"""
Vendi-RAG  —  Algorithm 1 from the paper.

Correctly implements every component described in Section 2 and the appendix:

  Section 2.1  Vendi Retrieval
    - VS_k(D)  = exp( H_1( eigenvalues(K/n) ) )            [Eq. 1]
    - VS_norm  = (VS - 1) / (n - 1)  in [0,1]              [Appendix A.5]
    - SS_norm  = (SS - SS_min) / (SS_max - SS_min)         [Appendix A.5]
    - VRS(D)   = s*VS_norm(D) + (1-s)*SS_norm(q,D)         [Eq. 2]
    - Greedy:  d* = argmax_{d in C, d not in D} VRS(D+{d}) [Sec 2.1]

  Section 2.2  Vendi-RAG loop
    • Init: q_1=q, s_1=0.8, best_quality=0
    • Each iter: retrieve → CoT → answer(q, D, r_{1:i}) → judge → track best
    • Dynamic s: s_target = clip(1-Q/max(best,ε), 0,1)     [Eq. 3a]
                 s_{t+1}  = β·s_t + (1-β)·s_target         [Eq. 3b]
    • Judge: Q_t = mean(coherence, relevance, alignment)/10 [Appendix B.3]
    • Query refinement: q_{i+1} = RefineQuery(q_i, â_i, r_i)

Quick start
-----------
  # Single question (uses hotpotqa vector DB):
  python vendi_rag.py --dataset hotpotqa \\
      --question "Who was the director of Inception?"

  # Full benchmark with dynamic s:
  python vendi_rag.py --dataset hotpotqa --model gpt-4o-mini

  # Fixed-s ablation (no EMA update):
  python vendi_rag.py --dataset hotpotqa --no-dynamic-s
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

import numpy as np
import pandas as pd
import scipy.linalg

# ── path: works whether run from project root or models/ ────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from decouple import config
from langchain_community.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate

from vectorDB.dataset_ingestion import Ingestor


# ════════════════════════════════════════════════════════════════════════════
#  Paper hyperparameters (Section 2.2 / Appendix A.3)
# ════════════════════════════════════════════════════════════════════════════
INITIAL_S = 0.8           # s_1: initial diversity weight
BETA = 0.3                # β:   EMA momentum
EPSILON = 0.01            # ε:   numerical floor for best_quality
TAU = 0.85                # τ:   early-stop quality threshold  (= 8.5/10)
MAX_ITERATIONS = 5        # N:   maximum iterations
K_CANDIDATES = 50         # |C|: candidate pool size
K_DOCS = 5                # k:   documents selected per iteration


# ════════════════════════════════════════════════════════════════════════════
#  Section 2.1  —  Vendi Score & Greedy VRS Selection
# ════════════════════════════════════════════════════════════════════════════

def _vendi_score(embs: np.ndarray) -> float:
    """
    VS_k(D) = exp( H_1( eigenvalues(K/n) ) )    [Eq. 1]

    K_ij = cos(d_i, d_j).  Assumes embs are L2-normalized → K_ii = 1.
    Eigenvalues of K/n are non-negative and sum to 1 (trace = n/n = 1).
    """
    n = len(embs)
    K = embs @ embs.T           # cosine-similarity matrix; K_ii = 1
    w = scipy.linalg.eigvalsh(K / n)   # ascending order
    w = w[w > 1e-12]            # numerical guard for near-zero eigenvalues
    return float(np.exp(-(w * np.log(w)).sum()))


def _vs_norm(embs: np.ndarray) -> float:
    """
    ṼS_k(D) = (VS_k(D) - 1) / (n - 1)  ∈ [0, 1]    [Appendix A.5]

    VS ∈ [1, n]:  minimum 1 when all docs identical, maximum n when orthogonal.
    Returns 0.0 for singleton sets (n=1) since VS=1 always there.
    """
    n = len(embs)
    if n <= 1:
        return 0.0
    return (_vendi_score(embs) - 1.0) / (n - 1.0)


def greedy_vrs_select(
    candidates: list,
    doc_embs: np.ndarray,   # (n, d)  L2-normalized
    query_emb: np.ndarray,  # (d,)    L2-normalized
    s: float,
    k: int,
) -> list:
    """
    Greedy VRS subset selection  [Section 2.1, Eq. 2].

    At each step add:
        d* = argmax_{d in C, d not in D}  VRS(D u {d})

    where

        VRS(D) = s · ṼS(D) + (1-s) · ṼSS(q, D)

    Normalization:
        ṼS(D)    = (VS(D) - 1) / (|D| - 1)              [Appendix A.5]
        ṼSS(q,D) = (SS(q,D) - SS_min) / (SS_max - SS_min)
        SS(q,D)  = mean cosine-sim between q and every d ∈ D
        SS_min/max are computed over the full candidate pool C.

    Cost:  O(k · |C| · k³)  — negligible for k≤10, |C|=50 (< 50 ms).
    """
    n = len(candidates)
    if n == 0:
        return []

    # Per-candidate cosine similarity with query (L2-normalized → dot product)
    ss_all = (doc_embs @ query_emb).astype(float)   # (n,)
    ss_min = float(ss_all.min())
    ss_max = float(ss_all.max())
    ss_range = max(ss_max - ss_min, 1e-9)

    selected_idx: List[int] = []
    selected_embs: List[np.ndarray] = []
    remaining = list(range(n))

    for _ in range(min(k, n)):
        best_vrs, best_i = -np.inf, None

        for i in remaining:
            # Trial set = already selected + candidate i
            trial_embs = np.array(selected_embs + [doc_embs[i]])  # (m, d)

            vs_n = _vs_norm(trial_embs)

            # SS = average query–doc cosine sim over the trial set
            ss_vals = [ss_all[j] for j in selected_idx] + [ss_all[i]]
            ss_n = (float(np.mean(ss_vals)) - ss_min) / ss_range

            vrs = s * vs_n + (1.0 - s) * ss_n

            if vrs > best_vrs:
                best_vrs, best_i = vrs, i

        if best_i is None:
            break

        selected_idx.append(best_i)
        selected_embs.append(doc_embs[best_i])
        remaining.remove(best_i)

    return [candidates[i] for i in selected_idx]


# ════════════════════════════════════════════════════════════════════════════
#  LLM chains  (prompts match the paper's appendix exactly)
# ════════════════════════════════════════════════════════════════════════════

def _make_chain(llm, template: str, input_vars: list, schemas: list):
    """Build a prompt | llm | structured-output-parser chain."""
    parser = StructuredOutputParser.from_response_schemas(
        [ResponseSchema(**s) for s in schemas]
    )
    prompt = PromptTemplate(
        template=template,
        input_variables=input_vars,
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt | llm | parser


def _build_cot_chain(llm):
    """r_i = GenerateCoT(q_i, D_i)  — chain-of-thought from current query + docs."""
    return _make_chain(
        llm,
        template=(
            "Generate step-by-step chain-of-thought reasoning to answer the "
            "question using only the provided documents.\n\n"
            "Documents:\n{documents}\n\n"
            "Question: {question}\n\n"
            "Output VALID JSON:\n{format_instructions}"
        ),
        input_vars=["documents", "question"],
        schemas=[{
            "name": "reasoning",
            "description": "Step-by-step chain-of-thought reasoning.",
            "type": "string",
        }],
    )


def _build_answer_chain(llm):
    """â_i = GenerateAnswer(q, D_i, r_{1:i})  — uses ORIGINAL question q."""
    return _make_chain(
        llm,
        template=(
            "Provide a concise and precise answer to the question using the "
            "retrieved documents and the accumulated reasoning chain.\n\n"
            "Documents:\n{documents}\n\n"
            "Accumulated Reasoning (all iterations):\n{reasoning}\n\n"
            "Question: {question}\n\n"
            "Output VALID JSON:\n{format_instructions}"
        ),
        input_vars=["documents", "reasoning", "question"],
        schemas=[{
            "name": "answer",
            "description": "Concise, direct answer to the question.",
            "type": "string",
        }],
    )


def _build_judge_chain(llm):
    """
    Q_i = LLM-Judge(â_i, q, D_i)  [Section 2.2 + Appendix B.3].

    Scores coherence (C), relevance (R), query alignment (QA) each on 1–10.
    Q_t = mean(C, R, QA) / 10  ∈ [0, 1].   Threshold τ = 0.85.
    """
    return _make_chain(
        llm,
        template=(
            "You are an expert LLM-based judge tasked with evaluating the quality "
            "of answers in a Retrieval-Augmented Generation (RAG) system. "
            "Your evaluation will consider the following aspects:\n\n"
            "1. Coherence (C): Assess whether the answer is logically consistent "
            "and flows smoothly, without conflicting statements or gaps in reasoning.\n\n"
            "2. Relevance (R): Evaluate how well the answer addresses the query "
            "based on the information from the retrieved documents.\n\n"
            "3. Query Alignment (QA): Determine how closely the answer aligns with "
            "the specific query asked, ensuring the response is focused and appropriate.\n\n"
            "Scoring: each criterion on a 1–10 integer scale (10 = best).\n\n"
            "Query: {query}\n"
            "Retrieved Documents:\n{documents}\n"
            "Answer: {answer}\n\n"
            "Output VALID JSON with integer scores:\n{format_instructions}"
        ),
        input_vars=["query", "documents", "answer"],
        schemas=[
            {"name": "coherence",      "description": "Coherence score 1–10.",       "type": "integer"},
            {"name": "relevance",      "description": "Relevance score 1–10.",       "type": "integer"},
            {"name": "query_alignment","description": "Query alignment score 1–10.", "type": "integer"},
        ],
    )


def _build_refine_chain(llm):
    """q_{i+1} = RefineQuery(q_i, â_i, r_i)  — target information gaps."""
    return _make_chain(
        llm,
        template=(
            "The current answer is incomplete or low quality. Reformulate the "
            "question to retrieve the missing information, guided by the partial "
            "answer and reasoning chain.\n\n"
            "Original Question: {question}\n"
            "Candidate Answer: {answer}\n"
            "Reasoning: {reasoning}\n\n"
            "Output VALID JSON:\n{format_instructions}"
        ),
        input_vars=["question", "answer", "reasoning"],
        schemas=[{
            "name": "refined_question",
            "description": "Refined question that targets the missing information.",
            "type": "string",
        }],
    )


# ════════════════════════════════════════════════════════════════════════════
#  VendiRAG  —  Algorithm 1
# ════════════════════════════════════════════════════════════════════════════

class VendiRAG:
    """
    Vendi-RAG: diversity-driven iterative retrieval for multi-hop QA.

    Implements Algorithm 1 from the paper exactly:

      Initialize  q_1 ← q,  s_1 ← 0.8,  best_quality ← 0

      For i = 1 … N:
        C      ← TopK-Similar(q_i, 𝒦, k=50)           # candidate pool
        D_i    ← GreedySelect(C, s_i, k=5)             # Eq. 2 (VRS)
        r_i    ← GenerateCoT(q_i, D_i)
        â_i    ← GenerateAnswer(q, D_i, r_{1:i})       # q = ORIGINAL query
        Q_i    ← LLM-Judge(â_i, q, D_i)               # mean(C,R,QA)/10
        if Q_i > best_quality: â*, best_quality ← â_i, Q_i
        if Q_i ≥ τ: return â*
        s_target  ← clip(1 − Q_i / max(best_quality, ε), 0, 1)   # Eq. 3
        s_{i+1}   ← β·s_i + (1−β)·s_target
        q_{i+1}   ← RefineQuery(q_i, â_i, r_i)

      Return â*

    Parameters
    ----------
    ingestor : Ingestor
        Must point to an already-built Chroma vector database.
    model : str
        OpenAI chat model identifier (e.g. "gpt-4o-mini", "gpt-4o").
    max_iterations : int
        N in the algorithm (paper experiments: 3–5).
    quality_threshold : float
        τ – early-stop threshold (paper: 0.85 = 8.5/10 raw).
    initial_s : float
        s_1 – starting diversity weight (paper: 0.8).
    k_candidates : int
        |C| – candidate pool per iteration (paper: 50).
    k_docs : int
        k – documents selected per iteration (paper: 5).
    beta : float
        β – EMA momentum (paper: 0.3).
    dynamic_s : bool
        True  → full Vendi-RAG with EMA-adjusted s.
        False → fixed s = initial_s (the "fixed-s" ablation baseline).
    verbose : bool
        Print per-iteration diagnostics.
    api_key : str or None
        OpenAI API key; falls back to OPENAI_API_KEY environment variable.
    """

    def __init__(
        self,
        ingestor: Ingestor,
        model: str = "gpt-4o-mini",
        max_iterations: int = MAX_ITERATIONS,
        quality_threshold: float = TAU,
        initial_s: float = INITIAL_S,
        k_candidates: int = K_CANDIDATES,
        k_docs: int = K_DOCS,
        beta: float = BETA,
        dynamic_s: bool = True,
        verbose: bool = True,
        api_key: Optional[str] = None,
    ):
        self.ingestor = ingestor
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        self.initial_s = initial_s
        self.k_candidates = k_candidates
        self.k_docs = k_docs
        self.beta = beta
        self.dynamic_s = dynamic_s
        self.verbose = verbose

        llm = ChatOpenAI(
            model=model,
            temperature=0,
            api_key=api_key or config("OPENAI_API_KEY"),
        )
        self._cot    = _build_cot_chain(llm)
        self._answer = _build_answer_chain(llm)
        self._judge  = _build_judge_chain(llm)
        self._refine = _build_refine_chain(llm)

    # ── public API ────────────────────────────────────────────────────────────

    def answer(self, question: str) -> dict:
        """
        Run Vendi-RAG (Algorithm 1) on a single question.

        Returns
        -------
        dict
            answer        : best answer string across all iterations
            quality       : best Q_t ∈ [0, 1]
            iterations    : number of iterations actually executed
            s_trajectory  : list of s values used, one per iteration
            history       : list of per-iteration detail dicts
        """
        s = self.initial_s
        best_quality = 0.0
        best_answer = ""
        all_reasoning: List[str] = []
        current_question = question
        s_trajectory: List[float] = []
        history: List[dict] = []

        for it in range(1, self.max_iterations + 1):
            if self.verbose:
                print(f"\n{'─' * 60}")
                print(f"[Iter {it}/{self.max_iterations}]  s={s:.3f}")
                print(f"  query: {current_question}")

            # ─ Step 1: top-|C| candidates → greedy VRS selection ────────────
            candidates, doc_embs, query_emb = self.ingestor.get_candidates(
                current_question, k=self.k_candidates
            )
            docs = greedy_vrs_select(
                candidates, doc_embs, query_emb, s=s, k=self.k_docs
            )
            context = "\n\n".join(d.page_content for d in docs)

            # ─ Step 2: chain-of-thought from current (possibly refined) query ─
            reasoning = self._invoke(
                self._cot,
                {"documents": context, "question": current_question},
                key="reasoning",
            )
            all_reasoning.append(reasoning)
            accumulated = "\n".join(
                f"[Iter {j + 1}] {r}" for j, r in enumerate(all_reasoning)
            )

            # ─ Step 3: answer from ORIGINAL question + all prior reasoning ───
            candidate_answer = self._invoke(
                self._answer,
                {"documents": context, "reasoning": accumulated, "question": question},
                key="answer",
            )

            # ─ Step 4: judge  Q_t = mean(C,R,QA)/10 ∈ [0,1] ─────────────────
            quality = self._judge_quality(question, context, candidate_answer)

            if self.verbose:
                print(f"  answer : {candidate_answer}")
                print(f"  Q      : {quality:.3f}")

            s_trajectory.append(s)
            history.append({
                "iteration": it,
                "s": s,
                "query": current_question,
                "answer": candidate_answer,
                "quality": quality,
                "reasoning": reasoning,
            })

            # ─ Step 5: best-answer tracking ──────────────────────────────────
            if quality > best_quality:
                best_quality = quality
                best_answer = candidate_answer

            # ─ Step 6: early stop ─────────────────────────────────────────────
            if quality >= self.quality_threshold:
                if self.verbose:
                    print(f"  ✓ early stop  Q={quality:.3f} ≥ τ={self.quality_threshold}")
                break

            if it == self.max_iterations:
                break

            # ─ Step 7: EMA update of s  (Eq. 3) ─────────────────────────────
            if self.dynamic_s:
                # best_quality is already updated above, so when Q_i is the new
                # best: s_target = clip(1 - Q_i/Q_i, 0,1) = 0  → s shrinks.
                # When Q_i < best_quality: s_target > 0 → more diversity.
                s_target = float(np.clip(
                    1.0 - quality / max(best_quality, EPSILON),
                    0.0, 1.0,
                ))
                s = self.beta * s + (1.0 - self.beta) * s_target
                if self.verbose:
                    print(f"  s_target={s_target:.3f}  →  s_{it + 1}={s:.3f}")

            # ─ Step 8: query refinement ───────────────────────────────────────
            refined = self._invoke(
                self._refine,
                {"question": current_question, "answer": candidate_answer, "reasoning": reasoning},
                key="refined_question",
                fallback=current_question,
            )
            current_question = refined

        return {
            "answer": best_answer,
            "quality": best_quality,
            "iterations": len(history),
            "s_trajectory": s_trajectory,
            "history": history,
        }

    def run_benchmark(
        self,
        questions: List[str],
        ground_truths: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Run Vendi-RAG on a list of questions with checkpointing support.

        Parameters
        ----------
        questions : list[str]
        ground_truths : list[str] or None
        output_path : CSV path for final results (None = don't save)
        checkpoint_path : CSV path written after every question (None = no checkpoint)

        Returns
        -------
        pd.DataFrame  with columns: question, generated_answer, quality, iterations[, ground_truth]
        """
        rows: List[dict] = []

        # Resume from checkpoint if it exists
        start = 0
        if checkpoint_path and os.path.exists(checkpoint_path):
            prev = pd.read_csv(checkpoint_path)
            rows = prev.to_dict("records")
            start = len(rows)
            print(f"Resuming from Q#{start + 1} ({start} already done).")

        for i, q in enumerate(questions[start:], start=start):
            print(f"\n{'=' * 60}")
            print(f"Q {i + 1}/{len(questions)}: {q}")

            result = self.answer(q)
            row: dict = {
                "question": q,
                "generated_answer": result["answer"],
                "quality": result["quality"],
                "iterations": result["iterations"],
            }
            if ground_truths is not None:
                row["ground_truth"] = ground_truths[i]
            rows.append(row)

            if checkpoint_path:
                pd.DataFrame(rows).to_csv(checkpoint_path, index=False)

        df = pd.DataFrame(rows)
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"\nResults saved → {output_path}")
        return df

    # ── internals ────────────────────────────────────────────────────────────

    def _judge_quality(self, question: str, context: str, answer: str) -> float:
        """
        Q_t = mean(coherence, relevance, query_alignment) / 10  ∈ [0, 1].
        Falls back to 0.5 on parse errors to avoid crashing the loop.
        """
        try:
            out = self._judge.invoke({
                "query": question,
                "documents": context,
                "answer": answer,
            })
            c  = float(out.get("coherence",       5))
            r  = float(out.get("relevance",        5))
            qa = float(out.get("query_alignment",  5))
            return (c + r + qa) / 30.0
        except Exception as exc:
            if self.verbose:
                print(f"  [judge error] {exc}")
            return 0.5

    def _invoke(self, chain, inputs: dict, key: str, fallback: str = "") -> str:
        try:
            return chain.invoke(inputs).get(key, fallback)
        except Exception as exc:
            if self.verbose:
                print(f"  [chain error ({key})] {exc}")
            return fallback


# ════════════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Vendi-RAG  —  Algorithm 1 from the paper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--dataset", default="hotpotqa",
        choices=["hotpotqa", "musique", "2wikimultihopqa"],
        help="Dataset whose vector DB to query.",
    )
    ap.add_argument("--model", default="gpt-4o-mini",
                    help="OpenAI chat model.")
    ap.add_argument("--max-iter", type=int, default=MAX_ITERATIONS,
                    help=f"Maximum iterations N (paper: {MAX_ITERATIONS}).")
    ap.add_argument("--k-docs", type=int, default=K_DOCS,
                    help=f"k: docs selected per iteration (paper: {K_DOCS}).")
    ap.add_argument("--k-candidates", type=int, default=K_CANDIDATES,
                    help=f"|C|: candidate pool size (paper: {K_CANDIDATES}).")
    ap.add_argument("--initial-s", type=float, default=INITIAL_S,
                    help=f"s_1: initial diversity weight (paper: {INITIAL_S}).")
    ap.add_argument("--no-dynamic-s", action="store_true",
                    help="Keep s fixed at --initial-s (ablation baseline).")
    ap.add_argument("--question", type=str, default=None,
                    help="Run on a single question instead of the full benchmark.")
    ap.add_argument("--output", type=str, default=None,
                    help="CSV output path (auto-named if omitted).")
    args = ap.parse_args()

    ingestor = Ingestor(
        dataset_path=os.path.join(_ROOT, f"processed_data/{args.dataset}/"),
        persist_directory=os.path.join(_ROOT, f"vectorDB/{args.dataset}"),
    )

    rag = VendiRAG(
        ingestor=ingestor,
        model=args.model,
        max_iterations=args.max_iter,
        k_docs=args.k_docs,
        k_candidates=args.k_candidates,
        initial_s=args.initial_s,
        dynamic_s=not args.no_dynamic_s,
        verbose=True,
    )

    if args.question:
        # ── single-question mode ─────────────────────────────────────────────
        res = rag.answer(args.question)
        print(f"\n{'=' * 60}")
        print(f"Answer     : {res['answer']}")
        print(f"Quality    : {res['quality']:.3f}  (τ={TAU})")
        print(f"Iterations : {res['iterations']}")
        print(f"s path     : {[f'{v:.3f}' for v in res['s_trajectory']]}")
    else:
        # ── benchmark mode ───────────────────────────────────────────────────
        data = ingestor.load_evaluation_data()
        tag = f"dyn" if not args.no_dynamic_s else f"fixed"
        out = args.output or os.path.join(
            _ROOT,
            f"results/vendi_rag_{args.dataset}"
            f"_s{args.initial_s}_{tag}_k{args.k_docs}.csv",
        )
        ckpt = out.replace(".csv", "_checkpoint.csv")
        df = rag.run_benchmark(
            questions=data["question_text"],
            ground_truths=data["ground_truth"],
            output_path=out,
            checkpoint_path=ckpt,
        )
        print(df.head())
