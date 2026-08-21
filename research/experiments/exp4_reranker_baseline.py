"""
Experiment 4 — Cross-encoder reranker baseline
===============================================
Addresses Reviewer ezYK's concern: can Vendi-RAG's gains be explained
by simply having access to a larger candidate pool (top-50) rather than
by the diversity-aware selection?

Design
------
  Same top-50 candidates produced by Vendi-RAG's dense retriever
  → reranked by an off-the-shelf cross-encoder (no diversity term)
  → top-k fed to the same QA LLM
  → single-step (no iterative loop)

Performance design (two-phase)
-------------------------------
  Phase 1 — Batch retrieval + reranking (CPU, sequential batch):
    For every question: dense search → top-50 candidates.
    All (question, passage) pairs scored in one cross-encoder batch.
    Scores + top-k passages cached in memory.

  Phase 2 — Parallel LLM calls (I/O-bound, ThreadPoolExecutor):
    For every question: CoT + answer using the pre-selected passages.
    No cross-encoder calls during this phase → no CPU contention.

This decouples CPU-bound and I/O-bound work, giving ~10–20× speedup
over the naïve thread-pool approach where 20 workers fight over the model.

Rerankers compared
------------------
  - cross-encoder/ms-marco-MiniLM-L-6-v2  (fast, MS-MARCO fine-tuned)
  - BAAI/bge-reranker-base                 (stronger, paper-recommended)

Usage
-----
  # Full run (both rerankers, all 3 datasets):
  python research/experiments/exp4_reranker_baseline.py

  # Single reranker, single dataset:
  python research/experiments/exp4_reranker_baseline.py --reranker bge --dataset hotpotqa

  # Quick test (5 questions):
  python research/experiments/exp4_reranker_baseline.py --max-questions 5

  # Summary table from existing results:
  python research/experiments/exp4_reranker_baseline.py --summary-only
"""

from __future__ import annotations

import argparse
import collections
import os
import re
import string
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Must be set before any HuggingFace / transformers import
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "research"))

from sentence_transformers import CrossEncoder

from vendirag import OpenAILLM, PromptedBackend

from ingestion import ChromaCorpus, for_dataset


# ════════════════════════════════════════════════════════════════════════════
#  Configuration
# ════════════════════════════════════════════════════════════════════════════

DATASETS = ["hotpotqa", "musique", "2wikimultihopqa"]

RERANKERS: Dict[str, str] = {
    "msmarco": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "bge":     "BAAI/bge-reranker-base",
}

MODEL           = "gpt-4o-mini"
K_CANDIDATES    = 50
K_DOCS          = 5
LLM_WORKERS     = 20    # concurrent LLM calls (I/O-bound)
RERANKER_BATCH  = 256   # pairs per cross-encoder forward pass

OUT_DIR = os.path.join(_ROOT, "results", "exp4")


# ════════════════════════════════════════════════════════════════════════════
#  Evaluation helpers
# ════════════════════════════════════════════════════════════════════════════

def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def _f1(pred: str, gold: str) -> float:
    p = collections.Counter(_normalize(pred).split())
    g = collections.Counter(_normalize(gold).split())
    common = p & g
    if not common:
        return 0.0
    prec = sum(common.values()) / sum(p.values())
    rec  = sum(common.values()) / sum(g.values())
    return 2 * prec * rec / (prec + rec)


def score_df(df: pd.DataFrame) -> dict:
    em, f1, acc = [], [], []
    for _, row in df.iterrows():
        pred = _normalize(str(row.get("generated_answer", "") or ""))
        gold = _normalize(str(row.get("ground_truth", "") or ""))
        if not pred or not gold:
            continue
        em.append(int(pred == gold))
        f1.append(_f1(pred, gold))
        acc.append(int(gold in pred))
    n = max(len(em), 1)
    return {
        "n":   n,
        "EM":  round(100 * sum(em) / n, 2),
        "F1":  round(100 * sum(f1) / n, 2),
        "Acc": round(100 * sum(acc) / n, 2),
    }


_print_lock = threading.Lock()

def _safe_print(*a, **kw):
    with _print_lock:
        print(*a, **kw)


# ════════════════════════════════════════════════════════════════════════════
#  Phase 1 — batch retrieval + cross-encoder reranking
# ════════════════════════════════════════════════════════════════════════════

def batch_retrieve_and_rerank(
    questions: List[str],
    corpus: ChromaCorpus,
    reranker: CrossEncoder,
    k_candidates: int = K_CANDIDATES,
    k_docs: int = K_DOCS,
    batch_size: int = RERANKER_BATCH,
) -> List[List[str]]:
    """
    For every question, retrieve top-k_candidates via dense search,
    then score all (question, passage) pairs in one batched cross-encoder
    call, and return the top-k_docs passages per question.

    Returns
    -------
    contexts : list[str]
        One joined context string per question.
    """
    total = len(questions)
    print(f"Phase 1: retrieving candidates for {total} questions …")

    # Per-question candidate lists
    all_candidates: List[list] = []
    for i, q in enumerate(questions):
        cands, _, _ = corpus.candidates(q, k_candidates)
        all_candidates.append(cands)
        if (i + 1) % 50 == 0:
            print(f"  Retrieved {i+1}/{total}")

    # Build flat list of (query, passage) pairs + index bookkeeping
    pairs: List[tuple] = []
    offsets: List[int] = [0]          # start index in pairs[] for question i
    for q, cands in zip(questions, all_candidates):
        for doc in cands:
            pairs.append((q, doc.text))
        offsets.append(len(pairs))

    print(f"Phase 1: scoring {len(pairs)} pairs with cross-encoder (batch={batch_size}) …")
    all_scores = reranker.predict(pairs, batch_size=batch_size, show_progress_bar=True)

    # Select top-k_docs per question
    contexts: List[str] = []
    for i, (q, cands) in enumerate(zip(questions, all_candidates)):
        start, end = offsets[i], offsets[i + 1]
        scores_i = all_scores[start:end]
        top_idx  = np.argsort(scores_i)[-k_docs:][::-1]
        top_docs = [cands[j] for j in top_idx]
        contexts.append("\n\n".join(d.text for d in top_docs))

    print("Phase 1 complete.")
    return contexts


# ════════════════════════════════════════════════════════════════════════════
#  Phase 2 — parallel LLM calls
# ════════════════════════════════════════════════════════════════════════════

def parallel_llm_calls(
    questions: List[str],
    contexts: List[str],
    ground_truths: Optional[List[str]],
    backend: PromptedBackend,
    llm_workers: int = LLM_WORKERS,
    checkpoint_path: Optional[str] = None,
    done_qs: Optional[set] = None,
) -> List[dict]:
    """
    Run CoT + answer generation for all questions in parallel.
    Returns list of result dicts, preserving input order.
    """
    total = len(questions)
    rows: List[dict] = []
    lock = threading.Lock()
    done_qs = done_qs or set()

    def _worker(idx: int, q: str, ctx: str, gt: Optional[str]) -> dict:
        # Same CoT + answer prompts the full pipeline uses, so the only thing
        # that differs from Vendi-RAG here is how the k documents were chosen.
        reasoning = backend.cot(q, ctx)
        ans = backend.answer(q, ctx, reasoning)

        row = {"question": q, "generated_answer": ans, "iterations": 1}
        if gt is not None:
            row["ground_truth"] = gt
        with lock:
            rows.append(row)
            if checkpoint_path:
                pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
        _safe_print(f"  [{idx+1:>4}/{total}] {q[:65]}")
        return row

    pending = [
        (i, q, ctx, gt)
        for i, (q, ctx, gt) in enumerate(
            zip(questions, contexts,
                ground_truths if ground_truths else [None] * total)
        )
        if q not in done_qs
    ]

    print(f"Phase 2: {len(pending)} LLM calls  (workers={llm_workers}) …")
    with ThreadPoolExecutor(max_workers=llm_workers) as pool:
        futures = {pool.submit(_worker, i, q, ctx, gt): i
                   for i, q, ctx, gt in pending}
        for f in as_completed(futures):
            exc = f.exception()
            if exc:
                _safe_print(f"  [THREAD ERROR] {exc}")

    print("Phase 2 complete.")
    return rows


# ════════════════════════════════════════════════════════════════════════════
#  One (reranker_key, dataset) run
# ════════════════════════════════════════════════════════════════════════════

def run_one(
    reranker_key: str,
    dataset: str,
    max_questions: Optional[int] = None,
    llm_workers: int = LLM_WORKERS,
) -> pd.DataFrame:
    reranker_name = RERANKERS[reranker_key]
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path  = os.path.join(OUT_DIR, f"reranker_{reranker_key}_{dataset}.csv")
    ckpt_path = os.path.join(OUT_DIR, f"reranker_{reranker_key}_{dataset}_checkpoint.csv")

    # Load data
    corpus = for_dataset(dataset, root=_ROOT)
    questions, ground_truths = corpus.load_questions()

    if max_questions:
        questions     = questions[:max_questions]
        ground_truths = ground_truths[:max_questions]

    total = len(questions)

    # Already fully done?
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if len(df) >= total:
            print(f"  [skip] {reranker_key}/{dataset} — {len(df)} rows already saved")
            return df
        print(f"  [partial] {reranker_key}/{dataset} — {len(df)}/{total}; rerunning …")
        os.remove(csv_path)

    # Load checkpoint for LLM-phase resume
    done_qs: set = set()
    llm_rows: List[dict] = []
    if os.path.exists(ckpt_path):
        prev = pd.read_csv(ckpt_path)
        llm_rows = prev.to_dict("records")
        done_qs  = {r["question"] for r in llm_rows}
        print(f"  LLM checkpoint: {len(done_qs)} questions already answered.")

    print(f"\n{'='*60}")
    print(f"Reranker : {reranker_name}")
    print(f"Dataset  : {dataset}  ({total} questions)")
    print(f"{'='*60}")

    ingestor.load_vectordb(ingestor.persist_directory)

    # ── Phase 1: batch retrieval + reranking (always reruns for simplicity) ──
    reranker = CrossEncoder(reranker_name, max_length=512, device="cpu")
    contexts = batch_retrieve_and_rerank(
        questions, corpus, reranker,
        k_candidates=K_CANDIDATES, k_docs=K_DOCS,
    )

    # ── Phase 2: parallel LLM calls ──────────────────────────────────────────
    backend = PromptedBackend(OpenAILLM(MODEL))

    new_rows = parallel_llm_calls(
        questions, contexts, list(ground_truths),
        backend,
        llm_workers=llm_workers,
        checkpoint_path=ckpt_path,
        done_qs=done_qs,
    )
    all_rows = llm_rows + new_rows

    df = pd.DataFrame(all_rows)
    df.to_csv(csv_path, index=False)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print(f"Saved → {csv_path}  ({len(df)} rows)")
    return df


# ════════════════════════════════════════════════════════════════════════════
#  Summary table
# ════════════════════════════════════════════════════════════════════════════

def print_table(results: dict) -> None:
    # Try to load Exp 1 Condition A for comparison
    exp1_rows: Dict[str, dict] = {}
    for ds in DATASETS:
        p = os.path.join(_ROOT, "results", "exp1", f"condA_{ds}.csv")
        if os.path.exists(p):
            exp1_rows[ds] = score_df(pd.read_csv(p))

    header = (
        f"{'Method':<44} | {'HotpotQA':>9} | {'MuSiQue':>9} | {'2Wiki':>9}"
    )
    sep = "=" * len(header)
    print(f"\n{sep}")
    print("Experiment 4 — Cross-encoder reranker vs. Vendi-RAG  (Accuracy %)")
    print(sep)
    print(header)
    print("-" * len(header))

    if exp1_rows:
        vals = " | ".join(
            f"{exp1_rows[ds]['Acc']:>9.2f}" if ds in exp1_rows else f"{'N/A':>9}"
            for ds in DATASETS
        )
        print(f"{'Exp1-A  Vendi-RAG fixed-s (iterative)':<44} | {vals}")

    for rk in RERANKERS:
        vals = " | ".join(
            f"{results[(rk,ds)]['Acc']:>9.2f}" if (rk, ds) in results else f"{'N/A':>9}"
            for ds in DATASETS
        )
        label = f"Reranker ({rk})  {RERANKERS[rk].split('/')[-1]}"
        print(f"{label:<44} | {vals}")

    print(sep)
    for metric in ["EM", "F1"]:
        print(f"\n  {metric}:")
        if exp1_rows:
            vals = "  ".join(
                f"{exp1_rows.get(ds,{}).get(metric,'N/A'):>7}" for ds in DATASETS
            )
            print(f"    Exp1-A : {vals}")
        for rk in RERANKERS:
            vals = "  ".join(
                f"{results.get((rk,ds),{}).get(metric,'N/A'):>7}" for ds in DATASETS
            )
            print(f"    {rk:7}: {vals}")
    print()


def save_summary_csv(results: dict) -> None:
    rows = [{"reranker": rk, "model": RERANKERS[rk], "dataset": ds, **sc}
            for (rk, ds), sc in results.items()]
    path = os.path.join(OUT_DIR, "exp4_summary.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Summary saved → {path}")


# ════════════════════════════════════════════════════════════════════════════
#  Entry point
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Exp 4 — Cross-encoder reranker baseline (batch + parallel LLM)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--reranker",      choices=list(RERANKERS), default=None)
    ap.add_argument("--dataset",       choices=DATASETS,        default=None)
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--llm-workers",   type=int, default=LLM_WORKERS)
    ap.add_argument("--summary-only",  action="store_true")
    args = ap.parse_args()

    reranker_keys = [args.reranker] if args.reranker else list(RERANKERS)
    datasets      = [args.dataset]  if args.dataset  else DATASETS

    results: dict = {}

    for rk in RERANKERS:
        for ds in DATASETS:
            p = os.path.join(OUT_DIR, f"reranker_{rk}_{ds}.csv")
            if os.path.exists(p):
                results[(rk, ds)] = score_df(pd.read_csv(p))

    if args.summary_only:
        if results:
            print_table(results)
            save_summary_csv(results)
        else:
            print("No results found in", OUT_DIR)
        sys.exit(0)

    for rk in reranker_keys:
        for ds in datasets:
            df = run_one(rk, ds,
                         max_questions=args.max_questions,
                         llm_workers=args.llm_workers)
            results[(rk, ds)] = score_df(df)
            sc = results[(rk, ds)]
            print(f"  EM={sc['EM']:.2f}%  F1={sc['F1']:.2f}%  Acc={sc['Acc']:.2f}%")

    if results:
        print_table(results)
        save_summary_csv(results)
