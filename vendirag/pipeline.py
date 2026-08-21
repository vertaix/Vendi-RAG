"""
Vendi-RAG — the iterative pipeline (Algorithm 1 of the paper).

Retrieval is only half the method.  The other half is a loop that treats the
diversity weight ``s`` as something to *steer* rather than tune:

    Initialize  q_1 <- q,  s_1 <- 0.8,  best_quality <- 0
    For i = 1..N:
        C   <- TopK-Similar(q_i, K, |C|)          # candidate pool
        D_i <- GreedyVRS(C, s_i, k)               # Eq. 2
        r_i <- GenerateCoT(q_i, D_i)
        a_i <- GenerateAnswer(q, D_i, r_1..i)     # the ORIGINAL question
        Q_i <- LLM-Judge(a_i, q, D_i)             # mean(C, R, QA) / 10
        track the best answer so far
        if Q_i >= tau: stop
        s_target <- clip(1 - Q_i / max(best_quality, eps), 0, 1)
        s_{i+1}  <- beta * s_i + (1 - beta) * s_target      # Eq. 3
        q_{i+1}  <- RefineQuery(q_i, a_i, r_i)
    Return the best answer

When the answer is good, ``s`` falls and retrieval consolidates around the
current reasoning path.  When it is poor, ``s`` rises and retrieval fans out to
look for the evidence that is missing.  ``s`` carries across iterations through
the EMA independently of query rewriting, so the diversity/relevance balance is
a property of the whole trajectory for a query.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence

import numpy as np

from .llm import Backend, HeuristicBackend, Judgement, PromptedBackend
from .retriever import VendiRetriever
from .types import Document

__all__ = ["VendiRAG", "VendiRAGResult", "Iteration"]

# Paper defaults (Section 2.2 / Appendix A.3).
INITIAL_S = 0.8
BETA = 0.3
EPSILON = 0.01
TAU = 0.85
MAX_ITERATIONS = 5
K_CANDIDATES = 50
K_DOCS = 5


@dataclass
class Iteration:
    """A full record of one pass through the loop."""

    index: int
    s: float
    query: str
    documents: List[Document]
    reasoning: str
    answer: str
    judgement: Judgement
    vendi_score: float
    mean_similarity: float
    s_next: Optional[float] = None
    s_target: Optional[float] = None

    @property
    def quality(self) -> float:
        return self.judgement.quality

    @property
    def context(self) -> str:
        return "\n\n".join(d.text for d in self.documents)


@dataclass
class VendiRAGResult:
    """What the loop returns: the best answer plus the full trajectory."""

    question: str
    answer: str
    quality: float
    iterations: List[Iteration] = field(default_factory=list)
    stopped_early: bool = False

    @property
    def n_iterations(self) -> int:
        return len(self.iterations)

    @property
    def s_trajectory(self) -> List[float]:
        return [it.s for it in self.iterations]

    @property
    def quality_trajectory(self) -> List[float]:
        return [it.quality for it in self.iterations]

    @property
    def best_iteration(self) -> Optional[Iteration]:
        return max(self.iterations, key=lambda it: it.quality, default=None)

    @property
    def documents(self) -> List[Document]:
        """Documents behind the returned answer."""
        best = self.best_iteration
        return best.documents if best else []

    def __str__(self) -> str:
        lines = [
            f"Q: {self.question}",
            f"A: {self.answer}",
            f"   quality={self.quality:.3f}  iterations={self.n_iterations}"
            f"{'  (early stop)' if self.stopped_early else ''}",
            f"   s: {' -> '.join(f'{s:.2f}' for s in self.s_trajectory)}",
        ]
        return "\n".join(lines)


class VendiRAG:
    """The Vendi-RAG pipeline.

    >>> from vendirag import VendiRAG, VendiRetriever
    >>> retriever = VendiRetriever.from_texts(corpus)              # doctest: +SKIP
    >>> rag = VendiRAG(retriever, llm=OpenAILLM("gpt-4o-mini"))    # doctest: +SKIP
    >>> print(rag.answer("Who led the Thule Expedition?"))         # doctest: +SKIP

    Parameters
    ----------
    retriever : VendiRetriever
        Supplies the candidate pool and the VRS selection.
    llm : LLM, callable, or None
        Drives chain of thought, answering, and query refinement.  Pass ``None``
        together with ``backend=HeuristicBackend()`` to run without any model.
    judge_llm : LLM or callable, optional
        A separate judge.  The quality target is *relative*, so a smaller model
        works here.
    backend : Backend, optional
        Full control over the four LLM steps; overrides ``llm``.
    max_iterations : int
        N.  The paper reports convergence in 3-4 iterations on average.
    quality_threshold : float
        tau.  Stopping is absolute, so a locally-best but poor answer can never
        end the loop early.
    initial_s : float
        s_1.  0.8 starts the loop diversity-leaning, which matters most on the
        first retrieval when nothing is known about the answer yet.
    k_docs, k_candidates : int
        k and |C| for each retrieval step.
    beta : float
        EMA momentum.  Higher means s moves more slowly.
    dynamic_s : bool
        ``False`` pins s to ``initial_s`` — the fixed-s variant, which the paper
        recommends as the default configuration since it captures most of the
        gain at no extra inference cost.
    early_stopping : bool
        ``False`` always runs exactly ``max_iterations``.
    refine_query : bool
        ``False`` re-retrieves with the original question every iteration.
    verbose : bool
        Print per-iteration diagnostics.
    on_iteration : callable, optional
        Called with each :class:`Iteration` as it completes — for progress
        bars, logging, or animating a run.
    """

    def __init__(
        self,
        retriever: VendiRetriever,
        llm: Any = None,
        judge_llm: Any = None,
        backend: Optional[Backend] = None,
        max_iterations: int = MAX_ITERATIONS,
        quality_threshold: float = TAU,
        initial_s: float = INITIAL_S,
        k_docs: int = K_DOCS,
        k_candidates: int = K_CANDIDATES,
        beta: float = BETA,
        epsilon: float = EPSILON,
        dynamic_s: bool = True,
        early_stopping: bool = True,
        refine_query: bool = True,
        prompts: Optional[dict] = None,
        verbose: bool = False,
        on_iteration: Optional[Callable[[Iteration], None]] = None,
    ):
        if backend is None:
            if llm is None:
                raise ValueError(
                    "pass an llm=..., or backend=HeuristicBackend() to run without a model"
                )
            backend = PromptedBackend(llm, judge_llm=judge_llm, prompts=prompts)
        self.retriever = retriever
        self.backend = backend
        self.max_iterations = int(max_iterations)
        self.quality_threshold = float(quality_threshold)
        self.initial_s = float(initial_s)
        self.k_docs = int(k_docs)
        self.k_candidates = int(k_candidates)
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.dynamic_s = bool(dynamic_s)
        self.early_stopping = bool(early_stopping)
        self.refine_query = bool(refine_query)
        self.verbose = bool(verbose)
        self.on_iteration = on_iteration

    # ── constructors ─────────────────────────────────────────────────────────

    @classmethod
    def from_texts(cls, texts: Sequence[str], llm: Any = None, s: float = INITIAL_S,
                   embedder: Any = None, **kwargs) -> "VendiRAG":
        """Build retriever and pipeline in one call."""
        retriever = VendiRetriever.from_texts(
            texts, embedder=embedder, s=s,
            k=kwargs.get("k_docs", K_DOCS),
            candidate_pool=kwargs.get("k_candidates", K_CANDIDATES),
        )
        return cls(retriever, llm=llm, initial_s=s, **kwargs)

    @classmethod
    def offline(cls, retriever: VendiRetriever, **kwargs) -> "VendiRAG":
        """A pipeline with no model behind it, for demos and tests.

        Uses :class:`~vendirag.llm.HeuristicBackend`, which chains entities
        through the retrieved text instead of reasoning about it.  Everything
        else — the judge signal, the EMA on ``s``, early stopping, query
        refinement — runs exactly as it does with a real LLM.
        """
        return cls(retriever, backend=HeuristicBackend(), **kwargs)

    # ── the loop ─────────────────────────────────────────────────────────────

    def answer(self, question: str) -> VendiRAGResult:
        """Run Algorithm 1 on one question."""
        s = self.initial_s
        best_quality, best_answer = 0.0, ""
        reasoning_chain: List[str] = []
        current_query = question
        result = VendiRAGResult(question=question, answer="", quality=0.0)

        for i in range(1, self.max_iterations + 1):
            # 1-2. candidate pool -> greedy VRS selection
            retrieval = self.retriever.retrieve_details(
                current_query, k=self.k_docs, s=s, candidate_pool=self.k_candidates
            )
            context = "\n\n".join(d.text for d in retrieval.documents)

            # 3. chain of thought from the *current* (possibly refined) query
            reasoning = self.backend.cot(current_query, context)
            reasoning_chain.append(reasoning)
            accumulated = "\n".join(
                f"[iteration {j + 1}] {r}" for j, r in enumerate(reasoning_chain)
            )

            # 4. answer the ORIGINAL question, from all reasoning so far
            candidate = self.backend.answer(question, context, accumulated)

            # 5. judge
            judgement = self.backend.judge(question, context, candidate)
            quality = judgement.quality

            record = Iteration(
                index=i, s=s, query=current_query,
                documents=list(retrieval.documents),
                reasoning=reasoning, answer=candidate, judgement=judgement,
                vendi_score=retrieval.vendi_score,
                mean_similarity=retrieval.mean_similarity,
            )

            # 6. best-answer tracking: a bad iteration can never corrupt output
            if quality > best_quality:
                best_quality, best_answer = quality, candidate

            if self.verbose:
                print(
                    f"[iter {i}] s={s:.3f}  Q={quality:.3f}  "
                    f"VS={retrieval.vendi_score:.2f}  a={candidate[:70]!r}"
                )

            # 7. absolute stopping rule
            if self.early_stopping and quality >= self.quality_threshold:
                record.s_next = s
                result.iterations.append(record)
                result.stopped_early = True
                if self.on_iteration:
                    self.on_iteration(record)
                break

            if i < self.max_iterations:
                # 8. EMA update of s (Eq. 3)
                if self.dynamic_s:
                    s_target = float(np.clip(
                        1.0 - quality / max(best_quality, self.epsilon), 0.0, 1.0
                    ))
                    record.s_target = s_target
                    s = self.beta * s + (1.0 - self.beta) * s_target
                record.s_next = s

                # 9. query refinement — runs regardless of s, so exploration
                #    continues even while retrieval is in exploitation mode
                if self.refine_query:
                    current_query = self.backend.refine(
                        current_query, candidate, reasoning
                    ) or current_query
            else:
                record.s_next = s

            result.iterations.append(record)
            if self.on_iteration:
                self.on_iteration(record)

        result.answer = best_answer
        result.quality = best_quality
        return result

    def batch(
        self,
        questions: Sequence[str],
        progress: bool = False,
    ) -> List[VendiRAGResult]:
        """Run the loop over many questions."""
        results = []
        total = len(questions)
        for i, q in enumerate(questions, start=1):
            if progress:
                print(f"[{i}/{total}] {q[:70]}")
            results.append(self.answer(q))
        return results

    # ── single-shot comparison baseline ──────────────────────────────────────

    def answer_single_shot(self, question: str, s: Optional[float] = None) -> VendiRAGResult:
        """One retrieval, one answer, no loop — the ablation to compare against."""
        saved = (self.max_iterations, self.dynamic_s, self.refine_query)
        self.max_iterations, self.dynamic_s, self.refine_query = 1, False, False
        try:
            if s is not None:
                saved_s, self.initial_s = self.initial_s, s
            out = self.answer(question)
        finally:
            self.max_iterations, self.dynamic_s, self.refine_query = saved
            if s is not None:
                self.initial_s = saved_s
        return out
