"""
Vendi Retrieval — the standalone retriever.

Similarity search returns the documents closest to the query, which on a real
corpus means near-duplicates: the retrieved set is redundant and multi-hop
questions lose the evidence they need.  Vendi retrieval instead scores the
*set*, trading relevance against set-level diversity through the Vendi
Retrieval Score (Eq. 2 of the paper):

    VRS(D) = s * VS~(D) + (1 - s) * SS~(q, D)

where ``VS~`` is the Vendi Score of D rescaled to [0, 1], ``SS~`` is the mean
query-document cosine similarity min-max normalized over the candidate pool,
and ``s`` in [0, 1] sets the trade-off (0 = pure relevance, 1 = pure diversity).

Because exact maximization over all k-subsets is intractable, documents are
added greedily:  d* = argmax_{d in C \\ D} VRS(D + {d}).

This module depends only on numpy.  Use it on its own, drop it in front of an
existing vector store with :func:`vendi_rerank`, or let
:class:`~vendirag.pipeline.VendiRAG` drive it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Union

import numpy as np

from .embeddings import Embedder, HashingEmbedder, auto_embedder, l2_normalize
from .types import Document, RetrievalResult
from .vendi import normalized_vendi_score, vendi_score

__all__ = ["vrs_select", "vendi_rerank", "mmr_select", "VendiRetriever",
           "VendiReranker", "SelectionStep"]


@dataclass
class SelectionStep:
    """One greedy step, recorded for inspection and visualization."""

    rank: int
    index: int
    vrs: float
    vs_norm: float
    ss_norm: float
    vendi_score: float


def vrs_select(
    doc_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    k: int = 5,
    s: float = 0.8,
    q: float = 1.0,
    return_trace: bool = False,
):
    """Greedily select ``k`` documents maximizing the Vendi Retrieval Score.

    Parameters
    ----------
    doc_embeddings : array of shape (n, d)
        Candidate pool embeddings.  Normalized internally, so raw vectors are
        fine.
    query_embedding : array of shape (d,)
    k : int
        Number of documents to select.
    s : float in [0, 1]
        Diversity weight.  ``s=0`` reduces to picking the ``k`` most similar
        documents; ``s=1`` ignores relevance within the pool entirely.
    q : float
        Renyi order of the Vendi Score (1.0 = standard).
    return_trace : bool
        Also return the per-step :class:`SelectionStep` records.

    Returns
    -------
    list[int] of selected indices in selection order, or
    ``(indices, trace)`` when ``return_trace`` is True.

    Notes
    -----
    Cost is ``O(k * |C| * k^3)`` — the eigendecomposition is on the *selected
    subset* (size <= k), never on the corpus, so this stays in the low
    milliseconds for the paper's ``k <= 10``, ``|C| = 50``.
    """
    X = np.asarray(doc_embeddings, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"doc_embeddings must be (n, d), got {X.shape}")
    n = len(X)
    if n == 0:
        return ([], []) if return_trace else []
    if not 0.0 <= s <= 1.0:
        raise ValueError(f"s must be in [0, 1], got {s}")

    X = l2_normalize(X, axis=1)
    query = l2_normalize(np.asarray(query_embedding, dtype=np.float64).reshape(1, -1))[0]

    # Relevance term, min-max normalized over the candidate pool (Appendix A.5).
    ss_all = X @ query
    ss_min, ss_max = float(ss_all.min()), float(ss_all.max())
    ss_range = max(ss_max - ss_min, 1e-9)
    ss_norm_all = (ss_all - ss_min) / ss_range

    k = min(int(k), n)
    selected: List[int] = []
    trace: List[SelectionStep] = []
    remaining = list(range(n))
    ss_running = 0.0  # sum of raw similarities of the selected set

    for rank in range(k):
        rem = np.asarray(remaining)
        m = len(selected)

        if m == 0:
            # A singleton set has VS = 1 exactly, so the diversity term is
            # constant here and the first pick comes down to relevance.
            vs_norms = np.zeros(len(rem))
            vs_raw = np.ones(len(rem))
        else:
            sel_X = X[selected]                      # (m, d)
            K_sel = sel_X @ sel_X.T                  # (m, m)
            cross = sel_X @ X[rem].T                 # (m, n_rem)
            vs_raw = np.empty(len(rem))
            K = np.empty((m + 1, m + 1), dtype=np.float64)
            K[:m, :m] = K_sel
            K[m, m] = 1.0
            for j in range(len(rem)):
                K[:m, m] = cross[:, j]
                K[m, :m] = cross[:, j]
                vs_raw[j] = _vendi_from_kernel(K, q)
            vs_norms = (vs_raw - 1.0) / m            # trial set has m + 1 items

        # Relevance of each trial set: mean raw similarity, then pool-normalized.
        ss_trial = (ss_running + ss_all[rem]) / (m + 1)
        ss_trial_norm = (ss_trial - ss_min) / ss_range

        vrs = s * vs_norms + (1.0 - s) * ss_trial_norm
        # Ties are broken by relevance.  This matters at s = 1, where the
        # objective says nothing about the first pick (a singleton set always
        # has VS = 1), and keeps selection stable elsewhere.
        best = int(np.argmax(vrs + 1e-9 * ss_norm_all[rem]))
        chosen = int(rem[best])

        selected.append(chosen)
        remaining.remove(chosen)
        ss_running += float(ss_all[chosen])

        if return_trace:
            trace.append(
                SelectionStep(
                    rank=rank,
                    index=chosen,
                    vrs=float(vrs[best]),
                    vs_norm=float(vs_norms[best]),
                    ss_norm=float(ss_trial_norm[best]),
                    vendi_score=float(vs_raw[best]),
                )
            )

    return (selected, trace) if return_trace else selected


def _vendi_from_kernel(K: np.ndarray, q: float = 1.0) -> float:
    """Vendi Score of a small unit-diagonal kernel (inlined hot path)."""
    n = K.shape[0]
    if n == 1:
        return 1.0
    w = np.linalg.eigvalsh(K / n)
    w = w[w > 1e-12]
    if q == 1.0:
        return float(np.exp(-(w * np.log(w)).sum()))
    return float(np.exp(np.log((w ** q).sum()) / (1.0 - q)))


def mmr_select(
    doc_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    k: int = 5,
    lambda_mult: float = 0.5,
) -> List[int]:
    """Maximal Marginal Relevance selection — provided as a baseline.

    MMR penalizes the *single* most similar already-selected document, a
    pairwise notion of novelty, where the VRS scores the whole set at once.
    Included so comparisons against Vendi retrieval can be run without pulling
    in another library.
    """
    X = l2_normalize(np.asarray(doc_embeddings, dtype=np.float64), axis=1)
    query = l2_normalize(np.asarray(query_embedding, dtype=np.float64).reshape(1, -1))[0]
    sims = X @ query
    n = len(X)
    k = min(int(k), n)

    selected = [int(np.argmax(sims))]
    remaining = [i for i in range(n) if i != selected[0]]
    while len(selected) < k and remaining:
        rem = np.asarray(remaining)
        redundancy = (X[rem] @ X[selected].T).max(axis=1)
        scores = lambda_mult * sims[rem] - (1.0 - lambda_mult) * redundancy
        chosen = int(rem[int(np.argmax(scores))])
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def vendi_rerank(
    query_embedding: np.ndarray,
    doc_embeddings: np.ndarray,
    documents: Optional[Sequence] = None,
    k: int = 5,
    s: float = 0.8,
    q: float = 1.0,
):
    """Re-rank candidates from *any* retriever with the Vendi Retrieval Score.

    The drop-in path for an existing stack: pull a candidate pool from your
    vector store (Chroma, FAISS, Pinecone, Elasticsearch, ...), hand the pool's
    embeddings here, and get back a diverse, relevant subset.

    Returns the selected documents when ``documents`` is given, otherwise the
    selected indices.

    >>> docs, embs = my_store.search(query, k=50)          # doctest: +SKIP
    >>> best = vendi_rerank(q_emb, embs, docs, k=5, s=0.8) # doctest: +SKIP
    """
    idx = vrs_select(doc_embeddings, query_embedding, k=k, s=s, q=q)
    if documents is None:
        return idx
    return [documents[i] for i in idx]


class VendiRetriever:
    """A complete diversity-aware retriever: index texts, retrieve a subset.

    >>> from vendirag import HashingEmbedder, VendiRetriever
    >>> retriever = VendiRetriever.from_texts([
    ...     "Cats purr when they are content.",
    ...     "Cats purr when they are content and relaxed.",
    ...     "Cats purr when they are content and calm.",
    ...     "Dogs bark when they are alert.",
    ... ], embedder=HashingEmbedder(), s=0.8, k=2)

    Relevance alone spends both slots on the same fact:

    >>> [d.text for d in retriever.similarity_search("Why do cats purr?", k=2)]
    ['Cats purr when they are content.', 'Cats purr when they are content and relaxed.']

    Scoring the set does not:

    >>> [d.text for d in retriever.retrieve("Why do cats purr?")]
    ['Cats purr when they are content.', 'Dogs bark when they are alert.']

    (The embedder is pinned here only so the example is reproducible; leave it
    out and the best available encoder is used.)

    Parameters
    ----------
    embedder : Embedder, optional
        Any object with ``encode(texts) -> (n, d)``.  Defaults to
        sentence-transformers when installed, else the built-in
        :class:`~vendirag.embeddings.HashingEmbedder`.
    s : float
        Default diversity weight (paper default 0.8).
    k : int
        Default number of documents to return.
    candidate_pool : int
        Size of the similarity-search pool the greedy selection runs over
        (``|C|`` in the paper, default 50).  Relevance therefore enters twice:
        once in forming the pool, once through the SS term.
    q : float
        Renyi order for the Vendi Score.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        s: float = 0.8,
        k: int = 5,
        candidate_pool: int = 50,
        q: float = 1.0,
    ):
        self.embedder = embedder if embedder is not None else auto_embedder()
        self.s = float(s)
        self.k = int(k)
        self.candidate_pool = int(candidate_pool)
        self.q = float(q)
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None

    # ── building the index ───────────────────────────────────────────────────

    @classmethod
    def from_texts(cls, texts: Sequence[str], metadatas: Optional[Sequence[dict]] = None,
                   **kwargs) -> "VendiRetriever":
        """Build a retriever directly from a list of strings."""
        docs = [
            Document(text=t, metadata=dict(metadatas[i]) if metadatas else {}, id=str(i))
            for i, t in enumerate(texts)
        ]
        return cls(**kwargs).index(docs)

    def index(
        self,
        documents: Sequence[Union[str, dict, Document]],
        embeddings: Optional[np.ndarray] = None,
    ) -> "VendiRetriever":
        """Embed and store ``documents``, replacing any existing index.

        Pass ``embeddings`` to reuse vectors you already have and skip encoding.
        """
        docs = [Document.coerce(d) for d in documents]
        if not docs:
            raise ValueError("cannot index an empty document list")
        for i, doc in enumerate(docs):
            if doc.id is None:
                doc.id = str(i)

        if embeddings is None:
            # The hashing embedder needs corpus statistics for its idf weights.
            if isinstance(self.embedder, HashingEmbedder):
                self.embedder.fit(d.text for d in docs)
            embeddings = self.embedder.encode([d.text for d in docs])

        X = l2_normalize(np.asarray(embeddings, dtype=np.float64), axis=1)
        if len(X) != len(docs):
            raise ValueError(
                f"got {len(X)} embeddings for {len(docs)} documents"
            )
        self.documents = docs
        self.embeddings = X
        return self

    def __len__(self) -> int:
        return len(self.documents)

    # ── retrieval ────────────────────────────────────────────────────────────

    def embed_query(self, query: str) -> np.ndarray:
        return l2_normalize(np.asarray(self.embedder.encode([query])[0]).reshape(1, -1))[0]

    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Plain top-k similarity search — the baseline Vendi retrieval improves on."""
        self._require_index()
        k = self.k if k is None else k
        sims = self.embeddings @ self.embed_query(query)
        idx = np.argsort(-sims)[:k]
        return [self._with_score(i, sims[i]) for i in idx]

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        s: Optional[float] = None,
        candidate_pool: Optional[int] = None,
    ) -> List[Document]:
        """Retrieve a relevant *and* diverse set of documents for ``query``."""
        return self.retrieve_details(query, k=k, s=s, candidate_pool=candidate_pool).documents

    def retrieve_details(
        self,
        query: str,
        k: Optional[int] = None,
        s: Optional[float] = None,
        candidate_pool: Optional[int] = None,
        return_trace: bool = False,
    ) -> RetrievalResult:
        """Like :meth:`retrieve`, but returns the diagnostics too.

        The :class:`~vendirag.types.RetrievalResult` carries the selected set's
        Vendi Score, its mean query similarity, and the candidate pool — enough
        to plot or debug a selection.
        """
        self._require_index()
        k = self.k if k is None else int(k)
        s = self.s if s is None else float(s)
        pool = self.candidate_pool if candidate_pool is None else int(candidate_pool)

        query_emb = self.embed_query(query)
        sims = self.embeddings @ query_emb
        pool_idx = np.argsort(-sims)[:min(pool, len(sims))]
        pool_embs = self.embeddings[pool_idx]

        out = vrs_select(pool_embs, query_emb, k=k, s=s, q=self.q, return_trace=return_trace)
        local_idx, trace = out if return_trace else (out, None)
        global_idx = [int(pool_idx[i]) for i in local_idx]

        chosen = self.embeddings[global_idx]
        mean_sim = float(np.mean(sims[global_idx])) if global_idx else 0.0
        vs = vendi_score(chosen, q=self.q, normalize=False) if global_idx else 0.0

        result = RetrievalResult(
            documents=[self._with_score(i, sims[i]) for i in global_idx],
            indices=global_idx,
            vendi_score=vs,
            mean_similarity=mean_sim,
            vrs=trace[-1].vrs if trace else float("nan"),
            s=s,
            candidate_indices=[int(i) for i in pool_idx],
            query_embedding=query_emb,
        )
        if return_trace:
            result.trace = trace  # type: ignore[attr-defined]
            # Trace indices are pool-local; map them to the index for callers.
            result.trace_global = global_idx  # type: ignore[attr-defined]
        return result

    def set_diversity(self, documents: Sequence[Union[str, Document]]) -> float:
        """Vendi Score of an arbitrary document set — the effective number of
        unique documents it contains.  Handy for measuring redundancy in
        *someone else's* retriever output."""
        texts = [d if isinstance(d, str) else Document.coerce(d).text for d in documents]
        if not texts:
            return 0.0
        return vendi_score(self.embedder.encode(texts), q=self.q)

    # ── internals ────────────────────────────────────────────────────────────

    def _require_index(self) -> None:
        if self.embeddings is None:
            raise RuntimeError("no documents indexed — call .index(documents) first")

    def _with_score(self, i: int, score: float) -> Document:
        doc = self.documents[i]
        return Document(text=doc.text, metadata=doc.metadata, id=doc.id, score=float(score))


class VendiReranker:
    """Vendi retrieval on top of a vector store you already have.

    :class:`VendiRetriever` keeps the whole corpus in memory, which is the
    wrong shape once the corpus is large or already lives in Chroma, FAISS,
    Pinecone, Elasticsearch, or pgvector.  This class keeps the store and adds
    only the selection step: you supply a function that returns a candidate
    pool, and it applies the greedy VRS on top.

    Parameters
    ----------
    candidate_fn : callable
        ``candidate_fn(query, n) -> (documents, doc_embeddings, query_embedding)``
        where ``doc_embeddings`` is ``(n, d)`` and ``query_embedding`` is
        ``(d,)``.  Normalization is handled here.
    s, k, candidate_pool, q :
        As on :class:`VendiRetriever`.

    >>> def candidates(query, n):                                # doctest: +SKIP
    ...     hits = my_index.search(query, n)
    ...     return hits.docs, hits.vectors, my_encoder(query)
    >>> reranker = VendiReranker(candidates, s=0.8, k=5)         # doctest: +SKIP
    >>> reranker.retrieve("who led the expedition?")             # doctest: +SKIP

    The pipeline accepts this in place of a :class:`VendiRetriever`, so the
    full Vendi-RAG loop runs against your store unchanged.
    """

    def __init__(
        self,
        candidate_fn: Callable[[str, int], tuple],
        s: float = 0.8,
        k: int = 5,
        candidate_pool: int = 50,
        q: float = 1.0,
    ):
        self.candidate_fn = candidate_fn
        self.s = float(s)
        self.k = int(k)
        self.candidate_pool = int(candidate_pool)
        self.q = float(q)

    def retrieve(self, query: str, k: Optional[int] = None, s: Optional[float] = None,
                 candidate_pool: Optional[int] = None) -> List[Document]:
        return self.retrieve_details(
            query, k=k, s=s, candidate_pool=candidate_pool
        ).documents

    def retrieve_details(
        self,
        query: str,
        k: Optional[int] = None,
        s: Optional[float] = None,
        candidate_pool: Optional[int] = None,
        return_trace: bool = False,
    ) -> RetrievalResult:
        k = self.k if k is None else int(k)
        s = self.s if s is None else float(s)
        pool = self.candidate_pool if candidate_pool is None else int(candidate_pool)

        raw_docs, doc_embs, query_emb = self.candidate_fn(query, pool)
        docs = [Document.coerce(d) for d in raw_docs]
        if not docs:
            return RetrievalResult([], [], 0.0, 0.0, float("nan"), s)

        doc_embs = l2_normalize(np.asarray(doc_embs, dtype=np.float64), axis=1)
        query_emb = l2_normalize(
            np.asarray(query_emb, dtype=np.float64).reshape(1, -1)
        )[0]

        out = vrs_select(doc_embs, query_emb, k=k, s=s, q=self.q, return_trace=return_trace)
        idx, trace = out if return_trace else (out, None)
        sims = doc_embs @ query_emb

        result = RetrievalResult(
            documents=[docs[i] for i in idx],
            indices=[int(i) for i in idx],
            vendi_score=vendi_score(doc_embs[idx], q=self.q, normalize=False) if idx else 0.0,
            mean_similarity=float(np.mean(sims[idx])) if idx else 0.0,
            vrs=trace[-1].vrs if trace else float("nan"),
            s=s,
            candidate_indices=list(range(len(docs))),
            query_embedding=query_emb,
        )
        if return_trace:
            result.trace = trace  # type: ignore[attr-defined]
        return result

    def set_diversity(self, documents: Sequence[Union[str, Document]]) -> float:
        raise NotImplementedError(
            "VendiReranker has no embedder of its own; score the set with "
            "vendirag.vendi_score(embeddings) instead"
        )
