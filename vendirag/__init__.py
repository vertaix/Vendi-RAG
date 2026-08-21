"""
Vendi-RAG — diversity-aware retrieval, and the loop that steers it.

Similarity search optimizes each document in isolation, so a retrieved set is
usually several restatements of the same fact.  Multi-hop questions need the
opposite: a few documents that say *different* things.  Vendi-RAG scores the
retrieved set as a whole with the Vendi Retrieval Score

    VRS(D) = s * VS~(D) + (1 - s) * SS~(q, D)

trading query relevance against the Vendi Score, an eigenvalue-based measure of
how many *effectively distinct* documents a set contains.

Two entry points, usable independently:

**The retriever, on its own** — numpy only, no LLM, no vector store::

    from vendirag import VendiRetriever

    retriever = VendiRetriever.from_texts(my_documents, s=0.8, k=5)
    docs = retriever.retrieve("Which instrument did the expedition carry?")

Already have a vector store?  Keep it, and re-rank its candidate pool::

    from vendirag import vendi_rerank

    docs = vendi_rerank(query_emb, pool_embs, pool_docs, k=5, s=0.8)

**The full pipeline** — the iterative loop with an LLM judge steering ``s``::

    from vendirag import VendiRAG, OpenAILLM

    rag = VendiRAG(retriever, llm=OpenAILLM("gpt-4o-mini"))
    print(rag.answer("In which town is the observatory that houses it?"))

Paper: https://arxiv.org/abs/2502.11228
"""

from .embeddings import (
    Embedder,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    auto_embedder,
)
from .llm import (
    AnthropicLLM,
    Backend,
    CallableLLM,
    HeuristicBackend,
    Judgement,
    OpenAILLM,
    PromptedBackend,
)
from .pipeline import Iteration, VendiRAG, VendiRAGResult
from .retriever import (
    SelectionStep,
    VendiReranker,
    VendiRetriever,
    mmr_select,
    vendi_rerank,
    vrs_select,
)
from .types import Document, RetrievalResult
from .vendi import normalized_vendi_score, vendi_score, vendi_score_from_kernel

__version__ = "0.2.0"

__all__ = [
    # retrieval
    "VendiRetriever",
    "VendiReranker",
    "vrs_select",
    "vendi_rerank",
    "mmr_select",
    "SelectionStep",
    # pipeline
    "VendiRAG",
    "VendiRAGResult",
    "Iteration",
    # scoring
    "vendi_score",
    "normalized_vendi_score",
    "vendi_score_from_kernel",
    # embedders
    "Embedder",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "auto_embedder",
    # LLM plumbing
    "OpenAILLM",
    "AnthropicLLM",
    "CallableLLM",
    "Backend",
    "PromptedBackend",
    "HeuristicBackend",
    "Judgement",
    # types
    "Document",
    "RetrievalResult",
    "__version__",
]
