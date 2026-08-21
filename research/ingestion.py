"""
Chroma-backed corpus for the paper's benchmarks.

The library keeps its index in memory, which is the wrong shape for the
HotpotQA / MuSiQue / 2WikiMultiHopQA corpora used in the paper.  This module
builds a persistent Chroma store over the preprocessed JSONL and exposes it as
a candidate source, so :class:`vendirag.VendiReranker` — and therefore the full
Vendi-RAG loop — runs against the store unchanged.

Setup follows Appendix A.4: ``all-mpnet-base-v2`` embeddings, Chroma persisted
to disk, 512-token chunks.

    python research/ingestion.py --dataset hotpotqa --build
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional, Tuple

# Must be set before sentence_transformers/transformers are imported, or an
# installed-but-unused TensorFlow gets pulled in and hangs the process.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np

from vendirag import Document, SentenceTransformerEmbedder, VendiReranker

DATASETS = ("hotpotqa", "musique", "2wikimultihopqa")
BATCH_SIZE = 5000


class ChromaCorpus:
    """Build, load, and query a persistent Chroma store.

    Parameters
    ----------
    dataset_path : str
        Directory holding ``test_subsampled.jsonl`` (see ``download/``).
    persist_directory : str
        Where Chroma keeps the index.
    model_name : str
        Sentence-transformers encoder; the paper uses ``all-mpnet-base-v2``.
    """

    def __init__(
        self,
        dataset_path: str,
        persist_directory: str,
        model_name: str = "all-mpnet-base-v2",
    ):
        self.dataset_path = dataset_path
        self.persist_directory = persist_directory
        self.embedder = SentenceTransformerEmbedder(model_name)
        self._store = None
        os.makedirs(persist_directory, exist_ok=True)

    # ── data ────────────────────────────────────────────────────────────────

    def load_passages(self) -> Tuple[List[str], List[dict]]:
        """Every context paragraph in the test split, with its metadata."""
        path = os.path.join(self.dataset_path, "test_subsampled.jsonl")
        with open(path, encoding="utf-8") as handle:
            samples = [json.loads(line) for line in handle]

        texts, metadatas = [], []
        for sample in samples:
            for context in sample["contexts"]:
                texts.append(context["paragraph_text"])
                metadatas.append({
                    "idx": context["idx"],
                    "title": context.get("title", "Unknown"),
                    "is_supporting": bool(context.get("is_supporting", False)),
                })
        return texts, metadatas

    def load_questions(self) -> Tuple[List[str], List[str]]:
        """Questions and their gold answers."""
        path = os.path.join(self.dataset_path, "test_subsampled.jsonl")
        with open(path, encoding="utf-8") as handle:
            samples = [json.loads(line) for line in handle]
        questions = [s["question_text"] for s in samples]
        answers = [s["answers_objects"][0]["spans"][0] for s in samples]
        return questions, answers

    # ── the store ───────────────────────────────────────────────────────────

    def build(self):
        """Embed every passage and persist the index.  Run once per dataset."""
        import chromadb

        texts, metadatas = self.load_passages()
        client = chromadb.PersistentClient(path=self.persist_directory)
        collection = client.get_or_create_collection("passages")
        done = collection.count()
        if done:
            print(f"{done} passages already indexed; resuming.")

        for start in range(done, len(texts), BATCH_SIZE):
            batch = texts[start:start + BATCH_SIZE]
            print(f"embedding {start}..{start + len(batch)} of {len(texts)}")
            collection.add(
                ids=[str(i) for i in range(start, start + len(batch))],
                documents=batch,
                metadatas=metadatas[start:start + len(batch)],
                embeddings=self.embedder.encode(batch).tolist(),
            )
        print(f"indexed {collection.count()} passages -> {self.persist_directory}")
        self._store = collection
        return collection

    @property
    def store(self):
        import chromadb

        if self._store is None:
            client = chromadb.PersistentClient(path=self.persist_directory)
            self._store = client.get_or_create_collection("passages")
            if self._store.count() == 0:
                raise RuntimeError(
                    f"no passages in {self.persist_directory}; run with --build first"
                )
        return self._store

    # ── the bridge into the library ─────────────────────────────────────────

    def candidates(self, query: str, n: int):
        """``(documents, doc_embeddings, query_embedding)`` for the top ``n`` hits.

        This is exactly the signature :class:`vendirag.VendiReranker` expects.
        """
        query_emb = self.embedder.encode([query])[0]
        hits = self.store.query(
            query_embeddings=[query_emb.tolist()],
            n_results=n,
            include=["documents", "metadatas", "embeddings"],
        )
        docs = [
            Document(text=text, metadata=meta or {}, id=doc_id)
            for text, meta, doc_id in zip(
                hits["documents"][0], hits["metadatas"][0], hits["ids"][0]
            )
        ]
        return docs, np.asarray(hits["embeddings"][0], dtype=np.float64), query_emb

    def retriever(self, s: float = 0.8, k: int = 5, candidate_pool: int = 50) -> VendiReranker:
        """Vendi retrieval over this Chroma store."""
        return VendiReranker(self.candidates, s=s, k=k, candidate_pool=candidate_pool)


def for_dataset(dataset: str, root: Optional[str] = None) -> ChromaCorpus:
    """Conventional paths: ``processed_data/<dataset>/`` and ``vectorDB/<dataset>``."""
    root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return ChromaCorpus(
        dataset_path=os.path.join(root, "processed_data", dataset),
        persist_directory=os.path.join(root, "vectorDB", dataset),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="hotpotqa", choices=DATASETS)
    parser.add_argument("--build", action="store_true", help="build the index")
    parser.add_argument("--query", default=None, help="try a query against it")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("-s", type=float, default=0.8)
    args = parser.parse_args()

    corpus = for_dataset(args.dataset)
    if args.build:
        corpus.build()
    if args.query:
        for i, doc in enumerate(corpus.retriever(s=args.s, k=args.k).retrieve(args.query), 1):
            print(f"{i}. [{doc.metadata.get('title')}] {doc.text[:160]}")
