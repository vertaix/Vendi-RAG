"""Selection behaviour of the Vendi retriever."""

import numpy as np
import pytest

from vendirag import (
    Document,
    HashingEmbedder,
    VendiReranker,
    VendiRetriever,
    mmr_select,
    vendi_rerank,
    vrs_select,
)


@pytest.fixture
def clustered():
    """Three tight clusters in 6-D; the query sits on the first."""
    rng = np.random.default_rng(0)
    centres = np.eye(3, 6)
    embs = np.vstack([c + 0.02 * rng.normal(size=(8, 6)) for c in centres])
    return embs, centres[0]


def test_s_zero_is_plain_similarity_ranking(clustered):
    embs, query = clustered
    picked = vrs_select(embs, query, k=5, s=0.0)
    unit = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    top5 = list(np.argsort(-(unit @ query))[:5])
    assert picked == [int(i) for i in top5]


def test_high_s_spreads_across_clusters(clustered):
    embs, query = clustered
    diverse = vrs_select(embs, query, k=3, s=0.9)
    clusters = {i // 8 for i in diverse}
    assert clusters == {0, 1, 2}
    greedy = vrs_select(embs, query, k=3, s=0.0)
    assert {i // 8 for i in greedy} == {0}


def test_first_pick_is_always_the_most_relevant(clustered):
    embs, query = clustered
    unit = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    best = int(np.argmax(unit @ query))
    for s in (0.0, 0.5, 1.0):
        assert vrs_select(embs, query, k=4, s=s)[0] == best


def test_selection_has_no_duplicates_and_respects_k(clustered):
    embs, query = clustered
    picked = vrs_select(embs, query, k=7, s=0.6)
    assert len(picked) == len(set(picked)) == 7
    assert len(vrs_select(embs, query, k=99, s=0.6)) == len(embs)


def test_empty_pool_and_bad_s(clustered):
    _, query = clustered
    assert vrs_select(np.zeros((0, 6)), query, k=3) == []
    with pytest.raises(ValueError):
        vrs_select(np.eye(4, 6), query, k=2, s=1.5)


def test_trace_records_every_step(clustered):
    embs, query = clustered
    picked, trace = vrs_select(embs, query, k=4, s=0.7, return_trace=True)
    assert [step.index for step in trace] == picked
    assert all(1.0 <= step.vendi_score <= len(picked) + 1e-9 for step in trace)


def test_rerank_returns_documents_when_given_them(clustered):
    embs, query = clustered
    docs = [Document(text=f"doc {i}") for i in range(len(embs))]
    out = vendi_rerank(query, embs, docs, k=3, s=0.8)
    assert len(out) == 3 and all(isinstance(d, Document) for d in out)
    assert vendi_rerank(query, embs, None, k=3, s=0.8) == [d for d in
        vrs_select(embs, query, k=3, s=0.8)]


def test_mmr_baseline_also_spreads(clustered):
    embs, query = clustered
    assert len({i // 8 for i in mmr_select(embs, query, k=3, lambda_mult=0.3)}) == 3


def test_retriever_drops_redundant_documents():
    texts = [
        "Cats purr when they are content.",
        "Cats purr when they are content and relaxed.",
        "Cats purr when they are content and calm.",
        "Dogs bark when they are alert.",
    ]
    retriever = VendiRetriever.from_texts(texts, embedder=HashingEmbedder(), k=2, candidate_pool=4)
    assert [d.text for d in retriever.similarity_search("Why do cats purr?", k=2)] == texts[:2]
    assert [d.text for d in retriever.retrieve("Why do cats purr?", s=0.8)] == [texts[0], texts[3]]


def test_retriever_preserves_ids_and_metadata():
    docs = [Document(text=f"text {i}", id=f"x{i}", metadata={"n": i}) for i in range(5)]
    retriever = VendiRetriever(embedder=HashingEmbedder(), k=2).index(docs)
    out = retriever.retrieve("text 3")
    assert all(d.id.startswith("x") for d in out)
    assert all("n" in d.metadata for d in out)


def test_retriever_requires_an_index():
    with pytest.raises(RuntimeError):
        VendiRetriever().retrieve("anything")
    with pytest.raises(ValueError):
        VendiRetriever().index([])


def test_retriever_accepts_precomputed_embeddings():
    embs = np.eye(4, 6)
    retriever = VendiRetriever(embedder=HashingEmbedder(), k=2).index(
        [f"d{i}" for i in range(4)], embeddings=embs)
    assert retriever.embeddings.shape == (4, 6)
    with pytest.raises(ValueError):
        VendiRetriever().index(["a", "b"], embeddings=np.eye(3, 6))


def test_reranker_matches_the_in_memory_retriever():
    texts = [f"document number {i} about topic {i % 4}" for i in range(40)]
    retriever = VendiRetriever.from_texts(texts, embedder=HashingEmbedder(), k=4, candidate_pool=20)

    def candidates(query, n):
        qe = retriever.embed_query(query)
        idx = np.argsort(-(retriever.embeddings @ qe))[:n]
        return [retriever.documents[i] for i in idx], retriever.embeddings[idx], qe

    reranker = VendiReranker(candidates, k=4, candidate_pool=20, s=0.7)
    assert [d.text for d in reranker.retrieve("topic 2")] == \
           [d.text for d in retriever.retrieve("topic 2", s=0.7)]
