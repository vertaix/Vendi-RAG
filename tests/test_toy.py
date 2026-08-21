"""The synthetic corpus, and the claim the demo rests on."""

import numpy as np
import pytest

from vendirag import HashingEmbedder, VendiRetriever
from vendirag.toy import evaluate_retrieval, make_corpus


def test_corpus_shape_is_deterministic():
    a, b = make_corpus(n_chains=6, seed=0), make_corpus(n_chains=6, seed=0)
    assert [d.text for d in a.documents] == [d.text for d in b.documents]
    assert len(a.documents) == 6 * (4 + 12)
    assert len(a.questions) == 18


def test_every_question_has_a_complete_gold_chain():
    corpus = make_corpus(n_chains=5, seed=1)
    ids = {d.id for d in corpus.documents}
    for q in corpus.questions:
        assert len(q.gold_ids) == q.n_hops
        assert set(q.gold_ids) <= ids
        assert q.answer


def test_answers_appear_in_their_gold_documents():
    corpus = make_corpus(n_chains=5, seed=1)
    for q in corpus.questions:
        evidence = " ".join(corpus.by_id(g).text for g in q.gold_ids)
        assert q.answer in evidence


def test_chains_share_no_entities():
    """The walk in HeuristicBackend relies on chains being disjoint."""
    from vendirag.llm import HeuristicBackend

    corpus = make_corpus(n_chains=12, seed=0)
    owners = {}
    for doc in corpus.documents:
        for entity in HeuristicBackend.entities(doc.text):
            owners.setdefault(entity, set()).add(doc.metadata["chain"])
    shared = {e: c for e, c in owners.items() if len(c) > 1}
    assert not shared, f"entities shared between chains: {sorted(shared)}"


def test_bad_chain_count_is_rejected():
    with pytest.raises(ValueError):
        make_corpus(n_chains=0)
    with pytest.raises(ValueError):
        make_corpus(n_chains=1000)


def test_similarity_search_falls_into_the_redundancy_trap():
    corpus = make_corpus(n_chains=20, seed=0)
    retriever = VendiRetriever(embedder=HashingEmbedder(), k=8, candidate_pool=50).index(corpus.documents)
    plain = evaluate_retrieval(
        corpus.questions, lambda q: retriever.similarity_search(q, k=8)
    )
    assert plain["hop_coverage"] < 0.05
    assert plain["distractor_rate"] > 0.9


def test_vendi_retrieval_escapes_it():
    corpus = make_corpus(n_chains=20, seed=0)
    retriever = VendiRetriever(embedder=HashingEmbedder(), k=8, candidate_pool=50).index(corpus.documents)
    plain = evaluate_retrieval(
        corpus.questions, lambda q: retriever.similarity_search(q, k=8)
    )
    diverse = evaluate_retrieval(
        corpus.questions, lambda q: retriever.retrieve(q, k=8, s=0.6)
    )
    assert diverse["hop_coverage"] > 5 * plain["hop_coverage"]
    mean_vs = np.mean([
        retriever.set_diversity([d.text for d in retriever.retrieve(q.question, k=8, s=0.6)])
        for q in corpus.questions[:20]
    ])
    assert mean_vs > 6.0
