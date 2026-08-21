"""The Vendi-RAG loop: control flow, the EMA on s, and the offline backend."""

import numpy as np
import pytest

from vendirag import (
    CallableLLM,
    HashingEmbedder,
    HeuristicBackend,
    Judgement,
    PromptedBackend,
    VendiRAG,
    VendiRetriever,
)
from vendirag.llm import extract_json
from vendirag.pipeline import Backend
from vendirag.toy import make_corpus


@pytest.fixture(scope="module")
def corpus():
    return make_corpus(n_chains=8, seed=0)


@pytest.fixture(scope="module")
def retriever(corpus):
    # Pinned so the suite is deterministic and downloads nothing.
    return VendiRetriever(
        embedder=HashingEmbedder(), k=8, candidate_pool=50
    ).index(corpus.documents)


class ScriptedBackend(Backend):
    """Returns a fixed sequence of quality scores, to drive s deterministically."""

    def __init__(self, qualities):
        self.qualities = list(qualities)
        self.calls = 0

    def cot(self, question, documents):
        return "reasoning"

    def answer(self, question, documents, reasoning):
        return f"answer {self.calls}"

    def judge(self, question, documents, answer):
        q = self.qualities[min(self.calls, len(self.qualities) - 1)]
        self.calls += 1
        return Judgement(q * 10, q * 10, q * 10)

    def refine(self, question, answer, reasoning):
        return question


def test_early_stop_when_quality_clears_tau(retriever):
    rag = VendiRAG(retriever, backend=ScriptedBackend([0.9]), max_iterations=5)
    result = rag.answer("anything")
    assert result.n_iterations == 1 and result.stopped_early
    assert result.quality == pytest.approx(0.9)


def test_no_early_stop_runs_every_iteration(retriever):
    rag = VendiRAG(retriever, backend=ScriptedBackend([0.9]),
                   max_iterations=4, early_stopping=False)
    assert rag.answer("anything").n_iterations == 4


def test_best_answer_survives_a_later_bad_iteration(retriever):
    rag = VendiRAG(retriever, backend=ScriptedBackend([0.6, 0.1, 0.1]),
                   max_iterations=3)
    result = rag.answer("anything")
    assert result.answer == "answer 0"
    assert result.quality == pytest.approx(0.6)


def test_s_rises_when_quality_falls_below_the_running_best(retriever):
    # 0.8 sets the best; 0.2 is then far below it, so s_target -> 0.75.
    rag = VendiRAG(retriever, backend=ScriptedBackend([0.8, 0.2, 0.2]),
                   max_iterations=3, initial_s=0.5, beta=0.3)
    trajectory = rag.answer("anything").s_trajectory
    assert trajectory[1] < trajectory[0]     # first answer is the best so far
    assert trajectory[2] > trajectory[1]     # quality dropped, diversify again


def test_fixed_s_never_moves(retriever):
    rag = VendiRAG(retriever, backend=ScriptedBackend([0.3, 0.9, 0.2]),
                   max_iterations=3, initial_s=0.8, dynamic_s=False,
                   early_stopping=False)
    assert rag.answer("anything").s_trajectory == [0.8, 0.8, 0.8]


def test_s_stays_inside_the_unit_interval(retriever):
    rag = VendiRAG(retriever, backend=ScriptedBackend([0.05, 0.9, 0.01, 0.5]),
                   max_iterations=4, initial_s=0.8, early_stopping=False)
    assert all(0.0 <= s <= 1.0 for s in rag.answer("anything").s_trajectory)


def test_pipeline_requires_an_llm_or_backend(retriever):
    with pytest.raises(ValueError):
        VendiRAG(retriever)


def test_offline_pipeline_answers_multi_hop_questions(corpus, retriever):
    rag = VendiRAG.offline(retriever, k_docs=8, k_candidates=50,
                           initial_s=0.8, dynamic_s=False)
    correct = [
        rag.answer(q.question).answer.strip().lower() == q.answer.strip().lower()
        for q in corpus.questions
    ]
    assert np.mean(correct) > 0.5


def test_diverse_retrieval_beats_pure_similarity_end_to_end(corpus, retriever):
    def accuracy(s):
        rag = VendiRAG.offline(retriever, k_docs=8, k_candidates=50,
                               initial_s=s, dynamic_s=False)
        return np.mean([
            rag.answer(q.question).answer.strip().lower() == q.answer.strip().lower()
            for q in corpus.questions
        ])

    assert accuracy(0.8) > accuracy(0.0) + 0.2


def test_prompted_backend_parses_json_and_records_calls():
    replies = {
        "chain-of-thought": '{"reasoning": "step one"}',
        "concise and precise": '```json\n{"answer": "42"}\n```',
        "expert LLM-based judge": 'Sure: {"coherence": 9, "relevance": 8, "query_alignment": 10}',
        "Reformulate": '{"refined_question": "narrower?"}',
    }

    def fake(prompt):
        for marker, reply in replies.items():
            if marker in prompt:
                return reply
        raise AssertionError("unexpected prompt")

    backend = PromptedBackend(CallableLLM(fake))
    assert backend.cot("q", "docs") == "step one"
    assert backend.answer("q", "docs", "r") == "42"
    assert backend.judge("q", "docs", "42").quality == pytest.approx(27 / 30)
    assert backend.refine("q", "a", "r") == "narrower?"
    assert len(backend.calls) == 4


def test_prompted_backend_survives_a_broken_llm():
    def broken(prompt):
        raise RuntimeError("api down")

    backend = PromptedBackend(CallableLLM(broken))
    assert backend.cot("q", "d") == ""
    assert backend.judge("q", "d", "a").quality == pytest.approx(0.5)
    assert backend.refine("q", "a", "r") == "q"


@pytest.mark.parametrize("text,expected", [
    ('{"a": 1}', {"a": 1}),
    ('```json\n{"a": 1}\n```', {"a": 1}),
    ('Here you go: {"a": 1}. Hope that helps!', {"a": 1}),
    ("not json at all", {}),
    ("", {}),
])
def test_extract_json(text, expected):
    assert extract_json(text) == expected


def test_heuristic_backend_is_deterministic(corpus, retriever):
    rag = VendiRAG.offline(retriever, k_docs=8, k_candidates=50)
    question = corpus.questions[0].question
    assert rag.answer(question).answer == rag.answer(question).answer


def test_heuristic_backend_reports_missing_evidence():
    backend = HeuristicBackend()
    docs = "The Thule Expedition departed in the spring of 1962."
    question = "Which instrument was used by the leader of the Thule Expedition?"
    answer = backend.answer(question, docs, "")
    assert answer.startswith("Insufficient")
    assert backend.judge(question, docs, answer).quality < 0.2
