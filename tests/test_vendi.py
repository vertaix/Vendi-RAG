"""Properties of the Vendi Score itself."""

import numpy as np
import pytest

from vendirag import normalized_vendi_score, vendi_score
from vendirag.vendi import entropy_q, vendi_score_from_kernel


def test_identical_items_score_one():
    X = np.tile(np.array([1.0, 2.0, 3.0]), (6, 1))
    assert vendi_score(X) == pytest.approx(1.0, abs=1e-6)


def test_orthogonal_items_score_n():
    X = np.eye(5)
    assert vendi_score(X) == pytest.approx(5.0, abs=1e-6)


@pytest.mark.parametrize("n", [2, 3, 7, 12])
def test_score_is_bounded_by_one_and_n(n):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 16))
    score = vendi_score(X)
    assert 1.0 - 1e-9 <= score <= n + 1e-9


def test_score_is_invariant_to_row_scaling():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(6, 8))
    scaled = X * rng.uniform(0.5, 4.0, size=(6, 1))
    assert vendi_score(X) == pytest.approx(vendi_score(scaled), abs=1e-9)


def test_adding_a_duplicate_adds_less_than_adding_a_novel_item():
    X = np.eye(4, 6)
    with_duplicate = np.vstack([X, X[0]])
    with_novel = np.vstack([X, np.eye(6)[4]])
    assert vendi_score(with_duplicate) < vendi_score(with_novel)


def test_normalized_score_spans_zero_to_one():
    assert normalized_vendi_score(np.tile([1.0, 0.0], (4, 1))) == pytest.approx(0.0, abs=1e-6)
    assert normalized_vendi_score(np.eye(4)) == pytest.approx(1.0, abs=1e-6)


def test_singleton_and_empty_sets():
    assert normalized_vendi_score(np.ones((1, 3))) == 0.0
    assert vendi_score(np.zeros((0, 3))) == 0.0
    assert vendi_score_from_kernel(np.ones((1, 1))) == 1.0


def test_entropy_orders():
    p = np.array([0.5, 0.3, 0.2])
    assert entropy_q(p, q=1) == pytest.approx(-(p * np.log(p)).sum())
    assert entropy_q(p, q="inf") == pytest.approx(-np.log(0.5))
    # Renyi entropy is non-increasing in q.
    assert entropy_q(p, q=0.5) >= entropy_q(p, q=1) >= entropy_q(p, q=2)
