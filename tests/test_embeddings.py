"""The built-in embedder."""

import numpy as np
import pytest

from vendirag import HashingEmbedder, auto_embedder
from vendirag.embeddings import l2_normalize


def test_encoding_is_deterministic_across_instances():
    a = HashingEmbedder(dim=256).encode(["hello world"])
    b = HashingEmbedder(dim=256).encode(["hello world"])
    assert np.allclose(a, b)


def test_rows_are_unit_length():
    X = HashingEmbedder(dim=128).encode(["one", "two words here", "three"])
    assert np.allclose(np.linalg.norm(X, axis=1), 1.0)


def test_similar_text_scores_higher_than_unrelated_text():
    embedder = HashingEmbedder(dim=512, use_idf=False)
    X = embedder.encode([
        "the expedition was led by a glaciologist",
        "the expedition was led by a geologist",
        "orange marmalade on toast",
    ])
    assert X[0] @ X[1] > X[0] @ X[2]


def test_idf_needs_fitting_and_changes_the_result():
    unfitted = HashingEmbedder(dim=256).encode(["the cat sat"])
    fitted = HashingEmbedder(dim=256).fit(["the cat sat", "the dog sat", "the bird sat"])
    assert not np.allclose(unfitted, fitted.encode(["the cat sat"]))


def test_empty_text_does_not_blow_up():
    X = HashingEmbedder(dim=64).encode(["", "something"])
    assert X.shape == (2, 64) and np.isfinite(X).all()


def test_l2_normalize_leaves_zero_rows_finite():
    out = l2_normalize(np.zeros((2, 3)))
    assert np.isfinite(out).all() and (out == 0).all()


def test_langchain_style_aliases():
    embedder = HashingEmbedder(dim=32)
    assert len(embedder.embed_query("hi")) == 32
    assert len(embedder.embed_documents(["a", "b"])) == 2


def test_auto_embedder_always_returns_something():
    assert hasattr(auto_embedder(), "encode")
