"""
Vendi Score primitives.

The Vendi Score (VS) of a set of items is the exponential of the Shannon
entropy of the eigenvalues of their normalized similarity matrix
(Friedman & Dieng, TMLR 2023):

    VS_k(D) = exp( -sum_i lambda_i log lambda_i ),

where lambda_1..lambda_n are the eigenvalues of K / n, with K_ij = k(d_i, d_j)
a positive semi-definite kernel with unit diagonal.  It is interpretable as the
*effective number of unique items* in the set: 1 when every item is identical,
n when the items are mutually orthogonal.

Only numpy is required here, so this module (and the retriever built on it) can
be used without pulling in any LLM, vector-store, or deep-learning dependency.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "entropy_q",
    "vendi_score_from_kernel",
    "vendi_score",
    "normalized_vendi_score",
]

_EIG_FLOOR = 1e-12


def entropy_q(p: np.ndarray, q: float | str = 1.0) -> float:
    """Renyi entropy of order ``q`` of a probability vector ``p``.

    ``q=1`` gives Shannon entropy (the default used by the Vendi Score),
    ``q="inf"`` gives min-entropy.  Zero entries are dropped.
    """
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    if q == "inf":
        return float(-np.log(np.max(p)))
    if float(q) == 1.0:
        return float(-(p * np.log(p)).sum())
    q = float(q)
    return float(np.log((p ** q).sum()) / (1.0 - q))


def vendi_score_from_kernel(K: np.ndarray, q: float | str = 1.0) -> float:
    """Vendi Score of a similarity matrix ``K`` with unit diagonal."""
    K = np.asarray(K, dtype=np.float64)
    n = K.shape[0]
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0
    # Symmetrize to kill any asymmetry from floating-point error before eigvalsh.
    K = 0.5 * (K + K.T)
    w = np.linalg.eigvalsh(K / n)
    w = w[w > _EIG_FLOOR]
    return float(np.exp(entropy_q(w, q=q)))


def vendi_score(X: np.ndarray, q: float | str = 1.0, normalize: bool = True) -> float:
    """Vendi Score of embeddings ``X`` of shape ``(n, d)`` under a cosine kernel.

    Parameters
    ----------
    X : array of shape (n, d)
        One row per item.
    q : float
        Renyi order.  ``q=1`` (default) is the standard Vendi Score.
    normalize : bool
        L2-normalize the rows first so the kernel has unit diagonal.  Set to
        ``False`` only when ``X`` is already normalized.

    Returns
    -------
    float in ``[1, n]`` — the effective number of unique items.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"expected a 2-D (n, d) array, got shape {X.shape}")
    if len(X) == 0:
        return 0.0
    if normalize:
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return vendi_score_from_kernel(X @ X.T, q=q)


def normalized_vendi_score(X: np.ndarray, q: float | str = 1.0, normalize: bool = True) -> float:
    """Vendi Score rescaled from ``[1, n]`` to ``[0, 1]``.

        VS~(D) = (VS(D) - 1) / (n - 1)

    This is the diversity term of the Vendi Retrieval Score (Appendix A.5 of the
    paper); the rescaling keeps it on the same footing as the relevance term.
    Singleton sets score 0.0, since VS is always exactly 1 there.
    """
    X = np.asarray(X)
    n = len(X)
    if n <= 1:
        return 0.0
    return (vendi_score(X, q=q, normalize=normalize) - 1.0) / (n - 1.0)
