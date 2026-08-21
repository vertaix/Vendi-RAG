"""Pluggable text embedders.

Anything with an ``encode(texts) -> (n, d) array`` method works as an embedder
in this package.  Two implementations ship here:

* :class:`HashingEmbedder` — pure numpy, no downloads, deterministic.  Good
  enough for demos, tests, and small corpora, and it lets the whole library run
  offline with numpy as the only dependency.
* :class:`SentenceTransformerEmbedder` — wraps ``sentence-transformers``
  (``all-mpnet-base-v2`` by default, the encoder used in the paper).

:func:`auto_embedder` picks the strongest one available.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Iterable, Optional, Sequence

import numpy as np

__all__ = [
    "Embedder",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "auto_embedder",
    "l2_normalize",
]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def l2_normalize(X: np.ndarray, axis: int = -1) -> np.ndarray:
    """L2-normalize along ``axis``, leaving all-zero rows untouched."""
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=axis, keepdims=True)
    return X / np.maximum(norms, 1e-12)


class Embedder:
    """Minimal embedder interface. Subclass or duck-type."""

    def encode(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    # Convenience aliases so LangChain-style callers keep working.
    def embed_documents(self, texts: Sequence[str]) -> list:
        return self.encode(list(texts)).tolist()

    def embed_query(self, text: str) -> list:
        return self.encode([text])[0].tolist()

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode(texts)


def _tokenize(text: str, ngram_max: int = 2) -> list:
    words = _TOKEN_RE.findall(text.lower())
    tokens = list(words)
    for n in range(2, ngram_max + 1):
        tokens.extend(" ".join(words[i:i + n]) for i in range(len(words) - n + 1))
    return tokens


def _bucket(token: str, dim: int) -> tuple:
    """Deterministic (bucket, sign) for a token.

    ``hashlib`` rather than ``hash()`` because the built-in is salted per
    process, which would make embeddings non-reproducible across runs.
    """
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    return value % dim, 1.0 if (value >> 63) & 1 else -1.0


class HashingEmbedder(Embedder):
    """Deterministic bag-of-ngrams embedder using the hashing trick.

    Sublinear term frequency with optional inverse document frequency, hashed
    into ``dim`` signed buckets and L2-normalized, so the dot product of two
    embeddings is their cosine similarity.

    Parameters
    ----------
    dim : int
        Output dimensionality.
    ngram_max : int
        Include word n-grams up to this length (2 captures short phrases).
    use_idf : bool
        Down-weight terms that appear in many documents.  Requires :meth:`fit`;
        :class:`~vendirag.retriever.VendiRetriever` fits automatically at index
        time.
    """

    def __init__(self, dim: int = 1024, ngram_max: int = 2, use_idf: bool = True):
        self.dim = int(dim)
        self.ngram_max = int(ngram_max)
        self.use_idf = bool(use_idf)
        self._idf: dict = {}
        self._default_idf: float = 1.0
        self._fitted = False

    def fit(self, texts: Iterable[str]) -> "HashingEmbedder":
        """Estimate inverse document frequencies from a corpus."""
        texts = list(texts)
        n_docs = max(len(texts), 1)
        df: Counter = Counter()
        for text in texts:
            df.update(set(_tokenize(text, self.ngram_max)))
        self._idf = {
            token: math.log((n_docs + 1) / (count + 1)) + 1.0
            for token, count in df.items()
        }
        # Terms unseen at fit time are treated as maximally informative.
        self._default_idf = math.log(n_docs + 1) + 1.0
        self._fitted = True
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        out = np.zeros((len(texts), self.dim), dtype=np.float64)
        for row, text in enumerate(texts):
            counts = Counter(_tokenize(text, self.ngram_max))
            for token, tf in counts.items():
                weight = 1.0 + math.log(tf)
                if self.use_idf and self._fitted:
                    weight *= self._idf.get(token, self._default_idf)
                bucket, sign = _bucket(token, self.dim)
                out[row, bucket] += sign * weight
        return l2_normalize(out, axis=1)


class SentenceTransformerEmbedder(Embedder):
    """Wraps a ``sentence-transformers`` model (the paper uses all-mpnet-base-v2)."""

    def __init__(
        self,
        model_name: str = "all-mpnet-base-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "SentenceTransformerEmbedder needs the 'sentence-transformers' "
                "package: pip install 'vendirag[embeddings]'"
            ) from exc
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return l2_normalize(np.asarray(vectors, dtype=np.float64), axis=1)


def auto_embedder(prefer: str = "sentence-transformers", **kwargs) -> Embedder:
    """Return the best embedder available in this environment.

    Falls back to :class:`HashingEmbedder` when ``sentence-transformers`` is not
    installed, so calling code never has to branch on what is present.
    """
    if prefer == "sentence-transformers":
        try:
            return SentenceTransformerEmbedder(**kwargs)
        except ImportError:
            pass
    return HashingEmbedder(**{k: v for k, v in kwargs.items()
                              if k in {"dim", "ngram_max", "use_idf"}})
