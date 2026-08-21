"""Lightweight data containers shared across the package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

__all__ = ["Document", "RetrievalResult"]


@dataclass
class Document:
    """A retrievable chunk of text plus arbitrary metadata.

    Duck-typed to interoperate with LangChain: ``page_content`` is available as
    a property, and :meth:`coerce` accepts LangChain ``Document`` objects, plain
    strings, or dicts.
    """

    text: str
    metadata: dict = field(default_factory=dict)
    id: Optional[str] = None
    score: Optional[float] = None

    @property
    def page_content(self) -> str:  # LangChain compatibility
        return self.text

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.text

    @classmethod
    def coerce(cls, obj: Any) -> "Document":
        """Build a ``Document`` from a string, dict, or any object exposing
        ``page_content`` / ``text``."""
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, str):
            return cls(text=obj)
        if isinstance(obj, dict):
            text = obj.get("text", obj.get("page_content", ""))
            return cls(
                text=text,
                metadata=obj.get("metadata", {}) or {},
                id=obj.get("id"),
            )
        text = getattr(obj, "page_content", None) or getattr(obj, "text", None)
        if text is None:
            raise TypeError(f"cannot coerce {type(obj).__name__} to Document")
        return cls(text=text, metadata=dict(getattr(obj, "metadata", {}) or {}))


@dataclass
class RetrievalResult:
    """Everything the retriever knows about one selection, for inspection."""

    documents: list
    indices: list
    #: Vendi Score of the selected set (effective number of unique documents).
    vendi_score: float
    #: Mean query-document cosine similarity of the selected set.
    mean_similarity: float
    #: Final value of the Vendi Retrieval Score objective.
    vrs: float
    #: Diversity weight used for this selection.
    s: float
    #: Candidate pool the selection was made from (indices into the index).
    candidate_indices: list = field(default_factory=list)
    query_embedding: Optional[np.ndarray] = None

    @property
    def texts(self) -> list:
        return [d.text for d in self.documents]

    def __len__(self) -> int:
        return len(self.documents)

    def __iter__(self):
        return iter(self.documents)
