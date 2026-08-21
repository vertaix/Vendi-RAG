"""
Plots and animations of a Vendi retrieval.

Optional module — needs ``matplotlib`` and ``pillow``::

    pip install "vendirag[viz]"

The interesting thing to look at is what happens to the *selected set* as the
diversity weight ``s`` sweeps from pure relevance to pure diversity, so that is
what :func:`make_selection_gif` animates.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import numpy as np

__all__ = ["project_2d", "project_relevance_spread", "sweep_s",
           "plot_selection", "make_selection_gif", "default_classify"]

# A palette that survives both GitHub themes.
INK = "#1c1c1e"
MUTED = "#9aa0a6"
OTHER = "#d6d9dd"
REDUNDANT = "#d93025"
EVIDENCE = "#e8710a"
PICK = "#1a73e8"
QUERY = "#111111"
GRID = "#e8eaed"

#: How the scatter colours a candidate document.
GROUPS = ("evidence", "redundant", "other")
_GROUP_STYLE = {
    "other":     dict(c=OTHER,     s=24,  m="o", label="unrelated", z=2),
    "redundant": dict(c=REDUNDANT, s=30,  m="o", label="near-duplicates", z=3),
    "evidence":  dict(c=EVIDENCE,  s=150, m="D", label="evidence", z=4),
}


def project_2d(X: np.ndarray, query: Optional[np.ndarray] = None):
    """PCA to two dimensions via SVD, keeping the query in the same frame."""
    X = np.asarray(X, dtype=np.float64)
    stack = X if query is None else np.vstack([X, query.reshape(1, -1)])
    centred = stack - stack.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    coords = centred @ vt[:2].T
    return (coords[:-1], coords[-1]) if query is not None else (coords, None)


def project_relevance_spread(X: np.ndarray, query: np.ndarray):
    """Lay candidates out on axes that say what the objective is trading off.

    The horizontal axis is cosine similarity to the query — the relevance term.
    The vertical axis is the leading direction of variation *orthogonal* to the
    query, which is where the diversity term does its work.  A relevance-only
    retriever therefore takes a vertical slice off the right-hand edge, and
    raising ``s`` fans the selection out along the vertical.

    Returns ``(coords, query_xy)``.
    """
    X = np.asarray(X, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64).ravel()
    query = query / max(np.linalg.norm(query), 1e-12)

    x = X @ query
    residual = X - np.outer(x, query)
    residual = residual - residual.mean(axis=0, keepdims=True)
    if residual.shape[0] > 1 and np.any(residual):
        _, _, vt = np.linalg.svd(residual, full_matrices=False)
        y = residual @ vt[0]
    else:
        y = np.zeros(len(X))
    # The query itself sits at similarity 1 with no orthogonal component.
    return np.column_stack([x, y]), np.array([1.0, 0.0])


def sweep_s(
    retriever,
    questions: Sequence,
    s_values: Sequence[float],
    k: int = 8,
    candidate_pool: int = 50,
    is_evidence: Optional[Callable] = None,
):
    """Run the retriever across ``s_values`` and score each setting.

    ``is_evidence(question, document) -> bool`` marks the documents that
    actually answer the question; defaults to ``metadata["role"] == "hop"``.

    Returns ``(vendi_scores, evidence_recall)``, both arrays aligned to
    ``s_values`` and averaged over ``questions``.
    """
    if is_evidence is None:
        def is_evidence(question, doc):  # noqa: ARG001
            return getattr(doc, "metadata", {}).get("role") == "hop"

    vs, recall = [], []
    for s in s_values:
        per_q_vs, per_q_hit = [], []
        for q in questions:
            text = q if isinstance(q, str) else q.question
            result = retriever.retrieve_details(
                text, k=k, s=s, candidate_pool=candidate_pool
            )
            per_q_vs.append(result.vendi_score)
            per_q_hit.append(any(is_evidence(q, d) for d in result.documents))
        vs.append(float(np.mean(per_q_vs)))
        recall.append(float(np.mean(per_q_hit)))
    return np.array(vs), np.array(recall)


def default_classify(question, doc) -> str:
    """Group a candidate as evidence for this question, a near-duplicate of the
    question's topic, or an unrelated document.

    Works off the toy corpus's metadata; pass your own ``classify`` to
    :func:`make_selection_gif` for a different corpus.
    """
    meta = getattr(doc, "metadata", {}) or {}
    gold = set(getattr(question, "gold_ids", []) or [])
    if gold and getattr(doc, "id", None) in gold:
        return "evidence"
    if not gold and meta.get("role") == "hop":
        return "evidence"
    chain = getattr(question, "chain", None)
    if chain is not None and meta.get("chain") == chain:
        return "redundant"
    return "other"


def plot_selection(
    ax,
    coords: np.ndarray,
    query_xy: np.ndarray,
    kinds: Sequence[str],
    selected: Sequence[int],
    title: str = "",
    legend: bool = True,
):
    """Scatter the candidate pool and ring the currently selected documents."""
    ax.clear()
    kinds = np.asarray(kinds)
    for group in GROUPS[::-1]:
        mask = kinds == group
        if not mask.any():
            continue
        style = _GROUP_STYLE[group]
        ax.scatter(coords[mask, 0], coords[mask, 1], s=style["s"], c=style["c"],
                   marker=style["m"],
                   edgecolors="white" if group == "evidence" else "none",
                   linewidths=1.0, zorder=style["z"], label=style["label"])
    sel = np.asarray(selected, dtype=int)
    if len(sel):
        ax.scatter(coords[sel, 0], coords[sel, 1], s=210, facecolors="none",
                   edgecolors=PICK, linewidths=2.0, zorder=5,
                   label="retrieved")
    if query_xy is not None:
        ax.scatter([query_xy[0]], [query_xy[1]], marker="*", s=340, c=QUERY,
                   edgecolors="white", linewidths=1.0, zorder=6, label="query")
    # A couple of far outliers would otherwise squash the whole pool into a
    # corner, so frame on the bulk and let them clip.
    for axis, values in ((ax.set_xlim, coords[:, 0]), (ax.set_ylim, coords[:, 1])):
        lo, hi = np.percentile(values, [2, 98])
        pad = 0.12 * max(hi - lo, 1e-9)
        axis(lo - pad, hi + pad)
    if title:
        ax.set_title(title, fontsize=9.5, color=MUTED, pad=7)
    ax.set_xlabel("relevance to the query  \u2192", fontsize=9, color=MUTED, labelpad=3)
    ax.set_ylabel("semantic spread  \u2192", fontsize=9, color=MUTED, labelpad=3)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(GRID)
    if legend:
        ax.legend(loc="lower left", fontsize=8.5, frameon=True, framealpha=0.94,
                  edgecolor=GRID, borderpad=0.5, labelspacing=0.4,
                  handletextpad=0.5)


def make_selection_gif(
    retriever,
    question,
    path: str = "vendi-retrieval.gif",
    questions: Optional[Sequence] = None,
    k: int = 8,
    candidate_pool: int = 50,
    s_values: Optional[Sequence[float]] = None,
    fps: float = 4.0,
    hold: int = 5,
    classify: Optional[Callable] = None,
    title: str = "Vendi retrieval: one knob, from redundant to complete",
    dpi: int = 100,
):
    """Animate a Vendi retrieval as the diversity weight ``s`` sweeps 0 -> 1.

    Left: the candidate pool, laid out by similarity to the query against the
    main axis of variation orthogonal to it, with the selected documents ringed.
    At ``s = 0`` the selection is a vertical slice off the right-hand edge; as
    ``s`` rises it fans out and reaches the evidence.

    Right: the same sweep aggregated over ``questions`` — set diversity and how
    often the evidence is retrieved.

    Parameters
    ----------
    retriever : VendiRetriever
        Already indexed.
    question : str, or an object with ``.question`` / ``.gold_ids`` / ``.chain``
        The single query drawn on the left.
    questions : sequence, optional
        Questions the aggregate curves are averaged over.  Defaults to
        ``[question]``.
    classify : callable, optional
        ``classify(question, doc)`` returning ``"evidence"``, ``"redundant"``,
        or ``"other"``.  Defaults to :func:`default_classify`.
    hold : int
        Extra frames held at each end of the sweep so the loop reads clearly.

    Returns the written path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    text = question if isinstance(question, str) else question.question
    classify = classify or default_classify
    if s_values is None:
        s_values = np.round(np.linspace(0.0, 1.0, 21), 3)
    s_values = list(s_values)
    if questions is None:
        questions = [question]

    # Everything expensive happens once, before the first frame is drawn.
    base = retriever.retrieve_details(text, k=k, s=0.0, candidate_pool=candidate_pool)
    pool_idx = base.candidate_indices
    pool_docs = [retriever.documents[i] for i in pool_idx]
    # The horizontal axis *is* query similarity here, so a query marker would
    # sit far off the pool's range and squash it; the axis label carries it.
    coords, _ = project_relevance_spread(
        retriever.embeddings[pool_idx], base.query_embedding
    )
    kinds = [classify(question, d) for d in pool_docs]

    selections, stats = [], []
    for s in s_values:
        result = retriever.retrieve_details(text, k=k, s=s, candidate_pool=candidate_pool)
        selections.append([pool_idx.index(i) for i in result.indices if i in pool_idx])
        stats.append((
            result.vendi_score,
            any(classify(question, d) == "evidence" for d in result.documents),
        ))
    curve_vs, curve_recall = sweep_s(
        retriever, questions, s_values, k=k, candidate_pool=candidate_pool,
        is_evidence=lambda q, d: classify(q, d) == "evidence",
    )

    frames = [0] * hold + list(range(len(s_values))) + [len(s_values) - 1] * hold

    fig = plt.figure(figsize=(9.6, 4.5), dpi=dpi, facecolor="white")
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.25, 1.0],
        left=0.045, right=0.975, top=0.755, bottom=0.105, wspace=0.16,
    )
    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_curve = fig.add_subplot(gs[0, 1])
    fig.suptitle(title, fontsize=13.5, color=INK, y=0.955, fontweight="bold")
    readout = fig.text(0.5, 0.855, "", ha="center", fontsize=12, color=INK)

    def draw(fi: int):
        i = frames[fi]
        s = s_values[i]
        vs, found = stats[i]
        plot_selection(ax_scatter, coords, None, kinds, selections[i])
        readout.set_text(
            f"s = {s:.2f}       {vs:.1f} of {k} retrieved documents are distinct"
            f"       evidence: {'found' if found else 'missed'}"
        )
        readout.set_color(EVIDENCE if found else MUTED)

        ax_curve.clear()
        ax_curve.plot(s_values, curve_vs / k, color=PICK, lw=2.2,
                      label="distinct documents")
        ax_curve.plot(s_values, curve_recall, color=EVIDENCE, lw=2.2,
                      label="evidence found")
        ax_curve.axvline(s, color=INK, lw=1.1, ls="--", alpha=0.65)
        ax_curve.scatter([s, s], [curve_recall[i], curve_vs[i] / k], s=44,
                         color=[EVIDENCE, PICK], zorder=5)
        ax_curve.set_xlim(-0.02, 1.02); ax_curve.set_ylim(-0.05, 1.12)
        ax_curve.set_xlabel("diversity weight   s", fontsize=9.5, color=INK, labelpad=2)
        ax_curve.set_yticks([0, 0.5, 1.0])
        ax_curve.tick_params(labelsize=8.5, colors=MUTED)
        ax_curve.grid(alpha=0.35, color=GRID)
        ax_curve.legend(fontsize=9, loc="lower right", frameon=False)
        ax_curve.set_title(f"across {len(questions)} questions",
                           fontsize=9.5, color=MUTED, pad=4)
        for spine in ax_curve.spines.values():
            spine.set_color(GRID)
        return []

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=1000 / fps, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return path
