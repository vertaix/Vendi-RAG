"""
The whole argument for Vendi-RAG, on synthetic data, in one runnable file.

Needs nothing but numpy (matplotlib only for ``--gif``): no API key, no model
download, no network.  Every number it prints is computed here, and the
generated corpus is deterministic, so the output is reproducible.

    python examples/toy_demo.py
    python examples/toy_demo.py --gif assets/vendi-retrieval.gif

The corpus (see ``vendirag.toy``) is a fictional world of Arctic expeditions.
Answering a question means chaining facts that live in separate documents, and
around each chain sits a thicket of near-duplicate documents that repeat the
expedition's name.  That thicket is what similarity search retrieves.
"""

from __future__ import annotations

import argparse

import numpy as np

from vendirag import HashingEmbedder, VendiRAG, VendiRetriever, mmr_select
from vendirag.toy import make_corpus

K = 8            # documents retrieved per query
POOL = 50        # candidate pool size |C|
RULE = "-" * 74


def banner(text: str) -> None:
    print(f"\n{RULE}\n{text}\n{RULE}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chains", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--distractors", type=int, default=12,
                        help="depth of the redundancy thicket around each chain")
    parser.add_argument("--gif", default=None, help="write the animation here")
    args = parser.parse_args()

    corpus = make_corpus(n_chains=args.chains, seed=args.seed,
                         n_distractors=args.distractors)
    # Pinned deliberately: the hashing embedder downloads nothing and gives the
    # same vectors on every machine, so these numbers are reproducible.  Swap in
    # SentenceTransformerEmbedder() to see the same mechanism with a real
    # encoder — a stronger encoder already handles some of the redundancy on its
    # own, so the gap narrows.
    retriever = VendiRetriever(
        embedder=HashingEmbedder(), k=K, candidate_pool=POOL
    ).index(corpus.documents)
    print(f"Corpus: {corpus.describe()}")
    print("Embedder: HashingEmbedder (deterministic, no download)")

    # Sections 1, 2 and 5 walk through a single question. Pick one where the two
    # retrievers actually disagree, so the contrast is visible rather than
    # asserted; the aggregate numbers in section 3 cover every question.
    def finds(q, **kwargs):
        gold = set(q.gold_ids)
        docs = (retriever.similarity_search(q.question, k=K) if not kwargs
                else retriever.retrieve(q.question, k=K, **kwargs))
        return any(d.id in gold for d in docs)

    question = next(
        (q for q in corpus.questions if not finds(q) and finds(q, s=0.8)),
        corpus.questions[0],
    )

    # ── 1. the failure mode ─────────────────────────────────────────────────
    banner("1. What plain similarity search returns")
    print(f"Q: {question.question}")
    print(f"   (a {question.n_hops}-hop question; > marks a document on its evidence chain)\n")
    top_k = retriever.similarity_search(question.question, k=K)
    for doc in top_k:
        mark = ">" if doc.id in set(question.gold_ids) else " "
        print(f"  {mark} {doc.text[:88]}")
    print(f"\n  {K} documents, but only "
          f"{retriever.set_diversity([d.text for d in top_k]):.1f} of them are "
          f"effectively distinct (Vendi Score) — the budget went on restatements.")
    print(f"  Evidence chain recovered: "
          f"{sum(d.id in set(question.gold_ids) for d in top_k)} of "
          f"{len(question.gold_ids)} documents.")

    # ── 2. the same query under Vendi retrieval ─────────────────────────────
    banner("2. The same pool, selected by the Vendi Retrieval Score (s = 0.8)")
    diverse = retriever.retrieve(question.question, k=K, s=0.8)
    for doc in diverse:
        mark = ">" if doc.id in set(question.gold_ids) else " "
        print(f"  {mark} {doc.text[:88]}")
    print(f"\n  {retriever.set_diversity([d.text for d in diverse]):.1f} of {K} "
          f"effectively distinct.")
    print(f"  Evidence chain recovered: "
          f"{sum(d.id in set(question.gold_ids) for d in diverse)} of "
          f"{len(question.gold_ids)} documents.")
    print("\n  Same candidate pool, same encoder, same budget. Only the selection rule")
    print("  changed — and the one document that names the leader is now in context.")

    # ── 3. across every question and every method ───────────────────────────
    banner(f"3. Retrieval across all {len(corpus.questions)} questions")

    def score(fn):
        hit = [any(d.id in set(q.gold_ids) for d in fn(q.question)) for q in corpus.questions]
        vs = [retriever.set_diversity([d.text for d in fn(q.question)]) for q in corpus.questions]
        return float(np.mean(hit)), float(np.mean(vs))

    def mmr(question_text: str, lam: float):
        qe = retriever.embed_query(question_text)
        pool = np.argsort(-(retriever.embeddings @ qe))[:POOL]
        idx = mmr_select(retriever.embeddings[pool], qe, k=K, lambda_mult=lam)
        return [retriever.documents[pool[i]] for i in idx]

    print(f"  {'method':<30}{'evidence found':>16}{'unique docs':>14}")
    methods = [("top-k similarity", lambda q: retriever.similarity_search(q, k=K))]
    methods += [(f"MMR (lambda = {lam})", lambda q, l=lam: mmr(q, l)) for lam in (0.5, 0.7)]
    methods += [(f"Vendi retrieval (s = {s})",
                 lambda q, s=s: retriever.retrieve(q, k=K, s=s, candidate_pool=POOL))
                for s in (0.2, 0.4, 0.6, 0.8, 1.0)]
    for name, fn in methods:
        recall, vs = score(fn)
        print(f"  {name:<30}{recall:>15.0%}{vs:>10.1f}/{K}")

    print("\n  Relevance alone almost never reaches the evidence: it spends the whole")
    print("  budget inside the thicket. Diversity-aware selection fixes that, and the")
    print("  optimum is interior — at s = 1 relevance is ignored and selection drifts")
    print("  off-topic, so recall falls again.")

    # ── 4. end to end ───────────────────────────────────────────────────────
    banner("4. End to end, with the offline reader (no API key)")
    print(f"  {'configuration':<30}{'exact match':>16}{'iterations':>14}")
    configs = [
        ("single-shot, top-k", dict(initial_s=0.0), True),
        ("single-shot, Vendi s = 0.8", dict(initial_s=0.8), True),
        ("iterative, top-k", dict(initial_s=0.0, dynamic_s=False), False),
        ("Vendi-RAG (fixed s = 0.8)", dict(initial_s=0.8, dynamic_s=False), False),
        ("Vendi-RAG (dynamic s)", dict(initial_s=0.8, dynamic_s=True), False),
    ]
    for name, kwargs, single in configs:
        rag = VendiRAG.offline(retriever, k_docs=K, k_candidates=POOL, **kwargs)
        results = [
            rag.answer_single_shot(q.question) if single else rag.answer(q.question)
            for q in corpus.questions
        ]
        em = np.mean([r.answer.strip().lower() == q.answer.strip().lower()
                      for r, q in zip(results, corpus.questions)])
        iters = np.mean([r.n_iterations for r in results])
        print(f"  {name:<30}{em:>15.0%}{iters:>14.1f}")

    print("\n  Neither half is sufficient on its own. One diverse retrieval cannot reach")
    print("  hop 3 no matter how diverse it is, and iterating over a redundant set just")
    print("  re-reads the same thicket. Together they answer most of the chains.")
    print("\n  These are synthetic numbers illustrating the mechanism, not benchmark")
    print("  results. For real-data numbers see the paper, or research/ to rerun them.")

    # ── 5. one trajectory in detail ─────────────────────────────────────────
    banner("5. One 4-hop question, iteration by iteration")
    hard = corpus.questions[2]
    rag = VendiRAG.offline(retriever, k_docs=K, k_candidates=POOL,
                           initial_s=0.8, dynamic_s=False)
    result = rag.answer(hard.question)
    print(f"Q: {hard.question}")
    print(f"   gold answer: {hard.answer}\n")
    for it in result.iterations:
        print(f"  iteration {it.index}  s={it.s:.2f}  VS={it.vendi_score:.1f}/{K}  "
              f"Q={it.quality:.2f}")
        print(f"    retrieving for : {it.query[:70]}")
        print(f"    answer         : {it.answer[:70]}")
    print(f"\n  final: {result.answer!r}  "
          f"({'correct' if result.answer == hard.answer else 'wrong'})")

    if args.gif:
        from vendirag.viz import make_selection_gif
        path = make_selection_gif(
            retriever, question, path=args.gif, questions=corpus.questions,
            k=K, candidate_pool=POOL,
        )
        print(f"\nAnimation written to {path}")


if __name__ == "__main__":
    main()
