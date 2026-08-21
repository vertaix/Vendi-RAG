"""``vendirag`` command line entry point.

    vendirag demo                     # run the synthetic benchmark, print results
    vendirag demo --gif assets/x.gif  # also render the animation
    vendirag ask "..." --corpus f.txt # retrieve against a file, one line per doc
"""

from __future__ import annotations

import argparse
import sys
from typing import List

import numpy as np


def _build_toy(args):
    from .embeddings import HashingEmbedder
    from .retriever import VendiRetriever
    from .toy import make_corpus

    corpus = make_corpus(n_chains=args.chains, seed=args.seed,
                         n_distractors=args.distractors)
    # Pinned to the hashing embedder so the demo downloads nothing and prints
    # the same numbers everywhere.
    retriever = VendiRetriever(
        embedder=HashingEmbedder(), k=args.k, candidate_pool=args.pool
    ).index(corpus.documents)
    return corpus, retriever


def cmd_demo(args) -> int:
    from .pipeline import VendiRAG
    from .retriever import mmr_select

    corpus, retriever = _build_toy(args)
    print(f"Corpus: {corpus.describe()}\n")

    def score(fn) -> tuple:
        hit, vs = [], []
        for q in corpus.questions:
            docs = fn(q.question)
            hit.append(any(d.id in set(q.gold_ids) for d in docs))
            vs.append(retriever.set_diversity([d.text for d in docs]))
        return float(np.mean(hit)), float(np.mean(vs))

    def mmr(question: str, lam: float) -> List:
        qe = retriever.embed_query(question)
        sims = retriever.embeddings @ qe
        pool = np.argsort(-sims)[:args.pool]
        idx = mmr_select(retriever.embeddings[pool], qe, k=args.k, lambda_mult=lam)
        return [retriever.documents[pool[i]] for i in idx]

    print(f"RETRIEVAL  (k={args.k}, |C|={args.pool}, {len(corpus.questions)} questions)")
    print(f"  {'method':<28} {'evidence found':>15} {'unique docs':>13}")
    rows = [("top-k similarity", lambda q: retriever.similarity_search(q, k=args.k))]
    rows += [(f"MMR (lambda={lam})", lambda q, l=lam: mmr(q, l)) for lam in (0.5, 0.7)]
    rows += [(f"Vendi retrieval (s={s})",
              lambda q, s=s: retriever.retrieve(q, k=args.k, s=s, candidate_pool=args.pool))
             for s in (0.2, 0.4, 0.6, 0.8, 1.0)]
    for name, fn in rows:
        rec, vs = score(fn)
        print(f"  {name:<28} {rec:>14.0%} {vs:>9.1f}/{args.k}")

    print(f"\nEND TO END  (offline reader, no API key)")
    print(f"  {'configuration':<28} {'exact match':>15} {'iterations':>13}")
    configs = [
        ("single-shot, top-k", dict(initial_s=0.0), True),
        ("single-shot, Vendi s=0.8", dict(initial_s=0.8), True),
        ("iterative, top-k", dict(initial_s=0.0, dynamic_s=False), False),
        ("Vendi-RAG (fixed s=0.8)", dict(initial_s=0.8, dynamic_s=False), False),
        ("Vendi-RAG (dynamic s)", dict(initial_s=0.8, dynamic_s=True), False),
    ]
    for name, kwargs, single in configs:
        rag = VendiRAG.offline(retriever, k_docs=args.k, k_candidates=args.pool, **kwargs)
        results = [
            rag.answer_single_shot(q.question) if single else rag.answer(q.question)
            for q in corpus.questions
        ]
        em = np.mean([
            r.answer.strip().lower() == q.answer.strip().lower()
            for r, q in zip(results, corpus.questions)
        ])
        iters = np.mean([r.n_iterations for r in results])
        print(f"  {name:<28} {em:>14.0%} {iters:>13.1f}")

    if args.gif:
        from .viz import make_selection_gif
        path = make_selection_gif(
            retriever, corpus.questions[0], path=args.gif,
            questions=corpus.questions, k=args.k, candidate_pool=args.pool,
        )
        print(f"\nAnimation written to {path}")
    return 0


def cmd_ask(args) -> int:
    from .retriever import VendiRetriever

    with open(args.corpus, encoding="utf-8") as handle:
        texts = [line.strip() for line in handle if line.strip()]
    retriever = VendiRetriever.from_texts(texts, k=args.k, candidate_pool=args.pool)
    result = retriever.retrieve_details(args.question, s=args.s)
    print(f"s={args.s}  Vendi Score={result.vendi_score:.2f}  "
          f"mean similarity={result.mean_similarity:.3f}\n")
    for i, doc in enumerate(result.documents, start=1):
        print(f"{i}. {doc.text}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="vendirag", description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)

    demo = subs.add_parser("demo", help="run the synthetic multi-hop benchmark")
    demo.add_argument("--chains", type=int, default=20, help="chains in the toy world")
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--distractors", type=int, default=12,
                      help="depth of the redundancy thicket around each chain")
    demo.add_argument("-k", type=int, default=8, help="documents retrieved per query")
    demo.add_argument("--pool", type=int, default=50, help="candidate pool size |C|")
    demo.add_argument("--gif", default=None, help="also write the animation here")
    demo.set_defaults(func=cmd_demo)

    ask = subs.add_parser("ask", help="Vendi retrieval over a newline-delimited file")
    ask.add_argument("question")
    ask.add_argument("--corpus", required=True, help="one document per line")
    ask.add_argument("-k", type=int, default=5)
    ask.add_argument("-s", type=float, default=0.8)
    ask.add_argument("--pool", type=int, default=50)
    ask.set_defaults(func=cmd_ask)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
