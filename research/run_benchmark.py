"""
Reproduce the paper's benchmark numbers.

Runs the Vendi-RAG loop from the library over a real dataset's Chroma index and
writes per-question predictions, checkpointing after every question so a long
run survives an interrupted API session.

    python research/run_benchmark.py --dataset hotpotqa --model gpt-4o-mini
    python research/run_benchmark.py --dataset musique --fixed-s      # fixed-s variant
    python research/run_benchmark.py --dataset hotpotqa --limit 20    # smoke test

Score the output with ``research/evaluation/cal_f1_em.py``.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from vendirag import OpenAILLM, VendiRAG

from ingestion import DATASETS, for_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default="hotpotqa", choices=DATASETS)
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI chat model")
    parser.add_argument("--judge-model", default=None, help="separate judge model")
    parser.add_argument("-k", "--k-docs", type=int, default=5, help="documents per iteration")
    parser.add_argument("--k-candidates", type=int, default=50, help="candidate pool |C|")
    parser.add_argument("-s", "--initial-s", type=float, default=0.8)
    parser.add_argument("--max-iter", type=int, default=5)
    parser.add_argument("--fixed-s", action="store_true",
                        help="pin s (the variant the paper recommends as default)")
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="first N questions only")
    parser.add_argument("--out", default=None, help="output CSV path")
    args = parser.parse_args()

    corpus = for_dataset(args.dataset)
    questions, answers = corpus.load_questions()
    if args.limit:
        questions, answers = questions[:args.limit], answers[:args.limit]

    rag = VendiRAG(
        corpus.retriever(s=args.initial_s, k=args.k_docs, candidate_pool=args.k_candidates),
        llm=OpenAILLM(args.model),
        judge_llm=OpenAILLM(args.judge_model) if args.judge_model else None,
        initial_s=args.initial_s,
        k_docs=args.k_docs,
        k_candidates=args.k_candidates,
        max_iterations=args.max_iter,
        dynamic_s=not args.fixed_s,
        early_stopping=not args.no_early_stop,
    )

    variant = "fixed" if args.fixed_s else "dynamic"
    out = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results",
        f"vendirag_{args.dataset}_{args.model}_s{args.initial_s}_{variant}_k{args.k_docs}.csv",
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)

    rows = []
    if os.path.exists(out):
        rows = pd.read_csv(out).to_dict("records")
        print(f"resuming after {len(rows)} completed questions")

    for i in range(len(rows), len(questions)):
        result = rag.answer(questions[i])
        rows.append({
            "question": questions[i],
            "ground_truth": answers[i],
            "generated_answer": result.answer,
            "quality": result.quality,
            "iterations": result.n_iterations,
            "s_trajectory": ";".join(f"{s:.3f}" for s in result.s_trajectory),
        })
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"[{i + 1}/{len(questions)}] Q={result.quality:.2f} "
              f"iters={result.n_iterations}  {result.answer[:60]}")

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
