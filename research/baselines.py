"""
Single-step retrieval baselines on the paper's datasets.

Every configuration reported alongside Vendi-RAG in Section 3 that does *not*
use the iterative loop lives here, so they share one retrieval path, one prompt,
and one output format and differ only in how the k documents are chosen:

* ``similarity``  plain top-k dense retrieval
* ``mmr``         Maximal Marginal Relevance over the same top-|C| pool
* ``vendi``       Vendi retrieval at a fixed s, single step (no loop)

    python research/baselines.py --dataset hotpotqa --method mmr --lambda 0.7

Score the output with ``research/evaluation/cal_f1_em.py``.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from vendirag import OpenAILLM, mmr_select, vrs_select
from vendirag.llm import extract_json

from ingestion import DATASETS, for_dataset

ANSWER_PROMPT = """Based on the provided context, answer the question concisely \
with the shortest span that fully answers it.

Context:
{context}

Question: {question}

Respond with JSON only, in exactly this form:
{{"answer": "<your concise answer>"}}"""


def select(method: str, docs, doc_embs, query_emb, k: int, s: float, lambda_mult: float):
    if method == "similarity":
        return list(range(min(k, len(docs))))       # the pool is already ranked
    if method == "mmr":
        return mmr_select(doc_embs, query_emb, k=k, lambda_mult=lambda_mult)
    if method == "vendi":
        return vrs_select(doc_embs, query_emb, k=k, s=s)
    raise ValueError(f"unknown method {method!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default="hotpotqa", choices=DATASETS)
    parser.add_argument("--method", default="vendi",
                        choices=("similarity", "mmr", "vendi"))
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("-k", "--k-docs", type=int, default=5)
    parser.add_argument("--k-candidates", type=int, default=50)
    parser.add_argument("-s", "--diversity", type=float, default=0.8,
                        help="s for --method vendi")
    parser.add_argument("--lambda", dest="lambda_mult", type=float, default=0.7,
                        help="lambda for --method mmr")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    corpus = for_dataset(args.dataset)
    questions, answers = corpus.load_questions()
    if args.limit:
        questions, answers = questions[:args.limit], answers[:args.limit]
    llm = OpenAILLM(args.model)

    tag = {"similarity": "topk", "mmr": f"mmr{args.lambda_mult}",
           "vendi": f"vendi{args.diversity}"}[args.method]
    out = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results",
        f"baseline_{args.dataset}_{args.model}_{tag}_k{args.k_docs}.csv",
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)

    rows = pd.read_csv(out).to_dict("records") if os.path.exists(out) else []
    if rows:
        print(f"resuming after {len(rows)} completed questions")

    for i in range(len(rows), len(questions)):
        docs, doc_embs, query_emb = corpus.candidates(questions[i], args.k_candidates)
        doc_embs = np.asarray(doc_embs)
        idx = select(args.method, docs, doc_embs, query_emb,
                     args.k_docs, args.diversity, args.lambda_mult)
        context = "\n\n".join(docs[j].text for j in idx)
        reply = llm.complete(ANSWER_PROMPT.format(context=context, question=questions[i]))
        rows.append({
            "question": questions[i],
            "ground_truth": answers[i],
            "generated_answer": extract_json(reply).get("answer", ""),
        })
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"[{i + 1}/{len(questions)}] {rows[-1]['generated_answer'][:60]}")

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
