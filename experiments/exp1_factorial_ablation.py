"""
Experiment 1 — 2×2 Factorial Ablation (Rebuttal, Highest Priority)
===================================================================
Isolates the contribution of (a) s-adaptation and (b) judge early-stopping.

Four conditions:
  A  fixed-s=0.8,  no early stopping,  3 iterations  (baseline)
  B  fixed-s=0.8,  LLM judge stops,    ≤5 iterations
  C  adaptive-s,   no early stopping,  3 iterations
  D  adaptive-s,   LLM judge stops,    ≤5 iterations  (full system)

Three datasets: hotpotqa, musique, 2wikimultihopqa  (500 each)

Model: gpt-4o-mini  (all four conditions — ablation, not backbone test)

Usage
-----
  # run all 12 condition×dataset pairs:
  python experiments/exp1_factorial_ablation.py

  # single condition on one dataset (for testing):
  python experiments/exp1_factorial_ablation.py --condition A --dataset hotpotqa

  # dry run on 5 questions per dataset:
  python experiments/exp1_factorial_ablation.py --max-questions 5
"""

from __future__ import annotations

import argparse
import collections
import os
import re
import string
import sys
from typing import Dict, List, Optional

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from vectorDB.dataset_ingestion import Ingestor
from models.vendi_rag import VendiRAG


# ════════════════════════════════════════════════════════════════════════════
#  Experiment configuration
# ════════════════════════════════════════════════════════════════════════════

DATASETS = ["hotpotqa", "musique", "2wikimultihopqa"]

#  Each entry: (dynamic_s, use_early_stopping, max_iterations)
CONDITIONS: Dict[str, dict] = {
    "A": dict(dynamic_s=False, use_early_stopping=False, max_iterations=3,
              label="Fixed-s, fixed-iter"),
    "B": dict(dynamic_s=False, use_early_stopping=True,  max_iterations=5,
              label="Fixed-s, judge-stop"),
    "C": dict(dynamic_s=True,  use_early_stopping=False, max_iterations=3,
              label="Adaptive-s, fixed-iter"),
    "D": dict(dynamic_s=True,  use_early_stopping=True,  max_iterations=5,
              label="Adaptive-s, judge-stop  [full system]"),
}

MODEL = "gpt-4o-mini"
K_DOCS = 5
K_CANDIDATES = 50
INITIAL_S = 0.8

OUT_DIR = os.path.join(_ROOT, "results", "exp1")


# ════════════════════════════════════════════════════════════════════════════
#  Evaluation helpers  (mirrors evaluation/cal_f1_em.py)
# ════════════════════════════════════════════════════════════════════════════

def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def _f1(pred: str, gold: str) -> float:
    p_toks = collections.Counter(_normalize(pred).split())
    g_toks = collections.Counter(_normalize(gold).split())
    common = p_toks & g_toks
    if not common:
        return 0.0
    prec = sum(common.values()) / sum(p_toks.values())
    rec  = sum(common.values()) / sum(g_toks.values())
    return 2 * prec * rec / (prec + rec)


def score_df(df: pd.DataFrame) -> dict:
    """Compute EM, F1, Accuracy, avg_iterations from a results DataFrame."""
    em, f1, acc = [], [], []
    for _, row in df.iterrows():
        pred = _normalize(str(row.get("generated_answer", "") or ""))
        gold = _normalize(str(row.get("ground_truth", "") or ""))
        if not pred or not gold:
            continue
        em.append(int(pred == gold))
        f1.append(_f1(pred, gold))
        acc.append(int(gold in pred))

    avg_iter = df["iterations"].mean() if "iterations" in df.columns else float("nan")
    n = max(len(em), 1)
    return {
        "n": n,
        "EM":   round(100 * sum(em) / n, 2),
        "F1":   round(100 * sum(f1) / n, 2),
        "Acc":  round(100 * sum(acc) / n, 2),
        "avg_iter": round(avg_iter, 2),
    }


# ════════════════════════════════════════════════════════════════════════════
#  Single condition × dataset run
# ════════════════════════════════════════════════════════════════════════════

def run_one(
    condition_id: str,
    dataset: str,
    max_questions: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run one (condition, dataset) pair.  Returns a scored DataFrame.
    Resumes from checkpoint if the CSV already exists.
    """
    cfg = CONDITIONS[condition_id]
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path  = os.path.join(OUT_DIR, f"cond{condition_id}_{dataset}.csv")
    ckpt_path = os.path.join(OUT_DIR, f"cond{condition_id}_{dataset}_checkpoint.csv")

    # ── already fully done? ──────────────────────────────────────────────────
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if "EM" in df.columns:           # scored already
            print(f"  [skip] {condition_id}/{dataset} — results exist")
            return df
        # file exists but not scored yet → fall through to scoring

    ingestor = Ingestor(
        dataset_path=os.path.join(_ROOT, f"processed_data/{dataset}/"),
        persist_directory=os.path.join(_ROOT, f"vectorDB/{dataset}"),
    )
    data = ingestor.load_evaluation_data()
    questions    = data["question_text"]
    ground_truths = data["ground_truth"]

    if max_questions:
        questions     = questions[:max_questions]
        ground_truths = ground_truths[:max_questions]

    rag = VendiRAG(
        ingestor=ingestor,
        model=MODEL,
        max_iterations=cfg["max_iterations"],
        dynamic_s=cfg["dynamic_s"],
        use_early_stopping=cfg["use_early_stopping"],
        initial_s=INITIAL_S,
        k_docs=K_DOCS,
        k_candidates=K_CANDIDATES,
        verbose=False,          # suppress per-step output during benchmarks
    )

    df = rag.run_benchmark(
        questions=questions,
        ground_truths=ground_truths,
        output_path=csv_path,
        checkpoint_path=ckpt_path,
    )

    # ── clean up checkpoint once complete ────────────────────────────────────
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    return df


# ════════════════════════════════════════════════════════════════════════════
#  Summary table
# ════════════════════════════════════════════════════════════════════════════

def print_table(results: dict) -> None:
    """
    Print the paper-style table:

    Condition              | HotpotQA Acc | ΔA  | MuSiQue Acc | ΔA  | 2Wiki Acc | ΔA  | avg_iter
    """
    header = f"{'Condition':<38} | {'HotpotQA':>9} | {'Δ':>6} | {'MuSiQue':>9} | {'Δ':>6} | {'2Wiki':>9} | {'Δ':>6} | avg_iter"
    print("\n" + "=" * len(header))
    print("Experiment 1 — 2×2 Factorial Ablation  (Accuracy %)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    baseline_acc = {}
    for cid in ["A", "B", "C", "D"]:
        row_parts = [f"{cid}: {CONDITIONS[cid]['label']:<34}"]
        iters_all = []
        for ds in DATASETS:
            key = (cid, ds)
            if key not in results:
                row_parts.append(f"{'N/A':>9} | {'':>6}")
                continue
            sc = results[key]
            acc = sc["Acc"]
            iters_all.append(sc["avg_iter"])
            if cid == "A":
                baseline_acc[ds] = acc
                delta_str = "   —  "
            else:
                delta = acc - baseline_acc.get(ds, acc)
                delta_str = f"{delta:+.2f}"
            row_parts.append(f"{acc:>9.2f} | {delta_str:>6}")
        avg_iter = sum(iters_all) / len(iters_all) if iters_all else float("nan")
        print(" | ".join(row_parts) + f" | {avg_iter:>6.2f}")

    print("=" * len(header))
    print()

    # also print EM and F1
    for metric in ["EM", "F1"]:
        print(f"\n--- {metric} ---")
        for cid in ["A", "B", "C", "D"]:
            vals = [f"{results.get((cid,ds), {}).get(metric,'N/A'):>7}" for ds in DATASETS]
            print(f"  {cid}: {' | '.join(vals)}")


def save_summary_csv(results: dict) -> None:
    rows = []
    for (cid, ds), sc in results.items():
        rows.append({"condition": cid, "label": CONDITIONS[cid]["label"],
                     "dataset": ds, **sc})
    df = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, "exp1_summary.csv")
    df.to_csv(path, index=False)
    print(f"Summary saved → {path}")


# ════════════════════════════════════════════════════════════════════════════
#  Entry point
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Exp 1 — 2×2 factorial ablation")
    ap.add_argument("--condition", choices=["A", "B", "C", "D"], default=None,
                    help="Run only this condition (default: all four)")
    ap.add_argument("--dataset",   choices=DATASETS, default=None,
                    help="Run only this dataset (default: all three)")
    ap.add_argument("--max-questions", type=int, default=None,
                    help="Limit questions per dataset (for quick testing)")
    args = ap.parse_args()

    conds   = [args.condition] if args.condition else list(CONDITIONS.keys())
    datasets = [args.dataset]  if args.dataset   else DATASETS

    results: dict = {}

    # ── load any already-done results ────────────────────────────────────────
    for cid in ["A", "B", "C", "D"]:
        for ds in DATASETS:
            csv_path = os.path.join(OUT_DIR, f"cond{cid}_{ds}.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                results[(cid, ds)] = score_df(df)

    # ── run requested conditions × datasets ──────────────────────────────────
    for cid in conds:
        for ds in datasets:
            print(f"\n{'='*60}")
            print(f"Condition {cid} ({CONDITIONS[cid]['label']}) | {ds}")
            print(f"{'='*60}")
            df = run_one(cid, ds, max_questions=args.max_questions)
            results[(cid, ds)] = score_df(df)
            sc = results[(cid, ds)]
            print(f"  EM={sc['EM']:.2f}%  F1={sc['F1']:.2f}%  Acc={sc['Acc']:.2f}%  avg_iter={sc['avg_iter']:.2f}")

    # ── print summary table ───────────────────────────────────────────────────
    if results:
        print_table(results)
        save_summary_csv(results)
