"""
EM / F1 / accuracy scoring for a predictions CSV.

The CSV needs a ``ground_truth`` and a ``generated_answer`` column, which is
what every runner in ``research/`` writes.

    python research/evaluation/cal_f1_em.py --predictions results/<file>.csv
"""

import argparse
import collections
import os
import re
import string
import sys
import time

import numpy as np
import pandas as pd
def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))
class Evaluation:
    def __init__(self, predictions, out_dir="."):
        """
        Parameters
        ----------
        predictions : str
            Path to a CSV with ``ground_truth`` and ``generated_answer`` columns.
        out_dir : str
            Where the per-row metrics and summary files are written.
        """
        self.predictions = predictions
        self.out_dir = out_dir
        self.name = os.path.splitext(os.path.basename(predictions))[0]
        self.df = pd.read_csv(predictions)
        missing = {"ground_truth", "generated_answer"} - set(self.df.columns)
        if missing:
            raise ValueError(f"{predictions} is missing column(s): {sorted(missing)}")
        os.makedirs(out_dir, exist_ok=True)
        self.one_step_time = None  # Placeholder for baseline one-step time, to be calculated dynamically

    def _path(self, prefix, ext):
        return os.path.join(self.out_dir, f"{prefix}_{self.name}.{ext}")

    def calculate_metrics(self):
        """
        Method to calculate F1, EM, and Accuracy (Acc).
        F1 measures overlapping words between predicted and ground truth.
        EM checks if the predicted answer is identical to the ground truth.
        Acc checks if the predicted answer contains the ground-truth answer.
        """
        self.df["f1_score"] = 0.0
        self.df["exact_match"] = 0
        self.df["accuracy"] = 0

        for index, row in self.df.iterrows():
            ground_truth = normalize_answer(row["ground_truth"]).strip().lower() if pd.notnull(row["ground_truth"]) else ""
            generated_answer = normalize_answer(row["generated_answer"]).strip().lower() if pd.notnull(row["generated_answer"]) else ""

            if not ground_truth or not generated_answer:
                continue  # Skip if either field is empty

            # F1 Score
            gt_tokens = set(ground_truth.split())
            pred_tokens = set(generated_answer.split())
            common_tokens = collections.Counter(gt_tokens) & collections.Counter(pred_tokens)
            precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
            recall = len(common_tokens) / len(gt_tokens) if gt_tokens else 0
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
            self.df.at[index, "f1_score"] = f1

            # Exact Match (EM)
            self.df.at[index, "exact_match"] = 1 if ground_truth == generated_answer else 0

            # Accuracy (Acc)
            self.df.at[index, "accuracy"] = 1 if ground_truth in generated_answer else 0
        self.df= self.df.dropna()

    def calculate_efficiency(self):
        """
        Method to calculate the number of retrieval-and-generate steps
        and the average time per query relative to the one-step approach.
        """
        self.df["retrieval_generate_steps"] = self.df["steps"].astype(int)  # Assuming "steps" column exists
        self.df["query_time"] = self.df["query_time"].astype(float)  # Assuming "query_time" column exists

        # Calculate one-step baseline time
        if self.one_step_time is None:
            self.one_step_time = self.df["query_time"].mean()  # Baseline: average of all times

        # Calculate time relative to the one-step approach
        self.df["relative_time"] = self.df["query_time"] / self.one_step_time

    def aggregate_results(self):
        """
        Method to calculate mean and std for all metrics.
        """
        metrics = ["f1_score", "exact_match", "accuracy"]
        aggregate_results = {
            "metric": [],
            "mean": [],
            "std": []
        }

        for metric in metrics:
            if metric in self.df.columns:
                aggregate_results["metric"].append(metric)
                aggregate_results["mean"].append(self.df[metric].mean())
                aggregate_results["std"].append(self.df[metric].std())

        # Save aggregate results
        aggregate_df = pd.DataFrame(aggregate_results)
        aggregate_df.to_csv(self._path("Aggregate_Performance", "csv"), index=False)

    def calculate_overall_performance(self):
        """
        Method to calculate the overall performance by combining F1, EM, and accuracy.
        Uses a weighted average of the metrics for retrieval and generation.
        """
        retrieval_weight = 0.5  # Adjust weights as needed
        generation_weight = 0.5

        # Ensure required columns exist
        if "f1_score" not in self.df.columns or "exact_match" not in self.df.columns:
            raise ValueError("F1 and Exact Match metrics are required for overall performance calculation.")

        # Compute the overall score
        self.df["overall_performance"] = (
            retrieval_weight * self.df["f1_score"] +
            generation_weight * self.df["exact_match"]
        )

        # Aggregate overall performance
        overall_mean = self.df["overall_performance"].mean()
        overall_std = self.df["overall_performance"].std()

        # Print and save results
        print(f"Overall Performance: Mean={overall_mean:.4f}, Std={overall_std:.4f}")
        with open(self._path("Overall_Performance", "txt"), "w") as file:
            file.write(f"Overall Performance: Mean={overall_mean:.4f}, Std={overall_std:.4f}\n")

    def evaluate(self):
        """
        Run all evaluations, save metrics, and aggregate results.
        """
        self.calculate_metrics()
        # self.calculate_efficiency()
        self.aggregate_results()
        self.calculate_overall_performance()

        # Save detailed results
        self.df.to_csv(self._path("Metrics", "csv"), index=False)
        print("Evaluation completed and results saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True,
                        help="CSV with ground_truth and generated_answer columns")
    parser.add_argument("--out-dir", default=".",
                        help="where to write the per-row metrics and summary")
    args = parser.parse_args()

    Evaluation(predictions=args.predictions, out_dir=args.out_dir).evaluate()
