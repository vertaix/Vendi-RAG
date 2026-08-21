# Reproducing the paper

The `vendirag` package at the repository root is the implementation; everything
here is the harness that runs it over the paper's datasets and scores the
output. Nothing in this directory is needed to *use* the library.

## 1. Install the research extras

```bash
pip install -e ".[research]"
cp .env.example .env      # then set OPENAI_API_KEY
```

## 2. Get the data

```bash
bash research/download/processed_data.sh   # preprocessed test splits, ~150 MB
bash research/download/raw_data.sh         # raw datasets, only if re-preprocessing
```

`research/processing_scripts/` turns raw HotpotQA / MuSiQue / 2WikiMultiHopQA
dumps into the `processed_data/<dataset>/test_subsampled.jsonl` format the rest
of the harness expects.

## 3. Build the index (once per dataset)

```bash
python research/ingestion.py --dataset hotpotqa --build
```

Embeds every context paragraph with `all-mpnet-base-v2` into a persistent Chroma
store under `vectorDB/<dataset>/`. Resumable — rerun after an interruption and it
picks up where it stopped.

## 4. Run

```bash
# Vendi-RAG, dynamic s (Algorithm 1)
python research/run_benchmark.py --dataset hotpotqa --model gpt-4o-mini

# Vendi-RAG, fixed s = 0.8 — the configuration the paper recommends as default
python research/run_benchmark.py --dataset hotpotqa --fixed-s

# Single-step baselines: top-k, MMR, and fixed-s Vendi retrieval
python research/baselines.py --dataset hotpotqa --method similarity
python research/baselines.py --dataset hotpotqa --method mmr --lambda 0.7
python research/baselines.py --dataset hotpotqa --method vendi -s 0.8
```

Every runner checkpoints to its output CSV after each question, so an
interrupted run resumes rather than restarting.

## 5. Score

```bash
python research/evaluation/cal_f1_em.py --predictions results/<file>.csv
```

## Experiments

| Script | What it isolates |
|---|---|
| `experiments/exp1_factorial_ablation.py` | 2x2 factorial: s-adaptation vs. judge-driven early stopping (Appendix C.3) |
| `experiments/exp4_reranker_baseline.py` | Cross-encoder reranking over the same top-50 pool, to separate diversity-awareness from pool size (Appendix C.2) |

Both predate the library refactor and still carry their own copies of the
retrieval and prompting code.
