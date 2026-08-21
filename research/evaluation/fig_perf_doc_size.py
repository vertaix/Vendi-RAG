import pandas as pd
import matplotlib.pyplot as plt

# Original data
data = {
    "Document Size": [2, 4, 6, 8, 10],

    # Best model (s₁=0.8)
    "ACC_Vendi-RAG, s₁=0.8":     [50.8, 54.0, 59.71, 61.7, 62.8],
    "F1_Vendi-RAG, s₁=0.8":      [51.3, 55.8, 61.8, 63.7, 64.39],
    "EM_Vendi-RAG, s₁=0.8":      [37.6, 41.2, 47.09, 48.3, 49.0],
    "Vendi_Vendi-RAG, s₁=0.8":   [1.81, 3.30, 4.423, 5.586, 6.8187],

    # Adaptive-RAG (baseline)
    "ACC_Adaptive-RAG":  [51.2, 53.0, 54.0, 54.3, 55.0],
    "F1_Adaptive-RAG":   [52.0, 55.0, 58.0, 59.0, 60.1],
    "EM_Adaptive-RAG":   [38.3, 40.1, 43.0, 46.1, 47.3],
    "Vendi_Adaptive-RAG":[1.74, 2.99, 3.981, 4.457, 5.205],

    # Vendi-RAG, s₁=0.3
    "ACC_Vendi-RAG, s₁=0.3": [51.5, 54.1, 56.5, 57.6, 58.3],
    "F1_Vendi-RAG, s₁=0.3":  [52.1, 56.0, 59.4, 60.5, 61.3],
    "EM_Vendi-RAG, s₁=0.3":  [38.5, 41.0, 45.2, 46.7, 47.9],
    "Vendi_Vendi-RAG, s₁=0.3":[1.76, 3.10, 4.10, 5.00, 6.0],

    # Vendi-RAG, s₁=1.0
    "ACC_Vendi-RAG, s₁=1.0": [51.9, 55.0, 58.1, 59.8, 60.7],
    "F1_Vendi-RAG, s₁=1.0":  [52.8, 56.9, 60.6, 62.0, 62.7],
    "EM_Vendi-RAG, s₁=1.0":  [38.9, 41.6, 46.3, 47.5, 48.2],
    "Vendi_Vendi-RAG, s₁=1.0":[1.78, 3.25, 4.30, 5.35, 6.5],
}

df = pd.DataFrame(data)

# Define metrics and their key prefixes
metrics = ["Exact Match", "F1-score", "Accuracy", "Vendi Score"]
metric_keys = ["EM", "F1", "ACC", "Vendi"]

# Define model variants to plot
model_variants = [
    ("Vendi-RAG, s₁=0.8", "#D35400", "-.", "v"),
    ("Vendi-RAG, s₁=1.0", "#9B59B6", "--", "d"),
    ("Vendi-RAG, s₁=0.3", "#3498DB", ":", "^"),
    ("Adaptive-RAG", "black", "--", "s"),
]

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(17, 13), sharex=True)
plt.rcParams.update({'font.size': 24})

for i, (metric, key) in enumerate(zip(metrics, metric_keys)):
    ax = axes[i//2, i%2]
    for name, color, linestyle, marker in model_variants:
        col_name = f"{key}_{name}"
        if col_name in df.columns:
            ax.plot(df["Document Size"], df[col_name], label=name, color=color,
                    linestyle=linestyle, marker=marker, linewidth=2.5)
    ax.set_title(metric, fontsize=27)
    ax.set_xlabel("Document Size", fontsize=27)
    if i<=2:
        ax.set_ylabel(f"{metric} (%)", fontsize=27)
    else:
        ax.set_ylabel(metric, fontsize=27)

    ax.grid(True, linestyle="--", linewidth=1)
    if i == 0:
        ax.legend(fontsize=24)

plt.tight_layout()
plt.savefig("per_doc_named_models.pdf", format="pdf", bbox_inches="tight")
plt.show()
