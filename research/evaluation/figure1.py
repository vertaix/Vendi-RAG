import numpy as np
import matplotlib.pyplot as plt

# Data for the datasets
datasets = ["2WikiMultiHopQA", "HotpotQA", "MuSiQue"]
models = [
    ("Vendi-RAG-4o", "Adaptive-RAG-4o"),
    ("Vendi-RAG-4o-mini", "Adaptive-RAG-4o-mini"),
    ("Vendi-RAG-3.5", "Adaptive-RAG-3.5")
]
metrics = ["F1-score", "Exact Match", "Accuracy"]

# Performance values for each model
data = {
    "2WikiMultiHopQA": {
        "Vendi-RAG-4o": [0.581, 0.484, 0.634],
        "Adaptive-RAG-4o": [0.62, 0.471, 0.593],
        "Vendi-RAG-4o-mini": [0.570, 0.464, 0.632],
        "Adaptive-RAG-4o-mini": [0.576, 0.45, 0.583],
        "Vendi-RAG-3.5": [0.589, 0.472, 0.614],
        "Adaptive-RAG-3.5": [0.6009, .4660, .5680],
    },
    "HotpotQA": {
        "Vendi-RAG-4o": [0.699, 0.565, 0.647],
        "Adaptive-RAG-4o": [0.634, 0.521, 0.605],
        "Vendi-RAG-4o-mini": [0.644, 0.490, 0.628],
        "Adaptive-RAG-4o-mini": [0.601, 0.473, 0.55],
        "Vendi-RAG-3.5": [0.570, 0.422, 0.584],
        "Adaptive-RAG-3.5": [0.5256, 0.4040, 0.47],
    },
    "MuSiQue": {
        "Vendi-RAG-4o": [0.428, 0.300, 0.375],
        "Adaptive-RAG-4o": [0.424, 0.285, 0.362],
        "Vendi-RAG-4o-mini": [0.392, 0.266, 0.354],
        "Adaptive-RAG-4o-mini": [0.366, 0.261, 0.35],
        "Vendi-RAG-3.5": [0.325, 0.242, 0.304],
        "Adaptive-RAG-3.5": [0.3260, 0.2180, 0.2960],
    },
}

# Define bar width and positions
bar_width = 0.1
x = np.arange(len(datasets))
colors = ["#D35400", "#E67E22", "#F39C12", "#7F8C8D", "#BDC3C7", "#A6ACAF"]
hatches = ["//", "xx", "\\\\", "||", "--", ".."]

# Create figure with 3 subplots (one per metric)
fig, axes = plt.subplots(3, 1, figsize=(16, 16), sharex=True)
sca=[[30,74],[20,60],[25,70]]
# Store handles for legend
bars_list = []
labels_list = []

for i, metric in enumerate(metrics):
    ax = axes[i]

    for j, (vendi_model, adaptive_model) in enumerate(models):
        vendi_values = [data[dataset][vendi_model][i] * 100 for dataset in datasets]
        adaptive_values = [data[dataset][adaptive_model][i] * 100 for dataset in datasets]

        # Set positions with pairs grouped closely together
        positions = x + (j - 1) * (bar_width * 2.8)

        bars1 = ax.bar(positions - bar_width/2, vendi_values, width=bar_width, 
                       color=colors[j], edgecolor="black", hatch=hatches[j])
        bars2 = ax.bar(positions + bar_width/2, adaptive_values, width=bar_width, 
                       color=colors[j + 3], edgecolor="black", hatch=hatches[j+3])

        # Add bars to legend lists only once
        if i == 0:
            bars_list.append(bars1[0])
            labels_list.append(vendi_model)
            bars_list.append(bars2[0])
            labels_list.append(adaptive_model)

        # Add text labels on top of bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 1, f"{height:.1f}", 
                        ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_ylabel(f"{metric} (%)", fontsize=25)
    ax.set_ylim( sca[i])
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.tick_params(axis='y', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
plt.rcParams.update({'font.size': 28})
# Set common x-axis labels
axes[-1].set_xticks(x)
axes[-1].set_xticklabels(datasets, fontsize=25)

# Add legend outside the plot
fig.legend(bars_list, labels_list, fontsize=25, loc="upper center", ncol=3, bbox_to_anchor=(0.52, 1.03))

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit legend
plt.savefig("vendiragllms.pdf", format="pdf", bbox_inches="tight")
plt.show()
