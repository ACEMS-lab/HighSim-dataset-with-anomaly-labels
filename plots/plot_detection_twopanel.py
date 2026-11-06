import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"

plt.rcParams["font.size"] = 14

plt.rcParams["axes.labelsize"] = 14

plt.rcParams["xtick.labelsize"] = 14

plt.rcParams["ytick.labelsize"] = 14

plt.rcParams["legend.fontsize"] = 14

plt.rcParams["axes.titlesize"] = 14
data = {
    "Method": [
    "IForest",

    "PCA",

    "AE",

    "LSTM-\nAE",

    "Raw-GRU-\nAE",

    "Raw-GRU-AE-\nGMM",

    "Utility-GRU-\nAE",

    "Utility-LSTM-\nAE-GMM",

    "Utility-GRU-\nAE-GMM"
    ],
    "n_pred": [299, 443, 542, 540, 522, 171, 189, 105, 113],
    "event_recall": [0.4538, 0.5773, 0.6795, 0.7261, 0.7546, 0.7546, 0.9487, 0.9231, 0.9487],
}

df = pd.DataFrame(data)

x = np.arange(len(df))
width = 0.65

fig, ax1 = plt.subplots(figsize=(12, 5.5))

bars = ax1.bar(
    x,
    df["n_pred"],
    width=width,
    label="Number of Detected Events",
    color="lightgray",
    edgecolor="black"
)

ax1.set_ylabel("Number of Detected Events")
ax1.set_xticks(x)
ax1.set_xticklabels(df["Method"], rotation=0, ha="center")
ax1.set_ylim(0, max(df["n_pred"]) * 1.15)

for bar in bars:
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        height + 8,
        f"{int(height)}",
        ha="center",
        va="bottom",
        fontsize=14
    )

ax2 = ax1.twinx()
ax2.plot(
    x,
    df["event_recall"],
    marker="o",
    linewidth=2.5,
    markersize=7,
    color="darkred",
    label="Proxy Reference Event Recall"
)

ax2.set_ylabel("Proxy Reference Event Recall")
ax2.set_ylim(0, 1.05)

for i, recall in enumerate(df["event_recall"]):
    ax2.text(
        i,
        recall - 0.04,
        f"{recall:.4f}",
        ha="center",
        va="top",
        fontsize=11,
        color="darkred"
    )

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

plt.tight_layout()
plt.savefig("baseline_event_comparison_no_rule.png", dpi=300, bbox_inches="tight")
plt.show()