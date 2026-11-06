"""
plot_spe_by_cluster.py —— 各驾驶状态（GMM 聚类）训练集 SPE 柱状图（英文、简化；样式参考论文 Fig.3b）。在 runs/<run> 下运行：

    python ../../tools/plot_spe_by_cluster.py [OUT=fig_spe_by_cluster.png]

- 数据：train_frame_spe.csv（New.stage_detect 写出：训练帧 SPE = 覆盖窗口最大值，cluster = 覆盖窗口众数）。
- 柱 = 该类训练帧 SPE 均值；误差线 = +1 SD（只画上侧，SPE 非负）；黑色短虚线 = 该类最终 UCL（方案 C 后，读 log_detect*.txt）。
- 类名与颜色同 plot_latent_clusters.py（cluster_names.json）。
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#3b6fb6", "#2a9d8f", "#8ab17d", "#e9a23b", "#c8553d", "#9b72aa", "#6c757d"]


def final_ucl():
    logs = sorted(glob.glob("log_detect*.txt"), key=os.path.getmtime)
    for f in reversed(logs):
        txt = open(f, errors="ignore").read()
        if "Cluster-specific frame UCL (final):" in txt:
            part = txt.split("Cluster-specific frame UCL (final):")[-1]
            return {int(k): float(v) for k, v in re.findall(r"Cluster (\d+): UCL=([0-9.]+)", part)}
    return {}


def main(out="fig_spe_by_cluster.png"):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 11})
    t = pd.read_csv("train_frame_spe.csv")
    names = json.load(open("cluster_names.json"))
    ucl = final_ucl()
    ks = sorted(int(k) for k in t["cluster"].dropna().unique())
    g = t.groupby("cluster")["spe"]
    mean, sd, share = g.mean(), g.std(), g.size() / len(t)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    x = np.arange(len(ks))
    for i, k in enumerate(ks):
        ax.bar(i, mean[k], width=0.55, color=COLORS[k % len(COLORS)], alpha=0.85, edgecolor="none")
        ax.errorbar(i, mean[k], yerr=[[0], [sd[k]]], color="0.3", capsize=5, lw=1)
        if os.environ.get("SHOW_UCL") == "1" and k in ucl:   # 默认不画（与论文参考图一致）
            ax.hlines(ucl[k], i - 0.32, i + 0.32, colors="k", linestyles="--", lw=1.2,
                      label="UCL (99%, floored at global)" if i == 0 else None)
    labels = [names.get(str(k), f"C{k}").replace(" ", "\n") for k in ks]
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("SPE"); ax.grid(True, alpha=0.25); ax.set_ylim(bottom=0)
    if os.environ.get("SHOW_UCL") == "1":
        ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout(); plt.savefig(out, dpi=300); plt.close()
    print(pd.DataFrame({"name": [names.get(str(k)) for k in ks], "share": share[ks].round(3).values,
                        "mean": mean[ks].round(3).values, "sd": sd[ks].round(3).values,
                        "ucl": [ucl.get(k) for k in ks]}).to_string(index=False))
    print("saved", out)


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []))
