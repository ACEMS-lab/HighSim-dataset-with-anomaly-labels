"""
plot_train_curves.py —— 不同潜变量维度 D 的训练曲线（训练 MSE、验证 MSE、潜变量总方差）。

用法：
    python tools/plot_train_curves.py runs/latentD OUT_PREFIX
读 runs/latentD/D*/train_history.csv（MSE 不含 L2 正则项），
输出 OUT_PREFIX_{train_mse,val_mse,latent_total_var}.png 三张单图及 OUT_PREFIX_all.png 三联图。
"""
import glob
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {8: ("#f4a582", 1.3), 16: ("#2c5fb3", 1.8), 32: ("#5ab045", 1.6)}
PANELS = [("train_mse", "Training MSE", "MSE"), ("val_mse", "Validation MSE", "MSE"),
          ("latent_total_var", "Total Variance of Latent Variables During Training", "Latent Variance")]


def main(root="runs/latentD", prefix="fig_latent_dim"):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 12,
                         "mathtext.fontset": "dejavuserif"})
    H = {}
    for f in sorted(glob.glob(os.path.join(root, "D*", "train_history.csv"))):
        d = pd.read_csv(f); H[int(d.latent_dim.iloc[0])] = d
    if not H:
        sys.exit("no train_history.csv found")

    def draw(ax, col, title, ylab):
        for D in sorted(H):
            c, lw = STYLE.get(D, ("0.4", 1.4))
            ax.plot(H[D].epoch, H[D][col], color=c, lw=lw, label=f"$D = {D}$")
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.25); ax.legend(fontsize=12)

    for col, title, ylab in PANELS:
        fig, ax = plt.subplots(figsize=(5.2, 4))
        draw(ax, col, title, ylab)
        plt.tight_layout(); plt.savefig(f"{prefix}_{col}.png", dpi=300); plt.close()
    fig, axs = plt.subplots(3, 1, figsize=(5.4, 11.5))
    for ax, (col, title, ylab), tag in zip(axs, PANELS, "abc"):
        draw(ax, col, title, ylab)
        ax.text(0.5, -0.3, f"({tag})", transform=ax.transAxes, ha="center", fontsize=13)
    plt.tight_layout(h_pad=2.5); plt.savefig(f"{prefix}_all.png", dpi=300); plt.close()
    print(pd.DataFrame({D: {"epochs": len(d), "best_val_mse": round(float(d.val_mse.min()), 4),
                            "final_train_mse": round(float(d.train_mse.iloc[-1]), 4), "final_latent_var": round(float(d.latent_total_var.iloc[-1]), 1)}
                        for D, d in sorted(H.items())}).T.to_string())


if __name__ == "__main__":
    main(*sys.argv[1:3])
