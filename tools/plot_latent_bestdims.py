"""
plot_latent_bestdims.py —— 在 GRU-AE 潜变量的原始维度中，挑选最能分开 GMM 驾驶状态的 2 维 / 3 维作图（坐标轴 = 真实潜变量 LVi）。
在 runs/<run> 目录下运行：

    python ../../tools/plot_latent_bestdims.py [N_PER_CLUSTER=400]

- 潜向量：验证集全部窗口（val_windows.npz 的窗口与 GMM 标签），按训练集同样方式标准化（各维减均值除标准差）。
- 选维：每类等量抽 N 个窗口；对所有 2 维组合（C(16,2)=120）与 3 维组合（C(16,3)=560）计算轮廓系数（silhouette），取最大者。
- 输出：fig_latent_best2d.png（99% 稳健椭圆，同 plot_latent_clusters.py）、fig_latent_best3d.png、latent_dim_ranking.csv。
- 平稳行驶（φ≡0）的窗口潜向量几乎相同，在原始维度中重合为一点（不加扰动，真实显示）。
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
from plot_latent_clusters import build_windows, ellipse, COLORS   # noqa: E402


def main(n_per=400):
    import tensorflow as tf
    from sklearn.metrics import silhouette_score
    from New import zero_recurrent_bias
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 11})
    w = np.load("val_windows.npz"); vid, start, lab = w["vid"], w["start"], w["lab"]
    norm = np.load("norm_params.npz")
    X = (build_windows(vid, start) - norm["mu"]) / norm["sd"]
    ae = zero_recurrent_bias(tf.keras.models.load_model("gruae_model.keras", compile=False))
    enc = tf.keras.Model(ae.input, ae.get_layer("latent_vector").output)
    Z = enc.predict(X, verbose=0, batch_size=2048)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)
    K = lab.max() + 1
    share = {k: float(np.mean(lab == k)) for k in range(K)}
    rng = np.random.default_rng(42)
    keep = np.concatenate([rng.choice(np.where(lab == k)[0], min(n_per, (lab == k).sum()), replace=False) for k in range(K)])
    Zs, ls = Z[keep], lab[keep]
    D = Z.shape[1]
    rows = []
    for dims in itertools.chain(itertools.combinations(range(D), 2), itertools.combinations(range(D), 3)):
        rows.append((len(dims), dims, silhouette_score(Zs[:, dims], ls)))
    r = pd.DataFrame(rows, columns=["n_dims", "dims", "silhouette"]).sort_values("silhouette", ascending=False)
    r["dims"] = r["dims"].map(lambda d: "+".join(f"LV{i + 1}" for i in d))
    r.to_csv("latent_dim_ranking.csv", index=False)
    print("all 16 dims silhouette:", round(silhouette_score(Zs, ls), 3))
    print(r.groupby("n_dims").head(3).to_string(index=False))
    best2 = [int(s[2:]) - 1 for s in r[r.n_dims == 2].iloc[0]["dims"].split("+")]
    best3 = [int(s[2:]) - 1 for s in r[r.n_dims == 3].iloc[0]["dims"].split("+")]
    names = json.load(open("cluster_names.json"))
    order = sorted(range(K), key=lambda k: -share[k])

    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    for k in order:
        idx = np.where(lab == k)[0]
        pick = rng.choice(idx, min(n_per, len(idx)), replace=False)
        c = COLORS[k % len(COLORS)]
        ax.scatter(Z[pick, best2[0]], Z[pick, best2[1]], s=6, color=c, alpha=0.55, lw=0,
                   label=f"{names.get(str(k), f'C{k}')} ({share[k]:.1%})")
        ellipse(ax, Z[idx][:, best2], c, ls="--" if str(k) in os.environ.get("MIXED", "6").split(",") else "-",
                fill=0.03 if str(k) in os.environ.get("MIXED", "6").split(",") else 0.10)
    ax.set_xlabel(f"LV{best2[0] + 1}"); ax.set_ylabel(f"LV{best2[1] + 1}")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="best", fontsize=9, markerscale=3)
    plt.tight_layout(); plt.savefig("fig_latent_best2d.png", dpi=300); plt.close()

    fig = plt.figure(figsize=(7.5, 6.2)); ax = fig.add_subplot(111, projection="3d")
    for k in order:
        idx = np.where(lab == k)[0]
        pick = rng.choice(idx, min(n_per, len(idx)), replace=False)
        ax.scatter(*(Z[pick, d] for d in best3), s=5, color=COLORS[k % len(COLORS)], alpha=0.55, lw=0,
                   label=f"{names.get(str(k), f'C{k}')} ({share[k]:.1%})")
    ax.set_xlabel(f"LV{best3[0] + 1}"); ax.set_ylabel(f"LV{best3[1] + 1}"); ax.set_zlabel(f"LV{best3[2] + 1}")
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.view_init(elev=22, azim=-55)
    ax.legend(loc="upper left", fontsize=8, markerscale=3)
    plt.tight_layout(); plt.savefig("fig_latent_best3d.png", dpi=300); plt.close()
    print("saved fig_latent_best2d.png, fig_latent_best3d.png")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 400)
