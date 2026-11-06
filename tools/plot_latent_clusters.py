"""
plot_latent_clusters.py —— GRU-AE 潜空间中的 GMM 驾驶状态（英文、简化）。在 runs/<run> 目录下运行：

    python ../../tools/plot_latent_clusters.py [OUT=fig_latent_clusters.png] [N_PER_CLUSTER=600]

- 窗口：验证集全部窗口（val_windows.npz 的 vid/start，与检测时同一组窗口与 GMM 标签）。
- 潜向量：gruae_model.keras 的 latent_vector 层。必须在训练时的同一设备上运行（Mac 上即 GPU/Metal）：
  该模型在隐藏 GPU 的进程中载入后输出完全不同（见 exper_fixed._baseline_task 注释）。
- LV1/LV2：标准化潜向量的前两个主成分。
- 每类随机抽 N_PER_CLUSTER 个窗口作散点（小类也看得见），椭圆 = 该类投影点的 95% 高斯置信椭圆。
- 类名读 cluster_names.json（按训练集各类画像命名；模型重训后类序号会变，须重新核对）。
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

MODEL_COLS = ["phi1", "phi2", "phi3", "phi4", "is_truck"]
T = 90
COLORS = ["#3b6fb6", "#2a9d8f", "#8ab17d", "#e9a23b", "#c8553d", "#9b72aa", "#6c757d"]


def build_windows(vid, start):
    df = pd.read_csv("phi_val.csv", usecols=["Vehicle_ID", "Frame_ID"] + MODEL_COLS)
    df["is_truck"] = df["is_truck"].fillna(0)
    df = df.sort_values(["Vehicle_ID", "Frame_ID"])
    by = {int(v): (g["Frame_ID"].to_numpy(int), g[MODEL_COLS].to_numpy(float)) for v, g in df.groupby("Vehicle_ID")}
    X = np.empty((len(vid), T, len(MODEL_COLS)))
    for i, (v, s) in enumerate(zip(vid, start)):
        fr, val = by[int(v)]
        j = np.searchsorted(fr, s)
        X[i] = val[j:j + T]
    return X


def ellipse(ax, pts, color, q=float(os.environ.get("ELLIPSE_CHI2", "9.210")), ls="-", fill=0.10):   # 默认 χ²(2) 99%   # χ²(2) 95%；稳健协方差（MCD），不让长尾离群点撑大椭圆
    from sklearn.covariance import MinCovDet
    sub = pts if len(pts) <= 5000 else pts[np.random.default_rng(0).choice(len(pts), 5000, replace=False)]
    try:
        m = MinCovDet(random_state=0).fit(sub); mu, cov = m.location_, m.covariance_
    except ValueError:
        mu, cov = sub.mean(0), np.cov(sub.T)
    cov = cov + np.eye(2) * 1e-4 * np.trace(cov)
    w, V = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
    w2, h2 = 2 * np.sqrt(q * w[1]), 2 * np.sqrt(q * w[0])
    ax.add_patch(Ellipse(mu, w2, h2, angle=ang, facecolor=color, alpha=fill, edgecolor="none"))
    ax.add_patch(Ellipse(mu, w2, h2, angle=ang, facecolor="none", edgecolor=color, lw=1.8, ls=ls))
    return mu, cov, q


def main(out="fig_latent_clusters.png", n_per=600):
    import tensorflow as tf
    w = np.load("val_windows.npz")
    vid, start, lab = w["vid"], w["start"], w["lab"]
    X = build_windows(vid, start)
    norm = np.load("norm_params.npz")
    Xn = (X - norm["mu"]) / norm["sd"]
    from New import zero_recurrent_bias
    ae = zero_recurrent_bias(tf.keras.models.load_model("gruae_model.keras", compile=False))
    enc = tf.keras.Model(ae.input, ae.get_layer("latent_vector").output)
    Z = enc.predict(Xn, verbose=0, batch_size=2048)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)
    share = {k: float(np.mean(lab == k)) for k in range(lab.max() + 1)}   # 真实占比（全部验证窗口），抽样前算
    proj = os.environ.get("PROJ", "tsne")
    if proj == "lda":      # 线性判别投影：选最能分开各类的两个线性方向（LV1/LV2）
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        pj = LinearDiscriminantAnalysis(n_components=2).fit(Z, lab); P = pj.transform(Z)
        info = f"LDA explained ratio {pj.explained_variance_ratio_[:2].round(3).tolist()}"
    elif proj == "tsne":   # 非线性，只用于展示：对抽样点做 t-SNE
        from sklearn.manifold import TSNE
        rng0 = np.random.default_rng(42)
        keep = np.concatenate([rng0.choice(np.where(lab == k)[0], min(n_per, (lab == k).sum()), replace=False)
                               for k in range(lab.max() + 1)])
        Z, lab = Z[keep], lab[keep]
        # 平稳行驶窗口（φ≡0）的潜向量几乎完全相同，t-SNE 会把重复点排成弧形（"月牙"）假象；
        #   降维前加 0.01 SD 的高斯扰动打破重复（仅用于展示，图注说明）。JITTER=0 关闭。
        jit = float(os.environ.get("JITTER", "0.01"))
        if jit > 0:
            Z = Z + np.random.default_rng(1).normal(0, jit, Z.shape)
        P = TSNE(2, perplexity=40, random_state=42, init="pca").fit_transform(Z); info = "t-SNE"
    else:
        from sklearn.decomposition import PCA
        pj = PCA(2, random_state=42).fit(Z); P = pj.transform(Z)
        info = f"PCA explained variance {pj.explained_variance_ratio_.round(3).tolist()}"
    names = json.load(open("cluster_names.json"))
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"]})
    order = sorted(share, key=lambda k: -share[k])
    dropped = {}
    for k in order:
        idx = np.where(lab == k)[0]
        if len(idx) < 3:
            continue
        pick = rng.choice(idx, min(n_per, len(idx)), replace=False)
        c = COLORS[k % len(COLORS)]
        mixed = os.environ.get("MIXED", "6").split(",")   # 散布在多类之间的混合类：虚线、淡填充
        mu_e, cov_e, q_e = ellipse(ax, P[idx], c, ls="--" if str(k) in mixed else "-", fill=0.03 if str(k) in mixed else 0.10)
        if os.environ.get("INSIDE_ONLY") == "1":
            # 只画落在本类椭圆内的点（展示用；被省略的比例打印出来，须写进图注）
            dlt = P[pick] - mu_e
            d2 = np.einsum("ij,jk,ik->i", dlt, np.linalg.inv(cov_e), dlt)
            ins_ = d2 <= q_e
            dropped[k] = (int((~ins_).sum()), len(pick))
            pick = pick[ins_]
        ax.scatter(P[pick, 0], P[pick, 1], s=6, color=c, alpha=0.55, lw=0,
                   label=f"{names.get(str(k), f'C{k}')} ({share[k]:.1%})")
    ax.set_xlabel("t-SNE 1" if proj == "tsne" else "LV1"); ax.set_ylabel("t-SNE 2" if proj == "tsne" else "LV2")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="best", fontsize=9, frameon=True, markerscale=3)
    plt.tight_layout(); plt.savefig(out, dpi=300); plt.close()
    if dropped:
        tot = sum(v[0] for v in dropped.values()), sum(v[1] for v in dropped.values())
        print("omitted points outside own-cluster ellipse:", {names.get(str(k), k): f"{a}/{n}" for k, (a, n) in dropped.items()},
              f"total {tot[0]}/{tot[1]} ({tot[0] / tot[1]:.1%})")
    print(f"saved {out}; {info}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "fig_latent_clusters.png", int(a[1]) if len(a) > 1 else 600)
