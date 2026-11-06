""""
GMM Cluster Visualization  –  t-SNE + per-cluster confidence ellipse
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Ellipse
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

FONT_SIZE = 16

matplotlib.rcParams.update({
    "font.family":     "serif",
    "font.serif":      ["Times New Roman", "DejaVu Serif"],
    "font.size":       FONT_SIZE,
    "axes.labelsize":  FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "axes.titlesize":  FONT_SIZE,
    "figure.dpi":      150,
    "pdf.fonttype":    42,
})

# 8 GMM clusters → 4 visualization groups
CLUSTER_TO_GROUP = {0: 1, 1: 2, 2: 0, 3: 0, 4: 3, 5: 1, 6: 2, 7: 3}

# 4-group display config
CLUSTER_CFG = {
    0: dict(label="Cruising",           color="#3070B0"),  # blue
    1: dict(label="Following",          color="#1A9896"),  # teal
    2: dict(label="Lane Changing",      color="#E08030"),  # orange
    3: dict(label="Mixed Interaction",  color="#9B59B6"),  # purple
}

FEATURE_COLS = ["phi1", "phi2", "phi3", "phi4"]

# ── Load SPE data for bar chart ───────────────────────────────────────────────
def load_spe_by_group():
    df_pred  = pd.read_csv("pred_frames_val.csv")
    df_clust = pd.read_csv("phi_val_with_cluster.csv",
                           usecols=["Vehicle_ID", "Frame_ID", "cluster"])
    df_m = df_pred.merge(df_clust, on=["Vehicle_ID", "Frame_ID"], how="left")
    df_m = df_m.dropna(subset=["cluster"])
    df_m["cluster"] = df_m["cluster"].astype(int).map(CLUSTER_TO_GROUP)
    df_m = df_m.dropna(subset=["cluster"])
    df_m["cluster"] = df_m["cluster"].astype(int)
    return df_m


CLIP_Q = 0.995
N_TSNE = 10_000

# ── [ORIGINAL] Load & build window features (phi mean/std → t-SNE) ──────────
# NOTE: Commented out — replaced by latent-vector approach below.
#       Original used phi mean/std per window as t-SNE input, which does NOT
#       match the actual GMM input space (encoder latent vectors).
#
# print("Loading ...")
# df_win = pd.read_csv("phi_val_window_cluster.csv")
# df_frm = pd.read_csv("phi_val_with_cluster.csv",
#                      usecols=["Vehicle_ID", "Frame_ID"] + FEATURE_COLS)
# records = []
# for _, row in df_win.iterrows():
#     vid=int(row["Vehicle_ID"]); sf=int(row["start_frame"])
#     ef=int(row["end_frame"]);   clk=int(row["cluster"])
#     sub = df_frm.query("Vehicle_ID==@vid and Frame_ID>=@sf and Frame_ID<=@ef")
#     if len(sub)==0: continue
#     rec = {"cluster": clk}
#     for c in FEATURE_COLS:
#         rec[f"{c}_mean"] = sub[c].mean()
#         rec[f"{c}_std"]  = sub[c].std(ddof=0)
#     records.append(rec)
# df = pd.DataFrame(records).dropna()
# df["cluster"] = df["cluster"].astype(int)
# df = df[df["cluster"].isin(CLUSTER_CFG)]
# feat_cols = [f"{c}_mean" for c in FEATURE_COLS] + \
#             [f"{c}_std"  for c in FEATURE_COLS]
# for c in feat_cols:
#     lo = df[c].quantile(1 - CLIP_Q)
#     hi = df[c].quantile(CLIP_Q)
#     df[c] = df[c].clip(lo, hi)
# parts = []
# for k in sorted(CLUSTER_CFG):
#     sub = df[df["cluster"] == k]
#     if len(sub) == 0: continue
#     n = min(len(sub), max(500, int(N_TSNE * len(sub) / len(df))))
#     parts.append(sub.sample(n, random_state=42))
# df_sub = pd.concat(parts, ignore_index=True)
# scaler = StandardScaler()
# X_sc = scaler.fit_transform(df_sub[feat_cols].values)
# tsne = TSNE(n_components=2, perplexity=40, learning_rate="auto",
#             init="pca", max_iter=1000, random_state=42, n_jobs=-1)
# Z = tsne.fit_transform(X_sc)
# df_sub["tx"] = Z[:, 0]; df_sub["ty"] = Z[:, 1]

# ── NEW: Encoder latent vectors → t-SNE (matches GMM input space) ───────────
WINDOW_SIZE = 90
STEP_SIZE   = 30

def _build_windows(df_phi, feature_cols, window_size, step):
    """Rebuild sliding windows identical to New2.py fixed_sliding_windows."""
    X, vids, sfs = [], [], []
    for vid, g in df_phi.groupby("Vehicle_ID"):
        g = g.sort_values("Frame_ID").reset_index(drop=True)
        feats  = g[feature_cols].to_numpy(dtype=float)
        frames = g["Frame_ID"].to_numpy()
        t = 0
        while t + window_size <= len(feats):
            X.append(feats[t:t + window_size])
            vids.append(int(vid))
            sfs.append(int(frames[t]))
            t += step
    return (np.array(X) if X else np.empty((0, window_size, len(feature_cols))),
            vids, sfs)

print("Loading phi features and cluster labels ...")
df_phi    = pd.read_csv("phi_val.csv")
df_wclust = pd.read_csv("phi_val_window_cluster.csv")
df_wclust["cluster"] = df_wclust["cluster"].astype(int).map(CLUSTER_TO_GROUP)
df_wclust = df_wclust.dropna(subset=["cluster"]).copy()
df_wclust["cluster"] = df_wclust["cluster"].astype(int)

# Build cluster lookup: (Vehicle_ID, start_frame) → group
clust_lookup = {
    (int(r["Vehicle_ID"]), int(r["start_frame"])): int(r["cluster"])
    for _, r in df_wclust.iterrows()
}

print("Building windows ...")
X_raw, vids, sfs = _build_windows(df_phi, FEATURE_COLS, WINDOW_SIZE, STEP_SIZE)

# Assign cluster labels; drop windows with no matching cluster
labels = np.array([clust_lookup.get((v, s), -1) for v, s in zip(vids, sfs)])
valid  = labels >= 0
X_raw, labels = X_raw[valid], labels[valid]

# Normalize with parameters saved by New2.py
norm_data = np.load("norm_params.npz")
mu, sd    = norm_data["mu"], norm_data["sd"]
X_n       = (X_raw - mu) / sd

# Load encoder (latent_vector layer of saved autoencoder)
print("Loading encoder ...")
ae      = tf.keras.models.load_model("gruae_model.keras", compile=False)
encoder = tf.keras.Model(ae.input, ae.get_layer("latent_vector").output,
                         name="encoder")
print("Running encoder inference ...")
Z_latent = encoder.predict(X_n, batch_size=512, verbose=0)   # (N, latent_dim)

# Normalize latent vectors before t-SNE
Z_mu  = Z_latent.mean(axis=0, keepdims=True)
Z_sd  = Z_latent.std(axis=0,  keepdims=True) + 1e-8
Z_n   = ((Z_latent - Z_mu) / Z_sd).astype(np.float32)

# Stratified subsample
rng   = np.random.default_rng(42)
parts = []
for k in sorted(CLUSTER_CFG):
    idx = np.where(labels == k)[0]
    if len(idx) == 0:
        continue
    n = min(len(idx), max(500, int(N_TSNE * len(idx) / len(labels))))
    chosen = rng.choice(idx, n, replace=False)
    parts.append({"Z": Z_n[chosen], "labels": labels[chosen]})

Z_sub   = np.vstack([p["Z"]      for p in parts])
lab_sub = np.concatenate([p["labels"] for p in parts])
print(f"  t-SNE input: {len(Z_sub):,} windows  (latent dim={Z_latent.shape[1]})")

# ── t-SNE ────────────────────────────────────────────────────────────────────
print("Running t-SNE (~30s) ...")
tsne = TSNE(n_components=2, perplexity=40, learning_rate="auto",
            init="pca", max_iter=1000, random_state=42, n_jobs=-1)
Z2   = tsne.fit_transform(Z_sub)
df_sub = pd.DataFrame({"tx": Z2[:, 0], "ty": Z2[:, 1], "cluster": lab_sub})
print("  Done.")

# ── Per-cluster: keep inner 90% for display & ellipse fitting ────────────────
def core_points(sub, q_lo=0.15, q_hi=0.85):
    """Keep points within [q_lo, q_hi] quantile range on both axes."""
    mask = (
        (sub["tx"] >= sub["tx"].quantile(q_lo)) &
        (sub["tx"] <= sub["tx"].quantile(q_hi)) &
        (sub["ty"] >= sub["ty"].quantile(q_lo)) &
        (sub["ty"] <= sub["ty"].quantile(q_hi))
    )
    return sub[mask]

# ── Ellipse drawing ───────────────────────────────────────────────────────────
def draw_ellipse(ax, x, y, color, n_std=2.0, lw=2.2, zorder=6):
    cov = np.cov(x, y)
    cx, cy = x.mean(), y.mean()
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-12)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    w = 2 * n_std * np.sqrt(eigvals[0])
    h = 2 * n_std * np.sqrt(eigvals[1])
    ellipse = Ellipse(xy=(cx, cy), width=w, height=h, angle=angle,
                      edgecolor=color, facecolor=color,
                      linewidth=lw, linestyle="-",
                      alpha=0.12, zorder=zorder-1)
    ax.add_patch(ellipse)
    ellipse2 = Ellipse(xy=(cx, cy), width=w, height=h, angle=angle,
                       edgecolor=color, facecolor="none",
                       linewidth=lw, linestyle="-",
                       alpha=0.85, zorder=zorder)
    ax.add_patch(ellipse2)

# ── Plot ─────────────────────────────────────────────────────────────────────
df_spe_all = load_spe_by_group()

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15.0, 6.0))

# Draw large clusters first (background), small last (foreground)
draw_order = sorted(CLUSTER_CFG,
                    key=lambda k: (df_sub["cluster"]==k).sum(), reverse=True)

label_positions = {}

for k in draw_order:
    cfg   = CLUSTER_CFG[k]
    sub   = df_sub[df_sub["cluster"] == k]
    core  = core_points(sub)          # trimmed 90% for scatter + ellipse
    color = cfg["color"]
    x, y  = core["tx"].values, core["ty"].values

    # Keep only points inside 1.2-sigma ellipse (Mahalanobis distance ≤ n_std)
    cx_e, cy_e = x.mean(), y.mean()
    cov = np.cov(x, y)
    try:
        cov_inv = np.linalg.inv(cov)
        diff = np.column_stack([x - cx_e, y - cy_e])
        maha = np.sqrt(np.einsum("ij,jk,ik->i", diff, cov_inv, diff))
        inside = maha <= 2.0   # keep points within 1.5-sigma
        x_in, y_in = x[inside], y[inside]
    except Exception:
        x_in, y_in = x, y

    dot_sz = 15 if len(x_in) > 500 else 35
    ax.scatter(x_in, y_in, c=color, s=dot_sz, alpha=0.55,
               linewidths=0, rasterized=True, zorder=3)

    # Centroid marker
    # ax.scatter(cx_e, cy_e, c=color, s=180, marker="D",
    #            edgecolors="white", linewidths=1.6, zorder=11)

    # Ellipse fitted to core points
    draw_ellipse(ax, x, y, color, n_std=2.0, lw=2.2, zorder=6)

    # Store centroid for label placement
    label_positions[k] = (x.mean(), y.mean())

ax.autoscale_view()
ax.set_xlabel("LV1", labelpad=6)
ax.set_ylabel("LV2", labelpad=6)
# ax.set_title(
#     r"Normal Driving Behavior Clusters",
#     pad=10, fontweight="bold")
ax.spines["top"].set_visible(True)
ax.spines["right"].set_visible(True)
ax.tick_params(length=0, labelbottom=False, labelleft=False)
ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.25)
# ax.grid(False)

handles = sorted(
    [
        Patch(facecolor=CLUSTER_CFG[k]["color"], edgecolor=CLUSTER_CFG[k]["color"],
              alpha=0.75, label=CLUSTER_CFG[k]["label"])
        for k in CLUSTER_CFG
    ],
    key=lambda h: h.get_label()
)
ax.legend(handles=handles, loc="upper right",
          framealpha=0.94, edgecolor="#CCCCCC", fontsize=FONT_SIZE)

# ── SPE bar chart (ax2) — 4 groups ───────────────────────────────────────────
bar_x  = np.arange(len(CLUSTER_CFG))
clip_q = 0.95

for i, k in enumerate(sorted(CLUSTER_CFG)):
    spe_k = df_spe_all.loc[df_spe_all["cluster"] == k, "frame_spe"].values
    ucl_k = df_spe_all.loc[df_spe_all["cluster"] == k, "ucl_frame"].mean()
    if len(spe_k) == 0:
        continue
    spe_k    = spe_k[spe_k <= np.quantile(spe_k, clip_q)]
    mean_spe = spe_k.mean()
    std_spe  = spe_k.std()
    yerr     = [[min(std_spe, mean_spe)], [std_spe]]

    ax2.bar(bar_x[i], mean_spe, yerr=yerr, width=0.55,
            color=CLUSTER_CFG[k]["color"], alpha=0.78, capsize=6,
            error_kw={"ecolor": "dimgray", "elinewidth": 1.5},
            label=CLUSTER_CFG[k]["label"])
    ax2.hlines(ucl_k, bar_x[i] - 0.30, bar_x[i] + 0.30,
               colors=CLUSTER_CFG[k]["color"], linestyles="--", linewidth=2.0)
    ax2.text(bar_x[i], ucl_k + 0.01, f"UCL={ucl_k:.3f}",
             ha="center", va="bottom", fontsize=FONT_SIZE - 4,
             color=CLUSTER_CFG[k]["color"])

ax2.set_xticks(bar_x)
ax2.set_xticklabels(
    [CLUSTER_CFG[k]["label"] for k in sorted(CLUSTER_CFG)],
    fontsize=FONT_SIZE
)
ax2.set_ylabel("SPE", fontsize=FONT_SIZE)
ax2.set_ylim(bottom=0, top=4.5)
ax2.spines["top"].set_visible(True)
ax2.spines["right"].set_visible(True)
ax2.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
ax2.tick_params(axis="x", length=0)

fig.tight_layout()
fig.savefig("fig_gmm_clusters.png", bbox_inches="tight", dpi=300)
print("Saved.")