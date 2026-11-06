
import os, sys, random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from sklearn.mixture import GaussianMixture
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family":    "serif",
    "font.serif":     ["Times New Roman"],
    "font.size":      16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "axes.titlesize":  16,
    "figure.dpi":     150,
    "pdf.fonttype":   42,
})

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

PHI_FEATURES = ["phi1", "phi2", "phi3", "phi4"]
LATENT_LIST  = [8, 16, 32]
COLORS       = ["#F4A582", "#2166AC", "#4DAC26"]   # orange / blue / green

# ── Same build_gruae as New2.py ───────────────────────────────────────────────
def build_gruae(time_steps, num_features, latent_dim=16, dropout=0.1,
                l2_enc=1e-3, l2_dec=1e-3):
    inp = layers.Input(shape=(time_steps, num_features))
    x   = layers.GRU(64, return_sequences=True, dropout=dropout,
                     kernel_regularizer=regularizers.l2(l2_enc))(inp)
    x   = layers.LayerNormalization()(x)
    x   = layers.GRU(32, return_sequences=False, dropout=dropout,
                     kernel_regularizer=regularizers.l2(l2_enc))(x)
    x   = layers.LayerNormalization()(x)
    z   = layers.Dense(latent_dim, activation="linear",
                       kernel_regularizer=regularizers.l2(l2_enc),
                       name="latent_vector")(x)
    h   = layers.Dense(time_steps * num_features, activation="linear",
                       kernel_regularizer=regularizers.l2(l2_dec))(z)
    out = layers.Reshape((time_steps, num_features))(h)
    encoder     = models.Model(inp, z,   name="encoder")
    autoencoder = models.Model(inp, out, name="autoencoder")
    return autoencoder, encoder

# ── Same LatentVarianceLogger as New2.py ─────────────────────────────────────
class LatentVarianceLogger(Callback):
    def __init__(self, encoder, X_train, sample_size=2000, seed=42):
        super().__init__()
        self.encoder     = encoder
        self.total_variances = []
        rng = np.random.default_rng(seed)
        n   = min(sample_size, len(X_train))
        self.sample_idx = rng.choice(len(X_train), size=n, replace=False)
        self.X_sample   = X_train[self.sample_idx]

    def on_epoch_end(self, epoch, logs=None):
        Z = self.encoder(self.X_sample, training=False).numpy()
        self.total_variances.append(float(np.var(Z, axis=0).sum()))

# ── Load phi_all.csv and split (same logic as New2.py) ───────────────────────
print("Loading phi_all.csv ...")
df_phi = pd.read_csv("phi_all.csv", low_memory=False)
for c in ["Vehicle_ID","Frame_ID","Global_Time"] + PHI_FEATURES:
    if c in df_phi.columns:
        df_phi[c] = pd.to_numeric(df_phi[c], errors="coerce")

gt_events = pd.read_csv("HighSim_events_Anomaly30_merge2s.csv", low_memory=False)
gt_events["Vehicle_ID"] = pd.to_numeric(gt_events["Vehicle_ID"], errors="coerce").astype(int)
gt_vids = set(gt_events["Vehicle_ID"].astype(int))

L1, S1 = 90, 30
cnt          = df_phi.groupby("Vehicle_ID").size()
eligible_ids = np.sort(cnt[cnt >= L1].index.to_numpy())
gt_vids_array = np.array([v for v in eligible_ids if v in gt_vids])
normal_ids    = np.array([v for v in eligible_ids if v not in gt_vids])

rng      = np.random.default_rng(SEED)
shuffled = rng.permutation(normal_ids)
split    = int(0.7 * len(normal_ids))
train_ids = shuffled[:split]
val_ids   = shuffled[split:]

df_phi_train      = df_phi[df_phi["Vehicle_ID"].isin(train_ids)].copy()
df_phi_val_normal = df_phi[df_phi["Vehicle_ID"].isin(val_ids) &
                           ~df_phi["Vehicle_ID"].isin(gt_vids)].copy()

def sliding_windows(df, window_size=L1, step_size=S1):
    X, meta = [], []
    for vid, g in df.groupby("Vehicle_ID"):
        g = g.sort_values("Frame_ID")
        feats  = g[PHI_FEATURES].to_numpy(dtype=float)
        frames = g["Frame_ID"].to_numpy(dtype=int)
        n = len(feats)
        if n < window_size: continue
        t = 0
        while t + window_size <= n:
            win = feats[t:t+window_size]
            fs  = frames[t:t+window_size]
            if np.all(np.diff(fs)==1) and not np.isnan(win).any():
                X.append(win)
                meta.append({"Vehicle_ID": int(vid),
                             "start_frame": int(fs[0]),
                             "end_frame":   int(fs[-1])})
            t += step_size
    return (np.array(X) if X else
            np.empty((0, window_size, len(PHI_FEATURES)))), meta

print("Building sliding windows ...")
X_train,      meta_train = sliding_windows(df_phi_train)
X_val_normal, _          = sliding_windows(df_phi_val_normal)

# Normalize (train stats only) — same as New2.py
train_flat = X_train.reshape(-1, len(PHI_FEATURES))
mu = train_flat.mean(axis=0, keepdims=True)
sd = train_flat.std(axis=0,  keepdims=True) + 1e-8
X_train_n      = (X_train      - mu) / sd
X_val_normal_n = (X_val_normal - mu) / sd

print(f"Train windows: {len(X_train_n):,}  Val-normal windows: {len(X_val_normal_n):,}")

# ── Sweep ─────────────────────────────────────────────────────────────────────
all_histories = {}

for lat in LATENT_LIST:
    print(f"\n{'='*50}")
    print(f"  Training latent d={lat} ...")
    print(f"{'='*50}")

    ae, encoder = build_gruae(L1, len(PHI_FEATURES),
                              latent_dim=lat, dropout=0.1)
    ae.compile(optimizer=Adam(1e-3),
               loss=tf.keras.losses.MeanSquaredError())

    var_logger = LatentVarianceLogger(encoder, X_train_n)

    history = ae.fit(
        X_train_n, X_train_n,
        validation_data=(X_val_normal_n, X_val_normal_n),
        epochs=500,
        batch_size=256,
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=15, min_lr=1e-5),
            var_logger,
        ],
        verbose=1
    )

    all_histories[lat] = {
        "train_loss":   history.history["loss"],
        "val_loss":     history.history["val_loss"],
        "latent_var":   var_logger.total_variances,
    }

    n_ep = len(history.history["loss"])
    print(f"  → epochs={n_ep}  "
          f"final_train={history.history['loss'][-1]:.5f}  "
          f"final_val={history.history['val_loss'][-1]:.5f}")

# Save histories
np.savez("latent_training_histories.npz",
         **{f"train_{lat}": all_histories[lat]["train_loss"] for lat in LATENT_LIST},
         **{f"val_{lat}":   all_histories[lat]["val_loss"]   for lat in LATENT_LIST},
         **{f"var_{lat}":   all_histories[lat]["latent_var"] for lat in LATENT_LIST})
print("\nSaved latent_training_histories.npz")

# ── Plot: 1×2  (train MSE | val MSE) with all 3 latent dims overlaid ─────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for ax, loss_key, title in [
    (axes[0], "train_loss", "Training MSE"),
    (axes[1], "val_loss",   "Validation MSE"),
]:
    for lat, color in zip(LATENT_LIST, COLORS):
        vals   = all_histories[lat][loss_key]
        epochs = list(range(1, len(vals)+1))
        lw     = 2.2 if lat == 16 else 1.5
        # label  = f"$d={lat}$" + (" (selected)" if lat == 16 else "")
        label = f"$D={lat}$"
        ax.plot(epochs, vals, color=color, lw=lw, label=label)

    ax.set_xlabel("Epoch", labelpad=4)
    ax.set_ylabel("MSE", labelpad=4)
    ax.set_title(title, pad=6)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.tick_params(length=3)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)
    ax.legend(framealpha=0.92, edgecolor="#CCCCCC")

# fig.suptitle("GRU-AE Convergence for Different Latent Dimensions",
#              fontsize=1, y=1.01)
fig.tight_layout()
fig.savefig("fig_latent_loss_curves.pdf", bbox_inches="tight")
fig.savefig("fig_latent_loss_curves.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("Saved fig_latent_loss_curves.pdf / .png")

# ── Optional: also plot latent variance ──────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(6, 4))
for lat, color in zip(LATENT_LIST, COLORS):
    vals   = all_histories[lat]["latent_var"]
    epochs = list(range(1, len(vals)+1))
    lw     = 2.2 if lat == 16 else 1.5
    # label  = f"$d={lat}$" + (" (selected)" if lat == 16 else "")
    label = f"$D={lat}$"
    ax2.plot(epochs, vals, color=color, lw=lw, label=label)

ax2.set_xlabel("Epoch", labelpad=4)
ax2.set_ylabel("Total Latent Variance", labelpad=4)
ax2.set_title("Total Variance of Latent Variables During Training", pad=6)
ax2.spines["top"].set_visible(True)
ax2.spines["right"].set_visible(True)
ax2.tick_params(length=3)
ax2.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)
ax2.legend(framealpha=0.92, edgecolor="#CCCCCC")
fig2.tight_layout()
fig2.savefig("fig_latent_variance_curves.pdf", bbox_inches="tight")
fig2.savefig("fig_latent_variance_curves.png", bbox_inches="tight", dpi=300)
plt.close(fig2)
print("Saved fig_latent_variance_curves.pdf / .png")