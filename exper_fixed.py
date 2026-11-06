
import os, random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

DATA_PATH      = "HighSim_frames_with_Anomaly30.csv"
# full GT（HardBrake/HardAccel + TTC<1.5s，已剔除轨迹首尾/重建失败等不可信帧）。
# 去噪后 accel-only GT 只剩个位数事件，不足以做方法对比。
GT_EVENTS_PATH = "HighSim_events_Anomaly30_merge2s.csv"
PHI_VAL_PATH   = "phi_all.csv"

WINDOW    = 90
STEP      = 30
FPS       = 30
UCL_Q     = 0.99
MIN_DUR   = 30
GAP_MERGE = int(round(float(os.environ.get("EVENT_GAP_MERGE_S", WINDOW / FPS)) * FPS))   # v16b：相邻事件间隔 ≤ 窗口长度 3 s 合并（与 New.EVENT_GAP_MERGE 相同）
GAP_TOL   = 15   # 与 New.GAP_TOL_FRAMES 一致：≤0.5 s 短间断先合并
SPE_RATIO_THRESH = None   # 事件 = SPE>UCL 连续 ≥1 s；不再要求平均 SPE ≥ 5×UCL（与 New.py 一致）

RAW_FEATURES = ["Speed_smooth", "Acc_from_vsmooth_g", "Local_X", "Space_Headway"]
PHI_FEATURES = ["phi1", "phi2", "phi3", "phi4", "is_truck"]   # 与 New.MODEL_COLS 一致
N_PHI        = 4   # 前 4 个通道被重构；is_truck 只作条件输入


# ── Data loading ──────────────────────────────────────────────────────────────
def load_real_data(path):
    df = pd.read_csv(path, low_memory=False)
    num_cols = [
        "Vehicle_ID", "Global_Time", "Frame_ID", "Local_X", "Local_Y",
        "Global_X", "Global_Y", "Width", "Length", "Class", "Speed",
        "Acceleration", "Lane_Num", "Preceding_Vehicle_ID", "Space_Headway",
        "Speed_smooth", "Acc_from_vsmooth_ftps2", "Acc_from_vsmooth_g",
        "HardBrake", "HardAccel", "FreewayDecel", "Anomaly_30",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values(["Vehicle_ID", "Global_Time", "Frame_ID"]).reset_index(drop=True)


def load_gt_events(path):
    ev = pd.read_csv(path, low_memory=False)
    for c in ["Vehicle_ID", "start_time", "end_time", "start_frame", "end_frame", "duration_s"]:
        if c in ev.columns:
            ev[c] = pd.to_numeric(ev[c], errors="coerce")
    ev = ev.dropna(subset=["Vehicle_ID", "start_frame", "end_frame", "start_time", "end_time"])
    ev["Vehicle_ID"] = ev["Vehicle_ID"].astype(int)
    return ev.sort_values(["Vehicle_ID", "start_time"]).reset_index(drop=True)


def clean_window_mask(meta, out_frames):
    """v8 与 New.py 一致：窗口内含任一 3σ 离群帧（o1–o4 任一）→ False。out_frames: {vid: 有序帧号数组}"""
    keep = np.ones(len(meta), bool)
    for i, m in enumerate(meta):
        f = out_frames.get(m["Vehicle_ID"])
        if f is not None and len(f):
            j = np.searchsorted(f, m["start_frame"])
            keep[i] = not (j < len(f) and f[j] <= m["end_frame"])
    return keep


def fixed_sliding_windows(df, feature_cols, window_size=WINDOW, step_size=STEP):
    X, meta = [], []
    for vid, g in df.dropna(subset=["Vehicle_ID"]).groupby("Vehicle_ID"):
        g      = g.sort_values("Frame_ID")
        feats  = g[feature_cols].to_numpy(dtype=float)
        frames = g["Frame_ID"].to_numpy(dtype=int)
        times  = g["Global_Time"].to_numpy(dtype=float) if "Global_Time" in g.columns else None
        n = len(feats)
        if n < window_size:
            continue
        t = 0
        while t + window_size <= n:
            win    = feats[t:t + window_size]
            fslice = frames[t:t + window_size]
            if not np.all(np.diff(fslice) == 1) or np.isnan(win).any():
                t += step_size
                continue
            X.append(win)
            meta.append({
                "Vehicle_ID":  int(vid),
                "start_frame": int(fslice[0]),
                "end_frame":   int(fslice[-1]),
                "start_time":  float(times[t]) if times is not None else np.nan,
            })
            t += step_size
    return (np.array(X) if X else np.empty((0, window_size, len(feature_cols)))), meta


def find_runs(mask):
    runs, i = [], 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < len(mask) and mask[j]:
            j += 1
        runs.append((i, j))
        i = j
    return runs


def merge_runs(runs, gap):
    if not runs:
        return []
    merged = [list(runs[0])]
    for s, e in runs[1:]:
        if s - merged[-1][1] <= gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def eventize(df_frame, min_dur=MIN_DUR, gap_merge=GAP_MERGE,
             min_spe_ratio=SPE_RATIO_THRESH):
    """
    Convert frame-level anomaly flags to events.
    Ratio filter (mean_spe / mean_ucl >= min_spe_ratio) is applied only when
    df_frame contains a 'ucl_frame' column — skipped for IForest and raw-feature
    methods where the SPE scale makes the threshold meaningless.
    """
    has_ucl = "ucl_frame" in df_frame.columns
    events  = []
    for vid, g in df_frame.groupby("Vehicle_ID"):
        g      = g.sort_values("Frame_ID").reset_index(drop=True)
        mask   = g["is_anom_frame"].to_numpy()
        frames = g["Frame_ID"].to_numpy()
        runs   = [(s, e) for s, e in merge_runs(find_runs(mask), GAP_TOL) if (e - s) >= min_dur]
        runs   = merge_runs(runs, gap_merge)
        for s, e in runs:
            if has_ucl and min_spe_ratio is not None:
                seg_spe  = g["frame_spe"].iloc[s:e].values
                seg_ucl  = g["ucl_frame"].iloc[s:e].values
                ucl_mean = seg_ucl.mean()
                if ucl_mean > 0 and seg_spe.mean() / ucl_mean < min_spe_ratio:
                    continue
            events.append({
                "Vehicle_ID":  int(vid),
                "start_frame": int(frames[s]),
                "end_frame":   int(frames[e - 1]),
                "duration_s":  (e - s) / FPS,
            })
    return (pd.DataFrame(events) if events
            else pd.DataFrame(columns=["Vehicle_ID", "start_frame", "end_frame", "duration_s"]))


def compute_total_hours(df):
    return float(df.groupby("Vehicle_ID")["Frame_ID"].count().sum()) / FPS / 3600


def compute_metrics(df_frame, pred_events, gt_events, total_hours, id_col="Vehicle_ID"):
    n_gt   = len(gt_events)
    n_pred = len(pred_events)
    tp_ev  = 0
    cov_list = []

    for _, ev in gt_events.iterrows():
        vid = int(ev[id_col])
        st, et = int(ev["start_frame"]), int(ev["end_frame"])
        dfv    = df_frame[
            (df_frame[id_col] == vid) &
            (df_frame["Frame_ID"] >= st) &
            (df_frame["Frame_ID"] <= et)
        ]
        gt_len = max(1, et - st + 1)
        n_hit  = int((dfv["is_anom_frame"] == 1).sum())
        cov    = n_hit / gt_len
        if n_hit > 0:
            tp_ev += 1
        cov_list.append(cov)

    fp_ev = 0
    for _, pev in pred_events.iterrows():
        vid    = int(pev[id_col])
        ps, pe = int(pev["start_frame"]), int(pev["end_frame"])
        gt_v   = gt_events[gt_events[id_col] == vid]
        overlap = any(
            not (pe < int(r["start_frame"]) or ps > int(r["end_frame"]))
            for _, r in gt_v.iterrows()
        )
        if not overlap:
            fp_ev += 1

    # 严格事件级指标（事件 ↔ 事件，时间区间有重叠即匹配）：
    #   recall    = 被至少一个检测事件重叠的 GT 事件数 / GT 事件数
    #   precision = 与至少一个 GT 事件重叠的检测事件数 / 检测事件数
    def _overlaps(a_vid, a_s, a_e, other):
        o = other[other[id_col] == a_vid]
        return bool(((o["start_frame"] <= a_e) & (o["end_frame"] >= a_s)).any())
    gt_hit = sum(_overlaps(int(r[id_col]), int(r["start_frame"]), int(r["end_frame"]), pred_events)
                 for _, r in gt_events.iterrows()) if n_pred else 0
    pred_tp = n_pred - fp_ev
    recall    = gt_hit / (n_gt + 1e-9)
    precision = pred_tp / (n_pred + 1e-9)
    f1        = 2 * recall * precision / (recall + precision + 1e-9)
    # 旧的宽松指标：GT 事件内只要有 1 帧被判异常就算命中（帧级命中率），仅作参考
    frame_hit_rate = tp_ev / (n_gt + 1e-9)
    far       = fp_ev / (total_hours + 1e-9)
    cov_arr   = np.array(cov_list)

    return {
        "n_gt":             n_gt,
        "n_pred":           n_pred,
        "event_recall":     round(recall,    4),
        "event_precision":  round(precision, 4),
        "event_f1":         round(f1,        4),
        "frame_hit_rate":   round(frame_hit_rate, 4),
        "false_alarm_rate": round(far,       4),
        "coverage_ge_0.3":  round(float((cov_arr >= 0.3).mean()), 4) if len(cov_arr) else 0.0,
        "coverage_ge_0.5":  round(float((cov_arr >= 0.5).mean()), 4) if len(cov_arr) else 0.0,
        "coverage_ge_0.8":  round(float((cov_arr >= 0.8).mean()), 4) if len(cov_arr) else 0.0,
        "mean_coverage":    round(float(cov_arr.mean()),          4) if len(cov_arr) else 0.0,
    }


# ── Frame df builders ─────────────────────────────────────────────────────────
def global_ucl_frame_level(meta, spe_2d, q=UCL_Q):
    """v16b：全局 UCL 与本文模型同一定义 —— 训练集每帧 SPE 取覆盖窗口的最大值，再取 q 分位数
    （= New.compute_cluster_ucl_frame_level 的 global_fallback）。原来对窗口×帧展平取分位数，
    重叠窗口重复计数且未取最大值，UCL 偏低（Utility_GRU_AE：2.11 vs 3.40）。"""
    from New import frame_spe_by_cluster
    d = frame_spe_by_cluster(meta, spe_2d, np.zeros(len(meta), int))
    return float(np.quantile(d["spe"].to_numpy(), q))


def frames_from_frame_spe(meta, spe_2d, ucl):
    """Per-frame SPE with a scalar UCL. Includes ucl_frame → ratio filter applies."""
    return frames_from_frame_spe_gmm(meta, spe_2d, np.full(len(meta), float(ucl)))


def frames_from_frame_spe_gmm(meta, spe_2d, ucl_per_window):
    """Per-frame SPE with per-window GMM UCL：每帧取 SPE/UCL 比值最大的覆盖窗口（与 New.py 同一规则）。"""
    from New import aggregate_window_frames
    return aggregate_window_frames(np.array([m["Vehicle_ID"] for m in meta]),
                                   np.array([m["start_frame"] for m in meta]), spe_2d, ucl_per_window)


def frames_from_window_score(meta, scores, ucl):
    """Window-level score uniformly distributed to all frames. No ucl_frame → ratio filter skipped."""
    rows = []
    for i, m in enumerate(meta):
        for f in range(int(m["start_frame"]), int(m["end_frame"]) + 1):
            rows.append({
                "Vehicle_ID":    int(m["Vehicle_ID"]),
                "Frame_ID":      f,
                "frame_spe":     float(scores[i]),
                "is_anom_frame": int(scores[i] > ucl),
            })
    return (pd.DataFrame(rows)
            .groupby(["Vehicle_ID", "Frame_ID"], as_index=False)
            .agg({"frame_spe": "max", "is_anom_frame": "max"}))


# ── B1: Rule Threshold ────────────────────────────────────────────────────────
def run_rule_baseline(df_raw_val, val_ids):
    print("\n── B1: Rule Threshold ──")
    df_v = df_raw_val[df_raw_val["Vehicle_ID"].isin(val_ids)].copy()
    hb = df_v["HardBrake"].fillna(0)    if "HardBrake"    in df_v.columns else 0
    ha = df_v["HardAccel"].fillna(0)    if "HardAccel"    in df_v.columns else 0
    fd = df_v["FreewayDecel"].fillna(0) if "FreewayDecel" in df_v.columns else 0
    df_v["is_anom_frame"] = ((hb > 0) | (ha > 0) | (fd > 0)).astype(int)
    df_v["frame_spe"] = (
        df_v["Acc_from_vsmooth_g"].abs() if "Acc_from_vsmooth_g" in df_v.columns
        else 0.0
    )
    df_frame = df_v[["Vehicle_ID", "Frame_ID", "is_anom_frame", "frame_spe"]].copy()
    return df_frame, eventize(df_frame)  # no ucl_frame → no ratio filter


# ── B2: Isolation Forest (phi features) ──────────────────────────────────────
def run_iforest(X_train_n, X_val_n, meta_val):
    print("\n── B2: Isolation Forest (raw features) ──")
    Xtr = X_train_n.reshape(X_train_n.shape[0], -1)
    Xva = X_val_n.reshape(X_val_n.shape[0], -1)
    clf = IsolationForest(n_estimators=200, contamination=0.05,
                          random_state=SEED, n_jobs=-1)
    clf.fit(Xtr)
    scores_tr = -clf.score_samples(Xtr)
    scores_va = -clf.score_samples(Xva)
    ucl = float(np.quantile(scores_tr, UCL_Q))
    print(f"  UCL: {ucl:.4f}")
    # IForest anomaly scores ≠ SPE → no ucl_frame → no ratio filter
    df_frame = frames_from_window_score(meta_val, scores_va, ucl)
    return df_frame, eventize(df_frame)


# ── B3: PCA-SPE (phi features, per-frame SPE via reshape) ────────────────────
def run_pca_spe(X_train_n, X_val_n, meta_val, meta_train, n_components=8):
    print(f"\n── B3: PCA-SPE (raw features, n_components={n_components}) ──")
    T, F = X_train_n.shape[1], X_train_n.shape[2]
    Xtr  = X_train_n.reshape(-1, T * F)
    Xva  = X_val_n.reshape(-1, T * F)
    pca  = PCA(n_components=n_components, random_state=SEED)
    pca.fit(Xtr)
    err_tr = Xtr - pca.inverse_transform(pca.transform(Xtr))
    err_va = Xva - pca.inverse_transform(pca.transform(Xva))
    spe_tr = (err_tr.reshape(-1, T, F) ** 2).sum(axis=2)  # (N, T)
    spe_va = (err_va.reshape(-1, T, F) ** 2).sum(axis=2)
    ucl    = global_ucl_frame_level(meta_train, spe_tr)
    print(f"  UCL: {ucl:.6f}")
    df_frame = frames_from_frame_spe(meta_val, spe_va, ucl)
    return df_frame, eventize(df_frame.drop(columns=["ucl_frame"]))


# ── AE builders ───────────────────────────────────────────────────────────────
def build_rnn_ae(time_steps, n_feat, latent_dim=16, dropout=0.1,
                 rnn_type="GRU", l2_reg=1e-3, n_out=None):
    """GRU/LSTM autoencoder matching New2.py architecture."""
    rnn = layers.GRU if rnn_type == "GRU" else layers.LSTM
    inp = layers.Input(shape=(time_steps, n_feat))
    x   = rnn(64, return_sequences=True,  dropout=dropout,
               kernel_regularizer=regularizers.l2(l2_reg))(inp)
    x   = layers.LayerNormalization()(x)
    x   = rnn(32, return_sequences=False, dropout=dropout,
               kernel_regularizer=regularizers.l2(l2_reg))(x)
    x   = layers.LayerNormalization()(x)
    z   = layers.Dense(latent_dim, activation="linear",
                       kernel_regularizer=regularizers.l2(l2_reg),
                       name="latent_vector")(x)
    n_out = n_feat if n_out is None else n_out
    h   = layers.Dense(time_steps * n_out, activation="linear",
                       kernel_regularizer=regularizers.l2(l2_reg))(z)
    out = layers.Reshape((time_steps, n_out))(h)
    encoder = models.Model(inp, z,   name=f"{rnn_type}_encoder")
    ae      = models.Model(inp, out, name=f"{rnn_type}_AE")
    return ae, encoder


def build_mlp_ae(input_dim, latent_dim=32):
    inp = layers.Input(shape=(input_dim,))
    x   = layers.Dense(256, activation="relu",
                        kernel_regularizer=regularizers.l2(1e-3))(inp)
    x   = layers.Dense(128, activation="relu",
                        kernel_regularizer=regularizers.l2(1e-3))(x)
    z   = layers.Dense(latent_dim, activation="linear", name="latent_vector")(x)
    x   = layers.Dense(128, activation="relu")(z)
    x   = layers.Dense(256, activation="relu")(x)
    out = layers.Dense(input_dim, activation="linear")(x)
    encoder = models.Model(inp, z,   name="MLP_encoder")
    ae      = models.Model(inp, out, name="MLP_AE")
    return ae, encoder


def train_ae(ae, Xtr, Xva_normal, epochs=200, bs=256):
    n_out = ae.output_shape[-1]           # 条件通道不作为重构目标
    ae.compile(optimizer=Adam(1e-3), loss="mse")
    ae.fit(
        Xtr, Xtr[..., :n_out],
        validation_data=(Xva_normal, Xva_normal[..., :n_out]),
        epochs=epochs, batch_size=bs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
        ],
        verbose=0,
    )
    return ae


def ae_frame_spe(ae, X_n):
    """Per-frame squared prediction error: returns (N, T)."""
    Xhat = ae.predict(X_n, verbose=0)
    return np.sum((X_n[..., :Xhat.shape[-1]] - Xhat) ** 2, axis=2)


# ── B4: MLP-AE (phi features, per-frame SPE via reshape) ─────────────────────
def run_mlp_ae(X_train_n, X_val_normal_n, X_val_n, meta_val, meta_train):
    print("\n── B4: MLP-AE (raw features, per-frame SPE) ──")
    T, F     = X_train_n.shape[1], X_train_n.shape[2]
    Xtr      = X_train_n.reshape(-1, T * F)
    Xva_norm = X_val_normal_n.reshape(-1, T * F)
    Xva      = X_val_n.reshape(-1, T * F)
    ae, _    = build_mlp_ae(T * F)
    ae.compile(optimizer=Adam(1e-3), loss="mse")
    ae.fit(
        Xtr, Xtr,
        validation_data=(Xva_norm, Xva_norm),
        epochs=200, batch_size=256,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
        ],
        verbose=0,
    )
    err_tr = Xtr - ae.predict(Xtr, verbose=0)
    err_va = Xva - ae.predict(Xva, verbose=0)
    spe_tr = (err_tr.reshape(-1, T, F) ** 2).sum(axis=2)
    spe_va = (err_va.reshape(-1, T, F) ** 2).sum(axis=2)
    ucl    = global_ucl_frame_level(meta_train, spe_tr)
    print(f"  UCL: {ucl:.6f}")
    df_frame = frames_from_frame_spe(meta_val, spe_va, ucl)
    return df_frame, eventize(df_frame.drop(columns=["ucl_frame"]))


# ── B5: Utility-LSTM-AE + Global UCL ─────────────────────────────────────────
def run_lstm_ae(X_train_n, X_val_normal_n, X_val_n, meta_val, n_feat, meta_train):
    print("\n── B5: LSTM-AE (raw features) + Global UCL ──")
    ae, _ = build_rnn_ae(WINDOW, n_feat, rnn_type="LSTM")
    ae    = train_ae(ae, X_train_n, X_val_normal_n)
    ucl    = global_ucl_frame_level(meta_train, ae_frame_spe(ae, X_train_n))
    print(f"  UCL: {ucl:.6f}")
    spe_va   = ae_frame_spe(ae, X_val_n)
    df_frame = frames_from_frame_spe(meta_val, spe_va, ucl)
    return df_frame, eventize(df_frame.drop(columns=["ucl_frame"]))


# ── GMM helpers ───────────────────────────────────────────────────────────────
def fit_gmm_bic(Z_n, k_max=20):   # v16b：上限 10→20，与 New.py 相同
    best_gmm, best_bic = None, np.inf
    min_ok = max(int(0.02 * len(Z_n)), 50)
    for k in range(1, k_max + 1):
        g = GaussianMixture(n_components=k, covariance_type="full",
                            random_state=SEED, reg_covar=1e-3)
        g.fit(Z_n)
        if not g.converged_:
            continue
        if np.bincount(g.predict(Z_n), minlength=k).min() < min_ok:
            continue
        bic = g.bic(Z_n)
        if bic < best_bic:
            best_bic = bic
            best_gmm = g
    if best_gmm is None:
        best_gmm = GaussianMixture(1, random_state=SEED, reg_covar=1e-3).fit(Z_n)
    print(f"  GMM k={best_gmm.n_components}, BIC={best_bic:.1f}")
    return best_gmm


def cluster_ucl_from_train(ae, encoder, X_train_n, meta_train, q=UCL_Q,
                            trim_q=None, global_cap_q=0.999, cap_mult=None):
    """
    GMM input = encoder latent vectors (matches New2.py online_scores_frame_level).
    UCL per cluster capped at cap_mult * global_q(global_cap_q).
    """
    spe = ae_frame_spe(ae, X_train_n)  # (N, T)

    phi_feat = encoder.predict(X_train_n, verbose=0)   # (N, latent_dim)
    Z_mu     = phi_feat.mean(axis=0, keepdims=True)
    Z_sd     = phi_feat.std(axis=0,  keepdims=True) + 1e-8
    Z_n      = ((phi_feat - Z_mu) / Z_sd).astype(np.float64)

    gmm    = fit_gmm_bic(Z_n)
    labels = gmm.predict(Z_n)

    rows = []
    for i, m in enumerate(meta_train):
        for t in range(spe.shape[1]):
            rows.append({
                "Vehicle_ID": m["Vehicle_ID"],
                "Frame_ID":   m["start_frame"] + t,
                "spe":        spe[i, t],
                "cluster":    labels[i],
            })
    df = (pd.DataFrame(rows)
          .groupby(["Vehicle_ID", "Frame_ID"], as_index=False)
          .agg({"spe": "max",
                "cluster": lambda x: x.value_counts().index[0]}))

    global_ucl_q    = float(np.quantile(df["spe"].values, global_cap_q))
    ucl_cap         = cap_mult * global_ucl_q if cap_mult else np.inf   # v8：训练已清洗，不封顶（与 New.py 一致）
    global_fallback = float(np.quantile(df["spe"].values, q))
    print(f"  Global SPE q{global_cap_q:.3f}={global_ucl_q:.6f}  UCL_cap={ucl_cap:.6f}")

    ucl = {}
    for j in range(gmm.n_components):
        spe_j = df.loc[df["cluster"] == j, "spe"].values
        if len(spe_j) < 200:
            ucl[j] = min(global_fallback, ucl_cap)
            print(f"  Cluster {j}: n={len(spe_j)} < 200 → fallback={ucl[j]:.6f}")
            continue
        cut   = float(np.quantile(spe_j, trim_q)) if trim_q else np.inf
        trim  = spe_j[spe_j <= cut]
        if len(trim) < 50:
            trim = spe_j
        base  = float(np.quantile(trim, q))
        final = min(base, ucl_cap)
        ucl[j] = final
        print(f"  Cluster {j}: n={len(spe_j):6d}  base={base:.6f}  capped={final:.6f}")

    from New import pool_degenerate_ucl
    ucl = pool_degenerate_ucl(ucl, X_train_n, labels, global_fallback)   # P5（与 New.py 相同规则）
    if os.environ.get("UCL_FLOOR_GLOBAL", "1") == "1":   # v16b 方案 C（与 New.py 相同）：每类 UCL 不低于全局
        ucl = {j: max(u, global_fallback) for j, u in ucl.items()}
    from gmm_outside import fit_outside_thresholds
    gmm.d2_thr_ = fit_outside_thresholds(gmm, Z_n)   # 类外判断阈值（与 New.py 相同规则）
    return gmm, Z_mu, Z_sd, ucl


def gruae_gmm_detect(ae, encoder, X_val_n, meta_val, gmm, Z_mu, Z_sd, cluster_ucl):
    """
    GMM input = encoder latent vectors (matches New2.py).
    Returns df with ucl_frame → ratio filter will apply.
    """
    phi_feat = encoder.predict(X_val_n, verbose=0)
    Z_n      = ((phi_feat - Z_mu) / (Z_sd + 1e-8)).astype(np.float64)
    # 与 New.py 相同：取最可能的类；在类内用该类 UCL，类外（马氏距离超出该类 P99）用最严格 UCL
    from gmm_outside import window_ucl
    ucl_per_window, outside, _ = window_ucl(gmm, Z_n, cluster_ucl, gmm.d2_thr_)
    print(f"  类外窗口: {outside.mean():.2%}")
    spe            = ae_frame_spe(ae, X_val_n)      # (N, T)
    df_frame       = frames_from_frame_spe_gmm(meta_val, spe, ucl_per_window)
    print(f"  Frame anomaly rate: {df_frame['is_anom_frame'].mean():.4f}")
    return df_frame


# ── AblA: Utility-GRU-AE + Global UCL  (shares ae_phi with Proposed) ─────────
def run_ablation_global_ucl(ae, X_train_n, X_val_n, meta_val, meta_train):
    print("\n── AblA: Utility-GRU-AE + Global UCL ──")
    ucl    = global_ucl_frame_level(meta_train, ae_frame_spe(ae, X_train_n))
    print(f"  Global UCL: {ucl:.6f}")
    spe_va   = ae_frame_spe(ae, X_val_n)
    df_frame = frames_from_frame_spe(meta_val, spe_va, ucl)
    return df_frame, eventize(df_frame.drop(columns=["ucl_frame"]))


# ── AblC: Raw-GRU-AE + Global UCL ────────────────────────────────────────────
def run_ablation_raw_feat(X_tr_raw_n, X_va_raw_normal_n, X_va_raw_n, meta_va_raw, meta_tr_raw):
    print("\n── Raw-GRU-AE + Global UCL ──")
    ae, _ = build_rnn_ae(WINDOW, len(RAW_FEATURES), rnn_type="GRU")
    ae    = train_ae(ae, X_tr_raw_n, X_va_raw_normal_n)
    ucl    = global_ucl_frame_level(meta_tr_raw, ae_frame_spe(ae, X_tr_raw_n))
    print(f"  UCL: {ucl:.6f}")
    spe_va   = ae_frame_spe(ae, X_va_raw_n)
    df_frame = frames_from_frame_spe(meta_va_raw, spe_va, ucl)
    # eventize without ucl_frame → no ratio filter (raw SPE scale differs from phi)
    return df_frame, eventize(df_frame.drop(columns=["ucl_frame"]))


# ── Raw-GRU-AE-GMM ────────────────────────────────────────────────────────────
def run_raw_gruae_gmm(X_tr_raw_n, X_va_raw_normal_n, X_va_raw_n,
                      meta_tr_raw, meta_va_raw):
    print("\n── Raw-GRU-AE-GMM ──")
    ae, enc = build_rnn_ae(WINDOW, len(RAW_FEATURES), rnn_type="GRU")
    ae      = train_ae(ae, X_tr_raw_n, X_va_raw_normal_n)
    gmm, Z_mu, Z_sd, ucl = cluster_ucl_from_train(ae, enc, X_tr_raw_n, meta_tr_raw)
    df_frame = gruae_gmm_detect(ae, enc, X_va_raw_n, meta_va_raw, gmm, Z_mu, Z_sd, ucl)
    return df_frame, eventize(df_frame.drop(columns=["ucl_frame"]))


# ── Utility-LSTM-AE-GMM ───────────────────────────────────────────────────────
def run_utility_lstm_ae_gmm(X_train_n, X_val_normal_n, X_val_n,
                             meta_train, meta_val):
    print("\n── Utility-LSTM-AE-GMM ──")
    n_feat  = X_train_n.shape[2]
    ae, enc = build_rnn_ae(WINDOW, n_feat, rnn_type="LSTM", n_out=N_PHI)
    ae      = train_ae(ae, X_train_n, X_val_normal_n)
    gmm, Z_mu, Z_sd, ucl = cluster_ucl_from_train(ae, enc, X_train_n, meta_train)
    df_frame = gruae_gmm_detect(ae, enc, X_val_n, meta_val, gmm, Z_mu, Z_sd, ucl)
    return df_frame, eventize(df_frame)


# ── Main ──────────────────────────────────────────────────────────────────────

# ───────────── 基线并行执行 ─────────────
BASELINE_METHODS = ["B2_IForest", "B3_PCA_SPE", "B4_MLP_AE", "B5_LSTM_AE", "Raw_GRU_AE",
                    "Raw_GRU_AE_GMM", "Utility_GRU_AE", "Utility_LSTM_AE_GMM"]
# 每个主进程一个缓存目录（子进程经环境变量继承），避免同一 workdir 下并行的两个基线进程互相删除缓存
#   （v15 夜间：Utility_GRU_AE 单独进程收尾时 rmtree 了共用的 _baseline_cache，导致其余进程报 FileNotFoundError）
_CACHE_DIR = os.environ.setdefault("BASELINE_CACHE_DIR", os.path.abspath(f"_baseline_cache_{os.getpid()}"))


def _save_cache(d):
    import pickle
    os.makedirs(_CACHE_DIR, exist_ok=True)
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            np.save(os.path.join(_CACHE_DIR, k + ".npy"), v)
        else:
            with open(os.path.join(_CACHE_DIR, k + ".pkl"), "wb") as f:
                pickle.dump(v, f)


def _load_cache(k):
    import pickle
    p = os.path.join(_CACHE_DIR, k + ".npy")
    if os.path.exists(p):
        return np.load(p, mmap_mode="r")
    with open(os.path.join(_CACHE_DIR, k + ".pkl"), "rb") as f:
        return pickle.load(f)


_RESULT_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_baseline_results_cache")
_TASK_INPUTS = {
    "B2_IForest": ["X_tr_raw_n", "X_va_raw_n", "meta_va_raw"],
    "B3_PCA_SPE": ["X_tr_raw_n", "X_va_raw_n", "meta_va_raw", "meta_tr_raw"],
    "B4_MLP_AE": ["X_tr_raw_n", "X_va_raw_normal_n", "X_va_raw_n", "meta_va_raw", "meta_tr_raw"],
    "B5_LSTM_AE": ["X_tr_raw_n", "X_va_raw_normal_n", "X_va_raw_n", "meta_va_raw", "meta_tr_raw"],
    "Raw_GRU_AE": ["X_tr_raw_n", "X_va_raw_normal_n", "X_va_raw_n", "meta_va_raw", "meta_tr_raw"],
    "Raw_GRU_AE_GMM": ["X_tr_raw_n", "X_va_raw_normal_n", "X_va_raw_n", "meta_tr_raw", "meta_va_raw"],
    "Utility_GRU_AE": ["X_tr_phi_n", "X_va_phi_n", "meta_va", "meta_tr", "@gruae_model.keras"],
    "Utility_LSTM_AE_GMM": ["X_tr_phi_n", "X_va_phi_normal_n", "X_va_phi_n", "meta_tr", "meta_va"],
}
_TASK_FUNCS = {"B2_IForest": "run_iforest", "B3_PCA_SPE": "run_pca_spe", "B4_MLP_AE": "run_mlp_ae",
               "B5_LSTM_AE": "run_lstm_ae", "Raw_GRU_AE": "run_ablation_raw_feat",
               "Raw_GRU_AE_GMM": "run_raw_gruae_gmm", "Utility_GRU_AE": "run_ablation_global_ucl",
               "Utility_LSTM_AE_GMM": "run_utility_lstm_ae_gmm"}


def _result_key(name):
    import hashlib, inspect, pickle
    h = hashlib.sha1(f"{name}|{SEED}|{UCL_Q}".encode())
    for k in _TASK_INPUTS[name]:
        if k.startswith("@"):
            with open(k[1:], "rb") as f:
                h.update(f.read())
            continue
        v = _load_cache(k)
        if isinstance(v, np.ndarray):
            h.update(str(v.shape).encode()); h.update(np.ascontiguousarray(v).tobytes())
        else:
            h.update(pickle.dumps(v))
    fn = globals()[_TASK_FUNCS[name]]
    h.update(inspect.getsource(fn).encode())
    for dep in ("train_ae", "ae_frame_spe", "build_rnn_ae", "build_mlp_ae", "fit_gmm_bic",
                "cluster_ucl_from_train", "gruae_gmm_detect", "frames_from_frame_spe",
                "frames_from_frame_spe_gmm", "frames_from_window_score", "eventize"):
        if dep in globals():
            h.update(inspect.getsource(globals()[dep]).encode())
    return h.hexdigest()[:16]


def _baseline_task(name):
    """在（子）进程中运行一个基线方法，写出逐帧结果，返回 (name, 指标)"""
    # v16b：Utility_GRU_AE 载入主模型（在 GPU/Metal 上训练保存）—— 在隐藏 GPU 的进程里载入后输出与 GPU 上完全不同
    #   （同一批训练窗口 SPE q99：GPU 2.38，CPU-only 31.4），所以它不受 BASELINE_CPU 影响，始终与主模型同一设备。
    if os.environ.get("BASELINE_CPU") == "1" and name != "Utility_GRU_AE":   # 与主模型训练并行时，基线只用 CPU，把 GPU 留给主模型
        try:
            tf.config.set_visible_devices([], "GPU")
        except (RuntimeError, ValueError):
            pass
    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        pass
    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
    C = _load_cache
    base_cols = ["Vehicle_ID", "Frame_ID", "frame_spe", "is_anom_frame"]
    cols = base_cols + ["ucl_frame"]
    # v12（用户确认 2026-11-04）：结果缓存 —— 键 = 该方法的输入数组/元数据 + 方法源码 + 随机种子的哈希。
    #   输入与代码都没变（如只改了 GT 或报告规则）就直接复用上次的逐帧结果，不重训。BASELINE_CACHE=0 关闭。
    key = _result_key(name)
    hit = os.path.join(_RESULT_CACHE, f"{name}_{key}.csv")
    if os.environ.get("BASELINE_CACHE", "1") == "1" and os.path.exists(hit):
        import shutil
        shutil.copy(hit, f"df_frame_{name}.csv")
        df = pd.read_csv(hit)
        print(f"── [{name}] cache hit ({key}) ──", flush=True)
        return name, compute_metrics(df, eventize(df.drop(columns=["ucl_frame"], errors="ignore")),
                                     C("gt_events_val"), C("total_hours"))
    print(f"── [{name}] start (pid {os.getpid()}) ──", flush=True)
    if name == "B2_IForest":
        df, ev = run_iforest(C("X_tr_raw_n"), C("X_va_raw_n"), C("meta_va_raw")); cols = base_cols
    elif name == "B3_PCA_SPE":
        df, ev = run_pca_spe(C("X_tr_raw_n"), C("X_va_raw_n"), C("meta_va_raw"), C("meta_tr_raw"))
    elif name == "B4_MLP_AE":
        df, ev = run_mlp_ae(C("X_tr_raw_n"), C("X_va_raw_normal_n"), C("X_va_raw_n"), C("meta_va_raw"), C("meta_tr_raw"))
    elif name == "B5_LSTM_AE":
        df, ev = run_lstm_ae(C("X_tr_raw_n"), C("X_va_raw_normal_n"), C("X_va_raw_n"), C("meta_va_raw"),
                             len(RAW_FEATURES), C("meta_tr_raw"))
    elif name == "Raw_GRU_AE":
        df, ev = run_ablation_raw_feat(C("X_tr_raw_n"), C("X_va_raw_normal_n"), C("X_va_raw_n"), C("meta_va_raw"), C("meta_tr_raw"))
    elif name == "Raw_GRU_AE_GMM":
        df, ev = run_raw_gruae_gmm(C("X_tr_raw_n"), C("X_va_raw_normal_n"), C("X_va_raw_n"),
                                   C("meta_tr_raw"), C("meta_va_raw"))
    elif name == "Utility_GRU_AE":
        from New import zero_recurrent_bias
        ae_phi = zero_recurrent_bias(tf.keras.models.load_model("gruae_model.keras", compile=False))
        df, ev = run_ablation_global_ucl(ae_phi, C("X_tr_phi_n"), C("X_va_phi_n"), C("meta_va"), C("meta_tr"))
    elif name == "Utility_LSTM_AE_GMM":
        df, ev = run_utility_lstm_ae_gmm(C("X_tr_phi_n"), C("X_va_phi_normal_n"), C("X_va_phi_n"),
                                         C("meta_tr"), C("meta_va"))
    else:
        raise ValueError(name)
    df[[c for c in cols if c in df.columns]].to_csv(f"df_frame_{name}.csv", index=False)
    os.makedirs(_RESULT_CACHE, exist_ok=True)
    import shutil
    shutil.copy(f"df_frame_{name}.csv", hit)
    m = compute_metrics(df, ev, C("gt_events_val"), C("total_hours"))
    print(f"── [{name}] finished ──", flush=True)
    return name, m


def main():
    # 1. Load raw data & phi features
    df_full = load_real_data(DATA_PATH)   # 原始特征帧在读入 φ 表后按 φ 表的车辆×帧筛选（见下）

    df_phi_all = pd.read_csv(PHI_VAL_PATH, low_memory=False)
    for c in ["Vehicle_ID", "Frame_ID", "Global_Time"] + PHI_FEATURES:
        if c in df_phi_all.columns:
            df_phi_all[c] = pd.to_numeric(df_phi_all[c], errors="coerce")
    # v16b（2026-11-05 基线核查）：原始特征基线与本文模型用同一组帧（φ 表中的车辆×帧）。
    #   原来的经纬度框是旧版遗留，比 φ 表少 8.5% 的验证帧（如 GT V1565 整段不在框内，原始特征基线不可能检出）。
    _keys = pd.MultiIndex.from_arrays([df_phi_all["Vehicle_ID"].astype(int), df_phi_all["Frame_ID"].astype(int)])
    filtered = df_full[pd.MultiIndex.from_arrays([df_full["Vehicle_ID"].astype(int),
                                                  df_full["Frame_ID"].astype(int)]).isin(_keys)].copy()
    print(f"原始特征帧（与 φ 表对齐）: {len(filtered)} / φ 表 {len(df_phi_all)}")
    if "is_truck" not in df_phi_all.columns:
        df_phi_all["is_truck"] = 0
    df_phi_all["is_truck"] = df_phi_all["is_truck"].fillna(0)

    gt_events = load_gt_events(GT_EVENTS_PATH)
    gt_vids   = set(gt_events["Vehicle_ID"].astype(int))

    # Use the exact same val_ids as New.py (read from pred_frames_val.csv)
    # to ensure all methods are evaluated on the identical vehicle set.
    # （pred_frames_val.csv 尚未生成时——基线与主模型训练并行预跑——用 New.split_ids 得到同一划分）
    if os.path.exists("pred_frames_val.csv"):
        df_pred_val  = pd.read_csv("pred_frames_val.csv", low_memory=False, usecols=["Vehicle_ID"])
        val_ids_set  = set(df_pred_val["Vehicle_ID"].dropna().astype(int).unique())
    else:
        import New
        val_ids_set = set(int(v) for v in New.split_ids(New.load_phi_all(), gt_events)[1])
        print(f"[precompute] pred_frames_val.csv 不存在，用 New.split_ids 划分：val {len(val_ids_set)} 辆")

    cnt      = df_phi_all.groupby("Vehicle_ID").size()
    eligible = np.sort(cnt[cnt >= WINDOW].index.to_numpy())

    val_ids   = np.array([v for v in eligible if v in val_ids_set])
    train_ids = np.array([v for v in eligible if v not in val_ids_set])

    df_phi_train      = df_phi_all[df_phi_all["Vehicle_ID"].isin(train_ids)].copy()
    df_phi_val        = df_phi_all[df_phi_all["Vehicle_ID"].isin(val_ids)].copy()
    df_phi_val_normal = df_phi_val[~df_phi_val["Vehicle_ID"].isin(gt_vids)].copy()

    total_hours   = compute_total_hours(df_phi_val)
    gt_events_val = gt_events[gt_events["Vehicle_ID"].isin(val_ids)].copy()

    print(f"Train: {len(train_ids)} veh | Val: {len(val_ids)} veh | "
          f"GT events: {len(gt_events_val)} | Hours: {total_hours:.2f}")

    # 2. Phi feature windows (three splits: train / val_normal / val_full)
    X_tr_phi,        meta_tr  = fixed_sliding_windows(df_phi_train,      PHI_FEATURES)
    X_va_phi_normal, meta_vn  = fixed_sliding_windows(df_phi_val_normal, PHI_FEATURES)
    # v8："先学正常" —— 所有方法的训练/早停窗口都去掉含 3σ 离群帧的窗口（与 New.py 相同）
    O_COLS = [c for c in ["o1", "o2", "o3", "o4"] if c in df_phi_all.columns]
    out_frames = {}
    if O_COLS:
        o = df_phi_all[df_phi_all[O_COLS].max(axis=1) > 0]
        out_frames = {int(v): np.sort(g["Frame_ID"].to_numpy(int)) for v, g in o.groupby("Vehicle_ID")}
    k = clean_window_mask(meta_tr, out_frames); X_tr_phi = X_tr_phi[k]; meta_tr = [m for m, b in zip(meta_tr, k) if b]
    k2 = clean_window_mask(meta_vn, out_frames); X_va_phi_normal = X_va_phi_normal[k2]
    print(f"干净窗口（phi）: train {k.sum()}/{len(k)}, val_normal {k2.sum()}/{len(k2)}")
    X_va_phi,        meta_va  = fixed_sliding_windows(df_phi_val,        PHI_FEATURES)

    # Load normalization from New2.py so all methods use identical preprocessing
    if os.path.exists("norm_params.npz"):
        norm = np.load("norm_params.npz")
        mu, sd = norm["mu"], norm["sd"]
    else:
        mu = X_tr_phi.reshape(-1, len(PHI_FEATURES)).mean(axis=0, keepdims=True)
        sd = X_tr_phi.reshape(-1, len(PHI_FEATURES)).std(axis=0,  keepdims=True) + 1e-8
        mu[:, N_PHI:] = 0.0; sd[:, N_PHI:] = 1.0
    X_tr_phi_n        = (X_tr_phi        - mu) / sd
    X_va_phi_normal_n = (X_va_phi_normal - mu) / sd
    X_va_phi_n        = (X_va_phi        - mu) / sd

    print(f"Phi windows — train: {len(X_tr_phi_n)}  "
          f"val_normal: {len(X_va_phi_normal_n)}  val_full: {len(X_va_phi_n)}")

    # 3. Raw feature windows (AblC / AblD)
    df_raw_tr        = filtered[filtered["Vehicle_ID"].isin(train_ids)].copy()
    df_raw_va_normal = filtered[
        filtered["Vehicle_ID"].isin(val_ids) &
        ~filtered["Vehicle_ID"].isin(gt_vids)
    ].copy()
    df_raw_va = filtered[filtered["Vehicle_ID"].isin(val_ids)].copy()
    for c in RAW_FEATURES:
        for df_ in [df_raw_tr, df_raw_va_normal, df_raw_va]:
            if c in df_.columns:
                df_[c] = pd.to_numeric(df_[c], errors="coerce")

    X_tr_raw,        meta_tr_raw = fixed_sliding_windows(df_raw_tr,        RAW_FEATURES)
    X_va_raw_normal, meta_vn_raw = fixed_sliding_windows(df_raw_va_normal, RAW_FEATURES)
    k = clean_window_mask(meta_tr_raw, out_frames); X_tr_raw = X_tr_raw[k]; meta_tr_raw = [m for m, b in zip(meta_tr_raw, k) if b]
    k2 = clean_window_mask(meta_vn_raw, out_frames); X_va_raw_normal = X_va_raw_normal[k2]
    print(f"干净窗口（raw）: train {k.sum()}/{len(k)}, val_normal {k2.sum()}/{len(k2)}")
    X_va_raw,        meta_va_raw = fixed_sliding_windows(df_raw_va,        RAW_FEATURES)

    mu_r = X_tr_raw.reshape(-1, len(RAW_FEATURES)).mean(axis=0, keepdims=True)
    sd_r = X_tr_raw.reshape(-1, len(RAW_FEATURES)).std(axis=0,  keepdims=True) + 1e-8
    X_tr_raw_n        = (X_tr_raw        - mu_r) / sd_r
    X_va_raw_normal_n = (X_va_raw_normal - mu_r) / sd_r
    X_va_raw_n        = (X_va_raw        - mu_r) / sd_r

    # 4. Load pre-trained GRU-AE from New2.py (gruae_model.keras)
    #    AblA uses this model with global UCL.
    #    Proposed loads pre-computed frame scores from pred_frames_val.csv (same model, GMM UCL).
    #    Both methods therefore use identical model weights — only UCL strategy differs.
    # （Utility_GRU_AE 在子进程中自行加载 gruae_model.keras）

    # 5. Run all methods
    results  = {}
    base_cols = ["Vehicle_ID", "Frame_ID", "frame_spe", "is_anom_frame"]

    df_b1, ev_b1 = run_rule_baseline(filtered, val_ids)
    results["Ref_GT_Rule (circular)"] = compute_metrics(df_b1, ev_b1, gt_events_val, total_hours)

    # B2~B5 + 消融：各方法互不依赖 → 多进程并行（BASELINE_WORKERS，默认 4；=1 时顺序执行）。
    # 输入数组存成 .npy，子进程以只读 mmap 打开，避免每个进程复制一份数据。
    cache = {"X_tr_raw_n": X_tr_raw_n, "X_va_raw_normal_n": X_va_raw_normal_n, "X_va_raw_n": X_va_raw_n,
             "X_tr_phi_n": X_tr_phi_n, "X_va_phi_normal_n": X_va_phi_normal_n, "X_va_phi_n": X_va_phi_n,
             "meta_va_raw": meta_va_raw, "meta_tr_raw": meta_tr_raw, "meta_tr": meta_tr, "meta_va": meta_va,
             "gt_events_val": gt_events_val, "total_hours": total_hours}
    _save_cache(cache)
    workers = int(os.environ.get("BASELINE_WORKERS", "4"))
    # BASELINE_ONLY=A,B 只重跑这几个方法；其余方法从已有 df_frame_<name>.csv 重新计算指标
    _only = os.environ.get("BASELINE_ONLY")
    todo = [n for n in BASELINE_METHODS if not _only or n in _only.split(",")]

    def _metrics_from_saved(name):
        d = pd.read_csv(f"df_frame_{name}.csv")
        print(f"  [reuse] {name}: df_frame_{name}.csv", flush=True)
        return compute_metrics(d, eventize(d.drop(columns=["ucl_frame"], errors="ignore")),
                               gt_events_val, total_hours)
    out = {}
    if workers > 1 and len(todo) > 1:
        try:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
                for name, m in ex.map(_baseline_task, todo):
                    out[name] = m
                    print(f"  [done] {name}: recall={m['event_recall']} precision={m['event_precision']}", flush=True)
        except Exception as e:   # 并行失败则退回顺序执行
            print(f"并行执行失败（{e!r}），改为顺序执行", flush=True)
            out = {}
    for name in BASELINE_METHODS:
        if name not in out:
            if name in todo:
                out[name] = _baseline_task(name)[1]
            elif os.environ.get("BASELINES_PRECOMPUTE") != "1":
                out[name] = _metrics_from_saved(name)
    import shutil
    shutil.rmtree(_CACHE_DIR, ignore_errors=True)
    if os.environ.get("BASELINES_PRECOMPUTE") == "1":   # 只预跑基线、写出 df_frame_<name>.csv，汇总留给 baselines 阶段
        print("[precompute] done:", todo, flush=True)
        return

    # 报告规则（report_rule.py）：所有方法统一——检测事件内至少一帧某维度 ≥3σ 才报告。
    # 同时输出不加规则的结果（experiment_results_norule.csv），看模型本身。
    import report_rule as rr
    of = rr.load_oflags("phi_all.csv")
    frames = {"Ref_GT_Rule (circular)": (df_b1, ev_b1)}
    # 对照：纯 3σ 规则（任一维度 ≥3σ 的帧即超限，连续 ≥1 s 为事件，不用任何模型）
    _oc = [c for c in rr.O_COLS if c in df_phi_val.columns]
    df_r3 = df_phi_val[["Vehicle_ID", "Frame_ID"]].copy()
    df_r3["is_anom_frame"] = (rr.evidence_cols(df_phi_val).max(axis=1) > 0).astype(int).to_numpy()
    df_r3["frame_spe"] = df_phi_val[[c for c in PHI_FEATURES[:N_PHI]]].max(axis=1).to_numpy()
    frames["Rule_3sigma"] = (df_r3, eventize(df_r3))
    for name in BASELINE_METHODS:
        d = pd.read_csv(f"df_frame_{name}.csv")
        frames[name] = (d, eventize(d.drop(columns=["ucl_frame"], errors="ignore")))
    print("\n── Proposed: Utility-GRU-AE-GMM (loading pred_frames_val.csv) ──")
    df_prop = pd.read_csv("pred_frames_val.csv", low_memory=False)
    for c in ["Vehicle_ID", "Frame_ID", "frame_spe", "ucl_frame", "is_anom_frame"]:
        if c in df_prop.columns:
            df_prop[c] = pd.to_numeric(df_prop[c], errors="coerce")
    df_prop = df_prop[df_prop["Vehicle_ID"].isin(val_ids)].copy()
    frames["Utility_GRU_AE_GMM"] = (df_prop, eventize(df_prop))

    # GT 口径（report_rule.load_gt）：主口径去掉"跟车同步急加减速"；原口径（全部 GT）另存 *_origGT.csv
    gt_main, _ = rr.load_gt(GT_EVENTS_PATH)
    gt_main_val = gt_events_val.merge(gt_main[["Vehicle_ID", "start_frame", "end_frame"]],
                                      on=["Vehicle_ID", "start_frame", "end_frame"])
    print(f"GT 主口径（去掉跟车同步急加减速）: {len(gt_main_val)} / 原口径 {len(gt_events_val)}")
    gt_p = rr.load_gt_phys(GT_EVENTS_PATH)
    gt_phys_val = (gt_events_val.merge(gt_p[["Vehicle_ID", "start_frame", "end_frame"]],
                                       on=["Vehicle_ID", "start_frame", "end_frame"]) if gt_p is not None else None)
    if gt_phys_val is not None:
        print(f"GT 物理锚点口径: {len(gt_phys_val)}")
    results_norule, results_rule, results_orig, results_phys = {}, {}, {}, {}
    for name, (d, ev) in frames.items():
        results_norule[name] = compute_metrics(d, ev, gt_main_val, total_hours)
        rep, cand = rr.split_events(ev, of)
        m = compute_metrics(rr.mask_frames(d, rep), rep, gt_main_val, total_hours)
        m["n_candidates"] = len(ev)
        results_rule[name] = m
        mo = compute_metrics(rr.mask_frames(d, rep), rep, gt_events_val, total_hours)
        mo["n_candidates"] = len(ev)
        results_orig[name] = mo
        if gt_phys_val is not None:
            mp = compute_metrics(rr.mask_frames(d, rep), rep, gt_phys_val, total_hours)
            mp["n_candidates"] = len(ev)
            results_phys[name] = mp
        print(f"  {name}: 候选 {len(ev)} → 报告 {len(rep)}  recall={m['event_recall']} precision={m['event_precision']}",
              flush=True)
    results = results_rule if rr.rule_on() else results_norule

    # 6. Results table
    df_res = (
        pd.DataFrame(results).T
        .reset_index()
        .rename(columns={"index": "Method"})
    )
    col_order = [
        "Method", "n_gt", "n_pred",
        "event_recall", "event_precision", "event_f1", "frame_hit_rate",
        "false_alarm_rate",
        "coverage_ge_0.3", "coverage_ge_0.5", "coverage_ge_0.8", "mean_coverage",
    ]
    col_order.insert(3, "n_candidates")
    df_res = df_res[[c for c in col_order if c in df_res.columns]]
    df_res.to_csv("experiment_results.csv", index=False)
    df_nr = pd.DataFrame(results_norule).T.reset_index().rename(columns={"index": "Method"})
    df_nr[[c for c in col_order if c in df_nr.columns]].to_csv("experiment_results_norule.csv", index=False)
    df_or = pd.DataFrame(results_orig).T.reset_index().rename(columns={"index": "Method"})
    df_or[[c for c in col_order if c in df_or.columns]].to_csv("experiment_results_origGT.csv", index=False)
    # v16b（用户确认 2026-11-05）：主表 = 两层。第一层"模型检测"（不加事件确认）所有方法同一口径，用于统计对比；
    #   第二层"确认"（单维或综合饱和）只对基于 φ 的方法给出（φ 是其自身输入）；原始特征方法第二层留空。
    PHI_METHODS = {"Rule_3sigma", "Utility_GRU_AE", "Utility_LSTM_AE_GMM", "Utility_GRU_AE_GMM"}
    # v16b（用户确认 2026-11-05）：不依赖 GT 的误报指标 —— 模型检测层事件内的超限帧中，四个 φ 全为 0（各维度都在正常范围）
    #   的帧占比。原始特征方法的超限约一半落在这类帧上（噪声型误报），φ 方法约 10%。
    _p0 = df_phi_val[["Vehicle_ID", "Frame_ID"]].assign(
        _phi0=(df_phi_val[[c for c in PHI_FEATURES[:N_PHI]]].abs().sum(axis=1) < 1e-9).to_numpy())
    def _alarm_phi0(d, ev):
        m = rr.mask_frames(d[["Vehicle_ID", "Frame_ID", "is_anom_frame"]], ev)
        m = m[m["is_anom_frame"] == 1].merge(_p0, on=["Vehicle_ID", "Frame_ID"], how="left")
        return round(float(m["_phi0"].mean()), 4) if len(m) else np.nan
    rows = []
    for name in frames:
        if name.startswith("Ref_GT_Rule"):
            continue
        m0, m1 = results_norule[name], results_rule[name]
        r = {"Method": name,
             "model_events": m0.get("n_pred"), "model_recall": m0.get("event_recall"),
             "model_precision": m0.get("event_precision"), "model_false_alarm_rate": m0.get("false_alarm_rate"),
             "alarm_frames_all_phi_zero": _alarm_phi0(*frames[name])}
        if name in PHI_METHODS:
            r.update({"confirmed_events": m1.get("n_pred"), "confirmed_recall": m1.get("event_recall"),
                      "confirmed_precision": m1.get("event_precision")})
        rows.append(r)
    df_main = pd.DataFrame(rows)
    df_main.to_csv("experiment_results_main.csv", index=False)
    if results_phys:
        df_ph = pd.DataFrame(results_phys).T.reset_index().rename(columns={"index": "Method"})
        df_ph[[c for c in col_order if c in df_ph.columns]].to_csv("experiment_results_physGT.csv", index=False)

    print("\n" + "=" * 65)
    print("EXPERIMENT RESULTS")
    print("=" * 65)
    print(df_res.to_string(index=False))
    print("\nSaved → experiment_results.csv")


if __name__ == "__main__":
    main()