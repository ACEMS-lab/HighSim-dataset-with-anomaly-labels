"""HighSim 高速公路异常驾驶检测主模型。

流程：φ1–φ4 效用特征 → GRU-AE 重构 → 潜空间 GMM 划分驾驶状态 → 分类控制限 UCL → 事件化与事件确认。
入口（由 pipeline.py 调用）：
  stage_features()  原始帧 → phi_all.csv
  stage_train()     训练 GRU-AE，保存 gruae_model.keras 与 norm_params.npz
  stage_detect()    GMM + UCL + 在线打分 + 事件化 + 评估
"""

import os, sys, random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("Using Python from:", sys.executable)

DATA_PATH            = "HighSim_frames_with_Anomaly30.csv"
GT_EVENTS_PATH       = "HighSim_events_Anomaly30_merge2s.csv"   # 完整 GT（加速度 + TTC），用于划分与主评估
GT_ACCEL_EVENTS_PATH = "HighSim_events_accel_only.csv"          # 仅加速度 GT，参考评估

# ===== 1) 数据读取 =====
def load_real_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=0, low_memory=False)
    numeric_cols = [
        "Vehicle_ID","Global_Time","Frame_ID",
        "Local_X","Local_Y","Global_X","Global_Y",
        "Width","Length","Class","Speed","Acceleration",
        "Lane_Num","Preceding_Vehicle_ID","Space_Headway","Anomaly",
        "Speed_smooth","Acc_from_vsmooth_ftps2","Acc_from_vsmooth_g",
        "HardBrake","HardAccel","TailgateTTC_sev","Anomaly_30"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values(["Vehicle_ID","Global_Time","Frame_ID"]).reset_index(drop=True)
    return df

def extract_utility_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算 φ1–φ4、各维等价 σ (z1–z4)、3σ 离群标记 (o1–o4) 与证据质量标记，具体公式见 utility_phi.py。

    必须传入包含所有同帧车辆的完整 DataFrame，否则同帧车流统计量（速度中位数、车道中心、邻车）会有偏差。
    """
    df = df.copy().sort_values(["Vehicle_ID", "Frame_ID"])
    df = df[df["Lane_Num"].notna()].copy()   # 保留匝道（Lane_Num=-1）

    import utility_phi as up
    trusted = pd.Series(True, index=df.index)
    for qc in ["Edge", "BadTrack", "BadVehicle"]:
        if qc in df.columns:
            trusted &= df[qc].fillna(0).eq(0)
    truck = df["is_truck"].fillna(0).astype(int) if "is_truck" in df.columns \
        else pd.Series(0, index=df.index)

    df["phi1"], info1 = up.compute_phi1(df, trusted, truck)
    w_follow = info1.pop("_parts")["w"]
    z1 = info1.pop("_z")

    # φ2 扣除跟随前车的正常加减速响应
    df["phi2"], z2, info2 = up.compute_phi2(df, trusted, truck, w_follow)
    z2_ctx = info2.pop("_z_ctx")
    az = pd.Series(z2, index=df.index)
    print(f"φ2: z={up.Z_SAT:g} → 1；φ2>0 {(df.loc[trusted, 'phi2'] > 0).mean():.2%}，"
          f"φ2=1 {(df.loc[trusted, 'phi2'] >= 1).mean():.3%}")

    df["phi3"], info3 = up.compute_phi3(df, trusted, truck)
    info3.pop("_parts", None)
    out3 = info3.pop("_out"); z3 = info3.pop("_z")
    df["phi4"], (out4, z4), info4 = up.compute_phi4(df, trusted)
    # 各维等价 σ（z1–z4），供训练集清洗（CLEAN_Z）与事件核查
    df["z1"] = np.asarray(z1, float).astype(np.float32)
    df["z2"] = np.abs(np.nan_to_num(az.to_numpy(float), nan=0.0)).astype(np.float32)
    df["z3"] = np.asarray(z3, float).astype(np.float32)
    df["z4"] = np.asarray(z4, float).astype(np.float32)

    # 各维 3σ 离群标记（拉依达准则）
    zthr = float(up.phi_from_z(up.Z_OUT))
    df["o1"] = (df["phi1"] >= zthr).astype(np.int8)
    df["o2"] = (df["phi2"] >= zthr).astype(np.int8)
    # o2ctx：只扣情境/前车、不加物理门槛的 φ2 3σ 标记，仅供 GT 判定（report_rule.gt_sync_follow），
    # 使 GT 不随 φ2 物理门槛变化；不进模型、不作证据。
    df["o2ctx"] = (np.asarray(z2_ctx, float) >= up.Z_OUT).astype(np.int8)
    df["o3"] = np.asarray(out3, bool).astype(np.int8)
    df["o4"] = np.asarray(out4, bool).astype(np.int8)
    oc = df.loc[trusted, ["o1", "o2", "o3", "o4"]]
    print(f"3σ 离群帧占比: {oc.mean().round(4).to_dict()}，任一维 {oc.max(axis=1).mean():.2%}")

    import json
    with open("phi_zmax.json", "w") as f:   # 标度参数，供核查脚本把 φ 换算回 z
        json.dump({"phi1": up.Z_SAT, "phi2": up.Z_SAT, "z_sat": up.Z_SAT, **info1, **info2, **info3, **info4}, f)

    keep = ["Vehicle_ID", "Frame_ID", "Global_Time", "phi1", "phi2", "phi3", "phi4", "is_truck",
            "o1", "o2", "o3", "o4", "z1", "z2", "z3", "z4", "o2ctx"]
    # 证据质量标记，供事件确认与核查（不影响 φ 与训练清洗）：
    #   dq    = 1：振铃修复区（Data.py RepairZone），该帧的 3σ 不作为证据
    #   cutin = 1：前车切入后 1.5 s 内，该帧的近距离（o4）证据记为他车引起
    df["dq"] = df["RepairZone"].fillna(0).astype(np.int8) if "RepairZone" in df else np.int8(0)
    df["cutin"] = df["CutIn"].fillna(0).astype(np.int8) if "CutIn" in df else np.int8(0)
    keep += ["dq", "cutin"]
    return df[[c for c in keep if c in df.columns]].copy()


# ===== 3) 滑窗 =====
def fixed_sliding_windows(
    df: pd.DataFrame,
    feature_cols,
    window_size=90,
    step_size=30,
    time_col="Global_Time",
    require_continuous=True
):
    """按车辆切固定长度滑窗；跳过帧号不连续或含 NaN 的窗口。返回 X (N, window_size, F) 与每窗口的元信息。"""
    X, meta = [], []

    for vid, vehicle_df in df.dropna(subset=["Vehicle_ID"]).groupby("Vehicle_ID"):
        vehicle_df = vehicle_df.sort_values("Frame_ID")
        features   = vehicle_df[feature_cols].to_numpy(dtype=float)
        frame_ids  = vehicle_df["Frame_ID"].to_numpy(dtype=int)
        has_time   = time_col in vehicle_df.columns
        times      = vehicle_df[time_col].to_numpy(dtype=float) if has_time else None

        n = len(features)
        if n < window_size:
            continue

        t = 0
        while t + window_size <= n:
            window = features[t:t + window_size]

            if require_continuous:
                frames = frame_ids[t:t + window_size]
                if not np.all(np.diff(frames) == 1):
                    t += step_size
                    continue

            if np.isnan(window).any():
                t += step_size
                continue

            sf = int(frame_ids[t])
            ef = int(frame_ids[t + window_size - 1])

            X.append(window)
            meta.append({
                "Vehicle_ID":  int(vid),
                "start_pos":   int(t),
                "end_pos":     int(t + window_size),
                "start_frame": sf,
                "end_frame":   ef,
                "start_time":  float(times[t])                  if has_time else np.nan,
                "end_time":    float(times[t + window_size - 1]) if has_time else np.nan,
            })
            t += step_size

    X = np.array(X) if len(X) > 0 else np.empty((0, window_size, len(feature_cols)))
    return X, meta

# ===== 4) GRU-AE =====
def build_gruae(time_steps, num_features, latent_dim=8, dropout=0.3,
                use_ln=True, l2_enc=1e-3, l2_dec=1e-3, num_out=None):
    """两层 GRU 编码器 + 全连接解码器。num_out 为被重构的通道数（条件通道只输入不重构）。"""
    num_out = num_features if num_out is None else num_out
    inp = layers.Input(shape=(time_steps, num_features), name="X")

    x = layers.GRU(64, return_sequences=True,  dropout=dropout,
                   kernel_regularizer=regularizers.l2(l2_enc))(inp)
    if use_ln: x = layers.LayerNormalization()(x)

    x = layers.GRU(32, return_sequences=False, dropout=dropout,
                   kernel_regularizer=regularizers.l2(l2_enc))(x)
    if use_ln: x = layers.LayerNormalization()(x)

    z = layers.Dense(latent_dim, activation="linear",
                     kernel_regularizer=regularizers.l2(l2_enc),
                     name="latent_vector")(x)

    h   = layers.Dense(time_steps * num_out, activation="linear",
                       kernel_regularizer=regularizers.l2(l2_dec),
                       name="fc_decoder")(z)
    out = layers.Reshape((time_steps, num_out), name="X_hat")(h)

    encoder     = models.Model(inp, z,   name="encoder")
    autoencoder = models.Model(inp, out, name="autoencoder_fcdec")
    return autoencoder, encoder

# ===== 5) 潜变量方差记录 =====
class LatentVarianceLogger(Callback):
    """每个 epoch 记录训练样本潜变量的总方差及各维最小/最大方差，用于监测潜空间塌缩。"""
    def __init__(self, encoder, X_train, sample_size=2000, seed=42):
        super().__init__()
        self.encoder    = encoder
        self.X_train    = X_train
        self.sample_size = min(sample_size, len(X_train))
        if self.sample_size > 0:
            rng = np.random.default_rng(seed)
            self.sample_idx = rng.choice(len(X_train), size=self.sample_size, replace=False)
        else:
            self.sample_idx = None

        self.total_variances   = []
        self.min_dim_variances = []
        self.max_dim_variances = []

    def on_epoch_end(self, epoch, logs=None):
        if self.sample_idx is None:
            return
        Xs = self.X_train[self.sample_idx]
        Z  = self.encoder(Xs, training=False).numpy()
        var_dims  = np.var(Z, axis=0)
        total_var = float(np.sum(var_dims))
        min_var   = float(np.min(var_dims))
        max_var   = float(np.max(var_dims))

        self.total_variances.append(total_var)
        self.min_dim_variances.append(min_var)
        self.max_dim_variances.append(max_var)

        print(f"[Epoch {epoch}] TotalVar={total_var:.6f}, "
              f"MinDimVar={min_var:.6f}, MaxDimVar={max_var:.6f}")

def frame_spe_by_cluster(meta, spe_2d, labels):
    """窗口逐帧 SPE (N,T) + 窗口聚类 → 逐帧结果：spe 取覆盖窗口的最大值，cluster 取覆盖窗口的众数（并列取最先出现的窗口）。"""
    N, T = spe_2d.shape
    vid = np.repeat(np.array([m["Vehicle_ID"] for m in meta]), T)
    fid = (np.array([m["start_frame"] for m in meta])[:, None] + np.arange(T)[None, :]).ravel()
    cl = np.repeat(np.asarray(labels), T)
    d = pd.DataFrame({"Vehicle_ID": vid, "Frame_ID": fid, "spe": np.asarray(spe_2d, float).ravel(),
                      "cluster": cl, "_o": np.arange(N * T)})
    spe = d.groupby(["Vehicle_ID", "Frame_ID"], as_index=False)["spe"].max()
    c = d.groupby(["Vehicle_ID", "Frame_ID", "cluster"], as_index=False).agg(n=("_o", "size"), o=("_o", "min"))
    c = c.sort_values(["Vehicle_ID", "Frame_ID", "n", "o"], ascending=[True, True, False, True])
    c = c.drop_duplicates(["Vehicle_ID", "Frame_ID"])[["Vehicle_ID", "Frame_ID", "cluster"]]
    return spe.merge(c, on=["Vehicle_ID", "Frame_ID"], how="left")


def _frame_cluster_mode(meta, labels, merge_map):
    """每帧取覆盖它的窗口聚类（经 merge_map 映射后）的众数，并列取最先出现的窗口。"""
    lab = np.array([merge_map.get(int(k), -1) for k in labels])
    st = np.array([m["start_frame"] for m in meta]); en = np.array([m["end_frame"] for m in meta])
    vd = np.array([m["Vehicle_ID"] for m in meta])
    keep = lab >= 0
    st, en, vd, lab = st[keep], en[keep], vd[keep], lab[keep]
    ln = en - st + 1
    off = np.repeat(np.cumsum(ln) - ln, ln)
    fid = np.repeat(st, ln) + (np.arange(ln.sum()) - off)
    d = pd.DataFrame({"Vehicle_ID": np.repeat(vd, ln), "Frame_ID": fid, "cluster": np.repeat(lab, ln),
                      "_o": np.arange(ln.sum())})
    c = d.groupby(["Vehicle_ID", "Frame_ID", "cluster"], as_index=False).agg(n=("_o", "size"), o=("_o", "min"))
    c = c.sort_values(["Vehicle_ID", "Frame_ID", "n", "o"], ascending=[True, True, False, True])
    return c.drop_duplicates(["Vehicle_ID", "Frame_ID"])[["Vehicle_ID", "Frame_ID", "cluster"]].reset_index(drop=True)


# ===== 6) GMM 选择 =====
def fit_gmm_with_bic(
    Z, k_min=1, k_max=15, covariance_type="full",
    random_state=42, min_cluster_frac=0.02, min_cluster_abs=50,
    reg_covar=1e-3, verbose=True
):
    """在 k_min..k_max 中按 BIC 选 GMM 分量数；剔除未收敛、最小类过小（< 2% 或 < min_cluster_abs）或协方差病态的 k。"""
    if Z is None or len(Z) == 0:
        raise ValueError("Input Z is empty.")

    n      = len(Z)
    min_ok = max(int(min_cluster_frac * n), min_cluster_abs)
    best_gmm, best_k, lowest_bic = None, None, np.inf

    # 各 k 相互独立（random_state 固定），并行拟合，结果与顺序执行相同
    def _fit_one(k):
        msgs = []
        gmm = GaussianMixture(n_components=k, covariance_type=covariance_type,
                              random_state=random_state, reg_covar=reg_covar)
        gmm.fit(Z)
        if not gmm.converged_:
            return k, None, None, [f"k={k} skip (not converged)"]
        labels = gmm.predict(Z)
        counts = np.bincount(labels, minlength=k)
        if counts.min() < min_ok:
            return k, None, None, [f"k={k} skip (min_cluster={counts.min()} < {min_ok})"]
        if covariance_type == "full":
            for i, cov in enumerate(gmm.covariances_):
                eigvals     = np.linalg.eigvalsh(cov)
                min_eig     = eigvals.min()
                cond_number = eigvals.max() / (min_eig + 1e-12)
                if min_eig < 1e-8 or cond_number > 1e8:
                    return k, None, None, [f"k={k} skip (cluster {i} unstable: "
                                           f"min_eig={min_eig:.2e}, cond={cond_number:.2e})"]
        bic  = gmm.bic(Z)
        aic  = gmm.aic(Z)
        probs = gmm.predict_proba(Z)
        msgs.append(f"k={k}, BIC={bic:.1f}, AIC={aic:.1f}, min_cluster={counts.min()}, "
                    f"mean_max_post={np.mean(probs.max(axis=1)):.3f}")
        return k, gmm, bic, msgs

    from joblib import Parallel, delayed
    n_jobs = int(os.environ.get("GMM_JOBS", "5"))
    res = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_fit_one)(k) for k in range(k_min, k_max + 1))
    for k, gmm, bic, msgs in sorted(res, key=lambda r: r[0]):
        if verbose:
            for m in msgs:
                print(m)
        if gmm is not None and bic < lowest_bic:
            lowest_bic = bic
            best_gmm   = gmm
            best_k     = k

    if best_gmm is None:
        raise RuntimeError("All k were rejected. Try relaxing cluster constraints.")

    if verbose:
        print(f"\n------------------------------------\n"
              f"Best k = {best_k}\nLowest BIC = {lowest_bic:.1f}\n"
              f"------------------------------------\n")
    return best_gmm, best_k

# ===== 7) 分类控制限 UCL =====
def pool_degenerate_ucl(cluster_ucl, X_train_n, labels, pooled_ucl, verbose=True):
    """退化类改用全局 UCL（pooled_ucl）。

    退化类指一半以上训练窗口输入全为 0（φ≡0，归一化后每维等于最小值）的类：其 SPE 只反映重构噪声，
    q 分位数接近 0，极小的 SPE 也会超限。环境变量 POOL_DEGENERATE=0 关闭。
    """
    if os.environ.get("POOL_DEGENERATE", "1") != "1":
        return cluster_ucl
    X = np.asarray(X_train_n)
    xmin = X.reshape(-1, X.shape[-1]).min(axis=0)
    zero_w = np.all(np.isclose(X, xmin[None, None, :], atol=1e-6), axis=(1, 2))
    labels = np.asarray(labels)
    out = dict(cluster_ucl)
    for j in out:
        sel = labels == j
        share = float(zero_w[sel].mean()) if sel.any() else 0.0
        if share > 0.5:
            if verbose:
                print(f"  退化类 C{j}（{share:.1%} 训练窗口 φ≡0）：UCL {out[j]:.6f} → 合并 {pooled_ucl:.6f}")
            out[j] = pooled_ucl
    return out


def compute_cluster_ucl_frame_level(
    X_train_n, meta_train, autoencoder, encoder, gmm,
    Z_mu, Z_sd, q=0.99, trim_q=None,
    global_cap_q=0.999, cap_mult=None,
    Xhat_train=None, Z_train_n=None, floor_q=None
):
    """分类帧级 UCL：取各聚类训练帧 SPE 的 q 分位数（默认 99%）。

    训练集已清洗为正常窗口，默认不截尾、不封顶、不设下限（trim_q / cap_mult / floor_q 为 None）。
    样本 < 200 帧的类用全局 q 分位数；退化类见 pool_degenerate_ucl。
    """
    if Xhat_train is None:
        Xhat_train = autoencoder.predict(X_train_n, verbose=0)
    frame_spe = spe_per_frame(X_train_n, Xhat_train)
    if Z_train_n is None:
        phi_feat  = encoder.predict(X_train_n, verbose=0)
        Z_train_n = ((phi_feat - Z_mu) / Z_sd).astype(np.float64)
    labels = gmm.predict(Z_train_n)

    df_train_frame = frame_spe_by_cluster(meta_train, frame_spe, labels)

    df_train_frame.to_csv("train_frame_spe.csv", index=False)   # 供控制限敏感性分析 tools/ucl_sweep.py
    global_ucl_q      = float(np.quantile(df_train_frame["spe"], global_cap_q))
    ucl_cap           = cap_mult * global_ucl_q if cap_mult else np.inf
    global_fallback   = float(np.quantile(df_train_frame["spe"], q))
    ucl_floor         = float(np.quantile(df_train_frame["spe"], floor_q)) if floor_q else 0.0
    print(f"Global SPE q{global_cap_q:.3f} = {global_ucl_q:.6f}, UCL cap = {ucl_cap:.6f}, "
          f"UCL floor (q{floor_q}) = {ucl_floor:.6f}")

    cluster_ucl = {}
    for j in range(gmm.n_components):
        spe_j = df_train_frame.loc[df_train_frame["cluster"] == j, "spe"].values
        if len(spe_j) < 200:
            print(f"Cluster {j} too small, fallback to global q={q}")
            cluster_ucl[j] = max(min(global_fallback, ucl_cap), ucl_floor)
            continue
        cut      = np.quantile(spe_j, trim_q) if trim_q else np.inf
        spe_trim = spe_j[spe_j <= cut]
        if len(spe_trim) < 50:
            spe_trim = spe_j
        base_ucl  = float(np.quantile(spe_trim, q))
        final_ucl = max(min(base_ucl, ucl_cap), ucl_floor)
        cluster_ucl[j] = final_ucl
        print(f"Cluster {j}: frame_samples={len(spe_j)}, "
              f"base_ucl={base_ucl:.6f}, capped_ucl={final_ucl:.6f}")

    cluster_ucl = pool_degenerate_ucl(cluster_ucl, X_train_n, labels, global_fallback)
    # 每类 UCL 不低于全局 q 分位数：GMM 只为本来波动大的驾驶状态放宽控制限，任何一类（含取各类最小值的类外窗口）
    # 都不比全局严，报出的帧是全局 UCL 报出帧的子集；否则低波动类的 UCL 过严，会报出大量各维都在 3σ 内的事件。
    # 环境变量 UCL_FLOOR_GLOBAL=0 关闭。
    if os.environ.get("UCL_FLOOR_GLOBAL", "1") == "1":
        for j in cluster_ucl:
            if cluster_ucl[j] < global_fallback:
                print(f"  UCL 下限（全局）：C{j} UCL {cluster_ucl[j]:.6f} → 全局 {global_fallback:.6f}")
                cluster_ucl[j] = global_fallback
    print("Cluster-specific frame UCL (final):")
    for k, v in cluster_ucl.items():
        print(f"  Cluster {k}: UCL={v:.6f}")
    return cluster_ucl

# ===== 8) 在线窗口打分 =====
def pick_max_ratio(df):
    """每帧只保留 SPE/UCL 比值最大的一行，保证 frame_spe 与 ucl_frame 来自同一窗口，并标记 is_anom_frame。"""
    r = df["frame_spe"].to_numpy() / np.maximum(df["ucl_frame"].to_numpy(), 1e-12)
    df = (df.assign(_r=r).sort_values("_r", ascending=False, kind="stable")
            .drop_duplicates(["Vehicle_ID", "Frame_ID"])
            .sort_values(["Vehicle_ID", "Frame_ID"]).drop(columns="_r").reset_index(drop=True))
    df["is_anom_frame"] = (df["frame_spe"] > df["ucl_frame"]).astype(int)
    return df


def aggregate_window_frames(vid, start, spe_2d, ucl_w):
    """窗口级逐帧 SPE (N,T) + 每窗口 UCL (N,) → 逐帧结果：每帧取比值 SPE/UCL 最大的覆盖窗口。"""
    N, T = spe_2d.shape
    df = pd.DataFrame({
        "Vehicle_ID": np.repeat(np.asarray(vid, dtype=int), T),
        "Frame_ID":   (np.asarray(start, dtype=int)[:, None] + np.arange(T)[None, :]).ravel(),
        "frame_spe":  np.asarray(spe_2d, dtype=float).ravel(),
        "ucl_frame":  np.repeat(np.asarray(ucl_w, dtype=float), T),
    })
    return pick_max_ratio(df)


def online_scores_frame_level(
    meta_list, X_windows_n, autoencoder, encoder,
    gmm, cluster_ucl, Z_mu, Z_sd, d2_thr=None, dump_path=None
):
    """窗口打分：编码 → GMM 归类定 UCL → 逐帧 SPE → 每帧取 SPE/UCL 最大的窗口。dump_path 给定时保存窗口级明细。"""
    phi_feat = encoder.predict(X_windows_n, verbose=0)
    Z_n = ((phi_feat - Z_mu) / Z_sd).astype(np.float64)
    # 类内窗口用该类 UCL，类外窗口（马氏距离超 d2_thr）用各类最小 UCL，见 gmm_outside.py
    from gmm_outside import window_ucl
    if d2_thr is None:
        d2_thr = np.full(gmm.n_components, np.inf)
    ucl_win, outside, lab_win = window_ucl(gmm, Z_n, cluster_ucl, d2_thr)

    Xhat      = autoencoder.predict(X_windows_n, verbose=0)
    frame_spe = spe_per_frame(X_windows_n, Xhat)     # (N, T)
    if dump_path:   # 供控制限敏感性分析 tools/ucl_sweep.py
        np.savez_compressed(dump_path, vid=np.array([m["Vehicle_ID"] for m in meta_list]),
                            start=np.array([m["start_frame"] for m in meta_list]),
                            lab=lab_win, outside=outside, spe=frame_spe.astype(np.float32))

    df_frame = aggregate_window_frames(
        np.array([m["Vehicle_ID"] for m in meta_list]),
        np.array([m["start_frame"] for m in meta_list]), frame_spe, ucl_win)

    print(f"Frame anomaly rate: {df_frame['is_anom_frame'].mean():.4f}  (类外窗口 {outside.mean():.2%})")
    return df_frame

# ===== 9) 事件化 =====
def find_runs(mask):
    """返回 0/1 序列中连续 1 段的 [start, end) 列表。"""
    runs, i, n = [], 0, len(mask)
    while i < n:
        if mask[i] == 0:
            i += 1; continue
        j = i + 1
        while j < n and mask[j] == 1:
            j += 1
        runs.append((i, j))
        i = j
    return runs

def merge_runs(runs, gap_merge):
    """合并间隔 ≤ gap_merge 帧的相邻段。"""
    if not runs:
        return []
    merged = [list(runs[0])]
    for s, e in runs[1:]:
        if s - merged[-1][1] <= gap_merge:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]

GAP_TOL_FRAMES = 15   # 超限帧之间 ≤0.5 s 的短间断视为同一段（SPE 在 UCL 附近抖动）

def eventize_from_frames(df_frame, min_dur=15, gap_merge=30,
                         fps=30, peak_threshold=None, gap_tol=None):
    """超限帧 → 事件：先合并 ≤ gap_tol 帧的短间断，保留总长 ≥ min_dur 帧的段，再合并间隔 ≤ gap_merge 帧的段。"""
    events = []
    for vid, g in df_frame.groupby("Vehicle_ID"):
        g      = g.sort_values("Frame_ID").reset_index(drop=True)
        mask   = g["is_anom_frame"].to_numpy()
        frames = g["Frame_ID"].to_numpy()

        gt_ = GAP_TOL_FRAMES if gap_tol is None else gap_tol
        runs = merge_runs(find_runs(mask), gt_)
        runs = [(s, e) for s, e in runs if (e - s) >= min_dur]
        runs = merge_runs(runs, gap_merge)

        for s, e in runs:
            seg_spe  = g["frame_spe"].iloc[s:e]
            peak_spe = float(seg_spe.max())
            mean_spe = float(seg_spe.mean())
            anom_ratio = float(g["is_anom_frame"].iloc[s:e].mean())
            if peak_threshold is not None and peak_spe < peak_threshold:
                continue
            events.append({
                "Vehicle_ID":      int(vid),
                "start_frame":     int(frames[s]),
                "end_frame":       int(frames[e - 1]),
                "duration_frames": e - s,
                "duration_s":      (e - s) / fps,
                "peak_frame_spe":  peak_spe,
                "mean_frame_spe":  mean_spe,
                "anom_frame_ratio": anom_ratio,
            })

    if not events:
        return pd.DataFrame()
    return (
        pd.DataFrame(events)
        .sort_values(["Vehicle_ID","start_frame"])
        .reset_index(drop=True)
    )

# ===== 10) 绘图 =====
def plot_training_curves(history, var_logger=None, save_prefix="fig_train"):
    plt.figure()
    plt.plot(history.history.get("loss", []), label="Train")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Val")
    plt.xlabel("Epoch"); plt.ylabel("MSE")
    plt.title("Training loss"); plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_loss.png", dpi=200)
    plt.close()

    if (var_logger is not None
            and hasattr(var_logger, "total_variances")
            and len(var_logger.total_variances) > 0):
        plt.figure()
        plt.plot(var_logger.total_variances)
        plt.xlabel("Epoch"); plt.ylabel("Variance")
        plt.title("Latent variance"); plt.tight_layout()
        plt.savefig(f"{save_prefix}_latent_variance.png", dpi=200)
        plt.close()


# ===== 11) 评估 =====
def load_gt_events(path, id_col="Vehicle_ID"):
    ev = pd.read_csv(path, low_memory=False)
    numeric_cols = [
        "Vehicle_ID","event_id","start_time","end_time","duration_s",
        "start_frame","end_frame","num_frames",
        "min_g","max_g","peak_abs_g","peak_g"
    ]
    for c in numeric_cols:
        if c in ev.columns:
            ev[c] = pd.to_numeric(ev[c], errors="coerce")
    ev = ev.dropna(subset=[id_col, "start_time", "end_time",
                           "start_frame", "end_frame"]).copy()
    ev[id_col] = ev[id_col].astype(int)
    ev = ev.sort_values([id_col, "start_time"]).reset_index(drop=True)
    return ev


def evaluate_alignment_frame(events_gt, df_frame, id_col="Vehicle_ID", fps=30):
    """逐个 GT 事件统计区间内预测超限帧数、命中与覆盖率，返回明细表与汇总。"""
    rows = []
    for _, ev in events_gt.iterrows():
        vid = int(ev[id_col])
        st  = int(ev["start_frame"])
        et  = int(ev["end_frame"])

        dfv = df_frame[
            (df_frame[id_col] == vid) &
            (df_frame["Frame_ID"] >= st) &
            (df_frame["Frame_ID"] <= et)
        ].copy()

        gt_len = max(1, et - st + 1)

        if len(dfv) == 0:
            rows.append({
                "Vehicle_ID": vid, "start_frame": st, "end_frame": et,
                "gt_len_frames": gt_len, "n_pred_anom_in_gt": 0,
                "hit": 0, "coverage_ratio": 0.0,
                "mean_spe_in_gt": np.nan, "max_spe_in_gt": np.nan,
                "mean_ucl_in_gt": np.nan,
            })
            continue

        pred_mask = (dfv["is_anom_frame"] == 1).to_numpy()
        n_hit     = int(pred_mask.sum())
        rows.append({
            "Vehicle_ID":       vid,
            "start_frame":      st,
            "end_frame":        et,
            "gt_len_frames":    gt_len,
            "n_pred_anom_in_gt": n_hit,
            "hit":              int(n_hit > 0),
            "coverage_ratio":   float(n_hit / gt_len),
            "mean_spe_in_gt":   float(dfv["frame_spe"].mean()),
            "max_spe_in_gt":    float(dfv["frame_spe"].max()),
            "mean_ucl_in_gt":   float(dfv["ucl_frame"].mean()),
        })

    df_eval = pd.DataFrame(rows)
    if len(df_eval) == 0:
        return df_eval, {"n_gt_events": 0}

    summary = {
        "n_gt_events":            int(len(df_eval)),
        "event_hit_rate":         float(df_eval["hit"].mean()),
        "mean_coverage_ratio":    float(df_eval["coverage_ratio"].mean()),
        "median_coverage_ratio":  float(df_eval["coverage_ratio"].median()),
        "coverage_ge_0.3_rate":   float((df_eval["coverage_ratio"] >= 0.3).mean()),
        "coverage_ge_0.5_rate":   float((df_eval["coverage_ratio"] >= 0.5).mean()),
        "coverage_ge_0.8_rate":   float((df_eval["coverage_ratio"] >= 0.8).mean()),
    }
    return df_eval, summary

# ===== 12) 流程各阶段 =====
L1 = 90          # 窗口长度（帧，3 s）
S1 = 30          # 窗口步长（帧，1 s）
# 同一车辆相邻事件间隔 ≤ 一个窗口长度（3 s）即合并：模型对一帧的判断来自覆盖它的 3 s 窗口，
# 间隔不足一个窗口的两段超限无法区分为两次独立行为。可用环境变量 EVENT_GAP_MERGE_S（秒）调整。
EVENT_GAP_MERGE = int(round(float(os.environ.get("EVENT_GAP_MERGE_S", L1 / 30)) * 30))
FEATURE_COLS = ["phi1", "phi2", "phi3", "phi4"]   # 被重构、计算 SPE 的特征
# 条件输入（是否货车）：只作编码器输入、不参与重构/SPE，让模型按车型学习正常，
# 又不会因货车样本少而把"是货车"本身当成异常。
COND_COLS  = ["is_truck"]
MODEL_COLS = FEATURE_COLS + COND_COLS
N_FEAT     = len(FEATURE_COLS)


def _on_metal_gpu():
    import sys as _sys
    return _sys.platform == "darwin" and len(tf.config.list_logical_devices("GPU")) > 0


def zero_recurrent_bias(model, verbose=True):
    """载入模型后调用，使 GRU 计算结果与设备无关。

    Mac GPU（tensorflow-metal）的融合 GRU 只用一组偏置，训练时两组偏置更新相同，因此 Metal 训练的模型
    b_in == b_rec，实际计算等价于循环偏置为 0 的单偏置 GRU。
    - Mac GPU 上：保持原样（置 0 会删掉唯一生效的偏置）。
    - 其他设备上：若 b_in == b_rec（Metal 训练的标志），把循环偏置置 0，与 Mac GPU 结果一致；
      非 Metal 训练的模型 b_in != b_rec，不改动。
    """
    if _on_metal_gpu():
        return model
    n = 0
    for l in model.layers:
        if isinstance(l, layers.GRU) and getattr(l, "reset_after", False):
            W, U, b = l.get_weights()
            if b.ndim == 2 and np.any(b[1] != 0) and np.allclose(b[0], b[1]):
                b = b.copy(); b[1] = 0.0; l.set_weights([W, U, b]); n += 1
    if verbose and n:
        print(f"[portable GRU] 非 Mac GPU：{n} 个 Metal 训练的 GRU 层循环偏置置 0（与 Mac GPU 计算一致）")
    return model


def spe_per_frame(X_n, Xhat):
    """逐帧 SPE：只在被重构的 φ 通道上计算（条件通道不计入）。返回 (N, T)。"""
    return np.sum((X_n[..., :Xhat.shape[-1]] - Xhat) ** 2, axis=2)
# 归一化标准差下限（φ 取值 0~1）：φ2 几乎恒为 0、标准差极小，不设下限会把普通加减速放大成数个 σ。
NORM_SD_FLOOR = 0.05


def stage_features():
    """原始帧 → phi_all.csv：φ 在全部车辆上计算，之后剔除不可信帧。"""
    df_full = load_real_data(DATA_PATH)
    print("Total vehicles (all classes):", df_full["Vehicle_ID"].nunique())

    # 先在全部帧上算 φ：不可信帧的车辆在物理上存在，仍要作为同帧邻车参与统计
    df_phi_all = extract_utility_features(df_full)

    # 剔除轨迹首尾 (Edge)、|a|>1g (BadTrack)、重建失败车辆 (BadVehicle)
    q = df_full.loc[df_phi_all.index, ["Edge", "BadTrack", "BadVehicle"]].fillna(0)
    bad = (q["Edge"] == 1) | (q["BadTrack"] == 1) | (q["BadVehicle"] == 1)
    print(f"Removing untrusted frames from phi_all: {int(bad.sum())} "
          f"({bad.mean():.2%}); BadVehicle ids: "
          f"{sorted(df_full.loc[df_full['BadVehicle'] == 1, 'Vehicle_ID'].unique().tolist())}")
    df_phi_all = df_phi_all[~bad.to_numpy()]

    df_phi_all = (
        df_phi_all
        .sort_values(["Vehicle_ID", "Frame_ID", "Global_Time"])
        .drop_duplicates(["Vehicle_ID", "Frame_ID"], keep="first")
        .reset_index(drop=True)
    )
    df_phi_all["Global_Time"] = pd.to_numeric(df_phi_all["Global_Time"], errors="coerce")
    df_phi_all.to_csv("phi_all.csv", index=False)
    print(f"Saved phi_all.csv ({len(df_phi_all)} rows)")
    print(df_phi_all[FEATURE_COLS].describe().round(4))


def load_phi_all():
    """读取 phi_all.csv 并转为数值类型；缺少 is_truck 时补 0。"""
    df_phi_all = pd.read_csv("phi_all.csv", low_memory=False)
    for c in ["Vehicle_ID", "Frame_ID", "Global_Time"] + MODEL_COLS:
        if c in df_phi_all.columns:
            df_phi_all[c] = pd.to_numeric(df_phi_all[c], errors="coerce")
    if "is_truck" not in df_phi_all.columns:
        df_phi_all["is_truck"] = 0
    df_phi_all["is_truck"] = df_phi_all["is_truck"].fillna(0)
    return df_phi_all


TRAIN_FRAC = 0.7


def split_ids(df_phi_all, gt_events):
    """train / val 车辆划分（确定性），model.py 各阶段与 baselines.py 共用。有 GT 的车辆全部进入验证集。"""
    gt_vids = set(gt_events["Vehicle_ID"].astype(int))
    cnt          = df_phi_all.groupby("Vehicle_ID").size()
    eligible_ids = np.sort(cnt[cnt >= L1].index.to_numpy())
    gt_vids_array = np.array([v for v in eligible_ids if v in gt_vids])
    normal_ids    = np.array([v for v in eligible_ids if v not in gt_vids])
    # 哈希划分：每辆车按 (SEED, Vehicle_ID) 的哈希得到 [0,1) 伪随机数，< TRAIN_FRAC 进训练集。
    # 每辆车的归属只取决于自身，GT 车辆集合变化不会让其他车重新洗牌。
    import hashlib
    u = np.array([int(hashlib.md5(f"{SEED}-{int(v)}".encode()).hexdigest()[:8], 16) / 2**32 for v in normal_ids])
    return normal_ids[u < TRAIN_FRAC], np.concatenate([normal_ids[u >= TRAIN_FRAC], gt_vids_array])


def prepare_splits_and_windows(df_phi_all, gt_events):
    """train / val 划分 + 滑窗 + 训练窗口过滤。train 与 detect 两个阶段共用，保证一致。"""
    train_ids, val_ids = split_ids(df_phi_all, gt_events)
    gt_vids = set(gt_events["Vehicle_ID"].astype(int))
    print(f"train_ids: {len(train_ids)}, val_ids: {len(val_ids)}")

    df_phi_train      = df_phi_all[df_phi_all["Vehicle_ID"].isin(train_ids)].copy()
    df_phi_val        = df_phi_all[df_phi_all["Vehicle_ID"].isin(val_ids)].copy()
    df_phi_val_normal = df_phi_val[~df_phi_val["Vehicle_ID"].isin(gt_vids)].copy()

    X_train, meta_train = fixed_sliding_windows(df_phi_train, MODEL_COLS, L1, S1)
    X_val_normal, meta_vn = fixed_sliding_windows(df_phi_val_normal, MODEL_COLS, L1, S1)
    X_val_full, meta_val = fixed_sliding_windows(df_phi_val, MODEL_COLS, L1, S1)
    for name, X in [("train", X_train), ("val_normal", X_val_normal), ("val_full", X_val_full)]:
        if X.shape[0] == 0:
            raise RuntimeError(f"No {name} windows produced.")

    # 先学正常：训练集与早停用的 val_normal 只保留干净窗口，任一帧任一维等价 σ ≥ CLEAN_Z 即剔除。
    # 否则频繁出现的异常会在 GMM 里自成一类并拉高该类 UCL。
    # 环境变量 CLEAN_Z：默认 3（拉依达 3σ），"none" 表示不清洗。
    cz = os.environ.get("CLEAN_Z", "3")
    O_COLS = ["z1", "z2", "z3", "z4"] if "z1" in df_phi_train.columns else ["o1", "o2", "o3", "o4"]
    def _clean(df_part, X, meta):
        O, meta_o = fixed_sliding_windows(df_part, O_COLS, L1, S1)
        assert len(meta_o) == len(meta)
        if cz.lower() == "none":
            keep = np.ones(len(meta), bool)
        elif O_COLS[0] == "z1":
            O = (O >= float(cz)).astype(float)
            keep = O.max(axis=(1, 2)) == 0
        else:
            keep = O.max(axis=(1, 2)) == 0
        by_dim = (O.max(axis=1) > 0).mean(axis=0)
        return X[keep], [m for m, k in zip(meta, keep) if k], keep, by_dim
    X_train, meta_train, keep, by_dim = _clean(df_phi_train, X_train, meta_train)
    print(f"[CLEAN_Z={cz}] 干净训练窗口: {keep.sum()} / {len(keep)}（剔除 {1 - keep.mean():.1%}；"
          f"含 3σ 离群帧的窗口占比 φ1–φ4: {np.round(by_dim, 3).tolist()}）")
    X_val_normal, _, keep_v, _ = _clean(df_phi_val_normal, X_val_normal, meta_vn)
    print(f"干净 val_normal 窗口: {keep_v.sum()} / {len(keep_v)}")
    print("Train windows:", X_train.shape[0], " Val normal:", X_val_normal.shape[0],
          " Val full:", X_val_full.shape[0])

    return dict(gt_vids=gt_vids, train_ids=train_ids, val_ids=val_ids,
                df_phi_train=df_phi_train, df_phi_val=df_phi_val,
                X_train=X_train, meta_train=meta_train,
                X_val_normal=X_val_normal, X_val_full=X_val_full, meta_val=meta_val)


def stage_train(epochs=200):
    """训练 GRU-AE（早停监测干净 val_normal），保存模型、归一化参数与训练曲线。"""
    gt_events  = load_gt_events(GT_EVENTS_PATH)
    df_phi_all = load_phi_all()
    d = prepare_splits_and_windows(df_phi_all, gt_events)
    d["df_phi_val"].to_csv("phi_val.csv", index=False)

    train_flat = d["X_train"].reshape(-1, len(MODEL_COLS))
    mu = train_flat.mean(axis=0, keepdims=True)
    sd = np.maximum(train_flat.std(axis=0, keepdims=True), NORM_SD_FLOOR)
    mu[:, N_FEAT:] = 0.0; sd[:, N_FEAT:] = 1.0     # 条件通道（0/1）不标准化
    print("norm mu:", mu.round(4), " sd (floored):", sd.round(4))
    np.savez("norm_params.npz", mu=mu, sd=sd)
    X_train_n      = (d["X_train"]      - mu) / sd
    X_val_normal_n = (d["X_val_normal"] - mu) / sd

    latent_dim = int(os.environ.get("LATENT_DIM", "16"))   # 潜变量维度，环境变量可调以做敏感性分析
    ae, encoder = build_gruae(L1, len(MODEL_COLS), latent_dim=latent_dim, dropout=0.1, num_out=N_FEAT)
    # 环境变量 WARM_START=<已有 gruae_model.keras>：结构相同则从其权重开始训练（仅调参用，正式结果从头训练）
    ws = os.environ.get("WARM_START")
    if ws:
        prev = tf.keras.models.load_model(ws, compile=False)
        try:
            ae.set_weights(prev.get_weights())
            print(f"热启动：从 {ws} 载入权重")
        except ValueError as e:
            print(f"热启动失败（结构不同：{e}），从头训练")
    ae.compile(optimizer=Adam(1e-3), loss=tf.keras.losses.MeanSquaredError(),
               metrics=[tf.keras.metrics.MeanSquaredError(name="mse")])   # 不含 L2 正则的纯 MSE，用于训练曲线
    var_logger = LatentVarianceLogger(encoder, X_train_n)
    history = ae.fit(
        X_train_n, X_train_n[..., :N_FEAT],
        validation_data=(X_val_normal_n, X_val_normal_n[..., :N_FEAT]),
        epochs=epochs, batch_size=256,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                             restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                                 patience=4, min_lr=1e-5),
            var_logger,
        ],
        verbose=2,
    )
    # 逐 epoch 训练记录：训练/验证 MSE、潜变量总方差
    pd.DataFrame({"epoch": np.arange(1, len(history.history["loss"]) + 1),
                  "train_mse": history.history["mse"], "val_mse": history.history["val_mse"],
                  "train_loss": history.history["loss"], "val_loss": history.history["val_loss"],
                  "latent_total_var": var_logger.total_variances[:len(history.history["loss"])],
                  "latent_dim": latent_dim}).to_csv("train_history.csv", index=False)
    ae.save("gruae_model.keras")   # 按训练设备原样保存；载入时由 zero_recurrent_bias 处理设备差异
    plot_training_curves(history, var_logger, save_prefix="gruae")
    print("Saved gruae_model.keras, norm_params.npz, gruae_loss.png")


def auto_cluster_labels(X_train_raw, labels, best_k):
    """按各类 φ 均值相对全体均值的倍数命名：最突出维度超过 1.5 倍则按该维命名，否则为巡航/跟驰。"""
    names = {0: "Speed Deviation", 1: "Harsh Accel", 2: "Lane Changing", 3: "Close Interaction"}
    wm = X_train_raw.mean(axis=1)
    gmean = wm.mean(axis=0) + 1e-9
    group = {}
    for k in range(best_k):
        sel = labels == k
        ratio = wm[sel].mean(axis=0) / gmean if sel.any() else np.zeros(4)
        j = int(np.argmax(ratio))
        group[k] = names[j] if ratio[j] > 1.5 else "Cruising/Following"
        print(f"  cluster {k}: n={int(sel.sum())}, phi_mean/global={np.round(ratio, 2)} -> {group[k]}")
    return group


def stage_detect():
    """载入模型 → 潜空间 GMM → 分类 UCL → 验证集与全部车辆在线打分 → 事件化与事件确认 → GT 对齐评估与诊断输出。"""
    gt_events = load_gt_events(GT_EVENTS_PATH)
    gt_events_accel = load_gt_events(GT_ACCEL_EVENTS_PATH)
    df_phi_all = load_phi_all()
    d = prepare_splits_and_windows(df_phi_all, gt_events)
    gt_vids, val_ids = d["gt_vids"], d["val_ids"]
    df_phi_train, df_phi_val = d["df_phi_train"], d["df_phi_val"]
    X_train, meta_train, meta_val = d["X_train"], d["meta_train"], d["meta_val"]
    feature_cols = FEATURE_COLS

    gt_events_val       = gt_events[gt_events["Vehicle_ID"].isin(val_ids)].copy()
    gt_events_accel_val = gt_events_accel[gt_events_accel["Vehicle_ID"].isin(val_ids)].copy()
    print(f"GT events full (val): {len(gt_events_val)} | accel-only (val): {len(gt_events_accel_val)}")

    norm = np.load("norm_params.npz")
    mu, sd = norm["mu"], norm["sd"]
    X_train_n    = (X_train         - mu) / sd
    X_val_full_n = (d["X_val_full"] - mu) / sd

    ae = zero_recurrent_bias(tf.keras.models.load_model("gruae_model.keras"))
    encoder = models.Model(ae.input, ae.get_layer("latent_vector").output, name="encoder")
    print("Model loaded.")

    # ── 潜空间 GMM ──
    phi_feat = encoder.predict(X_train_n, verbose=0)
    Z_mu = phi_feat.mean(axis=0, keepdims=True)
    Z_sd = phi_feat.std(axis=0, keepdims=True) + 1e-8
    Z_train_n = ((phi_feat - Z_mu) / Z_sd).astype(np.float64)
    Xhat_train = ae.predict(X_train_n, verbose=0)

    gmm, best_k = fit_gmm_with_bic(
        Z_train_n, k_min=1, k_max=20, random_state=SEED, reg_covar=1e-3   # 上限放宽，k 实际由最小类 ≥2% 的约束决定
    )
    print("Best phi-feature clusters:", best_k)


    labels_train_k = gmm.predict(Z_train_n)
    GROUP_MAP = auto_cluster_labels(X_train[..., :N_FEAT], labels_train_k, best_k)
    CLUSTER_MERGE_MAP = {i: i for i in range(best_k)}
    MERGED_NAMES = {k: f"C{k} {GROUP_MAP[k]}" for k in range(best_k)}
    n_merged = best_k
    _palette = ["#3070B0", "#4DAC26", "#E08030", "#9B59B6", "#E84040"]
    GROUP_COLORS = {g: _palette[i % len(_palette)]
                    for i, g in enumerate(sorted(set(GROUP_MAP.values())))}

    # ── 分类 UCL ──
    cluster_ucl = compute_cluster_ucl_frame_level(
        X_train_n, meta_train, autoencoder=ae, encoder=encoder,
        gmm=gmm, Z_mu=Z_mu, Z_sd=Z_sd, q=0.99,
        Xhat_train=Xhat_train, Z_train_n=Z_train_n
    )

    from gmm_outside import fit_outside_thresholds, OUTSIDE_Q
    d2_thr = fit_outside_thresholds(gmm, Z_train_n)
    print(f"GMM 类外阈值（各类训练窗口马氏距离² 的 P{OUTSIDE_Q*100:g}）: " + ", ".join(f"C{k}={t:.1f}" for k, t in enumerate(d2_thr)))

    labels_train = gmm.predict(Z_train_n)
    frame_spe_2d = spe_per_frame(X_train_n, Xhat_train)

    df_spe = frame_spe_by_cluster(meta_train, frame_spe_2d, labels_train)

    # 逐帧聚类标签，仅用于箱线图与 φ 均值诊断
    df_spe["merged"]  = df_spe["cluster"].map(CLUSTER_MERGE_MAP)
    df_spe_valid      = df_spe[df_spe["merged"] >= 0].copy()

    # ── 各类 SPE 箱线图 ──
    colors = ["#3070B0", "#4DAC26", "#E08030", "#9B59B6",
              "#E84040", "#40C0C0", "#C070C0", "#80A040",
              "#F0A000", "#6060D0"]
    fig, ax = plt.subplots(figsize=(max(8, n_merged * 1.5), 4))
    data_by_merged = []
    for new_k in range(n_merged):
        spe_k = df_spe_valid.loc[df_spe_valid["merged"] == new_k, "spe"].values
        if len(spe_k) == 0:
            data_by_merged.append(np.array([0.0]))
            print(f"{MERGED_NAMES[new_k]}: empty cluster, skipped")
            continue
        spe_k = spe_k[spe_k <= np.quantile(spe_k, 0.95)]
        data_by_merged.append(spe_k)
        print(f"{MERGED_NAMES[new_k]}: median={np.median(spe_k):.4f}, "
              f"mean={np.mean(spe_k):.4f}, UCL={cluster_ucl[new_k]:.4f}")

    bp = ax.boxplot(data_by_merged, patch_artist=True, notch=False,
                    medianprops={"color": "black", "lw": 1.5},
                    whiskerprops={"lw": 1.2}, capprops={"lw": 1.2})
    for new_k, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[new_k % len(colors)])
        patch.set_alpha(0.7)
    for new_k in range(n_merged):
        ax.hlines(cluster_ucl[new_k], new_k + 0.6, new_k + 1.4,
                  colors=colors[new_k % len(colors)], linestyles="--", linewidth=1.8)
    ax.set_xticklabels([MERGED_NAMES[k] for k in range(n_merged)],
                       fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("SPE", fontsize=12)
    ax.set_title("SPE by cluster (dashed: UCL)", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig("spe_by_cluster_frame.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ── 按类名分组的箱线图（组 UCL 取组内各类 UCL 均值）──
    group_names = sorted(set(GROUP_MAP.values()))
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    data_by_group, ucl_by_group = [], []
    for grp in group_names:
        cluster_ids = [k for k in range(best_k) if GROUP_MAP.get(k) == grp]
        spe_grp = np.concatenate([
            df_spe_valid.loc[df_spe_valid["cluster"].isin(cluster_ids), "spe"].values
        ])
        if len(spe_grp) > 0:
            spe_grp = spe_grp[spe_grp <= np.quantile(spe_grp, 0.95)]
        data_by_group.append(spe_grp if len(spe_grp) > 0 else np.array([0.0]))
        ucl_by_group.append(np.mean([cluster_ucl[k] for k in cluster_ids]))

    bp2 = ax2.boxplot(data_by_group, patch_artist=True, notch=False,
                      medianprops={"color": "black", "lw": 1.5},
                      whiskerprops={"lw": 1.2}, capprops={"lw": 1.2})
    for gi, grp in enumerate(group_names):
        bp2["boxes"][gi].set_facecolor(GROUP_COLORS[grp])
        bp2["boxes"][gi].set_alpha(0.75)
        ax2.hlines(ucl_by_group[gi], gi + 0.6, gi + 1.4,
                   colors=GROUP_COLORS[grp], linestyles="--", linewidth=2.0,
                   label=f"UCL {ucl_by_group[gi]:.2f}")
    ax2.set_xticklabels(group_names, fontsize=11)
    ax2.set_ylabel("SPE", fontsize=12)
    ax2.set_title("SPE by group", fontsize=11)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig("spe_by_group_frame.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved spe_by_group_frame.png (4-group visualization)")

    # ── 训练集逐帧聚类 ──
    df_cluster_train = _frame_cluster_mode(meta_train, labels_train, CLUSTER_MERGE_MAP)
    df_phi_train_clustered = df_phi_train.merge(
        df_cluster_train, on=["Vehicle_ID", "Frame_ID"], how="left"
    )
    df_phi_train_clustered.to_csv("phi_train_with_cluster.csv", index=False)
    print(f"Saved phi_train_with_cluster.csv  ({len(df_phi_train_clustered)} rows)")

    for new_k in range(n_merged):
        df_j = df_phi_train_clustered[df_phi_train_clustered["cluster"] == new_k]
        print(f"\n{MERGED_NAMES[new_k]} mean phi values:\n",
              df_j[feature_cols].mean(numeric_only=True))

    # ── 验证集在线打分 ──
    df_frame_val = online_scores_frame_level(
        meta_val, X_val_full_n,
        autoencoder=ae, encoder=encoder,
        gmm=gmm, cluster_ucl=cluster_ucl,
        Z_mu=Z_mu, Z_sd=Z_sd, d2_thr=d2_thr, dump_path="val_windows.npz"
    )
    df_frame_val.to_csv("pred_frames_val.csv", index=False)

    # ── 验证集窗口与逐帧聚类 ──
    phi_feat_val = encoder.predict(X_val_full_n, verbose=0)
    Z_val_n      = ((phi_feat_val - Z_mu) / Z_sd).astype(np.float64)
    labels_val   = gmm.predict(Z_val_n)
    probs_val    = gmm.predict_proba(Z_val_n)

    cluster_win_rows = []
    for w_idx, meta in enumerate(meta_val):
        merged_k = CLUSTER_MERGE_MAP.get(int(labels_val[w_idx]), -1)
        row = {
            "Vehicle_ID":  meta["Vehicle_ID"],
            "start_frame": meta["start_frame"],
            "end_frame":   meta["end_frame"],
            "cluster":     merged_k,
        }
        for new_k in range(n_merged):
            orig_ks = [k for k, v in CLUSTER_MERGE_MAP.items() if v == new_k]
            row[f"prob_cluster_{new_k}"] = float(
                sum(probs_val[w_idx, k] for k in orig_ks)
            )
        cluster_win_rows.append(row)

    df_window_cluster = pd.DataFrame(cluster_win_rows)
    df_window_cluster.to_csv("phi_val_window_cluster.csv", index=False)
    print(f"Saved phi_val_window_cluster.csv  ({len(df_window_cluster)} windows)")

    df_fc_val = _frame_cluster_mode(meta_val, labels_val, CLUSTER_MERGE_MAP)
    df_phi_val_with_cluster = df_phi_val.merge(
        df_fc_val, on=["Vehicle_ID", "Frame_ID"], how="left"
    )
    df_phi_val_with_cluster.to_csv("phi_val_with_cluster.csv", index=False)
    print(f"Saved phi_val_with_cluster.csv  "
          f"({len(df_phi_val_with_cluster)} rows, "
          f"cluster dist: "
          f"{df_phi_val_with_cluster['cluster'].value_counts().to_dict()})")

    # ── 验证集事件化 ──
    # 事件 = SPE > UCL 的帧连续 ≥ 1 s（min_dur=30），间隔 ≤ EVENT_GAP_MERGE 帧合并。
    MEAN_RATIO_THRESH = None   # 事件平均 SPE/UCL 下限；None = 不筛选（SPE > UCL 即异常）
    pred_events = eventize_from_frames(df_frame_val, min_dur=30, gap_merge=EVENT_GAP_MERGE, fps=30)
    if not pred_events.empty:
        event_ucl = []
        for _, ev in pred_events.iterrows():
            mask = (
                (df_frame_val["Vehicle_ID"] == ev["Vehicle_ID"]) &
                (df_frame_val["Frame_ID"] >= ev["start_frame"]) &
                (df_frame_val["Frame_ID"] <= ev["end_frame"])
            )
            event_ucl.append(df_frame_val.loc[mask, "ucl_frame"].mean())
        pred_events["event_ucl"]      = event_ucl
        pred_events["spe_ratio"]      = (pred_events["peak_frame_spe"]
                                         / pred_events["event_ucl"])
        pred_events["mean_spe_ratio"] = (pred_events["mean_frame_spe"]
                                         / pred_events["event_ucl"])
        if MEAN_RATIO_THRESH is not None:
            pred_events = pred_events[
                pred_events["mean_spe_ratio"] >= MEAN_RATIO_THRESH
            ].reset_index(drop=True)

    import report_rule as rr
    _of = rr.load_oflags("phi_all.csv") if rr.rule_on() else None
    # 两层输出：模型检测的全部事件 → pred_events_val_model.csv（confirmed 列标明是否确认，统计对比用）；
    # 通过事件确认的 → pred_events_val.csv（案例展示用）。
    if not pred_events.empty:
        _pm = pred_events.copy()
        _pm["confirmed"] = rr.has_evidence(_pm, _of).astype(int) if _of is not None else 1
        _pm.to_csv("pred_events_val_model.csv", index=False)
    if _of is not None and not pred_events.empty:
        _n0 = len(pred_events)
        pred_events, _cand = rr.split_events(pred_events, _of)
        _cand.to_csv("pred_events_val_candidates.csv", index=False)
        print(f"事件确认（单维或综合饱和）: 模型检测 {_n0} → 确认 {len(pred_events)}，"
              f"未确认 {len(_cand)}（pred_events_val_candidates.csv）")
    pred_events.to_csv("pred_events_val.csv", index=False)
    print(f"Predicted events val (mean_spe_ratio>={MEAN_RATIO_THRESH}): "
          f"{len(pred_events)}")

    # ── 训练集在线打分（分批，避免内存溢出）──
    print("\nRunning detection on train vehicles...")
    SCORE_BATCH = 4000
    train_chunks = []
    for _start in range(0, len(meta_train), SCORE_BATCH):
        _end = min(_start + SCORE_BATCH, len(meta_train))
        print(f"  batch {_start}–{_end} / {len(meta_train)}", end="\r", flush=True)
        _chunk = online_scores_frame_level(
            meta_train[_start:_end], X_train_n[_start:_end],
            autoencoder=ae, encoder=encoder,
            gmm=gmm, cluster_ucl=cluster_ucl,
            Z_mu=Z_mu, Z_sd=Z_sd, d2_thr=d2_thr
        )
        train_chunks.append(_chunk)
    print()
    df_frame_train = pd.concat(train_chunks, ignore_index=True)

    # ── 合并训练集与验证集 ──
    df_frame_all = pick_max_ratio(
        pd.concat([df_frame_train, df_frame_val], ignore_index=True)[
            ["Vehicle_ID", "Frame_ID", "frame_spe", "ucl_frame"]])
    df_frame_all.to_csv("pred_frames_all.csv", index=False)
    print(f"Saved pred_frames_all.csv  ({len(df_frame_all)} rows, "
          f"{df_frame_all['Vehicle_ID'].nunique()} vehicles)")

    # ── 全部车辆事件化 ──
    pred_events_all = eventize_from_frames(
        df_frame_all, min_dur=30, gap_merge=EVENT_GAP_MERGE, fps=30)
    if not pred_events_all.empty:
        event_ucl_all = []
        for _, ev in pred_events_all.iterrows():
            mask = (
                (df_frame_all["Vehicle_ID"] == ev["Vehicle_ID"]) &
                (df_frame_all["Frame_ID"]   >= ev["start_frame"]) &
                (df_frame_all["Frame_ID"]   <= ev["end_frame"])
            )
            event_ucl_all.append(df_frame_all.loc[mask, "ucl_frame"].mean())
        pred_events_all["event_ucl"]      = event_ucl_all
        pred_events_all["spe_ratio"]      = (pred_events_all["peak_frame_spe"]
                                             / pred_events_all["event_ucl"])
        pred_events_all["mean_spe_ratio"] = (pred_events_all["mean_frame_spe"]
                                             / pred_events_all["event_ucl"])
        if MEAN_RATIO_THRESH is not None:
            pred_events_all = pred_events_all[
                pred_events_all["mean_spe_ratio"] >= MEAN_RATIO_THRESH
            ].reset_index(drop=True)

    if _of is not None and not pred_events_all.empty:
        pred_events_all, _cand_all = rr.split_events(pred_events_all, _of)
        _cand_all.to_csv("pred_events_all_candidates.csv", index=False)
    pred_events_all.to_csv("pred_events_all.csv", index=False)
    print(f"Predicted events all (mean_spe_ratio>={MEAN_RATIO_THRESH}): "
          f"{len(pred_events_all)}  "
          f"({pred_events_all['Vehicle_ID'].nunique()} unique vehicles)")

    # ── 评估 ──
    # 主评估用完整 GT（急加减速 + TTC<1.5 s）；仅加速度 GT 事件很少，只作参考
    print("\n=== 评估：full GT (accel + TTC<1.5s) ===")
    ev_eval, summary = evaluate_alignment_frame(gt_events_val, df_frame_val)
    print("Alignment summary (full GT):", summary)
    ev_eval.to_csv("align_eval_frame.csv", index=False)
    ev_eval_acc, summary_acc = evaluate_alignment_frame(gt_events_accel_val, df_frame_val)
    print("Alignment summary (accel-only GT, reference):", summary_acc)
    ev_eval_acc.to_csv("align_eval_frame_accel.csv", index=False)

    print("\nPipeline finished.")

    # ── 诊断输出 ──
    print("\n详细每个 full GT 事件的覆盖情况:")
    print(ev_eval[["Vehicle_ID", "start_frame", "end_frame",
                   "gt_len_frames", "n_pred_anom_in_gt",
                   "hit", "coverage_ratio"]].to_string())

    if not pred_events.empty:
        print("\n=== 事件SPE超出倍数分布 ===")
        print(pred_events["spe_ratio"].describe())
        print("\n按spe_ratio排序的事件:")
        print(pred_events[["Vehicle_ID", "duration_s",
                            "peak_frame_spe", "event_ucl", "spe_ratio"]]
              .sort_values("spe_ratio", ascending=False).to_string())

    for _, ev in gt_events_val.iterrows():
        vid = int(ev["Vehicle_ID"])
        sf  = int(ev["start_frame"])
        ef  = int(ev["end_frame"])
        dfv = df_frame_val[
            (df_frame_val["Vehicle_ID"] == vid) &
            (df_frame_val["Frame_ID"]   >= sf)  &
            (df_frame_val["Frame_ID"]   <= ef)
        ]
        if len(dfv) == 0:
            print(f"VID={vid} [{sf}~{ef}]: 模型无此区间数据")
            continue
        print(f"VID={vid} [{sf}~{ef}]: "
              f"frames={len(dfv)}, "
              f"mean_spe={dfv['frame_spe'].mean():.4f}, "
              f"max_spe={dfv['frame_spe'].max():.4f}, "
              f"mean_ucl={dfv['ucl_frame'].mean():.4f}, "
              f"hit_frames={dfv['is_anom_frame'].sum()}")

    df_normal = df_frame_val[~df_frame_val["Vehicle_ID"].isin(gt_vids)]
    df_anomal = df_frame_val[ df_frame_val["Vehicle_ID"].isin(gt_vids)]

    print("\n=== 正常车辆SPE分布 ===")
    print(df_normal["frame_spe"].describe())
    print(f"SPE > 0.02 的帧比例: {(df_normal['frame_spe'] > 0.02).mean():.2%}")
    print(f"SPE > 0.1  的帧比例: {(df_normal['frame_spe'] > 0.1).mean():.2%}")
    print(f"SPE > 1.0  的帧比例: {(df_normal['frame_spe'] > 1.0).mean():.2%}")
    print(f"正常车辆中触发异常的车辆数: "
          f"{df_normal[df_normal['is_anom_frame']==1]['Vehicle_ID'].nunique()} / "
          f"{df_normal['Vehicle_ID'].nunique()}")

    print("\n=== GT异常车辆SPE分布 ===")
    print(df_anomal["frame_spe"].describe())

    print("\n=== UCL分布 ===")
    print(df_frame_val["ucl_frame"].describe())
    print(df_frame_val["ucl_frame"].value_counts().head(10))



def main():
    raise SystemExit("请使用统一入口: python pipeline.py --workdir runs/<name> --stage all|features|train|detect")


if __name__ == "__main__":
    main()

