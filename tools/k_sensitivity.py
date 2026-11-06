"""
k_sensitivity.py —— GMM 聚类数 k 的敏感性分析（v16b，用户确认 2026-11-05）。在 runs/<run> 目录下运行：

    python ../../tools/k_sensitivity.py [K_MIN=2] [K_MAX=12]

同一个已训练的 GRU-AE（gruae_model.keras，循环偏置置 0），GMM 依次取 k = K_MIN..K_MAX，**不加最小类占比约束**，
其余与 New.stage_detect 完全相同：潜向量按训练集标准化、full 协方差、reg_covar=1e-3、seed 42；
各类 UCL = 训练帧 SPE 的 q99 → P5 退化类合并 → 方案 C（不低于全局 q99）；类外窗口（马氏距离超出该类 P99）用各类最小 UCL；
验证集：每帧取 SPE/UCL 比值最大的覆盖窗口 → 连续 ≥1 s 为事件 → 事件确认（report_rule，单维或综合饱和）。
输出 k_sensitivity.csv：k、BIC、最小类占比、是否满足 2%、模型检测事件数 / GT 检出、确认事件数 / GT 检出。
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))


def main(k_min=2, k_max=12):
    import tensorflow as tf
    from sklearn.mixture import GaussianMixture
    from joblib import Parallel, delayed
    import New
    import report_rule as rr
    from gmm_outside import fit_outside_thresholds, window_ucl
    import exper_fixed as ef

    gt = New.load_gt_events(New.GT_EVENTS_PATH)
    d = New.prepare_splits_and_windows(New.load_phi_all(), gt)
    norm = np.load("norm_params.npz")
    Xtr = (d["X_train"] - norm["mu"]) / norm["sd"]
    Xva = (d["X_val_full"] - norm["mu"]) / norm["sd"]
    mtr, mva = d["meta_train"], d["meta_val"]
    ae = New.zero_recurrent_bias(tf.keras.models.load_model("gruae_model.keras", compile=False))
    enc = tf.keras.Model(ae.input, ae.get_layer("latent_vector").output)
    Ztr, Zva = enc.predict(Xtr, verbose=0, batch_size=2048), enc.predict(Xva, verbose=0, batch_size=2048)
    mu, sd = Ztr.mean(0, keepdims=True), Ztr.std(0, keepdims=True) + 1e-8
    Ztr, Zva = ((Ztr - mu) / sd).astype(np.float64), ((Zva - mu) / sd).astype(np.float64)
    spe_tr = New.spe_per_frame(Xtr, ae.predict(Xtr, verbose=0, batch_size=2048))
    spe_va = New.spe_per_frame(Xva, ae.predict(Xva, verbose=0, batch_size=2048))
    gq = float(np.quantile(New.frame_spe_by_cluster(mtr, spe_tr, np.zeros(len(mtr), int))["spe"], 0.99))
    print(f"global q99 = {gq:.6f}", flush=True)

    vids = set(int(v) for v in d["val_ids"])
    gt_val = gt[gt["Vehicle_ID"].isin(vids)]
    of = rr.load_oflags("phi_all.csv")
    vid_va = np.array([m["Vehicle_ID"] for m in mva]); st_va = np.array([m["start_frame"] for m in mva])

    def hits(E):
        return sum(bool(((E["Vehicle_ID"] == v) & (E["start_frame"] <= e) & (E["end_frame"] >= s)).any())
                   for v, s, e in zip(gt_val["Vehicle_ID"], gt_val["start_frame"], gt_val["end_frame"]))

    def fit(k):
        g = GaussianMixture(k, covariance_type="full", random_state=New.SEED, reg_covar=1e-3).fit(Ztr)
        return k, g

    fits = Parallel(n_jobs=int(os.environ.get("GMM_JOBS", "5")), backend="loky")(delayed(fit)(k) for k in range(k_min, k_max + 1))
    rows = []
    for k, g in sorted(fits, key=lambda r: r[0]):
        lab = g.predict(Ztr)
        share = np.bincount(lab, minlength=k) / len(lab)
        df = New.frame_spe_by_cluster(mtr, spe_tr, lab)
        ucl = {j: float(np.quantile(df.loc[df["cluster"] == j, "spe"], 0.99)) if (df["cluster"] == j).sum() >= 200 else gq
               for j in range(k)}
        ucl = New.pool_degenerate_ucl(ucl, Xtr, lab, gq, verbose=False)
        ucl = {j: max(u, gq) for j, u in ucl.items()}            # 方案 C
        uw, outside, _ = window_ucl(g, Zva, ucl, fit_outside_thresholds(g, Ztr))
        f = New.aggregate_window_frames(vid_va, st_va, spe_va, uw)
        ev = ef.eventize(f[["Vehicle_ID", "Frame_ID", "frame_spe", "is_anom_frame"]])
        rep, _ = rr.split_events(ev, of)
        r = dict(k=k, bic=round(g.bic(Ztr), 1), min_share=round(share.min(), 4), meets_2pct=bool(share.min() >= 0.02),
                 n_clusters_above_global=int(sum(u > gq for u in ucl.values())), outside_share=round(outside.mean(), 4),
                 model_events=len(ev), model_gt=hits(ev), confirmed_events=len(rep), confirmed_gt=hits(rep), n_gt=len(gt_val))
        rows.append(r); print(r, flush=True)
    # 对照：全局 UCL（k=1）
    f = New.aggregate_window_frames(vid_va, st_va, spe_va, np.full(len(mva), gq))
    ev = ef.eventize(f[["Vehicle_ID", "Frame_ID", "frame_spe", "is_anom_frame"]]); rep, _ = rr.split_events(ev, of)
    rows.insert(0, dict(k=1, bic=np.nan, min_share=1.0, meets_2pct=True, n_clusters_above_global=0, outside_share=0.0,
                        model_events=len(ev), model_gt=hits(ev), confirmed_events=len(rep), confirmed_gt=hits(rep), n_gt=len(gt_val)))
    pd.DataFrame(rows).to_csv("k_sensitivity.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 2, int(a[1]) if len(a) > 1 else 12)
