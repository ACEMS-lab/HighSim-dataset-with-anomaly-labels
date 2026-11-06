"""
phi1_local.py —— φ1 的参照车流：同一帧、同一车道、纵向 ±PHI1_WIN_FT 内的其他车辆（局部车流）。

为什么不用"整条车道"的中位数：观测路段长约 8000 ft，lane 1 前方（出口附近）常有排队（4–7 mph），
上游仍在自由流（60–70 mph）。整条车道中位数被排队车辆主导，上游正常行驶的车会被算成"比车流快
60–100 ft/s"，φ1 直接饱和（旧版 1.9% 的帧 φ1=1，其中 99% 在 lane 1 且全部是"比中位数快"）。
"""
import numpy as np
import pandas as pd

PHI1_WIN_FT = 500.0   # 局部车流的纵向范围（前后各 500 ft；60 mph 下约 6 s 车程）
PHI1_MIN_NB = 2       # 局部车流至少 2 辆其他车，否则无法比较 → z = 0（稀疏自由流，约 9.5% 的帧）


def local_speed_z(df, frame_col="Frame_ID", lane_col="Lane_Num", y_col="Local_Y", v_col="Speed_smooth",
                  win=PHI1_WIN_FT, min_nb=PHI1_MIN_NB, qlane=None):
    """返回 (dev, mad, n_nb)：本车速度 − 局部中位数、局部 MAD、局部邻车数（与 df.index 对齐）。
    qlane：按哪条车道取局部车流（默认本车车道）；变道过渡期可传入原车道/目标车道。"""
    order = np.argsort(df[frame_col].to_numpy(), kind="stable")
    fr = df[frame_col].to_numpy()[order]
    ln = df[lane_col].to_numpy(float)[order]
    ql = ln if qlane is None else np.asarray(qlane, float)[order]
    y = df[y_col].to_numpy(float)[order]
    v = df[v_col].to_numpy(float)[order]
    dev = np.full(len(df), np.nan); mad = np.full(len(df), np.nan); nnb = np.zeros(len(df), int)
    bounds = np.flatnonzero(np.diff(fr)) + 1
    for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(fr)]):
        n = b - a
        if n < 2:
            continue
        m = (ql[a:b, None] == ln[None, a:b]) & (np.abs(y[a:b, None] - y[None, a:b]) <= win)
        np.fill_diagonal(m, False)
        cnt = m.sum(1)
        V = np.where(m, v[None, a:b], np.nan)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(V, axis=1)
                md = np.nanmedian(np.abs(V - med[:, None]), axis=1)
        ok = cnt >= min_nb
        dev[a:b] = np.where(ok, v[a:b] - med, np.nan)
        mad[a:b] = np.where(ok, md, np.nan)
        nnb[a:b] = cnt
    out_dev = np.empty(len(df)); out_mad = np.empty(len(df)); out_n = np.empty(len(df), int)
    out_dev[order] = dev; out_mad[order] = mad; out_n[order] = nnb
    return (pd.Series(out_dev, index=df.index), pd.Series(out_mad, index=df.index),
            pd.Series(out_n, index=df.index))
