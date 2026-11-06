"""
phi1_local.py —— φ1 的局部车流速度偏差。

参照车流为同一帧、同一车道、纵向 ±PHI1_WIN_FT 内的其他车辆。不用整条车道中位数，
是因为约 8000 ft 的路段上下游车速差异大（出口附近排队、上游自由流），会使 φ1 失真。
主要入口：local_speed_z。
"""
import numpy as np
import pandas as pd

PHI1_WIN_FT = 500.0   # 局部车流纵向范围，ft（前后各 500 ft，60 mph 下约 6 s 车程）
PHI1_MIN_NB = 2       # 局部邻车数下限，不足时结果为 NaN


def local_speed_z(df, frame_col="Frame_ID", lane_col="Lane_Num", y_col="Local_Y", v_col="Speed_smooth",
                  win=PHI1_WIN_FT, min_nb=PHI1_MIN_NB, qlane=None):
    """返回 (dev, mad, n_nb)：本车速度 − 局部中位数、局部 MAD、局部邻车数，均与 df.index 对齐。
    qlane：取局部车流所用的车道，默认本车车道（变道时可传入原车道或目标车道）。"""
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
