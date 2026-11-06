"""
accel_thresholds.py —— 速度相关的加/减速度异常阈值曲线（参考 周荣贵等, 2024,
《基于轨迹数据的高速公路小客车异常驾驶行为》, 吉林大学学报(工学版), DOI 10.13229/j.cnki.jdxbgxb.20221504）

方法（与原文一致的部分）：
  1. 按速度区间分组；在每个区间内，对"加速样本(a>0)"和"减速样本(a<0, 取绝对值)"分别计算均值 μ 和标准差 σ；
  2. 阈值点 T_k = μ + kσ（k=1 一般异常，k=2 极端异常；另算 k=3 供参考）；
  3. 用三次多项式把各区间的阈值点回归成速度的连续函数 T_k(v)，使用时代入当前车速。
本实现的补充：
  - 按"组"分别拟合（原文只研究小客车）：0 = 小汽车·主线(lane 1-3)，1 = 货车（主线+匝道，匝道货车样本太少），
    2 = 小汽车·出口匝道(lane 0)。匝道上的减速/排队走停是正常操作，若与主线共用曲线会被大量误判为极端异常；
  - 速度区间宽度原文未能确认，这里默认 10 km/h（BIN_KMH），样本 < MIN_BIN_FRAMES 的区间不参与拟合；
  - 速度超出拟合区间时截断到区间端点，避免三次曲线外推发散；
  - 由 T_1(v)、T_2(v) 反推 σ(v) = T_2 − T_1、μ(v) = T_1 − σ(v)，得到连续的 z 值
    z = (|a| − μ(v)) / σ(v)，供 GT 判定和 φ2 使用。
"""
import json
import numpy as np
import pandas as pd

G_FTPS2        = 32.174
FTPS_PER_KMH   = 0.911344
BIN_KMH        = 10.0
MIN_BIN_FRAMES = 2000
KS             = (1, 2, 3)
SIDES          = ("acc", "dec")
_CTX = {0: "", 1: "·Δ1-3σ", 2: "·Δ3-5σ", 3: "·Δ≥5σ"}
_BASE = {0: "car-mainline", 1: "truck", 2: "car-ramp"}
_PBASE = {0: "Car (mainline)", 1: "Truck", 2: "Car (ramp)"}
_PCTX = {0: "", 1: ", gap 1-3σ", 2: ", gap 3-5σ", 3: ", gap ≥5σ"}
CLASS_NAMES    = {k + 3 * c: _BASE[k] + _CTX[c] for c in range(4) for k in range(3)}
PLOT_NAMES     = {k + 3 * c: _PBASE[k] + _PCTX[c] for c in range(4) for k in range(3)}
GROUPS         = (0, 1, 2)
CATCH_OFFSET   = 3
RAMP_LANE      = 0
# v14（用户确认 2026-11-05）：物理锚点 —— "异常加减速"须同时超出正常驾驶的物理范围（GT 第二口径与 φ2 共用）
#   减速：3.4 m/s²（AASHTO 停车视距设计减速度，"大多数驾驶员可舒适达到"；NCHRP Report 400 / Fambro et al. 1997）
#   加速：小汽车 2.2 m/s²（Long 2000, TRR 1737：高于平均水平驾驶员起步加速度 α）；货车 1.5 m/s²（同文，单体货车平均）
G_MS2          = 9.80665
PHYS_DEC_G     = 3.4 / G_MS2
PHYS_ACC_CAR_G = 2.2 / G_MS2
PHYS_ACC_TRK_G = 1.5 / G_MS2


def phys_exceed(acc_g, is_truck):
    """是否超出物理锚点（加速按车型，减速统一）。"""
    a = np.asarray(acc_g, float); t = np.nan_to_num(np.asarray(is_truck, float)) == 1
    return (a >= np.where(t, PHYS_ACC_TRK_G, PHYS_ACC_CAR_G)) | (a <= -PHYS_DEC_G)


def group_id(is_truck, lane, ctx=None):
    """拟合组：货车 → 1；小汽车在出口匝道(lane 0) → 2；其余小汽车 → 0。
    v12（P9 分级 + P11，用户确认 2026-11-04）：ctx = 情境等级 0–3（加速帧用追赶等级、减速帧用逼近等级，
    见 context_class）→ 组号 + CATCH_OFFSET·ctx。"""
    t = np.asarray(is_truck, int); ln = np.asarray(lane, float)
    g = np.where(t == 1, 1, np.where(ln == RAMP_LANE, 2, 0))
    if ctx is not None:
        g = g + CATCH_OFFSET * np.nan_to_num(np.asarray(ctx, float)).astype(int)
    return g


def context_class(deficit_sigma):
    """速度落差（以 1σ 为单位）→ 等级：<1σ → 0（稳态），1–3σ → 1，3–5σ → 2，≥5σ → 3。
    分界与全文统一标度相同（1σ 起算、3σ 离群、5σ 饱和）。"""
    d = np.nan_to_num(np.asarray(deficit_sigma, float), nan=0.0)
    return np.select([d >= 5, d >= 3, d >= 1], [3, 2, 1], 0)


def frame_ctx(acc_g, catch_cls, appr_cls):
    """加速帧取追赶等级，减速帧取逼近等级。"""
    a = np.asarray(acc_g, float)
    return np.where(a > 0, np.asarray(catch_cls, int), np.asarray(appr_cls, int))


def add_context_curves(curves, speed_ftps, acc_g, group, catch_cls, appr_cls, trusted=None):
    """v12（P9 分级 + P11）：情境不同，"正常加减速"的分布不同 ——
      加速侧：本车比参照速度（局部车流 / 前车）慢得越多（追赶：拥堵消散起步、换入快车道），正常加速越大；
      减速侧：本车比前方慢车 / 局部车流快得越多（逼近：远处慢车队前的提前减速），正常减速越大。
    按落差等级（context_class）分别用同一方法（周荣贵等 2024）拟合；等级 0 = 稳态。样本不足的组沿用总体曲线。
    v11 原型（只分"追赶/否"两档）对 GT 中"正常"的起步、变道提速基本无效：0.2–0.25 g 在全部追赶帧中仍 >5σ，
    因为追赶帧大多只是轻度追赶；分级后大落差等级的 5σ 线随之放宽，稳态等级收紧。"""
    base = np.asarray(group, int)
    out = {}
    for c in range(4):
        sa = fit_curves(speed_ftps, acc_g, np.where(np.asarray(catch_cls) == c, base, -1), trusted)
        sd = fit_curves(speed_ftps, acc_g, np.where(np.asarray(appr_cls) == c, base, -1), trusted)
        for k in GROUPS:
            acc = sa[k]["acc"] if len(sa[k]["acc"]["bins"]) >= 4 else curves[k]["acc"]
            dec = sd[k]["dec"] if len(sd[k]["dec"]["bins"]) >= 4 else curves[k]["dec"]
            out[k + CATCH_OFFSET * c] = {"acc": acc, "dec": dec}
    return out


def _r2(y, yhat, w):
    ybar = np.average(y, weights=w)
    ss_res = np.sum(w * (y - yhat) ** 2)
    ss_tot = np.sum(w * (y - ybar) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def fit_curves(speed_ftps, acc_g, group, trusted=None, bin_kmh=BIN_KMH,
               min_bin_frames=MIN_BIN_FRAMES):
    """返回 dict: curves[cls][side] = {coef_k: 三次多项式系数(高次在前), r2_k, v_lo, v_hi, bins: 表}"""
    v = np.asarray(speed_ftps, float); a = np.asarray(acc_g, float)
    t = np.asarray(group, int)
    ok = np.isfinite(v) & np.isfinite(a) & (v >= 0)
    if trusted is not None:
        ok &= np.asarray(trusted, bool)
    width = bin_kmh * FTPS_PER_KMH
    curves = {}
    for c in GROUPS:
        curves[c] = {}
        for side in SIDES:
            sel = ok & (t == c) & ((a > 0) if side == "acc" else (a < 0))
            mag = np.abs(a[sel]); vv = v[sel]
            b = np.floor(vv / width)
            df = pd.DataFrame({"b": b, "m": mag})
            st = df.groupby("b")["m"].agg(["mean", "std", "size"])
            st = st[st["size"] >= min_bin_frames]
            if len(st) < 4:   # 样本不足以拟合三次曲线（追赶状态的小组）
                curves[c][side] = {"bins": pd.DataFrame()}
                continue
            vc = (st.index.to_numpy(float) + 0.5) * width
            w = st["size"].to_numpy(float)
            entry = {"v_lo": float(vc.min()), "v_hi": float(vc.max()),
                     "bins": pd.DataFrame({"v_center_ftps": vc, "v_center_kmh": vc / FTPS_PER_KMH,
                                           "n": st["size"].to_numpy(), "mean": st["mean"].to_numpy(),
                                           "sd": st["std"].to_numpy()})}
            for k in KS:
                y = (st["mean"] + k * st["std"]).to_numpy()
                coef = np.polyfit(vc, y, 3, w=np.sqrt(w))
                entry[f"coef_{k}"] = coef.tolist()
                entry[f"r2_{k}"] = _r2(y, np.polyval(coef, vc), w)
                entry["bins"][f"T{k}"] = y
            curves[c][side] = entry
    return curves


def threshold(curves, cls, side, k, speed_ftps):
    e = curves[cls][side]
    v = np.clip(np.asarray(speed_ftps, float), e["v_lo"], e["v_hi"])
    return np.polyval(e[f"coef_{k}"], v)


def zscore(curves, speed_ftps, acc_g, group):
    """连续 z 值：加速样本相对加速曲线、减速样本相对减速曲线。z>=1 一般异常，z>=2 极端异常。"""
    v = np.asarray(speed_ftps, float); a = np.asarray(acc_g, float)
    t = np.asarray(group, int)
    z = np.zeros(len(a))
    for c in curves:
        for side in SIDES:
            sel = (t == c) & ((a > 0) if side == "acc" else (a < 0))
            if not sel.any() or "v_lo" not in curves[c][side]:
                continue
            t1 = threshold(curves, c, side, 1, v[sel]); t2 = threshold(curves, c, side, 2, v[sel])
            sd = np.maximum(t2 - t1, 1e-4)
            mu = t1 - sd
            z[sel] = (np.abs(a[sel]) - mu) / sd
    return z


def summary(curves, speed_ftps, acc_g, group, trusted=None):
    """打印拟合 R² 与各车型落在 1σ/2σ/3σ 曲线外侧的比例（对照原文表 1：加速 12.68/4.37/1.68%，减速 12.80/4.58/1.78%）。"""
    v = np.asarray(speed_ftps, float); a = np.asarray(acc_g, float); t = np.asarray(group, int)
    ok = np.ones(len(a), bool) if trusted is None else np.asarray(trusted, bool)
    lines = []
    for c in curves:
        for side in SIDES:
            if "v_lo" not in curves[c][side]:
                continue
            sel = ok & (t == c) & ((a > 0) if side == "acc" else (a < 0))
            mag = np.abs(a[sel])
            parts = []
            for k in KS:
                frac = float((mag > threshold(curves, c, side, k, v[sel])).mean())
                parts.append(f"{k}σ外 {frac:6.2%} (R²={curves[c][side][f'r2_{k}']:.2f})")
            lines.append(f"  {CLASS_NAMES[c]:12s} {'加速' if side == 'acc' else '减速'}: " + " | ".join(parts))
    print("速度-阈值三次曲线（周荣贵等 2024 方法）：\n" + "\n".join(lines))


def save(curves, path="accel_threshold_curves.json"):
    out = {}
    for c, d in curves.items():
        out[str(c)] = {}
        for side, e in d.items():
            ee = {k: v for k, v in e.items() if k != "bins"}
            ee["bins"] = e["bins"].to_dict(orient="list")
            out[str(c)][side] = ee
    with open(path, "w") as f:
        json.dump(out, f, indent=1)


def load(path="accel_threshold_curves.json"):
    with open(path) as f:
        raw = json.load(f)
    curves = {}
    for c, d in raw.items():
        curves[int(c)] = {}
        for side, e in d.items():
            e = dict(e); e["bins"] = pd.DataFrame(e["bins"])
            curves[int(c)][side] = e
    return curves


def plot(curves, path="accel_threshold_curves.png"):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(curves), 2, figsize=(11, 3.4 * len(curves)), sharex=True)
    for i, c in enumerate(sorted(curves)):
        for j, side in enumerate(SIDES):
            ax = axes[i, j]; e = curves[c][side]; b = e["bins"]
            if "v_lo" not in e:
                ax.set_axis_off(); continue
            vv = np.linspace(e["v_lo"], e["v_hi"], 200)
            for k, ls in zip(KS, ["-", "--", ":"]):
                ax.plot(b["v_center_kmh"], b[f"T{k}"], "o", ms=3, color=f"C{k-1}")
                ax.plot(vv / FTPS_PER_KMH, np.polyval(e[f"coef_{k}"], vv), ls, color=f"C{k-1}",
                        label=f"mean+{k}σ")
            ax.set_title(f"{PLOT_NAMES[c]}, {'accel' if side == 'acc' else 'decel'}")
            ax.set_ylabel("|a| (g)"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("Speed (km/h)")
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()
