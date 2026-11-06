"""
accel_thresholds.py —— 随速度变化的加/减速度异常阈值曲线。

方法参考 周荣贵等 (2024)《基于轨迹数据的高速公路小客车异常驾驶行为》(吉林大学学报(工学版),
DOI 10.13229/j.cnki.jdxbgxb.20221504)：按 10 km/h 速度分箱，在每箱内分别对加速、减速样本求
μ、σ，阈值点 T_k = μ + kσ，再用三次多项式拟合成 T_k(v)。
补充：按车型/车道分组拟合（小汽车主线、货车、小汽车出口匝道），并可按情境等级细分；
速度超出拟合范围时截断到端点；由 T_1、T_2 反推 μ(v)、σ(v) 得连续 z 值。
主要入口：fit_curves、add_context_curves、zscore、phys_exceed。
"""
import json
import numpy as np
import pandas as pd

G_FTPS2        = 32.174          # 1 g，ft/s²
FTPS_PER_KMH   = 0.911344
BIN_KMH        = 10.0            # 速度分箱宽度，km/h
MIN_BIN_FRAMES = 2000            # 样本不足此数的速度箱不参与拟合
KS             = (1, 2, 3)
SIDES          = ("acc", "dec")
_CTX = {0: "", 1: "·Δ1-3σ", 2: "·Δ3-5σ", 3: "·Δ≥5σ"}
_BASE = {0: "car-mainline", 1: "truck", 2: "car-ramp"}
_PBASE = {0: "Car (mainline)", 1: "Truck", 2: "Car (ramp)"}
_PCTX = {0: "", 1: ", gap 1-3σ", 2: ", gap 3-5σ", 3: ", gap ≥5σ"}
CLASS_NAMES    = {k + 3 * c: _BASE[k] + _CTX[c] for c in range(4) for k in range(3)}
PLOT_NAMES     = {k + 3 * c: _PBASE[k] + _PCTX[c] for c in range(4) for k in range(3)}
GROUPS         = (0, 1, 2)       # 0 小汽车主线，1 货车，2 小汽车出口匝道
CATCH_OFFSET   = 3               # 情境等级 c 的组号 = 基础组号 + 3c
RAMP_LANE      = 0               # 出口匝道车道号
# 物理锚点（单位 g）：异常加减速还须超出正常驾驶的物理范围
#   减速 3.4 m/s²：AASHTO 停车视距设计减速度（Fambro et al. 1997, NCHRP Report 400）
#   加速 小汽车 2.2 m/s²、货车 1.5 m/s²：Long (2000), TRR 1737
G_MS2          = 9.80665
PHYS_DEC_G     = 3.4 / G_MS2
PHYS_ACC_CAR_G = 2.2 / G_MS2
PHYS_ACC_TRK_G = 1.5 / G_MS2


def phys_exceed(acc_g, is_truck):
    """是否超出物理锚点（加速按车型，减速统一）。"""
    a = np.asarray(acc_g, float); t = np.nan_to_num(np.asarray(is_truck, float)) == 1
    return (a >= np.where(t, PHYS_ACC_TRK_G, PHYS_ACC_CAR_G)) | (a <= -PHYS_DEC_G)


def group_id(is_truck, lane, ctx=None):
    """拟合组：货车 → 1；出口匝道小汽车 → 2；其余小汽车 → 0。
    给出 ctx（情境等级 0–3）时组号再加 CATCH_OFFSET·ctx。"""
    t = np.asarray(is_truck, int); ln = np.asarray(lane, float)
    g = np.where(t == 1, 1, np.where(ln == RAMP_LANE, 2, 0))
    if ctx is not None:
        g = g + CATCH_OFFSET * np.nan_to_num(np.asarray(ctx, float)).astype(int)
    return g


def context_class(deficit_sigma):
    """速度落差（以 σ 为单位）→ 情境等级：<1 → 0（稳态），1–3 → 1，3–5 → 2，≥5 → 3。"""
    d = np.nan_to_num(np.asarray(deficit_sigma, float), nan=0.0)
    return np.select([d >= 5, d >= 3, d >= 1], [3, 2, 1], 0)


def frame_ctx(acc_g, catch_cls, appr_cls):
    """加速帧取追赶等级，减速帧取逼近等级。"""
    a = np.asarray(acc_g, float)
    return np.where(a > 0, np.asarray(catch_cls, int), np.asarray(appr_cls, int))


def add_context_curves(curves, speed_ftps, acc_g, group, catch_cls, appr_cls, trusted=None):
    """按情境等级分别拟合阈值曲线，返回以 组号 + CATCH_OFFSET·等级 为键的曲线字典。
    加速侧按追赶等级（本车比参照速度慢得越多，正常加速越大），减速侧按逼近等级
    （本车比前方车流快得越多，正常减速越大）。有效速度箱不足 4 个的组沿用 curves 中的总体曲线。"""
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
    """按组拟合阈值曲线。速度 ft/s，加速度 g。
    返回 curves[组][side] = {coef_k: 三次多项式系数(高次在前), r2_k, v_lo, v_hi, bins: 分箱统计表}。"""
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
            if len(st) < 4:   # 速度箱不足以拟合三次曲线
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
    """连续 z 值 (|a| − μ(v)) / σ(v)，加速样本用加速曲线、减速样本用减速曲线。"""
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
    """打印各组拟合 R² 及超出 1σ/2σ/3σ 曲线的样本比例（原文表 1：加速 12.68/4.37/1.68%，减速 12.80/4.58/1.78%）。"""
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
