"""
Data.py —— HighSim 轨迹预处理与 GT 事件生成。

流程：读取原始轨迹 → 速度修复与加速度（1 s 速度差分）→ 数据质量标记（Edge / BadTrack /
BadVehicle / 速度修复区）→ 基于位置的前车与 TTC → 逐帧异常标注 → 持续时间过滤 → 事件化。
GT：急减速 min(周荣贵 2024 速度-阈值曲线 z, 全局 z) ≥ GT_ACC_Z；同车道 TTC < TTC_SEV_S；均持续 ≥ GT_MIN_S。
单位：ft、ft/s、ft/s²；g = ft/s² ÷ 32.174。主要入口：main。
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# =========================
# 全局参数
# =========================
G_FTPS2           = 32.174   # 1g = 32.174 ft/s²
FPS               = 30       # 帧率

# ── GT ─────────────────────────────────────────────────────────
# 急减速：z ≥ GT_ACC_Z（原文 z ≥ 2 为极端异常，在本数据上过宽，取 5σ 作为确定异常）；
# TTC 冲突：TTC < TTC_SEV_S 且前车同车道。两者均须持续 ≥ GT_MIN_S。
GT_ACC_Z          = 5.0
GT_MIN_S          = 1.0      # s
HARD_BRAKE_G      = -0.65    # SHRP2 文献阈值，仅打印对照
HARD_ACCEL_G      =  0.58
TTC_CRIT_S        =  3.0     # 仅统计
TTC_SEV_S         =  1.5     # 严重冲突阈值，s（SSAM/FHWA）
# 重建噪声剔除：急变速段内原始加速度饱和帧占比 > NOISE_SAT_FRAC 且无邻车印证则剔除。
# 邻车印证 = 同车道 ±NB_DY_FT (ft) 内、同时段速度变化 > NB_DV_FTPS (ft/s) 的车辆。
NOISE_SAT_FRAC    =  0.5
NB_DY_FT          = 400.0
NB_DV_FTPS        = 15.0
# 数据方将每帧速度变化截断在 ±20 ft/s²；1 s 加速度达到上限说明整 1 s 被截断（真实急变速或重建伪影），
# 这类段需至少 NB_MIN_CAPPED 辆邻车印证才保留。
ACC_CAP_FTPS2     = 19.99
NB_MIN_CAPPED     = 2

# ── 平滑 ───────────────────────────────────────────────────────
# 原始 Speed 已由数据方做 2 s 滑动平均，再做 SG 平滑会压低峰值，故 SG_WINDOW = None（不再平滑）。
# 换用其他数据源时可设为奇数帧数启用 SG 平滑。
SG_WINDOW         = None
SG_POLYORDER      = 2

MIN_CONSEC_FRAMES     = int(round(GT_MIN_S * 30))   # 急变速连续帧下限
MIN_CONSEC_FRAMES_TTC = int(round(GT_MIN_S * 30))   # TTC 冲突连续帧下限
MERGE_GAP_S       = 2.0      # 事件合并间隔，s
CONTEXT_EXPAND_S  = 1.0      # 事件前后扩展，s

# ── 数据质量 ───────────────────────────────────────────────────
# 轨迹首尾各 1.5 s 标记为 Edge（2 s 滑动平均在两端各约 1 s 不完整，加 1 s 差分半窗 0.5 s），
# 这些帧不参与 GT 标注和模型训练/检测。
EDGE_TRIM_FRAMES  = 45
MAX_PLAUSIBLE_G   = 1.0      # 超过视为跟踪误差（BadTrack）
EXTRAP_MIN_FRAMES = 90       # 速度连续不变 ≥ 3 s 视为外推假轨迹
EXTRAP_MIN_SPEED  = 5.0      # ft/s，排除停车
# 原始 Acceleration 被重建算法限制在 ±20 ft/s²；饱和帧占比 > BAD_VEH_SAT_FRAC 的车辆
# 整车标记 BadVehicle（全体车辆该占比 P99 ≈ 5.8%）。
RAW_ACC_BOUND     = 20.0
BAD_VEH_SAT_FRAC  = 0.15
LEADER_LAT_TOL_FT = 6.0      # 前车判定：横向中心距 < 半车道宽（12 ft 车道）
LAT_SEP_FRAMES    = 15       # 横向分离速度的差分步长（0.5 s）
# 加速度 = 中心差分（前后各 ACC_HALF_FRAMES 帧，共 1 s）；不用原始 Acceleration 列（截断、量化、常饱和）。
ACC_HALF_FRAMES   = 15
TRUCK_CLASS       = 3        # Class：2 = 小汽车，3 = 货车

id_col    = "Vehicle_ID"
time_col  = "Global_Time"
frame_col = "Frame_ID"
speed_col = "Speed"          # 原始速度，ft/s


# =========================
# 1) 读取 & 过滤
# =========================
def load_and_filter(csv_path: str) -> pd.DataFrame:
    cols = [
        "Vehicle_ID", "Global_Time", "Frame_ID",
        "Local_X", "Local_Y", "Global_X", "Global_Y",
        "Width", "Length", "Class", "Speed", "Acceleration",
        "Lane_Num", "Preceding_Vehicle_ID", "Space_Headway"
    ]
    df = pd.read_csv(csv_path, header=None, names=cols)
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 第 7/8 列顺序在不同版本数据中不一致，按"车长 > 车宽"识别并纠正
    if df["Width"].median() > df["Length"].median():
        df = df.rename(columns={"Width": "Length", "Length": "Width"})
        print("检测到新版列顺序（第 7 列为车长），已交换 Width/Length")
    df["is_truck"] = (df["Class"] == TRUCK_CLASS).astype(int)
    vc = df.groupby(id_col)["is_truck"].first().value_counts().to_dict()
    print(f"车型：小汽车 {vc.get(0, 0)} 辆，货车 {vc.get(1, 0)} 辆；"
          f"车长中位数 {df['Length'].median():.1f} ft，车宽中位数 {df['Width'].median():.1f} ft")

    df = df.sort_values([id_col, time_col, frame_col]).reset_index(drop=True)
    print(f"读取后: {len(df)} 条记录，{df[id_col].nunique()} 辆车")
    return df


# =========================
# 2) 速度修复与加速度
# =========================
def _make_odd(n: int) -> int:
    return n if n % 2 == 1 else n - 1


def _acc_for_vehicle(g: pd.DataFrame) -> np.ndarray:
    """加速度 a(t) = [v(t+h) − v(t−h)] / [T(t+h) − T(t−h)]，h = ACC_HALF_FRAMES，v 为 Speed_smooth。
    轨迹两端退化为单侧差分（这些帧已被 Edge 剔除）。"""
    v = g["Speed_smooth"].to_numpy(dtype=float)
    t = g[time_col].to_numpy(dtype=float)
    n = len(v)
    if n < 2:
        return np.zeros(n)
    i  = np.arange(n)
    lo = np.clip(i - ACC_HALF_FRAMES, 0, n - 1)
    hi = np.clip(i + ACC_HALF_FRAMES, 0, n - 1)
    dt = t[hi] - t[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        acc = (v[hi] - v[lo]) / dt
    return np.nan_to_num(acc, nan=0.0, posinf=0.0, neginf=0.0)


# ── 速度振铃（锯齿）修复 ───────────────────────────────────────
# 速度阶跃经 ±20 ft/s² 截断后会在正负上限间来回振荡（锯齿）。振荡不全是噪声，故不删除，
# 而在振铃段内用段前、段后速度线性插值，保留净速度变化。
# 振铃 = 顶格段（|a| ≥ ACC_CAP_FTPS2 且连续 ≥ RING_MIN_RUN 帧）正负交替 ≥ RING_MIN_CHAIN 次、
# 相邻段间隔 ≤ RING_GAP 帧；修复范围向两侧各扩展 RING_PAD 帧。
RING_MIN_RUN   = 2
RING_GAP       = 15
RING_MIN_CHAIN = 3
RING_PAD       = 15


def ringing_mask(a_raw: np.ndarray) -> np.ndarray:
    """由原始加速度标记振铃段（含两侧 RING_PAD 帧）。"""
    s = np.where(np.abs(a_raw) >= ACC_CAP_FTPS2, np.sign(a_raw), 0)
    n = len(s); flag = np.zeros(n, bool); runs = []; i = 0
    while i < n:
        if s[i] == 0:
            i += 1; continue
        j = i
        while j < n and s[j] == s[i]:
            j += 1
        if j - i >= RING_MIN_RUN:
            runs.append((i, j, s[i]))
        i = j
    k = 0
    while k < len(runs):
        chain = [runs[k]]
        while (k + 1 < len(runs) and runs[k + 1][0] - chain[-1][1] <= RING_GAP
               and runs[k + 1][2] != chain[-1][2]):
            k += 1; chain.append(runs[k])
        if len(chain) >= RING_MIN_CHAIN:
            flag[max(0, chain[0][0] - RING_PAD):min(n, chain[-1][1] + RING_PAD)] = True
        k += 1
    return flag


def step_mask(a_raw: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """标记速度阶跃：不属于振铃的孤立顶格段（连续 RING_MIN_RUN–RING_GAP 帧），多为跟踪修正。
    与振铃同样插值修复；顶格超过 RING_GAP 帧视为真实急变速，不修复。"""
    s = np.where(np.abs(a_raw) >= ACC_CAP_FTPS2, np.sign(a_raw), 0)
    n = len(s); flag = np.zeros(n, bool); i = 0
    while i < n:
        if s[i] == 0:
            i += 1; continue
        j = i
        while j < n and s[j] == s[i]:
            j += 1
        if RING_MIN_RUN <= j - i <= RING_GAP and not ring[i:j].any():
            flag[max(0, i - RING_PAD):min(n, j + RING_PAD)] = True
        i = j
    return flag


def _repair_ringing(v: np.ndarray, flag: np.ndarray) -> np.ndarray:
    """flag 段内用段前、段后速度线性插值；段落在轨迹端点时取单侧值。"""
    v = v.copy(); n = len(v); i = 0
    while i < n:
        if not flag[i]:
            i += 1; continue
        j = i
        while j < n and flag[j]:
            j += 1
        a, b = i - 1, j            # 段前、段后的正常帧
        if a >= 0 and b < n:
            v[i:j] = np.interp(np.arange(i, j), [a, b], [v[a], v[b]])
        elif a >= 0:
            v[i:j] = v[a]
        elif b < n:
            v[i:j] = v[b]
        i = j
    return v


def _speed_smooth_for_vehicle(g: pd.DataFrame) -> np.ndarray:
    """修复振铃与阶跃后的速度；SG_WINDOW 非空时再做 SG 平滑。"""
    v = g[speed_col].to_numpy(dtype=float)
    if "Acceleration" in g:
        a_raw = g["Acceleration"].to_numpy(dtype=float)
        ring = ringing_mask(a_raw)
        v = _repair_ringing(v, ring | step_mask(a_raw, ring))
    if SG_WINDOW is None:
        return v
    w = _make_odd(min(SG_WINDOW, len(v)))
    if w <= SG_POLYORDER:
        return v
    return savgol_filter(v, window_length=w, polyorder=SG_POLYORDER, deriv=0, mode="interp")


def add_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    """新增 Ringing、SpeedStep、Speed_smooth、Acc_ftps2、Acc_g 列。"""
    df["Ringing"] = (
        df.groupby(id_col, group_keys=False)
          .apply(lambda g: pd.Series(ringing_mask(g["Acceleration"].to_numpy(dtype=float)), index=g.index))
    ).astype(int)
    print(f"速度振铃（锯齿）帧: {df['Ringing'].mean():.2%}，涉及车辆 {df.loc[df['Ringing'] == 1, id_col].nunique()}；"
          f"已在振铃段内对速度线性插值修复")
    df["SpeedStep"] = (
        df.groupby(id_col, group_keys=False)
          .apply(lambda g: pd.Series(step_mask(g["Acceleration"].to_numpy(dtype=float),
                                               ringing_mask(g["Acceleration"].to_numpy(dtype=float))), index=g.index))
    ).astype(int)
    print(f"速度阶跃（孤立顶格段 ±{RING_PAD / 30:.1f} s）帧: {df['SpeedStep'].mean():.2%}，"
          f"涉及车辆 {df.loc[df['SpeedStep'] == 1, id_col].nunique()}；同样线性插值修复")
    # 修复后速度，供加速度、TTC、φ 计算使用
    df["Speed_smooth"] = (
        df.groupby(id_col, group_keys=False)
          .apply(lambda g: pd.Series(_speed_smooth_for_vehicle(g), index=g.index))
    ).round(4)
    acc_ftps2 = (
        df.groupby(id_col, group_keys=False)
          .apply(lambda g: pd.Series(_acc_for_vehicle(g), index=g.index))
    )
    df["Acc_ftps2"] = acc_ftps2.round(4)
    df["Acc_g"]     = (df["Acc_ftps2"] / G_FTPS2).round(6)

    raw_g = df["Acceleration"] / G_FTPS2
    print(f"\n重算加速度 vs 原始 Acceleration 列：相关系数 {df['Acc_g'].corr(raw_g):.3f}，"
          f"原始列 |a|=20 ft/s² 饱和帧占比 {(df['Acceleration'].abs() >= RAW_ACC_BOUND - 1e-3).mean():.2%}")
    print("\n加速度分布（g），速度差/时间：")
    print(df["Acc_g"].describe())
    print("\n经验分位数（验证文献阈值是否在数据中有意义）：")
    for p in [99, 99.5, 99.9]:
        hi = df["Acc_g"].quantile(p / 100)
        lo = df["Acc_g"].quantile(1 - p / 100)
        print(f"  P{p}: {hi:+.4f}g  |  P{100-p}: {lo:+.4f}g")
    print(f"  文献阈值: HardBrake {HARD_BRAKE_G}g / HardAccel +{HARD_ACCEL_G}g")
    print(f"  触发 HardBrake 帧数 : {(df['Acc_g'] <= HARD_BRAKE_G).sum()}")
    print(f"  触发 HardAccel 帧数 : {(df['Acc_g'] >= HARD_ACCEL_G).sum()}")
    return df


# =========================
# 2b) 数据质量标记
# =========================
BOUNDARY_BIN_FT = 200.0    # 观测边界区的纵向分箱，ft
CURVE_BIN_FT = 100.0       # 车道中心曲率的纵向分箱，ft
MIN_BOUNDARY_BIN = 2000   # 每箱最少样本，与 accel_thresholds.MIN_BIN_FRAMES 相同


def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """新增数据质量列：Edge、BadTrack、Extrapolated、RepairZone、BoundaryZone、CurveZone、
    Lane_ctx（匝道组判定用车道）、SatFrac、BadVehicle。"""
    g   = df.groupby(id_col)
    pos = g.cumcount()
    n   = g[frame_col].transform("size")
    df["Edge"]     = ((pos < EDGE_TRIM_FRAMES) | (pos >= n - EDGE_TRIM_FRAMES)).astype(int)
    df["BadTrack"] = (df["Acc_g"].abs() > MAX_PLAUSIBLE_G).astype(int)
    # 外推假轨迹：速度连续 ≥ EXTRAP_MIN_FRAMES 帧完全不变且非停车，为跟踪丢失时的外推结果，计入 BadTrack
    df = df.sort_values([id_col, frame_col])
    same = (df["Speed"].diff() == 0) & (df[id_col].diff() == 0)
    same = same | same.shift(-1, fill_value=False) & (df[id_col].shift(-1) == df[id_col])
    grp = (same != same.shift()).cumsum()
    run_len = same.groupby(grp).transform("size").where(same, 0)
    df["Extrapolated"] = ((run_len >= EXTRAP_MIN_FRAMES) & (df["Speed"] > EXTRAP_MIN_SPEED)).astype(int)
    df["BadTrack"] = (df["BadTrack"] | df["Extrapolated"]).astype(int)
    # 速度修复区：修复段及其边界前后 EDGE_TRIM_FRAMES 帧。速度为重建值，这些帧不作为异常证据
    # （见 report_rule.py），但仍参与 φ 计算与训练清洗。
    ring = (df["Ringing"].fillna(0).astype(int) | df.get("SpeedStep", 0)).astype(int)
    bnd = (ring.diff().fillna(0) != 0) & (df[id_col].diff() == 0)
    near = bnd.astype(int).groupby(df[id_col]).transform(
        lambda s: s.rolling(2 * EDGE_TRIM_FRAMES + 1, center=True, min_periods=1).max())
    df["RepairZone"] = ((ring == 1) | (near > 0)).astype(int)
    print(f"振铃修复区（修复段 ±{EDGE_TRIM_FRAMES / 30:.1f} s）帧 {df['RepairZone'].mean():.2%}")
    # 观测区下游边界：按纵向分箱（主线 / 匝道分开）统计重建帧（RepairZone | BadTrack）占比，
    #   以上游半段各箱的最大占比为基准，下游半段超过该值的箱视为观测边界区，计入 Edge。
    yb = (df["Local_Y"] // BOUNDARY_BIN_FT) * BOUNDARY_BIN_FT
    lg = (df["Lane_Num"] == 0).astype(int)
    bad = ((df["RepairZone"] == 1) | (df["BadTrack"] == 1)).astype(float)
    share = bad.groupby([lg, yb]).agg(["mean", "size"])
    share = share[share["size"] >= MIN_BOUNDARY_BIN]["mean"]
    ymed = float(df["Local_Y"].median())
    ref = float(share[share.index.get_level_values(1) < ymed].max())
    hot = share[(share.index.get_level_values(1) >= ymed) & (share > ref)]
    key = pd.MultiIndex.from_arrays([lg, yb])
    bz = key.isin(hot.index)
    # 从首个边界箱起，下游所有箱都算（避免孤岛）
    first_y = {k: float(min(y for (kk, y) in hot.index if kk == k)) for k in set(hot.index.get_level_values(0))}
    bz = np.zeros(len(df), bool)
    for l, y0 in first_y.items():
        bz |= (lg.to_numpy() == l) & (yb.to_numpy() >= y0)
    df["BoundaryZone"] = bz.astype(int)
    df["Edge"] = (df["Edge"].astype(bool) | bz).astype(int)
    print(f"观测边界区（重建帧占比 > 上游最大 {ref:.1%}）: 起点 {first_y}（0=主线,1=匝道 lane 0），帧 {bz.mean():.2%} → 计入 Edge")
    # 分流段起点：每条车道按纵向分箱取车道中心（可信帧 Local_X 中位数），相邻箱中心偏移率 ≥ 3σ（稳健 σ）
    #   记为弯曲箱；连续弯曲段累计偏移 ≥ 半车道宽（LEADER_LAT_TOL_FT）时，视为车道在此开始分流。
    import utility_phi as _up
    t_ok = (df["Edge"] == 0) & df["Lane_Num"].notna()
    yc = (df["Local_Y"] // CURVE_BIN_FT) * CURVE_BIN_FT
    cen = df.loc[t_ok, "Local_X"].groupby([df.loc[t_ok, "Lane_Num"], yc[t_ok]]).agg(["median", "size"])
    cen = cen[cen["size"] >= MIN_BOUNDARY_BIN]["median"]
    sl = cen.groupby(level=0).diff().dropna()
    sig_c = _up.robust_sigma(sl.to_numpy())
    curve_start = {}
    for ln, ser in sl.groupby(level=0):
        ser = ser.droplevel(0)
        hot_c = (ser.abs() >= 3 * sig_c).to_numpy()
        ys = ser.index.to_numpy(); vals = ser.to_numpy()
        i = 0
        while i < len(ys):
            if hot_c[i]:
                j = i
                while j + 1 < len(ys) and hot_c[j + 1] and ys[j + 1] - ys[j] == CURVE_BIN_FT:
                    j += 1
                # 取段内累计偏移的最大幅度而非总和：分流末端车道重新编号会造成反向大跳，抵消总和
                if np.max(np.abs(np.cumsum(vals[i:j + 1]))) >= LEADER_LAT_TOL_FT and ln not in curve_start:
                    curve_start[ln] = float(ys[i])
                i = j + 1
            else:
                i += 1
    cz = np.zeros(len(df), bool)
    for ln, y0 in curve_start.items():
        cz |= (df["Lane_Num"].to_numpy() == ln) & (yc.to_numpy() >= y0)
    df["CurveZone"] = cz.astype(int)
    # 驶向出口的车在分流段内减速属正常：分流段起点下游、最终驶入匝道（lane 0）的车辆，
    #   其帧按匝道组（Lane_ctx = 0）使用匝道阈值曲线（GT 与 φ2 共用）。CurveZone 不做掩膜。
    y_div = min(curve_start.values()) if curve_start else np.inf
    exit_veh = df.groupby(id_col)["Lane_Num"].transform(lambda x: (x == 0).any())
    ramp_ctx = exit_veh.to_numpy() & (df["Local_Y"].to_numpy() >= y_div)
    df["Lane_ctx"] = np.where(ramp_ctx, 0, df["Lane_Num"].to_numpy(float))
    print(f"分流段（车道中心偏移率 ≥ 3σ = {3 * sig_c:.2f} ft/{CURVE_BIN_FT:.0f} ft、累计 ≥ {LEADER_LAT_TOL_FT} ft 的弯曲段）起点 "
          f"{curve_start} → 分流段起点 Y={y_div:.0f} ft；其下游驶出车辆的帧按匝道组：{ramp_ctx.mean():.2%} 帧"
          f"（其中原 Lane_Num≠0 {np.mean(ramp_ctx & (df['Lane_Num'].to_numpy() != 0)):.2%}）")
    print(f"外推假轨迹（速度连续 ≥ {EXTRAP_MIN_FRAMES / 30:.0f} s 不变）帧 {df['Extrapolated'].mean():.3%}，"
          f"车辆 {df.loc[df['Extrapolated'] == 1, id_col].nunique()}")
    sat = (df["Acceleration"].abs() >= RAW_ACC_BOUND - 1e-3)
    df["SatFrac"]    = sat.groupby(df[id_col]).transform("mean").round(4)
    df["BadVehicle"] = (df["SatFrac"] > BAD_VEH_SAT_FRAC).astype(int)
    bad_ids = sorted(df.loc[df["BadVehicle"] == 1, id_col].unique().tolist())
    print(f"BadVehicle (raw |a|=={RAW_ACC_BOUND} 占比 > {BAD_VEH_SAT_FRAC:.0%}): {bad_ids}")
    print(f"\n数据质量: Edge 帧 {df['Edge'].sum()} ({df['Edge'].mean():.2%}), "
          f"BadTrack(|a|>{MAX_PLAUSIBLE_G}g) 帧 {df['BadTrack'].sum()} "
          f"(其中非 Edge: {((df['BadTrack']==1)&(df['Edge']==0)).sum()})")
    return df


# =========================
# 3) TTC 计算：TTC = 净距 / (v_ego − v_leader)，仅在前车存在、净距 > 0、本车追近时计算
# =========================
def _position_leaders(df: pd.DataFrame):
    """按位置确定前车，返回每行的 (前车 ID, 净距 ft, 前车速度 ft/s)。"""
    n = len(df)
    order  = np.argsort(df[frame_col].to_numpy(), kind="stable")
    frames = df[frame_col].to_numpy()[order]
    vid = df[id_col].to_numpy(dtype=float)[order]
    y   = df["Local_Y"].to_numpy(dtype=float)[order]
    x   = df["Local_X"].to_numpy(dtype=float)[order]
    ln  = df["Length"].fillna(15.0).to_numpy(dtype=float)[order]
    v   = df["Speed_smooth"].to_numpy(dtype=float)[order]
    lead = np.full(n, np.nan); gap = np.full(n, np.nan); vl = np.full(n, np.nan)
    bounds = np.flatnonzero(np.diff(frames)) + 1
    for a, b in zip(np.r_[0, bounds], np.r_[bounds, n]):
        if b - a <= 1:
            continue
        dy = y[None, a:b] - y[a:b, None]                      # j 相对 i 的纵向距离
        ok = (dy > 0) & (np.abs(x[None, a:b] - x[a:b, None]) < LEADER_LAT_TOL_FT)
        dy = np.where(ok, dy, np.inf)
        j  = dy.argmin(axis=1)
        has = np.isfinite(dy[np.arange(b - a), j])
        idx = np.arange(a, b)[has]; jj = a + j[has]
        lead[idx] = vid[jj]
        # Local_Y 为车辆中心：净距 = 中心距 − 前车半长 − 本车半长
        gap[idx]  = y[jj] - y[idx] - ln[jj] / 2.0 - ln[idx] / 2.0
        vl[idx]   = v[jj]
    out = [np.empty(n) for _ in range(3)]
    for o, arr in zip(out, (lead, gap, vl)):
        o[order] = arr
    return out


def add_ttc(df: pd.DataFrame) -> pd.DataFrame:
    """按位置重新确定前车并计算 TTC。

    前车 = 同一帧中横向中心距 < LEADER_LAT_TOL_FT 且在本车前方的最近车辆。不用 Lane_Num
    （分流区不可靠）、Width（航拍框宽度虚高），也不用原始 Preceding_Vehicle_ID / Space_Headway
    （冲突帧中大多错误，原列保留不动）。
    新增列：Leader_ID；Gap_bumper（前车车尾到本车车头净距，ft）；Preceding_Speed、Rel_Speed、TTC；
    以及前车不可信的标记 Leader_bad / Leader_overlap / Leader_brief、前车切换的 Prev_Leader_Speed / CutIn、
    横向错开的 TTC_cleared。
    """
    df = df.sort_values([id_col, time_col, frame_col]).reset_index(drop=True)
    leader, gap, vlead = _position_leaders(df)
    df["Leader_ID"]       = leader
    df["Gap_bumper"]      = gap
    df["Preceding_Speed"] = vlead

    # 前车自身处于不可信帧（Edge / BadTrack / BadVehicle）时，前车速度、净距记为缺失
    qual = (df.set_index([frame_col, id_col])[["Edge", "BadTrack", "BadVehicle"]].max(axis=1)
            .groupby(level=[0, 1]).max())
    lkey = pd.MultiIndex.from_arrays([df[frame_col], df["Leader_ID"].fillna(-1).astype(int)])
    lbad = (qual.reindex(lkey).fillna(0).to_numpy() > 0) & df["Leader_ID"].notna().to_numpy()
    df["Leader_bad"] = lbad.astype(int)
    df.loc[lbad, ["Gap_bumper", "Preceding_Speed"]] = np.nan
    print(f"前车处于不可信帧 → 前车量记为缺失: {lbad.mean():.2%} 帧")

    # 车身重叠：同一前车的连续跟随段中出现净距 ≤ 0，说明位置/车长重建有误，整段前车量记为缺失
    lid = df["Leader_ID"].fillna(-1).to_numpy()
    newrun = np.r_[True, (lid[1:] != lid[:-1]) | (df[id_col].to_numpy()[1:] != df[id_col].to_numpy()[:-1])]
    run_id = np.cumsum(newrun)
    ovl = (df["Gap_bumper"] <= 0) & df["Leader_ID"].notna()
    bad_run = pd.Series(ovl.to_numpy(), index=df.index).groupby(run_id).transform("any").to_numpy() & df["Leader_ID"].notna().to_numpy()
    df["Leader_overlap"] = bad_run.astype(int)
    df.loc[bad_run, ["Gap_bumper", "Preceding_Speed"]] = np.nan
    print(f"车身重叠（净距 ≤ 0）的同一前车跟随段 → 前车量记为缺失: {bad_run.mean():.2%} 帧（{int(ovl.sum())} 个重叠帧）")
    # 短暂前车：同一前车持续 < GT_MIN_S，多为换道途中前车判定的闪变，前车量记为缺失
    run_len = pd.Series(np.ones(len(df)), index=df.index).groupby(run_id).transform("size").to_numpy()
    brief = df["Leader_ID"].notna().to_numpy() & (run_len < MIN_CONSEC_FRAMES_TTC)
    df["Leader_brief"] = brief.astype(int)
    df.loc[brief, ["Gap_bumper", "Preceding_Speed"]] = np.nan
    print(f"短暂前车（同一前车 < {GT_MIN_S:.0f} s）→ 前车量记为缺失: {brief.mean():.2%} 帧")

    # 前车切换后 EDGE_TRIM_FRAMES 帧内（约为跟驰反应延迟）：
    #   Prev_Leader_Speed = 原前车此刻速度（驾驶员尚未响应，φ1 同时参照新旧前车）；
    #   CutIn = 新前车更近或原来无前车，即近距离由他车切入引起（φ4 不计入本车）。
    g = df.groupby(id_col)
    L = df["Leader_ID"]; prevL = g["Leader_ID"].shift(); prevG = g["Gap_bumper"].shift()
    first = g.cumcount() == 0
    sw = (~first) & L.notna() & (L.fillna(-1) != prevL.fillna(-1))
    cut = sw & (prevL.isna() | (df["Gap_bumper"] < prevG))
    sw_frame = df[frame_col].where(sw); sw_old = prevL.where(sw); sw_cut = cut.astype(float).where(sw)
    sw_frame = sw_frame.groupby(df[id_col]).ffill(); sw_old = sw_old.groupby(df[id_col]).ffill()
    sw_cut = sw_cut.groupby(df[id_col]).ffill()
    since = df[frame_col] - sw_frame
    recent = since.notna() & (since <= EDGE_TRIM_FRAMES)
    vmap = df.set_index([frame_col, id_col])["Speed_smooth"].groupby(level=[0, 1]).first()
    okey = pd.MultiIndex.from_arrays([df[frame_col], sw_old.fillna(-1).astype(int)])
    v_old = vmap.reindex(okey).to_numpy()
    df["Prev_Leader_Speed"] = np.where(recent & sw_old.notna(), v_old, np.nan)
    df["CutIn"] = (recent & (sw_cut == 1)).astype(int)
    print(f"前车切换 {int(sw.sum())} 次（其中切入/新出现 {int(cut.sum())}）；切换后 1.5 s 内帧 {recent.mean():.2%}")

    df["Rel_Speed"] = df["Speed_smooth"] - df["Preceding_Speed"]
    valid = (
        df["Leader_ID"].notna() &
        (df["Gap_bumper"] > 0) &
        (df["Rel_Speed"] > 0.5)   # 排除数值噪声（< 0.5 ft/s 相对速度视为不追近）
    )
    df["TTC"] = np.nan
    df.loc[valid, "TTC"] = df.loc[valid, "Gap_bumper"] / df.loc[valid, "Rel_Speed"]
    # 横向错开：按当前横向分离速度（两车横向中心距在 LAT_SEP_FRAMES 内的变化率），
    #   错开所需时间 t_clear = (LEADER_LAT_TOL_FT − |Δx|) / 分离速度 < TTC 时，两车追上前已错开，TTC 记为缺失。
    xk = df.set_index([frame_col, id_col])["Local_X"]
    xk = xk[~xk.index.duplicated()]
    lk = pd.MultiIndex.from_arrays([df[frame_col], df["Leader_ID"].fillna(-1).astype(int)])
    dxa = pd.Series(np.abs(df["Local_X"].to_numpy(float) - xk.reindex(lk).to_numpy()), index=df.index)
    g2 = df.groupby(id_col)
    same_l = g2["Leader_ID"].shift(LAT_SEP_FRAMES).eq(df["Leader_ID"])
    vsep = ((dxa - dxa.groupby(df[id_col]).shift(LAT_SEP_FRAMES)) / (LAT_SEP_FRAMES / 30.0)).where(same_l)
    t_clear = np.where(vsep > 0, (LEADER_LAT_TOL_FT - dxa) / vsep, np.inf)
    cleared = df["TTC"].notna() & (t_clear < df["TTC"])
    print(f"横向错开先于追上（不在碰撞路径）: TTC 有效帧中 {cleared.sum() / max(df['TTC'].notna().sum(), 1):.2%}，"
          f"其中 TTC<{TTC_SEV_S}s 帧 {int((cleared & (df['TTC'] < TTC_SEV_S)).sum())} → 记为缺失")
    df["TTC_cleared"] = cleared.astype(int)
    df.loc[cleared, "TTC"] = np.nan
    mism = (df["Space_Headway"] > 0) & df["Leader_ID"].notna() & \
           (df["Preceding_Vehicle_ID"] != df["Leader_ID"])
    print(f"\n原始 Preceding_Vehicle_ID 与位置前车不一致的帧: {mism.mean():.2%}")

    ttc_valid = df["TTC"].dropna()
    print(f"\nTTC 有效帧数: {len(ttc_valid)} / {len(df)}")
    if len(ttc_valid) > 0:
        print(f"TTC 分布（秒）:\n{ttc_valid.describe()}")
        print(f"TTC < {TTC_CRIT_S}s 的帧数: {(ttc_valid < TTC_CRIT_S).sum()}")
        print(f"TTC < {TTC_SEV_S}s 的帧数: {(ttc_valid < TTC_SEV_S).sum()}")
    return df


# =========================
# 4) 异常标注
# =========================
def add_anomaly_labels(df: pd.DataFrame) -> pd.DataFrame:
    """拟合加减速阈值曲线（供 φ2 使用并保存），并逐帧标注 HardBrake、TailgateTTC_sev 等异常（未做持续时间过滤）。"""
    import accel_thresholds as at
    # 只在可信帧上标注（排除 Edge、BadTrack、BadVehicle）
    ok = ((df["Edge"] == 0) & (df["BadTrack"] == 0) & (df["BadVehicle"] == 0))

    # φ2 用的速度-阈值曲线：可信帧上拟合，分组为小汽车主线 / 货车 / 小汽车出口匝道
    grp = at.group_id(df["is_truck"], df.get("Lane_ctx", df["Lane_Num"]))
    curves = at.fit_curves(df["Speed_smooth"], df["Acc_g"], grp, ok)
    # 情境等级：参照速度上限 = max(局部车流中位数, 前车速度)，下限 = min(两者)；
    #   追赶落差 = (上限 − v)/σ，逼近落差 = (v − 下限)/σ，σ 为跟驰车相对局部车流的稳健 σ。
    #   按等级分别拟合曲线：加速帧用追赶等级，减速帧用逼近等级。
    import utility_phi as up
    from phi1_local import local_speed_z
    v = df["Speed_smooth"].to_numpy(float)
    dev, _, _ = local_speed_z(df)
    v_loc = v - dev.to_numpy()
    gp = df["Gap_bumper"].to_numpy(float)
    thw = np.where(np.isfinite(gp) & (gp > 0), gp / np.maximum(v, up.V_FLOOR), np.inf)
    s_c = up.robust_sigma((v - v_loc)[ok.to_numpy() & (thw <= up.THW_CONS) & np.isfinite(v_loc)])
    vl = np.where(np.isfinite(gp) & (gp > 0), df["Preceding_Speed"].to_numpy(float), np.nan)
    ref_up = np.fmax(v_loc, vl); ref_dn = np.fmin(v_loc, vl)
    df["CatchCls"] = at.context_class((ref_up - v) / s_c)
    df["ApprCls"] = at.context_class((v - ref_dn) / s_c)
    df["CatchUp"] = (df["CatchCls"] > 0).astype(int)
    o = ok.to_numpy()
    print(f"情境（σ = {s_c:.2f} ft/s）: 追赶等级 0–3 占可信帧 "
          f"{[round(float((df['CatchCls'].to_numpy()[o] == k).mean()), 3) for k in range(4)]}；逼近等级 "
          f"{[round(float((df['ApprCls'].to_numpy()[o] == k).mean()), 3) for k in range(4)]}")
    curves = at.add_context_curves(curves, df["Speed_smooth"], df["Acc_g"], grp, df["CatchCls"], df["ApprCls"], ok)
    grp = at.group_id(df["is_truck"], df.get("Lane_ctx", df["Lane_Num"]), at.frame_ctx(df["Acc_g"], df["CatchCls"], df["ApprCls"]))
    at.summary(curves, df["Speed_smooth"], df["Acc_g"], grp, ok)
    at.save(curves); at.plot(curves)
    df["Acc_z"] = at.zscore(curves, df["Speed_smooth"], df["Acc_g"], grp).round(4)
    df["Acc_level"] = np.select([df["Acc_z"] >= 2, df["Acc_z"] >= 1], [2, 1], 0)  # 1 一般异常 / 2 极端异常
    print(f"原文分级（可信帧）：一般异常(1σ~2σ) {(df.loc[ok, 'Acc_level'] == 1).mean():.2%}，"
          f"极端异常(≥2σ) {(df.loc[ok, 'Acc_level'] == 2).mean():.2%}")

    # GT 急减速：min(z_curve, z_glob) ≥ GT_ACC_Z。
    #   z_curve：速度-阈值曲线 z，只分车型（不分车道、不分情境，以免漏掉车队连锁急刹）；
    #   z_glob：分车型的全局 μ、σ。曲线两端三次多项式会下弯，取二者较小值避免把温和减速判为异常。
    #   急加速不进 GT（5σ 仅 0.13–0.18 g，多为普通加速）。
    a_g = df["Acc_g"].to_numpy(float); okv = ok.to_numpy(); trk = df["is_truck"].fillna(0).astype(int).to_numpy()
    curves_gt = at.fit_curves(df["Speed_smooth"], a_g, trk, ok)
    at.save(curves_gt, "accel_threshold_curves_gt.json")
    z_cur = at.zscore(curves_gt, df["Speed_smooth"], a_g, trk)
    z_glob = np.zeros(len(df))
    for tr in (0, 1):
        for sgn in (1, -1):
            sel = (trk == tr) & (np.sign(a_g) == sgn)
            ref = np.abs(a_g[sel & okv]); ref = ref[np.isfinite(ref)]; mu, sd = ref.mean(), ref.std()
            z_glob[sel] = (np.abs(a_g[sel]) - mu) / sd
            print(f"  GT 全局 5σ（{'货车' if tr else '小汽车'}{'加速' if sgn > 0 else '减速'}）: {mu + GT_ACC_Z * sd:.3f} g")
    df["Acc_zgt"] = np.round(np.minimum(z_cur, z_glob), 4)   # 加/减速两侧都存；GT 只用减速侧
    df["HardBrake"] = ((df["Acc_g"] < 0) & (df["Acc_zgt"] >= GT_ACC_Z) & ok).astype(int)
    df["HardAccel"] = 0
    print(f"  （仅对照，不进 GT）同口径急加速帧：{int(((df['Acc_g'] > 0) & (df['Acc_zgt'] >= GT_ACC_Z) & ok).sum())}")

    # TTC 冲突要求前车与本车同车道
    lane_of = df.set_index([frame_col, id_col])["Lane_Num"]
    key = pd.MultiIndex.from_arrays([df[frame_col], df["Leader_ID"].fillna(-1).astype(int)])
    lead_lane = lane_of.reindex(key).to_numpy()
    same_lane = pd.Series(lead_lane == df["Lane_Num"].to_numpy(), index=df.index)
    df["TailgateTTC"]     = ((df["TTC"] < TTC_CRIT_S).fillna(False) & ok).astype(int)
    df["TailgateTTC_sev"] = ((df["TTC"] < TTC_SEV_S).fillna(False) & same_lane & ok).astype(int)

    df["Anomaly"] = ((df["HardBrake"] == 1) | (df["HardAccel"] == 1) |
                     (df["TailgateTTC_sev"] == 1)).astype(int)

    print(f"\nAnomaly 标注（逐帧，未做持续时间过滤）：")
    print(f"  文献阈值对照: |a| 超 SHRP2 (-0.65/+0.58 g) 帧数 "
          f"{int(((df['Acc_g'] <= HARD_BRAKE_G) | (df['Acc_g'] >= HARD_ACCEL_G)).sum())}")
    print(f"  HardBrake (z>={GT_ACC_Z}) : {df['HardBrake'].sum()} 帧")
    print(f"  HardAccel (z>={GT_ACC_Z}) : {df['HardAccel'].sum()} 帧")
    print(f"  TailgateTTC(<{TTC_CRIT_S}s)     : {df['TailgateTTC'].sum()} 帧")
    print(f"  TailgateTTC(<{TTC_SEV_S}s, 同车道): {df['TailgateTTC_sev'].sum()} 帧")
    return df


# =========================
# 5) 持续时间过滤 → Anomaly_30
# =========================
def _keep_runs_minlen(arr: np.ndarray, min_len: int) -> np.ndarray:
    """只保留长度 ≥ min_len 的连续 1 段。"""
    out = np.zeros_like(arr)
    i = 0
    while i < len(arr):
        if arr[i] == 1:
            j = i
            while j < len(arr) and arr[j] == 1:
                j += 1
            if j - i >= min_len:
                out[i:j] = 1
            i = j
        else:
            i += 1
    return out


def _drop_noise_runs(df: pd.DataFrame, flag: pd.Series) -> pd.Series:
    """剔除缺乏邻车印证的急变速段（重建噪声）。n_nb = 同车道 ±NB_DY_FT 内、同时段速度变化 > NB_DV_FTPS 的邻车数：
      - 段内有截断帧或振铃/阶跃修复帧（幅度不可信）→ 需 n_nb ≥ NB_MIN_CAPPED；
      - 否则原始加速度饱和帧占比 > NOISE_SAT_FRAC → 需 n_nb ≥ 1。"""
    flag = flag.copy()
    by_frame = df[[frame_col, id_col, "Lane_Num", "Local_Y", "Speed_smooth"]].set_index(frame_col).sort_index()
    sat = df["Acceleration"].abs() >= RAW_ACC_BOUND - 1e-3
    capped = df["Acc_ftps2"].abs() >= ACC_CAP_FTPS2
    ring = df["Ringing"].fillna(0).astype(bool) if "Ringing" in df else pd.Series(False, index=df.index)
    if "SpeedStep" in df:
        ring = ring | df["SpeedStep"].fillna(0).astype(bool)
    n_runs = n_sat = n_cap = n_ring = 0
    for vid, g in df[flag == 1].groupby(id_col):
        f = g[frame_col].to_numpy()
        br = np.flatnonzero(np.diff(f) != 1) + 1
        for seg in np.split(np.arange(len(f)), br):
            n_runs += 1
            ix = g.index[seg]
            is_ring = bool(ring[ix].any())
            is_cap = bool(capped[ix].any()) or is_ring
            if not is_cap and sat[ix].mean() <= NOISE_SAT_FRAC:
                continue
            s0, e0 = int(f[seg[0]]), int(f[seg[-1]])
            w = by_frame.loc[s0 - 30:e0 + 90]
            lane = df.loc[ix, "Lane_Num"].mode()[0]; y0 = df.loc[ix, "Local_Y"].mean()
            nb = w[(w[id_col] != vid) & (w["Lane_Num"] == lane) & ((w["Local_Y"] - y0).abs() < NB_DY_FT)]
            rng = nb.groupby(id_col)["Speed_smooth"].agg(lambda x: x.max() - x.min()) if len(nb) else pd.Series(dtype=float)
            n_nb = int((rng > NB_DV_FTPS).sum())
            if n_nb < (NB_MIN_CAPPED if is_cap else 1):
                flag[ix] = 0
                if is_ring: n_ring += 1
                elif is_cap: n_cap += 1
                else: n_sat += 1
    print(f"  急加/减速段 {n_runs} 个：剔除 {n_sat} 个重建噪声（原始加速度饱和 > {NOISE_SAT_FRAC:.0%} 且无邻车印证），"
          f"{n_cap} 个截断段（加速度顶在 ±20 ft/s² 且邻车印证 < {NB_MIN_CAPPED} 辆），"
          f"{n_ring} 个振铃段（速度锯齿、邻车印证 < {NB_MIN_CAPPED} 辆）")
    return flag


def add_anomaly_filtered(df: pd.DataFrame) -> pd.DataFrame:
    """持续时间过滤与噪声剔除，生成 GT_acc、GT_ttc、Anomaly_30（全部 GT）、Anomaly_30_accel（仅加速度）。"""
    def _filter(col, min_frames):
        return (
            df.groupby(id_col, group_keys=False)
              .apply(lambda g: pd.Series(
                  _keep_runs_minlen(
                      g[col].fillna(0).astype(int).to_numpy(),
                      min_frames
                  ),
                  index=g.index
              ))
        )

    hb_f  = _filter("HardBrake",       MIN_CONSEC_FRAMES)
    ha_f  = _filter("HardAccel",       MIN_CONSEC_FRAMES)
    ttc_f = _filter("TailgateTTC_sev", MIN_CONSEC_FRAMES_TTC)
    acc_f = _drop_noise_runs(df, ((hb_f == 1) | (ha_f == 1)).astype(int))
    hb_f  = hb_f * acc_f
    ha_f  = ha_f * acc_f
    df["GT_acc"] = ((hb_f == 1) | (ha_f == 1)).astype(int)
    # GT_phys 与 GT_acc 相同，仅为下游兼容保留列名
    df["GT_phys"] = df["GT_acc"]
    print(f"GT 物理锚点口径：急加/减速帧 {int(df['GT_acc'].sum())} → 同时超出物理锚点且 ≥1 s {int(df['GT_phys'].sum())}")
    df["GT_ttc"] = (ttc_f == 1).astype(int)

    # 全部 GT（加速度 + TTC），用于 val 集划分
    df["Anomaly_30"] = (
        (hb_f == 1) | (ha_f == 1) | (ttc_f == 1)
    ).astype(int)

    # 仅加速度 GT，用于模型评估
    df["Anomaly_30_accel"] = (
        (hb_f == 1) | (ha_f == 1)
    ).astype(int)

    before       = df["Anomaly"].sum()
    after_full   = df["Anomaly_30"].sum()
    after_accel  = df["Anomaly_30_accel"].sum()
    print(f"\n连续帧过滤：")
    print(f"  HardBrake/HardAccel >= {MIN_CONSEC_FRAMES} 帧，TailgateTTC >= {MIN_CONSEC_FRAMES_TTC} 帧")
    print(f"  Anomaly_30      (full) : {before} → {after_full} 帧")
    print(f"  Anomaly_30_accel(accel): {before} → {after_accel} 帧")
    return df


# =========================
# 6) 事件化
# =========================
def _build_events_for_vehicle(g: pd.DataFrame, anomaly_col: str = "Anomaly_30") -> pd.DataFrame:
    """把单车的异常帧连成事件：间隔 ≤ MERGE_GAP_S 合并，前后扩展 CONTEXT_EXPAND_S，扩展后重叠再合并。"""
    g    = g.sort_values(time_col).copy()
    x    = g[anomaly_col].to_numpy(dtype=int)
    t    = g[time_col].to_numpy(dtype=float)
    fr   = g[frame_col].to_numpy(dtype=int)
    a    = g["Acc_g"].to_numpy(dtype=float)
    # severe 帧 = TailgateTTC_sev / HardBrake / HardAccel
    ttc_sev  = g["TailgateTTC_sev"].fillna(0).to_numpy(dtype=int) if "TailgateTTC_sev" in g.columns else np.zeros(len(g), dtype=int)
    hb       = g["HardBrake"].fillna(0).to_numpy(dtype=int)       if "HardBrake"       in g.columns else np.zeros(len(g), dtype=int)
    ha       = g["HardAccel"].fillna(0).to_numpy(dtype=int)        if "HardAccel"        in g.columns else np.zeros(len(g), dtype=int)
    sev      = np.clip(ttc_sev + hb + ha, 0, 1)
    gacc     = g["GT_acc"].fillna(0).to_numpy(dtype=int) if "GT_acc" in g.columns else np.zeros(len(g), dtype=int)
    gttc     = g["GT_ttc"].fillna(0).to_numpy(dtype=int) if "GT_ttc" in g.columns else np.zeros(len(g), dtype=int)
    gphys    = g["GT_phys"].fillna(0).to_numpy(dtype=int) if "GT_phys" in g.columns else np.zeros(len(g), dtype=int)

    t_min, t_max   = float(t.min()), float(t.max())
    fr_min, fr_max = int(fr.min()), int(fr.max())

    segs = []
    i = 0
    while i < len(x):
        if x[i] == 1:
            j = i
            while j < len(x) and x[j] == 1:
                j += 1
            sl = slice(i, j)
            k  = i + int(np.argmax(np.abs(a[sl])))
            segs.append({
                id_col:        g[id_col].iloc[0],
                "start_time":  float(t[i]),
                "end_time":    float(t[j - 1]),
                "start_frame": int(fr[i]),
                "end_frame":   int(fr[j - 1]),
                "min_g":       float(a[sl].min()),
                "max_g":       float(a[sl].max()),
                "peak_abs_g":  float(np.abs(a[sl]).max()),
                "peak_g":      float(a[k]),
                "severity":    "severe" if sev[sl].any() else "moderate",
                "gt_type":     "|".join(n for n, m in [("harsh_accel", gacc[sl].any()),
                                                       ("ttc_conflict", gttc[sl].any())] if m),
                "phys_harsh":  int(gphys[sl].any()),
            })
            i = j
        else:
            i += 1

    if not segs:
        return pd.DataFrame(columns=[
            id_col, "event_id", "start_time", "end_time", "duration_s",
            "start_frame", "end_frame", "num_frames",
            "min_g", "max_g", "peak_abs_g", "peak_g", "severity", "gt_type", "phys_harsh"
        ])

    segs = pd.DataFrame(segs).sort_values("start_time").reset_index(drop=True)

    # 间隔 ≤ MERGE_GAP_S 合并
    merged = [segs.iloc[0].to_dict()]
    for r in segs.iloc[1:].to_dict("records"):
        cur = merged[-1]
        if r["start_time"] - cur["end_time"] <= MERGE_GAP_S:
            cur["end_time"]  = max(cur["end_time"],  r["end_time"])
            cur["end_frame"] = max(cur["end_frame"], r["end_frame"])
            cur["min_g"]     = min(cur["min_g"],     r["min_g"])
            cur["max_g"]     = max(cur["max_g"],     r["max_g"])
            if r["peak_abs_g"] > cur["peak_abs_g"]:
                cur["peak_abs_g"] = r["peak_abs_g"]
                cur["peak_g"]     = r["peak_g"]
            if r["severity"] == "severe":
                cur["severity"] = "severe"
            cur["gt_type"] = "|".join(sorted(set(filter(None, (cur.get("gt_type", "") + "|" + r.get("gt_type", "")).split("|")))))
            cur["phys_harsh"] = int(max(cur.get("phys_harsh", 0), r.get("phys_harsh", 0)))
        else:
            merged.append(r)

    events = pd.DataFrame(merged)

    # ±CONTEXT_EXPAND_S 扩展
    events["start_time"]  = (events["start_time"]  - CONTEXT_EXPAND_S).clip(lower=t_min)
    events["end_time"]    = (events["end_time"]    + CONTEXT_EXPAND_S).clip(upper=t_max)
    events["start_frame"] = (
        (events["start_frame"] - CONTEXT_EXPAND_S * FPS)
        .clip(lower=fr_min).round().astype(int)
    )
    events["end_frame"] = (
        (events["end_frame"] + CONTEXT_EXPAND_S * FPS)
        .clip(upper=fr_max).round().astype(int)
    )

    # 扩展后重叠再合并
    events = events.sort_values("start_time").reset_index(drop=True)
    merged2 = [events.iloc[0].to_dict()]
    for r in events.iloc[1:].to_dict("records"):
        cur = merged2[-1]
        if r["start_time"] <= cur["end_time"]:
            cur["end_time"]  = max(cur["end_time"],  r["end_time"])
            cur["end_frame"] = max(cur["end_frame"], r["end_frame"])
            cur["min_g"]     = min(cur["min_g"],     r["min_g"])
            cur["max_g"]     = max(cur["max_g"],     r["max_g"])
            if r["peak_abs_g"] > cur["peak_abs_g"]:
                cur["peak_abs_g"] = r["peak_abs_g"]
                cur["peak_g"]     = r["peak_g"]
            if r["severity"] == "severe":
                cur["severity"] = "severe"
            cur["gt_type"] = "|".join(sorted(set(filter(None, (cur.get("gt_type", "") + "|" + r.get("gt_type", "")).split("|")))))
            cur["phys_harsh"] = int(max(cur.get("phys_harsh", 0), r.get("phys_harsh", 0)))
        else:
            merged2.append(r)

    events = pd.DataFrame(merged2)

    # num_frames / duration_s 由帧号计算
    events["num_frames"] = (events["end_frame"] - events["start_frame"] + 1).astype(int)
    events["duration_s"] = (events["num_frames"] / FPS).round(4)
    events["event_id"]   = np.arange(1, len(events) + 1)

    return events[[
        id_col, "event_id", "start_time", "end_time", "duration_s",
        "start_frame", "end_frame", "num_frames",
        "min_g", "max_g", "peak_abs_g", "peak_g", "severity", "gt_type", "phys_harsh"
    ]]


def eventize(df: pd.DataFrame, anomaly_col: str = "Anomaly_30") -> pd.DataFrame:
    events_df = (
        df.groupby(id_col, group_keys=False)
          .apply(lambda g: _build_events_for_vehicle(g, anomaly_col=anomaly_col))
          .reset_index(drop=True)
    )
    print(f"\n总事件数: {len(events_df)}，涉及 {events_df[id_col].nunique()} 辆车")
    if "gt_type" in events_df.columns and len(events_df):
        print("事件类型:", events_df["gt_type"].value_counts().to_dict())
    d = events_df["duration_s"]
    print("事件时长分布:")
    for lo, hi, label in [(0,1,"<1s"),(1,5,"1-5s"),(5,10,"5-10s"),(10,999,">10s")]:
        print(f"  {label}: {((d>=lo)&(d<hi)).sum()} 个")
    return events_df


# =========================
# 7) Sanity 图
# =========================
def plot_top_events(df: pd.DataFrame, events_df: pd.DataFrame,
                    out_dir: str = "sanity_plots"):
    """画持续最长和峰值最大的若干事件的速度、加速度、TTC 曲线。"""
    if events_df.empty:
        print("没有事件，跳过绘图")
        return
    os.makedirs(out_dir, exist_ok=True)

    to_plot = pd.concat([
        events_df.nlargest(3, "duration_s"),
        events_df.nlargest(3, "peak_abs_g"),
    ]).drop_duplicates(subset=[id_col, "event_id"]).head(9)

    for _, ev in to_plot.iterrows():
        vid = int(ev[id_col])
        st, et = float(ev["start_time"]), float(ev["end_time"])
        g   = df[df[id_col] == vid].sort_values(time_col)
        seg = g[(g[time_col] >= st - 2) & (g[time_col] <= et + 2)]
        if seg.empty:
            continue

        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

        t = seg[time_col] - st
        axes[0].plot(t, seg[speed_col], color="#185FA5")
        axes[0].axvspan(0, et - st, alpha=0.15, color="orange")
        axes[0].set_ylabel("Speed (ft/s)")
        axes[0].set_title(f"Vehicle {vid}")

        axes[1].plot(t, seg["Acc_g"], color="#A32D2D", alpha=0.85)
        axes[1].axvspan(0, et - st, alpha=0.15, color="orange")
        axes[1].set_ylabel("Accel (g)")

        axes[2].plot(t, seg["TTC"], color="#0F6E56", alpha=0.85)
        axes[2].axhline(TTC_SEV_S, color="red", ls=":", lw=0.8)
        axes[2].axvspan(0, et - st, alpha=0.15, color="orange")
        axes[2].set_ylabel("TTC (s)")
        axes[2].set_xlabel("Time (s)")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir,
                    f"vid{vid}_event{int(ev['event_id'])}.png"), dpi=150)
        plt.close(fig)

    print(f"已保存 {len(to_plot)} 张图到 {out_dir}/")


# =========================
# 主流程
# =========================
DEFAULT_RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "highsim_download", "HighSim_Data_20260601.csv")


def main(input_csv: str = DEFAULT_RAW):
    print("=" * 55)
    print("HighSim Proxy Reference")
    print(f"GT: 急减速 min(周荣贵曲线 z, 全局 z)>={GT_ACC_Z} | TTC<{TTC_SEV_S}s 同车道 | 持续>={GT_MIN_S}s")
    print("=" * 55)

    df = load_and_filter(input_csv)
    df = add_acceleration(df)
    df = add_quality_flags(df)
    df = add_ttc(df)
    df = add_anomaly_labels(df)
    df = add_anomaly_filtered(df)
    # 全部 GT 事件，用于 model.py 的 val 集划分
    events_full  = eventize(df, anomaly_col="Anomaly_30")
    # 仅加速度 GT 事件，用于 model.py 的模型评估
    events_accel = eventize(df, anomaly_col="Anomaly_30_accel")
    plot_top_events(df, events_full)

    # 列名对齐主模型
    df = df.rename(columns={
        "Acc_ftps2": "Acc_from_vsmooth_ftps2",
        "Acc_g":     "Acc_from_vsmooth_g",
    })

    frames_out      = "HighSim_frames_with_Anomaly30.csv"
    events_full_out  = "HighSim_events_Anomaly30_merge2s.csv"
    events_accel_out = "HighSim_events_accel_only.csv"

    df.to_csv(frames_out, index=False)
    events_full.to_csv(events_full_out, index=False)
    events_accel.to_csv(events_accel_out, index=False)

    print(f"\n输出:")
    print(f"  {frames_out}        ({len(df)} 行)")
    print(f"  {events_full_out}   ({len(events_full)} 个事件，full GT：加速度+TTC，用于val集划分)")
    print(f"  {events_accel_out}  ({len(events_accel)} 个事件，accel-only GT，用于模型评估)")
    return df, events_full, events_accel


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RAW
    main(csv_path)