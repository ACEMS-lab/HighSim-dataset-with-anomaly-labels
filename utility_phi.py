"""
utility_phi.py —— 效用特征 φ1–φ4（v7）

统一标度：每个 φ 都先换算成"相对正常驾驶偏离了几个 σ"（z），再映射
    φ = clip((z − 1) / (Z_SAT − 1), 0, 1)，Z_SAT = 5
即 1σ 以内为 0，5σ（与 GT"极端"同一标准）及以上为 1，中间线性。
旧版用全体帧 z 的 P99.99 作为封顶点：φ1 的尾巴被排队路段（邻车 MAD 极小 → z 可达 40+）拉得极长，
Z_MAX=43，自由流里快 20 mph 也只有 φ1≈0.14。

φ1 速度偏离（考虑周围环境）
    正常速度区间 [lo, hi]，超出区间的部分才算偏离（区间内 φ1=0）
      - 局部车流：同帧、同车道、纵向 ±500 ft 内其他车辆速度中位数（phi1_local.py）；
        变道过渡期（车道号切换前后 1.5 s）取原车道与目标车道中偏差较小者，避免切换瞬间跳变。
      - 自由流速度：同车道 × 车型 × 500 ft 路段内、前方无约束（车头时距 ≥ THW_FREE 或无前车）车辆的速度中位数
      - w：受约束程度。车头时距 ≤ THW_CONS → w=0（跟局部车流比）；≥ THW_FREE 或无前车 → w=1（跟自由流比），中间线性
      受约束：区间 = 局部车流速度；自由：区间 = [min(局部, 自由流), max(局部, 自由流)]，中间按 w 过渡
      → 前车慢但离得远、后车加速向自由流速度靠拢：在区间内，不算异常；开到远超自由流速度仍算异常。
    σ：受约束 = 相对局部车流偏差的稳健 σ；自由 = 自由车辆相对自由流速度偏差的稳健 σ（按车道×路段）。
    z = 超出区间的量 / σ + 1，所以超出区间 5σ 时 φ1=1。

φ2 急加/减速：周荣贵等(2024) 速度-阈值曲线的 z（Data.py → Acc_z），z=5 → 1。

φ3 横向行为 = max(车道保持, 变道横移速度, 蛇行)
    - 车道保持（v8）：车道内漂移正常；|偏移| 超过正常 P99（车道×车型×500 ft 路段）后线性增至
      半车道宽（压线，数据标定）时为 1。变道过程（车道号切换前后 LC_WIN_S 秒）内不计。
    - 变道横移速度：横向速度相对"一般变道的峰值横向速度"偏离几个 σ（横向位置始终相对同一条车道中心线计算，
      车道号切换时不跳变）。
    - 蛇行：A→B→A 回头变道（在 B 停留 ≥0.5 s、且横向确有 ≥ WEAVE_MIN_DX_FT 位移），间隔越短越高：
      间隔 ≤ 全体回头变道间隔的 P5 → 1，≥ P25 → 0。同方向连续变道（如逐条并入匝道）不计。

φ4 碰撞风险：纵向 TTC 项不变；侧向项的安全距离按数据标定（相邻车道并排车辆车身边缘间距的 P1 处 φ=0.5、P5 处 φ=0.1）。
    数据中 Width 为外接框（小汽车 8 ft、货车 12.5 ft，比实际车宽大），旧版固定 3 ft 安全距离使 10% 的正常并排 φ≥0.87。
"""
import numpy as np
import pandas as pd

Z_SAT = 5.0
FPS = 30
THW_CONS, THW_FREE = 2.0, 4.0
V_FLOOR = 44.0            # 30 mph（ft/s）
LC_TRANS_S = 1.5          # φ1：车道号切换前后按两条车道取参照
RELAX_TAU_S = 4.7         # v11 P7b：前车切换后速度差松弛时间常数（tools/relax_anchor.py：v10 数据 8365 次切换，0–10 s 指数拟合 4.66 s）
LC_WIN_S = 2.5            # φ3：变道过程窗口
YBIN_FT = 500.0
MIN_N = 3000
LAT_K = 15                # 横向速度差分步长（0.5 s）
WEAVE_MIN_DWELL = 15      # 在 B 车道至少 0.5 s
WEAVE_MIN_DX_FT = 6.0     # 横向位移至少半个车道
WEAVE_PAD = 30


def phi_from_z(z):
    return np.clip((np.asarray(z, float) - 1.0) / (Z_SAT - 1.0), 0.0, 1.0)


def robust_sigma(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def _seg_stat(df, val, ok, func, by_class=True, truck=None, lo=None, hi=None, label=""):
    """按 车道 × (车型) × 500 ft 路段 统计 func(val)，段中心之间沿 Y 线性插值；样本不足依次回退。"""
    out = pd.Series(np.nan, index=df.index)
    ybin = np.floor(df["Local_Y"] / YBIN_FT)
    for lane in df["Lane_Num"].dropna().unique():
        in_lane = df["Lane_Num"].eq(lane)
        lane_all = val[in_lane & ok]
        for c in ((0, 1) if by_class else (None,)):
            sel = in_lane if c is None else in_lane & truck.eq(c)
            if not sel.any():
                continue
            ref = val[sel & ok]
            whole = func(ref if len(ref) >= MIN_N else lane_all)
            yc, vb = [], []
            for b in np.unique(ybin[in_lane]):
                in_b = in_lane & ok & ybin.eq(b)
                r_c = val[in_b] if c is None else val[in_b & truck.eq(c)]
                if len(r_c) >= MIN_N:
                    vv = func(r_c)
                elif c == 0 and in_b.sum() >= MIN_N:
                    vv = func(val[in_b])
                else:
                    vv = whole
                yc.append((b + 0.5) * YBIN_FT); vb.append(vv)
            r = np.interp(df.loc[sel, "Local_Y"].to_numpy(float), yc, vb)
            if lo is not None or hi is not None:
                r = np.clip(r, lo, hi)
            out[sel] = r
            if label:
                print(f"  {label} lane {int(lane)}{'' if c is None else ' [' + ('car', 'truck')[c] + ']'}: "
                      f"整车道 {whole:.2f}，分段 {np.nanmin(vb):.2f}–{np.nanmax(vb):.2f}")
    return out


def _lane_switches(df):
    """每辆车的车道号切换：返回 DataFrame(row 位置 k, from, to)，df 须按 Vehicle_ID, Frame_ID 排序。"""
    vid = df["Vehicle_ID"].to_numpy(); ln = df["Lane_Num"].to_numpy(float)
    k = np.flatnonzero((vid[1:] == vid[:-1]) & (ln[1:] != ln[:-1])) + 1
    return pd.DataFrame({"k": k, "vid": vid[k], "frm": ln[k - 1], "to": ln[k]})


def _near_switch(df, sw, half):
    """每行到最近一次车道切换的帧距 ≤ half 的掩码，以及该切换另一侧的车道号"""
    n = len(df); mask = np.zeros(n, bool); other = np.full(n, np.nan)
    vid = df["Vehicle_ID"].to_numpy()
    for r in sw.itertuples():
        a = r.k - half; b = r.k + half
        a = max(a, 0); b = min(b, n)
        idx = np.arange(a, b); idx = idx[vid[idx] == r.vid]
        before = idx[idx < r.k]; after = idx[idx >= r.k]
        mask[idx] = True
        other[before] = r.to; other[after] = r.frm
    return mask, other


# ───────────────────────── φ1 ─────────────────────────
def compute_phi1(df, trusted, truck):
    from phi1_local import local_speed_z
    v = df["Speed_smooth"].to_numpy(float)
    dev_own, _, _ = local_speed_z(df)
    sw = _lane_switches(df)
    trans, other = _near_switch(df, sw, int(LC_TRANS_S * FPS))
    q = np.where(trans, other, np.nan)
    dev_oth, _, _ = local_speed_z(df, qlane=q)
    d_own = dev_own.to_numpy(); d_oth = dev_oth.to_numpy()
    use_oth = trans & np.isfinite(d_oth) & (~np.isfinite(d_own) | (np.abs(d_oth) < np.abs(d_own)))
    v_loc = np.where(use_oth, v - d_oth, v - d_own)

    gap = df["Gap_bumper"].to_numpy(float) if "Gap_bumper" in df else np.full(len(df), np.nan)
    # 车头时距按 max(v, V_FLOOR) 计算：排队时 3 mph、间距 30 ft 的车 THW 也有 7 s，但它显然受下游排队约束；
    # 加速度下限后，低速车需前方净空 ≥ THW_FREE·V_FLOOR（约 176 ft）才算"自由"。
    thw = np.where(np.isfinite(gap) & (gap > 0), gap / np.maximum(v, V_FLOOR), np.inf)
    w = np.clip((thw - THW_CONS) / (THW_FREE - THW_CONS), 0, 1)
    free = pd.Series(thw >= THW_FREE, index=df.index)
    print(f"φ1: 受约束(THW≤{THW_CONS}s) {np.mean(w == 0):.1%}，自由(THW≥{THW_FREE}s/无前车) {np.mean(w == 1):.1%}；"
          f"变道过渡帧 {trans.mean():.2%}（其中改用另一车道参照 {use_oth.sum() / max(trans.sum(), 1):.1%}）")
    v_free = _seg_stat(df, df["Speed_smooth"], trusted & free, np.median, truck=truck,
                       label="自由流速度 ft/s").to_numpy()
    # 正常速度区间 [lo, hi]：
    #   受约束(w=0)：区间退化为局部车流速度；
    #   自由(w=1)：上限 = max(局部车流, 自由流速度) —— 前方空、加速向自由流速度靠拢不算异常；
    #             下限 = min(局部车流, 自由流速度) —— 比周围车和自由流都慢才算异常。
    has = np.isfinite(v_loc)
    hi = np.where(has, (1 - w) * v_loc + w * np.maximum(v_loc, v_free), v_free)
    lo = np.where(has, (1 - w) * v_loc + w * np.minimum(v_loc, v_free), v_free)
    # v9：跟驰中（w<1）的车，跟着前车速度走也属正常。交通波里前车速度与 ±500 ft 车流中位数可差 10 mph，
    #     v8 中 φ1 的 3σ 离群帧 86% 处于跟驰状态，其中位数与前车只差 3.8 mph —— 是正常跟驰而非速度异常。
    #     正常区间扩为同时包含"局部车流"与"前车速度"；比两者都快/都慢才算偏离（逼近前车由 φ4 的 TTC 负责）。
    if "Preceding_Speed" in df:
        v_lead = df["Preceding_Speed"].to_numpy(float)
        use_l = np.isfinite(v_lead) & (w < 1)
        lo = np.where(use_l, np.minimum(lo, v_lead), lo)
        hi = np.where(use_l, np.maximum(hi, v_lead), hi)
    # v10（P2）：前车切换后 1.5 s 内，原前车速度也在正常区间内（驾驶员尚未来得及对新前车作出反应；
    #     v9 复核中多例 φ1 的 3σ 恰好始于前车/参照切换，Data.py → Prev_Leader_Speed）
    if "Prev_Leader_Speed" in df:
        v_old = df["Prev_Leader_Speed"].to_numpy(float)
        use_o = np.isfinite(v_old) & (w < 1)
        lo = np.where(use_o, np.minimum(lo, v_old), lo)
        hi = np.where(use_o, np.maximum(hi, v_old), hi)
    # v10（P4）：偏快一侧的上限在任何状态下都不低于路段自由流速度。跟驰中比前车/车流快但不超过正常自由流速度
    #     属于接近前车，其风险由 φ4（TTC）衡量，不在 φ1 重复计入；只有超过自由流速度才算速度偏离（超速）。
    #     （v10 特征统计：φ1 偏快 3σ 帧 82% 处于跟驰 THW≤2 s，中位间距 140 ft，TTC<3 s 仅 4.4%。）
    #     下限不变：比车流、前车、自由流都慢才算偏慢。
    hi = np.maximum(hi, np.where(np.isfinite(v_free), v_free, hi))
    # v11（P7b，用户确认 2026-11-04）：前车切换后的松弛过渡。换了前车（本车或他车变道）后，本车与新前车的速度差
    #     不是立刻消失，而是逐渐收敛（交通流"松弛现象"）。数据（tools/relax_anchor.py）：切换后本车−新前车速度差
    #     中位数相对稳态（切换后 10–20 s）的超出量按指数衰减，时间常数 RELAX_TAU_S。
    #     正常区间在切换后同时包含 新前车速度 + 切换时速度差·exp(−t/τ)（平滑递减，不硬切换）。
    #     只作用于依赖前车的参照；本车自身急加减速仍由 φ2 判定。
    if "Leader_ID" in df and "Preceding_Speed" in df:
        vid = df["Vehicle_ID"].to_numpy(); lid = df["Leader_ID"].fillna(-1).to_numpy()
        new_seg = np.r_[True, (vid[1:] != vid[:-1]) | (lid[1:] != lid[:-1])]
        first_seg = np.r_[True, vid[1:] != vid[:-1]]
        seg = np.cumsum(new_seg)
        pos = np.arange(len(df)) - pd.Series(np.arange(len(df))).groupby(seg).transform("min").to_numpy()
        v_l = df["Preceding_Speed"].to_numpy(float)
        dv0 = pd.Series(v - v_l).groupby(seg).transform("first").to_numpy()
        # 车辆的第一段（轨迹开始时已有的前车）不是切换
        is_first = pd.Series(first_seg).groupby(seg).transform("max").to_numpy().astype(bool)
        relax = (~is_first) & np.isfinite(dv0) & np.isfinite(v_l) & (lid >= 0) & (w < 1)
        path = v_l + dv0 * np.exp(-pos / (RELAX_TAU_S * FPS))
        lo = np.where(relax, np.minimum(lo, path), lo)
        hi = np.where(relax, np.maximum(hi, path), hi)
        print(f"φ1（P7b）: 前车切换后松弛过渡（τ={RELAX_TAU_S} s）适用帧 {relax.mean():.2%}")
    dev = np.maximum(v - hi, 0) + np.maximum(lo - v, 0)
    t = trusted.to_numpy()
    # σ：受约束 = 相对局部车流偏差的稳健 σ（全局）；自由 = 自由车辆相对自由流速度偏差的稳健 σ（按 车道×路段）
    s_c = robust_sigma((v - v_loc)[t & (w == 0)])
    r_free = pd.Series(np.abs(v - v_free), index=df.index)
    s_f = _seg_stat(df, r_free, trusted & free, lambda a: 1.4826 * np.median(np.abs(a)), by_class=False,
                    lo=s_c, hi=None, label="自由流速度 σ ft/s").to_numpy()
    sig = (1 - w) * s_c + w * s_f
    # v11（P7a，用户确认 2026-11-04）：偏快一侧的上限含路段自由流速度（P4），超出部分是"比自由流车辆快多少"，
    #     其离散程度应按自由流车辆的 σ（s_f）衡量，而不是跟驰车相对局部车流的 σ（s_c ≈ 2.5 mph）。
    #     v10 复核：本车 THW 1.6–2.1 s 被判为跟驰，超越 200 ft 外慢车时 φ1 用 s_c 而饱和（#31/41/44/34）。
    #     逼近前车的风险由 φ4（TTC）衡量。偏慢一侧不变。
    sig_fast = np.where(np.isfinite(s_f), np.maximum(sig, s_f), sig)
    z = np.where(v > hi, (v - hi) / sig_fast, dev / sig)   # 超出正常区间几个 σ；≤1σ → φ=0，≥5σ → φ=1
    phi = np.nan_to_num(phi_from_z(z), nan=0.0)
    sf_med = float(np.nanmedian(s_f[t & (w == 1)]))
    print(f"φ1: σ 受约束 {s_c * 0.6818:.2f} mph，自由（中位）{sf_med * 0.6818:.2f} mph → φ1=1 对应超出正常区间 "
          f"{Z_SAT * s_c * 0.6818:.1f} / {Z_SAT * sf_med * 0.6818:.1f} mph；φ1>0 {np.mean(phi[t] > 0):.2%}，"
          f"φ1=1 {np.mean(phi[t] >= 1):.3%}")
    return phi, {"sigma_cons_mph": s_c * 0.6818, "sigma_free_mph_median": sf_med * 0.6818,
                 "_parts": dict(v_loc=v_loc, v_free=v_free, w=w, thw=thw, dev=dev, sig=sig, lo=lo, hi=hi, s_f=s_f, s_c=s_c),
                 "_z": np.nan_to_num(z, nan=0.0)}


# ───────────────────────── φ2 ─────────────────────────
# v10（P1）：跟驰中跟着前车加减速是正常响应。响应延迟由数据估计（v9，399 对跟驰车 THW≤2 s、≥20 s）：
#   本车与前车加速度的互相关在延迟 1.5 s 处最大（r=0.73），各车对最佳延迟四分位范围 0.9–2.3 s。
#   正常区间 = [min(0, 前车在 t−2.3…t−0.9 s 的最小加速度), max(0, 同期最大加速度)]；
#   只把超出前车的那部分加速度按 周荣贵等(2024) 速度-阈值曲线换算成 z。跟驰判定与 φ1 相同（w<1，THW<4 s）。
#   GT（|a| z≥5）不受影响，仍按原始加速度定义。
FOLLOW_LAG_LO = 27   # 0.9 s
FOLLOW_LAG_HI = 69   # 2.3 s


def compute_phi2(df, trusted, truck, w):
    import accel_thresholds as at
    curves = at.load()
    # v12（P9 分级 + P11）：情境等级（Data.py → CatchCls / ApprCls）决定用哪条曲线，与 GT 一致
    a = df["Acc_from_vsmooth_g"].to_numpy(float)
    ctx = at.frame_ctx(a, df["CatchCls"], df["ApprCls"]) if "CatchCls" in df else None
    grp = at.group_id(truck, df["Lane_ctx"] if "Lane_ctx" in df else df["Lane_Num"], ctx)   # v13：分流段驶出车按匝道组
    v = df["Speed_smooth"].to_numpy(float)
    z_raw = at.zscore(curves, v, a, grp)
    a_lo = np.full(len(df), np.nan); a_hi = np.full(len(df), np.nan)
    if "Leader_ID" in df:
        acc = pd.Series(np.where(trusted.to_numpy(), a, np.nan), index=df.index)
        span = FOLLOW_LAG_HI - FOLLOW_LAG_LO + 1
        gb = acc.groupby(df["Vehicle_ID"])
        r_hi = gb.transform(lambda s: s.rolling(span, min_periods=span // 2).max())
        r_lo = gb.transform(lambda s: s.rolling(span, min_periods=span // 2).min())
        key = pd.MultiIndex.from_arrays([df["Vehicle_ID"].astype(int), df["Frame_ID"].astype(int)])
        H = pd.Series(r_hi.to_numpy(), index=key).groupby(level=[0, 1]).first()
        Lo = pd.Series(r_lo.to_numpy(), index=key).groupby(level=[0, 1]).first()
        lk = pd.MultiIndex.from_arrays([df["Leader_ID"].fillna(-1).astype(int),
                                        df["Frame_ID"].astype(int) - FOLLOW_LAG_LO])
        a_hi = H.reindex(lk).to_numpy(); a_lo = Lo.reindex(lk).to_numpy()
    use = np.isfinite(a_hi) & np.isfinite(a_lo) & (np.asarray(w) < 1)
    hi = np.maximum(np.nan_to_num(a_hi), 0); lo = np.minimum(np.nan_to_num(a_lo), 0)
    a_ex = np.where(use, a - np.clip(a, lo, hi), a)
    z = at.zscore(curves, v, a_ex, grp)
    z = np.where(a_ex == 0, 0.0, z)
    # v14（用户确认 2026-11-05）：物理锚点 —— 本车加速度未超出正常驾驶物理范围（AASHTO 3.4 m/s² 减速；
    #   Long 2000 起步加速 小汽车 2.2 / 货车 1.5 m/s²）时不算加减速异常（φ2 与 3σ 证据均为 0）。
    #   v12/v13 复核：0.13–0.25 g 的起步、变道提速、驶出减速虽超过情境 5σ 线，看起来都是普通驾驶。
    pe = at.phys_exceed(a, truck)
    print(f"φ2 物理锚点：超出的可信帧 {np.mean(pe[trusted.to_numpy()]):.3%}；情境 3σ 离群帧中超出物理锚点的 "
          f"{np.mean(pe[trusted.to_numpy() & (np.abs(np.nan_to_num(z)) >= Z_OUT)]):.1%}")
    z_ctx = np.abs(np.nan_to_num(z))      # 门槛前（只扣情境/前车）—— 供 GT 主口径"跟车同步"判定，口径不随物理门槛变化
    z = np.where(pe, z, 0.0)
    # v16（用户确认 2026-11-05）：减速侧改用 GT 同口径 z = min(周荣贵曲线 z, 全局 z)（Data.py → Acc_zgt），
    #   不扣前车响应、不加文献物理门槛 —— 车队连锁急刹（跟着前车刹）也计入，模型输入能看到全部 GT 急减速。
    #   加速侧不变（情境扣除 + 物理门槛）。
    if "Acc_zgt" in df:
        n_before = int(np.sum((phi_from_z(np.abs(z)) >= 1)[trusted.to_numpy() & (a < 0)]))
        z = np.where(a < 0, np.maximum(df["Acc_zgt"].to_numpy(float), 0.0), z)
        n_after = int(np.sum((phi_from_z(np.abs(z)) >= 1)[trusted.to_numpy() & (a < 0)]))
        print(f"φ2 v16 减速侧改用 GT 同口径 z：减速 φ2=1 的可信帧 {n_before} → {n_after}")
    phi = np.nan_to_num(phi_from_z(np.abs(z)), nan=0.0)
    t = trusted.to_numpy()
    o_raw = phi_from_z(np.abs(z_raw)) >= phi_from_z(Z_OUT); o_new = phi >= phi_from_z(Z_OUT)
    print(f"φ2: 跟驰且前车加速度可用 {use[t].mean():.1%} 帧；3σ 离群帧 {o_raw[t].mean():.3%} → {o_new[t].mean():.3%}"
          f"（按前车响应扣除后）；φ2=1 {np.mean(phi[t] >= 1):.3%}")
    return phi, np.abs(np.nan_to_num(z)), {"phi2_follow_share": float(use[t].mean()), "_z_ctx": z_ctx}


# ───────────────────────── φ3 ─────────────────────────
def _centre_interps(df, ybin=50.0, min_n=30):
    cs = {}
    for lane, g in df.groupby("Lane_Num"):
        b = np.floor(g["Local_Y"] / ybin)
        st = g.groupby(b)["Local_X"].agg(["median", "size"])
        st = st[st["size"] >= min_n]
        if len(st) == 0:
            cs[lane] = (np.array([0.0, 1e6]), np.repeat(g["Local_X"].median(), 2))
        else:
            cs[lane] = ((st.index.to_numpy(float) + 0.5) * ybin, st["median"].to_numpy(float))
    return cs


def _centre(cs, lane, y, margin=50.0):
    """车道中心线；超出有数据覆盖的纵向范围（±margin）返回 NaN —— 不外推。
    （v7 抽查 #1：匝道末端车辆稀少，中心线被平推，弯道本身被算成横向偏移，φ3 饱和。）"""
    out = np.full(len(y), np.nan)
    lane = np.asarray(lane, float); y = np.asarray(y, float)
    for L, (yc, xc) in cs.items():
        m = (lane == L) & (y >= yc[0] - margin) & (y <= yc[-1] + margin)
        if m.any():
            out[m] = np.interp(y[m], yc, xc)
    return out


def compute_phi3(df, trusted, truck):
    n = len(df)
    vid = df["Vehicle_ID"].to_numpy(); ln = df["Lane_Num"].to_numpy(float)
    x = df["Local_X"].to_numpy(float); y = df["Local_Y"].to_numpy(float)
    cs = _centre_interps(df)
    resid = x - _centre(cs, ln, y)
    sw = _lane_switches(df)
    in_lc, _ = _near_switch(df, sw, int(LC_WIN_S * FPS))

    # (a) 车道保持（v8）：车道内漂移属正常，只有"压线却没有完成变道"才计分。
    #   起点 = 正常车道保持 |偏移| 的 P99（车道×车型×500 ft 路段，变道过程外）；
    #   终点 = 半车道宽（车身中心到车道线；用每次车道号切换前一帧相对原车道中心的 |偏移| 中位数标定）。
    #   旧版用稳健 σ（≈0.8 ft）：偏移分布尾部很厚，车道内平缓漂移 3–4 ft 就到 φ3≈0.7，30% 的帧 φ3>0。
    abs_r = pd.Series(np.abs(resid), index=df.index)
    ok_keep = trusted & pd.Series(~in_lc, index=df.index)
    # 半车道宽只用主线车道之间（1/2/3）的切换标定；匝道分流处车道线位置与中心线 +半宽 不对应。
    main = sw[(sw.frm >= 1) & (sw.to >= 1)]
    kk = main.k.to_numpy() - 1
    d0 = np.abs(x[kk] - _centre(cs, main.frm.to_numpy(float), y[kk]))
    hw = float(np.nanmedian(d0)) if np.isfinite(d0).sum() >= 20 else 6.0
    p99 = _seg_stat(df, abs_r, ok_keep, lambda a: np.percentile(a[np.isfinite(a)], 99), truck=truck,
                    lo=0.3, hi=None, label="车道保持 |偏移| P99 ft").to_numpy()
    # 正常 P99 已接近半车道宽的路段（分流渐变段、匝道），中心线本身不可靠 → 不计车道保持（变道/蛇行项照常）
    usable = (hw - p99) >= 1.0
    keep = np.where(usable, np.clip((np.abs(resid) - p99) / np.maximum(hw - p99, 1.0), 0.0, 1.0), 0.0)
    keep = np.where(in_lc, 0.0, np.nan_to_num(keep, nan=0.0))
    p3s = _seg_stat(df, abs_r, ok_keep, lambda a: np.percentile(a[np.isfinite(a)], 100 * 0.99865), truck=truck,
                    lo=0.3, hi=None).to_numpy()     # 3σ 稀有度（单侧 P99.865）
    out_keep = usable & ~in_lc & (np.abs(np.nan_to_num(resid, nan=0.0)) >= p3s)
    p2s = _seg_stat(df, abs_r, ok_keep, lambda a: np.percentile(a[np.isfinite(a)], 100 * 0.97725), truck=truck,
                    lo=0.3, hi=None).to_numpy()     # 2σ 稀有度（单侧 P97.725）
    ar = np.abs(np.nan_to_num(resid, nan=0.0))
    z_keep = np.where(usable & ~in_lc, np.where(ar >= p3s, 3.0, np.where(ar >= p2s, 2.0, 0.0)), 0.0)
    t_ = trusted.to_numpy()
    print(f"φ3 半车道宽（主线变道切换前一帧 |偏移| 中位数）{hw:.2f} ft（{len(main)} 次）；"
          f"车道保持不可用（P99 距半车道宽 < 1 ft）的帧 {np.mean(~usable[t_]):.1%}")

    # (b) 横向速度：t 与 t−K 都相对 t 时刻车道的中心线
    k = LAT_K
    prev = np.arange(n) - k
    valid = (prev >= 0)
    prev = np.clip(prev, 0, n - 1)
    valid &= vid[prev] == vid
    r_prev = x[prev] - _centre(cs, ln, y[prev])
    vlat = np.where(valid, np.abs(resid - r_prev) / (k / FPS), np.nan)
    # 参照：每次变道过程中的峰值横向速度
    peaks = []
    for r in sw.itertuples():
        a = max(r.k - int(LC_WIN_S * FPS), 0); b = min(r.k + int(LC_WIN_S * FPS), n)
        idx = np.arange(a, b); idx = idx[(vid[idx] == r.vid) & trusted.to_numpy()[idx]]
        if len(idx) and np.isfinite(vlat[idx]).any():
            peaks.append(np.nanmax(vlat[idx]))
    peaks = np.asarray(peaks)
    med_p, s_p = np.median(peaks), robust_sigma(peaks)
    z_lat = np.nan_to_num((vlat - med_p) / s_p + 1.0, nan=0.0)
    lat = np.nan_to_num(phi_from_z(z_lat), nan=0.0)  # 一般变道峰值 → z=1 → 0
    lat = np.where(usable, lat, 0.0)   # 中心线不可靠路段（匝道弯道/分流渐变段）横向速度同样不可信（v7 抽查 #1）
    print(f"φ3 变道峰值横向速度: 中位数 {med_p:.2f} ft/s，σ {s_p:.2f} ft/s（{len(peaks)} 次变道）→ "
          f"φ=1 对应 {med_p + (Z_SAT - 1) * s_p:.2f} ft/s")

    # (c) 蛇行 A→B→A
    rev = []
    swl = list(sw.itertuples())
    for i in range(len(swl) - 1):
        a, b = swl[i], swl[i + 1]
        if a.vid != b.vid or b.to != a.frm or b.frm != a.to:
            continue
        dwell = b.k - a.k
        if dwell < WEAVE_MIN_DWELL:
            continue
        idx = np.arange(a.k, b.k)
        exc = np.abs(x[idx] - _centre(cs, np.full(len(idx), a.frm), y[idx]))
        if not np.isfinite(exc).any() or np.nanmax(exc) < WEAVE_MIN_DX_FT:
            continue
        rev.append((a.k, b.k, dwell / FPS))
    weave = np.zeros(n)
    if rev:
        dts = np.array([r[2] for r in rev])
        t_lo, t_hi = np.percentile(dts, 5), np.percentile(dts, 25)
        for ka, kb, dt in rev:
            val = np.clip((t_hi - dt) / max(t_hi - t_lo, 1e-6), 0, 1)
            lo_, hi_ = max(ka - WEAVE_PAD, 0), min(kb + WEAVE_PAD, n)
            idx = np.arange(lo_, hi_); idx = idx[vid[idx] == vid[ka]]
            weave[idx] = np.maximum(weave[idx], val)
        print(f"φ3 蛇行: 有效 A→B→A {len(rev)} 次，间隔 P5 {t_lo:.1f}s（→1）、P25 {t_hi:.1f}s（→0）；"
              f"φ>0 的回头变道 {(dts < t_hi).sum()} 次")
    phi = np.maximum.reduce([keep, lat, weave])
    t = trusted.to_numpy()
    print(f"φ3: 车道保持>0 {np.mean(keep[t] > 0):.2%}，横移速度>0 {np.mean(lat[t] > 0):.2%}，蛇行>0 {np.mean(weave[t] > 0):.2%}；"
          f"φ3=1 {np.mean(phi[t] >= 1):.3%}")
    out = out_keep | (lat >= phi_from_z(Z_OUT)) | (weave > 0)   # 3σ 离群：压线超 3σ、横移速度 ≥3σ、回头变道
    print(f"φ3: 3σ 离群帧 {np.mean(out[t]):.3%}")
    return phi, {"lc_peak_vlat_med": med_p, "lc_peak_vlat_sigma": s_p, "_parts": (keep, lat, weave), "_out": out,
                 "_z": np.maximum.reduce([z_keep, np.where(usable, z_lat, 0.0), np.where(weave > 0, 99.0, 0.0)])}


# ───────────────────────── φ4 侧向标定 ─────────────────────────
def calibrate_lateral(df, step=15, default_len=15.0, default_wid=6.0):
    """相邻车道、纵向车身重叠（并排）车辆的车身边缘间距分布 → (lat_safe=P1, k_lat 使 P5 处 φ=0.1)"""
    d = df[df["Frame_ID"] % step == 0]
    gaps = []
    for _, g in d.groupby("Frame_ID"):
        if len(g) < 2:
            continue
        x = g["Local_X"].to_numpy(float); y = g["Local_Y"].to_numpy(float)
        w = g["Width"].fillna(default_wid).to_numpy(float); l = g["Length"].fillna(default_len).to_numpy(float)
        L = g["Lane_Num"].to_numpy(float)
        adx = np.abs(y[:, None] - y[None, :]) - (l[:, None] + l[None, :]) / 2
        ady = np.abs(x[:, None] - x[None, :]) - (w[:, None] + w[None, :]) / 2
        m = (adx < 0) & (np.abs(L[:, None] - L[None, :]) == 1)
        np.fill_diagonal(m, False)
        gaps.append(ady[m])
    g = np.concatenate(gaps)
    p1, p5, p50 = np.percentile(g, [1, 5, 50])
    k = np.log(9.0) / max(p5 - p1, 1e-3)
    print(f"φ4 侧向标定: 并排车身边缘间距 P1 {p1:.2f} / P5 {p5:.2f} / P50 {p50:.2f} ft（{len(g)} 对）→ "
          f"lat_safe={p1:.2f} ft, k_lat={k:.2f}")
    return float(p1), float(k)


# ───────────────────────── 稀有度 z（偏斜/厚尾量用）─────────────────────────
# 对 TTC、侧向间距这类分布极度偏斜的量，均值±σ 没有意义。改用"稀有度"：
#   z = 使 P(正常参照 ≤ x) 等于正态分布单侧尾概率 sf(z) 的那个 z
# 即"x 在正常驾驶中出现得和正态 zσ 以外一样少"。z ≤ Z_OUT 的范围样本充足，直接用经验分布；
# 更极端处样本不足，用 1σ–3σ 区间上 z 与 x 的线性关系外推。
Z_OUT = 3.0   # 拉依达 3σ 准则：|z| ≥ 3 视为离群（训练集清洗与事件核查共用）


class RarityZ:
    """经验分布可靠的范围（比 x 更极端的参照样本 ≥ MIN_TAIL_N 个，尾概率相对标准误 ≤ 10%）直接用经验分布；
    更极端处用可靠范围最后 1σ 内 z 与 x 的线性关系外推（与经验段连续）。"""
    MIN_TAIL_N = 100

    def __init__(self, ref, name=""):
        from scipy.stats import norm
        self.norm = norm
        s = np.sort(np.asarray(ref, float)[np.isfinite(ref)])
        self.s, self.n = s, len(s)
        self.z_lim = float(norm.isf(self.MIN_TAIL_N / self.n))
        zz = np.linspace(self.z_lim - 1.0, self.z_lim, 21)
        xx = np.quantile(s, norm.sf(zz))
        self.b, self.a = np.polyfit(xx, zz, 1)
        self.x_lim = float(np.quantile(s, norm.sf(self.z_lim)))
        self.a = self.z_lim - self.b * self.x_lim          # 在 x_lim 处与经验段连续
        self.x1, self.x3 = float(np.quantile(s, norm.sf(1.0))), float(np.quantile(s, norm.sf(Z_OUT)))
        print(f"  稀有度 z [{name}]: 参照 {self.n} 个；1σ ↔ {self.x1:.2f}，3σ ↔ {self.x3:.2f}；"
              f"经验分布可靠至 {self.z_lim:.2f}σ（{self.x_lim:.2f}），之后外推，5σ ↔ {(5 - self.a) / self.b:.2f}")

    def __call__(self, x):
        x = np.asarray(x, float)
        F = np.searchsorted(self.s, x, side="right") / self.n
        z = self.norm.isf(np.clip(F, 1.0 / self.n, 1.0))
        z = np.where(x < self.x_lim, np.maximum(self.a + self.b * x, self.z_lim), z)
        return np.where(np.isfinite(x), z, -np.inf)


REAL_W_CAR, REAL_W_TRUCK = 6.0, 8.5   # ft：真实车宽（小汽车典型宽度；美国联邦货车宽度上限 102 in）


def _side_gap(df, default_len=15.0, drop_overlap=True, verbose=True):
    """每帧本车与相邻车道、纵向车身重叠（并排）车辆的最小车身侧向净距（ft）；无并排车 → NaN。
    数据的 Width 是外接框（小汽车 6–12 ft、货车 8–19 ft，货车中位 13 ft 已超过车道宽 11.7 ft），
    用它算净距会出现大量负值（P1 = −0.15 ft）。改用真实车宽后负值仅 0.02%，P0.1 ≈ 1 ft，物理上合理。"""
    order = np.argsort(df["Frame_ID"].to_numpy(), kind="stable")
    fr = df["Frame_ID"].to_numpy()[order]
    x = df["Local_X"].to_numpy(float)[order]; y = df["Local_Y"].to_numpy(float)[order]
    tk = df["is_truck"].fillna(0).to_numpy()[order] if "is_truck" in df else np.zeros(len(df))
    w = np.where(tk == 1, REAL_W_TRUCK, REAL_W_CAR)
    l = df["Length"].fillna(default_len).to_numpy(float)[order]
    L = df["Lane_Num"].to_numpy(float)[order]
    out = np.full(len(df), np.nan)
    nb = np.full(len(df), -1, np.int64)
    vids = df["Vehicle_ID"].to_numpy()[order].astype(np.int64)
    bounds = np.flatnonzero(np.diff(fr)) + 1
    for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(fr)]):
        if b - a < 2:
            continue
        adx = np.abs(y[a:b, None] - y[None, a:b]) - (l[a:b, None] + l[None, a:b]) / 2
        ady = np.abs(x[a:b, None] - x[None, a:b]) - (w[a:b, None] + w[None, a:b]) / 2
        m = (adx < 0) & (np.abs(L[a:b, None] - L[None, a:b]) == 1)
        np.fill_diagonal(m, False)
        G = np.where(m, ady, np.inf)
        j = G.argmin(1); g = G[np.arange(b - a), j]
        out[a:b] = np.where(np.isfinite(g), g, np.nan)
        nb[a:b] = np.where(np.isfinite(g), vids[a:b][j], -1)
    res = np.empty(len(df)); res[order] = out
    nbr = np.empty(len(df), np.int64); nbr[order] = nb
    # v15（2026-11-05）：并排车身重叠 —— 同一对并排车辆的一段连续并排中出现净距 ≤ 0（两车车身重叠，物理上不可能），
    #   说明这一对的横向位置/车宽误差已超过净距本身，整段侧向净距记为缺失（与 Data.py 前车 Leader_overlap 同一规则）。
    #   v14 复核：φ4 = 1 而无可见冲突的事件多为此类（净距 −0.2 … −1.7 ft，换道途中）。
    if drop_overlap:
        vid = df["Vehicle_ID"].to_numpy()
        fid = df["Frame_ID"].to_numpy()
        o2 = np.lexsort((fid, vid))
        v_s, n_s, r_s = vid[o2], nbr[o2], res[o2]
        f_s = fid[o2]
        new_run = np.r_[True, (v_s[1:] != v_s[:-1]) | (n_s[1:] != n_s[:-1]) | (np.diff(f_s) != 1)]
        rid = np.cumsum(new_run)
        bad = pd.Series(np.nan_to_num(r_s, nan=np.inf) <= 0).groupby(rid).transform("any").to_numpy() & (n_s >= 0)
        r_s = np.where(bad, np.nan, r_s)
        res2 = np.empty(len(df)); res2[o2] = r_s
        if verbose:
            print(f"侧向净距：并排车身重叠（净距 ≤ 0）的整段记为缺失 {bad.mean():.3%} 帧（重叠帧 {int((np.nan_to_num(res, nan=1) <= 0).sum())}）")
        res = res2
    return res


TTC_SAFE = 3.0   # s：文献中"正常跟驰/交会"的上界（瑞典冲突技术 >3 s 为正常交会；综述推荐阈值 3 s）


THW_BIN_KMH = 10.0          # 与 accel_thresholds.BIN_KMH 相同
THW_MIN_BIN = 2000          # 与 accel_thresholds.MIN_BIN_FRAMES 相同
THW_SUSTAIN = FPS           # 1 s
THW_S0_FT = 2.0 / 0.3048    # IDM 静止最小净距 s0 = 2 m（Treiber, Hennecke & Helbing 2000）


def _thw_term(df, t):
    """→ (φ_thw, z_thw)：持续 1 s 的车头时距按速度分箱的稀有度。"""
    n = len(df)
    v = df["Speed_smooth"].to_numpy(float)
    gp = df["Gap_bumper"].to_numpy(float) if "Gap_bumper" in df else np.full(n, np.nan)
    thw = np.where(np.isfinite(gp) & (gp > 0) & (v > 0), gp / np.maximum(v, 1e-6), np.nan)
    if "TTC_cleared" in df:
        thw = np.where(df["TTC_cleared"].fillna(0).to_numpy() == 1, np.nan, thw)
    s = pd.Series(thw, index=df.index)
    sus = s.groupby(df["Vehicle_ID"]).transform(
        lambda x: x.rolling(THW_SUSTAIN, min_periods=THW_SUSTAIN).max()).to_numpy()
    width = THW_BIN_KMH * 0.911344
    b = np.floor(v / width)
    ok = t & np.isfinite(sus)
    cnt = pd.Series(b[ok]).value_counts().sort_index()
    # 样本不足的箱并入相邻（向低速方向累积，直到 ≥ THW_MIN_BIN）
    edges, acc = [], 0
    for k, c in cnt.items():
        acc += c
        if acc >= THW_MIN_BIN:
            edges.append(k); acc = 0
    edges = np.array(edges if edges else [cnt.index.max()], float)
    grp = np.searchsorted(edges, np.where(np.isfinite(b), b, edges[-1]), side="left")
    grp = np.minimum(grp, len(edges) - 1)
    z = np.full(n, -np.inf); n_gate = 0
    for gi in np.unique(grp):
        sel = grp == gi
        ref = sus[sel & ok]
        if len(ref) < THW_MIN_BIN:
            continue
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            rz = RarityZ(ref, "THW")
        z[sel] = rz(sus[sel])
        # v12（用户确认 2026-11-04）：低速门槛 —— 该速度下 THW 的 3σ 对应净距（x3·v）小于静止最小净距
        #   s0 = 2 m（IDM，Treiber et al. 2000）时，"稀有的短时距"其实只是排队停走的正常车距，不作 THW 证据。
        #   v11 复核：v<30 ft/s 只由 THW 报出的 4 个事件全为排队正常贴近（净距 4–8 ft）。
        low = sel & (rz.x3 * v < THW_S0_FT)
        z[low] = 0.0
        n_gate += int((low & ok).sum())
    print(f"φ4（P10 低速门槛）: 3σ 净距 < s0={THW_S0_FT:.2f} ft 的帧 THW 项置 0：{n_gate} 帧（可信帧 {n_gate / max(ok.sum(), 1):.2%}）")
    z = np.where(np.isfinite(z), z, 0.0)
    phi = np.nan_to_num(phi_from_z(z), nan=0.0)
    print(f"φ4（P10）: THW 项 {len(edges)} 个速度箱；持续 1 s THW 的 3σ 离群帧 {np.mean((z >= Z_OUT)[t]):.3%}，"
          f"φ=1（5σ）{np.mean((phi >= 1)[t]):.3%}")
    for vb in (30, 60, 90, 120):   # 各速度下 3σ 对应的 THW（便于核对）
        sel = ok & (np.abs(v - vb) < 5)
        if sel.sum() > 1000:
            print(f"    v≈{vb} ft/s: 3σ ↔ THW {np.quantile(sus[sel], 0.00135):.2f} s，中位 {np.median(sus[sel]):.2f} s")
    return phi, z


def compute_phi4(df, trusted, ttc_gt=1.5, ttc_safe=TTC_SAFE):
    """φ4 碰撞风险 = max(纵向, 侧向)。强度用物理锚点、达到危险值即饱和为 1（极端情形样本少，不再细分）：
       纵向：同车道前车 TTC（GT 所用）。φ=1：TTC ≤ 1.5 s（严重冲突，van der Horst；FHWA SSAM，即 GT 标准）；
             φ=0：TTC ≥ 3 s（文献中正常跟驰/交会的上界）；中间对 TTC 线性。
             数据交叉验证：TTC 的 3σ 稀有度对应 2.41 s，落在 1.5–3 s 之间。
       侧向：与并排车辆的真实车身净距。φ=1：净距 ≤ 0（擦碰）；φ=0：≥ 正常 1σ 稀有度对应净距；中间线性。
       3σ 离群标记（训练清洗/核查用）仍按稀有度：TTC 或净距的稀有度 ≥ 3σ。"""
    t = trusted.to_numpy()
    ttc = df["TTC"].to_numpy(float) if "TTC" in df else np.full(len(df), np.nan)
    ttc = np.where(np.isfinite(ttc) & (ttc > 0), ttc, np.nan)
    zt = RarityZ(ttc[t], "TTC s")
    T1 = ttc_safe
    long_ = np.clip((T1 - ttc) / (T1 - ttc_gt), 0.0, 1.0)
    gap = _side_gap(df)
    zg = RarityZ(gap[t], "侧向净距 ft")
    G1 = zg.x1
    lat = np.clip((G1 - gap) / G1, 0.0, 1.0)
    # v11（P10，用户确认 2026-11-04）：车头时距（THW）项 —— 高速贴车跟驰时两车同速、TTC→∞，纵向 TTC 项为 0，
    #   φ1（同速在正常区间内）也为 0；v10 复核 #37（135 ft/s 下约 30 ft）即此类。
    #   THW = 净距 / 本车速度；持续性：取 1 s（与事件最短持续时间相同）滚动最大值，变道穿越等瞬时贴近不计；
    #   稀有度按速度分箱（10 km/h，与加速度阈值曲线相同；样本 < MIN_BIN_FRAMES 的箱并入相邻箱）分别估计，
    #   排队低速时的小间距不算稀有；强度用统一标度 φ = (z−1)/4（3σ 离群、5σ 饱和）。
    #   P8 判为"横向错开中"（TTC_cleared）的帧不计。
    thw_p, z_thw = _thw_term(df, t)
    phi = np.nan_to_num(np.fmax(np.fmax(long_, lat), thw_p), nan=0.0)
    z4 = np.fmax(np.fmax(zt(ttc), zg(gap)), z_thw); z4 = np.where(np.isfinite(z4), z4, 0.0)
    out = z4 >= Z_OUT
    print(f"φ4: 纵向 TTC ≥ {T1:.1f}s → 0，≤ {ttc_gt}s → 1（线性）；侧向净距 ≥ {G1:.2f}ft → 0，≤ 0 → 1；"
          f"φ4>0 {np.mean(phi[t] > 0):.2%}，φ4=1 {np.mean(phi[t] >= 1):.3%}；3σ 离群帧 {np.mean(out[t]):.3%}"
          f"（TTC ≤ {zt.x3:.2f}s 或净距 ≤ {zg.x3:.2f}ft）")
    return phi, (out, z4), {"ttc_phi0": T1, "ttc_phi1": ttc_gt, "ttc_at_z3": zt.x3,
                      "gap_phi0": G1, "gap_at_z3": zg.x3}
