"""
report_rule.py —— 报告规则（v9b，用户确认 2026-11-04）

检测事件 = SPE 超过本类控制限、连续 ≥1 s 的帧段（模型给出"候选"）。
报告为"异常驾驶事件"还需要：事件内至少一帧在某一维度达到 3σ（o1–o4 离群标记，3σ 拉依达准则），
且该帧不在振铃修复区、近距离证据不在前车切入后 1.5 s 内（v10，见 evidence_cols）。
只超 SPE、全程 <3σ 的候选（正常范围内的小波动）不报告，另存为 candidates 备查。

所有方法（本文方法与各基线）用同一条报告规则；REPORT_RULE=none 可关闭（只看模型本身）。
"""
import os
import numpy as np
import pandas as pd

O_COLS = ["o1", "o2", "o3", "o4"]
JOINT, JOINT_THR = None, None


# v10（P6，用户确认 2026-11-04）：轨迹末端修复区。2128/2182 条轨迹以振铃修复区结束（下游观测边界，
# 修复为端点保持的平直速度）；修复区之前一段的速度是"真实值→保持值"的过渡，复核中 A 组 7 个数据误报有 6 个落在这里。
# 范围由数据确定（tools/tail_repair_anchor.py）：以末端修复区起点前 3–10 s 的 3σ 帧率为远处参照，
# 每 0.5 s 分箱的 3σ 帧率在 0–2.0 s 内为参照的 1.13–1.74 倍，2.0–2.5 s 起回到参照水平（0.95） → 前推 2.0 s。
# 末端修复区及其前 2.0 s 的帧不作为证据（同 dq）；只影响报告规则与评估，不影响特征与训练。
# v12：速度阶跃修复（v11）后重算锚点 → 0 s（各 0.5 s 分箱 3σ 帧率均回到参照水平，见 runs/v11/anchor_tail.txt），
#   按"阈值须有数据锚点"关闭前推（只保留末端修复区本身不作证据，与 dq 相同）。
TAIL_PRE_FRAMES = 0


def add_tail(df):
    """加列 tail_rz：每辆车最后一段 dq（延伸到该车最后一帧）及其前 TAIL_PRE_FRAMES 帧 = 1。须传入整车数据。"""
    d = df[["Vehicle_ID", "Frame_ID"]].copy()
    d["dq"] = df["dq"].fillna(0).astype(int).to_numpy() if "dq" in df else 0
    d = d.sort_values(["Vehicle_ID", "Frame_ID"])
    g = d.groupby("Vehicle_ID")
    # 从车尾往前：dq 连续为 1 的末端段
    rev_break = (d["dq"] == 0).astype(int)[::-1].groupby(d["Vehicle_ID"][::-1]).cumsum()[::-1]
    in_tail_rz = rev_break == 0
    start = d["Frame_ID"].where(in_tail_rz).groupby(d["Vehicle_ID"]).transform("min")
    tail = start.notna() & (d["Frame_ID"] >= start - TAIL_PRE_FRAMES)
    out = df.copy()
    out["tail_rz"] = tail.reindex(df.index).astype(int).to_numpy()
    return out


def evidence_cols(df):
    """逐帧、逐维度的"可作证据"的 3σ 标记（v10）：
    - 振铃修复区（dq=1）的帧：任何维度都不作为证据（速度是重建值，边界处有阶跃）
    - 前车切入后 1.5 s 内（cutin=1）：近距离 o4 视为他车引起，不作为本车证据
    - 轨迹末端修复区及其前 2.0 s（tail_rz，P6；TAIL_MASK=0 关闭）：任何维度都不作为证据
    注意：tail_rz 须在整车数据上计算——传入切片前先调用 add_tail。
    返回与 O_COLS 同名的 0/1 DataFrame。"""
    E = df[[c for c in O_COLS if c in df.columns]].fillna(0).astype(int).copy()
    dq = df["dq"].fillna(0).astype(int) if "dq" in df else 0
    E = E.mul(1 - dq, axis=0) if "dq" in df else E
    if os.environ.get("TAIL_MASK", "1") == "1":
        if "tail_rz" not in df and {"Vehicle_ID", "Frame_ID", "dq"} <= set(df.columns):
            df = add_tail(df)
        if "tail_rz" in df:
            E = E.mul(1 - df["tail_rz"].fillna(0).astype(int), axis=0)
    if "o4" in E and "cutin" in df:
        E["o4"] = E["o4"] * (1 - df["cutin"].fillna(0).astype(int))
    return E


def rule_on():
    return os.environ.get("REPORT_RULE", "on").lower() != "none"   # v16（用户确认 2026-11-05）：用 5σ 证据规则（REPORT_LEVEL 默认 5）；REPORT_RULE=none 看模型全部输出


def load_oflags(path="phi_all.csv"):
    """每辆车：排好序的帧号 + "任一维度 3σ 离群"的累计和，用于 O(log n) 判断区间内是否有证据帧。"""
    P = ["phi1", "phi2", "phi3", "phi4"]
    Zc = ["z1", "z2", "z3", "z4"]
    df = pd.read_csv(path, usecols=lambda c: c in ["Vehicle_ID", "Frame_ID", "dq", "cutin"] + O_COLS + P + Zc)
    if os.environ.get("TAIL_MASK", "1") == "1" and "dq" in df:
        df = add_tail(df)
    ev = evidence_cols(df)
    # v15（2026-11-05，目标：报告事件大部分为真异常）：报告证据须达到 5σ —— 证据帧中该维度 φ = 1（统一标度 5σ 饱和，
    #   与 GT 的 5σ 同一锚点）。v14 全量复核：只有 3σ 证据的事件 98 个中真异常 5 个（5%）；有 5σ 证据的 36 个中 12 个（33%）。
    #   REPORT_LEVEL=3 恢复旧规则（任一维度 3σ）。
    if os.environ.get("REPORT_LEVEL", "5") == "5" and all(p in df for p in P):
        strong = np.column_stack([(df[p].to_numpy() >= 1 - 1e-9) for p in P])
        ev = ev & strong[:, [O_COLS.index(c) for c in ev.columns]]
        sat = (ev.max(axis=1) > 0).to_numpy()
        # v16b（用户确认 2026-11-05）：事件确认 = 单维饱和 或 综合饱和。单维都未饱和、但几个维度同时偏高的组合异常
        #   （v16 复核：蛇行、变道切入慢车队并急减速、急减速逼近前车等，12 个中约 9 个为真异常）也确认。
        #   综合强度 = sqrt(Σ min(z_k,5)²)（各维 z 截断于 5σ，与 φ=1 同一上限；不可信帧不计，同 evidence_cols）。
        #   阈值（数据锚点）：在未单维饱和的可信帧中，取使超阈帧比例等于单维饱和帧比例的分位数 → 综合饱和与单维饱和同样罕见。
        #   COMBINED_CONFIRM=0 关闭（只用单维饱和）。
        if os.environ.get("COMBINED_CONFIRM", "1") == "1" and all(z in df for z in Zc):
            trust = np.ones(len(df), bool)
            if "dq" in df:
                trust &= df["dq"].fillna(0).to_numpy() == 0
            if "tail_rz" in df:
                trust &= df["tail_rz"].fillna(0).to_numpy() == 0
            Zr = df[Zc].fillna(0).to_numpy(float)
            # v16b 修正（2026-11-05 复核）：φ3 蛇行（A→B→A 回头变道）在 z3 中记为占位值 99（= "≥3σ 离群"标记），
            #   其强度已按回头间隔分级在 φ3 里。截断成 5 会把每次回头变道都当 5σ，平稳的"变出又变回"也被确认
            #   （V1258/V1362/V918 等）。改用 φ3 对应的等价 z：z = 1 + (Z_SAT−1)·φ3（utility_phi.phi_from_z 的反函数）。
            if "phi3" in df:
                w = Zr[:, 2] >= 98
                Zr[w, 2] = 1.0 + 4.0 * df["phi3"].to_numpy(float)[w]
            Z = np.minimum(np.clip(Zr, 0, None), 5.0)
            if "cutin" in df:
                Z[:, 3] *= 1 - df["cutin"].fillna(0).to_numpy()
            Z[~trust] = 0
            comb = np.sqrt((Z ** 2).sum(1))
            rate = sat[trust].mean()
            thr = float(np.quantile(comb[trust & ~sat], 1 - rate))
            csat = trust & (comb >= thr)
            global JOINT, JOINT_THR   # 供复核图"证据"面板使用：每帧综合强度、是否单维饱和、阈值
            JOINT = df[["Vehicle_ID", "Frame_ID"]].assign(joint=comb, single_sat=sat.astype(int))
            JOINT_THR = thr
            print(f"[事件确认] 单维饱和帧 {rate:.4%}；综合强度阈值 {thr:.2f}（同频率）；"
                  f"只靠综合饱和的帧 {np.mean(csat & ~sat):.4%}", flush=True)
            df["_o"] = (sat | csat).astype(np.int32)
        else:
            df["_o"] = sat.astype(np.int32)
    else:
        df["_o"] = (ev.max(axis=1) > 0).astype(np.int32)
    df = df.sort_values(["Vehicle_ID", "Frame_ID"])
    return {int(v): (g["Frame_ID"].to_numpy(np.int64), np.concatenate([[0], np.cumsum(g["_o"].to_numpy())]))
            for v, g in df.groupby("Vehicle_ID")}


def has_evidence(events, of):
    if events is None or len(events) == 0:
        return np.zeros(0, bool)
    out = np.zeros(len(events), bool)
    for i, (v, s, e) in enumerate(zip(events["Vehicle_ID"].astype(int), events["start_frame"].astype(int),
                                      events["end_frame"].astype(int))):
        if v not in of:
            continue
        fr, cs = of[v]
        a, b = np.searchsorted(fr, s, "left"), np.searchsorted(fr, e, "right")
        out[i] = cs[b] - cs[a] > 0
    return out


def split_events(events, of):
    """→ (reported, candidates)"""
    if events is None or len(events) == 0:
        return events, events
    k = has_evidence(events, of)
    return events[k].reset_index(drop=True), events[~k].reset_index(drop=True)


def mask_frames(df_frame, reported):
    """只保留落在已报告事件内的超限帧（使逐帧覆盖率与报告的事件一致）。"""
    d = df_frame.copy()
    if reported is None or len(reported) == 0:
        d["is_anom_frame"] = 0
        return d
    keys = pd.MultiIndex.from_arrays([
        np.repeat(reported["Vehicle_ID"].astype(int).to_numpy(),
                  (reported["end_frame"] - reported["start_frame"] + 1).astype(int).to_numpy()),
        np.concatenate([np.arange(s, e + 1) for s, e in zip(reported["start_frame"].astype(int),
                                                           reported["end_frame"].astype(int))])])
    inside = pd.MultiIndex.from_arrays([d["Vehicle_ID"].astype(int), d["Frame_ID"].astype(int)]).isin(keys)
    d["is_anom_frame"] = (d["is_anom_frame"].astype(int) & inside).astype(int)
    return d


# ── GT 口径（v10，用户确认 2026-11-04）──────────────────────────────────────────
# 主口径：急加减速 GT（原始加速度 ±5σ）中，"跟车同步"的不算异常 —— 事件内本车加速度扣掉前车近期
# 加速度范围后（φ2 的 z2，见 utility_phi.compute_phi2）从未达到 3σ（o2 全为 0），即本车只是跟着前车加减速。
# 与 TTC 冲突重叠的事件（gt_type 含 ttc）保留。原口径（全部 GT）同时报告，作敏感性分析。
GT_PATH = "HighSim_events_Anomaly30_merge2s.csv"


def gt_sync_follow(gt, phi_path="phi_all.csv"):
    """→ bool 数组：该 GT 事件是否为"跟车同步急加减速"。"""
    cols = pd.read_csv(phi_path, nrows=0).columns
    oc = "o2ctx" if "o2ctx" in cols else "o2"   # v14：用未加物理门槛的情境 3σ 标记，主口径不随 φ2 门槛变化
    ph = pd.read_csv(phi_path, usecols=["Vehicle_ID", "Frame_ID", oc]).rename(columns={oc: "o2"})
    of = {int(v): (g["Frame_ID"].to_numpy(np.int64), np.concatenate([[0], np.cumsum(g["o2"].fillna(0).to_numpy())]))
          for v, g in ph.sort_values(["Vehicle_ID", "Frame_ID"]).groupby("Vehicle_ID")}
    o2_any = has_evidence(gt, of)
    return (gt["gt_type"].astype(str) == "harsh_accel").to_numpy() & ~o2_any


def load_gt(path=GT_PATH, phi_path="phi_all.csv"):
    """→ (gt_main, gt_orig)。gt_main 去掉跟车同步急加减速；GT_SYNC=keep 时两者相同。"""
    gt = pd.read_csv(path, low_memory=False)
    if os.environ.get("GT_SYNC", "keep").lower() == "keep":   # v16：GT 不再扣"跟车同步"
        return gt, gt
    m = gt_sync_follow(gt, phi_path)
    gt.assign(sync_follow=m.astype(int)).to_csv(path.replace(".csv", "_flagged.csv"), index=False)
    return gt[~m].reset_index(drop=True), gt


# v12（用户确认 2026-11-04）：复核后 GT —— 验证集全部主口径 GT 事件逐个复核（tools/plot_review.py OUT gt），
#   复核为 NORMAL（情境可解释的正常加减速）或 DATA（数据伪影）的事件去掉，只留确定的异常。
#   清单文件 gt_review_exclude.csv（Vehicle_ID,start_frame,end_frame,label,reason）放在工作目录。
GT_EXCLUDE_PATH = "gt_review_exclude.csv"


def load_gt_verified(path=GT_PATH, phi_path="phi_all.csv", exclude_path=GT_EXCLUDE_PATH):
    """→ 复核后 GT（主口径去掉复核为 NORMAL/DATA 的事件）；无清单文件时返回 None。"""
    if not os.path.exists(exclude_path):
        return None
    gt_main, _ = load_gt(path, phi_path)
    ex = pd.read_csv(exclude_path)
    key = set(zip(ex.Vehicle_ID.astype(int), ex.start_frame.astype(int)))
    m = np.array([(int(v), int(s)) in key for v, s in zip(gt_main.Vehicle_ID, gt_main.start_frame)], bool)
    print(f"复核后 GT：主口径 {len(gt_main)} − 复核去掉 {int(m.sum())} = {int((~m).sum())}")
    return gt_main[~m].reset_index(drop=True)


def load_gt_phys(path=GT_PATH, phi_path="phi_all.csv"):
    """v14（用户确认 2026-11-05，方案 2 的第二口径）：物理锚点 GT —— 主口径中 TTC 冲突全部保留；
    急加/减速只保留"情境 5σ 且超出物理锚点（AASHTO 3.4 m/s² 减速；Long 2000 加速 2.2/1.5 m/s²）持续 ≥1 s"的事件。"""
    gt_main, _ = load_gt(path, phi_path)
    if "phys_harsh" not in gt_main.columns:
        return None
    keep = gt_main["gt_type"].str.contains("ttc") | (gt_main["phys_harsh"].fillna(0) == 1)
    return gt_main[keep].reset_index(drop=True)
