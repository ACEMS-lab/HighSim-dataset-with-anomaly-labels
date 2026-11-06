"""
report_rule.py —— 事件确认规则与 GT 读取。

检测事件（候选）= SPE 超过控制限、连续 ≥1 s 的帧段。确认为异常驾驶事件需事件内至少一帧满足：
  - 单维饱和：某维 φ = 1（5σ）；或
  - 综合饱和：综合强度 √Σ min(z_k,5)² ≥ 阈值，阈值使超阈帧比例等于单维饱和帧比例。
修复区帧（dq）、轨迹末端修复区、前车切入后的 φ4 不作证据。未确认的候选另存备查。
所有方法用同一规则。环境变量：REPORT_RULE=none 关闭确认；REPORT_LEVEL=3 改用任一维 3σ 证据；
COMBINED_CONFIRM=0 只用单维饱和；TAIL_MASK=0 不屏蔽末端修复区。
"""
import os
import numpy as np
import pandas as pd

O_COLS = ["o1", "o2", "o3", "o4"]
JOINT, JOINT_THR = None, None


# 末端修复区之前额外不作证据的帧数。数据锚点（tools/tail_repair_anchor.py）：修复区之前各 0.5 s 分箱的
# 3σ 帧率与远处参照无差异，故取 0。只影响事件确认与评估，不影响特征与训练。
TAIL_PRE_FRAMES = 0


def add_tail(df):
    """加列 tail_rz：每辆车末端修复区（延伸到最后一帧的 dq 段）及其前 TAIL_PRE_FRAMES 帧为 1。须传入整车数据。"""
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
    """逐帧、逐维度可作证据的 3σ 标记，返回与 O_COLS 同名的 0/1 DataFrame。
    修复区（dq=1，速度为重建值）和末端修复区（tail_rz）的帧各维都不作证据；
    前车切入后 1.5 s 内（cutin=1）的 o4 视为他车引起，不作证据。
    tail_rz 须在整车数据上计算，传入切片前先调用 add_tail。"""
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
    return os.environ.get("REPORT_RULE", "on").lower() != "none"   # REPORT_RULE=none：不做确认，看模型全部输出


def load_oflags(path="phi_all.csv"):
    """每辆车：排好序的帧号 + 确认证据帧的累计和，用于 O(log n) 判断区间内是否有证据帧。"""
    P = ["phi1", "phi2", "phi3", "phi4"]
    Zc = ["z1", "z2", "z3", "z4"]
    df = pd.read_csv(path, usecols=lambda c: c in ["Vehicle_ID", "Frame_ID", "dq", "cutin"] + O_COLS + P + Zc)
    if os.environ.get("TAIL_MASK", "1") == "1" and "dq" in df:
        df = add_tail(df)
    ev = evidence_cols(df)
    # 单维饱和：证据帧中该维 φ = 1（φ 在 5σ 处饱和，与 GT 的 5σ 同一锚点）。REPORT_LEVEL=3 改用任一维 3σ。
    if os.environ.get("REPORT_LEVEL", "5") == "5" and all(p in df for p in P):
        strong = np.column_stack([(df[p].to_numpy() >= 1 - 1e-9) for p in P])
        ev = ev & strong[:, [O_COLS.index(c) for c in ev.columns]]
        sat = (ev.max(axis=1) > 0).to_numpy()
        # 综合饱和：各维都未饱和、但多维同时偏高的组合异常也确认。综合强度 = √Σ min(z_k,5)²（不可信帧不计）；
        #   阈值取未单维饱和的可信帧中使超阈比例等于单维饱和帧比例的分位数，使两类确认同样罕见。
        if os.environ.get("COMBINED_CONFIRM", "1") == "1" and all(z in df for z in Zc):
            trust = np.ones(len(df), bool)
            if "dq" in df:
                trust &= df["dq"].fillna(0).to_numpy() == 0
            if "tail_rz" in df:
                trust &= df["tail_rz"].fillna(0).to_numpy() == 0
            Zr = df[Zc].fillna(0).to_numpy(float)
            # 蛇行（回头变道）在 z3 中是占位值 99，强度已按回头间隔分级在 φ3 里；直接截断为 5 会把每次回头变道都当 5σ。
            #   改用 φ3 的等价 z = 1 + 4·φ3（utility_phi.phi_from_z 的反函数）。
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
            global JOINT, JOINT_THR   # 供作图使用：每帧综合强度、是否单维饱和、阈值
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
    """每个事件区间内是否有证据帧（of 为 load_oflags 的输出）。"""
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
    """按是否含证据帧把事件分为 (reported, candidates)。"""
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


# ── GT ──────────────────────────────────────────────────────────────────────
# "跟车同步急加减速"：事件内本车加速度扣掉前车近期加速度范围后（z2）从未达 3σ，即只是跟着前车加减速。
# GT_SYNC=keep（默认）时不剔除；否则主口径剔除这类事件，原口径作敏感性分析。
GT_PATH = "HighSim_events_Anomaly30_merge2s.csv"


def gt_sync_follow(gt, phi_path="phi_all.csv"):
    """→ bool 数组：该 GT 事件是否为"跟车同步急加减速"。"""
    cols = pd.read_csv(phi_path, nrows=0).columns
    oc = "o2ctx" if "o2ctx" in cols else "o2"   # 用未加物理门槛的情境 3σ 标记，GT 不随 φ2 门槛变化
    ph = pd.read_csv(phi_path, usecols=["Vehicle_ID", "Frame_ID", oc]).rename(columns={oc: "o2"})
    of = {int(v): (g["Frame_ID"].to_numpy(np.int64), np.concatenate([[0], np.cumsum(g["o2"].fillna(0).to_numpy())]))
          for v, g in ph.sort_values(["Vehicle_ID", "Frame_ID"]).groupby("Vehicle_ID")}
    o2_any = has_evidence(gt, of)
    return (gt["gt_type"].astype(str) == "harsh_accel").to_numpy() & ~o2_any


def load_gt(path=GT_PATH, phi_path="phi_all.csv"):
    """→ (gt_main, gt_orig)。GT_SYNC=keep（默认）时两者相同；否则 gt_main 去掉跟车同步急加减速，并写出 *_flagged.csv。"""
    gt = pd.read_csv(path, low_memory=False)
    if os.environ.get("GT_SYNC", "keep").lower() == "keep":
        return gt, gt
    m = gt_sync_follow(gt, phi_path)
    gt.assign(sync_follow=m.astype(int)).to_csv(path.replace(".csv", "_flagged.csv"), index=False)
    return gt[~m].reset_index(drop=True), gt



def load_gt_phys(path=GT_PATH, phi_path="phi_all.csv"):
    """物理锚点 GT：TTC 冲突全部保留；急加/减速只保留情境 5σ 且超出物理锚点
    （AASHTO 减速 3.4 m/s²；Long 2000 加速 2.2/1.5 m/s²）持续 ≥1 s 的事件。无 phys_harsh 列时返回 None。"""
    gt_main, _ = load_gt(path, phi_path)
    if "phys_harsh" not in gt_main.columns:
        return None
    keep = gt_main["gt_type"].str.contains("ttc") | (gt_main["phys_harsh"].fillna(0) == 1)
    return gt_main[keep].reset_index(drop=True)
