"""
plot_review.py —— 验证集事件的人工复核图，每个事件一张。

用法（在 runs/<run> 目录下）：
    python ../../tools/plot_review.py OUT_DIR [N_PER_GROUP=15|all|gt] [SEED=11]
  数字：每组随机抽样 —— A 仅模型报告、B 仅 3σ 规则报告、C 两者都报告（非 GT）、
        D 退化类改用合并 UCL 后不再报告的事件（需 pred_events_val_q099_pooled.csv）；
  all：全部报告事件（G 与 GT 重叠、N 不在 GT 中）；
  gt：全部 GT 事件（不显示是否检出，避免复核偏向）。
图中上半为物理量（鸟瞰、速度、加速度及 GT 5σ 阈值、横向位置、车距/TTC/侧向净距），
下半为模型量（SPE 与 UCL、φ1–φ4、事件确认的综合强度）。
输出：OUT_DIR/NNN_<group>_veh<id>.png、review_list.csv、index.html
"""
import os
import sys
import html

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import baselines as ef          # noqa: E402
import report_rule as rr          # noqa: E402
import accel_thresholds as at     # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 10, "axes.titlesize": 11, "legend.fontsize": 8.5,
                     "axes.grid": True, "grid.alpha": 0.3})
P = ["phi1", "phi2", "phi3", "phi4"]
PHI_STYLE = [("phi1", "φ1 Speed", "#a6c8e6"), ("phi2", "φ2 Acc", "#f0b46a"),
             ("phi3", "φ3 Lane", "#a9a9a9"), ("phi4", "φ4 Collision", "#7b4fb0")]
DN = ["speed", "accel", "lateral", "proximity"]
GROUP = {"A": "Model only", "B": "3σ rule only", "C": "Both", "D": "Dropped if low-UCL cluster used pooled UCL",
         "G": "Reported, matches GT", "N": "Reported, not in GT", "T": "Ground-truth event"}
PAD = 90  # 事件前后各 3 s


def overlaps(a, b):
    bg = {v: g for v, g in b.groupby("Vehicle_ID")}
    return np.array([(v in bg) and bool(((bg[v].start_frame <= e) & (bg[v].end_frame >= s)).any())
                     for v, s, e in zip(a.Vehicle_ID, a.start_frame, a.end_frame)], bool)


def shade(ax, t0, t1, color="#f4c7c3", label=None):
    ax.axvspan(t0, t1, color=color, alpha=0.45, lw=0, label=label)


def main(out_dir, n=15, seed=11):
    os.makedirs(out_dir, exist_ok=True)
    pf = pd.read_csv("pred_frames_val.csv")
    val = set(pf.Vehicle_ID.unique())
    ev = pd.read_csv("pred_events_val.csv")
    gt = rr.load_gt()[0]   # 主口径 GT（去掉跟车同步急加减速）
    ph = pd.read_csv("phi_all.csv")
    ph = ph[ph.Vehicle_ID.isin(val)].sort_values(["Vehicle_ID", "Frame_ID"]).reset_index(drop=True)
    rr.load_oflags("phi_all.csv")          # 载入事件确认的综合强度（证据面板用）
    if rr.JOINT is not None:
        ph = ph.merge(rr.JOINT, on=["Vehicle_ID", "Frame_ID"], how="left")
    E = rr.evidence_cols(ph)
    ph[[c + "_ev" for c in E.columns]] = E.to_numpy()
    d3 = ph[["Vehicle_ID", "Frame_ID"]].assign(is_anom_frame=(E.max(axis=1) > 0).astype(int).to_numpy(),
                                                frame_spe=0.0)
    r3 = ef.eventize(d3)
    A = ev[~overlaps(ev, r3)]
    B = r3[~overlaps(r3, ev)]
    C = ev[overlaps(ev, r3) & ~overlaps(ev, gt)]
    # D 组：退化类（φ 全 0 窗口过半）改用合并 UCL 后不再报告的非 GT 事件
    D = ev.iloc[0:0]
    if os.path.exists("pred_events_val_q099_pooled.csv"):
        pooled = pd.read_csv("pred_events_val_q099_pooled.csv")
        D = ev[~overlaps(ev, pooled) & ~overlaps(ev, gt)]
        p5_drop_all = int((~overlaps(ev, pooled)).sum())
        print(f"退化类合并 UCL：主流程报告 {len(ev)}，退化类（φ 全 0 窗口过半）改用合并 UCL 后报告 {len(pooled)}，不再报告 {p5_drop_all}（其中非 GT {len(D)}）")
    print({"A_model_only": len(A), "B_rule_only": len(B), "C_both_nonGT": len(C), "D_P5_dropped": len(D)})
    rng = np.random.default_rng(seed)
    pick = []
    if n == "gt":
        gv = gt[gt.Vehicle_ID.isin(val)]
        groups = [("T", gv)]
    else:
        groups = ([("G", ev[overlaps(ev, gt)]), ("N", ev[~overlaps(ev, gt)])] if n == "all"
                  else [("A", A), ("B", B), ("C", C), ("D", D)])
    for g, d in groups:
        if n in ("all", "gt"):
            pick.append(d[["Vehicle_ID", "start_frame", "end_frame"]].assign(group=g))
            continue
        k = min(n, len(d))
        if k == 0:
            continue
        s = d.iloc[np.sort(rng.choice(len(d), k, replace=False))][["Vehicle_ID", "start_frame", "end_frame"]].copy()
        s["group"] = g
        pick.append(s)
    pick = pd.concat(pick).reset_index(drop=True)
    pick.index += 1

    cols = ["Vehicle_ID", "Frame_ID", "Local_X", "Local_Y", "Lane_Num", "Speed_smooth", "Acc_from_vsmooth_g",
            "Leader_ID", "Gap_bumper", "Preceding_Speed", "TTC", "is_truck", "Length"]
    fr = pd.read_csv("HighSim_frames_with_Anomaly30.csv", usecols=lambda c: c in cols)
    # 侧向净距（utility_phi._side_gap），只在事件窗口涉及的帧上计算
    import utility_phi as up
    need = set()
    for e in pick.itertuples():
        need |= set(range(int(e.start_frame) - PAD, int(e.end_frame) + PAD + 1))
    sub = fr[fr.Frame_ID.isin(need)].copy()
    sub["SideGap"] = up._side_gap(sub, verbose=False)
    fr = fr.merge(sub[["Vehicle_ID", "Frame_ID", "SideGap"]], on=["Vehicle_ID", "Frame_ID"], how="left")
    byf = fr.set_index("Frame_ID").sort_index()
    me_all = (fr[fr.Vehicle_ID.isin(set(pick.Vehicle_ID))]
              .merge(ph.drop(columns=["is_truck", "Global_Time"], errors="ignore"), on=["Vehicle_ID", "Frame_ID"], how="left")
              .merge(pf, on=["Vehicle_ID", "Frame_ID"], how="left"))
    curves = at.load()
    rows = []
    for i, e in pick.iterrows():
        s0, e0 = e.start_frame - PAD, e.end_frame + PAD
        me = me_all[(me_all.Vehicle_ID == e.Vehicle_ID) & me_all.Frame_ID.between(s0, e0)].sort_values("Frame_ID")
        t = ((me.Frame_ID - e.start_frame) / 30).to_numpy()
        te = (e.end_frame - e.start_frame) / 30
        ins = me.Frame_ID.between(e.start_frame, e.end_frame).to_numpy()
        evc = [c + "_ev" for c in rr.O_COLS]
        dims = [nm for nm, c in zip(DN, evc) if c in me and me.loc[ins, c].max() > 0]
        truck = int(me.is_truck.max() > 0)

        fig = plt.figure(figsize=(11, 16))
        gs = GridSpec(6, 2, figure=fig, height_ratios=[1.05, 1, 1, 0.9, 0.9, 0.9], hspace=0.6, wspace=0.28, top=0.955, bottom=0.035)
        # ── Bird's eye view ──
        ax = fig.add_subplot(gs[0, :])
        win = byf.loc[e.start_frame:e.end_frame].reset_index()
        nb = win[win.Vehicle_ID != e.Vehicle_ID]
        y_in, x_in = me.Local_Y[ins], me.Local_X[ins]
        if len(y_in):
            ylo, yhi = y_in.min() - 150, y_in.max() + 150
            xc = x_in.mean()
            near = nb[nb.Local_Y.between(ylo, yhi) & ((nb.Local_X - xc).abs() < 18)]
            ldr = set(me.loc[ins, "Leader_ID"].dropna().astype(int))
            for v, g in near.groupby("Vehicle_ID"):
                c = "#f0a040" if v in ldr else "0.65"
                ax.plot(g.Local_Y, g.Local_X, color=c, lw=1.0)
                ax.plot(g.Local_Y.iloc[0], g.Local_X.iloc[0], "o", color=c, ms=3)
        pre, post = me.Frame_ID < e.start_frame, me.Frame_ID > e.end_frame
        ax.plot(me.Local_Y[pre], me.Local_X[pre], color="#9fd89f", lw=1.6, label="Ego (before event)")
        ax.plot(me.Local_Y[ins], me.Local_X[ins], color="#d62728", lw=2.4, label="Ego (event)")
        ax.plot(me.Local_Y[post], me.Local_X[post], color="#c5b3e6", lw=1.6, label="Ego (after event)")
        if len(y_in):
            ax.plot(y_in.iloc[0], x_in.iloc[0], "o", color="#d62728", mec="k", ms=6)
            ax.plot(y_in.iloc[-1], x_in.iloc[-1], "s", color="#d62728", mec="k", ms=6)
        ax.plot([], [], color="#f0a040", lw=1, label="Leader"); ax.plot([], [], color="0.65", lw=1, label="Others")
        ax.set_title("Bird's Eye View"); ax.set_xlabel("Longitudinal Position (ft)"); ax.set_ylabel("Lateral Position (ft)")
        ax.legend(loc="upper right", ncol=2)
        # ── Speed ──
        ax = fig.add_subplot(gs[1, 0])
        ax.plot(t, me.Speed_smooth, color="#1f5fa8", lw=1.6, label="Ego Speed")
        ax.plot(t, me.Preceding_Speed, "--", color="#f0a040", lw=1.3, label="Preceding Vehicle Speed")
        shade(ax, 0, te, label="Event window")
        ax.set_title("Speed"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Speed (ft/s)"); ax.legend()
        # ── Longitudinal acceleration ──
        ax = fig.add_subplot(gs[1, 1])
        ax.plot(t, me.Acc_from_vsmooth_g, color="#2ca02c", lw=1.6, label="Ego Vehicle Acc")
        v_ev = float(np.nanmedian(me.Speed_smooth[ins])) if ins.any() else float(np.nanmedian(me.Speed_smooth))
        grp = int(at.group_id(pd.Series([truck]), pd.Series([me.Lane_Num.mode().iloc[0]]))[0])
        for side, sgn in (("acc", 1), ("dec", -1)):
            t1 = at.threshold(curves, grp, side, 1, v_ev); t2 = at.threshold(curves, grp, side, 2, v_ev)
            thr = float(t1 + 4 * (t2 - t1))       # z = 5（GT 阈值）
            ax.axhline(sgn * thr, color="#d62728", ls=":", lw=1, label=f"{sgn * thr:+.2f} g (GT 5σ, steady state)")
        shade(ax, 0, te)
        ax.set_title("Longitudinal Acceleration"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Acceleration (g)")
        ax.legend(loc="lower left")
        # ── Lateral position ──
        ax = fig.add_subplot(gs[2, 0])
        ax.plot(t, me.Local_X, color="#8c6bc8", lw=1.6, label="Ego Vehicle Lateral Position")
        shade(ax, 0, te)
        ax.set_title("Lateral Position"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Lateral Position (ft)"); ax.legend()
        # ── Headway & TTC ──
        ax = fig.add_subplot(gs[2, 1])
        ax.plot(t, me.Gap_bumper, color="#3cb7c4", lw=1.5, label="Space Headway (ft)")
        ax.set_ylabel("Headway (ft)")
        a2 = ax.twinx(); a2.grid(False)
        a2.plot(t, me.TTC.clip(upper=20), color="#e377c2", lw=1.3, label="TTC (s)")
        if "SideGap" in me:
            a2.plot(t, me.SideGap.clip(upper=20), color="#7b4fb0", lw=1.2, ls="--", label="Side gap to adjacent vehicle (ft)")
        a2.axhline(1.5, color="#d62728", ls=":", lw=1); a2.axhline(3.0, color="#f0a040", ls=":", lw=1)
        a2.set_ylim(0, 20.5); a2.set_ylabel("TTC (s)")
        shade(ax, 0, te)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left")
        ax.set_title("Headway, TTC & side gap"); ax.set_xlabel("Time (s)")
        # ── SPE vs UCL ──
        ax = fig.add_subplot(gs[3, :])
        spe, ucl = me.frame_spe.to_numpy(float), me.ucl_frame.to_numpy(float)
        ax.plot(t, spe, color="#1f5fa8", lw=1.3, label="SPE")
        over = np.where(spe > ucl, spe, np.nan)
        ax.plot(t, over, color="#d62728", lw=1.8, label="SPE > UCL")
        ax.fill_between(t, 0, over, color="#d62728", alpha=0.25, lw=0)
        ax.plot(t, ucl, "--", color="k", lw=1, label="UCL")
        ax.axvspan(0, te, color="#f6d8b8", alpha=0.6, lw=0, label="Event window")
        ax.set_ylabel("SPE"); ax.set_xlabel("Time (s)"); ax.legend(loc="upper right", ncol=4)
        ax.set_title("Model: SPE vs UCL")
        # ── φ ──
        ax = fig.add_subplot(gs[4, :])
        for p, lab, c in PHI_STYLE:
            ax.plot(t, me[p], color=c, lw=1.8, label=lab)
        ax.axhline(0.5, color="0.4", ls=":", lw=0.8)
        ax.axvspan(0, te, color="#f6d8b8", alpha=0.6, lw=0)
        ax.set_ylim(-0.03, 1.05); ax.set_ylabel("Utility value"); ax.set_xlabel("Time (s)")
        ax.legend(loc="upper right", ncol=4); ax.set_title("Utility features (dotted: 3σ)")
        # ── 证据：事件确认用的综合效用强度 ──
        confirmed = None
        if "joint" in me and rr.JOINT_THR is not None:
            ax = fig.add_subplot(gs[5, :])
            jt = me.joint.to_numpy(float)
            ax.plot(t, jt, color="0.25", lw=1.4, label="Joint utility intensity")
            ax.axhline(rr.JOINT_THR, color="#d62728", ls="--", lw=1.1, label=f"Confirmation threshold ({rr.JOINT_THR:.2f})")
            hit = (jt >= rr.JOINT_THR) & ins
            ax.fill_between(t, 0, np.where(hit, jt, np.nan), color="#d62728", alpha=0.3, lw=0, label="Evidence frames")
            ss = (me.single_sat.fillna(0).to_numpy() > 0) & ins
            if ss.any():
                ax.plot(t[ss], jt[ss], "v", color="#d62728", ms=4, label="Single-dimension saturation")
            ax.axvspan(0, te, color="#f6d8b8", alpha=0.6, lw=0)
            ax.set_ylim(0, 10.5); ax.set_ylabel("Intensity"); ax.set_xlabel("Time (s)")
            ax.legend(loc="upper right", ncol=4); ax.set_title("Evidence: joint utility intensity  √Σz²  (each z capped at 5σ)")
            confirmed = bool(hit.any())

        fig.suptitle(f"#{i}  Vehicle {e.Vehicle_ID} ({'truck' if truck else 'car'})  ·  {GROUP[e.group]}  ·  "
                     f"{te:.1f} s  ·  3σ: {'+'.join(dims) or 'none'}"
                     + ("" if confirmed is None else ("  ·  confirmed" if confirmed else "  ·  not confirmed")), fontsize=13, y=0.985)
        fn = f"{i:03d}_{e.group}_veh{e.Vehicle_ID}.png"
        fig.savefig(os.path.join(out_dir, fn), dpi=85, bbox_inches="tight")
        plt.close(fig)
        rows.append(dict(no=i, group=e.group, Vehicle_ID=e.Vehicle_ID, start_frame=e.start_frame,
                         end_frame=e.end_frame, dur_s=round(te, 1), sigma3_dims="+".join(dims),
                         peak_g=round(float(me.loc[ins, "Acc_from_vsmooth_g"].abs().max()), 3),
                         min_ttc=round(float(me.loc[ins, "TTC"].min()), 2), file=fn,
                         claude_label="", your_judgement=""))
    L = pd.DataFrame(rows)
    L.to_csv(os.path.join(out_dir, "review_list.csv"), index=False)
    out = ["<html><head><meta charset='utf-8'><title>Review</title><style>body{font-family:serif;max-width:1000px;"
           "margin:auto;padding:16px}img{width:100%;border:1px solid #ccc}.cap{color:#444;margin:6px 0 28px}"
           "</style></head><body><h1>Manual review</h1>"]
    for g, d in L.groupby("group"):
        out.append(f"<h2>Group {g}: {GROUP[g]}</h2>")
        for r in d.itertuples():
            out.append(f"<img src='{r.file}'><div class='cap'>#{r.no} · veh {r.Vehicle_ID} · {r.dur_s} s · "
                       f"3σ: {html.escape(r.sigma3_dims or 'none')}</div>")
    out.append("</body></html>")
    open(os.path.join(out_dir, "index.html"), "w").write("\n".join(out))
    print(L.drop(columns=["file", "claude_label", "your_judgement"]).to_string(index=False))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], (a[1] if a[1] in ("all", "gt") else int(a[1])) if len(a) > 1 else 15, int(a[2]) if len(a) > 2 else 11)
