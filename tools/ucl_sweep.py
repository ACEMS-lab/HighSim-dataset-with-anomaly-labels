"""
ucl_sweep.py —— 控制限（UCL 分位数 q）敏感性分析。模型、GMM、类外判断都固定，只改 q。

需要 New.py detect 阶段输出的 train_frame_spe.csv（训练帧 SPE + 所属类）和 val_windows.npz（验证窗口明细）。
对每个 q：各类 UCL = 该类训练 SPE 的 q 分位数；类外窗口用最严格的类 UCL；每帧取 SPE/UCL 比值最大的覆盖窗口
（与 New.aggregate_window_frames 相同）；事件化规则与主流程相同（≤0.5 s 间断合并，≥1 s）。

输出指标（验证集）：
  - 与 GT 比：事件召回（分急加减速 / TTC）、精确率、F1
  - 报告规则（report_rule.py）：候选事件内至少一帧某维度 ≥3σ 才报告；报告事件中 φ=1 的为强证据
  - 验证车辆随机分两半 A/B 分别给出 F1，看结论是否稳定（避免在同一批数据上既挑 q 又报结果）

用法（在 runs/<run> 目录下）：python ../../tools/ucl_sweep.py [q1 q2 ...]
输出：ucl_sweep.csv
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import exper_fixed as ef  # noqa: E402

QS = [0.95, 0.975, 0.99, 0.995, 0.999]


def main(qs):
    import report_rule as rr
    tr = pd.read_csv("train_frame_spe.csv")
    W = np.load("val_windows.npz")
    vid, start, lab, outside, spe = W["vid"], W["start"], W["lab"], W["outside"], W["spe"]
    T = spe.shape[1]
    fv = np.repeat(vid, T); ff = (start[:, None] + np.arange(T)[None, :]).ravel()
    codes, uniq = pd.factorize(pd.MultiIndex.from_arrays([fv, ff]))
    spe_flat = spe.ravel().astype(float)

    gt, gt_o = rr.load_gt()   # 主口径（去掉跟车同步急加减速）/ 原口径
    vals = set(np.unique(vid))
    gt = gt[gt["Vehicle_ID"].isin(vals)]; gt_o = gt_o[gt_o["Vehicle_ID"].isin(vals)]
    gt_p = rr.load_gt_phys()
    gt_p = gt_p[gt_p["Vehicle_ID"].isin(vals)] if gt_p is not None else gt.iloc[0:0]
    P = ["phi1", "phi2", "phi3", "phi4"]
    ph = pd.read_csv("phi_all.csv", usecols=lambda c: c in ["Vehicle_ID", "Frame_ID", "dq", "cutin"] + P + rr.O_COLS)
    ph = rr.add_tail(ph[ph["Vehicle_ID"].isin(vals)]).set_index(["Vehicle_ID", "Frame_ID"]).sort_index()
    of = rr.load_oflags("phi_all.csv")
    hours = None
    rng = np.random.default_rng(0)
    half = {v: int(rng.random() < 0.5) for v in sorted(vals)}

    rows = []
    K = int(tr["cluster"].max()) + 1
    # P5：退化类 —— 该类一半以上训练窗口的 φ 全为 0（窗口 = train_frame_spe 中连续 T 行），
    #     其 SPE 只反映重构噪声，没有"正常波动"可估计 → 改用全部训练 SPE 的 q 分位数（合并估计）。
    pool = os.environ.get("POOL_DEGENERATE", "1") == "1"
    phi_tr = pd.read_csv("phi_all.csv", usecols=["Vehicle_ID", "Frame_ID"] + P)
    m = tr.merge(phi_tr, on=["Vehicle_ID", "Frame_ID"], how="left")
    wid = np.arange(len(m)) // T
    wz = (m[P].fillna(0).max(axis=1).groupby(wid).max() == 0)
    wc = m["cluster"].groupby(wid).first()
    zero_share = wz.groupby(wc).mean()
    degenerate = [int(k) for k, z in zero_share.items() if z > 0.5]
    print("各类训练窗口 φ 全为 0 的比例:", zero_share.round(3).to_dict(), "→ 退化类:", degenerate,
          "（POOL_DEGENERATE=%d）" % pool)
    for q in qs:
        ucl_k = np.array([tr.loc[tr["cluster"] == k, "spe"].quantile(q) if (tr["cluster"] == k).any() else np.inf
                          for k in range(K)])
        if pool:
            gq = tr["spe"].quantile(q)
            for k in degenerate:
                ucl_k[k] = gq
        ucl_w = np.where(outside, ucl_k.min(), ucl_k[lab])
        ucl_flat = np.repeat(ucl_w, T)
        # 与 New.aggregate_window_frames 相同：每帧取 SPE/UCL 比值最大的覆盖窗口
        ratio = spe_flat / np.maximum(ucl_flat, 1e-12)
        order = np.lexsort((-ratio, codes))
        first = order[np.r_[True, codes[order][1:] != codes[order][:-1]]]
        d = pd.DataFrame({"Vehicle_ID": uniq.get_level_values(0)[codes[first]],
                          "Frame_ID": uniq.get_level_values(1)[codes[first]],
                          "frame_spe": spe_flat[first], "ucl_frame": ucl_flat[first]})
        d["is_anom_frame"] = (d["frame_spe"] > d["ucl_frame"]).astype(int)
        d = d.sort_values(["Vehicle_ID", "Frame_ID"]).reset_index(drop=True)
        if hours is None:
            hours = ef.compute_total_hours(d)
        cand = ef.eventize(d)
        ev, _ = rr.split_events(cand, of)
        if abs(q - 0.99) < 1e-9:   # 供复核（tools/plot_review.py）比较：哪些事件只在原 UCL 下报告
            ev.to_csv("pred_events_val_q099_pooled.csv" if pool else "pred_events_val_q099.csv", index=False)
        dm = rr.mask_frames(d, ev)
        m = ef.compute_metrics(dm, ev, gt, hours)
        mo = ef.compute_metrics(dm, ev, gt_o, hours)
        mp = ef.compute_metrics(dm, ev, gt_p, hours) if len(gt_p) else {"event_recall": None, "event_precision": None, "event_f1": None}

        def hit(sub):
            return sum(((ev["Vehicle_ID"] == r.Vehicle_ID) & (ev["start_frame"] <= r.end_frame)
                        & (ev["end_frame"] >= r.start_frame)).any() for r in sub.itertuples())
        strong = 0
        for r in ev.itertuples():
            w = ph.loc[(r.Vehicle_ID, slice(r.start_frame, r.end_frame)), :]
            ok = (rr.evidence_cols(w) > 0).any().reindex(rr.O_COLS, fill_value=False).to_numpy()
            strong += int((w[P].max().to_numpy()[ok] >= 1 - 1e-9).any())
        f1h = []
        for h in (0, 1):
            vv = [v for v in vals if half[v] == h]
            mh = ef.compute_metrics(dm[dm["Vehicle_ID"].isin(vv)], ev[ev["Vehicle_ID"].isin(vv)],
                                    gt[gt["Vehicle_ID"].isin(vv)], hours / 2)
            f1h.append(mh["event_f1"])
        n = max(len(ev), 1)
        hs = gt.gt_type.str.contains("harsh"); ts = gt.gt_type.str.contains("ttc")
        rows.append(dict(q=q, ucl_C=";".join(f"{u:.3g}" for u in ucl_k),
                         n_candidates=len(cand), n_reported=len(ev), not_reported=len(cand) - len(ev),
                         recall=m["event_recall"], precision=m["event_precision"], f1=m["event_f1"],
                         f1_halfA=f1h[0], f1_halfB=f1h[1], fa_per_h=m.get("false_alarm_rate"),
                         recall_harsh=f"{hit(gt[hs])}/{hs.sum()}", recall_ttc=f"{hit(gt[ts])}/{ts.sum()}",
                         strong_share=round(strong / n, 3), cov08=m.get("coverage_ge_0.8"),
                         recall_origGT=mo["event_recall"], precision_origGT=mo["event_precision"],
                         f1_origGT=mo["event_f1"], n_physGT=len(gt_p), recall_physGT=mp["event_recall"],
                         precision_physGT=mp["event_precision"], f1_physGT=mp["event_f1"]))
        print(rows[-1], flush=True)
    R = pd.DataFrame(rows)
    R.to_csv("ucl_sweep_pooled.csv" if pool else "ucl_sweep.csv", index=False)
    print(R.to_string(index=False))


if __name__ == "__main__":
    main([float(x) for x in sys.argv[1:]] or QS)
