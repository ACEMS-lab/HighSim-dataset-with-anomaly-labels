"""
verify_events.py
----------------
检查模型检测到的异常事件是不是"真实异常"，而非数据噪声。

对 pred_events_all.csv 中每个事件给出：
  - 证据（v8，统一用拉依达 3σ 准则；事件内任一帧在该维度达到 3σ 离群即成立，见 utility_phi.Z_OUT）：
      speed     : 速度偏离正常区间 ≥ 3σ（o1）
      accel     : 加/减速 |z| ≥ 3（o2，周荣贵等曲线）
      lateral   : 压线超 3σ 稀有度且未完成变道 / 横移速度 ≥ 3σ / A→B→A 回头变道（o3）
      proximity : TTC 或并排侧向间距的稀有度 ≥ 3σ（o4）
    证据强度：该维度 φ 峰值达到 1（GT 级：加速度 5σ、TTC ≤ 1.5 s、其余 5σ）记为 strong，否则 moderate。
    单向变道、邻车同时变速不再算证据（v7 起单向变道属正常；邻车变速只用于噪声判断）。
  - 噪声标记（任一为真即可疑）：
      edge              : 事件落在轨迹首尾 1.5 s 内
      bad_vehicle       : 车辆原始加速度长期饱和在 ±20 ft/s²（重建失败）
      implausible_acc   : |加速度| > 1 g
      ramp_geometry     : 在出口匝道区且仅由 phi3 驱动、没有车道变化 —— 典型几何伪影
      raw_acc_saturated : 事件内原始 Acceleration 有 > 50% 帧顶在 ±20 ft/s² 重建上限，
                          且没有邻车印证 —— 疑似轨迹重建噪声
  - 邻车印证 neighbor_corroboration（证据）：同车道、纵向 ±400 ft 内，同时段（事件前 1 s 至后 3 s）
    至少 1 辆邻车速度变化 > 15 ft/s。噪声在车辆之间独立，多车同时变速说明是真实交通事件。

用法（在包含 pipeline 输出的目录下运行）：
    python verify_events.py [pred_events_all.csv]
输出：event_verification.csv + 屏幕汇总
"""
import sys
import numpy as np
import pandas as pd

FRAMES_PATH = "HighSim_frames_with_Anomaly30.csv"
PHI_PATH    = "phi_all.csv"

EDGE_FRAMES     = 45   # 与 Data.EDGE_TRIM_FRAMES 一致
RAMP_Y_FT       = 6400.0
G_HARD          = 0.3
TTC_CONFLICT_S  = 3.0
LAT_DISP_FT     = 6.0
PHI1_DEV_Z      = 3.0
PHI4_PROX       = 0.2
RAW_ACC_BOUND   = 20.0
SAT_FRAC        = 0.5
NB_DY_FT        = 400.0
NB_DV_FTPS      = 15.0

def main(events_path="pred_events_all.csv"):
    EVENTS_PATH = events_path
    cols = ["Vehicle_ID", "Frame_ID", "Local_X", "Local_Y", "Lane_Num",
            "Acc_from_vsmooth_g", "TTC", "BadVehicle", "Acceleration", "Speed_smooth", "is_truck"]
    fr = pd.read_csv(FRAMES_PATH, usecols=lambda c: c in cols, low_memory=False)
    fr = fr.sort_values(["Vehicle_ID", "Frame_ID"])
    g = fr.groupby("Vehicle_ID")
    fr["pos"] = g.cumcount()
    fr["n"]   = g["Frame_ID"].transform("size")
    phi = pd.read_csv(PHI_PATH, low_memory=False)
    OC = [c for c in ["o1", "o2", "o3", "o4"] if c in phi.columns]
    # v10：证据只用"可作证据"的 3σ 帧（振铃修复区不算；前车切入后 1.5 s 内的近距离不算），与 report_rule 一致
    import report_rule as rr
    phi[OC] = rr.evidence_cols(phi)[OC]
    fr = fr.merge(phi[["Vehicle_ID", "Frame_ID", "phi1", "phi2", "phi3", "phi4"] + OC],
                  on=["Vehicle_ID", "Frame_ID"], how="left")
    # 横向位移相对"车道中心线"计算（与 New.py 中 phi3 的中心线一致：按车道、每 50 ft 纵向分箱取中位数），
    # 否则匝道弯道本身会被误当作横向位移。
    fr["center"] = np.nan
    for lane, d in fr.groupby("Lane_Num"):
        b = np.floor(d["Local_Y"] / 50.0)
        st = d.groupby(b)["Local_X"].agg(["median", "size"])
        st = st[st["size"] >= 30]
        yc = (st.index.to_numpy(dtype=float) + 0.5) * 50.0
        fr.loc[d.index, "center"] = np.interp(d["Local_Y"].to_numpy(float), yc, st["median"].to_numpy(float))
    fr["lat_resid"] = fr["Local_X"] - fr["center"]
    by_vid = {v: d for v, d in fr.groupby("Vehicle_ID")}
    by_frame = fr.set_index("Frame_ID").sort_index()

    def neighbours_changing(v, s0, e0, lane, y0):
        w = by_frame.loc[s0 - 30:e0 + 90]
        nb = w[(w["Vehicle_ID"] != v) & (w["Lane_Num"] == lane) & ((w["Local_Y"] - y0).abs() < NB_DY_FT)]
        if nb.empty:
            return 0
        rng = nb.groupby("Vehicle_ID")["Speed_smooth"].agg(lambda x: x.max() - x.min())
        return int((rng > NB_DV_FTPS).sum())

    import json, os
    z1max = json.load(open("phi_zmax.json"))["phi1"] if os.path.exists("phi_zmax.json") else 10.0
    phi1_dev = (PHI1_DEV_Z - 1.0) / (z1max - 1.0)

    ev = pd.read_csv(EVENTS_PATH)
    rows = []
    for _, e in ev.iterrows():
        v = int(e["Vehicle_ID"])
        s = by_vid[v]
        w = s[(s["Frame_ID"] >= e["start_frame"]) & (s["Frame_ID"] <= e["end_frame"])]
        if w.empty:
            continue
        peak_g   = float(w["Acc_from_vsmooth_g"].abs().max())
        min_ttc  = float(w["TTC"].min()) if w["TTC"].notna().any() else np.nan
        lane_chg = int(w["Lane_Num"].nunique() > 1)
        # 车道变换时 Lane_Num 会切换、残差跳变；用不分车道的残差范围 + 车道号变化共同判断
        lat_disp = float(w["lat_resid"].max() - w["lat_resid"].min()) if not lane_chg else np.nan
        phis     = w[["phi1", "phi2", "phi3", "phi4"]].max()     # 峰值（v8；旧版用均值会稀释短时异常）
        in_ramp  = bool(w["Local_Y"].mean() > RAMP_Y_FT and (w["Lane_Num"] <= 1).all())
        lane_mode = int(w["Lane_Num"].mode()[0])
        n_nb     = neighbours_changing(v, int(e["start_frame"]), int(e["end_frame"]),
                                       lane_mode, float(w["Local_Y"].mean()))
        sat_frac = float((w["Acceleration"].abs() >= RAW_ACC_BOUND - 1e-3).mean()) \
            if "Acceleration" in w else 0.0

        ev_nb    = n_nb >= 1
        dims = [("speed", "o1", "phi1"), ("accel", "o2", "phi2"), ("lateral", "o3", "phi3"), ("proximity", "o4", "phi4")]
        ev_dim = {nm: bool(w[o].max() > 0) if o in w else False for nm, o, _ in dims}
        strong = any(ev_dim[nm] and phis[p] >= 1.0 - 1e-9 for nm, _, p in dims)
        ev_hard, ev_conf, ev_lane = ev_dim["accel"], ev_dim["proximity"], ev_dim["lateral"]

        f_edge   = bool((w["pos"].min() < EDGE_FRAMES) or (w["n"].iloc[0] - 1 - w["pos"].max() < EDGE_FRAMES))
        f_badveh = bool(w.get("BadVehicle", pd.Series([0])).max() == 1)
        f_impl   = peak_g > 1.0
        f_ramp   = bool(in_ramp and phis["phi3"] >= max(phis["phi1"], phis["phi4"]) and not ev_lane
                        and not ev_hard and not ev_conf)
        f_sat    = bool(sat_frac > SAT_FRAC and not ev_nb)

        evidence = [nm for nm, _, _ in dims if ev_dim[nm]]
        flags = [n for n, b in [("edge", f_edge), ("bad_vehicle", f_badveh),
                                ("implausible_acc", f_impl), ("ramp_geometry", f_ramp),
                                ("raw_acc_saturated", f_sat)] if b]
        verdict = "suspect_noise" if flags else ("supported" if evidence else "unexplained")
        strength = ("strong" if strong else "moderate") if verdict == "supported" else ""
        rows.append(dict(Vehicle_ID=v, start_frame=int(e["start_frame"]), end_frame=int(e["end_frame"]),
                         duration_s=round(float(e.get("duration_s", np.nan)), 2),
                         vehicle="truck" if w.get("is_truck", pd.Series([0])).max() == 1 else "car",
                         lane=lane_mode, Y_mean=round(float(w["Local_Y"].mean()), 0),
                         peak_abs_g=round(peak_g, 3), min_ttc_s=round(min_ttc, 2),
                         lat_disp_ft=round(lat_disp, 1), raw_sat_frac=round(sat_frac, 2),
                         n_neighbors_changing=n_nb,
                         phi1=round(phis["phi1"], 3), phi2=round(phis["phi2"], 3),
                         phi3=round(phis["phi3"], 3), phi4=round(phis["phi4"], 3),
                         evidence="|".join(evidence), noise_flags="|".join(flags), verdict=verdict,
                         strength=strength))

    R = pd.DataFrame(rows)
    R.to_csv("event_verification.csv", index=False)
    print(f"Events checked: {len(R)}")
    print(R["verdict"].value_counts().to_string())
    print("\nEvidence counts:")
    print(R["vehicle"].value_counts().to_string())
    print("Strength (supported):", R.loc[R["verdict"] == "supported", "strength"].value_counts().to_dict())
    for k in ["speed", "accel", "lateral", "proximity"]:
        print(f"  {k:16s}: {R['evidence'].str.contains(k).sum()}")
    print("Noise-flag counts:")
    for k in ["edge", "bad_vehicle", "implausible_acc", "ramp_geometry", "raw_acc_saturated"]:
        print(f"  {k:16s}: {R['noise_flags'].str.contains(k).sum()}")
    if (R["verdict"] != "supported").any():
        print(f"\nEvents needing a look: {(R['verdict'] != 'supported').sum()}（明细见 event_verification.csv）")
    print("\nSaved event_verification.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pred_events_all.csv")
