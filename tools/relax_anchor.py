"""
relax_anchor.py —— P7b 锚点：前车切换后，本车与新前车速度差的松弛时间常数 τ。

对每次前车切换（同一辆车 Leader_ID 改变，不含轨迹开始时已有的前车）、跟驰中（THW < THW_FREE）的帧，
统计切换后每 0.5 s 的 |本车速度 − 前车速度| 中位数；以切换后 10–20 s 的中位数为稳态，
超出量 e(t) 按 e0·exp(−t/τ) 拟合（0–10 s，对 log e 线性回归）。
用法（在 runs/<run> 下）：python ../../tools/relax_anchor.py
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import utility_phi as up  # noqa: E402

f = pd.read_csv("HighSim_frames_with_Anomaly30.csv",
                usecols=["Vehicle_ID", "Frame_ID", "Speed_smooth", "Preceding_Speed", "Leader_ID", "Gap_bumper",
                         "Edge", "BadTrack", "BadVehicle"]).sort_values(["Vehicle_ID", "Frame_ID"]).reset_index(drop=True)
ok = (f.Edge == 0) & (f.BadTrack == 0) & (f.BadVehicle == 0)
lid = f.Leader_ID.fillna(-1); vid = f.Vehicle_ID
seg = ((lid != lid.shift()) | (vid != vid.shift())).cumsum()
since = f.groupby(seg).cumcount()
first = seg.groupby(vid).transform("min") == seg
thw = f.Gap_bumper / np.maximum(f.Speed_smooth, up.V_FLOOR)
d = f[ok & ~first & f.Leader_ID.notna() & (thw < up.THW_FREE)].copy()
d["rel"] = (d.Speed_smooth - d.Preceding_Speed).abs()
d["t"] = (since[d.index] // 15) * 0.5
m = d.groupby("t")["rel"].median()
ref = d.loc[(since[d.index] >= 300) & (since[d.index] < 600), "rel"].median()
e = (m - ref).loc[:10.0]
e = e[e > 0]
slope = np.polyfit(e.index.to_numpy(float), np.log(e.to_numpy()), 1)[0]
print(f"切换 {int((~first & (since == 0)).sum())} 次；稳态 |Δv| 中位数 {ref:.2f} ft/s；0 s 时 {m.iloc[0]:.2f}")
print(f"→ τ = {-1 / slope:.2f} s（RELAX_TAU_S）")
