"""
tail_repair_anchor.py —— P6 锚点：轨迹末端修复区之前多远的帧受"真实值→保持值"过渡影响。

对以修复区（dq=1）结束的轨迹，统计末端修复区起点之前每 0.5 s 分箱的 3σ 帧率（o1–o4 任一），
以前 3–10 s 的平均帧率为远处参照；从起点往前，第一个帧率 ≤ 参照的分箱即为影响范围的终点。
用法（在 runs/<run> 下）：python ../../tools/tail_repair_anchor.py
"""
import numpy as np
import pandas as pd

ph = pd.read_csv("phi_all.csv", usecols=["Vehicle_ID", "Frame_ID", "dq", "o1", "o2", "o3", "o4"])
ph = ph.sort_values(["Vehicle_ID", "Frame_ID"])
ph["o"] = ph[["o1", "o2", "o3", "o4"]].max(axis=1).fillna(0)
rows = []
for v, g in ph.groupby("Vehicle_ID"):
    dq = g["dq"].fillna(0).to_numpy(); o = g["o"].to_numpy()
    if len(dq) == 0 or dq[-1] != 1:
        continue
    j = len(dq) - 1
    while j > 0 and dq[j - 1] == 1:
        j -= 1
    d = np.arange(1, min(300, j) + 1)
    rows.append(pd.DataFrame({"dist": d, "o": o[j - d]}))
R = pd.concat(rows)
R["bin"] = (R["dist"] - 1) // 15
rate = R.groupby("bin")["o"].mean()
ref = rate[(rate.index >= 6) & (rate.index < 20)].mean()
print(f"以修复区结束的轨迹 {len(rows)} / {ph.Vehicle_ID.nunique()}；远处参照（3–10 s）3σ 帧率 {ref:.4f}")
print((rate / ref).round(2).rename(lambda b: f"{b * 0.5:.1f}-{b * 0.5 + 0.5:.1f}s").head(8).to_string())
first = int(rate.index[(rate <= ref) & (rate.index < 6)].min())
print(f"→ 影响范围 {first * 0.5:.1f} s（TAIL_PRE_FRAMES = {first * 15}）")
