"""
merge_aanomaly.py —— 把 pred_events_all.csv 中检测事件覆盖的帧标为 AAnomaly=1，其余为 0。

用法（在 pipeline 输出目录下）：python merge_aanomaly.py
输出：HighSim_data_with_Anomaly(Final).csv
"""

import numpy as np
import pandas as pd

SMOOTH_CSV = "HighSim_frames_with_Anomaly30.csv"
EVENTS_CSV = "pred_events_all.csv"
OUT_CSV    = "HighSim_data_with_Anomaly(Final).csv"

df_smooth = pd.read_csv(SMOOTH_CSV, low_memory=False)
for c in ["Vehicle_ID", "Frame_ID", "Global_X", "Global_Y", "Class"]:
    if c in df_smooth.columns:
        df_smooth[c] = pd.to_numeric(df_smooth[c], errors="coerce")

print(f"Filtered vehicles in region: {df_smooth['Vehicle_ID'].nunique()}")

events = pd.read_csv(EVENTS_CSV, low_memory=False)
# 去掉首列列名中混入的多余字符
events.columns = [c.lstrip("那").strip() for c in events.columns]
events["Vehicle_ID"]   = pd.to_numeric(events["Vehicle_ID"],   errors="coerce")
events["start_frame"]  = pd.to_numeric(events["start_frame"],  errors="coerce")
events["end_frame"]    = pd.to_numeric(events["end_frame"],    errors="coerce")

aanomaly = np.zeros(len(df_smooth), dtype=np.int8)

for _, ev in events.iterrows():
    vid = ev["Vehicle_ID"]
    sf  = ev["start_frame"]
    ef  = ev["end_frame"]
    mask = (
        (df_smooth["Vehicle_ID"] == vid) &
        (df_smooth["Frame_ID"]  >= sf) &
        (df_smooth["Frame_ID"]  <= ef)
    )
    aanomaly[mask.values] = 1

df_smooth["AAnomaly"] = aanomaly
df_smooth.to_csv(OUT_CSV, index=False)

n_anom  = int(aanomaly.sum())
n_total = len(df_smooth)
print(f"Saved: {OUT_CSV}")
print(f"Total frames : {n_total}")
print(f"AAnomaly = 1 : {n_anom}  ({n_anom/n_total*100:.2f}%)")
print(f"AAnomaly = 0 : {n_total - n_anom}")