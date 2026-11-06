"""
duration_sensitivity.py —— GT 最短持续时间的敏感性分析（模型与检测结果固定，只改变 GT 定义）

对每个最短持续时间（默认 0.5 / 1 / 1.5 / 2 s）：按 Data.py 的同一套规则重新生成 GT 事件
（z>=5 急加/减速 + 同车道 TTC<1.5 s，去噪/截断规则不变），再与已有检测事件 pred_events_val.csv
做严格事件级对比（时间区间重叠即匹配），只统计验证集车辆。

说明：主分析中 val 集包含全部 GT 车辆（1 s 定义）。改变持续时间后，新增 GT 若落在训练车辆上，
这里不计入（模型训练时见过这些车辆）。

用法（在 runs/<run> 目录下）：
    python ../../tools/duration_sensitivity.py [0.5 1 1.5 2]
输出：duration_sensitivity.csv
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import Data  # noqa: E402


def gt_events_for(df, min_s):
    Data.MIN_CONSEC_FRAMES = Data.MIN_CONSEC_FRAMES_TTC = int(round(min_s * 30))
    d = Data.add_anomaly_filtered(df.copy())
    return Data.eventize(d, anomaly_col="Anomaly_30")


def overlaps(vid, s, e, other):
    o = other[other["Vehicle_ID"] == vid]
    return bool(((o["start_frame"] <= e) & (o["end_frame"] >= s)).any())


def main(durations):
    df = pd.read_csv("HighSim_frames_with_Anomaly30.csv", low_memory=False)
    df = df.rename(columns={"Acc_from_vsmooth_ftps2": "Acc_ftps2", "Acc_from_vsmooth_g": "Acc_g"})
    val_ids = set(pd.read_csv("pred_frames_val.csv", usecols=["Vehicle_ID"])["Vehicle_ID"].astype(int))
    pred = pd.read_csv("pred_events_val.csv")
    pred = pred[pred["Vehicle_ID"].isin(val_ids)]
    rows = []
    for m in durations:
        print(f"\n===== 最短持续 {m} s =====")
        gt = gt_events_for(df, m)
        gt_v = gt[gt["Vehicle_ID"].isin(val_ids)]
        gt_hit = sum(overlaps(r.Vehicle_ID, r.start_frame, r.end_frame, pred) for r in gt_v.itertuples())
        pred_tp = sum(overlaps(r.Vehicle_ID, r.start_frame, r.end_frame, gt_v) for r in pred.itertuples())
        rec = gt_hit / max(len(gt_v), 1); prec = pred_tp / max(len(pred), 1)
        by_type = {}
        for t in ["harsh_accel", "ttc_conflict"]:
            sub = gt_v[gt_v["gt_type"].str.contains(t)]
            hit = sum(overlaps(r.Vehicle_ID, r.start_frame, r.end_frame, pred) for r in sub.itertuples())
            by_type[t] = f"{hit}/{len(sub)}"
        rows.append(dict(min_duration_s=m, gt_events_all=len(gt), gt_events_val=len(gt_v),
                         gt_not_in_val=len(gt) - len(gt_v), pred_events=len(pred),
                         event_recall=round(rec, 3), event_precision=round(prec, 3),
                         event_f1=round(2 * rec * prec / (rec + prec + 1e-9), 3),
                         recall_harsh_accel=by_type["harsh_accel"], recall_ttc=by_type["ttc_conflict"]))
    R = pd.DataFrame(rows)
    R.to_csv("duration_sensitivity.csv", index=False)
    print("\n" + R.to_string(index=False))


if __name__ == "__main__":
    main([float(x) for x in sys.argv[1:]] or [0.5, 1.0, 1.5, 2.0])
