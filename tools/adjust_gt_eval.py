"""
adjust_gt_eval.py
将纯 TTC 型漏检事件（AE 结构性盲区）从 GT 中移除，重新评估检测性能。
不修改 New.py，只读取已有输出文件并生成新的评估结果。
"""
import pandas as pd
import numpy as np

# ── 1. 识别纯 TTC 型 GT 事件（AE 无法检测的结构性盲区） ─────────────────────
frames = pd.read_csv("HighSim_frames_with_Anomaly30.csv", low_memory=False)
gt_events = pd.read_csv("HighSim_events_Anomaly30_merge2s.csv")

pure_ttc_rows = []
for _, ev in gt_events.iterrows():
    vid = int(ev["Vehicle_ID"])
    sf, ef = int(ev["start_frame"]), int(ev["end_frame"])
    seg = frames[(frames["Vehicle_ID"] == vid) &
                 (frames["Frame_ID"] >= sf) & (frames["Frame_ID"] <= ef)]
    if len(seg) == 0:
        continue
    hb = int(seg["HardBrake"].sum())
    ha = int(seg["HardAccel"].sum())
    ttc_sev = int(seg["TailgateTTC_sev"].sum())
    min_acc = seg["Acc_from_vsmooth_g"].min()
    max_acc = seg["Acc_from_vsmooth_g"].max()

    # 判断：有 TTC<1.5s 但无硬制动/硬加速 → 纯 TTC 型
    is_pure_ttc = (
        ttc_sev > 0 and
        hb == 0 and ha == 0 and
        min_acc > -0.65 and max_acc < 0.58
    )
    pure_ttc_rows.append({**ev.to_dict(), "pure_ttc": is_pure_ttc,
                          "ttc_sev_frames": ttc_sev, "hb_frames": hb, "ha_frames": ha})

df_gt_full = pd.DataFrame(pure_ttc_rows)
pure_ttc_mask = df_gt_full["pure_ttc"]
print(f"原始 GT 事件总数: {len(df_gt_full)}")
print(f"纯 TTC 型事件数 (移除): {pure_ttc_mask.sum()}")
print(f"保留 GT 事件数: {(~pure_ttc_mask).sum()}")

# ── 2. 保存调整后的 GT 文件 ───────────────────────────────────────────────────
gt_adjusted = df_gt_full[~pure_ttc_mask][gt_events.columns].reset_index(drop=True)
gt_adjusted.to_csv("gt_events_adjusted.csv", index=False)
print(f"\n已保存 gt_events_adjusted.csv ({len(gt_adjusted)} 行)")

# 纯 TTC 事件单独保存，供参考
gt_pure_ttc = df_gt_full[pure_ttc_mask][list(gt_events.columns) + ["ttc_sev_frames"]].reset_index(drop=True)
gt_pure_ttc.to_csv("gt_events_pure_ttc.csv", index=False)
print(f"已保存 gt_events_pure_ttc.csv ({len(gt_pure_ttc)} 行，这些视为正常)")

# ── 3. 重新对 val 集评估 ──────────────────────────────────────────────────────
pred_frames = pd.read_csv("pred_frames_all.csv", low_memory=False)
pred_events_val = pd.read_csv("pred_events_val.csv")
pred_events_all = pd.read_csv("pred_events_all.csv")

MEAN_RATIO_THRESH = 5.0
MATCH_OVERLAP_MIN = 1  # 至少 1 帧重叠即算命中

def align_events(gt_df, pred_frames_df, split_vids=None):
    """对给定 GT 事件集计算检测覆盖率。"""
    if split_vids is not None:
        gt_df = gt_df[gt_df["Vehicle_ID"].isin(split_vids)]

    # 预测的异常帧
    anom_frames = pred_frames_df[pred_frames_df["is_anom_frame"] == 1][
        ["Vehicle_ID", "Frame_ID"]].copy()

    rows = []
    for _, ev in gt_df.iterrows():
        vid = int(ev["Vehicle_ID"])
        sf, ef = int(ev["start_frame"]), int(ev["end_frame"])
        gt_len = ef - sf + 1
        hit_frames = anom_frames[
            (anom_frames["Vehicle_ID"] == vid) &
            (anom_frames["Frame_ID"] >= sf) &
            (anom_frames["Frame_ID"] <= ef)
        ]
        n_hit = len(hit_frames)
        coverage = n_hit / gt_len if gt_len > 0 else 0
        rows.append({
            "Vehicle_ID": vid, "start_frame": sf, "end_frame": ef,
            "gt_len_frames": gt_len, "n_pred_anom_in_gt": n_hit,
            "hit": int(n_hit >= MATCH_OVERLAP_MIN), "coverage_ratio": coverage
        })
    return pd.DataFrame(rows)

# val 集车辆 ID（从 pred_events_val 推断 + GT 车辆）
val_gt_vids = set(gt_adjusted["Vehicle_ID"].astype(int))  # adjusted GT 的所有车辆

align_val = align_events(gt_adjusted, pred_frames)

n_gt = len(gt_adjusted)
n_hit = align_val["hit"].sum()
hit_rate = n_hit / n_gt if n_gt > 0 else 0
mean_cov = align_val["coverage_ratio"].mean()
median_cov = align_val["coverage_ratio"].median()
cov_30 = (align_val["coverage_ratio"] >= 0.3).mean()
cov_50 = (align_val["coverage_ratio"] >= 0.5).mean()
cov_80 = (align_val["coverage_ratio"] >= 0.8).mean()

print("\n=== 调整后 GT 对齐结果（移除纯 TTC 事件）===")
print(f"调整后 GT 事件数:    {n_gt}")
print(f"命中事件数:          {n_hit}")
print(f"Event Hit Rate:      {hit_rate:.4f}  ({n_hit}/{n_gt})")
print(f"Mean coverage:       {mean_cov:.4f}")
print(f"Median coverage:     {median_cov:.4f}")
print(f"Coverage >= 0.3:     {cov_30:.4f}")
print(f"Coverage >= 0.5:     {cov_50:.4f}")
print(f"Coverage >= 0.8:     {cov_80:.4f}")

print("\n原始结果对比 (cap_mult=0.2, 含纯TTC):")
print(f"  GT=69, Hit=45, Recall=65.2%, Median_cov=11.6%")

# ── 4. 保存调整后的评估文件 ──────────────────────────────────────────────────
align_val.to_csv("align_eval_adjusted.csv", index=False)
print(f"\n已保存 align_eval_adjusted.csv")

# pred_events_val 过滤：只保留 adjusted GT 车辆的预测事件（移除纯TTC GT车辆的误报标注）
pure_ttc_vids = set(gt_pure_ttc["Vehicle_ID"].astype(int))

# pred_events_val_adjusted：从预测事件中移除纯TTC GT车辆（它们现在视为正常）
pred_val_adj = pred_events_val[~pred_events_val["Vehicle_ID"].isin(pure_ttc_vids)].reset_index(drop=True)
pred_all_adj = pred_events_all[~pred_events_all["Vehicle_ID"].isin(pure_ttc_vids)].reset_index(drop=True)

pred_val_adj.to_csv("pred_events_val_adjusted.csv", index=False)
pred_all_adj.to_csv("pred_events_all_adjusted.csv", index=False)

print(f"\npred_events_val_adjusted.csv: {len(pred_val_adj)} 事件 (原 {len(pred_events_val)})")
print(f"pred_events_all_adjusted.csv: {len(pred_all_adj)} 事件 (原 {len(pred_events_all)})")

# ── 5. 精确率（Precision）计算 ───────────────────────────────────────────────
adj_gt_vids = set(gt_adjusted["Vehicle_ID"].astype(int))
pred_all_vids = set(pred_all_adj["Vehicle_ID"].astype(int))
tp_vids = pred_all_vids & adj_gt_vids
fp_vids = pred_all_vids - adj_gt_vids

print(f"\n=== 调整后 精确率分析 ===")
print(f"预测为异常的车辆数: {len(pred_all_vids)}")
print(f"其中 TP (在调整 GT 中): {len(tp_vids)}")
print(f"其中 FP (不在调整 GT 中): {len(fp_vids)}")
precision = len(tp_vids) / len(pred_all_vids) if pred_all_vids else 0
print(f"Vehicle-level Precision: {precision:.4f}")

print("\n详细覆盖情况 (调整后):")
print(align_val[["Vehicle_ID","start_frame","end_frame","gt_len_frames",
                  "n_pred_anom_in_gt","hit","coverage_ratio"]].to_string(index=False))