"""
plot_validation.py  —  Polished Version
=========================================
润色改动：
  - bin=2000ms + 高斯卷积二次平滑（sigma 可调）
  - SPE 超 UCL 部分用 fill_between 填充面积（更直观）
  - 4 条 φ 线全保留，各加半透明填充增加层次感
  - 配色升级（colorblind-friendly）
  - 去掉多余边框，grid 更轻，tick 方向向外
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter1d

FONT_SIZE = 16

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "pdf.fonttype": 42,
})


# =========================
# Helpers
# =========================
def _epoch_to_dt(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    x_valid = x.dropna()
    if len(x_valid) == 0:
        return pd.to_datetime(series, errors="coerce")
    unit = "ms" if np.nanmedian(x_valid) > 1e12 else "s"
    return pd.to_datetime(pd.to_numeric(series, errors="coerce"),
                          unit=unit, errors="coerce")


def _build_frame2time(df: pd.DataFrame, time_col="Global_Time") -> pd.Series:
    """Frame_ID -> datetime，从 dfu（phi_val）构建，确保有 Global_Time"""
    if time_col not in df.columns:
        return pd.Series(dtype="datetime64[ns]")
    tmp = (
        df[["Frame_ID", time_col]]
        .dropna()
        .drop_duplicates("Frame_ID")
        .sort_values("Frame_ID")
        .copy()
    )
    tmp["dt"] = _epoch_to_dt(tmp[time_col])
    tmp = tmp.dropna(subset=["dt"])
    return pd.Series(tmp["dt"].values, index=tmp["Frame_ID"].values)


def _nearest_time(frame2time: pd.Series, frame_id):
    if frame2time is None or len(frame2time) == 0:
        return None
    try:
        fid = int(frame_id)
    except (TypeError, ValueError):
        return None
    idx = frame2time.index.to_numpy()
    if fid in frame2time.index:
        return frame2time.loc[fid]
    j = int(np.argmin(np.abs(idx - fid)))
    return frame2time.iloc[j]


def _gauss_smooth(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0 or len(arr) < 3:
        return arr
    return gaussian_filter1d(arr.astype(float), sigma=sigma)


def smooth_by_timebin(
    dff: pd.DataFrame,
    dfu: pd.DataFrame,
    time_col: str = "Global_Time",
    bin_ms: int = 3000,
    gauss_sigma: float = 1.5,
):
    dff = dff.copy()
    dfu = dfu.copy()

    # dff 补时间列（pred_frames_val 无 Global_Time，从 dfu merge 补充）
    if time_col in dff.columns:
        dff["dt"] = _epoch_to_dt(dff[time_col])
    else:
        if time_col in dfu.columns:
            time_map = (
                dfu[["Frame_ID", time_col]]
                .drop_duplicates("Frame_ID")
                .rename(columns={time_col: "_t"})
            )
            dff = dff.merge(time_map, on="Frame_ID", how="left")
            dff["dt"] = _epoch_to_dt(dff["_t"])
            dff.drop(columns=["_t"], inplace=True)
        else:
            dff["dt"] = pd.NaT

    dff = dff.dropna(subset=["dt"]).set_index("dt").sort_index()
    dfu["dt"] = _epoch_to_dt(dfu[time_col])
    dfu = dfu.dropna(subset=["dt"]).set_index("dt").sort_index()

    rule = f"{int(bin_ms)}ms"

    spe_agg = {"frame_spe": "max", "ucl_frame": "max"}
    if "Frame_ID" in dff.columns:
        spe_agg["Frame_ID"] = "median"
    dff_s = (
        dff.resample(rule).agg(spe_agg)
        .dropna(subset=["frame_spe", "ucl_frame"])
        .reset_index()
    )

    phi_agg = {"phi1": "mean", "phi2": "mean", "phi3": "mean", "phi4": "mean"}
    if "Frame_ID" in dfu.columns:
        phi_agg["Frame_ID"] = "median"
    if time_col in dfu.columns:
        phi_agg[time_col] = "median"
    dfu_s = (
        dfu.resample(rule).agg(phi_agg)
        .dropna(subset=["phi1", "phi2", "phi3", "phi4"])
        .reset_index()
    )

    # 高斯二次平滑
    for col in ["frame_spe", "ucl_frame"]:
        if col in dff_s.columns:
            dff_s[col] = _gauss_smooth(dff_s[col].values, gauss_sigma)
    for col in ["phi1", "phi2", "phi3", "phi4"]:
        if col in dfu_s.columns:
            dfu_s[col] = _gauss_smooth(dfu_s[col].values, gauss_sigma)

    return dff_s, dfu_s


# =========================
# Color palette
# =========================
PALETTE = {
    "spe":       "#2C7BB6",   # blue SPE line
    "ucl":       "#000000",   # black UCL dashed
    "fill_anom": "#D7191C",   # red fill SPE > UCL
    "gt_span":   "#CCCCCC",   # light gray GT region
    "pred_span": "#FDCBA0",   # light salmon pred event
    "phi1":      "#AAC4E0",   # light blue-gray
    "phi2":      "#F5C07A",   # light orange
    "phi3":      "#B8B8B8",   # light gray
    "phi4":      "#7B61C4",   # purple (prominent)
}


# =========================
# Internal helpers
# =========================
def _draw_spans(ax, gt_evs, pred_evs, frame2time):
    for _, ev in gt_evs.iterrows():
        t1 = _nearest_time(frame2time, ev["start_frame"])
        t2 = _nearest_time(frame2time, ev["end_frame"])
        if t1 is not None and t2 is not None:
            ax.axvspan(t1, t2, color=PALETTE["gt_span"],   alpha=0.22, lw=0, zorder=0)
    for _, ev in pred_evs.iterrows():
        t1 = _nearest_time(frame2time, ev["start_frame"])
        t2 = _nearest_time(frame2time, ev["end_frame"])
        if t1 is not None and t2 is not None:
            ax.axvspan(t1, t2, color=PALETTE["pred_span"], alpha=0.28, lw=0, zorder=0)

def plot_all_validation_vehicles(
    df_phi_val, df_frame_val, gt_events_val, pred_events,
    output_dir="validation_plots",
    time_col="Global_Time",
    context_s=30,          # 事件前后各30s
    bin_ms=500,
    gauss_sigma=0.5,
    save_dpi=200,
    pred_only=False,       # True → 只画 pred_events 车辆，窗口以 pred 事件为中心
):
    os.makedirs(output_dir, exist_ok=True)

    # pred_only 模式：只遍历 pred_events 中的车辆
    if pred_only:
        vid_iter = sorted(pred_events["Vehicle_ID"].unique())
    else:
        vid_iter = sorted(df_frame_val["Vehicle_ID"].unique())

    for vid in vid_iter:

        dfu = df_phi_val[df_phi_val["Vehicle_ID"] == vid].sort_values("Frame_ID").copy()
        dff = df_frame_val[df_frame_val["Vehicle_ID"] == vid].sort_values("Frame_ID").copy()
        pred_evs = pred_events[pred_events["Vehicle_ID"] == vid]
        gt_evs   = gt_events_val[gt_events_val["Vehicle_ID"] == vid] if gt_events_val is not None else pd.DataFrame()

        if len(dff) == 0 or len(dfu) == 0:
            continue

        # pred_only 模式：只用 pred 事件作为绘图窗口中心
        if pred_only:
            all_evs = pred_evs.copy()
        else:
            all_evs = pd.concat([gt_evs, pred_evs]).drop_duplicates()
        if len(all_evs) == 0:
            continue

        context_frames = int(context_s * 30)  # 30fps

        # ── 每个事件单独画一张图 ──────────────────────────────────
        for ev_idx, (_, ev) in enumerate(all_evs.iterrows()):
            sf = int(ev["start_frame"]) - context_frames
            ef = int(ev["end_frame"])   + context_frames

            # 截取窗口
            dff_win = dff[(dff["Frame_ID"] >= sf) & (dff["Frame_ID"] <= ef)].copy()
            dfu_win = dfu[(dfu["Frame_ID"] >= sf) & (dfu["Frame_ID"] <= ef)].copy()

            if len(dff_win) == 0 or len(dfu_win) == 0:
                continue

            dff_s, dfu_s = smooth_by_timebin(
                dff_win, dfu_win, time_col=time_col,
                bin_ms=bin_ms, gauss_sigma=gauss_sigma,
            )
            if len(dff_s) == 0 or len(dfu_s) == 0:
                continue

            frame2time = _build_frame2time(dfu_win, time_col=time_col)

            fig, axs = plt.subplots(
                2, 1, figsize=(14, 7.5), sharex=True,
                gridspec_kw={"height_ratios": [1, 1], "hspace": 0.08},
            )

            # ── 上图：SPE + UCL ──────────────────────────────
            ax0 = axs[0]
            t   = dff_s["dt"].values
            spe = dff_s["frame_spe"].values
            ucl = dff_s["ucl_frame"].values

            _draw_spans(ax0, gt_evs, pred_evs, frame2time)

            ax0.fill_between(t, 0, spe, color=PALETTE["spe"], alpha=0.08, zorder=1)
            ax0.fill_between(
                t, ucl, spe,
                where=(spe > ucl), interpolate=True,
                color=PALETTE["fill_anom"], alpha=0.12,
                label="_nolegend_", zorder=2,
            )
            # Red line on top of SPE where SPE > UCL
            spe_red = np.where(spe > ucl, spe, np.nan)
            ax0.plot(t, spe_red, color=PALETTE["fill_anom"], lw=1.8,
                     label="SPE > UCL", zorder=4)
            ax0.plot(t, spe, color=PALETTE["spe"], lw=1.8, label="SPE", zorder=3)
            ax0.plot(t, ucl, color=PALETTE["ucl"], lw=1.3, ls="--",
                     alpha=0.9, label="UCL", zorder=3)

            pred_patch = Patch(facecolor=PALETTE["pred_span"], alpha=0.7,
                               edgecolor="none", label="Pred event")
            handles, labels = ax0.get_legend_handles_labels()
            ax0.legend(
                handles + [pred_patch],
                labels + ["Pred Event"],
                loc="upper right",
                frameon=False,
                fontsize=FONT_SIZE,
                ncol=2
            )

            ax0.set_ylabel("SPE", fontsize=FONT_SIZE)
            ax0.set_ylim(bottom=0)
            ax0.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.4)
            for spine in ax0.spines.values():
                spine.set_visible(True)

            # ── 下图：φ1–φ4 ──────────────────────────────────
            ax1 = axs[1]
            x   = dfu_s["dt"].values

            _draw_spans(ax1, gt_evs, pred_evs, frame2time)

            phi_cfg = [
                ("phi1", "φ1 Speed",     PALETTE["phi1"], 1.3, 1),
                ("phi2", "φ2 Acc",       PALETTE["phi2"], 1.3, 1),
                ("phi3", "φ3 Lane",      PALETTE["phi3"], 1.3, 1),
                ("phi4", "φ4 Collision", PALETTE["phi4"], 1.3, 1.00),
            ]
            for col, label, color, lw, alpha in phi_cfg:
                y = dfu_s[col].values
                ax1.plot(x, y, color=color, lw=lw, alpha=alpha,
                         label=label, zorder=3)

            ax1.set_ylabel("Utility value", fontsize=FONT_SIZE)
            ax1.set_ylim(-0.02, 1.10)
            ax1.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.4)
            for spine in ax1.spines.values():
                spine.set_visible(True)
            ax1.legend(
                ncol=2,
                fontsize=FONT_SIZE,
                frameon=False,
                loc="upper right"
            )

            # ── x轴：相对秒数，手动生成刻度避免重复 ─────────────
            x_num   = mdates.date2num(x)
            x_start = x_num[0]
            x_end   = x_num[-1]
            total_s = (x_end - x_start) * 86400
            tick_interval = max(1, int(context_s / 5))
            tick_s   = np.arange(0, total_s + tick_interval, tick_interval)
            tick_pos = x_start + tick_s / 86400
            # 只保留在数据范围内的刻度
            mask     = tick_pos <= x_end + 0.5 / 86400
            tick_pos = tick_pos[mask]
            tick_s   = tick_s[mask]
            for ax in axs:
                ax.set_xticks(tick_pos)
                ax.set_xticklabels([str(int(s)) for s in tick_s], fontsize=FONT_SIZE)
                ax.tick_params(axis="both", labelsize=FONT_SIZE)
                ax.set_xlim(x_start, x_end)
            ax1.set_xlabel("Time (s)", fontsize=FONT_SIZE)
            save_path = os.path.join(
                output_dir, f"Vehicle_{int(vid)}_event{ev_idx+1}.png"
            )
            fig.savefig(save_path, dpi=save_dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {save_path}")

    print(f"\nAll plots saved to: {output_dir}")
