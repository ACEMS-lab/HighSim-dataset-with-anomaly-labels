"""
SPE vs UCL — Check All Models for One Vehicle

Output:
  vehicle_17_all_models.png

Purpose:
  Show all comparison methods and the proposed Utility-GRU-AE-GMM
  for a selected vehicle, such as Vehicle 17.
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import warnings

warnings.filterwarnings("ignore")


# =========================
# Global style
# =========================
Fontsize = 16

matplotlib.rcParams.update({
    "font.family":     "serif",
    "font.serif":      ["Times New Roman", "DejaVu Serif"],
    "font.size":       Fontsize,
    "axes.titlesize":  Fontsize,
    "axes.labelsize":  Fontsize,
    "xtick.labelsize": Fontsize,
    "ytick.labelsize": Fontsize,
    "legend.fontsize": Fontsize,
    "figure.dpi":      150,
    "pdf.fonttype":    42,
})


# =========================
# Parameters
# =========================
FPS = 30
CONTEXT_S = 3.0

SPE_COLOR = "#2166AC"
UCL_COLOR = "#CC0000"
FA_COLOR  = "#FFCCCC"


# Fallback global UCL values (only used if ucl_frame column missing in CSV)
# B2~B5 now use raw features — update these values after a fresh run if needed
UCL_GLOBAL = {
    "B3_PCA_SPE":          None,
    "B4_MLP_AE":           None,
    "B5_LSTM_AE":          None,
    "Raw_GRU_AE":          None,
    "Raw_GRU_AE_GMM":      None,
    "Utility_GRU_AE":      None,
    "Utility_LSTM_AE_GMM": None,
}


# 3 × 3 grid: 8 comparison methods + Proposed in bottom-right cell
# Row 1: Traditional baselines (Raw)
METHODS_ROW1 = [
    ("B2_IForest", "IForest"),
    ("B3_PCA_SPE", "PCA"),
    ("B4_MLP_AE",  "AE"),
]

# Row 2: Temporal / deep baselines (Raw)
METHODS_ROW2 = [
    ("B5_LSTM_AE",     "LSTM-AE"),
    ("Raw_GRU_AE",     "Raw-GRU-AE"),
    ("Raw_GRU_AE_GMM", "Raw-GRU-AE-GMM"),
]

# Row 3: Ablations + architecture + Proposed
METHODS_ROW3 = [
    ("Utility_GRU_AE",      "Utility-GRU-AE"),
    ("Utility_LSTM_AE_GMM", "Utility-LSTM-AE-GMM"),
    ("Proposed",            "Utility-GRU-AE-GMM"),  # proposed goes here
]


# =========================
# Data loading
# =========================
def load_df(key):
    """
    Load frame-level results for one method.
    Proposed method uses pred_frames_val.csv.
    Other methods use df_frame_{key}.csv.
    """
    fname = "pred_frames_val.csv" if key == "Proposed" else f"df_frame_{key}.csv"

    try:
        df = pd.read_csv(fname, low_memory=False)

        for c in ["Vehicle_ID", "Frame_ID", "frame_spe", "is_anom_frame"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        if "ucl_frame" in df.columns:
            df["ucl_frame"] = pd.to_numeric(df["ucl_frame"], errors="coerce")

        return df

    except FileNotFoundError:
        print(f"WARNING: {fname} not found.")
        return None

def load_proxy_reference_vids():
    """
    Load proxy reference vehicles.
    """
    fname = "HighSim_events_Anomaly30_merge2s.csv"
    df_gt = pd.read_csv(fname)
    return set(df_gt["Vehicle_ID"].astype(int))


# =========================
# Window helpers
# =========================
def get_vehicle_frame_range(frames_dict, vid):
    """
    Get min/max Frame_ID available for this vehicle across all methods.
    """
    all_frames = []

    for df in frames_dict.values():
        if df is not None:
            sub = df[df["Vehicle_ID"] == int(vid)]["Frame_ID"].dropna()
            all_frames.extend(sub.astype(int).tolist())

    if not all_frames:
        return None, None

    return min(all_frames), max(all_frames)


def clip_center_to_available_range(center, frames_dict, vid, ctx):
    """
    Clip center frame so that [center - ctx, center + ctx]
    stays inside available data range.
    """
    min_f, max_f = get_vehicle_frame_range(frames_dict, vid)

    if min_f is None or max_f is None:
        return int(center)

    if max_f - min_f < 2 * ctx:
        return int((min_f + max_f) // 2)

    center = max(int(center), min_f + ctx)
    center = min(int(center), max_f - ctx)

    return int(center)


def get_window(df, vid, plot_sf, plot_ef):
    """
    Extract one plotting window.
    """
    if df is None:
        return None

    return (
        df[
            (df["Vehicle_ID"] == int(vid)) &
            (df["Frame_ID"] >= int(plot_sf)) &
            (df["Frame_ID"] <= int(plot_ef))
        ]
        .sort_values("Frame_ID")
        .copy()
    )


def get_best_center_for_vehicle(frames_dict, target_vid):
    """
    Choose a useful center frame for a vehicle.

    Priority:
      1. Median frame of all comparison-method anomaly frames.
      2. If no anomaly frames exist, use midpoint of available frames.
    """
    comparison_keys = [k for k, _ in METHODS_ROW1 + METHODS_ROW2 + METHODS_ROW3]

    anom_frames = []

    for key in comparison_keys:
        df = frames_dict.get(key)

        if df is None:
            continue

        sub = df[
            (df["Vehicle_ID"] == int(target_vid)) &
            (df["is_anom_frame"] == 1)
        ]

        if len(sub) > 0:
            anom_frames.extend(sub["Frame_ID"].dropna().astype(int).tolist())

    if anom_frames:
        return int(np.median(anom_frames))

    min_f, max_f = get_vehicle_frame_range(frames_dict, target_vid)

    if min_f is None or max_f is None:
        return None

    return int((min_f + max_f) // 2)


# =========================
# Plotting helpers
# =========================
def draw_ax(
    ax,
    sub,
    key,
    title,
    plot_sf,
    context_s=CONTEXT_S,
    is_proposed=False,
):
    """
    Draw SPE, UCL, and predicted-event shading for one subplot.
    """
    ax.set_title(
        title,
        fontweight="bold" if is_proposed else "normal",
        color="black",
        pad=4,
    )

    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.tick_params(length=3, labelsize=Fontsize)
    ax.grid(True, linestyle="--", linewidth=0.35, alpha=0.4)

    ax.axvline(context_s, color="#AAAAAA", lw=0.8, ls=":", zorder=2)

    if sub is None or len(sub) == 0:
        ax.text(
            0.5, 0.5, "No data",
            ha="center", va="center",
            transform=ax.transAxes,
            color="gray",
            fontsize=Fontsize,
        )
        ax.set_xlim(0, 2 * context_s)
        return

    frames = sub["Frame_ID"].values
    spe = sub["frame_spe"].values
    anom = sub["is_anom_frame"].values == 1
    time_s = (frames - plot_sf) / FPS

    # UCL: use per-frame UCL if available; otherwise use global scalar
    has_ucl_col = (
        "ucl_frame" in sub.columns and
        sub["ucl_frame"].notna().any()
    )

    if has_ucl_col:
        ucl_arr = sub["ucl_frame"].values
        use_step = True
    elif UCL_GLOBAL.get(key) is not None:
        ucl_arr = np.full_like(spe, UCL_GLOBAL[key], dtype=float)
        use_step = False
    else:
        ucl_arr = None
        use_step = False

    top_val = np.nanmax(spe)

    if ucl_arr is not None:
        top_val = max(top_val, np.nanmax(ucl_arr))

    if top_val <= 0 or not np.isfinite(top_val):
        top_val = 1.0

    y_top = top_val * (1.60 if use_step else 1.50)
    ax.set_ylim(0, y_top)

    # Predicted-event shading
    if anom.any():
        chg = np.diff(anom.astype(int), prepend=0, append=0)
        starts = np.where(chg == 1)[0]
        ends = np.where(chg == -1)[0]

        for s, e in zip(starts, ends):
            e_idx = min(e, len(time_s) - 1)
            ax.axvspan(
                time_s[s],
                time_s[e_idx],
                color=FA_COLOR,
                alpha=0.6,
                zorder=1,
            )

    # SPE line
    lw = 2.2 if is_proposed else 1.8
    ax.plot(time_s, spe, color=SPE_COLOR, lw=lw, zorder=4)

    # UCL line
    if ucl_arr is not None:
        if use_step:
            ax.step(
                time_s,
                ucl_arr,
                color=UCL_COLOR,
                lw=2.5,
                ls="--",
                where="post",
                zorder=6,
            )
        else:
            ax.axhline(
                ucl_arr[0],
                color=UCL_COLOR,
                lw=2.2,
                ls="--",
                zorder=6,
            )

    ax.set_xlim(0, 2 * context_s)


def add_legend(fig, y=0.975):
    """
    Add shared legend.
    """
    spe_line = Line2D([0], [0], color=SPE_COLOR, lw=2.0, label="SPE")
    ucl_line = Line2D([0], [0], color=UCL_COLOR, lw=1.8, ls="--", label="UCL")
    fa_patch = mpatches.Patch(color=FA_COLOR, alpha=0.7, label="Pred Event")

    fig.legend(
        handles=[spe_line, ucl_line, fa_patch],
        loc="upper right",
        bbox_to_anchor=(0.98, y),
        ncol=3,
        framealpha=0.92,
        edgecolor="#CCCCCC",
        fontsize=Fontsize,
        columnspacing=1.2,
        handlelength=2.0,
        borderpad=0.6,
        labelspacing=0.5,
    )


# =========================
# Main plotting function
# =========================
def plot_all_models_for_vehicle(
    target_vid=17,
    center_frame=None,
    output_name=None,
):
    """
    Plot all comparison methods + Proposed for a specific vehicle.
    """

    all_keys = [k for k, _ in METHODS_ROW1 + METHODS_ROW2 + METHODS_ROW3]
    frames_dict = {k: load_df(k) for k in all_keys}

    ctx = int(CONTEXT_S * FPS)

    if center_frame is None:
        center_frame = get_best_center_for_vehicle(frames_dict, target_vid)

        if center_frame is None:
            print(f"No data found for Vehicle {target_vid}.")
            return

    center = clip_center_to_available_range(
        center_frame,
        frames_dict,
        target_vid,
        ctx,
    )

    plot_sf = center - ctx
    plot_ef = center + ctx

    print(f"\nPlotting all models for Vehicle {target_vid}")
    print(f"  center_frame    = {center}")
    print(f"  plotting window = [{plot_sf}, {plot_ef}]")

    # Print detection status
    print("\nDetection status in this plotting window:")

    for key, label in METHODS_ROW1 + METHODS_ROW2 + METHODS_ROW3:
        df = frames_dict.get(key)
        sub = get_window(df, target_vid, plot_sf, plot_ef)

        if sub is None or len(sub) == 0:
            print(f"  {label:<24}: no data")
            continue

        n_anom = int((sub["is_anom_frame"] == 1).sum())
        status = "ANOMALY" if n_anom > 0 else "normal"

        print(f"  {label:<24}: {status}, anomalous frames = {n_anom}")

    # Plot — 3 × 3 grid; Proposed is bottom-right cell (highlighted)
    fig = plt.figure(figsize=(13, 9.0))

    gs = GridSpec(
        3, 3,
        figure=fig,
        hspace=0.55,
        wspace=0.35,
    )

    for row_idx, row_methods in enumerate([METHODS_ROW1, METHODS_ROW2, METHODS_ROW3]):
        for col, (key, label) in enumerate(row_methods):
            ax = fig.add_subplot(gs[row_idx, col])

            draw_ax(
                ax,
                get_window(frames_dict[key], target_vid, plot_sf, plot_ef),
                key,
                label,
                plot_sf,
                is_proposed=(key == "Proposed"),
            )

            ax.set_xlabel("Time (s)", labelpad=2, fontsize=Fontsize)

            if col == 0:
                ax.set_ylabel("SPE", labelpad=3, fontsize=Fontsize)

    add_legend(fig, y=0.975)

    fig.suptitle(
        f"Vehicle {target_vid}",
        fontsize=Fontsize,
        fontweight="bold",
        y=0.995,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.92])

    if output_name is None:
        output_name = f"vehicle_{int(target_vid)}_all_models.png"

    fig.savefig(output_name, bbox_inches="tight", dpi=300)

    print(f"\nSaved: {output_name}")
def find_raw_model_false_alarm_case_fast(
    raw_key="Raw_GRU_AE",
    exclude_proxy_vehicles=True,
):
    """
    Fast search:
    Find a vehicle where a raw-feature model flags anomalies,
    but the proposed Utility-GRU-AE-GMM does not flag the vehicle.

    raw_key options:
      - "Raw_GRU_AE"     -> Raw-GRU-AE
      - "Raw_GRU_AE_GMM" -> Raw-GRU-AE-GMM
    """

    all_keys = [k for k, _ in METHODS_ROW1 + METHODS_ROW2 + METHODS_ROW3]
    frames_dict = {k: load_df(k) for k in all_keys}

    df_raw = frames_dict.get(raw_key)
    df_prop = frames_dict.get("Proposed")

    if df_raw is None or df_prop is None:
        print("Missing raw model or proposed model data.")
        return None, None

    gt_vids = set()
    if exclude_proxy_vehicles:
        gt_vids = load_proxy_reference_vids()

    # Raw model anomalous frames
    raw_anom = df_raw[df_raw["is_anom_frame"] == 1][["Vehicle_ID", "Frame_ID"]].dropna()
    raw_anom["Vehicle_ID"] = raw_anom["Vehicle_ID"].astype(int)
    raw_anom["Frame_ID"] = raw_anom["Frame_ID"].astype(int)

    if exclude_proxy_vehicles:
        raw_anom = raw_anom[~raw_anom["Vehicle_ID"].isin(gt_vids)]

    # Vehicles flagged by Proposed
    prop_anom_vids = set(
        df_prop[df_prop["is_anom_frame"] == 1]["Vehicle_ID"]
        .dropna()
        .astype(int)
        .unique()
    )

    # Candidate vehicles:
    # raw model flags them, proposed does not flag them at all
    raw_counts = (
        raw_anom.groupby("Vehicle_ID")["Frame_ID"]
        .agg(["count", "median", "min", "max"])
        .reset_index()
    )

    raw_counts = raw_counts[~raw_counts["Vehicle_ID"].isin(prop_anom_vids)]

    if len(raw_counts) == 0:
        print(f"No vehicle found where {raw_key} is anomalous but Proposed is normal.")
        return None, None

    raw_counts = raw_counts.sort_values("count", ascending=False)

    print("\nRaw-model false-alarm candidates:")
    print(raw_counts.head(15))

    best = raw_counts.iloc[0]

    vid = int(best["Vehicle_ID"])
    center = int(best["median"])

    print("\nSelected raw-model false-alarm case:")
    print(f"  Vehicle_ID         = {vid}")
    print(f"  center_frame       = {center}")
    print(f"  raw model          = {raw_key}")
    print(f"  raw anomaly frames = {int(best['count'])}")
    print("  Proposed does not flag this vehicle.")

    return vid, center

# =========================
# Entry point
# =========================
if __name__ == "__main__":

    # Raw-GRU-AE-GMM abnormal, Proposed normal — illustrates utility features contribution
    vid, center = find_raw_model_false_alarm_case_fast(
        raw_key="Raw_GRU_AE_GMM",
        exclude_proxy_vehicles=True
    )

    if vid is not None:
        plot_all_models_for_vehicle(
            target_vid=vid,
            center_frame=center,
            output_name=f"vehicle_{vid}_raw_gru_ae_gmm_false_alarm.png"
        )

    plt.show()