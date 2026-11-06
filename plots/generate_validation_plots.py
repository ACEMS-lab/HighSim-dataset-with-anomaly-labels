"""
generate_validation_plots.py
============================
Generate multi-panel composite plots for manual review (blind annotation)
of detected anomaly events from the Utility-enhanced GRU-AE-GMM framework.

Inputs (expected in the same directory unless paths are changed):
    HighSim_frames_with_Anomaly30.csv   - full frame-level data (do NOT use Excel-truncated CSVs)
    pred_events_val.csv                  - detected events on the validation set
    phi_all.csv                          - utility features phi1..phi4
    pred_frames_val.csv                  - frame-level SPE and UCL on the validation set

Outputs:
    plots/case_XXX.png                   - one composite plot per case
    annotation_template.xlsx             - blank annotation form
    case_key.csv                         - de-anonymisation key (DO NOT share before annotation)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.signal import savgol_filter

plt.rcParams.update({
    "font.family":     "serif",
    "font.serif":      ["Times New Roman"],
    "font.size":       14,
    "axes.titlesize":  14,
    "axes.labelsize":  14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
SEED            = 42
FPS             = 30                       # frames per second
BUFFER_FRAMES   = 2 * FPS                  # ±2 s buffer around event
LONG_RANGE_FT   = 164.0                    # bird's-eye view longitudinal half-range
LAT_RANGE_FT    = 20.0                     # bird's-eye view lateral half-range
N_NORMAL_CTRL   = 40                       # number of normal control windows
CTRL_WIN_LEN    = 90                       # normal control window length (frames) = 3s

# Hard accel / decel thresholds from Perez et al. 2017 (units: g)
THR_HARD_ACCEL  = 0.58
THR_HARD_DECEL  = -0.75

# Reference safety distances / TTC (consistent with the manuscript)
TTC_THRESHOLD_S = 3.0
LONG_SAFE_FT    = 15.0


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _read_csv_safe(path, **kw):
    """
    Read CSV while stripping UTF-8 BOM from the first header
    (some pipelines / Excel exports leave a \ufeff in front of the first column).
    """
    df = pd.read_csv(path, encoding="utf-8-sig", **kw)
    df.columns = [c.lstrip("\ufeff").strip() for c in df.columns]
    return df


def load_inputs(args):
    print("Loading inputs ...")
    df_raw    = _read_csv_safe(args.raw_csv,    low_memory=False)
    df_events = _read_csv_safe(args.events_csv)
    df_phi    = _read_csv_safe(args.phi_csv,    low_memory=False)
    df_frame  = _read_csv_safe(args.frame_csv,  low_memory=False)

    # numeric coercion on the key columns
    num_raw = ["Vehicle_ID","Frame_ID","Global_Time","Local_X","Local_Y",
               "Width","Length","Speed","Speed_smooth","Acceleration",
               "Acc_from_vsmooth_g","Lane_Num",
               "Preceding_Vehicle_ID","Space_Headway","Preceding_Speed"]
    for c in num_raw:
        if c in df_raw.columns:
            df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")

    df_raw = df_raw.sort_values(["Vehicle_ID","Frame_ID"]).reset_index(drop=True)

    print(f"  raw frames     : {len(df_raw):,}  ({df_raw['Vehicle_ID'].nunique()} vehicles)")
    print(f"  detected events: {len(df_events):,}   columns={list(df_events.columns)}")
    print(f"  phi rows       : {len(df_phi):,}")
    print(f"  frame preds    : {len(df_frame):,}")

    # ---- Sanity checks -----------------------------------------------
    if "Vehicle_ID" not in df_events.columns:
        raise ValueError(
            f"events file '{args.events_csv}' does not contain a 'Vehicle_ID' column. "
            f"Found: {list(df_events.columns)}"
        )

    # detect Excel-truncated raw csv (1,048,575 lines = Excel row limit)
    if len(df_raw) <= 1_048_575:
        # cross-check against phi_all which should reflect the *full* dataset
        if len(df_phi) > len(df_raw) * 1.5:
            print("\n  ⚠️  WARNING: raw CSV appears to be TRUNCATED.")
            print(f"     raw has {len(df_raw):,} rows but phi has {len(df_phi):,} rows.")
            print(f"     The bird's-eye view will be missing neighbours for most events.")
            print(f"     Please use the *complete* frame-level CSV instead "
                  f"(e.g. HighSim_frames_with_Anomaly30.csv).\n")

    # check val/all consistency: warn if the events file looks like the 'all' file
    # (train+val), since validation should mirror the per-paper test-set evaluation
    if "all" in os.path.basename(args.events_csv).lower():
        print("\n  ℹ️  NOTE: you are using an 'all' events file (train+val).")
        print(f"     For consistency with the paper's test-set evaluation, consider")
        print(f"     --events_csv pred_events_val.csv and --frame_csv pred_frames_val.csv.\n")

    return df_raw, df_events, df_phi, df_frame


def _build_vehicle_index(df, col="Vehicle_ID"):
    """
    For a sorted dataframe, build {vid: (start_pos, end_pos)} so we can
    slice each vehicle's rows by integer position instead of boolean mask.
    Speeds up per-case lookup from O(N) to O(1) on a 7M-row table.
    """
    df = df.sort_values([col, "Frame_ID"]).reset_index(drop=True)
    vids = df[col].to_numpy()
    # positions where vehicle id changes
    change_idx = np.flatnonzero(np.r_[True, vids[1:] != vids[:-1]])
    starts = change_idx
    ends   = np.r_[change_idx[1:], len(df)]
    index_map = {int(vids[s]): (int(s), int(e)) for s, e in zip(starts, ends)}
    return df, index_map


def _slice_vehicle(df, index_map, vid, sf_buf, ef_buf):
    """Get a vehicle's rows within [sf_buf, ef_buf] using the prebuilt index."""
    if vid not in index_map:
        return df.iloc[0:0]
    s, e = index_map[vid]
    sub = df.iloc[s:e]
    # sub is already sorted by Frame_ID
    f = sub["Frame_ID"].to_numpy()
    mask = (f >= sf_buf) & (f <= ef_buf)
    return sub.iloc[mask]


def _neighbours_in_window(df_raw, frame_index, vid, sf_buf, ef_buf,
                          y_lo, y_hi, x_lo, x_hi):
    """
    Find all NON-ego vehicles with at least one frame in [sf_buf, ef_buf]
    falling inside the view box. Uses the prebuilt frame->position index.
    """
    # slice rows by frame range
    f_min = max(sf_buf, frame_index["f_min"])
    f_max = min(ef_buf, frame_index["f_max"])
    if f_max < f_min:
        return df_raw.iloc[0:0]
    lo = frame_index["frame_to_pos"].get(f_min)
    hi = frame_index["frame_to_pos"].get(f_max)
    # fall back to linear search if exact frame missing
    if lo is None:
        f_arr = frame_index["sorted_frames"]
        lo_idx = np.searchsorted(f_arr, f_min, side="left")
        if lo_idx >= len(f_arr):
            return df_raw.iloc[0:0]
        lo = frame_index["frame_to_pos"][int(f_arr[lo_idx])]
    if hi is None:
        f_arr = frame_index["sorted_frames"]
        hi_idx = np.searchsorted(f_arr, f_max, side="right") - 1
        if hi_idx < 0:
            return df_raw.iloc[0:0]
        hi = frame_index["frame_to_pos"][int(f_arr[hi_idx])] + frame_index["frame_counts"][int(f_arr[hi_idx])]
    else:
        hi = hi + frame_index["frame_counts"][f_max]

    sub = df_raw.iloc[lo:hi]
    mask = (
        (sub["Vehicle_ID"] != vid) &
        sub["Local_Y"].between(y_lo, y_hi) &
        sub["Local_X"].between(x_lo, x_hi)
    )
    return sub.loc[mask]


def _build_frame_index(df_raw):
    """
    Build per-frame position index. Assumes df_raw is sorted by Frame_ID.
    """
    df_sorted = df_raw.sort_values(["Frame_ID","Vehicle_ID"]).reset_index(drop=True)
    f = df_sorted["Frame_ID"].to_numpy()
    change = np.flatnonzero(np.r_[True, f[1:] != f[:-1]])
    starts = change
    ends   = np.r_[change[1:], len(df_sorted)]
    frame_to_pos  = {int(f[s]): int(s) for s in starts}
    frame_counts  = {int(f[s]): int(e - s) for s, e in zip(starts, ends)}
    sorted_frames = np.array(sorted(frame_to_pos.keys()))
    return df_sorted, {
        "frame_to_pos":  frame_to_pos,
        "frame_counts":  frame_counts,
        "sorted_frames": sorted_frames,
        "f_min": int(sorted_frames[0]),
        "f_max": int(sorted_frames[-1]),
    }



def sample_normal_controls(df_raw_byveh, raw_veh_idx,
                           df_events, df_frame_byveh,
                           n=N_NORMAL_CTRL,
                           win_len=CTRL_WIN_LEN, rng=None):
    """
    Sample random windows from vehicles that were NEVER detected as anomalous,
    to act as negative controls in the blind review pool.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    # vehicles with any detected event = exclude from control pool
    detected_vids = set(df_events["Vehicle_ID"].astype(int).unique())
    exclude_vids = detected_vids

    # use the prebuilt vehicle position index to compute (min,max,count) cheaply
    veh_meta = []
    fr_col = df_raw_byveh["Frame_ID"].to_numpy()
    for vid, (s, e) in raw_veh_idx.items():
        if vid in exclude_vids:
            continue
        n_rows = e - s
        if n_rows < win_len + 2 * BUFFER_FRAMES:
            continue
        veh_meta.append((vid, int(fr_col[s]), int(fr_col[e - 1])))

    if len(veh_meta) < n:
        print(f"  WARNING: only {len(veh_meta)} candidate normal vehicles, "
              f"sampling with replacement")
        choices = rng.choice(len(veh_meta), size=n, replace=True)
    else:
        choices = rng.choice(len(veh_meta), size=n, replace=False)

    ctrl_rows = []
    for idx in choices:
        vid, f_min, f_max = veh_meta[idx]
        lo = f_min + BUFFER_FRAMES
        hi = f_max - win_len - BUFFER_FRAMES
        if hi <= lo:
            continue
        sf = int(rng.integers(lo, hi))
        ef = sf + win_len - 1
        ctrl_rows.append({
            "Vehicle_ID":  int(vid),
            "start_frame": sf,
            "end_frame":   ef,
            "source":      "normal_control",
        })

    df_ctrl = pd.DataFrame(ctrl_rows)
    print(f"  sampled {len(df_ctrl)} normal control windows")
    return df_ctrl


def build_case_pool(df_events, df_ctrl, rng=None):
    """
    Merge detected events + normal controls, shuffle, and assign anonymous IDs.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    df_det = df_events[["Vehicle_ID","start_frame","end_frame"]].copy()
    df_det["source"] = "detected_event"

    pool = pd.concat([df_det, df_ctrl], ignore_index=True)
    pool = pool.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    pool["case_id"] = [f"case_{i+1:03d}" for i in range(len(pool))]

    # reorder columns
    pool = pool[["case_id","Vehicle_ID","start_frame","end_frame","source"]]
    print(f"  total cases in review pool: {len(pool)}  "
          f"(detected={ (pool['source']=='detected_event').sum() }, "
          f"controls={ (pool['source']=='normal_control').sum() })")
    return pool


# -------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------
def get_window_data(df_raw_byveh, raw_veh_idx,
                    df_raw_byfr,  raw_frame_idx,
                    df_phi_byveh, phi_veh_idx,
                    df_frame_byveh, frame_veh_idx,
                    vid, sf, ef):
    """Pull all data needed for one case, with ±buffer. Uses prebuilt indices."""
    sf_buf = sf - BUFFER_FRAMES
    ef_buf = ef + BUFFER_FRAMES

    ego = _slice_vehicle(df_raw_byveh, raw_veh_idx, vid, sf_buf, ef_buf)
    ego = ego.sort_values("Frame_ID").reset_index(drop=True)
    if len(ego) == 0:
        return None

    ego_phi   = _slice_vehicle(df_phi_byveh,   phi_veh_idx,   vid, sf_buf, ef_buf)
    ego_frame = _slice_vehicle(df_frame_byveh, frame_veh_idx, vid, sf_buf, ef_buf)

    # neighbours: view box centred on ego median position
    y_center = ego["Local_Y"].median()
    x_center = ego["Local_X"].median()
    nbrs = _neighbours_in_window(
        df_raw_byfr, raw_frame_idx, vid, sf_buf, ef_buf,
        y_center - LONG_RANGE_FT, y_center + LONG_RANGE_FT,
        x_center - LAT_RANGE_FT,  x_center + LAT_RANGE_FT,
    )

    return {
        "ego":       ego,
        "ego_phi":   ego_phi,
        "ego_frame": ego_frame,
        "nbrs":      nbrs,
        "sf":        sf,
        "ef":        ef,
        "sf_buf":    sf_buf,
        "ef_buf":    ef_buf,
        "y_center":  y_center,
        "x_center":  x_center,
    }


def plot_case(case_id, vid, sf, ef, source, data, out_dir, show_phi=False):
    """Generate one composite multi-panel figure for blind review."""
    if data is None or len(data["ego"]) == 0:
        return False

    ego, nbrs   = data["ego"], data["nbrs"]
    ego_phi     = data["ego_phi"]
    ego_frame   = data["ego_frame"]
    y_center    = data["y_center"]
    x_center    = data["x_center"]

    # convert time axis to seconds, relative to event start
    t0_global = ego.loc[ego["Frame_ID"] == sf, "Global_Time"].values
    if len(t0_global) == 0:
        t0_global = ego["Global_Time"].iloc[0]
    else:
        t0_global = t0_global[0]
    ego["t_rel"] = (ego["Global_Time"] - t0_global)
    if len(ego_phi):
        ego_phi   = ego_phi.merge(ego[["Frame_ID","t_rel"]], on="Frame_ID", how="left")
    if len(ego_frame):
        ego_frame = ego_frame.merge(ego[["Frame_ID","t_rel"]], on="Frame_ID", how="left")
    if len(nbrs):
        nbrs = nbrs.merge(ego[["Frame_ID","t_rel"]], on="Frame_ID", how="left")

    t_event_start = 0.0
    t_event_end   = (ef - sf) / FPS

    n_panels = 6 if show_phi else 5
    fig = plt.figure(figsize=(13, 10 if show_phi else 9))
    gs  = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.30)

    # ----- Panel 1: bird's-eye view -----
    ax1 = fig.add_subplot(gs[0, :])
    # neighbour trajectories
    if len(nbrs) > 0:
        for nv, ng in nbrs.groupby("Vehicle_ID"):
            ng = ng.sort_values("Frame_ID")
            ax1.plot(ng["Local_Y"], ng["Local_X"],
                     color="0.65", lw=0.9, alpha=0.65, zorder=1)
            # mark neighbour at event start
            at_start = ng[ng["Frame_ID"] == sf]
            if len(at_start) > 0:
                ax1.scatter(at_start["Local_Y"], at_start["Local_X"],
                            s=22, color="0.45", zorder=2)
    # ego trajectory
    ego_pre   = ego[ego["Frame_ID"] <  sf]
    ego_evt   = ego[(ego["Frame_ID"] >= sf) & (ego["Frame_ID"] <= ef)]
    ego_post  = ego[ego["Frame_ID"] >  ef]
    ax1.plot(ego_pre["Local_Y"],  ego_pre["Local_X"],
             color="#2ca02c", lw=1.5, alpha=0.5, label="Ego (before event)")
    ax1.plot(ego_evt["Local_Y"],  ego_evt["Local_X"],
             color="#d62728", lw=2.6, label="Ego Vehicle (event)")
    ax1.plot(ego_post["Local_Y"], ego_post["Local_X"],
             color="#9467bd", lw=1.5, alpha=0.5, label="Ego (after event)")
    # ego start / end markers
    if len(ego_evt):
        ax1.scatter(ego_evt["Local_Y"].iloc[0],  ego_evt["Local_X"].iloc[0],
                    s=70, color="#d62728", marker="o", zorder=4, edgecolor="k")
        ax1.scatter(ego_evt["Local_Y"].iloc[-1], ego_evt["Local_X"].iloc[-1],
                    s=70, color="#d62728", marker="s", zorder=4, edgecolor="k")
    ax1.set_xlim(y_center - LONG_RANGE_FT, y_center + LONG_RANGE_FT)
    ax1.set_ylim(x_center - LAT_RANGE_FT,  x_center + LAT_RANGE_FT)
    ax1.set_xlabel("Longitudinal Position (ft)")
    ax1.set_ylabel("Lateral Position (ft)")
    ax1.set_title(f"Bird's Eye View")
    ax1.legend(loc="upper right", fontsize=14, framealpha=0.85)
    ax1.grid(True, alpha=0.3)

    # ----- Panel 2: speed -----
    ax2 = fig.add_subplot(gs[1, 0])
    # on-the-fly SG smoothing (window=51, polyorder=2, same as Data.py)
    _v = ego["Speed"].to_numpy(dtype=float)
    _w = min(51, len(_v))
    _w = _w if _w % 2 == 1 else _w - 1
    if _w >= 3:
        _v_smooth = savgol_filter(_v, window_length=_w, polyorder=2, mode="interp")
        ax2.plot(ego["t_rel"], _v_smooth, color="#1f77b4", lw=2.0,
                 label="Ego Speed")
    else:
        ax2.plot(ego["t_rel"], ego["Speed"], color="#1f77b4", lw=1.5, label="Ego Speed")
    if "Preceding_Speed" in ego.columns and ego["Preceding_Speed"].notna().any():
        ax2.plot(ego["t_rel"], ego["Preceding_Speed"],
                 color="#ff7f0e", lw=1.2, ls="--", label="Preceding Vehicle Speed", alpha=0.85)
    ax2.axvspan(t_event_start, t_event_end, color="red", alpha=0.12,
                label="event window")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Speed (ft/s)")
    ax2.set_title("Speed")
    ax2.legend(fontsize=14); ax2.grid(True, alpha=0.3)

    # ----- Panel 3: acceleration (in g) -----
    ax3 = fig.add_subplot(gs[1, 1])
    if "Acc_from_vsmooth_g" in ego.columns and ego["Acc_from_vsmooth_g"].notna().any():
        ax3.plot(ego["t_rel"], ego["Acc_from_vsmooth_g"],
                 color="#2ca02c", lw=1.5, label="Ego Vehicle Acc")
    else:
        # fallback: convert raw acceleration ft/s^2 to g (1g ≈ 32.174 ft/s^2)
        ax3.plot(ego["t_rel"], ego["Acceleration"] / 32.174,
                 color="#2ca02c", lw=1.5, label="Ego Vehicle Acc")
    ax3.axhline(THR_HARD_ACCEL, color="red", ls=":", lw=1, alpha=0.7,
                label=f"+{THR_HARD_ACCEL}g")
    ax3.axhline(THR_HARD_DECEL, color="red", ls=":", lw=1, alpha=0.7,
                label=f"{THR_HARD_DECEL}g")
    ax3.axvspan(t_event_start, t_event_end, color="red", alpha=0.12)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Acceleration (g)")
    ax3.set_title("Longitudinal Acceleration")
    ax3.legend(fontsize=14); ax3.grid(True, alpha=0.3)

    # ----- Panel 4: lateral position + lane lines -----
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(ego["t_rel"], ego["Local_X"], color="#9467bd", lw=1.5, label="Ego Vehicle Lateral Position")
    # lane num as a secondary y-axis hint
    if "Lane_Num" in ego.columns:
        # rough lane-centre estimate: median Local_X per lane within this window
        for lane_no, lane_grp in ego.groupby("Lane_Num"):
            if pd.isna(lane_no):
                continue
            c = lane_grp["Local_X"].median()
            ax4.axhline(c, color="0.5", ls="--", lw=0.7, alpha=0.6)
    ax4.axvspan(t_event_start, t_event_end, color="red", alpha=0.12)
    ax4.set_ylim(x_center - LAT_RANGE_FT, x_center + LAT_RANGE_FT)
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Lateral Position (ft)")
    ax4.set_title("Lateral Position")
    ax4.legend(fontsize=14); ax4.grid(True, alpha=0.3)

    # ----- Panel 5: gap to preceding vehicle + TTC -----
    ax5 = fig.add_subplot(gs[2, 1])
    if "Space_Headway" in ego.columns:
        ax5.plot(ego["t_rel"], ego["Space_Headway"], color="#17becf",
                 lw=1.5, label="Space Headway (ft)")
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Headway (ft)")
    ax5.tick_params(axis="y", labelcolor="black")
    ax5.axvspan(t_event_start, t_event_end, color="red", alpha=0.12)
    ax5.axhline(LONG_SAFE_FT, color="#17becf", ls=":", lw=1, alpha=0.6)

    # TTC overlay
    ax5b = None
    if "Preceding_Speed" in ego.columns and ego["Preceding_Speed"].notna().any():
        ax5b = ax5.twinx()
        dv  = ego["Speed"] - ego["Preceding_Speed"]
        gap = ego["Space_Headway"] - ego["Length"]  # bumper-to-bumper
        ttc = np.where(dv > 0.1, gap / dv, np.nan)
        ttc = np.clip(ttc, 0, 20)  # cap for readability
        ax5b.plot(ego["t_rel"], ttc, color="#e377c2", lw=1.3, ls="-",
                  label="TTC (s)", alpha=0.85)
        ax5b.axhline(TTC_THRESHOLD_S, color="#e377c2", ls=":", lw=1, alpha=0.6)
        ax5b.set_ylabel("TTC (s)")
        ax5b.tick_params(axis="y", labelcolor="black")
        ax5b.set_ylim(0, 20)

    # merge handles from both axes so TTC appears in the legend
    h1, l1 = ax5.get_legend_handles_labels()
    h2, l2 = (ax5b.get_legend_handles_labels() if ax5b is not None else ([], []))
    ax5.legend(h1 + h2, l1 + l2, fontsize=14, loc="upper left")
    ax5.grid(True, alpha=0.3)
    ax5.set_title("Headway & TTC")

    # No phi panel during blind review
    fig.suptitle(f"{case_id}   —   review pool entry "
                 f"(window length {(ef-sf)/FPS:.1f} s, "
                 f"±{BUFFER_FRAMES/FPS:.0f} s buffer)",
                 fontsize=14, fontweight="bold")

    out_path = os.path.join(out_dir, f"{case_id}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


# -------------------------------------------------------------------
# Annotation template + key
# -------------------------------------------------------------------
def write_annotation_template(pool, out_path):
    """
    Blank Excel/CSV with case_id column and empty rating columns for annotators.
    """
    template = pd.DataFrame({
        "case_id": pool["case_id"],
        "rating":            ["" for _ in range(len(pool))],
        "primary_dimension": ["" for _ in range(len(pool))],
        "notes":             ["" for _ in range(len(pool))],
    })
    try:
        template.to_excel(out_path, index=False)
        print(f"  Wrote annotation template: {out_path}")
    except Exception:
        # fallback to csv if openpyxl missing
        alt = out_path.replace(".xlsx", ".csv")
        template.to_csv(alt, index=False)
        print(f"  Wrote annotation template (CSV fallback): {alt}")

    # Also write a short README explaining the rating scale
    readme = out_path.replace(".xlsx", "_README.txt")
    with open(readme, "w") as f:
        f.write("Blind annotation form\n")
        f.write("=====================\n\n")
        f.write("For each case_XXX.png, fill in the row with this case_id.\n\n")
        f.write("Column 'rating' must be one of:\n")
        f.write("  1  Clearly safety-critical\n")
        f.write("  2  Likely safety-relevant\n")
        f.write("  3  Ambiguous / borderline\n")
        f.write("  4  Normal driving variation\n\n")
        f.write("Column 'primary_dimension' (only for ratings 1 or 2):\n")
        f.write("  speed       - speed deviation from surrounding traffic\n")
        f.write("  longitudinal- hard accel / hard decel\n")
        f.write("  lane        - abnormal lateral position / lane keeping\n")
        f.write("  interaction - dangerously close to / fast-approaching another vehicle\n")
        f.write("  compound    - more than one dimension simultaneously\n\n")
        f.write("Column 'notes': free text (optional).\n\n")
        f.write("Please complete annotation BEFORE consulting case_key.csv.\n")
    print(f"  Wrote annotation README:   {readme}")


def write_case_key(pool, out_path):
    """Hidden key for de-anonymisation after annotation is complete."""
    pool.to_csv(out_path, index=False)
    print(f"  Wrote de-anonymisation key: {out_path}  (DO NOT SHARE with annotators)")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_csv",    default="HighSim_frames_with_Anomaly30.csv",
                    help="Full frame-level CSV (do NOT use the Excel-truncated one)")
    ap.add_argument("--events_csv", default="pred_events_all.csv",
                    help="All detected events (train+val)")
    ap.add_argument("--phi_csv",    default="phi_all.csv")
    ap.add_argument("--frame_csv",  default="pred_frames_all.csv",
                    help="Full frame-level predictions (train+val)")
    ap.add_argument("--out_dir",    default="validation_plots")
    ap.add_argument("--n_normal",   type=int, default=N_NORMAL_CTRL)
    ap.add_argument("--seed",       type=int, default=SEED)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    plots_dir = os.path.join(args.out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # 1) load everything
    df_raw, df_events, df_phi, df_frame = load_inputs(args)

    # 1b) prebuild indices (one-time cost, makes per-case lookup near-instant)
    print("\nBuilding indices on large frames ...")
    df_raw_byveh,   raw_veh_idx   = _build_vehicle_index(df_raw)
    df_raw_byfr,    raw_frame_idx = _build_frame_index(df_raw)
    df_phi_byveh,   phi_veh_idx   = _build_vehicle_index(df_phi)
    df_frame_byveh, frame_veh_idx = _build_vehicle_index(df_frame)
    print("  indices ready.")

    # 2) sample normal controls (only from vehicles present in the frame-level
    #    prediction file, i.e. the val set — to keep validation pool consistent
    #    with the events being reviewed)
    print("\nSampling normal controls ...")
    val_vids = set(df_frame["Vehicle_ID"].astype(int).unique())
    raw_veh_idx_val = {v: idx for v, idx in raw_veh_idx.items() if v in val_vids}
    print(f"  raw vehicles restricted to val set: {len(raw_veh_idx_val)} "
          f"(out of {len(raw_veh_idx)} total)")
    df_ctrl = sample_normal_controls(df_raw_byveh, raw_veh_idx_val,
                                     df_events, df_frame_byveh,
                                     n=args.n_normal, rng=rng)

    # 3) build anonymous pool
    print("\nBuilding anonymous case pool ...")
    pool = build_case_pool(df_events, df_ctrl, rng=rng)

    # 4) generate plots
    print(f"\nGenerating {len(pool)} composite plots ...")
    n_ok = n_skip = 0
    for i, row in pool.iterrows():
        vid = int(row["Vehicle_ID"])
        sf  = int(row["start_frame"])
        ef  = int(row["end_frame"])
        data = get_window_data(
            df_raw_byveh, raw_veh_idx,
            df_raw_byfr,  raw_frame_idx,
            df_phi_byveh, phi_veh_idx,
            df_frame_byveh, frame_veh_idx,
            vid, sf, ef,
        )
        ok = plot_case(row["case_id"], vid, sf, ef, row["source"],
                       data, plots_dir, show_phi=False)
        if ok:
            n_ok += 1
        else:
            n_skip += 1
        if (i+1) % 10 == 0:
            print(f"  [{i+1}/{len(pool)}] done, ok={n_ok}, skip={n_skip}")
    print(f"\nPlots written: {n_ok}  (skipped: {n_skip})")
    print(f"Plot directory: {plots_dir}")

    # 5) write annotation template + key
    print("\nWriting annotation files ...")
    write_annotation_template(pool, os.path.join(args.out_dir, "annotation_template.xlsx"))
    write_case_key(pool,            os.path.join(args.out_dir, "case_key.csv"))

    print("\nDone.")


if __name__ == "__main__":
    main()