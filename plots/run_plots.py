
import pandas as pd
from plot_validation import plot_all_validation_vehicles

PHI_VAL_PATH        = "phi_all.csv"
PRED_FRAMES_VAL_PATH = "pred_frames_all.csv"
PRED_EVENTS_VAL_PATH = "pred_events_all.csv"
GT_EVENTS_PATH      = "HighSim_events_Anomaly30_merge2s.csv"   # full GT (去噪后 accel-only 仅剩个位数事件)
OUTPUT_DIR          = "pred_only_plots"

def load_gt_events(path, id_col="Vehicle_ID"):
    ev = pd.read_csv(path, low_memory=False)

    numeric_cols = [
        "Vehicle_ID", "event_id", "start_time", "end_time", "duration_s",
        "start_frame", "end_frame", "num_frames",
        "min_g", "max_g", "peak_abs_g", "peak_g"
    ]
    for c in numeric_cols:
        if c in ev.columns:
            ev[c] = pd.to_numeric(ev[c], errors="coerce")

    # FIX 🟡-3: 补 start_frame / end_frame（与主流程 load_gt_events 保持一致）
    ev = ev.dropna(subset=[id_col, "start_time", "end_time",
                            "start_frame", "end_frame"]).copy()
    ev[id_col] = ev[id_col].astype(int)
    ev = ev.sort_values([id_col, "start_time"]).reset_index(drop=True)
    return ev


def main():
    df_phi_val   = pd.read_csv(PHI_VAL_PATH,         low_memory=False)
    df_frame_val = pd.read_csv(PRED_FRAMES_VAL_PATH,  low_memory=False)
    pred_events  = pd.read_csv(PRED_EVENTS_VAL_PATH,  low_memory=False)
    gt_events    = load_gt_events(GT_EVENTS_PATH)

    val_ids       = df_frame_val["Vehicle_ID"].unique()
    gt_events_val = gt_events[gt_events["Vehicle_ID"].isin(val_ids)].copy()

    plot_all_validation_vehicles(
        df_phi_val    = df_phi_val,
        df_frame_val  = df_frame_val,
        gt_events_val = gt_events_val,
        pred_events   = pred_events,
        output_dir    = OUTPUT_DIR,
        time_col      = "Global_Time",
        context_s     = 5,
        bin_ms        = 500,
        gauss_sigma   = 0.5,
        pred_only     = True,
    )


if __name__ == "__main__":
    main()
