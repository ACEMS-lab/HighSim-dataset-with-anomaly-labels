"""v12：训练 / 验证（正常车）分布对照 —— 哈希划分后检查两边是否覆盖相同的交通状态（KS 统计量；越小越相近）。"""
import sys, os, numpy as np, pandas as pd
from scipy.stats import ks_2samp
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import New
ph = New.load_phi_all()
gt = New.load_gt_events(New.GT_EVENTS_PATH)
tr, va = New.split_ids(ph, gt)
gv = set(gt["Vehicle_ID"].astype(int))
fr = pd.read_csv("HighSim_frames_with_Anomaly30.csv", usecols=["Vehicle_ID", "Speed_smooth", "Gap_bumper", "Lane_Num", "is_truck"])
fr["THW"] = np.where(fr.Gap_bumper > 0, fr.Gap_bumper / np.maximum(fr.Speed_smooth, 1), np.nan)
A = fr[fr.Vehicle_ID.isin(tr)]; B = fr[fr.Vehicle_ID.isin(va) & ~fr.Vehicle_ID.isin(gv)]
print(f"训练车 {len(tr)}，验证车 {len(va)}（其中正常车 {len(set(va) - gv)}）")
for c in ["Speed_smooth", "THW", "Lane_Num"]:
    a, b = A[c].dropna(), B[c].dropna()
    print(f"  {c:12s} KS={ks_2samp(a.sample(min(len(a), 200000), random_state=0), b.sample(min(len(b), 200000), random_state=0)).statistic:.3f}  "
          f"中位数 训练 {a.median():.2f} / 验证 {b.median():.2f}")
print(f"  货车占比 训练 {A.groupby('Vehicle_ID').is_truck.first().mean():.3f} / 验证 {B.groupby('Vehicle_ID').is_truck.first().mean():.3f}")
P = pd.read_csv("phi_all.csv", usecols=["Vehicle_ID", "phi1", "phi2", "phi3", "phi4"])
for c in ["phi1", "phi2", "phi3", "phi4"]:
    a = P[P.Vehicle_ID.isin(tr)][c]; b = P[P.Vehicle_ID.isin(va) & ~P.Vehicle_ID.isin(gv)][c]
    print(f"  {c}: >0 占比 训练 {(a > 0).mean():.3%} / 验证 {(b > 0).mean():.3%}")
