"""新 GT 候选：周荣贵等(2024) 连续速度-阈值曲线（加/减速样本分开，μ+kσ 分箱后三次多项式拟合），
不扣情境，z ≥ 5 持续 ≥ 1 s。另算"同时超过全局 5σ（同样加/减速样本分开）"。分小汽车 / 货车两组。"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "."); import accel_thresholds as at
src, out = sys.argv[1], sys.argv[2]
cols = ["Vehicle_ID","Frame_ID","Acc_from_vsmooth_g","is_truck","Speed_smooth","Lane_Num","Preceding_Speed",
        "Gap_bumper","Edge","BadTrack","Extrapolated","RepairZone","BadVehicle"]
df = pd.read_csv(src, usecols=cols).sort_values(["Vehicle_ID","Frame_ID"])
df = df[df[cols[8:]].fillna(0).sum(1) == 0].reset_index(drop=True)
df["tr"] = df.is_truck.fillna(0).astype(int)
v, a, grp = df.Speed_smooth.to_numpy(), df.Acc_from_vsmooth_g.to_numpy(), df.tr.to_numpy()
curves = at.fit_curves(v, a, grp)
z = at.zscore(curves, v, a, grp)
df["zs"] = np.sign(a) * z
# 全局 5σ（加/减速样本分开）
glob = {}
for tr in (0, 1):
    for side, m in (("acc", (grp == tr) & (a > 0)), ("dec", (grp == tr) & (a < 0))):
        x = np.abs(a[m]); glob[(tr, side)] = x.mean() + 5 * x.std()
df["gacc"] = np.where(grp == 1, glob[(1, "acc")], glob[(0, "acc")])
df["gdec"] = np.where(grp == 1, glob[(1, "dec")], glob[(0, "dec")])
print("全局 5σ (g):", {f"{'truck' if k[0] else 'car'}-{k[1]}": round(v_, 3) for k, v_ in glob.items()})
# 5σ 曲线在若干速度上的值
for tr, nm in ((0, "car"), (1, "truck")):
    for side in at.SIDES:
        vv = np.array([10, 20, 30, 40, 50, 60, 70, 80]) / 0.6818
        t1 = at.threshold(curves, tr, side, 1, vv); t2 = at.threshold(curves, tr, side, 2, vv); t5 = t1 + 4 * (t2 - t1)
        print(f"{nm} {side} 5σ 曲线 @10..80 mph:", np.round(t5, 3))
g = df.groupby("Vehicle_ID")
cont = (g.Frame_ID.diff(29) == 29)
df["za"] = g.zs.transform(lambda s: s.rolling(30).min()).where(cont)
df["zd"] = g.zs.transform(lambda s: s.rolling(30).max()).where(cont)
ag = df.Acc_from_vsmooth_g
df["ga"] = (g.Acc_from_vsmooth_g.transform(lambda s: s.rolling(30).min()) >= df.gacc).where(cont, False)
df["gd"] = (g.Acc_from_vsmooth_g.transform(lambda s: s.rolling(30).max()) <= -df.gdec).where(cont, False)
EDGES = [0,15,30,45,60,75,200]; LAB = ["0-15","15-30","30-45","45-60","60-75",">75"]
df["vb"] = pd.cut(df.Speed_smooth*0.6818, EDGES, right=False, labels=False)
def events(col, sgn, gcol):
    m = (df[col]*sgn >= 5).fillna(False).to_numpy(); vid = df.Vehicle_ID.to_numpy(); f = df.Frame_ID.to_numpy()
    gm = df[gcol].to_numpy(bool); ev, i = [], 0
    while i < len(m):
        if m[i]:
            j = i
            while j+1 < len(m) and m[j+1] and vid[j+1] == vid[i] and f[j+1] == f[j]+1: j += 1
            ev.append((vid[i], f[i]-29, f[j], int(df.vb.iloc[i]), int(df.tr.iloc[i]), bool(gm[i:j+1].any()))); i = j+1
        else: i += 1
    return pd.DataFrame(ev, columns=["vid","s","e","vb","tr","also_global"])
acc, dec = events("za", 1, "ga"), events("zd", -1, "gd")
for nm, E in (("急加速", acc), ("急减速", dec)):
    print(f"{nm}: 曲线 5σ {len(E)} 个（货车 {int(E.tr.sum())}），其中同时超全局 5σ {int(E.also_global.sum())} 个；"
          f"按速度 {E.groupby('vb').size().rename(index=lambda i: LAB[i]).to_dict()}；"
          f"同时超全局 按速度 {E[E.also_global].groupby('vb').size().rename(index=lambda i: LAB[i]).to_dict()}")
acc.to_csv(f"{out}/accel_events.csv", index=False); dec.to_csv(f"{out}/decel_events.csv", index=False)

INK, MUTED, EGO, LEAD, BAND = "#1f2328", "#6e7781", "#1f5fae", "#9aa3ad", "#f2d7a6"
# 图 1：阈值曲线（小汽车）
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, side, sg in ((axs[0], "acc", 1), (axs[1], "dec", -1)):
    e = curves[0][side]; b = e["bins"]; vv = np.linspace(e["v_lo"], e["v_hi"], 200)
    for k, ls in ((1, ":"), (2, "--")):
        ax.plot(vv*0.6818, sg*at.threshold(curves, 0, side, k, vv), color=MUTED, lw=1, ls=ls, label=f"{k}σ (paper)")
    t1 = at.threshold(curves, 0, side, 1, vv); t2 = at.threshold(curves, 0, side, 2, vv)
    ax.plot(vv*0.6818, sg*(t1+4*(t2-t1)), color=EGO, lw=2, label="5σ curve")
    ax.scatter(b.v_center_ftps*0.6818, sg*(b["mean"]+5*b["sd"]), s=12, color=EGO, zorder=3, label="5σ per bin")
    ax.axhline(sg*glob[(0, side)], color=INK, lw=1, ls="-.", label="Global 5σ")
    ax.set_xlabel("Speed (mph)", color=MUTED); ax.set_ylabel("Accel (g)", color=MUTED)
    ax.set_title(f"Car {'acceleration' if side=='acc' else 'deceleration'} thresholds", fontsize=10, loc="left")
    ax.grid(alpha=0.25); ax.legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(f"{out}/threshold_curves.png", dpi=120); plt.close(fig)

def sheet(E, sgn, title, fn, ncol=4):
    n = len(E); nr = int(np.ceil(n/ncol))
    fig, axs = plt.subplots(nr*3, ncol, figsize=(4.2*ncol, 2.0*3*nr), squeeze=False, gridspec_kw=dict(hspace=0.3, wspace=0.28))
    for k, r in enumerate(E.itertuples()):
        R, C = (k//ncol)*3, k % ncol
        w = df[(df.Vehicle_ID == r.vid) & (df.Frame_ID.between(r.s-240, r.e+240))]
        t = (w.Frame_ID - r.s)/30.0
        a1, a2, a3 = axs[R, C], axs[R+1, C], axs[R+2, C]
        for ax in (a1, a2, a3):
            ax.axvspan(0, (r.e-r.s)/30.0, color=BAND, alpha=0.6, lw=0)
            ax.tick_params(labelsize=7, colors=MUTED); [s.set_color("#d0d7de") for s in ax.spines.values()]; ax.grid(alpha=0.25, lw=0.5)
        a1.plot(t, w.Speed_smooth*0.6818, color=EGO, lw=1.6, label="Subject")
        a1.plot(t, w.Preceding_Speed*0.6818, color=LEAD, lw=1.4, ls="--", label="Leader")
        a1.set_ylabel("Speed (mph)", fontsize=7, color=MUTED)
        pk = (w.Acc_from_vsmooth_g*sgn).max()*sgn
        lane = int(w.Lane_Num.mode().iloc[0]); trk = " truck" if r.tr else ""
        a1.set_title(f"V{r.vid}{trk}, lane {lane}, {LAB[r.vb]} mph, peak {pk:+.2f} g" + ("  [also global]" if r.also_global else ""),
                     fontsize=8, color=INK, loc="left")
        if k == 0: a1.legend(fontsize=6, frameon=False)
        side = "acc" if sgn > 0 else "dec"
        t1 = at.threshold(curves, r.tr, side, 1, w.Speed_smooth); t2 = at.threshold(curves, r.tr, side, 2, w.Speed_smooth)
        a2.plot(t, w.Acc_from_vsmooth_g, color=EGO, lw=1.6)
        a2.plot(t, sgn*(t1+4*(t2-t1)), color=INK, lw=1, ls=":", label="5σ curve")
        a2.axhline(sgn*glob[(r.tr, side)], color=MUTED, lw=1, ls="-.", label="Global 5σ")
        a2.axhline(0, color="#d0d7de", lw=0.8); a2.set_ylabel("Accel (g)", fontsize=7, color=MUTED)
        if k == 0: a2.legend(fontsize=6, frameon=False)
        a3.plot(t, w.Gap_bumper, color=EGO, lw=1.6)
        a3.set_ylabel("Gap (ft)", fontsize=7, color=MUTED); a3.set_xlabel("Time from event start (s)", fontsize=7, color=MUTED)
    for k in range(n, nr*ncol):
        for d in range(3): axs[(k//ncol)*3+d, k % ncol].axis("off")
    fig.suptitle(title, fontsize=12, color=INK, x=0.01, ha="left", y=1.0)
    fig.savefig(fn, dpi=100, bbox_inches="tight"); plt.close(fig)
A = acc if len(acc) <= 24 else acc.groupby("vb", group_keys=False).apply(lambda d: d.sample(min(4, len(d)), random_state=0))
sheet(A.sort_values("vb"), 1, f"Harsh acceleration, paper curve 5σ ≥1 s: {len(A)} of {len(acc)} events", f"{out}/accel.png")
D = dec.groupby("vb", group_keys=False).apply(lambda d: d.sample(min(2, len(d)), random_state=0))
sheet(D.sort_values("vb"), -1, f"Harsh deceleration, paper curve 5σ ≥1 s: 2 random per speed bin (of {len(dec)})", f"{out}/decel.png")
G = dec[dec.also_global].sort_values(["vb", "vid"]).reset_index(drop=True)
for i in range(0, len(G), 16):
    sheet(G.iloc[i:i+16], -1, f"New GT harsh deceleration (curve 5σ AND global 5σ, ≥1 s): {i+1}-{min(i+16, len(G))} of {len(G)}",
          f"{out}/gt_decel_{i//16+1}.png")
print("plotted", len(A), len(D))
