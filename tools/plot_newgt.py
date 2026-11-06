"""新 GT 候选（按速度分段 5σ、持续 ≥1 s）的复核图：急加速全部 + 急减速每个速度段 2 个。"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
src, out = sys.argv[1], sys.argv[2]
cols = ["Vehicle_ID","Frame_ID","Acc_from_vsmooth_g","is_truck","Speed_smooth","Lane_Num","Preceding_Speed",
        "Gap_bumper","Leader_ID","Edge","BadTrack","Extrapolated","RepairZone","BadVehicle"]
raw = pd.read_csv(src, usecols=cols).sort_values(["Vehicle_ID","Frame_ID"])
df = raw[raw[cols[9:]].fillna(0).sum(1) == 0].copy()
df["tr"] = df.is_truck.fillna(0).astype(int)
EDGES = [0,15,30,45,60,75,200]; LAB = ["0-15","15-30","30-45","45-60","60-75",">75"]
df["vb"] = pd.cut(df.Speed_smooth*0.6818, EDGES, right=False, labels=False)
st = df.groupby(["tr","vb"]).Acc_from_vsmooth_g.agg(["mean","std"])
df = df.join(st, on=["tr","vb"])
df["z"] = (df.Acc_from_vsmooth_g - df["mean"]) / df["std"]
g = df.groupby("Vehicle_ID")["z"]
df["za"] = g.transform(lambda s: s.rolling(30).min()); df["zd"] = g.transform(lambda s: s.rolling(30).max())
def events(col, sgn):
    m = (df[col]*sgn >= 5).fillna(False).to_numpy(); v = df.Vehicle_ID.to_numpy(); f = df.Frame_ID.to_numpy()
    ev, i = [], 0
    while i < len(m):
        if m[i]:
            j = i
            while j+1 < len(m) and m[j+1] and v[j+1] == v[i] and f[j+1] == f[j]+1: j += 1
            ev.append((v[i], f[i]-29, f[j], int(df.vb.iloc[i]))); i = j+1
        else: i += 1
    return pd.DataFrame(ev, columns=["vid","s","e","vb"])
acc, dec = events("za", 1), events("zd", -1)
dec_s = dec.groupby("vb", group_keys=False).apply(lambda d: d.sample(min(2, len(d)), random_state=0))
acc.assign(kind="accel").to_csv(f"{out}/accel_events.csv", index=False)
dec.assign(kind="decel").to_csv(f"{out}/decel_events.csv", index=False)
INK, MUTED, EGO, LEAD, BAND = "#1f2328", "#6e7781", "#1f5fae", "#9aa3ad", "#f2d7a6"
def sheet(E, sgn, title, fn, ncol=4):
    n = len(E); nr = int(np.ceil(n/ncol))
    fig, axs = plt.subplots(nr*3, ncol, figsize=(4.2*ncol, 2.0*3*nr), squeeze=False,
                            gridspec_kw=dict(hspace=0.25, wspace=0.28))
    for k, r in enumerate(E.itertuples()):
        R, C = (k//ncol)*3, k % ncol
        w = df[(df.Vehicle_ID == r.vid) & (df.Frame_ID.between(r.s-240, r.e+240))]
        t = (w.Frame_ID - r.s)/30.0
        a1, a2, a3 = axs[R, C], axs[R+1, C], axs[R+2, C]
        for a in (a1, a2, a3):
            a.axvspan(0, (r.e-r.s)/30.0, color=BAND, alpha=0.6, lw=0)
            a.tick_params(labelsize=7, colors=MUTED); [s.set_color("#d0d7de") for s in a.spines.values()]
            a.grid(alpha=0.25, lw=0.5)
        a1.plot(t, w.Speed_smooth*0.6818, color=EGO, lw=1.6, label="Subject")
        a1.plot(t, w.Preceding_Speed*0.6818, color=LEAD, lw=1.4, ls="--", label="Leader")
        a1.set_ylabel("Speed (mph)", fontsize=7, color=MUTED)
        pk = (w.Acc_from_vsmooth_g*sgn).max()*sgn
        lane = int(w.Lane_Num.mode().iloc[0]); trk = " truck" if w.tr.iloc[0] else ""
        a1.set_title(f"V{r.vid}{trk}, lane {lane}, {LAB[r.vb]} mph, peak {pk:+.2f} g", fontsize=8, color=INK, loc="left")
        if k == 0: a1.legend(fontsize=6, frameon=False, loc="best")
        a2.plot(t, w.Acc_from_vsmooth_g, color=EGO, lw=1.6)
        a2.plot(t, w["mean"] + sgn*5*w["std"], color=INK, lw=1, ls=":", label="5σ (this speed)")
        a2.axhline(0, color="#d0d7de", lw=0.8)
        a2.set_ylabel("Accel (g)", fontsize=7, color=MUTED)
        if k == 0: a2.legend(fontsize=6, frameon=False, loc="best")
        a3.plot(t, w.Gap_bumper, color=EGO, lw=1.6)
        a3.set_ylabel("Gap (ft)", fontsize=7, color=MUTED); a3.set_xlabel("Time from event start (s)", fontsize=7, color=MUTED)
    for k in range(n, nr*ncol):
        for d in range(3): axs[(k//ncol)*3+d, k % ncol].axis("off")
    fig.suptitle(title, fontsize=12, color=INK, x=0.01, ha="left", y=0.995)
    fig.savefig(fn, dpi=110, bbox_inches="tight"); plt.close(fig)
sheet(acc.sort_values("vb"), 1, f"Harsh acceleration: all {len(acc)} events (speed-binned 5σ, ≥1 s; shaded = event)", f"{out}/accel_all.png")
sheet(dec_s.sort_values("vb"), -1, f"Harsh deceleration: 2 random events per speed bin (of {len(dec)})", f"{out}/decel_sample.png")
print("accel", len(acc), "decel", len(dec), "sampled", len(dec_s))
