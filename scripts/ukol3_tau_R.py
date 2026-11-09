#!/usr/bin/env python3
"""Task 3: capacitance C from tau(R) over all resistors.

Usage (after ukol3_analyza.py has written the <file>.json summaries):
  python3 ukol3_tau_R.py data/ukol3_rc_puvodni/rc_*.json [--tol 56000=0.05 ...]

Steps:
  1. repeats of the same resistor are merged into one point (same physical R, so its
     R error is fully correlated and must not count twice):
       random part  rep, stat  -> sqrt(sum of squares) / n
       systematic   meth, dir, clock -> mean over the repeats
  2. every resistor has a tolerance tol (marking F = 1 %; unmarked ones assumed, see
     TOL); standard uncertainty sigma_R = R tol / sqrt(3) (rectangular)
  3. weighted fit through the origin tau = C R with effective variance
       sigma_eff^2 = sigma_tau^2 + (C sigma_R)^2   (iterated, R errors dominate)
     chi^2/dof is printed; a free intercept tau0 is fitted only as a check
  4. C from the 22.1k (F, 1 %) resistor alone = calibration value C_cal; the R of the
     other resistors is then derived as R = tau / C_cal (they all share the 1 % of the
     22.1k, i.e. the errors are common-mode)
Graphs: tau_R.png (tau vs R and C per resistor), rc_vsechny_R.png (one discharge and
one charge per R in real time and in t / tau).
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# tolerance (relative half-width) and how it is known
TOL = {10e3: (0.10, "neznámá (značení „N“), předpoklad ±10 %"),
       22.1e3: (0.01, "F = ±1 % (potisk 22K1 F)"),
       56e3: (0.05, "neznámá (bez písmene), předpoklad ±5 %"),
       100e3: (0.10, "neznámá (značení „N“), předpoklad ±10 %"),
       240e3: (0.10, "neznámá (značení „N“), předpoklad ±10 %")}
CAL = 22.1e3

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--tol", nargs="*", default=[], help="override, e.g. 56000=0.05")
a = ap.parse_args()
for item in a.tol:
    r, t = item.split("=")
    TOL[float(r)] = (float(t), "zadáno ručně")

files = [f for f in a.files if "vadny" not in os.path.basename(f)]
runs = [json.load(open(f)) for f in files]
outdir = os.path.dirname(files[0])

# ---- 1. merge repeats per resistor ----
pts = {}
for r in runs:
    pts.setdefault(r["R"], []).append(r)
P = []
for R, rs in sorted(pts.items()):
    n = len(rs)
    tau = np.mean([r["tau"] for r in rs])
    rand = np.sqrt(sum(r["comp"]["rep"] ** 2 + r["comp"]["stat"] ** 2 for r in rs)) / n
    if n > 1:                           # spread of the repeats, if larger
        rand = max(rand, np.std([r["tau"] for r in rs], ddof=1) / np.sqrt(n))
    syst = np.mean([np.sqrt(r["comp"]["meth"] ** 2 + r["comp"]["dir"] ** 2
                            + r["comp"]["clock"] ** 2) for r in rs])
    etau = np.hypot(rand, syst)
    tol, why = TOL.get(R, (0.10, "neznámá, předpoklad ±10 %"))
    P.append(dict(R=R, tau=tau, etau=etau, rand=rand, syst=syst, n_runs=n,
                  n_edges=sum(r["n"] for r in rs), tol=tol, sR=R * tol / np.sqrt(3),
                  why=why, ppt=np.mean([r["pts_per_tau"] for r in rs]), runs=rs))
R = np.array([p["R"] for p in P])
T = np.array([p["tau"] for p in P])
ET = np.array([p["etau"] for p in P])
SR = np.array([p["sR"] for p in P])

print("### Body τ(R) (opakování sloučena)")
print("| R nominal | tolerance R | záznamů | hran | bodů na τ | τ [ms] | u(τ) [ms] | "
      "u(τ)/τ | C = τ/R_nom [µF] | u(C) [µF] |")
print("|---|---|---|---|---|---|---|---|---|---|")
for p in P:
    C = p["tau"] / p["R"]
    eC = C * np.hypot(p["etau"] / p["tau"], p["tol"] / np.sqrt(3))
    p["C"], p["eC"] = C, eC
    print(f"| {p['R'] / 1e3:g}k | {p['why']} | {p['n_runs']} | {p['n_edges']} | "
          f"{p['ppt']:.1f} | {p['tau'] * 1e3:.2f} | {p['etau'] * 1e3:.2f} | "
          f"{p['etau'] / p['tau']:.2%} | {C * 1e6:.3f} | {eC * 1e6:.3f} |")

# ---- 3. effective-variance fits ----
C = np.sum(R * T) / np.sum(R * R)
for _ in range(20):
    s2 = ET**2 + (C * SR) ** 2
    C = np.sum(R * T / s2) / np.sum(R * R / s2)
eC = 1 / np.sqrt(np.sum(R * R / s2))
chi2 = np.sum((T - C * R) ** 2 / s2)
dof = len(R) - 1
print(f"\nfit through origin (effective variance): C = {C * 1e6:.4f} ± {eC * 1e6:.4f} µF, "
      f"chi2/dof = {chi2:.2f}/{dof} = {chi2 / dof:.2f}")
k, q = C, 0.0
for _ in range(20):
    s2b = ET**2 + (k * SR) ** 2
    (k, q), cov = np.polyfit(R, T, 1, w=1 / np.sqrt(s2b), cov="unscaled")
print(f"check, free intercept: C = {k * 1e6:.4f} ± {np.sqrt(cov[0, 0]) * 1e6:.4f} µF, "
      f"tau0 = {q * 1e3:.2f} ± {np.sqrt(cov[1, 1]) * 1e3:.2f} ms (expected 0)")
ku, qu = np.polyfit(R, T, 1)
print(f"unweighted line (nominal R, for comparison only): C = {ku * 1e6:.4f} µF, "
      f"tau0 = {qu * 1e3:.2f} ms")

# ---- 4. calibration on the 1 % resistor ----
pc = next(p for p in P if p["R"] == CAL)
Ccal, eCcal = pc["C"], pc["eC"]
print(f"\nC from {CAL / 1e3:g}k (F) alone: C_cal = {Ccal * 1e6:.4f} ± {eCcal * 1e6:.4f} µF "
      f"(τ {pc['etau'] / pc['tau']:.2%}, R {pc['tol'] / np.sqrt(3):.2%}); "
      f"difference to the all-point fit {(C - Ccal) / Ccal:+.2%}")
print("\n### Odvozené odpory R = τ / C_cal (všechny sdílejí 0,6 % nejistoty C_cal)")
print("| R nominal | R = τ/C_cal [kΩ] | u [kΩ] | u/R | odchylka od nominálu | v toleranci? |")
print("|---|---|---|---|---|---|")
for p in P:
    if p["R"] == CAL:
        continue
    Rd = p["tau"] / Ccal
    eR = Rd * np.hypot(p["etau"] / p["tau"], eCcal / Ccal)
    dev = Rd / p["R"] - 1
    print(f"| {p['R'] / 1e3:g}k | {Rd / 1e3:.2f} | {eR / 1e3:.2f} | {eR / Rd:.2%} | "
          f"{dev:+.2%} | {'ano' if abs(dev) <= p['tol'] else 'NE'} (±{p['tol']:.0%}) |")

# ---- graphs ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
ax1.errorbar(R / 1e3, T * 1e3, xerr=SR / 1e3, yerr=ET * 1e3, fmt="o", ms=5,
             color="#333333", capsize=3, lw=1)
rr = np.linspace(0, R.max() * 1.05, 10)
ax1.plot(rr / 1e3, C * rr * 1e3, "-", color="#1565c0", lw=1.5,
         label=f"τ = C·R, C = {C * 1e6:.3f} ± {eC * 1e6:.3f} µF")
ax1.set_xlabel("R nominální [kΩ]")
ax1.set_ylabel("τ [ms]")
ax1.set_title("časová konstanta vs. odpor (chyby R = tolerance/√3)")
ax1.legend(frameon=False, loc="upper left")
x = np.arange(len(P))
ax2.axhspan((Ccal - eCcal) * 1e6, (Ccal + eCcal) * 1e6, color="#1565c0", alpha=0.12,
            label=f"C z 22,1k (±1 %) = {Ccal * 1e6:.3f} ± {eCcal * 1e6:.3f} µF")
ax2.errorbar(x, [p["C"] * 1e6 for p in P], yerr=[p["eC"] * 1e6 for p in P], fmt="o",
             ms=5, color="#333333", capsize=3, lw=1)
ax2.set_xticks(x, [f"{p['R'] / 1e3:g}k\n±{p['tol']:.0%}" for p in P])
ax2.set_ylabel("C = τ / R_nominal [µF]")
ax2.set_title("C z jednotlivých odporů (chyba hlavně z tolerance R)")
ax2.legend(frameon=False, loc="upper left", fontsize=8)
for ax in (ax1, ax2):
    ax.grid(True, color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)
fig.tight_layout()
out1 = os.path.join(outdir, "tau_R.png")
fig.savefig(out1, dpi=150)
plt.close(fig)

# one charge + one discharge per R: real time and normalised time t / tau
cols = plt.get_cmap("viridis")(np.linspace(0.05, 0.85, len(P)))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
for p, c in zip(P, cols):
    ex = p["runs"][0]["example"]
    for d, e in ex.items():
        full = 5.0 if d == "vyb" else -5.0
        tt = np.array(e["t"]) + e["tau"] * np.log(full / e["A"])   # align to full step
        uu = np.array(e["u"])
        lab = f"{p['R'] / 1e3:g}k: τ = {p['tau'] * 1e3:.0f} ms" if d == "vyb" else None
        ax1.plot(tt, uu, ".", ms=2, color=c, label=lab)
        ax2.plot(tt / e["tau"], uu, ".", ms=2, color=c)
xx = np.linspace(0, 6, 200)
ax2.plot(xx, 5 * np.exp(-xx), "-", color="#c62828", lw=1, label="5 V · e^(−t/τ)")
ax2.plot(xx, 5 * (1 - np.exp(-xx)), "-", color="#c62828", lw=1)
ax2.axvline(1, color="#999999", lw=0.8, ls="--")
ax1.set_xlim(0, 5)
ax2.set_xlim(0, 6)
ax1.set_xlabel("t od přepnutí [s]")
ax2.set_xlabel("t / τ")
for ax in (ax1, ax2):
    ax.set_ylabel("U_C [V]")
    ax.grid(True, color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)
ax1.set_title("nabíjení a vybíjení pro všechny R")
ax2.set_title("totéž v čase t/τ: všechny křivky splynou")
ax1.legend(frameon=False, loc="center right", fontsize=8)
ax2.legend(frameon=False, loc="center right", fontsize=8)
fig.tight_layout()
out2 = os.path.join(outdir, "rc_vsechny_R.png")
fig.savefig(out2, dpi=150)
print("\ngraphs ->", os.path.abspath(out1), os.path.abspath(out2))
