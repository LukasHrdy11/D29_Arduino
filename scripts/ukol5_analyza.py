#!/usr/bin/env python3
"""Task 5: evaluate photodiode linearity from data/ukol5_fotodioda/linearita.csv.

Distance series (tags 'max', 'max_rep2' = LED aimed at the signal maximum):
  model U = A / (r + d)^n   (n = 2 for a point source and a linear detector;
  with 1/r^2 geometry, U ~ I^gamma gives n = 2*gamma)
Superposition test (tags 'sup*', LED fixed at 10 cm): modulated amplitude vs
DC background level; for a linear detector the amplitude is constant.
Writes linearita.png and superpozice.png next to the data.
"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "data", "ukol5_fotodioda")
rows = list(csv.DictReader(open(os.path.join(D, "linearita.csv"))))
for r in rows:
    for k in ("r_cm", "delta", "sd_delta", "adc_off", "adc_max"):
        r[k] = float(r[k])

AIM_REL = 0.03      # aiming reproducibility (10 cm repeated: 20.78 vs 20.13)
DR = 0.1            # distance reading uncertainty [cm]
SAT = 1000          # ADC max above this -> saturated point
SCATTER = 0.56      # signal with barrier (10 cm): reflections + leakage [codes]

# ---------------------------------------------------------------- distance
aimed = [r for r in rows if r["tag"] in ("max", "max_rep2")]
by_r = {}
for r in aimed:
    by_r.setdefault(r["r_cm"], []).append(r["delta"])
pts, rejected = [], []
for rc, vals in sorted(by_r.items()):
    vals = np.array(vals)
    good = vals[vals > 0.9 * vals.max()]          # drop local-maximum aimings
    rejected += [(rc, v) for v in vals if v <= 0.9 * vals.max()]
    pts.append((rc, good.mean(), len(good)))
r_all = np.array([p[0] for p in pts])
U_all = np.array([p[1] for p in pts])
sat = np.array([any(x["adc_max"] >= SAT for x in aimed if x["r_cm"] == rc) for rc in r_all])
unaimed = [(r["r_cm"], r["delta"]) for r in rows if r["tag"] == "" and r["r_cm"] >= 2]

def model(r, A, d, n):
    return A / (r + d) ** n

def sigma(r, U, d, n):
    # aiming (relative) + distance reading propagated through the slope + stat
    return np.sqrt((AIM_REL * U) ** 2 + (n * U / (r + d) * DR) ** 2 + 0.03 ** 2)

fit_mask = (~sat) & (r_all >= 2)
r_f, U_f = r_all[fit_mask], U_all[fit_mask]
p = [U_f[0] * (r_f[0] + 0.5) ** 2, 0.5, 2.0]
for _ in range(5):                                  # iterate: sigma depends on d, n
    s = sigma(r_f, U_f, p[1], p[2])
    p, cov = curve_fit(model, r_f, U_f, p0=p, sigma=s, absolute_sigma=True)
s = sigma(r_f, U_f, p[1], p[2])
chi2 = np.sum(((U_f - model(r_f, *p)) / s) ** 2); dof = len(r_f) - 3
perr = np.sqrt(np.diag(cov))
A, d, n = p
# fixed n = 2 for comparison
p2, cov2 = curve_fit(lambda r, A, d: model(r, A, d, 2.0), r_f, U_f, p0=p[:2],
                     sigma=sigma(r_f, U_f, d, 2.0), absolute_sigma=True)
chi2_2 = np.sum(((U_f - model(r_f, *p2, 2.0)) / sigma(r_f, U_f, p2[1], 2.0)) ** 2)

print("Distance series (aimed), delta in ADC codes:")
for rc, U, k in pts:
    flag = " SATURATED" if rc in r_all[sat] else ("" if rc in r_f else " (not fitted)")
    print(f"  r = {rc:5.2f} cm  U = {U:8.2f}  (n_rep = {k}){flag}"
          + ("" if not (rc in r_f) else f"  fit {model(rc, *p):8.2f}  resid {100*(U/model(rc,*p)-1):+5.1f} %"))
print(f"  rejected local-maximum aimings: {rejected}")
print(f"fit U = A/(r+d)^n, r = {r_f.min():g}..{r_f.max():g} cm: "
      f"A = {A:.0f} ± {perr[0]:.0f}, d = {d:.2f} ± {perr[1]:.2f} cm, n = {n:.3f} ± {perr[2]:.3f}, "
      f"chi2/dof = {chi2:.1f}/{dof}")
print(f"  -> gamma = n/2 = {n/2:.3f} ± {perr[2]/2:.3f}")
print(f"fit with n = 2 fixed: A = {p2[0]:.0f}, d = {p2[1]:.2f} ± {np.sqrt(cov2[1,1]):.2f} cm, "
      f"chi2/dof = {chi2_2:.1f}/{len(r_f)-2}")
for rc, U in zip(r_all[nf_mask := (~fit_mask) & (~sat)], U_all[nf_mask]):
    m = model(rc, *p)
    print(f"  not fitted r = {rc} cm: U = {U:.0f}, extrapolated fit {m:.0f} ± {sigma(rc, m, d, n):.0f}"
          f" ({100*(U/m-1):+.0f} %)")
for rc, U in zip(r_all[sat], U_all[sat]):
    print(f"  saturated r = {rc} cm: U = {U:.0f} codes = {U*5/1024:.2f} V (at the rail; "
          f"extrapolation below 2 cm is not reliable, so no comparison with the fit)")
print(f"  scatter/reflection signal {SCATTER} codes = {100*SCATTER/U_f.min():.1f} % of the weakest point")

# -------------------------------------------------------------- superposition
sup = [r for r in rows if r["tag"].startswith("sup")]
sup.sort(key=lambda r: r["adc_off"])
print("\nSuperposition (LED fixed at 10 cm, box open):")
for r in sup:
    print(f"  {r['tag']:14s} background {r['adc_off']:6.1f}  amplitude {r['delta']:7.3f} ± {r['sd_delta']:.3f}")
same_geo = [r for r in sup if r["tag"] in ("sup3", "sup4", "sup5", "sup6", "sup7_zpet1")]
bg = np.array([r["adc_off"] for r in same_geo]); am = np.array([r["delta"] for r in same_geo])
k, q = np.polyfit(bg, am, 1)
rel = k / q * 1e2
print(f"  same phone position (sup3-7): amplitude = {q:.3f} + {k:.5f}*bg  "
      f"-> {rel*100:.2f} % per 100 codes; over {bg.min():.0f}-{bg.max():.0f}: "
      f"{100*(am.max()/am.min()-1):.1f} %")
# local gamma: amplitude ~ dU/dI ~ I^(gamma-1); I ~ (bg + amplitude/2)
Itot = bg + am / 2
g1 = np.polyfit(np.log(Itot), np.log(am), 1)[0]
print(f"  local exponent: d ln(amp)/d ln(level) = {g1:.4f} -> gamma = {1+g1:.3f}")

# ------------------------------------------------------------------- plots
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
rr = np.linspace(1, 14, 200)
a0 = ax[0]
a0.errorbar(r_f, U_f, sigma(r_f, U_f, d, n), fmt="o", color="C0", label="doladěno na maximum (fit)")
nf = (~fit_mask) & (~sat)
if nf.any():
    a0.plot(r_all[nf], U_all[nf], "s", color="C2", label="doladěno, mimo fit")
a0.plot(r_all[sat], U_all[sat], "v", color="C3", ms=9, label="saturace (4,9 V)")
if unaimed:
    a0.plot(*zip(*unaimed), "x", color="gray", label="bez doladění")
if rejected:
    a0.plot(*zip(*rejected), "+", color="k", ms=9, label="místní maximum (vyřazeno)")
a0.plot(rr, model(rr, *p), "-", color="C1",
        label=f"A/(r+d)^n: d = {d:.2f} cm, n = {n:.2f}±{perr[2]:.2f}")
a0.axhline(1023, color="C3", lw=0.6, ls=":")
a0.set(xscale="log", yscale="log", xlabel="r [cm]", ylabel="signál on−off [kód ADC]",
       title="Signál fotodiody vs. vzdálenost LED")
a0.legend(fontsize=8)
sec = a0.secondary_yaxis("right", functions=(lambda c: c * 5.0 / 1024 / 1e6 * 1e6,
                                               lambda i: i * 1024 / 5.0))
sec.set_ylabel("fotoproud [µA] (R_L = 1 MΩ)")

a1 = ax[1]
a1.errorbar(r_f, 100 * (U_f / model(r_f, *p) - 1), 100 * sigma(r_f, U_f, d, n) / U_f, fmt="o")
a1.axhline(0, color="k", lw=0.6)
a1.set(xlabel="r [cm]", ylabel="reziduum [%]", title=f"Rezidua fitu (χ²/ν = {chi2:.1f}/{dof})")

a2 = ax[2]
a2.errorbar([r["adc_off"] for r in sup], [r["delta"] for r in sup],
            [r["sd_delta"] for r in sup], fmt="o")
for r in sup:
    a2.annotate(r["tag"].replace("sup", ""), (r["adc_off"], r["delta"]), fontsize=7,
                xytext=(3, 3), textcoords="offset points")
bb = np.linspace(0, bg.max() * 1.05, 10)
a2.plot(bb, q + k * bb, "--", color="C1", label=f"lineární trend (body 3–7): {rel*100:+.2f} %/100 kódů")
a2.set(xlabel="stálé pozadí (LED off) [kód ADC]", ylabel="amplituda blikání [kód ADC]",
       title="Test superpozice (LED pevně na 10 cm)")
a2.legend(fontsize=8)
fig.tight_layout()
out = os.path.join(D, "linearita.png")
fig.savefig(out, dpi=130)
print(f"\nplot -> {out}")

# linearisation 1/sqrt(U) vs r (straight line for n = 2)
fig, a = plt.subplots(figsize=(6, 4.5))
a.plot(r_f, 1 / np.sqrt(U_f), "o", label="doladěno (fit)")
if unaimed:
    ru, Uu = zip(*unaimed); a.plot(ru, 1 / np.sqrt(Uu), "x", color="gray", label="bez doladění")
a.plot(rr, (rr + p2[1]) / np.sqrt(p2[0]), "-", label=f"n = 2: průsečík r = −d = {-p2[1]:.2f} cm")
a.set(xlabel="r [cm]", ylabel="1/√U  [kód$^{-1/2}$]", title="Linearizace: pro 1/r² přímka")
a.legend(fontsize=8); fig.tight_layout()
out2 = os.path.join(D, "linearizace.png"); fig.savefig(out2, dpi=130)
print(f"plot -> {out2}")
