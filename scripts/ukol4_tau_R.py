#!/usr/bin/env python3
"""Task 4: tau(R), C, R_pin, leakage and charge/discharge asymmetry from all runs.

  python3 ukol4_tau_R.py data/ukol4_rc_buffer/rc_*.json

Needs the <file>.json written by ukol4_analyza.py (and the CSVs next to them for the
local-tau test) and, for the comparison, data/ukol3_rc_puvodni/rc_*.json.

  1. repeats of one resistor merged (as in ukol3_tau_R.py): random part rep, stat
     -> sqrt(sum sq)/n (or spread of the repeats, if larger); systematic meth, dir,
     clock -> mean over the repeats (same R, same setup, fully correlated)
     charging / discharging alone: same, without the dir component
  2. effective-variance fit through the origin tau = C R (sigma_R = R tol / sqrt 3)
     for the mean, charging and discharging; free intercept tau0 = C R_pin as a check
  3. C_cal from the 22.1k (F, 1 %); resistance ratios R_i / R_22.1k = tau_i / tau_22.1k
  4. same resistors in task 3 (5V / GND wire, no pin in series) vs task 4 (D7 in
     series): delta tau = C R_pin is independent of the R tolerance
  5. leakage R_leak across C: charged level L = L0 R_leak / (R + R_leak) -> slope of
     the pre-step level of the discharging captures vs R
  6. local tau in voltage bands for charging and discharging: equal at the same
     voltage -> voltage dependent cause (C(U), ADC); different -> direction dependent
Graphs: tau_R.png, srovnani_10k_ukol3_ukol4.png, lokalni_tau.png
"""
import argparse, glob, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOL = {10e3: 0.10, 22.1e3: 0.01, 56e3: 0.05, 100e3: 0.10, 240e3: 0.10}   # as task 3
CAL = 22.1e3
ROOT = os.path.join(os.path.dirname(__file__), "..")

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
a = ap.parse_args()
runs = [json.load(open(f)) for f in a.files]
outdir = os.path.dirname(a.files[0])


def merge(rs, key=None):
    """Merged tau and its random / systematic error for a list of runs of one R.
    key None = mean of all captures (with dir), 'R' / 'F' = one direction only."""
    n = len(rs)
    if key is None:
        taus = [r["tau"] for r in rs]
        rand = np.sqrt(sum(r["comp"]["rep"] ** 2 + r["comp"]["stat"] ** 2 for r in rs)) / n
        syst = np.mean([np.sqrt(r["comp"]["meth"] ** 2 + r["comp"]["dir"] ** 2
                                + r["comp"]["clock"] ** 2) for r in rs])
    else:
        taus, rr = [], []
        for r in rs:
            t = np.array([c["tau"] for c in r["caps"] if c["ok"] and c["ph"] == key])
            taus.append(t.mean())
            rr.append(t.std(ddof=1) / np.sqrt(len(t)))
        rand = np.sqrt(np.sum(np.square(rr))) / n
        syst = np.mean([np.hypot(r["comp"]["meth"], r["comp"]["clock"]) for r in rs])
    if n > 1:
        rand = max(rand, np.std(taus, ddof=1) / np.sqrt(n))
    return np.mean(taus), np.hypot(rand, syst)


def fit_origin(R, T, ET, SR):
    C = np.sum(R * T) / np.sum(R * R)
    for _ in range(30):
        s2 = ET ** 2 + (C * SR) ** 2
        C = np.sum(R * T / s2) / np.sum(R * R / s2)
    return C, 1 / np.sqrt(np.sum(R * R / s2)), np.sum((T - C * R) ** 2 / s2)


def fit_free(R, T, ET, SR, C):
    k = C
    for _ in range(30):
        s2 = ET ** 2 + (k * SR) ** 2
        (k, q), cov = np.polyfit(R, T, 1, w=1 / np.sqrt(s2), cov="unscaled")
    return k, q, np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])


groups = {}
for r in runs:
    groups.setdefault(r["R"], []).append(r)
Rs = np.array(sorted(groups))
SR = np.array([R * TOL[R] / np.sqrt(3) for R in Rs])
res = {}
for key in (None, "R", "F"):
    m = [merge(groups[R], key) for R in Rs]
    res[key] = (np.array([x[0] for x in m]), np.array([x[1] for x in m]))
T, ET = res[None]

print("### Body τ(R) (opakování sloučena)")
print("| R | záznamů | τ [ms] | u(τ) [ms] | u/τ | τ nabíjení [ms] | τ vybíjení [ms] | "
      "(nab − vyb)/τ |")
print("|---|---|---|---|---|---|---|---|")
for i, R in enumerate(Rs):
    tR, eR = res["R"][0][i], res["R"][1][i]
    tF, eF = res["F"][0][i], res["F"][1][i]
    print(f"| {R / 1e3:g}k | {len(groups[R])} | {T[i] * 1e3:.3f} | {ET[i] * 1e3:.3f} | "
          f"{ET[i] / T[i]:.2%} | {tR * 1e3:.3f} ± {eR * 1e3:.3f} | {tF * 1e3:.3f} ± "
          f"{eF * 1e3:.3f} | {(tR - tF) / T[i]:+.3%} |")

print("\n### Fity τ = C·R (efektivní rozptyl, chyby R = tolerance/√3)")
fits = {}
for key, lab in ((None, "průměr"), ("R", "nabíjení"), ("F", "vybíjení")):
    t, et = res[key]
    C, eC, chi2 = fit_origin(Rs, t, et, SR)
    k, q, ek, eq = fit_free(Rs, t, et, SR, C)
    fits[key] = (C, eC)
    print(f"{lab}: C = {C * 1e6:.4f} ± {eC * 1e6:.4f} µF, chi2/dof = {chi2:.2f}/{len(Rs) - 1}; "
          f"volný úsek: tau0 = {q * 1e3:+.2f} ± {eq * 1e3:.2f} ms "
          f"(-> R_pin = {q / k:+.0f} ± {eq / k:.0f} Ω)")

ic = list(Rs).index(CAL)
Ccal = T[ic] / CAL
eCcal = Ccal * np.hypot(ET[ic] / T[ic], TOL[CAL] / np.sqrt(3))
print(f"\nC z 22,1k (F) samotného: C_cal = {Ccal * 1e6:.4f} ± {eCcal * 1e6:.4f} µF "
      f"(τ {ET[ic] / T[ic]:.2%}, R {TOL[CAL] / np.sqrt(3):.2%})")

# ---- task 3 for comparison ----
t3 = {}
for f in glob.glob(os.path.join(ROOT, "data/ukol3_rc_puvodni/rc_*.json")):
    if "vadny" in f:
        continue
    j = json.load(open(f))
    t3.setdefault(j["R"], []).append(j)
T3 = {}
for R, rs in t3.items():
    n = len(rs)
    tau = np.mean([r["tau"] for r in rs])
    rand = np.sqrt(sum(r["comp"]["rep"] ** 2 + r["comp"]["stat"] ** 2 for r in rs)) / n
    if n > 1:
        rand = max(rand, np.std([r["tau"] for r in rs], ddof=1) / np.sqrt(n))
    syst = np.mean([np.sqrt(r["comp"]["meth"] ** 2 + r["comp"]["dir"] ** 2
                            + r["comp"]["clock"] ** 2) for r in rs])
    T3[R] = (tau, np.hypot(rand, syst), np.mean([r["pts_per_tau"] for r in rs]),
             rs[0])

print("\n### Poměry odporů R_i / R_22,1k = τ_i / τ_22,1k (tolerance R se neuplatní)")
print("| R | úkol 4: poměr | u | → R = poměr · 22,1k [kΩ] | úkol 3: poměr | u |")
print("|---|---|---|---|---|---|")
for i, R in enumerate(Rs):
    if R == CAL:
        continue
    q4 = T[i] / T[ic]
    # full tau errors of both points (conservative: the clock part would cancel)
    e4 = q4 * np.hypot(ET[i] / T[i], ET[ic] / T[ic])
    a3, e3a = T3[R][0], T3[R][1]
    b3, e3b = T3[CAL][0], T3[CAL][1]
    q3 = a3 / b3
    e3 = q3 * np.hypot(e3a / a3, e3b / b3)
    print(f"| {R / 1e3:g}k | {q4:.5f} | {e4:.5f} ({e4 / q4:.2%}) | {q4 * 22.1:.3f} | "
          f"{q3:.5f} | {e3:.5f} ({e3 / q3:.2%}) |")

print("\n### Úkol 3 (vodič 5V/GND) vs úkol 4 (pin D7): Δτ = τ4 − τ3 = C·R_pin")
print("| R | τ úkol 3 [ms] | τ úkol 4 [ms] | Δτ [ms] | u(Δτ) [ms] | Δτ/τ | bodů na τ (3 → 4) |")
print("|---|---|---|---|---|---|---|")
D, ED = [], []
for i, R in enumerate(Rs):
    tau3, e3, p3, _ = T3[R]
    d, ed = T[i] - tau3, np.hypot(ET[i], e3)
    D.append(d)
    ED.append(ed)
    ppt4 = np.mean([r["pts_per_tau"] for r in groups[R]])
    print(f"| {R / 1e3:g}k | {tau3 * 1e3:.2f} ± {e3 * 1e3:.2f} | {T[i] * 1e3:.3f} ± "
          f"{ET[i] * 1e3:.3f} | {d * 1e3:+.2f} | {ed * 1e3:.2f} | {d / T[i]:+.2%} | "
          f"{p3:.1f} → {ppt4:.0f} |")
D, ED = np.array(D), np.array(ED)
w = 1 / ED ** 2
dm, edm = np.sum(w * D) / np.sum(w), 1 / np.sqrt(np.sum(w))
chi2d = np.sum((D - dm) ** 2 * w)
print(f"vážený průměr Δτ = {dm * 1e3:+.3f} ± {edm * 1e3:.3f} ms (chi2/dof = {chi2d:.1f}/"
      f"{len(D) - 1}) -> R_pin = {dm / Ccal:+.0f} ± {edm / Ccal:.0f} Ω")
rel = D / T
erel = ED / T
wr = 1 / erel ** 2
print(f"relativně: vážený průměr Δτ/τ = {np.sum(wr * rel) / np.sum(wr):+.3%} ± "
      f"{1 / np.sqrt(np.sum(wr)):.3%}")

# ---- 5. leakage from the charged level ----
L, EL, RL = [], [], []
for R in Rs:
    lv = [c["base"] for r in groups[R] for c in r["caps"] if c["ph"] == "F"]
    L.append(np.mean(lv))
    EL.append(np.std(lv, ddof=1) / np.sqrt(len(lv)))
L, EL = np.array(L), np.array(EL)
(sl, L0), cov = np.polyfit(Rs, L, 1, w=1 / EL, cov="unscaled")
esl = np.sqrt(cov[0, 0])
print("\n### Svod kondenzátoru: nabitá úroveň (průměr 50 vzorků před vybitím) vs R")
print("| R | úroveň [kód] | u [kód] |")
print("|---|---|---|")
for R, l, el in zip(Rs, L, EL):
    print(f"| {R / 1e3:g}k | {l:.3f} | {el:.3f} |")
lim = abs(sl) + 2 * esl                          # 2 sigma bound on |slope|
Rleak_min = L0 / lim
print(f"sklon = {sl * 1e3 * 1e3:+.3f} ± {esl * 1e6:.3f} kódu / MΩ -> R_leak > "
      f"{Rleak_min / 1e9:.2f} GΩ (2σ); vliv na τ u 240k < {240e3 / Rleak_min:.3%}")

# ---- 6. local tau in voltage bands ----
def load_csv(fn):
    meta, rows = {}, {}
    for ln in open(fn):
        if ln.startswith("#"):
            meta.update(re.findall(r"(\w+)=(\S+)", ln))
        elif ln[0] in "RF":
            p, r, i, v = ln.strip().split(",")
            rows.setdefault((p, int(r)), []).append(int(v))
    return meta, rows


edges = np.array([0.90, 0.70, 0.50, 0.35, 0.22, 0.12, 0.06])   # fraction of step left
loc = {"R": [], "F": []}
for r in runs:
    fn = os.path.join(outdir, r["name"] + ".csv")
    meta, rows = load_csv(fn)
    pre, dt, k = int(meta["pre"]), int(meta["dt_us"]) * 1e-6, r["clock"]
    for c in r["caps"]:
        if not c["ok"]:
            continue
        u = np.array(rows[(c["ph"], c["rep"])][pre:], float)
        t = np.arange(len(u)) * dt * k
        y = (u - c["B"]) / c["A"]
        out = []
        for hi, lo in zip(edges[:-1], edges[1:]):
            m = (y <= hi) & (y > lo)
            s = np.polyfit(t[m], np.log(y[m]), 1)[0]
            out.append((-1 / s / r["tau"], np.mean(u[m]) * 5 / 1023))
        loc[c["ph"]].append(out)
print("\n### Lokální τ v pásmech napětí (τ_lok / τ_záznamu, průměr přes všechny průběhy)")
print("| pásmo (zbývá skoku) | nabíjení: U [V] | τ_lok/τ | vybíjení: U [V] | τ_lok/τ |")
print("|---|---|---|---|---|")
LR, LF = np.array(loc["R"]), np.array(loc["F"])
for j, (hi, lo) in enumerate(zip(edges[:-1], edges[1:])):
    print(f"| {hi:.2f}–{lo:.2f} | {LR[:, j, 1].mean():.2f} | {LR[:, j, 0].mean():.4f} ± "
          f"{LR[:, j, 0].std(ddof=1) / np.sqrt(len(LR)):.4f} | {LF[:, j, 1].mean():.2f} | "
          f"{LF[:, j, 0].mean():.4f} ± {LF[:, j, 0].std(ddof=1) / np.sqrt(len(LF)):.4f} |")

# ---- graphs ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
C = fits[None][0]
ax1.errorbar(Rs / 1e3, T * 1e3, xerr=SR / 1e3, yerr=ET * 1e3, fmt="o", ms=5,
             color="#333333", capsize=3, lw=1)
rr = np.linspace(0, Rs.max() * 1.05, 10)
ax1.plot(rr / 1e3, C * rr * 1e3, "-", color="#1565c0", lw=1.5,
         label=f"τ = C·R, C = {C * 1e6:.3f} ± {fits[None][1] * 1e6:.3f} µF")
ax1.set(xlabel="R nominální [kΩ]", ylabel="τ [ms]",
        title="úkol 4: τ vs. R (chyby R = tolerance/√3)")
ax1.legend(frameon=False, loc="upper left")
x = np.arange(len(Rs))
for key, col, off, lab in (("R", "#c62828", -0.12, "nabíjení"), ("F", "#1565c0", 0.12,
                                                                    "vybíjení")):
    t, et = res[key]
    ax2.errorbar(x + off, (t / Rs / Ccal - 1) * 100, yerr=et / t * 100, fmt="o", ms=4,
                 color=col, capsize=3, lw=1, label=lab)
ax2.errorbar(x, (np.array([T3[R][0] for R in Rs]) / Rs / Ccal - 1) * 100,
             yerr=[T3[R][1] / T3[R][0] * 100 for R in Rs], fmt="s", ms=4, color="#999999",
             capsize=3, lw=1, label="úkol 3 (průměr)")
ax2.axhline(0, color="#333333", lw=0.8)
ax2.set_xticks(x, [f"{R / 1e3:g}k" for R in Rs])
ax2.set(ylabel="τ/(R_nom·C_cal) − 1 [%]",
        title="skutečné R vůči nominálu (C_cal z 22,1k)")
ax2.legend(frameon=False, fontsize=8)
for ax in (ax1, ax2):
    ax.grid(True, color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)
fig.tight_layout()
o1 = os.path.join(outdir, "tau_R.png")
fig.savefig(o1, dpi=150)
plt.close(fig)

# 10k overlay task 3 vs task 4
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
r4 = groups[10e3][0]
r3 = T3[10e3][3]
for d, col in (("vyb", "#1565c0"), ("nab", "#c62828")):
    e3, e4 = r3["example"][d], r4["example"][d]
    full = 5.0 if d == "vyb" else -5.0
    t3a = np.array(e3["t"]) + e3["tau"] * np.log(full / e3["A"])
    ax1.plot(t3a * 1e3, e3["u"], "s", ms=4, mfc="none", color="#999999",
             label="úkol 3: posílání po lince" if d == "vyb" else None)
    s4 = 5 / 1023
    ax1.plot(np.array(e4["t"]) * 1e3, np.array(e4["u"]) * s4, ".", ms=2, color=col,
             label=("úkol 4: buffer, DT = 0,5 ms" if d == "vyb" else None))
ax1.set(xlim=(0, 120), xlabel="t od kroku [ms]", ylabel="U_C [V]",
        title="10k: stejný kondenzátor, dvě metody vzorkování")
ax1.legend(frameon=False, fontsize=8, loc="center right")
labels = ["úkol 3", "úkol 4"]
vals = [T3[10e3][0] * 1e3, T[0] * 1e3]
errs = [T3[10e3][1] * 1e3, ET[0] * 1e3]
ax2.errorbar([0, 1], vals, yerr=errs, fmt="o", ms=6, color="#333333", capsize=5)
for i in range(2):
    ax2.annotate(f"{vals[i]:.2f} ± {errs[i]:.2f} ms", (i, vals[i]), xytext=(12, -4),
                 textcoords="offset points", fontsize=9)
ax2.set_xticks([0, 1], labels)
ax2.set_xlim(-0.5, 1.8)
ax2.set(ylabel="τ (10k) [ms]", title="τ pro 10k a jeho nejistota")
for ax in (ax1, ax2):
    ax.grid(True, color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)
fig.tight_layout()
o2 = os.path.join(outdir, "srovnani_10k_ukol3_ukol4.png")
fig.savefig(o2, dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 4.5))
for L_, col, lab in ((LR, "#c62828", "nabíjení"), (LF, "#1565c0", "vybíjení")):
    ax.errorbar(L_[:, :, 1].mean(0), (L_[:, :, 0].mean(0) - 1) * 100,
                yerr=L_[:, :, 0].std(0, ddof=1) / np.sqrt(len(L_)) * 100, fmt="o-", ms=4,
                color=col, capsize=3, lw=1, label=lab)
ax.axhline(0, color="#333333", lw=0.8)
ax.set(xlabel="napětí na kondenzátoru v pásmu [V]", ylabel="τ_lok / τ_fit − 1 [%]",
       title="lokální τ podle napětí (všechny R, 60 průběhů na směr)")
ax.legend(frameon=False)
ax.grid(True, color="#dddddd", lw=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
o3 = os.path.join(outdir, "lokalni_tau.png")
fig.savefig(o3, dpi=150)
print("\ngraphs ->", *(os.path.abspath(o) for o in (o1, o2, o3)))
