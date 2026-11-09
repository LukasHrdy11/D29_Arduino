#!/usr/bin/env python3
"""Figures for the D29 protocol ("Arduino ve fyzice") from the saved measurement data.

Reads CSV/JSON files written by the ukol*_*.py scripts and writes vector PDFs
graf_*.pdf plus a text summary protokol_grafy_vystup.txt with the numbers quoted
in the protocol (recomputed here, so the text can be checked against the data).

    python3 scripts/protokol_grafy.py [--data DIR] [--out DIR]

Defaults: data = ../outputs (project) or ../data (public repository),
out = ../protokol (project) or ../figures (repository).
Graph style comes from praktikum_lib (palette from ../styl_grafu.json next to it).
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
import praktikum_lib as pl  # noqa: E402  (sets the matplotlib style on import)
import matplotlib.pyplot as plt  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402


def first_dir(*cands):
    for c in cands:
        if os.path.isdir(c):
            return c
    return cands[-1]


ap = argparse.ArgumentParser()
ap.add_argument("--data", default=first_dir(os.path.join(HERE, "..", "data"),
                                            os.path.join(HERE, "..", "data")))
ap.add_argument("--out", default=first_dir(os.path.join(HERE, "..", "protokol"),
                                           os.path.join(HERE, "..", "figures")))
a = ap.parse_args()
D, OUT = a.data, a.out
os.makedirs(OUT, exist_ok=True)
LOG = []
LSB_MV = 5000 / 1024          # 1 ADC code in mV for Vcc = 5.0 V
W1, H1 = 6.4, 3.9             # one panel, printed at 0.85 text width
W2, H2 = 7.6, 3.5             # two panels, printed at full text width


def out(line=""):
    LOG.append(line)
    print(line)


def save(fig, name, rect=None):
    fig.tight_layout(rect=rect)
    fig.savefig(os.path.join(OUT, name), metadata={"CreationDate": None})  # reproducible files
    plt.close(fig)
    out(f"-> {name}")


def bp(i):
    """pl.body(i) without errorbar-only arguments, for ax.plot."""
    return {k: v for k, v in pl.body(i).items() if k not in ("capsize", "elinewidth", "ecolor", "capthick")}


LOGF = plt.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))   # 0,5 / 10 on log axes


def P(*parts):
    return os.path.join(D, *parts)


def fmtc(x, nd):
    """Number with decimal comma (for labels)."""
    return f"{x:.{nd}f}".replace(".", ",")


# ============================================================ 1. VA characteristic of LEDs
out("## 1. VA charakteristika LED (graf_va_led.pdf)")


def load_va(path):
    meta, rows = {}, []
    for line in open(path):
        line = line.strip()
        m = re.match(r"#\s*(\w+)=(.*)", line)
        if m:
            meta[m.group(1)] = m.group(2)
        elif line and line[0].isdigit():
            rows.append([float(x) for x in line.split(",")])
    d = np.array(rows)
    vcc, R = float(meta["vcc_mV"]) / 1000, float(meta.get("R_ohm", 220))
    u1 = d[:, 2] * vcc / 1024
    return (d[:, 1] - d[:, 2]) * vcc / 1024, u1 / R * 1e3     # U_LED [V], I [mA]


def u_va(u, i, vcc=5.03, R=220.0):
    """Full standard uncertainty of a VA point (as in eq. u_U of the protocol):
    common Vcc scale (10 % uniform), ADC accuracy ±2 LSB uniform on A0 and A1,
    R tolerance 1 % uniform. Returns u_U [V], u_I [mA]."""
    lsb, sv = vcc / 1024, 0.10 / np.sqrt(3)
    a = 2 / np.sqrt(3) * lsb
    uU = np.sqrt((u * sv) ** 2 + 2 * a ** 2)
    uI = np.sqrt((i * sv) ** 2 + (a / R * 1e3) ** 2 + (i * 0.01 / np.sqrt(3)) ** 2)
    return uU, uI


def u_at(u, i, level):
    """Voltage where the (monotonic part of the) sweep first reaches current level [mA]."""
    j = np.argmax(i >= level)
    return np.interp(level, i[j - 1:j + 1], u[j - 1:j + 1])


def u_lin(u, i, f=0.5):
    top = i > f * i.max()
    k, q = np.polyfit(u[top], i[top], 1)
    return -q / k, k, q


LED = [("cervena", "červená"), ("zluta", "žlutá"), ("modra", "modrá")]
LED_BARVA = ["#c62828", "#c99700", "#1f4fb4"]   # colour of the LED itself
fig, ax1 = plt.subplots(figsize=(W2, H2 * 1.35))
for s, (key, lab) in enumerate(LED):
    files = sorted(glob.glob(P("ukol2_va_led", f"{key}_propustny*.csv")))
    vals = [u_lin(*load_va(f))[0] for f in files]
    out(f"  {lab}: U_lin ze {len(files)} sweepů = " + ", ".join(f"{v:.4f}" for v in vals)
        + f" V, průměr {np.mean(vals):.4f} V")
    u, i = load_va(files[0])
    sel = slice(None, None, 8)
    ax1.plot(u[sel], i[sel], ls="none", marker=pl.body(s)["marker"], ms=2.2,
             color=LED_BARVA[s], alpha=0.55, label=lab)
    # error bars (full uncertainty incl. the common Vcc scale) on a sparse subset
    eb = np.unique([np.argmax(i >= lv) for lv in (0.5, 2, 4, 6, 8, 10, 12) if i.max() >= lv])
    uU, uI = u_va(u[eb], i[eb])
    ax1.errorbar(u[eb], i[eb], xerr=uU, yerr=uI, fmt="none",
                 ecolor=LED_BARVA[s], elinewidth=0.8, capsize=2)
    U0, k, q = u_lin(u, i)
    uu = np.linspace(U0, u.max() + 0.05, 20)
    ax1.plot(uu, k * uu + q, color="white", lw=3.4, zorder=4)          # halo over the points
    ax1.plot(uu, k * uu + q, color=LED_BARVA[s], ls=(0, (5, 2)), lw=1.8, zorder=5,
             label=f"přímka pro $U_\\mathrm{{lin}}$ ({lab})")
    u01, u1 = u_at(u, i, 0.1), u_at(u, i, 1.0)
    ax1.plot([u01, u1], [0.1, 1.0], ls="none", marker="D", ms=5, mfc="white", mec="k",
             mew=1.0, zorder=6, label="$U_{0,1}$ a $U_1$" if s == 0 else None)
LAMBDA = {"cervena": 630, "zluta": 590, "modra": 470}     # nominal wavelengths / nm
out("  1240/λ (nominální λ): " + ", ".join(f"{k_} {1239.84 / v:.3f} V" for k_, v in LAMBDA.items()))
q_meas, eq_meas = 1.406, 0.010          # U_lin blue/red with its (Vcc-free) uncertainty, VYSLEDKY ukol2
q_pred = LAMBDA["cervena"] / LAMBDA["modra"]
out(f"  poměr U_lin modrá/červená {q_meas} ± {eq_meas} vs. λ_č/λ_m = {q_pred:.3f}: "
    f"rozdíl {(q_meas - q_pred) / eq_meas:.1f} σ; vysvětlil by ho λ_č = {q_meas * 470:.0f} nm "
    f"nebo λ_m = {630 / q_meas:.0f} nm")
ur, ir = load_va(P("ukol2_va_led", "cervena_zaverny.csv"))
out(f"  závěrný směr: max |A1 kód| = {np.max(np.abs(ir)) * 220 / 1e3 / (5.03 / 1024):.0f}"
    f" -> |I| < 0,5 LSB / 220 Ω = {0.5 * 5.03 / 1024 / 220 * 1e6:.0f} µA")
ax1.set(xlabel="$U$ / V", ylabel="$I$ / mA", xlim=(1.5, 3.05), ylim=(-0.4, 14.5))
ax1.xaxis.set_major_formatter(pl.Carka())
pl.legenda_nad_grafem(ax1, ncol=3)
pl.carky_na_osach(ax1)
save(fig, "graf_va_led.pdf")

# ============================================================ 2.+3. tau(R), tasks 3 and 4
out("\n## 2. τ(R), úkoly 3 a 4 (graf_tau_R.pdf, graf_srovnani_10k.pdf)")
TOL = {10e3: 0.10, 22.1e3: 0.01, 56e3: 0.05, 100e3: 0.10, 240e3: 0.10}
CAL = 22.1e3
runs4 = [json.load(open(f)) for f in sorted(glob.glob(P("ukol4_rc_buffer", "rc_*.json")))]


def merge(rs, key=None):
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


g4 = {}
for r in runs4:
    g4.setdefault(r["R"], []).append(r)
Rs = np.array(sorted(g4))
SR = np.array([R * TOL[R] / np.sqrt(3) for R in Rs])
res = {k: np.array([merge(g4[R], k) for R in Rs]).T for k in (None, "R", "F")}
T4, ET4 = res[None]

g3 = {}
for f in glob.glob(P("ukol3_rc_puvodni", "rc_*.json")):
    j = json.load(open(f))
    g3.setdefault(j["R"], []).append(j)
T3, ET3, PPT3 = [], [], []
for R in Rs:
    rs = g3[R]
    n = len(rs)
    rand = np.sqrt(sum(r["comp"]["rep"] ** 2 + r["comp"]["stat"] ** 2 for r in rs)) / n
    if n > 1:
        rand = max(rand, np.std([r["tau"] for r in rs], ddof=1) / np.sqrt(n))
    syst = np.mean([np.sqrt(r["comp"]["meth"] ** 2 + r["comp"]["dir"] ** 2
                            + r["comp"]["clock"] ** 2) for r in rs])
    T3.append(np.mean([r["tau"] for r in rs]))
    ET3.append(np.hypot(rand, syst))
    PPT3.append(np.mean([r["pts_per_tau"] for r in rs]))
T3, ET3 = np.array(T3), np.array(ET3)

out("| R | τ3 / ms | τ4 / ms | τ4 nab / ms | τ4 vyb / ms | bodů na τ (3) |")
for i, R in enumerate(Rs):
    out(f"| {R / 1e3:g}k | {T3[i] * 1e3:.2f} ± {ET3[i] * 1e3:.2f} | {T4[i] * 1e3:.3f} ± "
        f"{ET4[i] * 1e3:.3f} | {res['R'][0][i] * 1e3:.3f} ± {res['R'][1][i] * 1e3:.3f} | "
        f"{res['F'][0][i] * 1e3:.3f} ± {res['F'][1][i] * 1e3:.3f} | {PPT3[i]:.1f} |")
C4, eC4, chi4 = fit_origin(Rs, T4, ET4, SR)
C3, eC3, chi3 = fit_origin(Rs, T3, ET3, SR)
ic = list(Rs).index(CAL)
Ccal = T4[ic] / CAL
eCcal = Ccal * np.hypot(ET4[ic] / T4[ic], TOL[CAL] / np.sqrt(3))
out(f"  úkol 3: C = {C3 * 1e6:.4f} ± {eC3 * 1e6:.4f} µF, chi2 = {chi3:.2f}/4")
out(f"  úkol 4: C = {C4 * 1e6:.4f} ± {eC4 * 1e6:.4f} µF, chi2 = {chi4:.2f}/4")
out(f"  úkol 4: C_cal(22,1k) = {Ccal * 1e6:.4f} ± {eCcal * 1e6:.4f} µF")
for i, R in enumerate(Rs):
    if R != CAL:
        q = T4[i] / T4[ic]
        e = q * np.hypot(ET4[i] / T4[i], ET4[ic] / T4[ic])
        out(f"  R/R22 {R / 1e3:g}k: {q:.5f} ± {e:.5f} -> {q * 22.1:.3f} kΩ")
Dt = T4 - T3
EDt = np.hypot(ET4, ET3)
w = 1 / EDt ** 2
out(f"  Δτ = τ4 − τ3: vážený průměr {np.sum(w * Dt) / np.sum(w) * 1e3:+.3f} ± "
    f"{1 / np.sqrt(np.sum(w)) * 1e3:.3f} ms")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W2, H2))
ax1.errorbar(Rs / 1e3, T4 * 1e3, xerr=SR / 1e3, yerr=ET4 * 1e3, label="úkol 4 (buffer)",
             **pl.body(0))
rr = np.linspace(0, Rs.max() * 1.05, 10)
ax1.plot(rr / 1e3, C4 * rr * 1e3, label=r"$\tau = RC$", **pl.cara(0))
ax1.set(xlabel=r"$R_\mathrm{nom}$ (značení) / k$\Omega$", ylabel=r"$\tau$ / ms")
pl.legenda_nad_grafem(ax1)
pl.carky_na_osach(ax1)
x = np.arange(len(Rs))
dev = {}
for s, (key, lab, off) in enumerate((("R", "úkol 4, nabíjení", -0.12),
                                     ("F", "úkol 4, vybíjení", 0.12))):
    t, et = res[key]
    dev[key] = (t / Rs / Ccal - 1) * 100
    ax2.errorbar(x + off, dev[key], yerr=et / t * 100, label=lab, **pl.body(s))
ax2.errorbar(x, (T3 / Rs / Ccal - 1) * 100, yerr=ET3 / T3 * 100, label="úkol 3",
             **{**pl.body(2), "mfc": "none", "alpha": 0.6})
# charging vs. discharging of the same resistor: bracket with the difference in %
for j in range(len(Rs)):
    lo, hi = dev["R"][j], dev["F"][j]
    ax2.plot([x[j] - 0.12, x[j] + 0.12], [lo, hi], color="k", lw=1.1, zorder=6)
    rel = (res["F"][0][j] - res["R"][0][j]) / res[None][0][j] * 100
    ax2.annotate(f"{fmtc(rel, 2)} %", (x[j] + 0.2, (lo + hi) / 2 - 0.25), fontsize=7.5,
                 va="center", ha="left")
    out(f"  vyb − nab {Rs[j] / 1e3:g}k: {rel:.3f} % τ")
ax2.axhline(0, color="0.35", lw=0.8)
ax2.set_xticks(x, [fmtc(R / 1e3, 1).replace(",0", "") for R in Rs])
ax2.set_xlim(-0.5, len(Rs) - 0.2)
ax2.set(xlabel=r"$R_\mathrm{nom}$ (značení) / k$\Omega$",
        ylabel=r"$\tau/(R_\mathrm{nom}C_\mathrm{cal}) - 1$ / %")
pl.legenda_nad_grafem(ax2, ncol=2)
ax2.yaxis.set_major_formatter(pl.Carka())
save(fig, "graf_tau_R.pdf")

# whole record of task 3, 56k: repeated charging and discharging by the hand-moved wire
fig, ax = plt.subplots(figsize=(W2, H2 * 0.95))
d56 = np.genfromtxt(P("ukol3_rc_puvodni", "rc_56k_rep1.csv"), delimiter=",", comments="#",
                    skip_header=5)
t56 = (d56[:, 0] - d56[0, 0]) / 1e3 * 1.001807          # Arduino ms -> s, clock factor k
ax.plot(t56, d56[:, 1], color=pl.barva(0), lw=0.8)
ax.plot(t56, d56[:, 1], ls="none", marker=".", ms=1.2, color=pl.barva(0))
ax.set(xlabel="$t$ / s", ylabel="$U_C$ / V", xlim=(0, t56[-1]))
pl.carky_na_osach(ax)
nab = int(np.sum(np.diff((d56[:, 1] > 2.5).astype(int)) == 1))
out(f"  56k rep1 (úkol 3): {len(d56)} vzorků za {t56[-1]:.1f} s, přechodů nahoru přes 2,5 V: {nab}")
save(fig, "graf_56k_zaznam.pdf")

# 10k: both methods overlaid
fig, ax = plt.subplots(figsize=(W1, H1 * 0.95))
r4 = g4[10e3][0]
r3 = g3[10e3][0]
for d in ("vyb", "nab"):
    e3, e4 = r3["example"][d], r4["example"][d]
    full = 5.0 if d == "vyb" else -5.0
    t3a = np.array(e3["t"]) + e3["tau"] * np.log(full / e3["A"])
    ax.plot(np.array(e4["t"]) * 1e3, np.array(e4["u"]) * 5 / 1023, ls="none", marker=".",
            ms=2.5, color=pl.barva(0), label="úkol 4: buffer, $\\Delta t = 0{,}5$ ms")
    ax.plot(t3a * 1e3, e3["u"], ls="none", marker="s", ms=4.5, mfc="none", color=pl.barva(1),
            label="úkol 3: původní metoda")
ax.set(xlim=(0, 120), xlabel="$t$ / ms", ylabel="$U_C$ / V")
pl.legenda_nad_grafem(ax)
pl.carky_na_osach(ax)
save(fig, "graf_srovnani_10k.pdf")


# local tau in voltage bands
def load_rc4(fn):
    meta, rows = {}, {}
    for ln in open(fn):
        if ln.startswith("#"):
            meta.update(re.findall(r"(\w+)=(\S+)", ln))
        elif ln[0] in "RF":
            p_, r_, _, v = ln.strip().split(",")
            rows.setdefault((p_, int(r_)), []).append(int(v))
    return meta, rows


edges = np.array([0.90, 0.70, 0.50, 0.35, 0.22, 0.12, 0.06])
loc = {"R": [], "F": []}
for r in runs4:
    meta, rows = load_rc4(P("ukol4_rc_buffer", r["name"] + ".csv"))
    pre, dt, k = int(meta["pre"]), int(meta["dt_us"]) * 1e-6, r["clock"]
    for c in r["caps"]:
        if not c["ok"]:
            continue
        u = np.array(rows[(c["ph"], c["rep"])][pre:], float)
        t = np.arange(len(u)) * dt * k
        y = (u - c["B"]) / c["A"]
        o = []
        for hi, lo in zip(edges[:-1], edges[1:]):
            m = (y <= hi) & (y > lo)
            sl = np.polyfit(t[m], np.log(y[m]), 1)[0]
            o.append((-1 / sl / r["tau"], np.mean(u[m]) * 5 / 1023))
        loc[c["ph"]].append(o)
# (figure dropped from the protocol, the numbers are quoted in the text)
for s, (key, lab) in enumerate((("R", "nabíjení"), ("F", "vybíjení"))):
    L_ = np.array(loc[key])
    m = (L_[:, :, 0].mean(0) - 1) * 100
    e = L_[:, :, 0].std(0, ddof=1) / np.sqrt(len(L_)) * 100
    out(f"  lokální τ {lab} ({len(L_)} průběhů): " + ", ".join(
        f"{u_:.2f} V: {v:+.2f} ± {e_:.2f} %" for u_, v, e_ in zip(L_[:, :, 1].mean(0), m, e)))

# ============================================================ 4. photodiode linearity
out("\n## 4. Linearita fotodiody (graf_linearita.pdf)")
rows = list(csv.DictReader(open(P("ukol5_fotodioda", "linearita.csv"))))
for r in rows:
    for k_ in ("r_cm", "delta", "sd_delta", "adc_off", "adc_max"):
        r[k_] = float(r[k_])
AIM, DR = 0.03, 0.1
aimed = [r for r in rows if r["tag"] in ("max", "max_rep2")]
by_r = {}
for r in aimed:
    by_r.setdefault(r["r_cm"], []).append(r["delta"])
pts, rejected = [], []
for rc, vals in sorted(by_r.items()):
    vals = np.array(vals)
    good = vals[vals > 0.9 * vals.max()]
    rejected += [(rc, v) for v in vals if v <= 0.9 * vals.max()]
    pts.append((rc, good.mean()))
r_all, U_all = np.array(pts).T
sat = np.array([any(x["adc_max"] >= 1000 for x in aimed if x["r_cm"] == rc) for rc in r_all])


def model(r, A, d, n):
    return A / (r + d) ** n


def sig(r, U, d, n):
    return np.sqrt((AIM * U) ** 2 + (n * U / (r + d) * DR) ** 2 + 0.03 ** 2)


fm = (~sat) & (r_all >= 2)
r_f, U_f = r_all[fm], U_all[fm]
p = [U_f[0] * (r_f[0] + 0.5) ** 2, 0.5, 2.0]
for _ in range(5):
    p, cov = curve_fit(model, r_f, U_f, p0=p, sigma=sig(r_f, U_f, p[1], p[2]),
                       absolute_sigma=True)
sU = sig(r_f, U_f, p[1], p[2])
chi2 = np.sum(((U_f - model(r_f, *p)) / sU) ** 2)
pe = np.sqrt(np.diag(cov))
out(f"  fit U = A/(r+d)^n: d = {p[1]:.2f} ± {pe[1]:.2f} cm, n = {p[2]:.3f} ± {pe[2]:.3f}, "
    f"chi2 = {chi2:.1f}/{len(r_f) - 3}; gamma = {p[2] / 2:.3f} ± {pe[2] / 2:.3f}")
n_good = {rc: sum(1 for v in by_r[rc] if v > 0.9 * max(by_r[rc])) for rc in by_r}


def pm(x, u):
    """x ± u rounded to the uncertainty (2 digits if it starts with 1, else 1), decimal comma."""
    e = int(np.floor(np.log10(u)))
    nd = max(0, -(e - (1 if u / 10 ** e < 2 else 0)))
    q = 10.0 ** -nd
    return f"${fmtc(round(x / q) * q, nd)} \\pm {fmtc(round(u / q) * q, nd)}$"


out("  tabulka (r | ΔN ± 3 % | U/mV = I/nA ± 3 % | počet měření | poznámka):")
for rc, U in zip(r_all, U_all):
    s_ = rc in r_all[sat]
    note = "saturace" if s_ else ("mimo fit" if rc < 2 else "")
    dn = f"${U:.0f}$" if s_ else pm(U, AIM * U)
    uu = f"${U * LSB_MV:.0f}$" if s_ else pm(U * LSB_MV, AIM * U * LSB_MV)
    out(f"    ${fmtc(rc, 1 if rc % 1 else 0)}$ & {dn} & {uu} & {n_good[rc]} & {note} \\\\")
sup = sorted([r for r in rows if r["tag"].startswith("sup")], key=lambda r: r["adc_off"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W2, H2))
ax1.set(xscale="log", yscale="log")
ax1.errorbar(r_f, U_f, yerr=sU, label="ve fitu", **pl.body(0))
nf = (~fm) & (~sat)
ax1.plot(r_all[nf], U_all[nf], label="mimo fit", **bp(1))
ax1.plot(r_all[sat], U_all[sat], label="saturace", **bp(2))
rr = np.linspace(1.5, 14, 200)
ax1.plot(rr, model(rr, *p), label="fit $A/(r+d)^n$", **pl.cara(0))
ax1.set(xlabel="$r$ / cm", ylabel=r"$\Delta N$ / kód")
pl.legenda_nad_grafem(ax1, ncol=2)
ax1.xaxis.set_major_formatter(LOGF)
ax2.errorbar([r["adc_off"] for r in sup], [r["delta"] for r in sup],
             yerr=[max(r["sd_delta"], 0.02) for r in sup], label="superpozice, LED na 10 cm",
             **pl.body(0))
ax2.set(xlabel=r"$N_\mathrm{pozadí}$ / kód", ylabel=r"$\Delta N$ / kód")
pl.legenda_nad_grafem(ax2)
pl.carky_na_osach(ax2)
save(fig, "graf_linearita.pdf")
for r in sup:
    out(f"  {r['tag']:12s} pozadí {r['adc_off']:6.1f}  amplituda {r['delta']:.3f} ± {r['sd_delta']:.3f}")

# ============================================================ 5. averaging sigma(N)
out("\n## 5. Průměrování σ(N) (graf_sigma_N.pdf)")
LAB = {"3v3": "pin 3V3", "fd_tma": "fotodioda, tma", "fd_okoli": "fotodioda, zářivka",
       "fd_LED_trvale": "fotodioda, LED", "fr_tma": "fotorezistor, tma",
       "fr_LED_trvale": "fotorezistor, LED"}
prum = [r for r in csv.DictReader(open(P("ukol6a_prumerovani", "prumerovani.csv")))
        if r["tag"] in LAB]
fig, ax = plt.subplots(figsize=(W1, H1 * 1.1))
ax.set(xscale="log", yscale="log")
for s, t in enumerate(LAB):
    rr_ = [r for r in prum if r["tag"] == t]
    N = np.array([int(r["N"]) for r in rr_])
    sg = np.array([float(r["sigma"]) for r in rr_])
    nb = np.array([int(r["blocks"]) for r in rr_])
    ax.errorbar(N, sg, yerr=sg / np.sqrt(2 * (nb - 1)), label=LAB[t], **pl.body(s))
    out(f"  {t:14s} průměr {float(rr_[0]['mean']):7.2f}: " + ", ".join(
        f"σ({n_})={v:.3f}" for n_, v in zip(N, sg)))
nn = np.array([1, 1000])
ax.plot(nn, 0.389 / np.sqrt(nn), label=r"$\sigma_1/\sqrt{N}$ (pin 3V3)", **pl.cara(0))
ax.set(xlabel="$N$", ylabel=r"$\sigma_N$ / kód")
ax.xaxis.set_major_formatter(LOGF)
ax.yaxis.set_major_formatter(LOGF)
pl.legenda_nad_grafem(ax, ncol=2)
save(fig, "graf_sigma_N.pdf")

# ============================================================ 6. Bode plot
out("\n## 6. Frekvenční odezva (graf_bode.pdf)")


def load_f(tag):
    r = list(csv.DictReader(open(P("ukol6_lockin", f"frekvence_{tag}_vysledky.csv"))))[:-1]
    return [np.array([float(x[k_]) for x in r]) for k_ in ("f_true", "R", "sd_R", "phase_deg")]


def mean_of(tag):
    rr_ = [r for r in csv.DictReader(open(P("ukol6a_prumerovani", "prumerovani.csv")))
           if r["tag"] == tag and r["N"] == "1"]
    return float(rr_[-1]["mean"])


def f_cross(f, y, level=1 / np.sqrt(2)):
    i = np.flatnonzero((y[:-1] >= level) & (y[1:] < level))[0]
    t = (y[i] - level) / (y[i] - y[i + 1])
    return np.exp(np.log(f[i]) + t * (np.log(f[i + 1]) - np.log(f[i])))


fd, Rd, sd, pd = load_f("fotodioda")
fr, Rr, sr, pr = load_f("fotorezistor")
H = mean_of("fr_LED_trvale") - mean_of("fr_tma")
Rdc = 2 / np.pi * H
fc_dc = f_cross(fr, Rr / Rdc)
kph = np.polyfit(fd, pd, 1)[0]
out(f"  fotorezistor: H = {H:.1f} kódu, R(0,5 Hz)/R_DC = {Rr[0] / Rdc:.3f}, 0,707 vůči DC při "
    f"{fc_dc:.2f} Hz, vůči 0,5 Hz při {f_cross(fr, Rr / Rr[0]):.2f} Hz; 50 Hz: {Rr[-1] / Rdc:.3f},"
    f" {pr[-1]:.1f}°; fáze 45° při {f_cross(fr, 1 - pr / 45, 0):.1f} Hz")
out(f"  fotodioda: R(50)/R(0,5) = {Rd[-1] / Rd[0]:.4f}, fáze {kph:.3f} °/Hz -> "
    f"zpoždění {kph / 360 * 1e3:.3f} ms")
out(f"  nejistota fáze σ_A/A: fotodioda {np.degrees(sd / Rd).min():.3f}–{np.degrees(sd / Rd).max():.3f}°,"
    f" fotorezistor {np.degrees(sr / Rr).min():.3f}–{np.degrees(sr / Rr).max():.3f}°;"
    f" σ_A/A_0 max {max((sd / Rd[0]).max(), (sr / Rdc).max()):.4f}")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W2, H2))
ax1.set(xscale="log", yscale="log")
ax1.errorbar(fd, Rd / Rd[0], yerr=sd / Rd[0], label="fotodioda", **pl.body(0))
ax1.errorbar(fr, Rr / Rdc, yerr=sr / Rdc, label="fotorezistor", **pl.body(1))
ff = np.logspace(np.log10(0.4), np.log10(60), 200)
ax1.plot(ff, 1 / np.hypot(1, ff / fc_dc), label="model 1. řádu", **pl.cara(1))
ax1.axhline(1 / np.sqrt(2), color="0.35", lw=0.7, ls="--")
ax1.set(xlabel="$f$ / Hz", ylabel=r"$A/A_0$")
ax1.xaxis.set_major_formatter(LOGF)
ax1.yaxis.set_major_formatter(LOGF)
ax2.set(xscale="log")
ax2.errorbar(fd, pd, yerr=np.degrees(sd / Rd), label="fotodioda", **pl.body(0))
ax2.errorbar(fr, pr, yerr=np.degrees(sr / Rr), label="fotorezistor", **pl.body(1))
ax2.plot(ff, np.degrees(np.arctan(ff / fc_dc)), label="model 1. řádu", **pl.cara(1))
ax2.set(xlabel="$f$ / Hz", ylabel=r"$\varphi$ / °")
ax2.xaxis.set_major_formatter(LOGF)
fig.legend(*ax2.get_legend_handles_labels(), ncol=3, loc="upper center", frameon=False)
save(fig, "graf_bode.pdf", rect=(0, 0, 1, 0.9))

# ============================================================ 7. weak signal
out("\n## 7. Signál pod šumem (graf_slaby_signal.pdf)")
FS = 1000.0


def demod(x, P_, k0=0):
    k_ = np.arange(len(x)) + k0
    w_ = 2 * np.pi / P_
    z = 2 * np.mean((x - x.mean()) * np.exp(1j * w_ * k_))
    r_ = ((k_ % P_) < P_ // 2).astype(float)
    zr = 2 * np.mean((r_ - r_.mean()) * np.exp(1j * w_ * k_))
    return z * np.conj(zr) / abs(zr)


def load_stream(fn):
    d = np.array([[int(v) for v in ln.split(",")] for ln in open(fn) if ln[0].isdigit()])
    return d[:, 0], d[:, 1].astype(float)


js = json.load(open(P("ukol6_lockin", "slaby_hlavni.json")))
ref, x = load_stream(P("ukol6_lockin", "slaby_hlavni.csv"))
Pp = 2 * int(round(FS / (2 * 7.0)))
e = np.flatnonzero(np.diff(ref) == 1)
xs = x[e[0] + 1:]
nper = len(xs) // Pp
xs = xs[:nper * Pp]
sT = np.array(js["sigmaT"])
sl, icpt = np.polyfit(np.log(sT[:, 0]), np.log(sT[:, 1]), 1)
cum = []
for npp in np.unique(np.geomspace(4, nper, 30).astype(int)):
    z = demod(xs[:npp * Pp], Pp)
    # sigma(T): measured slope of sigma_I(T), anchored to the final uncertainty err_IQ
    cum.append((npp * Pp / FS, z.real, z.imag, js["err_IQ"] * (npp / nper) ** sl))
cum = np.array(cum)
zf = demod(xs, Pp)
wave = xs.reshape(nper, Pp).mean(axis=0)
ph = np.arange(Pp) / Pp
on = ph < 0.5
skok = wave[on][2:].mean() - wave[~on][2:].mean()
out(f"  I = {zf.real:.4f}, Q = {zf.imag:.4f} (json {js['I']}, {js['Q']} ± {js['err_IQ']}), "
    f"pozadí {xs.mean():.2f}, σ surových dat {js['raw_sigma']}, sklon σ(T) {sl:.3f}")
out(f"  průměrný průběh přes {nper} period: skok zap−vyp (bez 2 vzorků po hraně) {skok:.3f} kódu, "
    f"nejistota bodu {js['raw_sigma'] / np.sqrt(nper):.3f}; první vzorek po rozsvícení "
    f"{wave[0] - wave[on][2:].mean():+.2f}, po zhasnutí {wave[Pp // 2] - wave[~on][2:].mean():+.2f}")
snr = np.abs(cum[:, 1]) / cum[:, 3]
out("  SNR(T): " + ", ".join(f"{t:.1f} s: {v:.1f}" for t, v in zip(cum[::5, 0], snr[::5]))
    + f", konec {cum[-1, 0]:.0f} s: {snr[-1]:.1f}")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W2, H2))
ax1.axvspan(0, 0.5, color=pl.barva(1), alpha=0.12, lw=0, label="LED svítí")
ax1.errorbar(ph, wave, yerr=js["raw_sigma"] / np.sqrt(nper), label=f"průměr {nper} period",
             **{**pl.body(0), "ms": 2.5, "capsize": 0})
ax1.set(xlim=(0, 1), xlabel="fáze periody", ylabel="$N$ / kód")
ax1.xaxis.set_major_formatter(pl.Carka())
ax1.yaxis.set_major_formatter(pl.Carka())
lo, hi = wave.min(), wave.max()
ax1.set_ylim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))
pl.legenda_nad_grafem(ax1, ncol=1)
ax2.set(xscale="log")
for k_, lab, i_ in ((1, "$I(T)$", 0), (2, "$Q(T)$", 1)):
    ax2.fill_between(cum[:, 0], cum[:, k_] - cum[:, 3], cum[:, k_] + cum[:, 3],
                     color=pl.barva(i_), alpha=0.2, lw=0)
    ax2.plot(cum[:, 0], cum[:, k_], label=lab, **bp(i_))
ax2.axhline(0, color="0.35", lw=0.8)
ax2.set(xlabel="$T$ / s", ylabel="$I$, $Q$ / kód")
ax2.xaxis.set_major_formatter(LOGF)
ax2.yaxis.set_major_formatter(pl.Carka())
ax3 = ax2.twinx()
ax3.plot(cum[:, 0], snr, label=r"SNR $=|I|/\sigma_I$", **{**bp(2), "linestyle": "-", "linewidth": 0.8})
ax3.set_ylabel(r"SNR", color=pl.barva(2))
ax3.tick_params(axis="y", colors=pl.barva(2))
h1, l1 = ax2.get_legend_handles_labels()
h3, l3 = ax3.get_legend_handles_labels()
ax2.legend(h1 + h3, l1 + l3, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01),
           frameon=False, columnspacing=1.2, borderaxespad=0.0)
save(fig, "graf_slaby_signal.pdf")

# ============================================================ 8. noise spectra
out("\n## 8. Spektrum šumu (graf_spektrum.pdf)")
fig, ax = plt.subplots(figsize=(W1, H1))
ax.set(yscale="log")
for s, (fn, lab) in enumerate((("3v3.csv", "pin 3V3"), ("fd_tma.csv", "fotodioda, tma"))):
    _, xx = load_stream(P("ukol6a_prumerovani", fn))
    win = np.hanning(len(xx))
    A = 2 * np.abs(np.fft.rfft((xx - xx.mean()) * win)) / win.sum()
    f_ = np.fft.rfftfreq(len(xx), 1 / FS)
    # max in 0.5 Hz bins keeps the narrow lines visible in a small vector file
    edges_f = np.arange(0, 500.5, 0.5)
    idx = np.digitize(f_, edges_f)
    Am = np.array([A[idx == j].max() for j in range(1, len(edges_f))])
    ax.plot(edges_f[:-1] + 0.25, Am, color=pl.barva(s), lw=0.6, label=lab)
    if fn == "fd_tma.csv":
        for f0 in (50, 159, 428):
            j = np.argmax(np.where(np.abs(edges_f[:-1] + 0.25 - f0) < 2, Am, 0))
            fpk, apk = edges_f[j] + 0.25, Am[j]
            ax.annotate(f"{f0} Hz", xy=(fpk, apk * 1.15), xytext=(fpk, apk * 4.5), ha="center",
                        fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.8, color="0.15"))
            out(f"  čára {f0} Hz: vrchol {fpk:.2f} Hz, {apk:.3f} kódu")
    out(f"  {lab}: medián amplitudy 2–490 Hz {np.median(A[(f_ > 2) & (f_ < 490)]):.5f} kódu, "
        f"50 Hz {A[np.abs(f_ - 50.07) < 0.3].max():.4f}")
ax.set(xlim=(0, 500), ylim=(None, 6), xlabel="$f$ / Hz", ylabel="$A$ / kód")
pl.legenda_nad_grafem(ax, ncol=2)
ax.xaxis.set_major_formatter(pl.Carka())
save(fig, "graf_spektrum.pdf")

with open(os.path.join(OUT, "protokol_grafy_vystup.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(LOG) + "\n")
