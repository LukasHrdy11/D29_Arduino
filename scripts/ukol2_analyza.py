#!/usr/bin/env python3
"""Task 2: LED I-V curves (linear + semilog), threshold voltages with uncertainties.

Usage:
  python3 ukol2_analyza.py file1.csv [file2.csv ...] [-o out.png]
                           [--vcc 5.03] [--vcc-tol 0.10] [--R-tol 0.01] [--adc-lsb 2]

Quantities (forward sweeps):
  U_lin   intercept with I = 0 of a line fitted to I > f*I_max (f = 0.5; tangent method)
  U_0.1   U where I = 0.1 mA, U_1  U where I = 1 mA
          (local linear fit of U vs log10 I over points within a factor 1.5 of the level)
  R_dyn   1 / slope of the line fit (series resistance at high current)

Uncertainties (standard, 1 sigma):
  stat    from the fit covariance (data noise)
  method  U_lin only: half range of U_lin over f = 0.4, 0.5, 0.6
  Vcc     Monte Carlo: Vcc * (1 + uniform(-tol, tol)); tol = 10 % bandgap tolerance
          unless --vcc from a multimeter is given with a smaller --vcc-tol
  ADC     Monte Carlo: independent offset uniform(-adc_lsb, adc_lsb) LSB on A0 and A1
          (ATmega328P absolute accuracy +-2 LSB)
  R       Monte Carlo: R * (1 + uniform(-R_tol, R_tol)), R_tol = 1 % (brown band)
  total   quadrature sum of all components
Reversed-LED files (name contains 'zaverny') are only plotted (3rd quadrant) and an
upper limit of the reverse current is printed.
"""
import argparse, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"cervena": "#c62828", "zluta": "#c9a000", "modra": "#1565c0"}
LABELS = {"cervena": "červená", "zluta": "žlutá", "modra": "modrá"}
QTY = ["U_lin", "U_0.1", "U_1"]
rng = np.random.default_rng(1)


def load(path):
    meta, rows = {}, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            m = re.match(r"#\s*(\w+)=(.*)", line)
            if m:
                meta[m.group(1)] = m.group(2)
            elif line and line[0].isdigit():
                rows.append([float(x) for x in line.split(",")])
    d = np.array(rows)
    return d[:, 1], d[:, 2], float(meta["vcc_mV"]) / 1000, float(meta.get("R_ohm", 220))


def to_ui(a0, a1, vcc, R):
    u1 = a1 * vcc / 1024
    return (a0 - a1) * vcc / 1024, u1 / R * 1e3


def fit_lin(u, i, f):
    top = i > f * i.max()
    (k, q), cov = np.polyfit(u[top], i[top], 1, cov=True)
    J = np.array([q / k**2, -1 / k])          # d(-q/k)/d(k, q)
    return -q / k, np.sqrt(J @ cov @ J), k


def at_level(u, i, level):
    sel = (i > level / 1.5) & (i < level * 1.5)
    x = np.log10(i[sel]) - np.log10(level)
    (b, c), cov = np.polyfit(x, u[sel], 1, cov=True)
    return c, np.sqrt(cov[1, 1])               # value and stat error at x = 0


def quantities(u, i):
    U_lin, s_lin, k = fit_lin(u, i, 0.5)
    U01, s01 = at_level(u, i, 0.1)
    U1, s1 = at_level(u, i, 1.0)
    return dict(zip(QTY, [U_lin, U01, U1])), dict(zip(QTY, [s_lin, s01, s1])), k


def mc(a0, a1, vcc, R, n, vcc_tol=0.0, R_tol=0.0, lsb=0.0):
    out = {q: [] for q in QTY}
    for _ in range(n):
        v = vcc * (1 + rng.uniform(-vcc_tol, vcc_tol))
        r = R * (1 + rng.uniform(-R_tol, R_tol))
        b0 = a0 + rng.uniform(-lsb, lsb)
        b1 = a1 + rng.uniform(-lsb, lsb)
        val, _, _ = quantities(*to_ui(b0, b1, v, r))
        for q in QTY:
            out[q].append(val[q])
    return {q: np.std(out[q], ddof=1) for q in QTY}


ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("-o", "--out")
ap.add_argument("--vcc", type=float, help="override Vcc [V] (e.g. multimeter reading)")
ap.add_argument("--vcc-tol", type=float, default=0.10, help="relative half-width")
ap.add_argument("--R-tol", type=float, default=0.01)
ap.add_argument("--adc-lsb", type=float, default=2.0)
ap.add_argument("--mc", type=int, default=400, help="Monte Carlo draws per source")
a = ap.parse_args()
out = a.out or (os.path.splitext(a.files[0])[0] + ".png" if len(a.files) == 1 else
                os.path.join(os.path.dirname(a.files[0]), "va_led_srovnani.png"))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
groups = {}
for path in a.files:
    a0, a1, vcc, R = load(path)
    if a.vcc:
        vcc = a.vcc
    u, i = to_ui(a0, a1, vcc, R)
    name = os.path.splitext(os.path.basename(path))[0]
    key = next((k for k in COLORS if k in name), None)
    col, lab = COLORS.get(key, "#555555"), LABELS.get(key, name)

    if "zaverny" in name:                  # LED reversed: plot in the 3rd quadrant
        lsb_mA = vcc / 1024 / R * 1e3
        print(f"{name}: N={len(u)}, U down to {-u.max():.3f} V, max A1 code {a1.max():.2f}"
              f" -> |I| < 0.5 LSB / R = {0.5 * lsb_mA * 1e3:.0f} µA")
        ax1.plot(-u, -i, ".", ms=2, color=col, alpha=0.5)
        continue

    val, stat, k = quantities(u, i)
    meth = {q: 0.0 for q in QTY}
    lins = [fit_lin(u, i, f)[0] for f in (0.4, 0.5, 0.6)]
    meth["U_lin"] = (max(lins) - min(lins)) / 2
    s_vcc = mc(a0, a1, vcc, R, a.mc, vcc_tol=a.vcc_tol)
    s_adc = mc(a0, a1, vcc, R, a.mc, lsb=a.adc_lsb)
    s_R = mc(a0, a1, vcc, R, a.mc, R_tol=a.R_tol)
    print(f"{name}: N={len(u)}, Vcc={vcc:.3f} V (tol ±{a.vcc_tol:.1%}), "
          f"I_max={i.max():.2f} mA, R_dyn={1e3 / k:.0f} ohm, "
          + ", ".join(f"{q}={val[q]:.4f}" for q in QTY))
    base = re.sub(r"_\d+$", "", name)
    first = base not in groups
    groups.setdefault(base, dict(lab=lab, rows=[], rdyn=[]))
    groups[base]["rows"].append((val, stat, meth, s_vcc, s_adc, s_R))
    groups[base]["rdyn"].append(1e3 / k)

    ax1.plot(u, i, ".", ms=2, color=col, alpha=0.3)
    if first:
        uu = np.linspace(val["U_lin"], u.max(), 10)
        ax1.plot(uu, k * (uu - val["U_lin"]), "--", lw=1.2, color=col)
        groups[base]["legend"] = (col, lab)
    pos = i > 0.01
    ax2.semilogy(u[pos], i[pos], ".", ms=2, color=col, alpha=0.3)

# Combine repeats of the same LED: value = mean; components:
#   rep  = sd / sqrt(n) of the repeats (standard error of the mean, n >= 2)
#   stat = rms(stat) / sqrt(n); method, Vcc, ADC, R are common to all repeats
#          (correlated), so their mean is taken, not reduced by sqrt(n).
print()
for base, g in groups.items():
    n = len(g["rows"])
    print(f"### {g['lab']} ({base}, n = {n} sweeps, R_dyn = {np.mean(g['rdyn']):.0f} ohm)")
    print("| veličina | hodnota [V] | opakování | stat | metoda | Vcc | ADC | R | **celkem** |")
    print("|---|---|---|---|---|---|---|---|---|")
    for q in QTY:
        vals = np.array([r[0][q] for r in g["rows"]])
        rep = vals.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        st = np.sqrt(np.mean([r[1][q] ** 2 for r in g["rows"]]) / n)
        comp = [rep, st] + [np.mean([r[j][q] for r in g["rows"]]) for j in (2, 3, 4, 5)]
        tot = np.sqrt(np.sum(np.square(comp)))
        print(f"| {q} | {vals.mean():.4f} | " + " | ".join(f"{c:.4f}" for c in comp)
              + f" | **{tot:.3f}** |")
        if q == "U_lin":
            g["U_lin"] = vals.mean()
    col, lab = g["legend"]
    ax1.plot([], [], ".", ms=8, color=col,
             label=f"{lab}: U_lin = {g['U_lin']:.2f} V ({n}×)")
    print()

for ax in (ax1, ax2):
    ax.set_xlabel("U_LED [V]")
    ax.set_ylabel("I [mA]")
    ax.grid(True, color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)
ax1.set_ylim(bottom=-0.3)
ax1.set_title("VA charakteristika (lineárně)")
ax2.set_title("VA charakteristika (semilog, I > 10 µA)")
ax1.legend(loc="upper left", frameon=False)
fig.tight_layout()
fig.savefig(out, dpi=150)
print("\ngraph ->", os.path.abspath(out))
