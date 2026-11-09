#!/usr/bin/env python3
"""Task 7.1 (protocol): combined Bode plot of photodiode and photoresistor + -3 dB frequency.

Normalisation: (a) to R(0.5 Hz) as in the plan, (b) for the photoresistor also to
the true DC step from task 6.0 runs (fr_LED_trvale - fr_tma): an ideal square wave
of height H has a fundamental of amplitude (2/pi) H.
Compares with a single-pole low-pass 1/sqrt(1+(f/fc)^2), phase atan(f/fc).
"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
O = os.path.join(HERE, "..", "data")
L = os.path.join(O, "ukol6_lockin")

def load(tag):
    r = list(csv.DictReader(open(os.path.join(L, f"frekvence_{tag}_vysledky.csv"))))[:-1]
    return (np.array([float(x["f_true"]) for x in r]), np.array([float(x["R"]) for x in r]),
            np.array([float(x["sd_R"]) for x in r]), np.array([float(x["phase_deg"]) for x in r]))

def mean_of(tag):
    rows = [r for r in csv.DictReader(open(os.path.join(O, "ukol6a_prumerovani", "prumerovani.csv")))
            if r["tag"] == tag and r["N"] == "1"]
    return float(rows[-1]["mean"])

def f_cross(f, y, level=1 / np.sqrt(2)):
    i = np.flatnonzero((y[:-1] >= level) & (y[1:] < level))
    if not len(i): return np.nan
    i = i[0]                                   # interpolate in log f
    t = (y[i] - level) / (y[i] - y[i + 1])
    return np.exp(np.log(f[i]) + t * (np.log(f[i + 1]) - np.log(f[i])))

fd, Rd, sd, pd = load("fotodioda")
fr, Rr, sr, pr = load("fotorezistor")
H = mean_of("fr_LED_trvale") - mean_of("fr_tma")
R_dc = 2 / np.pi * H
print(f"photoresistor DC step H = {H:.1f} codes -> ideal fundamental (2/pi)H = {R_dc:.1f}; "
      f"R(0.5 Hz) = {Rr[0]:.1f} = {Rr[0]/R_dc:.3f} of DC")
for name, f, y in [("fotodioda / R(0.5)", fd, Rd / Rd[0]), ("fotorezistor / R(0.5)", fr, Rr / Rr[0]),
                   ("fotorezistor / DC", fr, Rr / R_dc)]:
    fc = f_cross(f, y)
    print(f"  {name:24s}: 0.707 at f = {fc:.2f} Hz" if np.isfinite(fc) else f"  {name}: no crossing")
    if name.endswith("DC"):
        fc_dc = fc
fc_ph = f_cross(fr, 1 - pr / 45, 0)          # phase = 45 deg
print(f"  photoresistor phase = 45 deg at f = {fc_ph:.1f} Hz (single pole: same as 0.707 point)")
print(f"  single pole with fc = {fc_dc:.2f} Hz at 50 Hz: {1/np.hypot(1, 50/fc_dc):.3f}, "
      f"{np.degrees(np.arctan(50/fc_dc)):.0f} deg; measured {Rr[-1]/R_dc:.3f}, {pr[-1]:.0f} deg")
# photodiode phase -> constant delay
k = np.polyfit(fd, pd, 1)[0]
print(f"  photodiode phase slope {k:.3f} deg/Hz -> delay {k/360*1e3:.3f} ms")

ff = np.logspace(np.log10(0.4), np.log10(60), 200)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
ax[0].errorbar(fd, Rd / Rd[0], sd / Rd[0], fmt="o-", label="fotodioda (1 MΩ)")
ax[0].errorbar(fr, Rr / Rr[0], sr / Rr[0], fmt="s-", label="fotorezistor (10k), norm. na 0,5 Hz")
ax[0].plot(fr, Rr / R_dc, "^--", color="C1", alpha=0.6, label="fotorezistor, norm. na DC skok")
ax[0].plot(ff, 1 / np.hypot(1, ff / fc_dc), ":", color="gray",
           label=f"1. řád (RC), f_c = {fc_dc:.1f} Hz")
ax[0].axhline(1 / np.sqrt(2), color="k", lw=0.6, ls="--")
ax[0].set(xscale="log", yscale="log", xlabel="f [Hz]", ylabel="relativní amplituda",
          title="Bodeho graf: amplituda")
ax[0].grid(True, which="both", alpha=0.3); ax[0].legend(fontsize=8)
ax[1].plot(fd, pd, "o-", label=f"fotodioda (zpoždění {k/360*1e3:.2f} ms)")
ax[1].plot(fr, pr, "s-", label="fotorezistor")
ax[1].plot(ff, np.degrees(np.arctan(ff / fc_dc)), ":", color="gray", label=f"1. řád, f_c = {fc_dc:.1f} Hz")
ax[1].axhline(45, color="k", lw=0.6, ls="--")
ax[1].set(xscale="log", xlabel="f [Hz]", ylabel="fázové zpoždění [°]", title="Bodeho graf: fáze")
ax[1].grid(True, which="both", alpha=0.3); ax[1].legend(fontsize=8)
fig.tight_layout()
out = os.path.join(L, "bode_srovnani.png"); fig.savefig(out, dpi=130); print(f"-> {out}")
