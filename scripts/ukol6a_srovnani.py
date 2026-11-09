#!/usr/bin/env python3
"""Task 6 (protocol): compare sigma(N) of all valid runs (prumerovani.csv) in one log-log plot
and print the power share of the main interference lines for each raw record."""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "data", "ukol6a_prumerovani")
LABEL = {"3v3": "3V3 pin (ADC sám)", "fd_tma": "fotodioda, tma",
         "fd_tma_ON_zakryta": "fotodioda, tma, ON-LED zakrytá",
         "fd_okoli": "fotodioda, okolní světlo (zářivka)",
         "fd_LED_trvale": "fotodioda, LED trvale (~350 kódů)",
         "fr_tma": "fotorezistor, tma", "fr_LED_trvale": "fotorezistor, LED trvale (~380 kódů)"}
LINES = [50.07, 100.14, 159.2, 427.6]

rows = [r for r in csv.DictReader(open(os.path.join(D, "prumerovani.csv")))
        if "NEPLATNE" not in r["tag"]]
tags = list(dict.fromkeys(r["tag"] for r in rows))
fig, ax = plt.subplots(figsize=(7.5, 5))
print(f"{'tag':20s} {'mean':>8} {'sigma1':>7} {'s100':>7} {'s1000':>7} "
      + " ".join(f"{f:>7.0f}Hz" for f in LINES) + "  <1Hz")
for t in tags:
    rr = [r for r in rows if r["tag"] == t]
    N = np.array([int(r["N"]) for r in rr]); s = np.array([float(r["sigma"]) for r in rr])
    ax.loglog(N, s, "o-", label=f"{LABEL.get(t, t)}: {float(rr[0]['mean']):.0f} kódů")
    x = np.array([int(l.split(",")[1]) for l in open(os.path.join(D, rr[0]["file"]))
                  if l[0].isdigit()], float)
    xc = x - x.mean(); P = np.abs(np.fft.rfft(xc)) ** 2; f = np.fft.rfftfreq(len(x), 1e-3)
    tot = P[1:].sum()
    sh = [100 * P[np.abs(f - c) < 0.3].sum() / tot for c in LINES]
    low = 100 * P[(f > 0) & (f < 1)].sum() / tot
    sd = dict(zip(N, s))
    print(f"{t:20s} {x.mean():8.2f} {s[0]:7.3f} {sd.get(100, np.nan):7.3f} {sd.get(1000, np.nan):7.3f} "
          + " ".join(f"{v:9.1f}" for v in sh) + f"  {low:5.1f}   (% of variance)")
nn = np.array([1, 1000])
ax.loglog(nn, 0.389 / np.sqrt(nn), "k:", lw=1, label="sklon −0,5 (bílý šum)")
ax.axhline(1, color="gray", lw=0.6, ls="--")
ax.text(1.1, 1.08, "1 kód ADC", fontsize=8, color="gray")
ax.set(xlabel="N (počet průměrovaných vzorků, 1 kHz)", ylabel="σ průměru [kód ADC]",
       title="Průměrování: σ(N) pro různé zdroje signálu")
ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
fig.tight_layout()
out = os.path.join(D, "srovnani_sigma_N.png"); fig.savefig(out, dpi=130)
print(f"-> {out}")
