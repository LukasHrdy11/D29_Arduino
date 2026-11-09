#!/usr/bin/env python3
"""Task 2: quality check of LED sweeps (run after every measurement).

Usage:
  python3 ukol2_kontrola.py file1.csv [file2.csv ...]

Per file checks (OK / WARN):
  range     A0 starts near 0 (< 5 codes) and reaches the 5V stop (> 1015 codes)
  knee      points with 0.05 mA < I < 0.9 I_max: count >= 100 and largest gap in
            U_LED < 20 mV (knob not turned too fast through the knee)
  backturn  INFO only: largest drop of A0 below its running maximum (knob turned
            back). Harmless for a resistive circuit; hysteresis (LED heating)
            would show up in the outlier check.
  outliers  points whose U_LED deviates from the running median in I order
            (31 points, I > 0.05 mA) by more than 10 mV (~2 LSB). Measured
            horizontally because on the steep part a 1 mV shift changes I a lot.
  vcc       bandgap Vcc within 4.9 - 5.1 V
Then, for files of the same LED, the spread of U_lin, U_0.1, U_1 across repeats.
"""
import os, re, sys
import numpy as np
from scipy.ndimage import median_filter


def load(path):
    meta, rows = {}, []
    for line in open(path):
        line = line.strip()
        m = re.match(r"#\s*(\w+)=(.*)", line)
        if m:
            meta[m.group(1)] = m.group(2)
        elif line and line[0].isdigit():
            rows.append([float(x) for x in line.split(",")])
    d = np.array(rows)
    return d, float(meta["vcc_mV"]) / 1000, float(meta.get("R_ohm", 220))


def thresholds(u, i):
    top = i > 0.5 * i.max()
    k, q = np.polyfit(u[top], i[top], 1)
    res = [-q / k]
    for lev in (0.1, 1.0):
        s = (i > lev / 1.5) & (i < lev * 1.5)
        res.append(np.polyfit(np.log10(i[s] / lev), u[s], 1)[1])
    return res


groups = {}
for path in sys.argv[1:]:
    d, vcc, R = load(path)
    name = os.path.splitext(os.path.basename(path))[0]
    a0, a1 = d[:, 1], d[:, 2]
    u, i = (a0 - a1) * vcc / 1024, a1 * vcc / 1024 / R * 1e3
    msgs = []

    ok = a0[:20].min() < 5 and a0.max() > 1015
    msgs.append(("range", ok, f"A0 start {a0[:20].min():.1f}, max {a0.max():.1f}"))

    knee = (i > 0.05) & (i < 0.9 * i.max())
    gap = np.diff(np.sort(u[knee])).max() * 1e3 if knee.sum() > 1 else np.inf
    msgs.append(("knee", knee.sum() >= 100 and gap < 20,
                 f"{knee.sum()} points, max gap {gap:.1f} mV"))

    drop = (np.maximum.accumulate(a0) - a0).max()
    info = f"backturn max {drop:.0f} codes (info)"

    o = np.argsort(i)
    o = o[i[o] > 0.05]
    dev = np.abs(u[o] - median_filter(u[o], size=31, mode="nearest")) * 1e3
    n_out = int((dev > 10).sum())
    msgs.append(("outliers", n_out == 0,
                 f"{n_out} points > 10 mV off the curve (max {dev.max():.1f} mV)"))

    msgs.append(("vcc", 4.9 < vcc < 5.1, f"{vcc:.3f} V"))

    bad = [m for m in msgs if not m[1]]
    print(f"{name}: {'OK' if not bad else 'WARN'}")
    for tag, good, txt in msgs:
        print(f"   {'ok  ' if good else 'WARN'} {tag:9s} {txt}")
    print(f"   info {info}")
    base = re.sub(r"_\d+$", "", name)
    groups.setdefault(base, []).append(thresholds(u, i))

for base, vals in groups.items():
    if len(vals) < 2:
        continue
    v = np.array(vals)
    print(f"\n{base}: {len(v)} repeats")
    for j, q in enumerate(["U_lin", "U_0.1", "U_1"]):
        print(f"   {q:6s} " + " ".join(f"{x:.4f}" for x in v[:, j])
              + f"   mean {v[:, j].mean():.4f}  sd {v[:, j].std(ddof=1) * 1e3:.1f} mV"
              + f"  range {np.ptp(v[:, j]) * 1e3:.1f} mV")
