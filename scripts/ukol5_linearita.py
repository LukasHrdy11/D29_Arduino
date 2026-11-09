#!/usr/bin/env python3
"""Task 5: photodiode linearity, one point per run (sketch sketches/stream).

Usage:
  python3 ukol5_linearita.py <r_cm> [--tag dark] [-f 7] [-t 20]
Opens the port (board resets), waits for the sketch, modulates the LED on D9
with a square wave at f, records `ref,adc` lines for T seconds, saves the raw
CSV and appends one row to data/ukol5_fotodioda/linearita.csv.

Two estimates of the optical signal (both in ADC codes):
  delta = mean(ADC | LED on) - mean(ADC | LED off), per reference period,
          uncertainty = spread of per-period values / sqrt(#periods)
  R     = sine lock-in amplitude of the fundamental; for an ideal square wave
          R = (2/pi) * delta, so delta_li = (pi/2) * R is comparable to delta.
Ambient light is the same in both halves of a period, so it cancels.
"""
import argparse, csv, os, time
from datetime import datetime
import numpy as np
import serial

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "ukol5_fotodioda")
SKIP = 2          # samples dropped after each LED edge (settling)

ap = argparse.ArgumentParser()
ap.add_argument("r_cm", type=float, help="distance LED tip -> photodiode window [cm]")
ap.add_argument("--tag", default="", help="free label, e.g. dark, rep2")
ap.add_argument("-f", "--freq", type=float, default=7.0)
ap.add_argument("-t", "--time", type=float, default=20.0)
ap.add_argument("-p", "--port", default="/dev/ttyACM0")
ap.add_argument("--RL", type=float, default=1e6, help="load resistor [ohm]")
a = ap.parse_args()

os.makedirs(OUT, exist_ok=True)
name = f"lin_r{a.r_cm:g}cm" + (f"_{a.tag}" if a.tag else "")
raw = os.path.join(OUT, name + ".csv")

# ------------------------------------------------------------------ capture
with serial.Serial(a.port, 115200, timeout=1) as s:
    t0 = time.time()
    while time.time() - t0 < 8:                      # wait for setup() status
        if s.readline().decode(errors="ignore").startswith("# fs="):
            break
    else:
        raise SystemExit("sketch did not answer (is sketches/stream uploaded?)")
    s.write(f"f{a.freq}\n".encode()); s.flush()
    meta, rows = [], []
    t_start = time.time() + 0.5                      # skip the first 0.5 s
    while time.time() < t_start + a.time:
        line = s.readline().decode(errors="ignore").strip()
        if line.startswith("#"):
            meta.append(line); continue
        p = line.split(",")
        if len(p) == 2 and p[0].isdigit() and p[1].isdigit() and time.time() > t_start:
            rows.append((int(p[0]), int(p[1])))
    s.write(b"p\n"); s.flush(); time.sleep(0.2)
    while s.in_waiting:
        line = s.readline().decode(errors="ignore").strip()
        if line.startswith("#"):
            meta.append(line)
    s.write(b"f0\n"); s.flush()                      # LED off afterwards

with open(raw, "w") as fh:
    fh.write(f"# r_cm={a.r_cm} RL={a.RL:g} time={datetime.now().isoformat(timespec='seconds')}\n")
    for m in meta:
        fh.write(m + "\n")
    fh.write("ref,adc\n")
    for r, x in rows:
        fh.write(f"{r},{x}\n")

d = np.array(rows)
ref, x = d[:, 0], d[:, 1].astype(float)
fs = 1000.0
f_true = a.freq
overruns = None
for m in meta:
    for kv in m.lstrip("# ").split():
        k, _, v = kv.partition("=")
        if k == "f_true": f_true = float(v)
        if k == "overruns": overruns = int(v)
rate = len(x) / a.time

# ------------------------------------------------- per-period on/off difference
# cut the record into whole reference periods starting at a rising edge
edges = np.flatnonzero(np.diff(ref) == 1) + 1
deltas = []
for i0, i1 in zip(edges[:-1], edges[1:]):
    seg_r, seg_x = ref[i0:i1], x[i0:i1]
    fall = np.flatnonzero(seg_r == 0)
    if len(fall) == 0:
        continue
    on = seg_x[SKIP:fall[0]]
    off = seg_x[fall[0] + SKIP:]
    if len(on) and len(off):
        deltas.append(on.mean() - off.mean())
deltas = np.array(deltas)
delta, sd_delta = deltas.mean(), deltas.std(ddof=1) / np.sqrt(len(deltas))

# ------------------------------------------------ sine lock-in, whole periods
xs, rs = x[edges[0]:edges[-1]], ref[edges[0]:edges[-1]]
P = int(round(fs / f_true))
n = len(xs)
w = 2 * np.pi / P                                    # f_true = fs / P exactly
c, sn = np.cos(w * np.arange(n)), np.sin(w * np.arange(n))
xc = xs - xs.mean()
X, Y = 2 * np.mean(xc * c), 2 * np.mean(xc * sn)
R = np.hypot(X, Y)
rc, rsn = 2 * np.mean((rs - rs.mean()) * c), 2 * np.mean((rs - rs.mean()) * sn)
phase = (np.degrees(np.arctan2(Y, X) - np.arctan2(rsn, rc)) + 180) % 360 - 180
# uncertainty of R from 1 s blocks of whole periods
nb = max(1, round(fs / P)) * P
k = n // nb
Rb = [np.hypot(2 * np.mean(xc[i*nb:(i+1)*nb] * c[i*nb:(i+1)*nb]),
               2 * np.mean(xc[i*nb:(i+1)*nb] * sn[i*nb:(i+1)*nb])) for i in range(k)]
sd_R = np.std(Rb, ddof=1) / np.sqrt(k)
delta_li = np.pi / 2 * R

# levels and raw noise without the SKIP samples right after each edge
settled = np.ones(len(x), bool)
for e in np.flatnonzero(np.diff(ref) != 0) + 1:
    settled[e:e + SKIP] = False
on_lvl = x[(ref == 1) & settled].mean()
off_lvl = x[(ref == 0) & settled].mean()
sd_off = x[(ref == 0) & settled].std()
lsb_V = 5.0 / 1024
row = dict(
    r_cm=a.r_cm, tag=a.tag, f_true=f"{f_true:.4f}", T_s=a.time, n=len(x),
    rate=f"{rate:.0f}", overruns=overruns, periods=len(deltas),
    adc_off=f"{off_lvl:.2f}", adc_on=f"{on_lvl:.2f}", adc_max=int(x.max()),
    raw_sd_off=f"{sd_off:.2f}",
    delta=f"{delta:.4f}", sd_delta=f"{sd_delta:.4f}",
    R=f"{R:.4f}", sd_R=f"{sd_R:.4f}", delta_li=f"{delta_li:.4f}",
    phase_deg=f"{phase:.1f}",
    U_mV=f"{delta * lsb_V * 1e3:.3f}", I_nA=f"{delta * lsb_V / a.RL * 1e9:.3f}",
    file=os.path.basename(raw),
)
table = os.path.join(OUT, "linearita.csv")
new = not os.path.exists(table)
with open(table, "a", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=row.keys())
    if new:
        wr.writeheader()
    wr.writerow(row)

print(f"r = {a.r_cm:g} cm {a.tag}: {len(x)} samples ({rate:.0f}/s), overruns = {overruns}, "
      f"f = {f_true:.4f} Hz, {len(deltas)} periods")
print(f"  ADC off = {off_lvl:.1f}, on = {on_lvl:.1f}, max = {x.max():.0f}, raw sd(off) = {sd_off:.2f}")
print(f"  delta (on-off)   = {delta:.3f} ± {sd_delta:.3f} codes  = {delta*lsb_V*1e3:.2f} mV"
      f" = {delta*lsb_V/a.RL*1e9:.2f} nA")
print(f"  lock-in (pi/2)R  = {delta_li:.3f} ± {np.pi/2*sd_R:.3f} codes, phase = {phase:.1f} deg,"
      f" ratio = {delta_li/delta if delta else float('nan'):.3f}")
warn = []
if rate < 950: warn.append("samples lost (rate < 950/s)")
if overruns: warn.append(f"{overruns} overruns")
if x.max() >= 1000: warn.append("near ADC full scale -> saturation?")
if on_lvl * lsb_V > 4.0: warn.append("U(on) > 4 V -> photodiode may leave linear range")
print("  CHECK: " + ("OK" if not warn else "; ".join(warn)))
print(f"  -> {raw}\n  -> {table}")
