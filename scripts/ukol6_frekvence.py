#!/usr/bin/env python3
"""Task 7.1 (protocol): frequency response of a light sensor (sketch sketches/stream, LED on D9).

Usage (from the repository root, venv active):
  python3 scripts/ukol6_frekvence.py <sensor_tag> [-t 20]
  python3 scripts/ukol6_frekvence.py <sensor_tag> --file raw.csv   # re-analyse only
Opens the port once and steps the LED square-wave frequency through FREQS
(plus 0.5 Hz again at the end as a drift check). Per frequency: drop the first
SETTLE s, record T s, then
  R, phase  = lock-in amplitude/phase of the fundamental relative to the reference
              (whole periods, w = 2 pi / P exactly; ideal square wave: (pi/2) R = on-off)
  delta     = mean(on) - mean(off), 2 samples skipped after each edge
  wave      = mean waveform over one period (for plotting the shape)
Saves raw `f_req,ref,adc` to data/ukol6_lockin/frekvence_<tag>.csv,
results to frekvence_<tag>_vysledky.csv and the Bode plot frekvence_<tag>.png.
"""
import argparse, csv, os, time
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "ukol6_lockin")
FREQS = [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 0.5]
SKIP = 2

ap = argparse.ArgumentParser()
ap.add_argument("tag", help="sensor label, e.g. fotodioda, fotorezistor")
ap.add_argument("-t", "--time", type=float, default=20.0, help="seconds per frequency")
ap.add_argument("--settle", type=float, default=1.5)
ap.add_argument("-p", "--port", default="/dev/ttyACM0")
ap.add_argument("--file")
a = ap.parse_args()
os.makedirs(OUT, exist_ok=True)
raw = a.file or os.path.join(OUT, f"frekvence_{a.tag}.csv")

if not a.file:
    import serial
    data = []
    with serial.Serial(a.port, 115200, timeout=1) as s:
        t0 = time.time()
        while time.time() - t0 < 8:
            if s.readline().decode(errors="ignore").startswith("# fs="):
                break
        else:
            raise SystemExit("sketch did not answer (is sketches/stream uploaded?)")
        for i, f in enumerate(FREQS):
            s.write(f"f{f}\n".encode()); s.flush()
            t_start = time.time() + a.settle
            n = 0
            while time.time() < t_start + a.time:
                line = s.readline().decode(errors="ignore").strip()
                p = line.split(",")
                if len(p) == 2 and p[0].isdigit() and p[1].isdigit() and time.time() > t_start:
                    data.append((i, int(p[0]), int(p[1]))); n += 1
            print(f"  {f:5g} Hz: {n} samples", flush=True)
        s.write(b"p\n"); s.flush(); time.sleep(0.2)
        status = [l for l in s.read(s.in_waiting).decode(errors="ignore").splitlines() if l.startswith("#")]
        s.write(b"f0\n"); s.flush()
    with open(raw, "w") as fh:
        fh.write(f"# tag={a.tag} T={a.time} settle={a.settle} "
                 f"time={datetime.now().isoformat(timespec='seconds')} freqs={','.join(map(str, FREQS))}\n")
        for l in status:
            fh.write(l + "\n")
        fh.write("step,ref,adc\n")
        for i, r, x in data:
            fh.write(f"{i},{r},{x}\n")
    print("  " + (status[-1] if status else "no status"))

with open(raw) as fh:
    hdr = fh.readline()
    freqs = [float(v) for v in hdr.split("freqs=")[1].split()[0].split(",")]
    d = np.array([[int(v) for v in l.split(",")] for l in fh if l[0].isdigit()])

res, waves = [], []
for i, f in enumerate(freqs):
    seg = d[d[:, 0] == i]
    ref, x = seg[:, 1], seg[:, 2].astype(float)
    half = int(round(1000 / (2 * f))); P = 2 * half; f_true = 1000 / P
    edges = np.flatnonzero(np.diff(ref) == 1) + 1
    xs, rs = x[edges[0]:edges[-1]], ref[edges[0]:edges[-1]].astype(float)
    n = len(xs); k = np.arange(n); w = 2 * np.pi / P
    c, sn = np.cos(w * k), np.sin(w * k)
    xc = xs - xs.mean()
    X, Y = 2 * np.mean(xc * c), 2 * np.mean(xc * sn)
    rc, rsn = 2 * np.mean((rs - rs.mean()) * c), 2 * np.mean((rs - rs.mean()) * sn)
    R = np.hypot(X, Y)
    phase = (np.degrees(np.arctan2(Y, X) - np.arctan2(rsn, rc)) + 180) % 360 - 180
    # uncertainty from blocks of whole periods (~2 s, at least 1 period)
    nb = max(1, int(round(2000 / P))) * P
    kb = n // nb
    Rb = [np.hypot(2 * np.mean(xc[j*nb:(j+1)*nb] * c[j*nb:(j+1)*nb]),
                   2 * np.mean(xc[j*nb:(j+1)*nb] * sn[j*nb:(j+1)*nb])) for j in range(kb)]
    sd_R = np.std(Rb, ddof=1) / np.sqrt(kb) if kb > 1 else np.nan
    # on-off per whole period
    dl = []
    for i0, i1 in zip(edges[:-1], edges[1:]):
        sr, sx = ref[i0:i1], x[i0:i1]; fa = np.flatnonzero(sr == 0)
        if len(fa) and fa[0] > SKIP and len(sx) - fa[0] > SKIP:
            dl.append(sx[SKIP:fa[0]].mean() - sx[fa[0] + SKIP:].mean())
    wave = xs[:(n // P) * P].reshape(-1, P).mean(axis=0)
    waves.append((f_true, wave))
    res.append(dict(step=i, f_req=f, f_true=round(f_true, 4), periods=n // P, mean=round(xs.mean(), 2),
                    R=round(R, 4), sd_R=round(sd_R, 4), R_pi2=round(np.pi / 2 * R, 3),
                    phase_deg=round(phase, 2), delta=round(np.mean(dl), 3),
                    wave_pp=round(wave.max() - wave.min(), 2)))

R0 = np.mean([r["R"] for r in res if r["f_req"] == freqs[0]])
print(f"\n{a.tag}: reference R(0.5 Hz) = mean of first/last = {R0:.3f}")
print(f"  {'f_true':>7} {'per':>4} {'mean':>7} {'R':>8} {'±':>6} {'R/R0':>6} {'phase':>7} {'on-off':>7} {'pi/2R':>7}")
for r in res:
    r["R_rel"] = round(r["R"] / R0, 4)
    print(f"  {r['f_true']:7.3f} {r['periods']:4d} {r['mean']:7.1f} {r['R']:8.3f} {r['sd_R']:6.3f} "
          f"{r['R_rel']:6.3f} {r['phase_deg']:7.1f} {r['delta']:7.2f} {r['R_pi2']:7.2f}")
first, last = res[0]["R"], res[-1]["R"]
print(f"  drift check 0.5 Hz start/end: {first:.3f} / {last:.3f} ({100*(last/first-1):+.2f} %)")
warn = []
if abs(last / first - 1) > 0.03: warn.append("R(0.5 Hz) changed > 3 % during the run (drift / something moved)")
if d[:, 2].max() >= 1000: warn.append("ADC near full scale")
print("  CHECK: " + ("OK" if not warn else "; ".join(warn)))

with open(os.path.join(OUT, f"frekvence_{a.tag}_vysledky.csv"), "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=res[0].keys()); wr.writeheader(); wr.writerows(res)

# ------------------------------------------------------------------ plots
main = res[:-1]
ft = np.array([r["f_true"] for r in main]); Rr = np.array([r["R_rel"] for r in main])
eR = np.array([r["sd_R"] for r in main]) / R0; ph = np.array([r["phase_deg"] for r in main])
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
ax[0].errorbar(ft, Rr, eR, fmt="o-")
ax[0].plot(res[-1]["f_true"], res[-1]["R_rel"], "s", color="C3", label="0,5 Hz znovu na konci")
ax[0].axhline(1 / np.sqrt(2), color="gray", ls="--", lw=0.8, label="0,707")
ax[0].set(xscale="log", xlabel="f [Hz]", ylabel="R / R(0,5 Hz)", title=f"{a.tag}: amplituda")
ax[0].legend(fontsize=8); ax[0].grid(True, which="both", alpha=0.3)
ax[1].plot(ft, ph, "o-")
ax[1].set(xscale="log", xlabel="f [Hz]", ylabel="fáze [°]", title="fáze vůči referenci")
ax[1].grid(True, which="both", alpha=0.3)
for f_, wv in waves[:-1]:
    if f_ in (0.5, 2.0, 7.042253521126761, 20.0, 50.0) or abs(f_ - 7.04) < 0.01:
        tt = np.arange(len(wv)) / len(wv)
        ax[2].plot(tt, (wv - wv.min()) / (wv.max() - wv.min() or 1), label=f"{f_:.3g} Hz")
ax[2].set(xlabel="fáze periody (LED svítí 0–0,5)", ylabel="normovaný průběh",
          title="průměrný tvar jedné periody")
ax[2].legend(fontsize=8)
fig.tight_layout()
png = os.path.join(OUT, f"frekvence_{a.tag}.png"); fig.savefig(png, dpi=130)
print(f"  -> {raw}\n  -> {png}")
