#!/usr/bin/env python3
"""Task 6 (protocol): averaging and noise (sketch sketches/stream, LED off unless -f/--const).

Usage (from the repository root, venv active):
  python3 scripts/ukol6a_prumerovani.py <tag> [-t 150] [-f 0] [--const]
  python3 scripts/ukol6a_prumerovani.py <tag> --file raw.csv     # re-analyse only
Records `ref,adc` at 1 kHz for T seconds, saves the raw CSV and cuts the SAME
record into consecutive blocks of N samples (N = 1, 3, 10, ..., 1000).
For every N: mean and standard deviation sigma(N) of the block means.
White noise -> sigma(N) = sigma(1)/sqrt(N) (slope -0.5 in log-log).
Writes <tag>.png and appends rows to data/ukol6a_prumerovani/prumerovani.csv.
"""
import argparse, csv, os, time
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "ukol6a_prumerovani")
NS = [1, 3, 10, 30, 100, 300, 1000]
LSB_MV = 5000 / 1024

ap = argparse.ArgumentParser()
ap.add_argument("tag", help="label, e.g. 3v3, fd_tma, fd_svetlo")
ap.add_argument("-t", "--time", type=float, default=150.0)
ap.add_argument("-f", "--freq", type=float, default=0.0)
ap.add_argument("--const", action="store_true", help="LED constantly on")
ap.add_argument("-p", "--port", default="/dev/ttyACM0")
ap.add_argument("--file", help="analyse an existing raw CSV instead of measuring")
a = ap.parse_args()
os.makedirs(OUT, exist_ok=True)
raw = a.file or os.path.join(OUT, f"{a.tag}.csv")

if not a.file:
    import serial
    meta, rows = [], []
    with serial.Serial(a.port, 115200, timeout=1) as s:
        t0 = time.time()
        while time.time() - t0 < 8:
            if s.readline().decode(errors="ignore").startswith("# fs="):
                break
        else:
            raise SystemExit("sketch did not answer (is sketches/stream uploaded?)")
        s.write(b"h\n" if a.const else f"f{a.freq}\n".encode()); s.flush()
        t_start = time.time() + 0.5
        last = 0
        while time.time() < t_start + a.time:
            line = s.readline().decode(errors="ignore").strip()
            if line.startswith("#"):
                meta.append(line); continue
            p = line.split(",")
            if len(p) == 2 and p[0].isdigit() and p[1].isdigit() and time.time() > t_start:
                rows.append((int(p[0]), int(p[1])))
            el = int(time.time() - t_start)
            if el >= last + 30:
                last = el; print(f"  {el} s / {a.time:.0f} s", flush=True)
        s.write(b"p\n"); s.flush(); time.sleep(0.2)
        while s.in_waiting:
            line = s.readline().decode(errors="ignore").strip()
            if line.startswith("#"):
                meta.append(line)
        s.write(b"f0\n"); s.flush()
    with open(raw, "w") as fh:
        fh.write(f"# tag={a.tag} T={a.time} f={a.freq} const={a.const} "
                 f"time={datetime.now().isoformat(timespec='seconds')}\n")
        for m in meta:
            fh.write(m + "\n")
        fh.write("ref,adc\n")
        for r, x in rows:
            fh.write(f"{r},{x}\n")
    overruns = [m for m in meta if "overruns=" in m]
    print(f"saved {len(rows)} samples ({len(rows)/a.time:.0f}/s); last status: "
          f"{overruns[-1] if overruns else '?'}")

with open(raw) as fh:
    data = [l.strip().split(",") for l in fh if l[0].isdigit()]
x = np.array([int(d[1]) for d in data], float)

# ------------------------------------------------------------------ stats
codes, counts = np.unique(x.astype(int), return_counts=True)
res = []
for N in NS:
    k = len(x) // N
    if k < 10:
        continue
    m = x[:k * N].reshape(k, N).mean(axis=1)
    res.append((N, k, m.mean(), m.std(ddof=1)))
s1 = res[0][3]
# drift: linear trend over the whole record (per minute), from 1 s means
k1 = len(x) // 1000
sec = x[:k1 * 1000].reshape(k1, 1000).mean(axis=1)
slope = np.polyfit(np.arange(k1), sec, 1)[0] * 60

print(f"\n{a.tag}: {len(x)} samples, mean = {x.mean():.3f} codes "
      f"= {x.mean()*LSB_MV:.1f} mV, raw sigma = {s1:.3f} codes")
print("  histogram: " + ", ".join(f"{c}: {n}" for c, n in zip(codes, counts)))
print(f"  drift of 1 s means: {slope:+.3f} codes/min, range {sec.min():.2f}..{sec.max():.2f}")
print(f"  {'N':>5} {'blocks':>6} {'mean':>9} {'sigma':>8} {'s1/sqrtN':>9} {'ratio':>6}  sigma[mV]")
for N, k, mu, sd in res:
    ideal = s1 / np.sqrt(N)
    print(f"  {N:5d} {k:6d} {mu:9.3f} {sd:8.4f} {ideal:9.4f} {sd/ideal if ideal else float('nan'):6.2f}"
          f"  {sd*LSB_MV:7.3f}")
Nf = np.array([r[0] for r in res]); sf = np.array([r[3] for r in res])
ok = sf > 0
slope_log = np.polyfit(np.log(Nf[ok]), np.log(sf[ok]), 1)[0] if ok.sum() >= 2 else float("nan")
print(f"  log-log slope of sigma(N) over all N: {slope_log:.3f} (white noise: -0.5)")
warn = []
if s1 < 0.3: warn.append("raw noise < 0.3 LSB -> ADC 'frozen', averaging cannot go below 1 LSB")
if len(x) < 0.95 * 1000 * (a.time if not a.file else len(x) / 1000): warn.append("samples lost")
print("  CHECK: " + ("OK" if not warn else "; ".join(warn)))

table = os.path.join(OUT, "prumerovani.csv")
new = not os.path.exists(table)
with open(table, "a", newline="") as fh:
    w = csv.writer(fh)
    if new:
        w.writerow(["tag", "N", "blocks", "mean", "sigma", "sigma_ideal", "sigma_mV",
                    "raw_sigma", "drift_per_min", "file"])
    for N, k, mu, sd in res:
        w.writerow([a.tag, N, k, f"{mu:.4f}", f"{sd:.5f}", f"{s1/np.sqrt(N):.5f}",
                    f"{sd*LSB_MV:.4f}", f"{s1:.4f}", f"{slope:.4f}", os.path.basename(raw)])

# ------------------------------------------------------------------ plot
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
ax[0].bar(codes, counts, width=0.8)
ax[0].set(xlabel="kód ADC", ylabel="počet vzorků", title=f"{a.tag}: histogram surových dat",
          yscale="log")
ax[0].set_xticks(codes)
ax[1].loglog(Nf[ok], sf[ok], "o-", label="naměřeno")
nn = np.array([1, 1000])
ax[1].loglog(nn, s1 / np.sqrt(nn), "--", label=f"bílý šum σ₁/√N (sklon −0,5)")
ax[1].set(xlabel="N (počet průměrovaných vzorků)", ylabel="σ průměru [kód ADC]",
          title=f"σ(N), sklon {slope_log:.2f}")
sec_ax = ax[1].secondary_yaxis("right", functions=(lambda c: c * LSB_MV, lambda v: v / LSB_MV))
sec_ax.set_ylabel("σ [mV]")
ax[1].legend(fontsize=8); ax[1].grid(True, which="both", alpha=0.3)
ax[2].plot(np.arange(k1), sec, ".-", ms=3)
ax[2].set(xlabel="čas [s]", ylabel="průměr za 1 s (N = 1000) [kód]",
          title=f"Drift: {slope:+.3f} kódu/min")
fig.tight_layout()
png = os.path.join(OUT, f"{a.tag}.png")
fig.savefig(png, dpi=130)
print(f"  -> {raw}\n  -> {png}\n  -> {table}")
