#!/usr/bin/env python3
"""Task 7.2 (protocol): weak optical signal below the noise level, lock-in (sketch sketches/stream).

Usage (from the repository root, venv active):
  python3 scripts/ukol6_slaby_signal.py live                 # tuning, Ctrl+C to stop
  python3 scripts/ukol6_slaby_signal.py mer <tag> -t 180     # measurement + analysis
  python3 scripts/ukol6_slaby_signal.py mer <tag> -t 60 -f 0 # control: LED off
  python3 scripts/ukol6_slaby_signal.py ana <file.csv>       # re-analyse only

live: every second prints the background level (mean), share of zero codes
      (ADC clipping), raw noise sigma, the 2 s lock-in amplitude and the
      cumulative in-phase/quadrature amplitude since start.
mer:  records T s (sample-count based, the first 1 s is dropped), saves raw
      `ref,adc` to data/ukol6_lockin/slaby_<tag>.csv, then analyses:
      I/Q (in-phase / quadrature relative to the LED reference) over whole
      periods, 5 s blocks, sigma(T) for block lengths of 1..N periods,
      SNR(T), amplitude spectrum, raw noise, clipping. PNG + JSON next to CSV.
With -f 0 the LED stays off and the data are demodulated with a synthetic
reference at the nominal frequency (same phase convention).
"""
import argparse, json, os, sys, time
from datetime import datetime
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "ukol6_lockin")
FS = 1000.0
BLOCK_S = 5.0

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["live", "mer", "ana"])
ap.add_argument("arg", nargs="?", help="tag (mer) or CSV file (ana)")
ap.add_argument("-t", "--time", type=float, default=180.0)
ap.add_argument("-f", "--freq", type=float, default=7.0)
ap.add_argument("-p", "--port", default="/dev/ttyACM0")
a = ap.parse_args()


def half_of(f):
    return int(round(FS / (2 * f)))


def open_port():
    import serial
    s = serial.Serial(a.port, 115200, timeout=1)
    t0 = time.time()
    while time.time() - t0 < 8:
        if s.readline().decode(errors="ignore").startswith("# fs="):
            return s
    raise SystemExit("sketch did not answer (is sketches/stream uploaded? Serial Monitor closed?)")


def parse(line):
    p = line.strip().split(",")
    if len(p) == 2 and p[0].isdigit() and p[1].isdigit():
        return int(p[0]), int(p[1])
    return None


def pooled_sd(zb):
    """sigma of one block from I and Q together, each about its own mean
    (the signal sits in I, so the two means differ)."""
    zb = np.asarray(zb)
    dev = np.r_[zb.real - zb.real.mean(), zb.imag - zb.imag.mean()]
    return np.sqrt(np.sum(dev ** 2) / (2 * len(zb) - 2))


def demod(x, P, k0=0):
    """Complex amplitude z = X + iY of the fundamental (period P samples),
    rotated to the phase of the ideal LED reference (ref = 1 in the first
    half of every period starting at k0). Re z = in-phase I, Im z = quadrature Q."""
    k = np.arange(len(x)) + k0
    w = 2 * np.pi / P
    xc = x - x.mean()
    z = 2 * np.mean(xc * np.exp(1j * w * k))   # X + iY with X=2<x cos>, Y=2<x sin>
    r = (((k % P) < P // 2).astype(float))
    zr = 2 * np.mean((r - r.mean()) * np.exp(1j * w * k))
    return z * np.conj(zr) / abs(zr)


# ------------------------------------------------------------------ live
if a.mode == "live":
    P = 2 * half_of(a.freq)
    s = open_port()
    s.write(f"f{a.freq}\n".encode()); s.flush()
    data, t_print, start = [], time.time(), None
    print(f"lock-in {FS / P:.3f} Hz. Target: background 10-30 codes, zeros < 1 %. Ctrl+C = stop")
    print(f"{'t[s]':>5} {'mean':>6} {'zeros':>6} {'sigma':>6} {'R_2s':>6} | {'I_cum':>7} {'Q_cum':>7} {'±':>6}")
    try:
        while True:
            v = parse(s.readline().decode(errors="ignore"))
            if v:
                data.append(v)
            if time.time() - t_print < 1.0 or len(data) < 2 * P + 10:
                continue
            t_print = time.time()
            d = np.array(data)
            if start is None:
                e = np.flatnonzero(np.diff(d[:, 0]) == 1)
                if not len(e):
                    continue
                start = e[0] + 1
            x = d[start:, 1].astype(float)
            nper = len(x) // P
            if nper < 2:
                continue
            x = x[:nper * P]
            last = x[-(min(nper, int(2 * FS / P))) * P:]
            z2 = demod(last, P)
            zc = demod(x, P)
            # uncertainty of the cumulative value from 1 s blocks of whole periods
            m = max(1, int(FS / P)); nb = nper // m
            zb = np.array([demod(x[j*m*P:(j+1)*m*P], P) for j in range(nb)])
            err = pooled_sd(zb) / np.sqrt(nb) if nb > 2 else np.nan
            w = x[-2000:]
            print(f"{len(x) / FS:5.0f} {w.mean():6.1f} {100 * np.mean(w == 0):5.1f}% {w.std():6.2f} "
                  f"{abs(z2):6.2f} | {zc.real:7.3f} {zc.imag:7.3f} {err:6.3f}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        s.write(b"f0\n"); s.flush(); s.close()
        print("\nLED off")
    sys.exit()

# ------------------------------------------------------------------ record
if a.mode == "mer":
    if not a.arg:
        raise SystemExit("give a tag, e.g. mer hlavni")
    os.makedirs(OUT, exist_ok=True)
    raw = os.path.join(OUT, f"slaby_{a.arg}.csv")
    if os.path.exists(raw):
        raise SystemExit(f"{raw} exists, choose another tag")
    s = open_port()
    s.write(f"f{a.freq}\n".encode()); s.flush()
    need = int((a.time + 1.0) * FS)
    data, bad, t0 = [], 0, time.time()
    print(f"recording {a.time:.0f} s at f = {a.freq} Hz ... do not move above the setup")
    while len(data) < need:
        line = s.readline().decode(errors="ignore")
        v = parse(line)
        if v:
            data.append(v)
            if len(data) % 10000 == 0:
                print(f"  {len(data) / FS:5.0f} s", flush=True)
        elif line and not line.startswith("#"):
            bad += 1
    s.write(b"p\n"); s.flush(); time.sleep(0.3)
    status = [l for l in s.read(s.in_waiting).decode(errors="ignore").splitlines() if l.startswith("# fs=")]
    s.write(b"f0\n"); s.flush(); s.close()
    data = data[int(FS):]
    with open(raw, "w") as fh:
        fh.write(f"# tag={a.arg} f_req={a.freq} T={a.time} bad_lines={bad} "
                 f"time={datetime.now().isoformat(timespec='seconds')} wall={time.time() - t0:.1f}\n")
        for l in status:
            fh.write(l + "\n")
        fh.write("ref,adc\n")
        for r, x in data:
            fh.write(f"{r},{x}\n")
    print("  " + (status[-1] if status else "no status line"))
else:
    raw = a.arg

# ------------------------------------------------------------------ analysis
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

hdr = [l for l in open(raw) if l.startswith("#")]
f_req = float(hdr[0].split("f_req=")[1].split()[0])
overruns = next((int(l.split("overruns=")[1].split()[0]) for l in hdr if "overruns=" in l), None)
bad = int(hdr[0].split("bad_lines=")[1].split()[0])
rows = [l for l in open(raw) if l[0].isdigit()]
d = np.array([[int(v) for v in l.split(",")] for l in rows])
ref, x = d[:, 0], d[:, 1].astype(float)

f_nom = f_req if f_req > 0 else (a.freq if a.freq > 0 else 7.0)
P = 2 * half_of(f_nom); f_true = FS / P
e = np.flatnonzero(np.diff(ref) == 1)
start = e[0] + 1 if len(e) else 0
xs, rs = x[start:], ref[start:]
nper = len(xs) // P
xs, rs = xs[:nper * P], rs[:nper * P]
T = len(xs) / FS
led_on = rs.any()

z = demod(xs, P)
I, Q = z.real, z.imag

# blocks of whole periods ~5 s
mB = max(1, int(round(BLOCK_S * FS / P)))
nB = nper // mB
zB = np.array([demod(xs[j*mB*P:(j+1)*mB*P], P) for j in range(nB)])
sig5 = pooled_sd(zB)          # sigma of one 5 s block (I and Q pooled)
eI = sig5 / np.sqrt(nB)

# sigma(T): block length 1, 2, 4 ... periods, needs >= 6 blocks
sT = []
m = 1
while nper // m >= 6:
    nb = nper // m
    zb = np.array([demod(xs[j*m*P:(j+1)*m*P], P) for j in range(nb)])
    sT.append((m * P / FS, pooled_sd(zb), nb))
    m *= 2
sT = np.array(sT)
sl, ic = np.polyfit(np.log(sT[:, 0]), np.log(sT[:, 1]), 1)
# cumulative SNR(T) = |I(T)| / sigma(T) for growing measurement time
cum = []
for npp in np.unique(np.geomspace(1, nper, 25).astype(int)):
    zc = demod(xs[:npp * P], P)
    sc = np.exp(ic) * (npp * P / FS) ** sl
    cum.append((npp * P / FS, zc.real, zc.imag, sc))
cum = np.array(cum)

# raw noise and clipping
on = rs == 1
raw_sigma = np.sqrt(0.5 * (xs[on].var() + xs[~on].var())) if led_on else xs.std()
zeros = np.mean(xs == 0)
mean = xs.mean()
# on-off without the first 2 samples after each edge (electrical edge spike, see 6.2)
ph_k = np.arange(len(xs)) % P
keep = (ph_k % (P // 2)) >= 2
on_off = xs[on & keep].mean() - xs[~on & keep].mean() if led_on else np.nan
edge_spike = float(np.mean(xs[ph_k == 0]) - np.mean(xs[ph_k == P // 2])) / 2

# spectrum (Hann), amplitude in codes
win = np.hanning(len(xs))
F = np.fft.rfft((xs - mean) * win)
fr = np.fft.rfftfreq(len(xs), 1 / FS)
A = 2 * np.abs(F) / win.sum()

def peak(f0, bw=0.5):
    sel = (fr > f0 - bw) & (fr < f0 + bw)
    return float(A[sel].max()) if sel.any() else np.nan

floor = np.median(A[(fr > 2) & (fr < 490)])
lines = {f"{v:g}": round(peak(v), 4) for v in (f_true, 3 * f_true, 50, 100, 150, 159, 200, 250, 428)}

halves = [demod(xs[:nper // 2 * P], P).real, demod(xs[nper // 2 * P:], P).real]

res = dict(file=os.path.basename(raw), f_true=round(f_true, 4), T_s=round(T, 1), periods=int(nper),
           led_modulated=bool(led_on), mean=round(mean, 2), zeros_pct=round(100 * zeros, 2),
           raw_sigma=round(raw_sigma, 3), I=round(I, 4), Q=round(Q, 4), err_IQ=round(eI, 4),
           R=round(abs(z), 4), phase_deg=round(np.degrees(np.angle(z)), 1),
           on_off=round(on_off, 4) if led_on else None, edge_spike=round(edge_spike, 3),
           SNR=round(I / eI, 1), signal_over_rawnoise=round(abs(I) / raw_sigma, 3),
           sigma_5s=round(sig5, 4), n_blocks5=int(nB), sigmaT_slope=round(sl, 3),
           sigmaT=[[round(t, 3), round(v, 5), int(n)] for t, v, n in sT],
           I_first_half=round(halves[0], 4), I_second_half=round(halves[1], 4),
           spectrum_floor=round(floor, 5), spectrum_lines=lines,
           overruns=overruns, bad_lines=bad)
json.dump(res, open(raw.replace(".csv", ".json"), "w"), indent=1)

print(f"\n{res['file']}: T = {T:.0f} s, {nper} periods at {f_true:.3f} Hz, LED {'modulated' if led_on else 'OFF'}")
print(f"  background mean {mean:.2f} codes, zeros {100*zeros:.1f} %, raw noise sigma {raw_sigma:.3f} codes")
print(f"  I (in-phase) = {I:+.4f} ± {eI:.4f} codes   Q (quadrature) = {Q:+.4f} ± {eI:.4f}")
print(f"  R = {abs(z):.4f}, phase = {np.degrees(np.angle(z)):+.1f} deg, SNR = I/err = {I/eI:.1f}")
if led_on:
    print(f"  on-off (2 samples after edges skipped) = {on_off:+.4f} (ideal square: (pi/2) I = {np.pi/2*I:+.4f})")
print(f"  first sample after edge, half of (rise - fall) = {edge_spike:+.3f} codes (electrical edge spike)")
print(f"  signal / raw noise = {abs(I)/raw_sigma:.3f};  sigma(5 s block) = {sig5:.4f}, "
      f"sigma(T) slope = {sl:.3f} (ideal -0.5)")
print(f"  halves: I = {halves[0]:+.4f} / {halves[1]:+.4f}")
print("  spectrum lines [codes]: " + ", ".join(f"{k} Hz: {v:.4f}" for k, v in lines.items())
      + f"  (floor {floor:.4f})")
warn = []
if overruns: warn.append(f"{overruns} overruns")
if bad: warn.append(f"{bad} corrupted lines")
if zeros > 0.01: warn.append(f"{100*zeros:.1f} % zeros: ADC clipping, raise the background")
if abs(Q) > 3 * eI: warn.append("Q differs from 0 by > 3 sigma (phase shift or interference at f)")
if abs(halves[0] - halves[1]) > 3 * eI * np.sqrt(2) * 2: warn.append("halves differ (drift / moved)")
if abs(sl + 0.5) > 0.12: warn.append(f"sigma(T) slope {sl:.2f} far from -0.5")
print("  CHECK: " + ("OK" if not warn else "; ".join(warn)))

# ------------------------------------------------------------------ plot
fig, ax = plt.subplots(2, 3, figsize=(17, 9))
tt = np.arange(1000) / FS
ax[0, 0].plot(tt, xs[:1000], lw=0.6, label="ADC")
ax[0, 0].plot(tt, mean + 3 * raw_sigma * (rs[:1000] - 0.5), lw=1, color="C3", alpha=0.7,
              label="LED (reference, schematic)")
ax[0, 0].set(xlabel="t [s]", ylabel="ADC [code]", title="raw data, first 1 s")
ax[0, 0].legend(fontsize=8)
wave = xs.reshape(nper, P).mean(axis=0)
ph = np.arange(P) / P
ew = raw_sigma / np.sqrt(nper)
ax[0, 1].plot(ph, wave, ".", ms=3, label=f"mean over {nper} periods")
ax[0, 1].fill_between(ph, wave - ew, wave + ew, alpha=0.2)
ax[0, 1].axvspan(0, 0.5, color="C3", alpha=0.08, label="LED on")
ax[0, 1].set(xlabel="phase of period", ylabel="ADC [code]", title="period-averaged waveform")
ax[0, 1].legend(fontsize=8)
tb = (np.arange(nB) + 0.5) * mB * P / FS
ax[0, 2].errorbar(tb, zB.real, sig5, fmt="o", ms=4, label="I (in-phase)")
ax[0, 2].errorbar(tb + 0.6, zB.imag, sig5, fmt="s", ms=3, alpha=0.6, label="Q (quadrature)")
ax[0, 2].axhspan(I - eI, I + eI, color="C0", alpha=0.2)
ax[0, 2].axhline(0, color="k", lw=0.6)
ax[0, 2].set(xlabel="t [s]", ylabel="amplitude [code]", title=f"{BLOCK_S:g} s blocks, I = {I:.3f} ± {eI:.3f}")
ax[0, 2].legend(fontsize=8)
ax[1, 0].loglog(sT[:, 0], sT[:, 1], "o", label="measured")
ax[1, 0].loglog(sT[:, 0], np.exp(ic) * sT[:, 0] ** sl, "-", label=f"fit, slope {sl:.2f}")
ax[1, 0].loglog(sT[:, 0], sT[0, 1] * (sT[:, 0] / sT[0, 0]) ** -0.5, "--", color="gray", label="slope -0.5")
ax[1, 0].axhline(abs(I), color="C3", lw=0.8, label="|I| (signal)")
ax[1, 0].set(xlabel="averaging time T [s]", ylabel="sigma of I, Q [code]", title="uncertainty vs time")
ax[1, 0].legend(fontsize=8); ax[1, 0].grid(True, which="both", alpha=0.3)
ax[1, 1].plot(cum[:, 0], cum[:, 1], "o-", ms=3, label="I(T)")
ax[1, 1].fill_between(cum[:, 0], cum[:, 1] - cum[:, 3], cum[:, 1] + cum[:, 3], alpha=0.2)
ax[1, 1].plot(cum[:, 0], cum[:, 2], "s-", ms=2, alpha=0.6, label="Q(T)")
ax[1, 1].axhline(0, color="k", lw=0.6)
ax[1, 1].set(xscale="log", xlabel="measurement time T [s]", ylabel="amplitude [code]",
             title="cumulative result ± sigma(T)")
ax2 = ax[1, 1].twinx()
ax2.plot(cum[:, 0], np.abs(cum[:, 1]) / cum[:, 3], "^-", color="C2", ms=3, label="SNR")
ax2.set_ylabel("SNR = |I| / sigma", color="C2")
ax[1, 1].legend(fontsize=8, loc="upper left")
ax[1, 2].semilogy(fr, A, lw=0.4)
for f0, c in [(f_true, "C3"), (50, "gray"), (100, "gray"), (150, "gray"), (159, "C1"), (428, "C1")]:
    ax[1, 2].axvline(f0, color=c, lw=0.6, ls=":")
ax[1, 2].set(xlim=(0, 500), xlabel="f [Hz]", ylabel="amplitude [code]",
             title=f"spectrum (red: {f_true:.2f} Hz, gray: mains, orange: 159/428)")
fig.suptitle(f"{res['file']}: mean {mean:.1f}, raw sigma {raw_sigma:.2f}, "
             f"I = {I:.3f} ± {eI:.3f} code, Q = {Q:.3f}, SNR {I/eI:.1f}")
fig.tight_layout()
png = raw.replace(".csv", ".png"); fig.savefig(png, dpi=120)
print(f"  -> {raw}\n  -> {png}")
