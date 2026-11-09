#!/usr/bin/env python3
"""Task 2: record an LED I-V sweep (turn the pot slowly while it runs).

Usage:
  python3 ukol2_va_led.py <name> [-t 60] [--R 220]
Writes data/ukol2_va_led/<name>.csv: '# key=value' metadata, then
t_ms,a0,a1 (block-averaged ADC codes). Prints a short summary.
"""
import argparse, os, time
import numpy as np
import serial

ap = argparse.ArgumentParser()
ap.add_argument("name", help="e.g. cervena_propustny")
ap.add_argument("-t", type=float, default=60, help="duration [s]")
ap.add_argument("--R", type=float, default=220.0, help="shunt resistor [ohm]")
ap.add_argument("-p", "--port", default="/dev/ttyACM0")
a = ap.parse_args()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data",
                   "ukol2_va_led", a.name + ".csv")
os.makedirs(os.path.dirname(out), exist_ok=True)

meta, rows = [], []
with serial.Serial(a.port, 115200, timeout=3) as s:
    # wait for the start of metadata (board resets on port open)
    t_end = time.time() + 10
    while time.time() < t_end:
        line = s.readline().decode(errors="ignore").strip()
        if "# vcc_mV=" in line:
            meta.append(line[line.index("#"):])
            break
    else:
        raise SystemExit("no metadata (is va_led uploaded and the monitor closed?)")
    print(f"RECORDING {a.t:.0f} s: turn the knob slowly now ...", flush=True)
    t_end = time.time() + a.t
    while time.time() < t_end:
        line = s.readline().decode(errors="ignore").strip()
        if line.startswith("#"):
            meta.append(line)
        elif line.count(",") == 2:
            try:
                rows.append([float(x) for x in line.split(",")])
            except ValueError:
                pass

vcc = int(meta[0].split("=")[1]) / 1000
with open(out, "w") as f:
    f.write("\n".join(meta) + f"\n# R_ohm={a.R}\n# name={a.name}\nt_ms,a0,a1\n")
    for r in rows:
        f.write(f"{int(r[0])},{r[1]:.2f},{r[2]:.2f}\n")

d = np.array(rows)
u0, u1 = d[:, 1] * vcc / 1024, d[:, 2] * vcc / 1024
i_mA = u1 / a.R * 1e3
print(f"saved {len(d)} points -> {out}")
print(f"Vcc = {vcc:.3f} V, U_A0 {u0.min():.3f}..{u0.max():.3f} V, "
      f"U_LED max {(u0 - u1).max():.3f} V, I max {i_mA.max():.2f} mA")
if i_mA.max() > 20:
    print("WARNING: current above 20 mA, do not turn the knob that far next time")
