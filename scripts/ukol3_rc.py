#!/usr/bin/env python3
"""Task 3: record the original-method RC stream ("millis;voltage" at 9600 baud).

Usage:
  python3 ukol3_rc.py <name> --R 10e3 [-t 30]
Flip the switch to 5V (charging), wait, flip to GND (discharging), wait.
Writes data/ukol3_rc_puvodni/<name>.csv: '# key=value' metadata, then
t_ms,U,pc_s (U as printed by the sketch, 2 decimals; pc_s = PC receive time). Prints a short summary
including the real sampling period (differences of millis()).
"""
import argparse, os, time
import numpy as np
import serial

ap = argparse.ArgumentParser()
ap.add_argument("name", help="e.g. rc_10k_rep1")
ap.add_argument("--R", type=float, required=True, help="series resistor [ohm]")
ap.add_argument("-t", type=float, default=30, help="duration [s]")
ap.add_argument("-p", "--port", default="/dev/ttyACM0")
a = ap.parse_args()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data",
                   "ukol3_rc_puvodni", a.name + ".csv")
os.makedirs(os.path.dirname(out), exist_ok=True)

for _ in range(40):                      # wait up to 20 s for the board to (re)appear
    if os.path.exists(a.port):
        break
    time.sleep(0.5)
else:
    raise SystemExit(f"port {a.port} not found - is the USB cable plugged in?")
rows = []
with serial.Serial(a.port, 9600, timeout=2) as s:
    time.sleep(2.0)                      # board resets on port open
    s.reset_input_buffer()
    print(f"RECORDING {a.t:.0f} s: flip the switch to 5V, wait, then to GND ...",
          flush=True)
    t_end = time.time() + a.t
    while time.time() < t_end:
        line = s.readline().decode(errors="ignore").strip()
        parts = line.split(";")
        if len(parts) == 2:
            try:
                rows.append((int(parts[0]), float(parts[1]), time.time()))
            except ValueError:
                pass

d = np.array(rows)
d = d[1:]                                # first line may be partial
with open(out, "w") as f:
    f.write(f"# method=original_millis_9600\n# R_ohm={a.R}\n# name={a.name}\n"
            "# pc_s = PC receive time (time.time) for clock calibration\nt_ms,U,pc_s\n")
    for t, u, pc in d:
        f.write(f"{int(t)},{u:.2f},{pc:.4f}\n")
dt = np.diff(d[:, 0])
print(f"saved {len(d)} points -> {out}")
print(f"U {d[:, 1].min():.2f}..{d[:, 1].max():.2f} V, sampling period: mean "
      f"{dt.mean():.2f} ms, sd {dt.std():.2f} ms, min {dt.min():.0f}, max {dt.max():.0f} ms")
# clock check: Arduino millis vs PC time (receive latency jitter is a few ms and
# averages out over a 60 s line fit)
pc = d[:, 2] - d[0, 2]
ms = (d[:, 0] - d[0, 0]) / 1000
k = np.polyfit(ms, pc, 1)[0]
print(f"clock: Arduino counted {ms[-1]:.3f} s while PC measured {pc[-1]:.3f} s "
      f"-> PC/Arduino = {k:.5f} ({(k - 1) * 100:+.3f} %)")
