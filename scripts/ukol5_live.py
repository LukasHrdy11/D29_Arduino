#!/usr/bin/env python3
"""Task 5: live readout of the optical signal for aiming the LED (sketch sketches/stream).

Usage (in your own terminal, Ctrl+C to stop):
  python3 scripts/ukol5_live.py
Every 0.5 s prints delta = mean(LED on) - mean(LED off) over the last ~1 s
(whole 7 Hz periods), the maximum so far and a bar. Tilt the LED gently until
the value peaks. Ambient light cancels, so the box may stay open while aiming.
"""
import sys, time
from collections import deque
import numpy as np
import serial

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
SKIP = 2

s = serial.Serial(PORT, 115200, timeout=1)
t0 = time.time()
while time.time() - t0 < 8:
    if s.readline().decode(errors="ignore").startswith("# fs="):
        break
s.write(b"f7\n"); s.flush()
buf = deque(maxlen=1136)          # 8 periods of 142 samples ~ 1.1 s
best, t_print = 0.0, time.time()
try:
    while True:
        p = s.readline().decode(errors="ignore").strip().split(",")
        if len(p) == 2 and p[0].isdigit() and p[1].isdigit():
            buf.append((int(p[0]), int(p[1])))
        if time.time() - t_print < 0.5 or len(buf) < 400:
            continue
        t_print = time.time()
        d = np.array(buf)
        ref, x = d[:, 0], d[:, 1]
        ok = np.ones(len(x), bool)
        for e in np.flatnonzero(np.diff(ref) != 0) + 1:
            ok[e:e + SKIP] = False
        delta = x[(ref == 1) & ok].mean() - x[(ref == 0) & ok].mean()
        best = max(best, delta)
        bar = "#" * int(60 * delta / best) if best > 0 else ""
        print(f"\r signal {delta:7.2f}   max {best:7.2f}  |{bar:<60}|", end="", flush=True)
except KeyboardInterrupt:
    pass
finally:
    s.write(b"f0\n"); s.flush(); s.close()
    print(f"\nmax = {best:.2f} codes, LED off")
