#!/usr/bin/env python3
"""Task 1.2: record one potentiometer position and append a row to the table.

Usage:
  python3 ukol1_potenciometr.py <label> [--dmm 2.483] [-n 50]
Each run resets the board (port open), reads '# vcc_mV=', then N samples.
Appends to data/ukol1_seznameni/potenciometr.csv.
"""
import argparse, csv, os, time
import numpy as np
import serial

ap = argparse.ArgumentParser()
ap.add_argument("label", help="position name, e.g. p1, stred")
ap.add_argument("--dmm", type=float, default=None, help="multimeter voltage [V]")
ap.add_argument("-n", type=int, default=50)
ap.add_argument("-p", "--port", default="/dev/ttyACM0")
ap.add_argument("-o", "--out", default=os.path.join(os.path.dirname(__file__),
                "..", "data", "ukol1_seznameni", "potenciometr.csv"))
a = ap.parse_args()

vcc = None
codes = []
with serial.Serial(a.port, 115200, timeout=3) as s:
    t0 = time.time()
    while len(codes) < a.n and time.time() - t0 < 15:
        line = s.readline().decode(errors="ignore").strip()
        if "# vcc_mV=" in line:            # first line may carry reset garbage
            vcc = int(line.split("# vcc_mV=")[1]) / 1000
            codes = []                      # only samples after metadata
        elif vcc is not None and "," in line and not line.startswith("#"):
            try:
                codes.append(int(line.split(",")[0]))
            except ValueError:
                pass
if vcc is None or not codes:
    raise SystemExit("no data (is the read_voltage sketch uploaded and the monitor closed?)")

c = np.array(codes)
mean, sd = c.mean(), c.std(ddof=1)
u5 = mean * 5.0 / 1023                     # official example formula
uv = mean * vcc / 1024                     # using bandgap-estimated Vcc
row = dict(label=a.label, n=len(c), adc_mean=f"{mean:.2f}", adc_sd=f"{sd:.2f}",
           adc_min=c.min(), adc_max=c.max(), vcc_bandgap=f"{vcc:.3f}",
           U_5V=f"{u5:.4f}", U_vcc=f"{uv:.4f}",
           U_dmm="" if a.dmm is None else f"{a.dmm:.4f}")
new = not os.path.exists(a.out)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=row.keys())
    if new:
        w.writeheader()
    w.writerow(row)
print(f"{a.label}: ADC = {mean:.2f} ± {sd:.2f} (min {c.min()}, max {c.max()}), "
      f"Vcc(bandgap) = {vcc:.3f} V, U(5V) = {u5:.4f} V, U(Vcc) = {uv:.4f} V"
      + ("" if a.dmm is None else f", U(DMM) = {a.dmm:.4f} V"))
