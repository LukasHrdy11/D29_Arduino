#!/usr/bin/env python3
"""Task 1: potentiometer table with uncertainties (markdown to stdout).

Usage:
  python3 ukol1_vysledky.py [csv] [--vcc 5.03] [--vcc-tol 0.10] [--adc-lsb 2]

U = mean_code * Vcc / 1024 (Vcc from the bandgap in each run unless --vcc given).
Standard uncertainties of the mean code [LSB]:
  stat   sd / sqrt(n)
  quant  1/sqrt(12)   (rounding to an integer code, uniform +-0.5 LSB)
  ADC    adc_lsb/sqrt(3)  (absolute accuracy +-2 LSB, uniform)
and of Vcc: Vcc * vcc_tol / sqrt(3) (bandgap tolerance +-10 %, uniform).
u(U)^2 = (Vcc/1024)^2 (stat^2 + quant^2 + ADC^2) + (U * vcc_tol/sqrt(3))^2
Codes 0 and 1023 are saturated: only a bound is given.
"""
import argparse, csv, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("csv", nargs="?", default=os.path.join(os.path.dirname(__file__), "..",
                "data", "ukol1_seznameni", "potenciometr.csv"))
ap.add_argument("--vcc", type=float)
ap.add_argument("--vcc-tol", type=float, default=0.10)
ap.add_argument("--adc-lsb", type=float, default=2.0)
a = ap.parse_args()

rows = sorted(csv.DictReader(open(a.csv)), key=lambda r: float(r["adc_mean"]))
print("| poloha | kód ADC | u_stat [LSB] | Vcc [V] | U [V] | u(ADC+kvant.) [V] "
      "| u(Vcc) [V] | **u(U) [V]** |")
print("|---|---|---|---|---|---|---|---|")
for r in rows:
    n, m, sd = int(r["n"]), float(r["adc_mean"]), float(r["adc_sd"])
    vcc = a.vcc or float(r["vcc_bandgap"])
    lsb = vcc / 1024
    if m <= 0 or m >= 1023:
        lo, hi = (0, lsb) if m <= 0 else (1023 * lsb, None)
        bound = f"< {hi * 1e3:.1f} mV" if m <= 0 else f"≥ {lo:.3f} V (≈ Vcc)"
        print(f"| {r['label']} | {m:.0f} (saturace) | – | {vcc:.3f} | {bound} | – | – | – |")
        continue
    s_stat = sd / np.sqrt(n)
    s_dig = lsb * np.sqrt(s_stat**2 + 1 / 12 + (a.adc_lsb / np.sqrt(3))**2)
    U = m * lsb
    s_vcc = U * a.vcc_tol / np.sqrt(3)
    tot = np.hypot(s_dig, s_vcc)
    print(f"| {r['label']} | {m:.2f} | {s_stat:.3f} | {vcc:.3f} | {U:.4f} | {s_dig:.4f} "
          f"| {s_vcc:.4f} | **{tot:.3f}** |")
