#!/usr/bin/env python3
"""Task 4 side experiment: output resistance of pins D7 and D8 (sketch sketches/rpin).

  python3 ukol4_rpin.py record rpin_220 --RL 220      # records and analyses
  python3 ukol4_rpin.py analyse data/ukol4_rpin/rpin_220.csv [...]

All voltages are ADC codes (ratiometric, the scale cancels):
  c_H = A1 with D7 HIGH and no current (unloaded HIGH = Vcc), c_L = ADC zero level.
  HL (D7 HIGH, D8 LOW):  I = (A1 - A2)/R_L,  R7_HIGH = (c_H - A1)/I,  R8_LOW = (A2 - c_L)/I
  LH (D7 LOW,  D8 HIGH): I = (A2 - A1)/R_L,  R8_HIGH = (c_H - A2)/I,  R7_LOW = (A1 - c_L)/I
The measured R includes the jumper wire and breadboard contacts between the pin and
the sense row; this is exactly what is in series with R in the RC measurement.
Uncertainty: spread of the cycles / sqrt(n); ADC quantisation of a static voltage
(+-0.29 LSB per reading, no dithering -> does not average out); ADC linearity
(+-1 LSB assumed on differences); tolerance of R_L (rectangular). For LOW the
unloaded reading is clipped at code 0, so the ADC zero level -1.07 +- 0.30 codes is
taken from the discharge asymptotes B of task 4 (mean of 30 fits, sd 0.07).
"""
import argparse, glob, json, os, sys, time
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "ukol4_rpin")
ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["record", "analyse"])
ap.add_argument("names", nargs="+")
ap.add_argument("--RL", type=float, help="load resistor [ohm] (record)")
ap.add_argument("--RL-tol", type=float, default=0.01)
a = ap.parse_args()


def record(name, RL):
    import serial
    ports = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
    if not ports:
        sys.exit("No /dev/ttyACM* or /dev/ttyUSB* found. Is the board plugged in?")
    ser = serial.Serial(ports[0], 115200, timeout=2)
    lines, started = [], False
    while True:
        ln = ser.readline().decode(errors="replace").strip()
        if ln.startswith("# rpin"):
            started = True
        if not started or not ln:
            continue
        if ln == "END":
            break
        lines.append(ln)
    ser.close()
    fn = os.path.join(OUT, name + ".csv")
    with open(fn, "w") as fh:
        fh.write(f"# name={name}\n# RL_ohm={RL:g}\n")
        fh.write(lines[0] + "\nstate,cycle,A1,A2\n")
        fh.write("\n".join(lines[1:]) + "\n")
    print("saved ->", os.path.abspath(fn))
    return fn


def analyse(fn):
    RL = None
    d = {}
    for ln in open(fn):
        if ln.startswith("# RL_ohm="):
            RL = float(ln.split("=")[1])
        elif ln[0] in "OHL" and not ln.startswith("state"):
            s, c, a1, a2 = ln.strip().split(",")
            d.setdefault(s, []).append((float(a1), float(a2)))
    d = {k: np.array(v) for k, v in d.items()}
    cH, cL = d["O_H"][:, 0], d["O_L"][:, 0]
    hl, lh = d["HL"], d["LH"]
    I_hl = (hl[:, 0] - hl[:, 1]) / RL            # codes per ohm
    I_lh = (lh[:, 1] - lh[:, 0]) / RL
    # the unloaded LOW level is clipped at code 0, so the true zero level is taken
    # from the discharge asymptotes of task 4 (OFFSET); HIGH uses the measured c_H
    R = {"D7 HIGH": ((cH - hl[:, 0]) / I_hl, hl[:, 0], I_hl, False),
         "D8 LOW": ((hl[:, 1] - OFFSET) / I_hl, hl[:, 1], I_hl, True),
         "D8 HIGH": ((cH - lh[:, 1]) / I_lh, lh[:, 1], I_lh, False),
         "D7 LOW": ((lh[:, 0] - OFFSET) / I_lh, lh[:, 0], I_lh, True)}
    full = cH.mean() - OFFSET
    mA = I_hl.mean() * VCC / full * 1e3
    print(f"\n=== {os.path.basename(fn)}: R_L = {RL:g} Ω, {len(cH)} cycles, I = {mA:.2f} mA")
    print(f"unloaded HIGH c_H = {cH.mean():.3f} (sd {cH.std(ddof=1):.3f}); unloaded LOW "
          f"reads {cL.mean():.3f} (clipped), zero level used {OFFSET:.2f} ± {EOFFSET:.2f}")
    print("| pin, stav | R [Ω] | opakování | kvantování | linearita ADC | nula ADC | R_L | "
          "celkem | pozn. |")
    print("|---|---|---|---|---|---|---|---|---|")
    out = {}
    for k, (r, raw, I, low) in R.items():
        Im = I.mean()
        rep = r.std(ddof=1) / np.sqrt(len(r))
        q = np.sqrt(2) * 0.29 / Im             # two static readings, +-0.29 LSB each
        lin = 1.0 / Im                         # +-1 LSB on the difference
        off = EOFFSET / Im if low else 0.0
        rl = r.mean() * a.RL_tol / np.sqrt(3)
        tot = np.sqrt(rep ** 2 + q ** 2 + lin ** 2 + off ** 2 + rl ** 2)
        note = ""
        if low and raw.mean() < 3:
            note = "NEPOUŽITELNÉ: napětí pod nulou ADC (oříznuto)"
        out[k] = dict(R=r.mean(), eR=tot, I_mA=mA, ok=not note)
        print(f"| {k} | {r.mean():.2f} | {rep:.2f} | {q:.2f} | {lin:.2f} | {off:.2f} | "
              f"{rl:.2f} | **{tot:.2f}** | {note} |")
    return out


OFFSET, EOFFSET = -1.07, 0.30    # ADC zero level [codes]: task 4 discharge asymptotes
VCC = 5.0                        # only to print the current in mA

if a.mode == "record":
    analyse(record(a.names[0], a.RL))
    sys.exit()

res = {os.path.basename(fn): analyse(fn) for fn in a.names}
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 4.5))
sty = {"D7 HIGH": ("#c62828", "o", -0.04), "D8 HIGH": ("#c62828", "s", 0.04),
       "D7 LOW": ("#1565c0", "o", -0.04), "D8 LOW": ("#1565c0", "s", 0.04)}
for k, (col, mk, off) in sty.items():
    pts = [(r[k]["I_mA"], r[k]["R"], r[k]["eR"]) for r in res.values() if r[k]["ok"]]
    I, R, E = np.array(pts).T
    ax.errorbar(I * (1 + off), R, yerr=E, fmt=mk, ms=6, color=col, capsize=3, lw=1,
                mfc="none" if mk == "s" else col, label=k)
ax.set_xscale("log")
ax.set(xlabel="proud pinem [mA]", ylabel="výstupní odpor pinu [Ω]",
       title="výstupní odpor pinů D7 a D8 (zátěž 220 Ω, 1 kΩ, 10 kΩ)", ylim=(0, None))
ax.legend(frameon=False, ncol=2)
ax.grid(True, color="#dddddd", lw=0.6, which="both")
ax.set_axisbelow(True)
fig.tight_layout()
png = os.path.join(OUT, "rpin.png")
fig.savefig(png, dpi=150)
json.dump(res, open(os.path.join(OUT, "rpin.json"), "w"), indent=1)
print("\ngraph ->", os.path.abspath(png))
