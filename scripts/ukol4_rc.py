#!/usr/bin/env python3
"""Task 4: record buffered RC step responses from sketches/rc_buffer.

  python3 ukol4_rc.py rc_10k_rep1 --R 10e3 --dt 500 --settle 2000 [--reps 3]

Writes data/ukol4_rc_buffer/<name>.csv:
  '# key=value' metadata, '# clock us=<micros> pc=<PC raw monotonic time>' lines (resonator
  calibration), header 'phase,rep,i,adc', data rows.
"""
import argparse, glob, os, sys, time
import serial

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "ukol4_rc_buffer")

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--R", type=float, required=True, help="nominal resistance [ohm]")
ap.add_argument("--dt", type=int, required=True, help="sampling period [us]")
ap.add_argument("--settle", type=int, required=True, help="wait before each step [ms]")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("-p", "--port")
a = ap.parse_args()

ports = [a.port] if a.port else sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
if not ports:
    sys.exit("No /dev/ttyACM* or /dev/ttyUSB* found. Is the board plugged in?")
os.makedirs(OUT, exist_ok=True)
out = os.path.join(OUT, a.name + ".csv")

ser = serial.Serial(ports[0], 115200, timeout=1)   # opening resets the board
t_wait = time.time() + 5
while time.time() < t_wait:
    if ser.readline().decode(errors="replace").strip() == "# ready":
        break
else:
    sys.exit("sketch did not answer '# ready' (is rc_buffer uploaded?)")
ser.write(f"c {a.dt} {a.settle} {a.reps}\n".encode())

dur = a.reps * 2 * (a.settle / 1e3 + 500 * a.dt / 1e6 + 0.6) + 1
print(f"measuring ~{dur:.0f} s: {a.reps}x charging + {a.reps}x discharging, "
      f"DT = {a.dt} us, window after step = {450 * a.dt / 1e3:.0f} ms")
lines, rows = [], 0
while True:
    raw = ser.readline()
    pc = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)   # not slewed by NTP
    line = raw.decode(errors="replace").strip()
    if not line:                           # readline timeout during settle, keep waiting
        continue
    if line == "END":
        break
    if line.startswith("# clock"):
        line += f" pc={pc:.6f}"
    if line.startswith("#"):
        if "capture" in line or "dt_us" in line:
            print(line)
    else:
        rows += 1
    lines.append(line)
ser.close()

with open(out, "w") as fh:
    fh.write(f"# method=buffered_rc_buffer\n# pc_clock=monotonic_raw\n# name={a.name}\n# R_ohm={a.R:g}\n")
    data_started = False
    for ln in lines:
        if not ln.startswith("#") and not data_started:
            fh.write("phase,rep,i,adc\n")
            data_started = True
        fh.write(ln + "\n")
print(f"saved {rows} samples -> {os.path.abspath(out)}")
