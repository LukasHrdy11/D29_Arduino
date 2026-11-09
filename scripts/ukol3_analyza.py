#!/usr/bin/env python3
"""Task 3: RC time constant from the original-method stream (t_ms,U).

Usage:
  python3 ukol3_analyza.py file1.csv [file2.csv ...] [--clock-tol 0.005] [--R-tol 0.01]

Per file:
  * edges are found with hysteresis (U < 1 V = low, U > 4 V = high); the edge start is
    the first sample that left the previous plateau by more than 0.03 V
  * every edge is fitted with U = B + A exp(-(t - t_s)/tau) (A, B, tau free, so the
    unknown switching moment between two samples is absorbed in A)
  * fit window: from the first sample with less than 60 % of the 5 V step remaining
    (skips the ~20-30 ms contact bounce of the hand-moved wire; any later part of a
    single-RC curve is the same exponential) to t_s + 8 tau (capped at the next edge)
  * quality check (printed): plateau before the edge >= 5 tau, >= 3 points inside the
    first tau, fit residual rms, tau spread between edges -> OK / WARN
Uncertainties of tau (standard, 1 sigma):
  rep     sd / sqrt(n) over the edges of one direction (or all edges)
  stat    rms of the fit errors / sqrt(n)
  method  quadrature of half differences: window end 5 tau vs 8 tau and fit start at
          40 % vs 60 % remaining (mean over edges)
  clock   if the CSV has PC receive times (pc_s), the Arduino clock is calibrated against
          the PC clock (line fit, floor 1e-4) and all times are corrected; otherwise
          --clock/--clock-err, or the resonator tolerance +-0.5 % rectangular
  dir     |tau_charge - tau_discharge| / 2 (combined value only)
Writes <file>.png (graph) and <file>.json (summary for ukol3_tau_R.py, which does C).
"""
import argparse, json, os, re
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = {"nab": "#c62828", "vyb": "#1565c0"}
LAB = {"nab": "nabíjení", "vyb": "vybíjení"}


def load(path):
    meta, rows = {}, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            m = re.match(r"#\s*(\w+)=(.*)", line)
            if m:
                meta[m.group(1)] = m.group(2)
            elif line and line[0].isdigit():
                rows.append([float(x) for x in line.split(",")])
    d = np.array(rows)
    pc = d[:, 2] if d.shape[1] > 2 else None
    return d[:, 0] / 1000.0, d[:, 1], float(meta["R_ohm"]), pc


def clock_factor(t, pc):
    """real seconds per Arduino second from the PC receive times (line fit)."""
    x, y = t - t[0], pc - pc[0]
    (k, q), cov = np.polyfit(x, y, 1, cov=True)
    return k, np.sqrt(cov[0, 0])


def model(t, A, B, tau):
    return B + A * np.exp(-t / tau)


def find_edges(t, u, lo=1.0, hi=4.0, dep=0.03):
    """Return list of (direction, start index) using hysteresis states."""
    state = None
    cross = []
    for k, v in enumerate(u):
        s = "L" if v < lo else ("H" if v > hi else state)
        if state is not None and s != state:
            cross.append(("nab" if s == "H" else "vyb", k))
        state = s
    edges = []
    for direction, k in cross:
        j = k
        # walk back to the plateau: the first sample that left it is the edge start
        while j > 0 and abs(u[j - 1] - u[max(0, j - 30):j].min()
                            if direction == "nab" else
                            u[max(0, j - 30):j].max() - u[j - 1]) > dep:
            j -= 1
        edges.append((direction, j))
    return edges


def fit_edge(t, u, s, end, tau_guess, n_tau, target, f_start):
    """Fit from the first sample with |U - target| < f_start * 5 V (skips contact bounce
    at the start; any later part of a single-RC curve is the same exponential, only A
    changes), up to t_s + n_tau * tau. Time origin stays at the edge start t_s."""
    stop = min(end, np.searchsorted(t, t[s] + n_tau * tau_guess))
    s2 = s + int(np.argmax(np.abs(u[s:stop] - target) < f_start * 5.0))
    tt, uu = t[s2:stop] - t[s], u[s2:stop]
    A0 = uu[0] - uu[-1]
    p, cov = curve_fit(model, tt, uu, p0=[A0, uu[-1], tau_guess])
    return p, np.sqrt(np.diag(cov)), tt, uu


ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--clock-tol", type=float, default=0.005,
                help="resonator tolerance used when the clock is not calibrated")
ap.add_argument("--clock", type=float, help="calibrated real/Arduino time factor")
ap.add_argument("--clock-err", type=float, default=1e-4)
ap.add_argument("--f-start", type=float, default=0.6,
                help="fit starts where the remaining step is below this fraction of 5 V")
a = ap.parse_args()

summary = []
for path in a.files:
    t, u, R, pc = load(path)
    if pc is not None:
        kclk, eclk = clock_factor(t, pc)
        src = "measured in this file"
    elif a.clock is not None:
        kclk, eclk, src = a.clock, a.clock_err, "--clock"
    else:
        kclk, eclk, src = 1.0, a.clock_tol / np.sqrt(3), "nominal (resonator tolerance)"
    eclk = max(eclk, 1e-4)        # floor: PC clock / USB latency drift
    t = t * kclk                  # Arduino time -> real time
    name = os.path.splitext(os.path.basename(path))[0]
    dt = np.diff(t) * 1e3
    edges = find_edges(t, u)
    starts = [s for _, s in edges] + [len(t)]
    res = []
    for i, (direction, s) in enumerate(edges):
        end = starts[i + 1]
        # rough tau: time to cover 63 % of the step
        seg = u[s:end]
        target = seg[0] + 0.632 * ((5.0 if direction == "nab" else 0.0) - seg[0])
        k63 = np.argmax(seg >= target) if direction == "nab" else np.argmax(seg <= target)
        tau0 = max(t[s + k63] - t[s], 0.005)
        tgt = 5.0 if direction == "nab" else 0.0
        p8, e8, tt, uu = fit_edge(t, u, s, end, tau0, 8, tgt, a.f_start)
        p8, e8, tt, uu = fit_edge(t, u, s, end, p8[2], 8, tgt, a.f_start)
        p5, _, _, _ = fit_edge(t, u, s, end, p8[2], 5, tgt, a.f_start)
        p4, _, _, _ = fit_edge(t, u, s, end, p8[2], 8, tgt, a.f_start - 0.2)
        tau = p8[2]
        meth = np.hypot((p8[2] - p5[2]) / 2, (p8[2] - p4[2]) / 2)
        resid = uu - model(tt, *p8)
        # residual allowed: print rounding / ADC (20 mV) + 1.5 ms of time error (millis
        # has 1 ms resolution, so on the steep start a timing error looks like a voltage one)
        slope = np.abs(p8[0] / tau * np.exp(-tt / tau))
        bad = np.abs(resid) > 0.02 + slope * 1.5e-3
        plateau = t[s] - (t[starts[i - 1]] if i > 0 else t[0])
        n_first = np.sum(tt < tt[0] + tau)
        ok_pl = plateau >= 5 * tau or i == 0 and plateau >= 5 * tau
        res.append(dict(dir=direction, s=s, tau=tau, etau=e8[2], tau4=p4[2], meth=meth, A=p8[0],
                        B=p8[1], rms=resid.std(), maxres=np.abs(resid).max(), nbad=int(bad.sum()),
                        plateau=plateau, n_first=n_first, full=(t[end - 1] - t[s]) >= 5 * tau,
                        tt=tt, uu=uu, p=p8))

    print(f"## {name}: R = {R / 1e3:.1f} kOhm, N = {len(t)}, T = {t[-1] - t[0]:.1f} s")
    print(f"clock: real/Arduino = {kclk:.5f} ± {eclk:.5f} ({src}); times corrected")
    print(f"sampling period: mean {dt.mean():.2f} ms, sd {dt.std():.2f}, "
          f"min {dt.min():.0f}, max {dt.max():.0f} ms")
    print("| # | směr | τ [ms] | ± fit | τ(start 40 %) | U_start→U_konec [V] | rms res [mV] | "
          "max res [mV] | odlehlé | plató před [s] | bodů v 1. τ | kvalita |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    med_rms = np.median([r["rms"] for r in res])
    for i, r in enumerate(res):
        flags = []
        if r["rms"] > 2 * med_rms:
            flags.append(f"rms > 2× medián ({med_rms * 1e3:.1f} mV)")
        if i > 0 and r["plateau"] < 5 * r["tau"]:
            flags.append("krátké plató")
        if not r["full"]:
            flags.append("nedoběhlo")
        if r["n_first"] < 3:
            flags.append("málo bodů")
        if r["nbad"] > 0:
            flags.append(f"{r['nbad']} odlehlé body")
        r["ok"] = not flags
        r["why"] = ", ".join(flags)
        print(f"| {i + 1} | {LAB[r['dir']]} | {r['tau'] * 1e3:.2f} | {r['etau'] * 1e3:.2f} | "
              f"{r['tau4'] * 1e3:.2f} | {r['uu'][0]:.2f}→{r['B']:.2f} | {r['rms'] * 1e3:.1f} | "
              f"{r['maxres'] * 1e3:.1f} | {r['nbad']} | {r['plateau']:.2f} | {r['n_first']} | "
              f"{'OK' if r['ok'] else 'WARN: ' + ', '.join(flags)} |")

    good = [r for r in res if r["ok"]]
    print(f"\nedges: {len(res)} found, {len(good)} OK")
    groups = {"nab": [r for r in good if r["dir"] == "nab"],
              "vyb": [r for r in good if r["dir"] == "vyb"], "vše": good}
    diff = ed = None
    if len(groups["nab"]) > 1 and len(groups["vyb"]) > 1:
        tn = np.array([r["tau"] for r in groups["nab"]])
        tv = np.array([r["tau"] for r in groups["vyb"]])
        diff = tn.mean() - tv.mean()
        ed = np.sqrt(tn.var(ddof=1) / len(tn) + tv.var(ddof=1) / len(tv))
    print("| skupina | n | τ [ms] | opakování | stat | metoda | směr | hodiny | **celkem** |")
    print("|---|---|---|---|---|---|---|---|---|")
    for g, rs in groups.items():
        n = len(rs)
        if n == 0:
            continue
        taus = np.array([r["tau"] for r in rs])
        tau = taus.mean()
        rep = taus.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        st = np.sqrt(np.mean([r["etau"] ** 2 for r in rs]) / n)
        meth = np.mean([r["meth"] for r in rs])
        # direction: charge and discharge should give the same tau; half their
        # difference is taken as a systematic component of the combined value
        dirc = abs(diff) / 2 if (g == "vše" and diff is not None) else 0.0
        clk = tau * eclk
        tot = np.sqrt(rep**2 + st**2 + meth**2 + dirc**2 + clk**2)
        print(f"| {LAB.get(g, g)} | {n} | {tau * 1e3:.2f} | {rep * 1e3:.3f} | {st * 1e3:.3f} | "
              f"{meth * 1e3:.3f} | {dirc * 1e3:.3f} | {clk * 1e3:.3f} | **{tot * 1e3:.2f}** |")
        if g == "vše":
            summary.append(dict(
                name=name, R=R, tau=tau, etau=tot, n=n, n_found=len(res),
                comp=dict(rep=rep, stat=st, meth=meth, dir=dirc, clock=clk),
                diff=diff, ediff=ed, clock=kclk, eclock=eclk, clock_src=src,
                dt_mean=dt.mean() * kclk, dt_sd=dt.std() * kclk,
                pts_per_tau=tau / (dt.mean() * kclk / 1e3),
                example={d: dict(t=(rs_[0]["tt"]).tolist(), u=rs_[0]["uu"].tolist(),
                                 A=rs_[0]["A"], B=rs_[0]["B"], tau=rs_[0]["tau"])
                         for d, rs_ in (("nab", groups["nab"]), ("vyb", groups["vyb"]))
                         if rs_},
                rejected=[dict(i=k + 1, dir=r["dir"], tau=r["tau"], why=r["why"])
                          for k, r in enumerate(res) if not r["ok"]]))
    if diff is not None:
        print(f"\ncharge - discharge: {diff * 1e3:.2f} ± {ed * 1e3:.2f} ms "
              f"({abs(diff) / ed:.1f} sigma)")
    print(f"tau spread over OK edges: sd {np.std([r['tau'] for r in good], ddof=1) * 1e3:.2f} ms"
          f" ({np.std([r['tau'] for r in good], ddof=1) / np.mean([r['tau'] for r in good]):.1%})")

    # ---- graph: overview + aligned edges (linear and semilog) ----
    fig = plt.figure(figsize=(11, 7.5))
    ax0 = fig.add_subplot(2, 1, 1)
    ax1 = fig.add_subplot(2, 2, 3)
    ax2 = fig.add_subplot(2, 2, 4)
    ax0.plot(t - t[0], u, "-", color="#888888", lw=0.6)
    ax0.plot(t - t[0], u, ".", color="#333333", ms=1.5)
    for r in res:
        ax0.axvline(t[r["s"]] - t[0], color=COL[r["dir"]], lw=0.6, alpha=0.5)
    ax0.set_xlabel("t [s]")
    ax0.set_ylabel("U_C [V]")
    ax0.set_title(f"{name}: celý záznam (svislé čáry = nalezené hrany)")
    for r in res:
        if not r["ok"]:
            continue
        c = COL[r["dir"]]
        full = 5.0 if r["dir"] == "vyb" else -5.0
        r["shift"] = r["tau"] * np.log(full / r["A"])  # time already elapsed at t_s
        ts = r["tt"] + r["shift"]
        ax1.plot(ts * 1e3, r["uu"], ".", ms=3, color=c, alpha=0.35)
        x = (r["uu"] - r["B"]) / full
        pos = x > 0.003
        ax2.semilogy(ts[pos] * 1e3, x[pos], ".", ms=3, color=c, alpha=0.35)
    for g in ("nab", "vyb"):
        if groups[g]:
            tau = np.mean([r["tau"] for r in groups[g]])
            full = 5.0 if g == "vyb" else -5.0
            B = np.mean([r["B"] for r in groups[g]])
            xx = np.linspace(0, 8 * tau, 300)
            ax1.plot(xx * 1e3, model(xx, full, B, tau), "-", lw=1.5, color=COL[g],
                     label=f"{LAB[g]}: τ = {tau * 1e3:.1f} ms ({len(groups[g])}×)")
            ax2.semilogy(xx * 1e3, np.exp(-xx / tau), "-", lw=1.5, color=COL[g])
    ax2.axhline(np.exp(-1), color="#999999", lw=0.8, ls="--")
    ax2.text(0, np.exp(-1) * 1.08, " e⁻¹ = 36,8 %", color="#666666", fontsize=8)
    ax1.set_xlabel("t od přepnutí [ms] (hrany zarovnány na skok 5 V)")
    ax1.set_ylabel("U_C [V]")
    ax1.set_title("všechny hrany přes sebe + fit")
    ax1.legend(frameon=False, loc="center right")
    ax2.set_xlabel("t od přepnutí [ms]")
    ax2.set_ylabel("|U − U_konec| / 5 V")
    ax2.set_title("zbývající podíl skoku (semilog → přímka)")
    for ax in (ax0, ax1, ax2):
        ax.grid(True, color="#dddddd", lw=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    png = os.path.splitext(path)[0] + ".png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print("graph ->", os.path.abspath(png), "\n")

# ---- per-file summary for ukol3_tau_R.py ----
for sm in summary:
    path = next(p for p in a.files if os.path.splitext(os.path.basename(p))[0] == sm["name"])
    with open(os.path.splitext(path)[0] + ".json", "w") as f:
        json.dump(sm, f, indent=1)
