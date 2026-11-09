#!/usr/bin/env python3
"""Task 4: fit tau from buffered RC step captures (outputs of ukol4_rc.py).

  python3 ukol4_analyza.py data/ukol4_rc_buffer/rc_10k_rep1.csv [...]

Per capture (phase R = charging, F = discharging):
  t_i = i * DT * k, step at i = PRE (known exactly, D7 is switched by the program)
  fit U = B + A exp(-(t - t_PRE) / tau) on samples i >= PRE (A, B, tau free; ADC codes)
  k = clock factor PC seconds / Arduino seconds from the '# clock' lines
Quality checks per capture: overruns = 0, residual rms <= 1 LSB, flat baseline
(sd of the PRE samples <= 1 LSB), window after the step >= 5 tau.
Uncertainty of tau per file (components added in quadrature):
  rep    sd of the captures / sqrt(n)
  stat   rms of the fit errors / sqrt(n)
  meth   half-differences: window end (whole window vs 4 tau) and fit start
         (from the step vs from 80 % of the step remaining), in quadrature
  dir    |tau_charge - tau_discharge| / 2
  clock  error of the clock factor k (fit of the clock stamps), floor 1e-4
         (PC stamps from CLOCK_MONOTONIC_RAW; older files: --clock k from raw runs)
Writes <file>.png and <file>.json next to the CSV.
"""
import argparse, json, os, re
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--clock", type=float, help="use this clock factor k instead of the file's "
                "own stamps (files recorded with the NTP-slewed PC clock time.time())")
ap.add_argument("--clock-err", type=float, default=1e-4)
a = ap.parse_args()


def load(fn):
    meta, clk, caps, rows = {}, [], {}, []
    for ln in open(fn):
        ln = ln.strip()
        if ln.startswith("#"):
            for k, v in re.findall(r"(\w+)=(\S+)", ln):
                meta[k] = v
            m = re.match(r"# clock us=(\d+) pc=([\d.]+)", ln)
            if m:
                clk.append((int(m[1]) * 1e-6, float(m[2])))
            m = re.match(r"# capture (\w)(\d+) overruns=(\d+) max_late_us=(\d+)", ln)
            if m:
                caps[(m[1], int(m[2]))] = dict(overruns=int(m[3]), late=int(m[4]))
        elif ln and ln[0] in "RF":
            p, r, i, v = ln.split(",")
            rows.append((p, int(r), int(i), int(v)))
    return meta, np.array(clk), caps, rows


def clock_factor(clk):
    """PC time vs Arduino micros(): slope = true seconds per Arduino second."""
    us, pc = clk[:, 0], clk[:, 1]
    (k, q), cov = np.polyfit(us, pc, 1, cov=True)
    span = us[-1] - us[0]
    ek = max(np.sqrt(cov[0, 0]), 1e-4)
    return k, ek, span


def model(t, A, B, tau):
    return B + A * np.exp(-t / tau)


def fit(t, u, tau0):
    p, cov = curve_fit(model, t, u, p0=(u[0] - u[-1], u[-1], tau0))
    return p, np.sqrt(np.diag(cov))


for fn in a.files:
    meta, clk, caps, rows = load(fn)
    name = os.path.splitext(os.path.basename(fn))[0]
    R = float(meta["R_ohm"])
    dt, pre = int(meta["dt_us"]) * 1e-6, int(meta["pre"])
    k, ek, span = clock_factor(clk)
    raw = meta.get("pc_clock") == "monotonic_raw"
    if a.clock:
        print(f"(own stamps {'raw' if raw else 'NTP-slewed'}: k = {k:.6f}; overridden)")
        k, ek, src = a.clock, a.clock_err, "zadáno (--clock)"
    else:
        src = "vlastní razítka, " + ("surové hodiny PC" if raw else "POZOR: hodiny PC s NTP")
    print(f"\n=== {name}: R = {R / 1e3:g}k, DT = {dt * 1e6:.0f} us, "
          f"clock factor k = {k:.6f} ± {ek:.6f} (over {span:.0f} s; Arduino "
          f"{'slow' if k > 1 else 'fast'} by {abs(k - 1):.3%}; {src})")

    res = []
    for (ph, rep), info in sorted(caps.items(), key=lambda x: (x[0][1], x[0][0] != "R")):
        adc = np.array([v for p, r, i, v in rows if p == ph and r == rep], float)
        t_all = np.arange(len(adc)) * dt * k
        base = adc[:pre]
        t, u = t_all[pre:] - t_all[pre], adc[pre:]
        tau0 = 3.5e-6 * R
        (A, B, tau), (eA, eB, etau) = fit(t, u, tau0)
        r = u - model(t, A, B, tau)
        # method variants
        m4 = t <= 4 * tau
        tau_w = fit(t[m4], u[m4], tau)[0][2]
        s80 = np.argmax(np.abs(u - B) <= 0.8 * abs(A))
        tau_s = fit(t[s80:] - t[s80], u[s80:], tau)[0][2]
        win = t[-1] / tau
        pts = np.sum(t <= tau)
        why = []
        if info["overruns"]:
            why.append(f"overruns={info['overruns']}")
        if np.std(r) > 1.0:
            why.append(f"rms {np.std(r):.2f} LSB > 1")
        if np.std(base) > 1.0:
            why.append(f"baseline sd {np.std(base):.2f} LSB")
        if win < 5:
            why.append(f"window {win:.1f} tau < 5")
        res.append(dict(ph=ph, rep=rep, tau=tau, etau=etau, A=A, B=B, rms=np.std(r),
                        base_sd=np.std(base), base=np.mean(base), win=win, pts=pts,
                        dw=tau_w - tau, ds=tau_s - tau, late=info["late"],
                        ok=not why, why=", ".join(why), t=t, u=u, r=r, t_all=t_all,
                        adc=adc))

    print("| hrana | τ [ms] | fit ± [ms] | rms rezidua [LSB] | sd základny [LSB] | "
          "okno [τ] | bodů na τ | max zpoždění [µs] | OK? |")
    print("|---|---|---|---|---|---|---|---|---|")
    for e in res:
        lab = ("nab" if e["ph"] == "R" else "vyb") + str(e["rep"] + 1)
        print(f"| {lab} | {e['tau'] * 1e3:.3f} | {e['etau'] * 1e3:.3f} | {e['rms']:.2f} | "
              f"{e['base_sd']:.2f} | {e['win']:.1f} | {e['pts']} | {e['late']} | "
              f"{'ano' if e['ok'] else 'NE: ' + e['why']} |")

    ok = [e for e in res if e["ok"]]
    T = np.array([e["tau"] for e in ok])
    n = len(T)
    tau = T.mean()
    tR = np.array([e["tau"] for e in ok if e["ph"] == "R"])
    tF = np.array([e["tau"] for e in ok if e["ph"] == "F"])
    comp = dict(
        rep=T.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0,
        stat=np.sqrt(np.mean([e["etau"] ** 2 for e in ok])) / np.sqrt(n),
        meth=np.hypot(np.mean([e["dw"] for e in ok]) / 2, np.mean([e["ds"] for e in ok]) / 2),
        dir=abs(tR.mean() - tF.mean()) / 2 if len(tR) and len(tF) else 0.0,
        clock=tau * ek / k)
    etau_tot = np.sqrt(sum(v ** 2 for v in comp.values()))
    print(f"\nτ = {tau * 1e3:.3f} ± {etau_tot * 1e3:.3f} ms ({etau_tot / tau:.2%}), "
          f"{n}/{len(res)} captures OK")
    print("components [ms]: " + ", ".join(f"{kk} {v * 1e3:.4f}" for kk, v in comp.items()))
    print(f"charging {tR.mean() * 1e3:.3f} ms (sd {tR.std(ddof=1) * 1e3 if len(tR) > 1 else 0:.3f}),"
          f" discharging {tF.mean() * 1e3:.3f} ms (sd {tF.std(ddof=1) * 1e3 if len(tF) > 1 else 0:.3f})"
          f", difference {(tR.mean() - tF.mean()) * 1e3:+.3f} ms")

    # ---- graph ----
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5))
    for e in res:
        c = "#c62828" if e["ph"] == "R" else "#1565c0"
        axs[0, 0].plot(e["t_all"] * 1e3, e["adc"], ".", ms=1.5, color=c)
        axs[1, 0].plot(e["t"] * 1e3, e["r"], ".", ms=1.5, color=c, alpha=0.6)
        y = np.abs(e["u"] - e["B"]) / abs(e["A"])
        m = y > 0.003
        axs[0, 1].semilogy(e["t"][m] / e["tau"], y[m], ".", ms=1.5, color=c)
    axs[0, 0].plot([], [], ".", color="#c62828", label="nabíjení (D7 → 5 V)")
    axs[0, 0].plot([], [], ".", color="#1565c0", label="vybíjení (D7 → 0 V)")
    axs[0, 0].axvline(pre * dt * k * 1e3, color="#999999", lw=0.8, ls="--")
    axs[0, 0].set(xlabel="t od začátku záznamu [ms]", ylabel="ADC [kód]",
                  title=f"{name}: {len(res)} průběhů, DT = {dt * 1e6:.0f} µs")
    axs[0, 0].legend(frameon=False, fontsize=8)
    xx = np.linspace(0, 6, 50)
    axs[0, 1].semilogy(xx, np.exp(-xx), "-", color="#333333", lw=1, label="e^(−t/τ)")
    axs[0, 1].set(xlabel="t / τ", ylabel="zbývající část skoku",
                  title="semilog: přímka = čistá exponenciála", xlim=(0, 7))
    axs[0, 1].legend(frameon=False, fontsize=8)
    axs[1, 0].axhline(0, color="#333333", lw=0.8)
    axs[1, 0].set(xlabel="t od kroku [ms]", ylabel="reziduum [LSB]",
                  title="rezidua fitu (±0,5 LSB = kvantování)")
    x = np.arange(len(res))
    axs[1, 1].errorbar(x, [e["tau"] * 1e3 for e in res], yerr=[e["etau"] * 1e3 for e in res],
                       fmt="o", ms=4, color="#333333", capsize=3)
    for i, e in enumerate(res):
        axs[1, 1].plot(i, e["tau"] * 1e3, "o", ms=4,
                       color="#c62828" if e["ph"] == "R" else "#1565c0")
    axs[1, 1].axhspan((tau - etau_tot) * 1e3, (tau + etau_tot) * 1e3, color="#999999",
                      alpha=0.2, label=f"τ = {tau * 1e3:.3f} ± {etau_tot * 1e3:.3f} ms")
    axs[1, 1].set_xticks(x, [("nab" if e["ph"] == "R" else "vyb") + str(e["rep"] + 1)
                             for e in res])
    axs[1, 1].set(ylabel="τ [ms]", title="τ jednotlivých průběhů")
    axs[1, 1].legend(frameon=False, fontsize=8)
    for ax in axs.flat:
        ax.grid(True, color="#dddddd", lw=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    png = os.path.splitext(fn)[0] + ".png"
    fig.savefig(png, dpi=150)
    plt.close(fig)

    ex = {}
    for e in ok:
        d = "nab" if e["ph"] == "R" else "vyb"
        if d not in ex:
            ex[d] = dict(t=e["t"].tolist(), u=e["u"].tolist(), A=e["A"], B=e["B"],
                         tau=e["tau"])
    js = dict(name=name, R=R, dt=dt, tau=tau, etau=etau_tot, n=n, n_found=len(res),
              comp=comp, tauR=tR.mean(), tauF=tF.mean(), clock=k, eclock=ek, clock_src=src,
              rms=float(np.mean([e["rms"] for e in ok])),
              pts_per_tau=float(np.mean([e["pts"] for e in ok])),
              caps=[dict(ph=e["ph"], rep=e["rep"], tau=e["tau"], etau=e["etau"],
                         A=e["A"], B=e["B"], base=e["base"],
                         rms=e["rms"], ok=e["ok"], why=e["why"]) for e in res],
              example=ex)
    json.dump(js, open(os.path.splitext(fn)[0] + ".json", "w"), indent=1)
    print("graph ->", os.path.abspath(png))
