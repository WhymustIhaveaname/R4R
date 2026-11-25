"""Identify (J, K_P, K_D) for every arm on a nearly uniform grid of physics time steps h.

For each target h and nominal control frequency, the integer decimation closest to
1 / (f_ctrl_nominal * h) is used with the integer f_ctrl (Hz) closest to 1 / (decimation * h).
With --delay, also identify a command delay d (results/flow_mujoco_delay.json).
With --cross-seed, refit every point of the file from the other control rates' answers
(fitting.cross_seed), so that what scatters by control rate is the simulator's doing.
"""

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from common import (ARMS, PARAM0, Sim, first_order_prediction, first_order_prediction_delay,
                    ground_truth, ref_sequence, run_many, start_points)
import fitting  # noqa: E402  (common.py puts the repo root on the path)

F_CTRL = (10, 20, 50, 100, 200)  # nominal control frequencies (Hz)
H_TARGET = np.arange(1, 21) * 0.25e-3  # target physics time step h = 0.25, 0.5, ..., 5 ms


def schedule(f_nominal, h_target):
    """Integer (f_ctrl, decimation) whose physics step 1 / (f_ctrl * decimation) is closest to h_target."""
    f = 1.0 / h_target
    dec = max(1, round(f / f_nominal))
    return round(f / dec), dec
WITH_DELAY = "--delay" in sys.argv
OUT = Path(__file__).parent / "results" / ("flow_mujoco_delay.json" if WITH_DELAY else "flow_mujoco.json")

# A second engine only has to supply a simulator with the same rollout signature; the sweep
# itself is shared.  A simulator that can roll several parameter sets out at once offers
# rollout_batch(refs, f_ctrl, dec, rows), and one that can also give each its own command,
# rollout_many(jobs) with jobs of (refs, f_ctrl, dec, params) at a common step.
make_sim = Sim
# Called with a chain's rows so far after every grid point, so a chain that is interrupted
# still leaves behind everything it had finished.
on_row = lambda rows: None


def chains(arm, f_nominals, done=None):
    """Continuation from small to large h, one chain per nominal control frequency.

    The chains advance together: at each target h, the fits that share a physics step run to
    the end side by side, every generation of all of them in one rollout, which is what lets a
    simulator that steps many copies run several chains for the price of one; then the next
    physics step, so that the simulator changes its step once per group, not once per
    generation.  The fits are independent and each has its own random stream, so the order
    changes nothing in them.  `done` maps a nominal frequency
    to rows already fitted; those grid points are skipped, but their random starting points are
    still drawn, so a resumed chain is the same chain.
    """
    sim = make_sim(arm)
    state = {}
    for f in f_nominals:
        rows = list((done or {}).get(f, []))
        state[f] = {"rows": rows, "prev": np.array(rows[-1]["params"]) if rows else None,
                    "rng": np.random.default_rng(1000 * ARMS.index(arm) + f)}
    for h_target in H_TARGET:
        tasks = []
        for f in f_nominals:
            st = state[f]
            f_ctrl, dec = schedule(f, h_target)
            h = 1.0 / (f_ctrl * dec)
            predicted = (first_order_prediction_delay(arm, h) if WITH_DELAY
                         else first_order_prediction("B", h))  # the paper's formula, dt = h/2
            starts = start_points(h, predicted, st["prev"], st["rng"], WITH_DELAY)
            if any(r["f_ctrl"] == f_ctrl and r["dec"] == dec for r in st["rows"]):
                continue
            refs = ref_sequence(f_ctrl)
            gt = ground_truth(refs, f_ctrl, dec)
            tasks.append({"f": f, "f_ctrl": f_ctrl, "dec": dec, "h": h, "refs": refs, "gt": gt,
                          "fit": fitting.fit_steps(gt, h, starts, PARAM0, with_delay=WITH_DELAY),
                          "t0": time.perf_counter()})
        for group in by_step(tasks):
            for t in group:
                t["ask"] = next(t["fit"])
            active = list(group)
            while active:
                jobs = [(t["refs"], t["f_ctrl"], t["dec"], p) for t in active for p in t["ask"]]
                got, i = run_many(sim, jobs), 0
                for t in list(active):
                    answer, i = got[i:i + len(t["ask"])], i + len(t["ask"])
                    try:
                        t["ask"] = t["fit"].send(answer)
                    except StopIteration as result:
                        t["result"] = result.value
                        active.remove(t)
        for group in by_step(tasks):
            measured = run_many(sim, [(t["refs"], t["f_ctrl"], t["dec"], PARAM0) for t in group])
            for t, meas in zip(group, measured):
                t["result"]["rmse_measured"] = (float(np.sqrt(np.mean((meas - t["gt"]) ** 2)))
                                                if meas is not None else None)
        for t in tasks:
            st, fit = state[t["f"]], t["result"]
            fit.update(arm=arm, f_ctrl_nominal=t["f"], f_ctrl=t["f_ctrl"], dec=t["dec"],
                       h=t["h"], f=1.0 / t["h"], seconds=time.perf_counter() - t["t0"])
            st["rows"].append(fit)
            on_row(st["rows"])
            st["prev"] = np.array(fit["params"])
            p = st["prev"][:3] / PARAM0
            d = f" d/h={st['prev'][3] / t['h']:.4f}" if WITH_DELAY else ""
            print(f"{arm} f_ctrl={t['f_ctrl']:3d}x{t['dec']:<3d} h={t['h'] * 1e3:.3f}ms: "
                  f"J*/J={p[0]:.4f} KP*/KP={p[1]:.4f} KD*/KD={p[2]:.4f}{d} "
                  f"rmse={fit['rmse']:.2e} ({fit['seconds']:.1f}s)", flush=True)
    return {f: state[f]["rows"] for f in f_nominals}


def by_step(tasks):
    """The tasks grouped by physics step, so each group is one rollout."""
    groups = {}
    for t in tasks:
        groups.setdefault(t["f_ctrl"] * t["dec"], []).append(t)
    return list(groups.values())


def chain(args, done=None):
    """One chain: chains() for a single nominal control frequency."""
    arm, f_nominal = args
    return chains(arm, [f_nominal], {f_nominal: done or []})[f_nominal]


def cross_seed(rows, on_row=lambda: None):
    """fitting.cross_seed over one scheme's chains, on this module's simulator."""
    sim = make_sim(rows[0]["arm"])

    def problem(r):
        refs = ref_sequence(r["f_ctrl"])
        return (ground_truth(refs, r["f_ctrl"], r["dec"]),
                lambda J, kp, kd, delay=0.0: sim.rollout(refs, r["f_ctrl"], r["dec"], J, kp, kd,
                                                         delay),
                lambda rows: run_many(sim, [(refs, r["f_ctrl"], r["dec"], p) for p in rows]))
    return fitting.cross_seed(rows, problem, PARAM0, with_delay=WITH_DELAY, on_row=on_row)


def cross_seed_arm(rows):
    print(f"{rows[0]['arm']}: {cross_seed(rows)} of {len(rows)} fits moved", flush=True)
    return rows


if __name__ == "__main__" and "--cross-seed" in sys.argv:
    rows = json.loads(OUT.read_text())
    with Pool(len(ARMS)) as pool:
        rows = [r for rs in pool.map(cross_seed_arm, [[r for r in rows if r["arm"] == a]
                                                      for a in ARMS]) for r in rs]
    OUT.write_text(json.dumps(rows, indent=1, default=float))
elif __name__ == "__main__":
    jobs = [(arm, fc) for arm in ARMS for fc in F_CTRL]
    with Pool(len(jobs)) as pool:
        rows = [r for rs in pool.map(chain, jobs) for r in rs]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=1, default=float))
    print(f"wrote {len(rows)} fits to {OUT}")
