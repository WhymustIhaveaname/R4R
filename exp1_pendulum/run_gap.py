"""Trajectory error of measured, analytically renormalized, and identified parameters.

At each physics time step h, the same pendulum is simulated with
  measured      (J, K_P, K_D) copied from the physical rod,
  renormalized  the first-order prediction for that scheme, with no fitting at all,
  identified    the best fit to the real trajectory,
and compared against the continuous-time ground truth.

One parameter set has to serve five drive frequencies at once, so identification cannot
specialize to a single command. The control frequency is fixed at 50 Hz and only the decimation
varies, so every run sees the same commands and the same real trajectories; h = 1 / (50 * dec).
A fine grid of small h is also run with measured parameters only, so that the step size at
which measured parameters reach the renormalized accuracy is measured rather than extrapolated.
"""

import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from common import (ARMS, DRIVES, PARAM0, Sim, first_order_prediction,
                    first_order_prediction_delay, ground_truth, identify, pooled_rmse,
                    ref_sequence, run_many)

F_CTRL = 50  # Hz, the legged_gym value; held fixed so the command never changes
# h = 1 / (50 * decimation): 20 ms down to 0.2 ms, i.e. h K_D / J from 2.85 down to 0.03.
DECIMATION = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50, 60, 80, 100)
DECIMATION_FINE = (125, 160, 200, 250, 320, 400, 500)  # measured parameters only, up to 25 kHz
F_FIT_MAX = 1000  # Hz; the figure shows f up to here, so only these steps are fitted
OUT = Path(__file__).parent / "results" / "gap_mujoco.json"

# Another engine only has to supply a simulator with the same rollout signature (see
# common.run_many for the batched forms it may also offer); the sweep itself is shared.
make_sim = Sim
# RK4 holds the torque over the step, so it shifts no command and gets no delay to fit.
WITH_DELAY = {"A": True, "B": False, "C": True}
# Called with the rows so far after every grid point, so a run that is interrupted still
# leaves behind everything it had finished.
on_row = lambda rows, fine: None


def batch_over(sim, data, dec):
    """Each parameter row's trajectories over every command, all from one batched rollout."""
    def run(rows):
        got = run_many(sim, [(refs, F_CTRL, dec, p) for p in rows for refs, _ in data])
        per = [got[k * len(data):(k + 1) * len(data)] for k in range(len(rows))]
        return [None if any(t is None for t in ts) else np.concatenate(ts) for ts in per]
    return run


def renormalized(arm, h):
    """First-order parameters for this scheme, or None where they are unphysical."""
    p = first_order_prediction("B", h) if arm == "B" else first_order_prediction_delay(arm, h)
    return None if p[0] <= 0 else p  # arm C drives the effective inertia negative at h > 2J/K_D


def datasets(dec):
    """One (command, ground truth) pair per drive frequency, all at the same resolution."""
    out = []
    for f_ref in DRIVES:
        refs = ref_sequence(F_CTRL, f_ref)
        out.append((refs, ground_truth(refs, F_CTRL, dec)))
    return out


def run_arm(arm, done=None):
    """`done` is a {"main", "fine"} of rows already computed, whose grid points are skipped."""
    sim = make_sim(arm)
    with_delay = WITH_DELAY[arm]
    rows = list((done or {}).get("main", []))
    fitted = [r for r in rows if r.get("params_identified") is not None]
    prev = np.array(min(fitted, key=lambda r: r["dec"])["params_identified"]) if fitted else None
    for dec in sorted(DECIMATION, reverse=True):  # continuation from fine to coarse steps
        if any(r["dec"] == dec for r in rows):
            continue
        h = 1.0 / (F_CTRL * dec)
        data = datasets(dec)
        t0 = time.perf_counter()
        ren = renormalized(arm, h)
        starts = [np.append(PARAM0, 0.0) if with_delay else PARAM0]
        if ren is not None:
            starts.append(ren)
        if prev is not None:
            starts.append(prev)
        gt = np.concatenate([g for _, g in data])

        def rollout(J, kp, kd, delay=0.0, _data=data):
            out = [sim.rollout(refs, F_CTRL, dec, J, kp, kd, delay) for refs, _ in _data]
            return None if any(t is None for t in out) else np.concatenate(out)

        fit = (identify(rollout, gt, h, starts, with_delay=with_delay,
                        batch=batch_over(sim, data, dec))
               if 1.0 / h <= F_FIT_MAX + 1e-6 else {"rmse": None, "params": None})
        prev = np.array(fit["params"]) if fit["params"] is not None else prev
        rows.append({
            "arm": arm, "h": h, "f": 1.0 / h, "f_ctrl": F_CTRL, "dec": dec,
            "rmse_measured": pooled_rmse(sim, data, F_CTRL, dec, PARAM0),
            "rmse_renormalized": pooled_rmse(sim, data, F_CTRL, dec, ren) if ren is not None else None,
            "rmse_identified": fit["rmse"],
            "params_renormalized": list(ren) if ren is not None else None,
            "params_identified": fit["params"],
            "seconds": time.perf_counter() - t0,
        })
        r = rows[-1]
        print(f"{arm} dec={dec:<4d} h={h * 1e3:6.3f}ms: measured {r['rmse_measured']} "
              f"renorm {r['rmse_renormalized']} identified {r['rmse_identified']} "
              f"({r['seconds']:.1f}s)", flush=True)
        on_row(rows, [])

    fine = list((done or {}).get("fine", []))
    for dec in DECIMATION_FINE:
        if any(r["dec"] == dec for r in fine):
            continue
        h = 1.0 / (F_CTRL * dec)
        fine.append({"arm": arm, "h": h, "f": 1.0 / h, "f_ctrl": F_CTRL, "dec": dec,
                     "rmse_measured": pooled_rmse(sim, datasets(dec), F_CTRL, dec, PARAM0)})
        print(f"{arm} fine dec={dec}: measured {fine[-1]['rmse_measured']:.2e}", flush=True)
        on_row(rows, fine)
    return rows, fine


if __name__ == "__main__":
    with Pool(len(ARMS)) as pool:
        out = pool.map(run_arm, ARMS)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"main": [r for rows, _ in out for r in rows],
                               "fine": [r for _, fine in out for r in fine]}, indent=1, default=float))
    print(f"wrote {OUT}")
