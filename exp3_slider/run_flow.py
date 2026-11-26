"""Effective parameters of each scheme against the physics step, in MuJoCo and Genesis.

The grid and the fitter are experiment 1's: target h = 0.25, 0.5, ..., 5 ms, each realized with
the integer decimation closest to 1 / (f_ctrl_nominal h) and the integer control rate closest
to 1 / (decimation h), for five nominal control rates, and fitted by ../fitting.py, each chain
starting every fit from its previous answer.  Only the parameters the theory gives a
counterterm are fitted (common.FITTED): the rotor inertia J, and the command delay d and the
factor lam on the contact's stiffness, damping and regularization for Euler and implicitfast,
K_D for RK4; the others are held at their measured values.  Both coordinates are in the
residual, the block's as the angle x / L it subtends at the hinge.

One (engine, scheme, nominal control rate) chain per process; each saves after every grid point
and resumes from what is on disk.  Genesis has no RK4.
Usage:  uv run python run_flow.py --engine mujoco --arm rk4_explicit --fctrl 50
        uv run --extra genesis python run_flow.py --engine genesis --arm implicitfast_native --fctrl 50
        uv run python run_flow.py --merge
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common as c  # noqa: E402
import fitting  # noqa: E402

ENGINES = ("mujoco", "genesis")
F_CTRL = (10, 20, 50, 100, 200)  # nominal control rates (Hz), as in experiment 1
H_TARGET = np.arange(1, 21) * 0.25e-3
NOMINAL = np.array([c.J0, c.KP0, c.KD0, 0.0, 1.0])
RESULTS = Path(__file__).parent / "results"


def schedule(f_nominal, h_target):
    f = 1.0 / h_target
    dec = max(1, round(f / f_nominal))
    return round(f / dec), dec


def start_points(arm, h, prev, rng):
    """The nominal parameters, the first-order prediction, inflated inertias, which keep the
    explicit PD stable at the coarse steps, two random restarts, and the chain's last answer,
    each cut to the parameters the scheme carries."""
    n = max(c.FITTED[arm]) + 1
    pred = c.first_order_prediction(arm, h)
    starts = [NOMINAL, pred]
    starts += [np.append(c.J0 + k * h * c.KD0, pred[1:]) for k in (1, 2, 4)]
    for _ in range(2):
        p = pred.copy()
        p[0] = c.J0 * np.exp(rng.normal(0, 0.5))
        if 2 in c.FITTED[arm]:
            p[2] = c.KD0 * np.exp(rng.normal(0, 0.5))
        if 3 in c.FITTED[arm]:
            p[3] = rng.uniform(0, h)
        if 4 in c.FITTED[arm]:
            p[4] = pred[4] * np.exp(rng.normal(0, 0.3))
        starts.append(p)
    starts = [np.asarray(s, float)[:n] for s in starts]
    return starts + ([np.asarray(prev, float)] if prev is not None else [])


def make_sim(engine, arm):
    if engine == "genesis":
        from genesis_sim import GenesisSlider
        return GenesisSlider(arm, fitting.POPSIZE)  # one CMA-ES generation per rollout
    return c.Sim(arm)


def chain(engine, arm, f_nominal):
    path = RESULTS / "flow_parts" / f"flow_{engine}_{arm}_{f_nominal:03d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(path.read_text()) if path.exists() else []
    sim = make_sim(engine, arm)
    # the same starts in both engines; Euler's, at index 0, are the ones it always had
    rng = np.random.default_rng(1000 * c.ARMS.index(arm) + f_nominal)
    n = max(c.FITTED[arm]) + 1
    fixed = tuple(k for k in range(n) if k not in c.FITTED[arm])
    prev = None
    for h_target in H_TARGET:
        f_ctrl, dec = schedule(f_nominal, h_target)
        h = 1.0 / (f_ctrl * dec)
        starts = start_points(arm, h, prev, rng)  # drawn even when the point is done
        done = [r for r in rows if r["f_ctrl"] == f_ctrl and r["dec"] == dec]
        if done:
            prev = np.array(done[0]["params"])
            continue
        refs = c.ref_sequence(f_ctrl)
        gt = c.ground_truth(refs, f_ctrl, dec) * c.OBSERVE

        def run(params):
            out = sim.rollout_many([(refs, f_ctrl, dec, p) for p in params])
            return [None if t is None else t * c.OBSERVE for t in out]

        t0 = time.perf_counter()
        fit = fitting.drive(fitting.fit_steps(gt, h, starts, c.PARAM0, with_delay=n > 3,
                                              fixed=fixed, with_contact=n > 4), run)
        measured = run([NOMINAL[:n]])[0]
        fit.update(engine=engine, arm=arm, f_ctrl_nominal=f_nominal, f_ctrl=f_ctrl,
                   dec=dec, h=h, f=1.0 / h, seconds=time.perf_counter() - t0,
                   rmse_measured=(float(np.sqrt(np.mean((measured - gt) ** 2)))
                                  if measured is not None else None))
        rows.append(fit)
        path.write_text(json.dumps(rows, indent=1, default=float))
        prev = np.array(fit["params"])
        # each fitted shift over its first-order prediction
        shift = (prev - NOMINAL[:n]) / (c.first_order_prediction(arm, h) - NOMINAL)[:n]
        ratios = "  ".join(f"{name} {shift[k]:.4f}" for k, name in
                           enumerate(("J", "K_P", "K_D", "d", "lam")) if k in c.FITTED[arm])
        print(f"{engine} {arm} {f_ctrl:3d}x{dec:<3d} h={h * 1e3:.3f} ms: shift/theory {ratios}  "
              f"rmse {fit['rmse']:.2e} (measured {fit['rmse_measured']:.2e})  "
              f"{fit['seconds']:.0f} s", flush=True)
    return rows


def merge():
    for engine in ENGINES:
        parts = sorted((RESULTS / "flow_parts").glob(f"flow_{engine}_*.json"))
        rows = [r for p in parts for r in json.loads(p.read_text())]
        out = RESULTS / f"flow_{engine}.json"
        out.write_text(json.dumps(rows, indent=1, default=float))
        print(f"merged {len(rows)} fits from {len(parts)} chains into {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=ENGINES, default="mujoco")
    ap.add_argument("--arm", choices=c.ARMS, default="euler_explicit")
    ap.add_argument("--fctrl", type=int, choices=F_CTRL, default=50)
    ap.add_argument("--merge", action="store_true", help="concatenate the per-chain files")
    a = ap.parse_args()
    merge() if a.merge else chain(a.engine, a.arm, a.fctrl)
