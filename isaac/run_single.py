"""Isaac Sim version of experiment 1: effective parameters of a PD-controlled pendulum.

The grid is the one exp1_pendulum/run_flow.py uses, so the two engines are directly comparable:
for each target h and nominal control frequency, the integer decimation closest to
1 / (f_ctrl_nominal * h) with the integer f_ctrl (Hz) closest to 1 / (decimation * h).  Running
several control rates at the same h is what exposes whether a shift depends on h alone.

Usage (inside the Isaac venv):  python run_single.py --arm A
Writes results/isaac_single_grid_<arm>.json next to the MuJoCo results.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--arm", default="A", choices=("A", "B", "C"))
parser.add_argument("--t-fit", type=float, default=1.0, help="seconds per drive frequency")
parser.add_argument("--no-delay", action="store_true", help="fit only (armature, K_P, K_D)")
parser.add_argument("--h-step", type=float, default=0.25, help="ms between grid points")
parser.add_argument("--h-min", type=float, default=0.5,
                    help="ms; the smallest steps cost 1/h each and carry the least signal")
parser.add_argument("--points", type=int, default=20)
parser.add_argument("--resume", action="store_true",
                    help="keep the fits already in the output file and only fill the gaps")
parser.add_argument("--all-drives", action="store_true",
                    help="fit one parameter set to all five drive frequencies at once")
parser.add_argument("--fctrl", type=int, nargs="+", default=[10, 20, 50, 100, 200],
                    help="nominal control frequencies (Hz); the figure draws 50, 100 and 200")
args = parser.parse_args()
app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent  # the repo, wherever it is checked out
sys.path.insert(0, str(ROOT / "isaac"))
sys.path.insert(0, str(ROOT / "exp1_pendulum"))

from pendulum_isaac import (J0, KD0, KP0, IsaacPendulum, first_order_prediction,  # noqa: E402
                            fit_problem)
import common as mj  # noqa: E402  (the MuJoCo module supplies the shared reference solution)

F_CTRL_NOMINAL = args.fctrl  # MuJoCo's are 10, 20, 50, 100 and 200 Hz
H_TARGET = np.arange(1, args.points + 1) * args.h_step * 1e-3
H_TARGET = H_TARGET[H_TARGET >= args.h_min * 1e-3 - 1e-12]
WITH_DELAY = not args.no_delay
OUT = ROOT / "exp1_pendulum" / "results" / f"isaac_single_grid_{args.arm}.json"


def schedule(f_nominal, h_target):
    """Integer (f_ctrl, decimation) whose physics step is closest to h_target."""
    f = 1.0 / h_target
    dec = max(1, round(f / f_nominal))
    return round(f / dec), dec


def main():
    from isaacsim.core.api import World

    world = World(physics_dt=1 / 200, stage_units_in_meters=1.0, backend="numpy")
    ndim = 4 if WITH_DELAY else 3
    # The fitter advances all eight starting points at once, each asking for one environment
    # per central-difference probe, so a round of the fit is a single rollout.
    sim = IsaacPendulum(world, args.arm, 8 * 2 * ndim)

    rows = json.loads(OUT.read_text()) if args.resume and OUT.exists() else []
    done = {(r["f_ctrl_nominal"], round(r["h"], 12)) for r in rows}
    if done:
        print(f"resuming with {len(rows)} fits already on disk", flush=True)
    for f_nominal in F_CTRL_NOMINAL:
        rng = np.random.default_rng(1000 * "ABC".index(args.arm) + f_nominal)
        prev = None
        for h_target in H_TARGET:
            f_ctrl, dec = schedule(f_nominal, h_target)
            h = 1.0 / (f_ctrl * dec)
            if (f_nominal, round(h, 12)) in done:
                continue
            sim.set_dt(h)
            drives = mj.DRIVES if args.all_drives else (mj.F_REF,)
            gt, rollout, batch = fit_problem(sim, f_ctrl, dec, drives, args.t_fit)

            measured = np.append([J0, KP0, KD0], [0.0] * WITH_DELAY)
            predicted = first_order_prediction(args.arm, h, WITH_DELAY)
            starts = mj.start_points(h, predicted, prev, rng, WITH_DELAY)
            t0 = time.perf_counter()
            out = mj.identify(rollout, gt, h, starts, with_delay=WITH_DELAY, batch=batch)
            prev = np.array(out["params"])

            rmse = lambda p: (float(np.sqrt(np.mean((rollout(*p) - gt) ** 2))))
            rows.append({"arm": args.arm, "h": h, "f": 1.0 / h, "f_ctrl_nominal": f_nominal,
                         "f_ctrl": f_ctrl, "dec": dec,
                         "params_identified": out["params"], "rmse_identified": out["rmse"],
                         "params_renormalized": predicted.tolist(),
                         "rmse_measured": rmse(measured), "rmse_renormalized": rmse(predicted),
                         "seconds": time.perf_counter() - t0})
            r = rows[-1]
            print(f"{args.arm} f_ctrl={f_ctrl:3d}x{dec:<4d} h={h * 1e3:6.3f} ms: "
                  f"dJ/(h K_D/2) {(r['params_identified'][0] - J0) / (h / 2 * KD0):7.3f}  "
                  f"fit rmse {r['rmse_identified']:.3e} ({r['seconds']:.0f}s)", flush=True)
            OUT.write_text(json.dumps(rows, indent=1, default=float))
    print(f"wrote {len(rows)} fits to {OUT}")


main()
app.close()
