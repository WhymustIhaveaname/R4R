"""Isaac version of exp1_pendulum/run_gap.py: the error of measured, renormalized and fitted parameters.

The grid and the fitter are MuJoCo's: control at 50 Hz, the same decimations, one parameter set
fitted to all five drive frequencies at once, and a fine grid run with measured parameters only.
As in the Isaac flow, the error is read once per control step, over 1 s of each command.

Run (inside the Isaac venv):  python run_gap_single.py A
Writes exp1_pendulum/results/isaac_gap_<arm>.json after every grid point, and run again it
only fills in the points that file lacks.
"""

import json
import sys
import time
from pathlib import Path

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent  # the repo, wherever it is checked out
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "isaac"))
sys.path.insert(0, str(ROOT / "exp1_pendulum"))

import common as mj  # noqa: E402
import run_gap as mujoco_gap  # noqa: E402  (the grid)
from pendulum_isaac import IsaacPendulum, first_order_prediction, fit_problem  # noqa: E402

ARM = sys.argv[1] if len(sys.argv) > 1 else "A"
T_FIT = 1.0  # s of each command, as in run_single.py
F_CTRL = mujoco_gap.F_CTRL
OUT = ROOT / "exp1_pendulum" / "results" / f"isaac_gap_{ARM}.json"
MEASURED = np.append(mj.PARAM0, 0.0)


def main():
    from isaacsim.core.api import World

    world = World(physics_dt=1 / 200, stage_units_in_meters=1.0, backend="numpy")
    sim = IsaacPendulum(world, ARM, 3 * 2 * 4)   # three starts, each probing 2 x 4 parameters
    out = json.loads(OUT.read_text()) if OUT.exists() else {"main": [], "fine": []}
    done = {r["dec"] for r in out["main"]}, {r["dec"] for r in out["fine"]}

    def problem(dec):
        sim.set_dt(1.0 / (F_CTRL * dec))
        gt, rollout, batch = fit_problem(sim, F_CTRL, dec, mj.DRIVES, T_FIT)

        def rmse(p):
            """None where the rollout blew up, as MuJoCo's pooled_rmse reports it."""
            if p is None:
                return None
            theta = rollout(*p)
            ok = np.all(np.isfinite(theta)) and np.max(np.abs(theta)) < 1e3
            return float(np.sqrt(np.mean((theta - gt) ** 2))) if ok else None
        return gt, rollout, batch, rmse

    prev = (np.array(min(out["main"], key=lambda r: r["dec"])["params_identified"])
            if out["main"] else None)
    for dec in sorted(mujoco_gap.DECIMATION, reverse=True):  # from fine to coarse steps
        if dec in done[0]:
            continue
        h = 1.0 / (F_CTRL * dec)
        gt, rollout, batch, rmse = problem(dec)
        ren = first_order_prediction(ARM, h)
        ren = ren if ren[0] > 0 else None   # the inertia cannot be negative
        starts = [MEASURED] + ([ren] if ren is not None else []) + ([prev] if prev is not None else [])
        t0 = time.perf_counter()
        fit = mj.identify(rollout, gt, h, starts, with_delay=True, batch=batch)
        prev = np.array(fit["params"])
        out["main"].append({
            "arm": ARM, "h": h, "f": 1.0 / h, "f_ctrl": F_CTRL, "dec": dec,
            "rmse_measured": rmse(MEASURED), "rmse_renormalized": rmse(ren),
            "rmse_identified": fit["rmse"],
            "params_renormalized": None if ren is None else ren.tolist(),
            "params_identified": fit["params"], "seconds": time.perf_counter() - t0})
        r = out["main"][-1]
        print(f"{ARM} dec={dec:<4d} h={h * 1e3:6.3f}ms: measured {r['rmse_measured']} "
              f"renorm {r['rmse_renormalized']} identified {r['rmse_identified']:.2e} "
              f"({r['seconds']:.0f}s)", flush=True)
        OUT.write_text(json.dumps(out, indent=1, default=float))

    for dec in mujoco_gap.DECIMATION_FINE:
        if dec in done[1]:
            continue
        h = 1.0 / (F_CTRL * dec)
        out["fine"].append({"arm": ARM, "h": h, "f": 1.0 / h, "f_ctrl": F_CTRL, "dec": dec,
                            "rmse_measured": problem(dec)[3](MEASURED)})
        print(f"{ARM} fine dec={dec}: measured {out['fine'][-1]['rmse_measured']}", flush=True)
        OUT.write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {OUT}", flush=True)


main()
app.close()
