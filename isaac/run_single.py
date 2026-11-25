"""Experiment 1 in Isaac Sim, using exp1_pendulum/run_flow.py and run_gap.py unchanged.

The sweeps, the grids, the commands, the reference solution, the starting points and the fitter
all come from the MuJoCo code; this supplies a simulator with the same rollout signature and
each PhysX scheme's own first-order prediction, which seeds the fits.  Every scheme fits a
command delay, since every one of them shifts the command.

One process per scheme, because an Isaac app holds 4.6 GB however few environments it steps:
it runs the five flow chains side by side, where every chain at the same physics step shares
one rollout, then cross-seeds them, and with --gap the behaviour sweep, whose five commands
also share one rollout.  Each stage saves after every grid point and resumes from what is on
disk.

Usage (inside the Isaac venv):
    python run_single.py --arm A [--gap]
    python run_single.py --merge
"""

import argparse
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--arm", default="A", choices=("A", "B", "C"))
parser.add_argument("--gap", action="store_true", help="also run the behaviour sweep")
parser.add_argument("--merge", action="store_true")
args = parser.parse_args()

ROOT = Path(__file__).resolve().parent.parent  # the repo, wherever it is checked out
RESULTS = ROOT / "exp1_pendulum" / "results"
PARTS = RESULTS / "isaac_parts"
OUT = RESULTS / "flow_isaac.json"
N_ENVS = 5 * 32   # five chains or five commands, a CMA-ES generation of 32 each


def merge():
    rows = [r for f in sorted(PARTS.glob("flow_isaac_[ABC]_*.json"))
            for r in json.loads(f.read_text())]
    OUT.write_text(json.dumps(rows, indent=1, default=float))
    print(f"merged {len(rows)} fits into {OUT}")


def main():
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    from isaacsim.core.api import World

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "isaac"))
    sys.path.insert(0, str(ROOT / "exp1_pendulum"))
    import run_flow as rf
    import run_gap
    from pendulum_isaac import IsaacSim, first_order_prediction

    world = World(physics_dt=1 / 200, stage_units_in_meters=1.0, backend="numpy")
    sim = IsaacSim(world, args.arm, N_ENVS)
    save = lambda path, data: path.write_text(json.dumps(data, indent=1, default=float))
    PARTS.mkdir(parents=True, exist_ok=True)
    part = lambda f: PARTS / f"flow_isaac_{args.arm}_{f:03d}.json"
    load = lambda path, empty: json.loads(path.read_text()) if path.exists() else empty

    rf.WITH_DELAY = True
    rf.first_order_prediction_delay = first_order_prediction
    rf.make_sim = lambda arm: sim
    rf.on_row = lambda rows: save(part(rows[0]["f_ctrl_nominal"]), rows)
    rows = rf.chains(args.arm, rf.F_CTRL, {f: load(part(f), []) for f in rf.F_CTRL})

    files = {part(f): rs for f, rs in rows.items()}
    every = sorted((r for rs in files.values() for r in rs), key=lambda r: r["h"])
    moved = rf.cross_seed(every, on_row=lambda: [save(f, rs) for f, rs in files.items()])
    print(f"{args.arm}: {moved} of {len(every)} fits moved", flush=True)

    if not args.gap:
        return app.close()
    gap = RESULTS / f"gap_isaac_{args.arm}.json"
    run_gap.make_sim = lambda arm: sim
    run_gap.WITH_DELAY = {arm: True for arm in "ABC"}
    run_gap.renormalized = lambda arm, h: (p if (p := first_order_prediction(arm, h))[0] > 0
                                           else None)
    run_gap.on_row = lambda rows, fine: save(gap, {"main": rows, "fine": fine})
    run_gap.run_arm(args.arm, load(gap, None))
    app.close()


if __name__ == "__main__":
    merge() if args.merge else main()
