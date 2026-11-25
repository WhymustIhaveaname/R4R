"""Experiment 1 swept on Genesis, using exp1_pendulum/run_flow.py unchanged.

The sweep, the grid, the starting points and the fitter all come from the MuJoCo code; this
supplies the simulator and each scheme's first-order prediction, which is MuJoCo's for the same
step map and seeds the fit.  Every scheme fits a command delay, as MuJoCo's Euler and
implicitfast do.

One (scheme, nominal control frequency) chain per process: a scene has one physics step, and
a step of 32 environments costs 0.05 ms on one CPU core, so the chains run side by side on as
many cores.  Each saves after every grid point and resumes from what is on disk.
Usage:  uv run --extra genesis python run_single.py --arm euler_explicit --fctrl 50
        uv run --extra genesis python run_single.py --arm euler_explicit --cross-seed
        uv run --extra genesis python run_single.py --merge
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pendulum_genesis as pg  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--arm", choices=pg.ARMS, default=pg.ARMS[0])
ap.add_argument("--fctrl", type=int, default=50)
ap.add_argument("--cross-seed", action="store_true",
                help="once all five chains of the scheme are done: refit from each other's answers")
ap.add_argument("--merge", action="store_true", help="concatenate the per-chain files")
args = ap.parse_args()

RESULTS = pg.ROOT / "exp1_pendulum" / "results"
PARTS = RESULTS / "genesis_parts"
OUT = RESULTS / "flow_genesis.json"


def merge():
    parts = [p for arm in pg.ARMS for p in sorted(PARTS.glob(f"flow_genesis_{arm}_*.json"))]
    rows = [r for f in parts for r in json.loads(f.read_text())]
    OUT.write_text(json.dumps(rows, indent=1, default=float))
    print(f"merged {len(rows)} fits from {len(parts)} chains into {OUT}")


def main():
    if args.merge:
        return merge()
    import fitting
    import run_flow as rf

    rf.ARMS = pg.ARMS
    rf.WITH_DELAY = True
    rf.first_order_prediction_delay = pg.first_order_prediction_delay
    rf.make_sim = lambda arm: pg.GenesisPendulum(arm, fitting.POPSIZE)  # one CMA-ES generation
    PARTS.mkdir(parents=True, exist_ok=True)
    save = lambda path, rows: path.write_text(json.dumps(rows, indent=1, default=float))
    t0 = time.perf_counter()
    if args.cross_seed:
        files = {f: json.loads(f.read_text())
                 for f in sorted(PARTS.glob(f"flow_genesis_{args.arm}_*.json"))}
        rows = sorted((r for rs in files.values() for r in rs), key=lambda r: r["h"])
        moved = rf.cross_seed(rows, on_row=lambda: [save(f, rs) for f, rs in files.items()])
        print(f"{args.arm}: {moved} of {len(rows)} fits moved in "
              f"{(time.perf_counter() - t0) / 60:.0f} min", flush=True)
    else:
        part = PARTS / f"flow_genesis_{args.arm}_{args.fctrl:03d}.json"
        rf.on_row = lambda rows: save(part, rows)
        rows = rf.chain((args.arm, args.fctrl),
                        json.loads(part.read_text()) if part.exists() else None)
        print(f"wrote {len(rows)} fits in {(time.perf_counter() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
