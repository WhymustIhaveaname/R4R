"""Experiment 1 swept on SuperDex Physics, using exp1_pendulum/run_flow.py unchanged.

The sweep, the grid, the starting points and the fitter all come from the MuJoCo code; this
supplies the simulator, a batched Jacobian, and each scheme's first-order prediction, which
seeds the fit.  Every scheme fits a command delay, since every one of them shifts the command.

Two solver settings:
  default  as shipped: FP32, and Mochi's Newton stops after 4 iterations or at its tolerance
  tight    FP64, with Newton iterated until its step stalls at round-off
and the three link models of pendulum_superdex.LINKS.  The rod cannot run backward Euler with
Mochi's own PD, whose inertia flows below the rod's own: Mochi's armature cannot be negative.

One (setting, link, arm, nominal control frequency) chain per process.
Usage:  uv run --extra superdex python run_single.py --setting default --link rod \
            --arm backward_euler_explicit --fctrl 50
        uv run --extra superdex python run_single.py --setting default --link rod \
            --arm backward_euler_explicit --cross-seed
        uv run --extra superdex python run_single.py --merge
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--setting", choices=("default", "tight"), default="tight")
ap.add_argument("--link", choices=("rod", "quarter", "hundredth"), default="rod")
# as in pendulum_superdex, which cannot be imported before the precision is set
ARMS = ("backward_euler_explicit", "backward_euler_native", "bdf2_native")
ap.add_argument("--arm", choices=ARMS, default=ARMS[0])
ap.add_argument("--fctrl", type=int, default=50)
ap.add_argument("--cross-seed", action="store_true",
                help="once all five chains of the arm are done: refit from each other's answers")
ap.add_argument("--merge", action="store_true", help="concatenate the per-chain files")
args = ap.parse_args()
# The precision is fixed when superdex is imported.
os.environ["SUPERDEX_PRECISION"] = "fp64" if args.setting == "tight" else "fp32"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "exp1_pendulum"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

RESULTS = ROOT / "exp1_pendulum" / "results"
PARTS = RESULTS / "superdex_parts"


def stem(setting, link):
    return f"flow_superdex_{setting}_{link}"


def merge():
    for setting in ("default", "tight"):
        for link in ("rod", "quarter", "hundredth"):
            parts = [p for arm in ARMS
                     for p in sorted(PARTS.glob(f"{stem(setting, link)}_{arm}_*.json"))]
            if not parts:
                continue
            rows = [r for f in parts for r in json.loads(f.read_text())]
            out = RESULTS / f"{stem(setting, link)}.json"
            out.write_text(json.dumps(rows, indent=1, default=float))
            print(f"merged {len(rows)} fits from {len(parts)} chains into {out}")


def main():
    if args.merge:
        return merge()
    import fitting
    import run_flow as rf
    import pendulum_superdex as sdx

    rf.ARMS = sdx.ARMS
    rf.WITH_DELAY = True
    rf.first_order_prediction_delay = sdx.first_order_prediction_delay
    # one copy per member of a CMA-ES generation
    rf.make_sim = lambda arm: sdx.SuperDexPendulum(arm, fitting.POPSIZE,
                                                   tight=args.setting == "tight",
                                                   link_inertia=sdx.LINKS[args.link])
    PARTS.mkdir(parents=True, exist_ok=True)
    save = lambda path, rows: path.write_text(json.dumps(
        [dict(r, setting=args.setting, link=args.link) for r in rows], indent=1, default=float))
    t0 = time.perf_counter()
    if args.cross_seed:
        files = {f: json.loads(f.read_text()) for f in sorted(
            PARTS.glob(f"{stem(args.setting, args.link)}_{args.arm}_*.json"))}
        rows = sorted((r for rs in files.values() for r in rs), key=lambda r: r["h"])
        moved = rf.cross_seed(rows, on_row=lambda: [save(f, rs) for f, rs in files.items()])
        print(f"{args.arm}: {moved} of {len(rows)} fits moved", flush=True)
    else:
        part = PARTS / f"{stem(args.setting, args.link)}_{args.arm}_{args.fctrl:03d}.json"
        rf.on_row = lambda rows: save(part, rows)
        rows = rf.chain((args.arm, args.fctrl),
                        json.loads(part.read_text()) if part.exists() else None)
        print(f"wrote {len(rows)} fits in {(time.perf_counter() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
