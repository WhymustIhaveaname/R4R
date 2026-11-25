"""Experiment 1's behaviour sweep in SuperDex, using exp1_pendulum/run_gap.py unchanged.

The grid, the five commands, the reference solution and the fitter all come from the MuJoCo
code; this supplies the simulator, a batched rollout, and each Mochi scheme's own first-order
prediction.  Every scheme fits a command delay, since every one of them shifts the command.

Usage:  uv run --extra superdex python run_gap_single.py --link rod --arm backward_euler_explicit
Writes exp1_pendulum/results/gap_superdex_<setting>_<link>_<arm>.json after every grid point,
and resumes from it.
"""

import argparse
import json
import os
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--setting", choices=("default", "tight"), default="default")
ap.add_argument("--link", choices=("rod", "quarter", "hundredth"), default="rod")
ap.add_argument("--arm", choices=("backward_euler_explicit", "backward_euler_native",
                                  "bdf2_native"), default="backward_euler_explicit")
args = ap.parse_args()
os.environ["SUPERDEX_PRECISION"] = "fp64" if args.setting == "tight" else "fp32"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "exp1_pendulum"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fitting  # noqa: E402
import pendulum_superdex as sdx  # noqa: E402
import run_gap  # noqa: E402

OUT = (ROOT / "exp1_pendulum" / "results"
       / f"gap_superdex_{args.setting}_{args.link}_{args.arm}.json")

run_gap.make_sim = lambda arm: sdx.SuperDexPendulum(arm, fitting.POPSIZE,
                                                    tight=args.setting == "tight",
                                                    link_inertia=sdx.LINKS[args.link])
run_gap.WITH_DELAY = {arm: True for arm in sdx.ARMS}
run_gap.renormalized = lambda arm, h: sdx.first_order_prediction_delay(arm, h)
run_gap.on_row = lambda rows, fine: OUT.write_text(
    json.dumps({"main": rows, "fine": fine}, indent=1, default=float))

if __name__ == "__main__":
    run_gap.run_arm(args.arm, json.loads(OUT.read_text()) if OUT.exists() else None)
