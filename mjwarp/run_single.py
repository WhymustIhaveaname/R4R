"""Experiment 1 swept on MuJoCo Warp, using exp1_pendulum/run_flow.py unchanged.

The sweep, the grid, the starting points and the fitter all come from the MuJoCo code; the only
thing supplied here is a simulator with the same rollout signature and a batched rollout, so the
two engines cannot differ by anything but their stepping.

One process per arm.  A 1-DOF model leaves the GPU launch-bound, so a step of 160 worlds costs
what a step of 32 does, and every world keeps its own time step: the five chains of an arm
share every rollout, whatever their steps, and then the arm is cross-seeded.  Both resume
from the per-chain files on disk.

Usage (inside the MuJoCo Warp venv):
    python run_single.py --arm A [--delay]
    python run_single.py --merge [--delay]                # concatenate the per-chain files
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "exp1_pendulum"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fitting  # noqa: E402
import run_flow as rf  # noqa: E402

NSTEP_MAX = int(round(4.0 / 0.25e-3)) + 1          # 4 s of command at the finest step
RESULTS = ROOT / "exp1_pendulum" / "results"
PARTS = RESULTS / "mjwarp_parts"


def flag(name, default=None):
    return next((a.split("=", 1)[1] if "=" in a else sys.argv[sys.argv.index(a) + 1]
                 for a in sys.argv if a.split("=", 1)[0] == f"--{name}"), default)


def stem():
    return "flow_mjwarp_delay" if rf.WITH_DELAY else "flow_mjwarp"


def merge():
    # the [ABC] keeps "flow_mjwarp_*" from also matching "flow_mjwarp_delay_*"
    parts = sorted(PARTS.glob(f"{stem()}_[ABC]_*.json"))
    rows = [r for f in parts for r in json.loads(f.read_text())]
    out = RESULTS / f"{stem()}.json"
    out.write_text(json.dumps(rows, indent=1, default=float))
    print(f"merged {len(rows)} fits from {len(parts)} chains into {out}")


def main():
    if "--merge" in sys.argv:
        return merge()
    from pendulum_mjwarp import WarpPendulum

    arm = flag("arm", "A")
    # A generation of CMA-ES for each of the five chains, in one rollout.
    sim = WarpPendulum(arm, nworld=len(rf.F_CTRL) * fitting.POPSIZE, nstep_max=NSTEP_MAX)
    rf.make_sim = lambda a: sim
    PARTS.mkdir(parents=True, exist_ok=True)
    save = lambda path, rows: path.write_text(json.dumps(rows, indent=1, default=float))
    part = lambda f: PARTS / f"{stem()}_{arm}_{f:03d}.json"
    t0 = time.perf_counter()
    rf.on_row = lambda rows: save(part(rows[0]["f_ctrl_nominal"]), rows)
    rows = rf.chains(arm, rf.F_CTRL, {f: json.loads(part(f).read_text()) if part(f).exists()
                                      else [] for f in rf.F_CTRL})
    print(f"{arm}: flow done in {(time.perf_counter() - t0) / 60:.0f} min", flush=True)
    files = {part(f): rs for f, rs in rows.items()}
    every = sorted((r for rs in files.values() for r in rs), key=lambda r: r["h"])
    moved = rf.cross_seed(every, on_row=lambda: [save(f, rs) for f, rs in files.items()])
    print(f"{arm}: {moved} of {len(every)} fits moved", flush=True)


if __name__ == "__main__":
    main()
