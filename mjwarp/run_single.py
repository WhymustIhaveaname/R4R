"""Experiment 1 swept on MuJoCo Warp, using exp1_pendulum/run_flow.py unchanged.

The sweep, the grid, the starting points and the fitter all come from the MuJoCo code; the only
thing supplied here is a simulator with the same rollout signature and a batched rollout, so the
two engines cannot differ by anything but their stepping.

One (arm, nominal control frequency) chain per process, because a 1-DOF model leaves the GPU
launch-bound rather than compute-bound: several processes overlap almost perfectly, while a
single process stepping more worlds would not go any faster.

Usage (inside the MuJoCo Warp venv):
    python run_single.py --arm A --fctrl 50 [--delay]
    python run_single.py --arm A --cross-seed [--delay]   # once all five chains of the arm are done
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
    # A generation of CMA-ES is one rollout of POPSIZE worlds.
    rf.make_sim = lambda a: WarpPendulum(a, nworld=fitting.POPSIZE, nstep_max=NSTEP_MAX)
    PARTS.mkdir(parents=True, exist_ok=True)
    save = lambda path, rows: path.write_text(json.dumps(rows, indent=1, default=float))
    t0 = time.perf_counter()
    if "--cross-seed" in sys.argv:
        files = {f: json.loads(f.read_text()) for f in sorted(PARTS.glob(f"{stem()}_{arm}_*.json"))}
        rows = sorted((r for rs in files.values() for r in rs), key=lambda r: r["h"])
        moved = rf.cross_seed(rows, on_row=lambda: [save(f, rs) for f, rs in files.items()])
        print(f"{arm}: {moved} of {len(rows)} fits moved", flush=True)
    else:
        f_nominal = int(flag("fctrl", "50"))
        part = PARTS / f"{stem()}_{arm}_{f_nominal:03d}.json"
        rf.on_row = lambda rows: save(part, rows)
        rows = rf.chain((arm, f_nominal), json.loads(part.read_text()) if part.exists() else None)
        print(f"wrote {len(rows)} fits in {(time.perf_counter() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
