"""Cross-seed the Isaac flow of one scheme (../fitting.py, cross_seed).

Every grid point is refitted from the answers the other control rates found, so a flow that
still scatters by control rate does so because of PhysX, not because a chain settled in a
poorer minimum.

Run (inside the Isaac venv):  python cross_seed.py A
Rewrites exp1_pendulum/results/isaac_single_grid_<arm>.json in place.
"""

import json
import sys
from pathlib import Path

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

ROOT = Path(__file__).resolve().parent.parent  # the repo, wherever it is checked out
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "isaac"))
sys.path.insert(0, str(ROOT / "exp1_pendulum"))

import common as mj  # noqa: E402
import fitting  # noqa: E402
from pendulum_isaac import IsaacPendulum, fit_problem  # noqa: E402

ARM = sys.argv[1] if len(sys.argv) > 1 else "A"
T_FIT = 1.0  # run_single.py's default
OUT = ROOT / "exp1_pendulum" / "results" / f"isaac_single_grid_{ARM}.json"


def main():
    from isaacsim.core.api import World

    world = World(physics_dt=1 / 200, stage_units_in_meters=1.0, backend="numpy")
    sim = IsaacPendulum(world, ARM, 2 * 2 * 4)   # two starts, each probing 2 x 4 parameters
    rows = sorted(json.loads(OUT.read_text()), key=lambda r: r["h"])

    def problem(r):
        sim.set_dt(r["h"])
        return fit_problem(sim, r["f_ctrl"], r["dec"], (mj.F_REF,), T_FIT)

    moved = fitting.cross_seed(rows, problem, mj.PARAM0, key="params_identified",
                               rmse_key="rmse_identified")
    OUT.write_text(json.dumps(rows, indent=1, default=float))
    print(f"{moved} of {len(rows)} fits moved", flush=True)


main()
app.close()
