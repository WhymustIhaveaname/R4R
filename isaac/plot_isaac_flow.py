"""Isaac flow figure, drawn by the same code as the MuJoCo ones (../figures.py).

Three PhysX schemes, all first order, all fitting (rotor inertia, K_P, K_D, command delay).
The measured joint has no rotor inertia and no delay, so those two parameters start at zero.
Each scheme gets its own counterterms, derived in exp1_pendulum/derive_second_order.py:
an explicitly held torque makes PhysX behave as semi-implicit Euler (the step map was measured,
see probe_step_map_isaac.py), while PhysX's own drive is the implicit spring of
DyCpuGpuArticulation.h, whose damping is K_D + delta K_P rather than K_D.  PGS runs that spring
at delta = h; TGS advances time inside its position iterations, so there delta = h / N.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from figures import flow_figure, use_house_style  # noqa: E402
sys.path.insert(0, str(ROOT / "exp1_pendulum"))
from plot_flow import F_CTRL_COLOR, F_TICKS, H_MAX  # noqa: E402  (MuJoCo's grid)

RESULTS = ROOT / "exp1_pendulum" / "results"
KP0, KD0 = 80.0, 2.0
J0 = 0.45359237 * 0.3048 ** 2 / 3  # the measured rod inertia
# The shared fitter works in total inertia, so the stored first parameter is J, not the
# rotor inertia the joint carries on top of the rod.
BASE = np.array([J0, KP0, KD0, 0.0])
TGS_ITERS = 32  # Isaac's default solverPositionIterationCount
# (key, column title, which step map it realizes, substeps the drive sees)
SCHEMES = [("A", "explicit PD + TGS", "euler", 1),
           ("B", "PhysX drive + TGS", "drive", TGS_ITERS),
           ("C", "PhysX drive + PGS", "drive", 1)]
# Counterterms from exp1_pendulum/derive_second_order.py, as (per ms, per ms^2) of the
# relative shift.  PhysX's drive is an implicit spring run at delta = h / N, but gravity is not
# substepped - eENABLE_EXTERNAL_FORCES_EVERY_ITERATION_TGS is off by default, so it is applied
# once over the whole step.  That is what makes N enter the h^2 terms the way it does; at N = 1
# the expressions reduce to PGS exactly.  An explicitly held torque instead makes PhysX behave
# as semi-implicit Euler, whatever the iteration count (measured in probe_step_map_isaac.py).


def drive_theory(n):
    return {0: (-KD0 / (2 * n) / J0 * 1e-3,
                (-KP0 / 12 + KP0 / (6 * n**2) + KD0**2 / (12 * J0 * n**2)) / J0 * 1e-6),
            1: (0.0, KP0 * (1 - n**2) / (12 * J0 * n**2) * 1e-6),
            2: (-KP0 / n / KD0 * 1e-3, KP0 * (2 - n**2) / (12 * J0 * n**2) * 1e-6),
            3: (0.5 / n, 0.0)}


EULER_THEORY = {0: (KD0 / 2 / J0 * 1e-3, (J0 * KP0 + KD0**2) / (12 * J0) / J0 * 1e-6),
                1: (0.0, 0.0),
                2: (0.0, KP0 / (12 * J0) * 1e-6),
                3: (0.5, 0.0)}


def theory(kind, n):
    return EULER_THEORY if kind == "euler" else drive_theory(n)


# (parameter index, y label, unit).  The theory comes from theory() above.
ABSOLUTE = [(0, r"rotor inertia  ($10^{-3}$ kg m$^2$)", 1e-3),
            (2, r"$\Delta K_D = K_D^\star - K_D$  ($10^{-2}$ N m s/rad)", 1e-2),
            (1, r"$\Delta K_P = K_P^\star - K_P$  (N m/rad)", 1.0),
            (3, r"command delay $d^\star$ (ms)", 1e-3)]
# Each shift as a fraction of the measured value. The joint carries no rotor inertia of its own,
# so that row is normalized by the rod inertia J, exactly as the MuJoCo figure's row is.
RELATIVE = [(0, r"rotor inertia $/\, J$", J0),
            (2, r"$\Delta K_D / K_D$", KD0),
            (1, r"$\Delta K_P / K_P$", KP0),
            (3, r"command delay $d^\star$ (ms)", 1e-3)]
ROWS = RELATIVE
KP_ROW = 2  # the third row of the figure, whose y range follows the inertia row above it

CAPTION = ("Isaac Sim 5.1 / PhysX, single pendulum, on the same grid and fit as the MuJoCo figure.\n"
           "First-order theory (solid); fit (dashed), $c_1 h + c_2 h^2$ through the origin, so "
           "$c_1$ is the measured first-order slope.\n"
           "An explicitly held torque makes PhysX behave as semi-implicit Euler; its own drive "
           "is an implicit spring whose damping is $K_D + \\delta K_P$, run at "
           f"$\\delta = h/{TGS_ITERS}$ under TGS and $\\delta = h$ under PGS.")


def column(key, name, kind, substeps, rows):
    rows = sorted((r for r in rows if r["arm"] == key), key=lambda r: r["h"])
    h = np.array([r["h"] * 1e3 for r in rows])
    shift = np.array([r["params"] for r in rows]) - BASE
    t = theory(kind, substeps)
    return {"title": f"{name}\nfitted: rotor inertia, $K_P$, $K_D$, $d$", "h": h,
            "rows": [(shift[:, k] / unit, t[k]) for k, _, unit in ROWS],
            "rmse": np.array([r["rmse"] for r in rows]),
            "groups": np.array([r["f_ctrl_nominal"] for r in rows])}


def main(absolute=False):
    global ROWS
    ROWS = ABSOLUTE if absolute else RELATIVE
    use_house_style()
    rows = json.loads((RESULTS / "flow_isaac.json").read_text())
    columns = [column(*scheme, rows) for scheme in SCHEMES]
    # Show K_P on the scale the same scheme's rotor inertia moves on, where it is a flat line.
    ranges = []
    for col in columns:
        y = np.asarray(col["rows"][0][0])
        lo, hi = min(0.0, y.min()), max(0.0, y.max())
        pad = 0.05 * (hi - lo)
        ranges.append((lo - pad, hi + pad))
    stem_out = "fig_flow_isaac_single_absolute" if absolute else "fig_flow_isaac_single"
    flow_figure(columns, [label for _, label, _ in ROWS],
                [RESULTS / f"{stem_out}.{e}" for e in ("png", "pdf")],
                H_MAX, CAPTION, F_TICKS, quadratic=True, show_rmse=False,
                # K_P does flow here: under TGS the h^2 counterterm is -K_P^2/12J.
                no_fit=(),
                row_ylim=None if absolute else {KP_ROW: ranges},
                point_colors=F_CTRL_COLOR,
                legend_entries=[(rf"$f_\mathrm{{ctrl}} \approx$ {fc} Hz", c)
                                for fc, c in F_CTRL_COLOR.items()])
    print("wrote", RESULTS / f"{stem_out}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--absolute", action="store_true",
                    help="plot the raw shifts instead of fractions of the measured value")
    a = ap.parse_args()
    main(a.absolute)
