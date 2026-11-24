"""SuperDex flow figures, drawn by the same code as the MuJoCo and Isaac ones (../figures.py).

One figure per solver setting and link model, one column per scheme, on MuJoCo's grid.
The counterterms come from exp1_pendulum/derive_second_order.py: backward Euler with a held
torque (A) has semi-implicit Euler's, and with Mochi's own PD (B) it has PhysX's PGS drive's,
both to second order.  BDF2 (C) has no first-order shift of the parameters; only the command
moves, by -h/2.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "exp1_pendulum"))

from common import J0, KD0, KP0, PARAM0  # noqa: E402
from figures import flow_figure, use_house_style  # noqa: E402
from plot_flow import F_CTRL_COLOR, F_TICKS, H_MAX, HEAD, RELATIVE  # noqa: E402

RESULTS = ROOT / "exp1_pendulum" / "results"
ARM_LABEL = {"A": "A: backward Euler + explicit PD",
             "B": "B: backward Euler + native PD",
             "C": "C: BDF2 + native PD"}
LINK_LABEL = {"rod": "the rod", "quarter": "a lever carrying J/4",
              "hundredth": "a lever carrying J/100"}
SETTING_LABEL = {"default": "as shipped (FP32, at most 4 Newton iterations)",
                 "tight": "FP64, Newton converged to round-off"}
# (per ms, per ms^2) of each relative shift, keyed by parameter index (J, K_P, K_D, d).
J2 = (J0 * KP0 + KD0**2) / (12 * J0) / J0 * 1e-6
THEORY = {"A": {0: (KD0 / 2 / J0 * 1e-3, J2), 1: (0.0, 0.0), 2: (0.0, KP0 / (12 * J0) * 1e-6),
                3: (0.5, 0.0)},
          "B": {0: (-KD0 / 2 / J0 * 1e-3, J2), 1: (0.0, 0.0),
                2: (-KP0 / KD0 * 1e-3, KP0 / (12 * J0) * 1e-6), 3: (0.5, 0.0)},
          "C": {0: (0.0, 0.0), 1: (0.0, 0.0), 2: (0.0, 0.0), 3: (-0.5, 0.0)}}
KP_ROW = 2  # the third row of the figure, whose y range follows the inertia row above it


def column(arm, rows):
    rows = sorted((r for r in rows if r["arm"] == arm), key=lambda r: r["h"])
    h = np.array([r["h"] * 1e3 for r in rows])
    shift = np.array([r["params"] for r in rows]) - np.append(PARAM0, 0.0)
    return {"title": f"{ARM_LABEL[arm]}\nfitted: $J, K_P, K_D, d$", "h": h,
            "rows": [(shift[:, k] / unit, THEORY[arm][k]) for k, _, unit in RELATIVE],
            "rmse": np.array([r["rmse"] for r in rows]),
            "groups": np.array([r["f_ctrl_nominal"] for r in rows])}


def main(setting, link):
    use_house_style()
    rows = json.loads((RESULTS / f"flow_superdex_{setting}_{link}.json").read_text())
    columns = [column(arm, rows) for arm in "ABC" if any(r["arm"] == arm for r in rows)]
    # Show K_P on the scale the same scheme's J moves on, where it is a flat line.
    ranges = []
    for col in columns:
        y = np.asarray(col["rows"][0][0])
        lo, hi = min(0.0, y.min()), max(0.0, y.max())
        pad = 0.05 * (hi - lo)
        ranges.append((lo - pad, hi + pad))
    caption = (f"SuperDex Physics 1.0.1, {SETTING_LABEL[setting]}; the pendulum's link is "
               f"{LINK_LABEL[link]}.\n" + HEAD +
               "Theory (solid) to second order for A and B; for C, first order.  "
               "Fit (dashed): $c_1 h + c_2 h^2$ through the origin.")
    stem = f"fig_flow_superdex_{setting}_{link}"
    flow_figure(columns, [label for _, label, _ in RELATIVE],
                [RESULTS / f"{stem}.{e}" for e in ("png", "pdf")],
                H_MAX, caption, F_TICKS, point_colors=F_CTRL_COLOR, quadratic=True,
                show_rmse=False, no_fit=(KP_ROW,), row_ylim={KP_ROW: ranges},
                legend_entries=[(rf"$f_\mathrm{{ctrl}} \approx$ {fc} Hz", c)
                                for fc, c in F_CTRL_COLOR.items()])
    print("wrote", RESULTS / f"{stem}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("setting", choices=tuple(SETTING_LABEL))
    ap.add_argument("link", choices=tuple(LINK_LABEL))
    a = ap.parse_args()
    main(a.setting, a.link)
