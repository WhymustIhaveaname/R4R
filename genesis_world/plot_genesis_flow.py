"""Genesis flow figure, drawn by the same code as the MuJoCo one (exp1_pendulum/plot_flow.py).

Genesis's Euler and implicitfast are MuJoCo's step maps, so the columns, the theory and the
drawing are MuJoCo's; only the points come from Genesis.  approximate_implicitfast, Genesis's
default, is not swept: on this pendulum it is implicitfast to round-off.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "exp1_pendulum"))

from figures import flow_figure, use_house_style  # noqa: E402
from plot_flow import (CAPTION, F_CTRL_COLOR, F_TICKS, H_MAX, KP_ROW, RELATIVE, RESULTS,  # noqa: E402
                       column)

ARMS = ("euler_explicit", "implicitfast_native")


def main():
    use_house_style()
    rows = json.loads((RESULTS / "flow_genesis.json").read_text())
    columns = [column(arm, rows, True, second_order=True) for arm in ARMS]
    # Show K_P on the scale the same scheme's J moves on, sign included, where it is a flat line.
    ranges = []
    for col in columns:
        y = np.asarray(col["rows"][0][0])
        lo, hi = min(0.0, y.min()), max(0.0, y.max())
        pad = 0.05 * (hi - lo)
        ranges.append((lo - pad, hi + pad))
    caption = "Genesis 1.5.0 as shipped (FP32), on the CPU.\n" + CAPTION
    flow_figure(columns, [label for _, label, _ in RELATIVE],
                [RESULTS / f"fig_flow_genesis.{e}" for e in ("png", "pdf")],
                H_MAX, caption, F_TICKS, point_colors=F_CTRL_COLOR, quadratic=True,
                show_rmse=False, no_fit=(KP_ROW,), row_ylim={KP_ROW: ranges},
                legend_entries=[(rf"$f_\mathrm{{ctrl}} \approx$ {fc} Hz", c)
                                for fc, c in F_CTRL_COLOR.items()])
    print("wrote", RESULTS / "fig_flow_genesis.png")


if __name__ == "__main__":
    main()
