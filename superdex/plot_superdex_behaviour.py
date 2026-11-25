"""SuperDex behaviour figure, drawn by the same code as the MuJoCo and Isaac ones (../figures.py).

The pendulum as built and as shipped: the rod, FP32, Mochi's own Newton tolerances, and the
explicit-PD scheme.  Panel (a) is the trajectory error at 200 Hz, panel (b) the error of the
measured and predicted parameter sets against the simulation frequency.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "exp1_pendulum"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pendulum_superdex as sdx  # noqa: E402
from common import F_REF, PARAM0, ground_truth, ref_sequence  # noqa: E402
from figures import MEASURED, RENORM, behaviour_figure, use_house_style  # noqa: E402
from run_gap import F_CTRL  # noqa: E402

RESULTS = ROOT / "exp1_pendulum" / "results"
SCHEME = "backward_euler_explicit"
DEMO_DEC = 4  # h = 5 ms: 200 Hz physics, as in the MuJoCo and Isaac figures
DEMO_T = 2.0  # s shown
F_MAX = 1000


def main():
    use_house_style()
    rows = sorted(json.loads((RESULTS / f"gap_superdex_default_rod_{SCHEME}.json").read_text())
                  ["main"], key=lambda r: r["h"])
    row = next(r for r in rows if r["dec"] == DEMO_DEC)
    sim = sdx.SuperDexPendulum(SCHEME, 1, tight=False)
    refs = ref_sequence(F_CTRL, F_REF)
    gt = ground_truth(refs, F_CTRL, DEMO_DEC)
    renorm = sdx.first_order_prediction_delay(SCHEME, row["h"])
    error = {"t": np.arange(1, len(gt) + 1) * row["h"], "xlim": DEMO_T,
             "ylabel": r"simulated $-$ real (rad)",
             "title": "SuperDex at $f$ = 200 Hz, backward Euler + explicit PD",
             "series": [(sim.rollout(refs, F_CTRL, DEMO_DEC, *p) - gt, c, n)
                        for p, c, n in (((*PARAM0, 0.0), MEASURED, "without renormalization"),
                                        (renorm, RENORM, "renormalized parameters"))]}

    series = []
    # No fitted parameter set: the flow depends on the control rate, so no fit carries over.
    for key, color, label in (("rmse_measured", MEASURED, "without renormalization"),
                              ("rmse_renormalized", RENORM, "renormalized (no fitting)")):
        ok = [r for r in rows if r.get(key) is not None and r["f"] <= F_MAX]
        series.append(([r["f"] for r in ok], [r[key] for r in ok], color, label))
    lost = [r for r in rows if r.get("rmse_measured") is None]
    lo, hi = row["rmse_renormalized"], row["rmse_measured"]
    gap = {"series": series, "ylim": (1e-6, 1), "legend_loc": "upper right",
           "ylabel": "trajectory RMSE, all five commands (rad)",
           "diverge_f": max(r["f"] for r in lost) if lost else None,
           "annotation": (f"{hi / lo:.0f}$\\times$ smaller,\nwithout fitting", row["f"], lo, hi)}

    behaviour_figure(error, gap,
                     [RESULTS / f"fig_behaviour_superdex.{e}" for e in ("png", "pdf")], F_MAX)
    print("wrote", RESULTS / "fig_behaviour_superdex.png")


if __name__ == "__main__":
    main()
