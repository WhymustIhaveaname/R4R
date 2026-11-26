"""Flow of the effective parameters with the physics step: one figure for MuJoCo, one for Genesis.

One column per scheme, one row per parameter: the rotor inertia, the PD damping, the command
delay, and the factor lam on the contact's stiffness, damping and regularization.  A scheme
fits only the parameters the theory gives a counterterm (common.FITTED).  The drawing is the
shared ../figures.py's, as in experiments 1 and 2.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common as c  # noqa: E402
from figures import flow_figure, use_house_style  # noqa: E402

RESULTS = Path(__file__).parent / "results"
H_MAX = 5.0  # ms
F_CTRL_COLOR = {10: "#86b6ef", 20: "#5598e7", 50: "#2a78d6", 100: "#1c5cab", 200: "#104281"}
F_TICKS = [(4000, "4k"), (1000, "1k"), (500, "500"), (400, "400"), (200, "200")]
NOMINAL = np.array([c.J0, c.KP0, c.KD0, 0.0, 1.0])
# (parameter index, y label, unit)
ROWS = [(0, r"$\Delta J / J$", c.J0),
        (2, r"$\Delta K_D / K_D$", c.KD0),
        (3, r"command delay $d^\star$ (ms)", 1e-3),
        (4, r"contact factor $\lambda^\star - 1$", 1.0)]
SYMBOL = {0: "$J$", 2: "$K_D$", 3: "$d$", 4: r"$\lambda$"}
# why a row is not fitted, where the theory gives it no counterterm
NOT_FITTED = {2: "no counterterm in this scheme", 3: "no command advance in this scheme",
              4: "the contact needs no counterterm"}
# Schemes with a first-order term that no parameter absorbs: their points scatter by control
# rate, so a fit through them would suggest a law; they get the theory line only.
SCATTERED = ("implicitfast_native",)
ENGINES = {"mujoco": ("MuJoCo", c.ARMS),
           "genesis": ("Genesis 1.5.0 (FP32), its contact set to MuJoCo's",
                       ("euler_explicit", "implicitfast_native"))}
CAPTION = ("{engine}.\nTarget $h$ = 0.25, 0.5, ..., 5 ms ($f$ = 200-4000 Hz), realized with "
           "integer $f_\\mathrm{{ctrl}}$ (Hz) and integer decimation.\n"
           "$\\lambda$ scales the contact's stiffness, damping and regularization.  Parameters "
           "not fitted are held at their measured values.\n"
           "Residual over 4 s, both coordinates, the block's as $x / L$.\n"
           "First-order theory (solid); implicitfast also has a first-order term that no "
           "parameter absorbs.\n"
           "Fit (dashed): $c_1 h + c_2 h^2$ through the origin, so $c_1$ is the measured "
           "first-order slope; none where the points scatter by control rate.")


def column(arm, rows):
    rows = sorted((r for r in rows if r["arm"] == arm), key=lambda r: r["h"])
    h = np.array([r["h"] * 1e3 for r in rows])
    params = np.array([np.append(r["params"], NOMINAL[len(r["params"]):]) for r in rows])
    shift = params - NOMINAL
    slope = c.first_order_prediction(arm, 1e-3) - NOMINAL  # per ms
    cells = [(shift[:, k] / unit, slope[k] / unit) if k in c.FITTED[arm] else NOT_FITTED[k]
             for k, _, unit in ROWS]
    fitted = ", ".join(SYMBOL[k] for k in c.FITTED[arm])
    return {"title": f"{c.ARM_LABEL[arm]}\nfitted: {fitted}", "h": h, "rows": cells,
            "groups": np.array([r["f_ctrl_nominal"] for r in rows]),
            "rmse": np.array([r["rmse"] for r in rows])}


def main():
    use_house_style()
    for engine, (name, arms) in ENGINES.items():
        rows = json.loads((RESULTS / f"flow_{engine}.json").read_text())
        columns = [column(arm, rows) for arm in arms]
        no_fit = [(i, j) for j, arm in enumerate(arms) if arm in SCATTERED for i in range(len(ROWS))]
        flow_figure(columns, [label for _, label, _ in ROWS],
                    [RESULTS / f"fig_flow_{engine}.{e}" for e in ("png", "pdf")],
                    H_MAX, CAPTION.format(engine=name), F_TICKS, point_colors=F_CTRL_COLOR,
                    quadratic=True, show_rmse=False, no_fit=no_fit,
                    legend_entries=[(rf"$f_\mathrm{{ctrl}} \approx$ {fc} Hz", col)
                                    for fc, col in F_CTRL_COLOR.items()])
        print("wrote", RESULTS / f"fig_flow_{engine}.png")


if __name__ == "__main__":
    main()
