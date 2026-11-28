"""Accuracy against cost on the Jello cube: one line per method, every parameter renormalized.

Each point is one (cells per side, step h) of one method, from results/speed_accuracy/*.json:
  x       wall seconds of its stepping loop per simulated second, 1 / the row's speed, on the
          cluster (one CPU core for MuJoCo, one B300 GPU for Genesis); above 100 not run
  left    the frequency residual: E* fitted to the sway and twist frequencies of the same
          model's continuum (sway at the video's 8.010 Hz, twist 10.087 Hz); the RMS of the two
          relative errors left: the discretization's error alone.  Against the video's twist
          (9.992 Hz) the continuum itself is off by FLOOR, which no engine can beat but by
          cancellation, so the video is not the target here; that floor is drawn as a line.
  right   the damping residual: every damping parameter the engine offers fitted to the two
          damping ratios; the RMS of the two relative errors left.  The damping model's.
          Both from runs at the fitted parameters.
A method's line joins its cell counts, coarse to fine, each at the largest step tried there, not
counting MPM's and PBD's steps chosen for their own damping; its other points are faint.

  python plot_speed_accuracy.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common as c  # noqa: E402
from figures import INK2, clear_legends, style, use_house_style  # noqa: E402

DATA = c.RESULTS / "speed_accuracy"
# colour by engine; marker by method, shared by the two engines' like methods: a circle for
# symplectic Euler on linear tetrahedra, a square for backward Euler on them
ENGINE = {"mujoco": "#2a78d6", "genesis": "#eb6834"}
METHODS = {   # name: (label, engine, marker), the README's order
    "mj_discrete": ("MuJoCo flex, discrete", "mujoco", "s"),
    "mj_euler": ("MuJoCo flex, Euler", "mujoco", "o"),
    "mj_trilinear": ("MuJoCo flex, trilinear", "mujoco", "^"),
    "mj_quadratic": ("MuJoCo flex, quadratic", "mujoco", "v"),
    "gs_fem_explicit": ("Genesis FEM, explicit", "genesis", "o"),
    "gs_fem_implicit": ("Genesis FEM, implicit", "genesis", "s"),
    "gs_ipc": ("Genesis FEM, IPC", "genesis", "D"),
    "gs_mpm": ("Genesis MPM", "genesis", "P"),
    "gs_pbd": ("Genesis PBD", "genesis", "X"),
}


SWAY = 8.010                                   # Hz, the video's sway, which sets E
TWIST_VIDEO = 9.992                            # Hz, the video's twist, same pluck


def continuum_twist():
    """The same model's continuum twist at the E that puts its sway at the video's."""
    c_sway, c_twist = json.loads((c.CACHE / "jello_continuum_modes.json").read_text())
    return SWAY * c_twist / c_sway


def freq_residual(f, target):
    """RMS relative error of the frequencies f against target after the best common scale
    (a stiffness multiplier scales every frequency alike), in percent."""
    q = np.asarray(f) / np.asarray(target)
    k = q.sum() / (q * q).sum()
    return 100 * np.sqrt(np.mean((k * q - 1) ** 2))


def load():
    rows = [json.loads(p.read_text()) for p in sorted(DATA.glob("*.json"))]
    rows = [r for r in rows if not r.get("cut") and r.get("finite", True)]   # unstable runs have no accuracy
    twist = continuum_twist()
    for r in rows:
        if r.get("f_sway") and r.get("f_twist"):
            r["err_freq_model_pct"] = freq_residual((r["f_sway"], r["f_twist"]), (SWAY, twist))
    return rows


OWN_DAMPING = "own sway zeta"   # MPM's and PBD's steps at which their own sway damping is the video's


def on_line(rows):
    """Of one method's rows, the one drawn on its line at each cell count: the largest step, not
    counting the steps chosen for their own damping; coarse to fine."""
    out = {}
    for r in rows:
        if str(r.get("step_rule", "")).startswith(OWN_DAMPING):
            continue
        if r["cells"] not in out or r["h_s"] > out[r["cells"]]["h_s"]:
            out[r["cells"]] = r
    return [out[n] for n in sorted(out)]


def panel(ax, rows, key, ylabel, title, floor=None, clip=None):
    style(ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    for name, (label, engine, marker) in METHODS.items():
        mine = [r for r in rows if r["method"] == name and r.get(key) is not None and np.isfinite(r[key])]
        if not mine:
            continue
        color = ENGINE[engine]
        point = lambda r: (1 / r["speed"], r[key] if clip is None else max(r[key], clip))   # noqa: E731
        line = on_line(mine)
        f = [point(r) for r in line]
        rest = [point(r) for r in mine if all(r is not x for x in line)]
        if rest:
            ax.plot(*zip(*rest), marker, ms=4, color=color, alpha=0.3, mec="none", zorder=2)
        ax.plot(*zip(*f), "-", color=color, lw=2, zorder=3)
        ax.plot(*zip(*f), marker, ms=8, color=color, mec="white", mew=1.0, zorder=4, label=label)
    if floor is not None:
        ax.axhline(floor, color=INK2, lw=1.0, ls=":", zorder=1)   # the caption says what it is
    plain = FuncFormatter(lambda v, _: f"{v:g}")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(plain)
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if f"{v:.0e}"[0] in "25" else ""))
    ax.tick_params(axis="y", which="minor", labelsize=8)
    ax.set_xlabel("wall time / simulated time")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)


def main():
    use_house_style()
    rows = load()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    floor = freq_residual((SWAY, continuum_twist()), (SWAY, TWIST_VIDEO))
    panel(axes[0], rows, "err_freq_model_pct", "frequency residual (%)", "frequencies: the discretization", floor)
    panel(axes[1], rows, "err_damp_pct", "damping residual (%)", "damping ratios: the damping model", clip=0.1)
    axes[1].set_ylim(bottom=0.08)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0.86, 0.93))
    fig.text(0.43, 0.01,
             "Jello cube.  Each line: one method, from coarse to fine cells, each at the largest step tried; "
             "faint: its smaller steps,\nand MPM's and PBD's steps at which their own sway damping is the video's.\n"
             "Wall time: on the cluster, one CPU core for MuJoCo 4.0.0, one B300 GPU for Genesis 1.5.0; "
             "more than 100 times the simulated time not run.\n"
             r"Frequency residual: $E^\star$ fitted to the sway (8.010 Hz, the video's) and twist "
             f"({continuum_twist():.3f} Hz) of the same model's continuum; the RMS of the two relative errors left.\n"
             f"Dotted: that continuum against the video's twist (9.992 Hz), {floor:.2f}%: the model's own error.\n"
             r"Damping residual: every damping parameter the engine offers fitted to the video's sway "
             r"($\zeta$ 0.0151) and twist ($\zeta$ 0.0154); the RMS of the two relative errors left;"
             "\nbelow 0.1% drawn at 0.1%: Genesis FEM implicit, with both Rayleigh coefficients, meets both (to 1e-5%), "
             "but at 1 ms on 8 or more cells,\nwhere backward Euler's stability bound limits its stiffness damping.",
             ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.22, 0.86, 1))
    clear_legends(fig)
    for e in ("png", "pdf"):
        fig.savefig(c.RESULTS / f"fig_speed_accuracy.{e}", dpi=200)
    plt.close(fig)
    print("wrote", c.RESULTS / "fig_speed_accuracy.png", len(rows), "points")


if __name__ == "__main__":
    main()
