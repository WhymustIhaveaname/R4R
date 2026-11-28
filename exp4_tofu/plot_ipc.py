"""Genesis 1.5.0's FEM through the IPC coupler (libuipc): the free vibration and the Jello cube.

fig_flow_gs_ipc:     E* against h, divided by its h -> 0 value from the mesh's modes; the damping
                     the scheme gives, which no parameter sets; and E* / T(h) against dx, every
                     run.  The theory is undamped backward Euler on one mode, exactly.
fig_jello_gs_ipc:    the Jello video's sway against the cube in libuipc, nominal and renormalized.
                     (fig_indent_gs_ipc is drawn by plot_indent.py from run_ipc.py's data.)

  python plot_ipc.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common as c  # noqa: E402
import genesis_ipc as gi  # noqa: E402
import run_flow as rf  # noqa: E402
import run_ipc as ri  # noqa: E402
from figures import FIT_COLOR, INK, INK2, clear_legends, num, style, use_house_style  # noqa: E402

DX_COLOR = {10.0: "#86b6ef", 5.0: "#5598e7", 3.333: "#2a78d6", 2.5: "#1c5cab", 2.0: "#104281"}
DX_SIZE = {10.0: 11.0, 5.0: 9.0, 3.333: 7.5, 2.5: 6.0, 2.0: 4.5}   # rings where they coincide
MISFIT_MAX = 1e-6


def key(dx):
    return round(dx * 1e3, 3)


def pct(x):
    return f"{x:.2g}%" if x >= 0.01 else f"{x:.0e}%"


def f_axis(ax):
    top = ax.secondary_xaxis("top")
    ticks = [(4000, "4k"), (1000, "1k"), (500, "500"), (200, "200")]
    top.set_xticks([1e3 / f for f, _ in ticks])
    top.set_xticklabels([t for _, t in ticks], fontsize=8, color=INK2)
    top.set_xlabel("$f$ (Hz)", fontsize=8, color=INK2, labelpad=2)


def points(ax, rows, x, y):
    for k, color in sorted(DX_COLOR.items(), key=lambda kc: -DX_SIZE[kc[0]]):
        sel = [r for r in rows if key(r["dx"]) == k]
        if sel:
            ax.plot([x(r) for r in sel], [y(r) for r in sel], "o", ms=DX_SIZE[k], color=color, mec="white",
                    mew=0.7, zorder=3)


def flow_figure(rows):
    hs = np.linspace(1e-6, 5.2e-3, 300)
    theory = np.array([gi.backward_euler_time_factor(h) for h in hs])
    h_star = ri.h_star()
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.3))
    for ax in axes:
        style(ax)

    ax = axes[0]
    ax.axhline(0, color="#e4e3df", lw=1, zorder=0)
    ax.plot(hs * 1e3, 100 * (theory[:, 0] - 1), color=INK, lw=1.4, zorder=1, label="theory")
    points(ax, rows, lambda r: r["h"] * 1e3, lambda r: 100 * (r["E_ratio"] / r["S_theory"] - 1))
    worst = max(abs(r["E_ratio"] / (r["S_theory"] * r["time_theory"][0]) - 1) for r in rows)
    ax.set_ylabel(r"$E^\star / E^\star_{h \to 0} - 1$ (%)")
    ax.set_title(f"the stiffness\nruns within {pct(100 * worst)} of the theory", fontsize=10.5)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    ax = axes[1]
    ax.plot(hs * 1e3, theory[:, 1], color=INK, lw=1.4, zorder=1, label="theory")
    points(ax, rows, lambda r: r["h"] * 1e3, lambda r: r["zeta"])
    ax.axhline(c.ZETA0, color=INK, lw=1, ls=":", zorder=0)
    ax.text(5.1, c.ZETA0 - 0.004, "tofu's own damping, 0.10", ha="right", va="top", fontsize=8.5, color=INK2)
    ax.plot([h_star * 1e3] * 2, [0, c.ZETA0], color=INK2, lw=0.8, ls=":", zorder=0)
    ax.text(h_star * 1e3 + 0.08, 0.012, f"$h^\\star$ = {h_star * 1e3:.3f} ms", fontsize=8.5, color=INK2)
    worst = max(abs(r["zeta"] / r["time_theory"][1] - 1) for r in rows)
    ax.set_ylabel(r"$\zeta_\mathrm{IPC}$")
    ax.set_ylim(0, None)
    ax.set_title(f"the damping, which no parameter sets\nruns within {pct(100 * worst)} of the theory",
                 fontsize=10.5)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    for ax in axes[:2]:
        ax.set_xlim(0, 5.2)
        ax.set_xlabel("$h = 1/f$ (ms)", fontsize=9)
        f_axis(ax)

    ax = axes[2]
    table = rf.theory_table()
    ax.set_xlim(0, 10.5)
    ax.set_xlabel(r"$\Delta x$ (mm)", fontsize=9)
    ax.axhline(0, color="#e4e3df", lw=1, zorder=0)
    x = np.array([r["dx"] * 1e3 for r in rows])
    e = 100 * np.array([r["E_ratio"] / r["time_theory"][0] - 1 for r in rows])
    S = 100 * np.array([r["S_theory"] - 1 for r in rows])
    dxs = np.array(rf.DXS) * 1e3
    St = 100 * (np.array([table[f"{d:.6f}_volume"]["S"] for d in rf.DXS]) - 1)
    ax.plot(dxs, St, "o", ms=12, mfc="none", mec=INK, mew=1.2, zorder=2, label="modal analysis")
    A = np.stack([x ** 2, x ** 3], 1)
    a = np.linalg.lstsq(A, e, rcond=None)[0]
    r2 = 1 - np.sum((e - A @ a) ** 2) / np.sum((e - e.mean()) ** 2)
    xd = np.linspace(0, 10.5, 200)
    ax.plot(xd, a[0] * xd ** 2 + a[1] * xd ** 3, color=FIT_COLOR, lw=2, ls="--", zorder=1,
            label=f"fit: $c_2 \\Delta x^2 + c_3 \\Delta x^3$, $R^2$ = {r2:.4f}\n$c_2$ = {num(a[0])}, $c_3$ = {num(a[1])}")
    n_h = len({r["h"] for r in rows})
    ax.plot(x, e, "o", ms=5, color="#2a78d6", mec="white", mew=0.6, zorder=3,
            label=f"{len(rows)} runs, {n_h} steps $h$:\nwithin {pct(np.abs(e - S).max())} of the modal analysis")
    ax.set_ylabel(r"$E^\star / (E\,T(h)) - 1$ (%)")
    ax.set_title("the cells", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")

    fig.suptitle("Genesis 1.5.0, FEM through the IPC coupler (libuipc 0.0.30: backward Euler, no damping)",
                 fontsize=11.5, y=0.995)
    dots = [plt.Line2D([], [], ls="none", marker="o", ms=DX_SIZE[k], color=col, mec="white")
            for k, col in DX_COLOR.items()]
    fig.legend(dots, [rf"$\Delta x$ = {k:g} mm" for k in DX_COLOR], loc="upper center", ncol=5, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.945))
    caption = (r"$E^\star$ (Pa): the Young's modulus given to the simulator so that the block's lowest mode has the "
               r"continuum's frequency, 14.4 Hz at $E$ = 11.3 kPa."
               "\n" + r"$E^\star_{h \to 0}$ (Pa): $E^\star$ at the same $\Delta x$ as $h \to 0$, from the modal "
               r"analysis: the lowest mode of the same mesh, an eigenproblem."
               "\n" + r"$\zeta_\mathrm{IPC}$ (dimensionless): the damping ratio of the mode, backward Euler's own; "
               r"$h^\star$ (ms): the step at which it is the continuum's 0.10."
               "\n" + r"$T(h)$ (dimensionless): the theory's $E^\star / E^\star_{h \to 0}$.  Theory: undamped "
               r"backward Euler on one mode, its eigenvalue $1/(1 - i\omega h)$ exactly, $\omega$ (rad/s) the mesh's mode at $E^\star$.")
    fig.text(0.5, 0.008, caption, ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.13, 1, 0.92))
    clear_legends(fig)
    for e_ in ("png", "pdf"):
        fig.savefig(c.RESULTS / f"fig_flow_gs_ipc.{e_}", dpi=200)
    plt.close(fig)
    print("wrote", c.RESULTS / "fig_flow_gs_ipc.png")


def jello_figure(root="/data/datasets/vvt_jello_cube"):
    """The Jello video's sway against the cube in libuipc, nominal (E, h = 1 ms) and renormalized
    (E*, h*), each drawn with its own frequency and damping and the video's amplitude and phase,
    as plot_jello.py draws MuJoCo's."""
    import jello_sim as js
    import plot_jello as pj
    from figures import RENORM
    path = c.RESULTS / "ipc_jello.json"
    if not path.exists() or not Path(root).exists():
        return
    d = json.loads(path.read_text())
    runs = {r["name"]: r for r in d["runs"]}
    t, q = pj.video_sway(root)
    fig, ax = plt.subplots(figsize=(10, 4.4))
    style(ax)
    ren, nom = runs["renormalized"], runs["nominal"]
    ax.plot(t, q, "o", ms=1.6, color=INK, mec="none", zorder=1,
            label=f"video, 4000 fps: f {js.F_VIDEO:.2f} Hz, $\\zeta$ {js.ZETA_VIDEO:.3f}")
    ax.plot(t, pj.matched(ren["f"], ren["zeta"], t, q), color=RENORM, lw=1.6, zorder=3,
            label=f"IPC, renormalized $E^\\star$, $h^\\star$ = {ren['h'] * 1e3:.3f} ms: f {ren['f']:.2f} Hz, "
                  f"$\\zeta$ {ren['zeta']:.3f}")
    ax.plot(t, pj.matched(nom["f"], nom["zeta"], t, q), color="#b4b4b4", lw=1.2, zorder=2,
            label=f"IPC, nominal $E$, $h$ = 1 ms: f {nom['f']:.2f} Hz, $\\zeta$ {nom['zeta']:.3f}")
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("t (s)")
    ax.set_ylabel("sway (pixels)")
    ax.set_title("A Jello cube's sway: Genesis 1.5.0, FEM through the IPC coupler, 10 cells a side", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.text(0.5, 0.01,
             f"$E$ (Pa): Young's modulus from the video's frequency and the continuum, {d['E'] / 1e3:.1f} kPa at "
             f"$\\rho$ = {js.RHO:.0f} kg/m$^3$.\n"
             f"IPC has no damping parameter: its damping is backward Euler's, set by the step $h$ (ms); "
             f"$h^\\star$: the step at which it is the video's.\n"
             f"$E^\\star = E\\,S\\,T(h^\\star)$ = {d['S'] * d['T']:.4f} $E$, $S$ from the sway mode of the same mesh, "
             f"$T$ undamped backward Euler's: nothing fitted.",
             ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    clear_legends(fig)
    for e_ in ("png", "pdf"):
        fig.savefig(c.RESULTS / f"fig_jello_gs_ipc.{e_}", dpi=200)
    plt.close(fig)
    print("wrote", c.RESULTS / "fig_jello_gs_ipc.png")


def main():
    use_house_style()
    rows = [json.loads(p.read_text()) for p in sorted(ri.FLOW_PARTS.glob("*.json"))]
    left_out = [r for r in rows if not r["misfit"] < MISFIT_MAX]
    for r in left_out:
        print(f"left out: dx {r['dx'] * 1e3:.2f} mm, h {r['h'] * 1e3:g} ms, misfit {r['misfit']:.1e}")
    rows = [r for r in rows if r["misfit"] < MISFIT_MAX]
    if rows:
        flow_figure(rows)
    jello_figure()


if __name__ == "__main__":
    main()
