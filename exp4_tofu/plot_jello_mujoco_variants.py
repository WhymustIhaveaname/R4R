"""Figures for MuJoCo's other flex variants.

fig_jello_mj_{euler,trilinear,quadratic}:  as fig_jello.png (plot_jello.py), one per variant of
    jello_mujoco_variants.py: the video's sway, and MuJoCo's run nominal and renormalized.
fig_flow_mj_trilinear_quadratic:  the tofu block's free vibration in the two interpolated flexes
    (flow_mujoco_variants.py, results/flow_mj_interp.json): E* and zeta* against h, and the
    spatial factor S of each variant from its linearization.

  python plot_jello_mujoco_variants.py /data/datasets/vvt_jello_cube
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common as c  # noqa: E402
import jello_mujoco_variants as jv  # noqa: E402
import jello_sim as js  # noqa: E402
from figures import INK, INK2, RENORM, clear_legends, style, use_house_style  # noqa: E402
from plot_jello import matched, video_sway  # noqa: E402

TITLE = {"euler": "Euler (symplectic, damping explicit), h = {h}, 10 cells a side",
         "trilinear": "discrete, trilinear (8 nodes), h = {h}",
         "quadratic": "discrete, quadratic (27 nodes), h = {h}"}


def h_text(h):
    return f"{h * 1e3:g} ms"


def jello_figure(variant, t, q):
    s = jv.Setup(variant)
    h = jv.H_FIGURE[variant]
    dur = t[-1] + 0.01
    (f0, z0, _, _), (f1, z1, _, _), T, beta_star = s.pair(h, dur, series=True)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    style(ax)
    ax.plot(t, q, "o", ms=1.6, color=INK, mec="none", zorder=1,
            label=f"video, 4000 fps: f {js.F_VIDEO:.2f} Hz, $\\zeta$ {js.ZETA_VIDEO:.3f}")
    ax.plot(t, matched(f1, z1, t, q), color=RENORM, lw=1.6, zorder=3,
            label=f"MuJoCo, renormalized $E^\\star$, $\\beta^\\star$: f {f1:.2f} Hz, $\\zeta$ {z1:.3f}")
    ax.plot(t, matched(f0, z0, t, q), color="#b4b4b4", lw=1.2, zorder=2,
            label=f"MuJoCo, nominal $E$, $\\beta$: f {f0:.2f} Hz, $\\zeta$ {z0:.3f}")
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("t (s)")
    ax.set_ylabel("sway (pixels)")
    ax.set_title(f"A Jello cube's sway: MuJoCo 4.0.0, {TITLE[variant].format(h=h_text(h))}", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    first = (f"$E$ (Pa): Young's modulus from the video's frequency and the continuum, {s.E / 1e3:.1f} kPa at "
             f"$\\rho$ = {js.RHO:.0f} kg/m$^3$.  $\\beta$ (s): damping in proportion to stiffness, "
             f"$2\\zeta/\\omega$ = {s.beta * 1e3:.2f} ms.")
    if variant == "euler":
        second = (f"$E^\\star = E\\,S\\,T(h)$ = {s.S * T:.4f} $E$ ($S$ from the sway mode of MuJoCo's mesh, "
                  f"$T$ symplectic Euler's),  $\\beta^\\star = \\beta + h^2\\beta\\omega^2/12$ = {beta_star * 1e3:.4f} ms.")
        hm = jv.h_max_euler(jv.w_max_full(s.E * s.S * T), beta_star)
        third = (f"\nEuler is stable while $\\omega_{{max}}^2 h (h + 2\\beta^\\star) \\leq 4$, $\\omega_{{max}}$ "
                 f"(rad/s) the mesh's highest frequency at $E^\\star$: for $h \\leq$ {hm * 1e3:.3f} ms.")
    else:
        n = 8 if variant == "trilinear" else 27
        second = (f"$E^\\star = E\\,S\\,T(h)$ = {s.S * T:.4f} $E$ ($S$ from the sway mode of MuJoCo's {n} nodes, "
                  f"$T$ backward Euler's),  $\\beta^\\star = \\beta - h + h^2\\beta\\omega^2/12$ = {beta_star * 1e3:.2f} ms.")
        ft, fc = s.twists(s.E * s.S * T)
        third = (f"\nThe {n} nodes' sway overlaps the sway of 10 P1 cells a side by {s.overlap:.3f} "
                 f"(the best field of the {n} nodes by {s.span_overlap:.3f}).\n"
                 f"At $E^\\star$ their twist is at {ft:.1f} Hz, the continuum's at {fc:.1f} Hz.")
    fig.text(0.5, 0.01, first + "\n" + second + third, ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.08 if not third else 0.14, 1, 1))
    clear_legends(fig)
    for e in ("png", "pdf"):
        fig.savefig(c.RESULTS / f"fig_jello_mj_{variant}.{e}", dpi=200)
    plt.close(fig)
    print("wrote", c.RESULTS / f"fig_jello_mj_{variant}.png",
          f"h {h} S {s.S:.5f} T {T:.6f} beta* {beta_star:.6e} f0 {f0:.4f} z0 {z0:.4f} f1 {f1:.4f} z1 {z1:.4f}",
          flush=True)


VARIANT_COLOR = {"trilinear": "#2a78d6", "quadratic": "#eb6834"}
MARKER = {"discrete": "o", "Euler": "s"}
SCHEME = {"discrete": "backward_euler_beta", "Euler": "symplectic_explicit_beta"}


def flow_figure():
    rows = json.loads((c.RESULTS / "flow_mj_interp.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    for ax in axes:
        style(ax)
    hs = np.linspace(0, 4.2e-3, 200)
    for integ, ls_h in (("discrete", hs), ("Euler", hs[hs <= 0.8e-3])):
        th = np.array([c.time_prediction(SCHEME[integ], h) for h in ls_h])
        axes[0].plot(ls_h * 1e3, 100 * (th[:, 0] - 1), color=INK, lw=1.3, zorder=1)
        axes[1].plot(ls_h * 1e3, 100 * (th[:, 1] / c.ZETA0 - 1), color=INK, lw=1.3, zorder=1)
    for ax, x, y, name in ((axes[0], 3.2, 1.75, "discrete"), (axes[0], 0.88, -0.78, "Euler"),
                           (axes[1], 2.6, -95, "discrete"), (axes[1], 0.88, -9, "Euler")):
        ax.text(x, y, name, color=INK2, fontsize=8.5)
    for r in rows:
        col, mk = VARIANT_COLOR[r["variant"]], MARKER[r["integrator"]]
        size = 9 if r["variant"] == "trilinear" else 5
        axes[0].plot(r["h"] * 1e3, 100 * (r["E_ratio"] / r["S_theory"] - 1), mk, ms=size, color=col,
                     mec="white", mew=0.6, zorder=3)
        axes[1].plot(r["h"] * 1e3, 100 * (r["zeta"] / c.ZETA0 - 1), mk, ms=size, color=col,
                     mec="white", mew=0.6, zorder=3)
    for ax in axes[:2]:
        ax.set_xlim(0, 4.2)
        ax.set_xlabel("$h$ (ms)")
    axes[0].set_ylabel(r"$E^\star / E^\star_{h \to 0} - 1$ (%)")
    axes[1].set_ylabel(r"$\zeta^\star / \zeta - 1$ (%)")
    handles = [Line2D([], [], color=INK, lw=1.3, label="theory")]
    handles += [Line2D([], [], ls="none", marker="o", ms=9 if v == "trilinear" else 5, color=VARIANT_COLOR[v],
                       mec="white", label=v) for v in VARIANT_COLOR]
    handles += [Line2D([], [], ls="none", marker=MARKER[i], ms=6, color=INK2, mec="white", label=i)
                for i in MARKER]
    axes[0].legend(handles=handles, frameon=False, fontsize=8.5, loc="upper left")

    ax = axes[2]
    names = list(VARIANT_COLOR)
    worst = 0.0
    for k, v in enumerate(names):
        sel = [r for r in rows if r["variant"] == v]
        S = sel[0]["S_theory"]
        ax.plot(k, S, "o", ms=14, mfc="none", mec=INK, mew=1.2, zorder=2,
                label="linearized MuJoCo, modal analysis" if k == 0 else None)
        e = np.array([r["E_ratio"] / r["time_theory"][0] for r in sel])
        worst = max(worst, np.abs(e / S - 1).max())
        ax.plot(np.full(len(e), k), e, "o", ms=5, color=VARIANT_COLOR[v], mec="white", mew=0.6, zorder=3)
        ax.annotate(f"$S$ = {S:.3f}\noverlap {sel[0]['overlap']:.3f}", (k, S), xytext=(14, -4),
                    textcoords="offset points", fontsize=8.5, color=INK2, va="center")
    ax.plot([], [], "o", ms=5, color=INK2, mec="white",
            label=f"{len(rows)} runs: $E^\\star / (E\\,T(h))$,\nwithin {100 * worst:.2g}% of it")
    ax.set_yscale("log")
    ax.set_xlim(-0.5, 1.9)
    ax.set_ylim(0.4, 20)
    ax.set_xticks(range(len(names)), names)
    ax.set_ylabel(r"$E^\star_{h \to 0} / E = S$")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.suptitle("Free vibration of the tofu block: MuJoCo 4.0.0, interpolated flexes (one cell)", fontsize=11)
    cap = (r"$E^\star$ (Pa): the Young's modulus given to the simulator so that its mode closest in shape to the "
           r"continuum's lowest elastic mode has that mode's frequency, 14.4 Hz at $E$ = 11.3 kPa." "\n"
           r"$E^\star_{h \to 0} = S\,E$: $E^\star$ as $h \to 0$, from the model's own modes (MuJoCo's mass matrix "
           r"and the tangent of its elastic force)." "\n"
           r"$\zeta^\star$ (dimensionless): the damping ratio given to the simulator so that the mode decays as the "
           r"continuum's, $\zeta$ = 0.10.  $T(h)$ (dimensionless): the theory's factor $E^\star / E^\star_{h \to 0}$." "\n"
           r"Overlap: of the model's mode, read at the vertices, with the continuum's.  Euler's steps are 0.25 to "
           r"0.85 of its stability limit.")
    fig.text(0.5, 0.01, cap, ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.15, 1, 0.96))
    for e in ("png", "pdf"):
        fig.savefig(c.RESULTS / f"fig_flow_mj_trilinear_quadratic.{e}", dpi=200)
    plt.close(fig)
    print("wrote", c.RESULTS / "fig_flow_mj_trilinear_quadratic.png")


def main(root):
    use_house_style()
    flow_figure()
    t, q = video_sway(root)
    for variant in sys.argv[2:] or ("trilinear", "quadratic", "euler"):
        jello_figure(variant, t, q)


if __name__ == "__main__":
    main(sys.argv[1])
