"""The Jello video against Genesis's four methods (genesis_jello.py), one figure each, as fig_jello.

Each run is a single damped sway; it is drawn with its own frequency and damping, and with the
video's amplitude and phase, which alone are fitted (plot_jello.matched).

  python plot_jello_genesis.py /data/datasets/vvt_jello_cube [arm ...]
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common as c  # noqa: E402
import genesis_jello as gj  # noqa: E402
from figures import INK, INK2, RENORM, clear_legends, style, use_house_style  # noqa: E402
from plot_jello import matched, video_sway  # noqa: E402

NAME = {"gs_fem_explicit": "FEM explicit", "gs_fem_implicit": "FEM implicit", "gs_mpm": "MPM", "gs_pbd": "PBD"}


def ms(h):
    return f"{h * 1e3:.3g} ms"


def labels(arm, r):
    """(title, renormalized label, nominal label, caption lines) of one arm."""
    E, h = r["E"], r["h"]
    head = (f"$E$ (Pa): Young's modulus from the video's frequency and the continuum, {E / 1e3:.1f} kPa at "
            f"$\\rho$ = {gj.RHO:.0f} kg/m$^3$.")
    if arm == "gs_fem_explicit":
        title = f"Genesis 1.5.0, FEM explicit, h = {ms(h)}, 10 cells a side"
        ren, nom = "renormalized $E^\\star$, $c$", "nominal $E$, $c$"
        cap = [head + f"  $c$ (1/s): the velocity's decay $e^{{-ch}}$ per step, $2\\zeta\\omega$ = "
                      f"{r['damping_nominal']:.2f} /s.",
               f"$E^\\star = E\\,S\\,T(h)$ = {r['E_ratio']:.4f} $E$ ($S$ from the sway mode of Genesis's mesh, "
               f"$T$ symplectic Euler's), $c$ unchanged.  $h$: half of $2/\\omega_{{max}}$ = {ms(r['h_max'])}."]
    elif arm == "gs_fem_implicit":
        title = f"Genesis 1.5.0, FEM implicit, h = {ms(h)}, 10 cells a side"
        ren, nom = "renormalized $E^\\star$, $\\beta^\\star$", "nominal $E$, $\\beta$"
        cap = [head + f"  $\\beta$ (s): damping in proportion to stiffness, $2\\zeta/\\omega$ = "
                      f"{r['damping_nominal'] * 1e3:.2f} ms.",
               f"$E^\\star = E\\,S\\,T(h)$ = {r['E_ratio']:.4f} $E$ ($S$ from the sway mode of Genesis's mesh, "
               f"$T$ backward Euler's),  $\\beta^\\star = \\beta - h + h^2\\beta\\omega^2/12$ = "
               f"{r['damping_renormalized'] * 1e3:.2f} ms."]
    elif arm == "gs_mpm":
        n = r["particles_per_side"]
        title = (f"Genesis 1.5.0, MPM, h = {ms(h)}, {r['dx'] * 1e3:g} mm cells, "
                 f"{n[0]} x {n[1]} x {n[2]} particles")
        ren, nom = "renormalized $E^\\star$", "nominal $E$"
        cap = [head + "  No damping parameter.",
               f"$E^\\star$ = {r['E_ratio']:.4f} $E$: the modulus at which MLS-MPM's step, linearized about rest, "
               f"sways at {gj.F_VIDEO:.2f} Hz.",
               f"$h$: the step at which that linearized step damps the sway by $\\zeta$ = {r['zeta_theory']:.4g}.  "
               "The base is held by a slab of fixed particles."]
    else:
        title = f"Genesis 1.5.0, PBD, h = {ms(h)}, 10 cells a side"
        ren, nom = "renormalized $E^\\star$", "nominal $E$"
        cap = [head + "  No damping parameter.",
               f"$\\alpha$ (m/N): an edge's compliance, nominal $1/(E\\,\\Delta x)$, $\\Delta x$ = "
               f"{r['dx_nominal'] * 1e3:.1f} mm; $E^\\star$ = {r['E_ratio']:.1f} $E$ means $\\alpha E/E^\\star$.  "
               f"No $E^\\star$ exists above $h$ = {ms(r['h_max'])}.",
               f"$E^\\star$: the compliance at which PBD's step, linearized about rest, sways at {gj.F_VIDEO:.2f} Hz;  "
               f"$h$: the step at which it damps the sway by $\\zeta$ = {r['zeta_theory']:.4g}."]
    return title, ren, nom, cap


def run_label(name, run):
    if not np.isfinite(run["f"]):
        return f"Genesis, {name}: unstable"
    return f"Genesis, {name}: f {run['f']:.2f} Hz, $\\zeta$ {run['zeta']:.3f}"


def figure(arm, t, q):
    r = json.loads(gj.out_path(arm).read_text())
    title, ren, nom, cap = labels(arm, r)
    fig, ax = plt.subplots(figsize=(10, 4.2 + 0.15 * (len(cap) - 2)))
    style(ax)
    ax.plot(t, q, "o", ms=1.6, color=INK, mec="none", zorder=1,
            label=f"video, 4000 fps: f {gj.F_VIDEO:.2f} Hz, $\\zeta$ {gj.ZETA_VIDEO:.3f}")
    for key, name, color, lw, z in (("renormalized", ren, RENORM, 1.6, 3), ("nominal", nom, "#b4b4b4", 1.2, 2)):
        run = r[key]
        if np.isfinite(run["f"]):
            ax.plot(t, matched(run["f"], run["zeta"], t, q), color=color, lw=lw, zorder=z, label=run_label(name, run))
        else:
            ax.plot([], [], color=color, lw=lw, label=run_label(name, run))
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("t (s)")
    ax.set_ylabel("sway (pixels)")
    ax.set_title(f"A Jello cube's sway: {title}", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.text(0.5, 0.01, "\n".join(cap), ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.02 + 0.033 * len(cap), 1, 1))
    clear_legends(fig)
    for e in ("png", "pdf"):
        fig.savefig(c.RESULTS / f"fig_jello_{arm}.{e}", dpi=200)
    plt.close(fig)
    print("wrote", c.RESULTS / f"fig_jello_{arm}.png")


def main(root, arms):
    use_house_style()
    t, q = video_sway(root)
    for arm in arms or [a for a in gj.ARMS if gj.out_path(a).exists()]:
        figure(arm, t, q)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
