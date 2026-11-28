"""The Jello video: the cube's sway in the video, and MuJoCo's with and without the theory.

The video's sway is the largest component of the cube's motion (video_modes.py) once filtered to
6.5 to 9.5 Hz, from the release of the top-front pluck, in pixels at the point that moves most.
Each MuJoCo run (jello_sim.py) is a single damped sway; it is drawn with its own frequency and
damping, and with the video's amplitude and phase, which alone are fitted.

  python plot_jello.py /data/datasets/vvt_jello_cube
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common as c  # noqa: E402
import jello_sim as js  # noqa: E402
from figures import INK, INK2, RENORM, clear_legends, style, use_house_style  # noqa: E402

H = 1e-3


def video_sway(root):
    z = np.load(Path(root) / "top_front_pluck" / "motion.npz", allow_pickle=False)
    xy, dt, start = z["xy"], float(z["dt"]), int(z["start"])
    b, a = butter(4, [6.5, 9.5], btype="band", fs=1 / dt)
    y = filtfilt(b, a, xy - xy.mean(1, keepdims=True), axis=1)[:, start:]
    U, S, Vt = np.linalg.svd(y, full_matrices=False)
    trim = int(0.1 / dt)                                  # the filter's ends
    q = S[0] * Vt[0] * np.abs(U[:, 0]).max()              # pixels, at the point that moves most
    return np.arange(y.shape[1])[:-trim] * dt, q[:-trim]


def matched(f, zeta, t, q):
    """The simulation's sway, a exp(-zeta w t) sin(w_d t + p) with its own f and zeta, given the
    video's amplitude and phase by least squares."""
    w = 2 * np.pi * f
    env = np.exp(-zeta * w * t)
    wd = w * np.sqrt(1 - zeta**2)
    A = np.stack([env * np.sin(wd * t), env * np.cos(wd * t)], 1)
    return A @ np.linalg.lstsq(A, q, rcond=None)[0]


def main(root):
    use_house_style()
    t, q = video_sway(root)
    C = js.continuum_constant()
    E = js.RHO * (js.F_VIDEO / C) ** 2
    C_mj, X, shape = js.mujoco_sway_constant()
    S = (C / C_mj) ** 2
    w = 2 * np.pi * js.F_VIDEO
    T, beta_star = js.backward_euler(w, js.ZETA_VIDEO, H)
    dur = t[-1] + 0.01
    f0, z0, t0, q0 = js.run(E, 2 * js.ZETA_VIDEO / w, H, X, shape, dur, series=True)
    f1, z1, t1, q1 = js.run(E * S * T, beta_star, H, X, shape, dur, series=True)

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
    ax.set_title("A Jello cube's sway: MuJoCo 4.0.0, discrete, h = 1 ms, 10 cells a side", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.text(0.5, 0.01,
             f"$E$ (Pa): Young's modulus from the video's frequency and the continuum, {E / 1e3:.1f} kPa at "
             f"$\\rho$ = {js.RHO:.0f} kg/m$^3$.  $\\beta$ (s): damping in proportion to stiffness, "
             f"$2\\zeta/\\omega$ = {2 * js.ZETA_VIDEO / w * 1e3:.2f} ms.\n"
             f"$E^\\star = E\\,S\\,T(h)$ = {S * T:.4f} $E$ ($S$ from the sway mode of MuJoCo's mesh, $T$ backward Euler's),"
             f"  $\\beta^\\star = \\beta - h + h^2\\beta\\omega^2/12$ = {beta_star * 1e3:.2f} ms.",
             ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    clear_legends(fig)
    for e in ("png", "pdf"):
        fig.savefig(c.RESULTS / f"fig_jello.{e}", dpi=200)
    print("wrote", c.RESULTS / "fig_jello.png", f"f0 {f0:.4f} z0 {z0:.4f} f1 {f1:.4f} z1 {z1:.4f}")


if __name__ == "__main__":
    main(sys.argv[1])
