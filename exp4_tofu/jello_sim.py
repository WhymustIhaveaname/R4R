"""Step 2's test: MuJoCo given the parameters the theory renormalizes, on a cube it never saw.

The Jello cube of video_modes.py, 4.9 x 4.7 x 4.5 cm on a fixed base, is put in MuJoCo with
10 cells a side and the discrete integrator.  Its material comes from the video alone: the
lowest sway at f = 8.01 Hz with zeta = 0.015 (the top-front pluck), and the continuum's sway
constant C (jello_model.py, Richardson over two meshes), give E / rho = (f / C)^2; rho = 1080
kg/m^3 is a Jell-O's.  MuJoCo is then run twice:

  nominal        E and beta = 2 zeta / w, as the material is
  renormalized   E* = E S T(h), beta* (theory.py's backward Euler), with S = (C / C_P1)^2 from
                 the P1 sway on MuJoCo's own mesh with its equal masses: nothing fitted

and its sway's frequency and damping are read as common.measure reads them.

  python jello_sim.py
"""

import mujoco
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

import jello_model as jm
import p1

F_VIDEO, ZETA_VIDEO = 8.01, 0.015       # Hz, the top-front pluck's sway (video_modes.py)
RHO = 1080.0
CELLS = 10
NU = 0.45


def continuum_constant():
    """The sway's f at E / rho = 1, Richardson-extrapolated over two meshes with masses lumped by
    volume, whose error is of second order."""
    a = min(f for f, k in jm.modes((16, 16, 15), NU) if k.startswith("sway"))
    b = min(f for f, k in jm.modes((24, 24, 22), NU) if k.startswith("sway"))
    return b + (b - a) / ((24 / 16) ** 2 - 1)


def xml(E, beta, h):
    count = " ".join(str(CELLS + 1) for _ in range(3))
    spacing = " ".join(repr(float(s / CELLS)) for s in jm.SIDES)
    return f"""<mujoco>
  <option timestep="{float(h)!r}" gravity="0 0 0" integrator="discrete"/>
  <worldbody>
    <flexcomp name="cube" type="grid" dim="3" dof="full" count="{count}" spacing="{spacing}"
              mass="{float(RHO * np.prod(jm.SIDES))!r}" radius="0.0005">
      <contact contype="0" conaffinity="0" selfcollide="none"/>
      <elasticity young="{float(E)!r}" poisson="{NU!r}" damping="{float(beta)!r}"/>
      <pin gridrange="0 0 0 {CELLS} {CELLS} 0"/>
    </flexcomp>
  </worldbody>
</mujoco>"""


def mujoco_sway_constant():
    """The P1 sway's f at E / rho = 1 on MuJoCo's mesh, with MuJoCo's equal vertex masses."""
    X, tets = p1.mesh_from_mujoco(None, xml(1.0, 1e-3, 1e-3))
    X = X - X.min(0)
    K, _ = p1.assemble(X, tets, 1.0, NU)
    m = np.full(len(X), np.prod(jm.SIDES) / len(X))         # rho = 1 shared equally, pinned ones too
    free = np.flatnonzero(np.repeat(X[:, 2] > 1e-9, 3))
    w2, V = eigsh(K[free][:, free], k=4, M=diags(np.repeat(m, 3)[free]), sigma=0.0, which="LM")
    w = np.sqrt(np.sort(w2))
    shape = np.zeros((len(X), 3))
    shape.ravel()[free] = V[:, np.argmin(w2)]
    return w[0] / (2 * np.pi), X, shape


def backward_euler(w, zeta, h):
    """theory.py's counterterms of backward Euler with beta K damping, for one mode."""
    b = 2 * zeta / w
    return (1 + h * b * w**2 / 2 + h**2 * w**2 * (2 * b**2 * w**2 - 1) / 12,
            b - h + h**2 * b * w**2 / 12)


def measure_at(t, q, w0):
    """common.measure, started from the frequency w0: (w_n, zeta) of a q(t) = a exp(-zeta w t)
    sin(w_d t + p), amplitude and phase free."""
    from scipy.optimize import curve_fit

    def model(t, a, s, wd, ph):
        return a * np.exp(-s * t) * np.sin(wd * t + ph)
    p, _ = curve_fit(model, t, q, p0=[np.abs(q).max(), ZETA_VIDEO * w0, w0, 0.0], maxfev=20000)
    a, s, wd, ph = p
    wn = np.hypot(s, wd)
    return float(wn), float(s / wn)


def run(E, beta, h, X, shape, duration=1.5, series=False):
    """(f_n Hz, zeta) of MuJoCo's cube started in its P1 sway; with series, also (t, q)."""
    m = mujoco.MjModel.from_xml_string(xml(E, beta, h))
    d = mujoco.MjData(m)
    vb = m.flex_vertbodyid
    free = np.flatnonzero(m.body_dofnum[vb] > 0)
    dof = m.body_dofadr[vb[free]]
    qadr = np.stack([m.jnt_qposadr[m.dof_jntid[dof + k]] for k in range(3)], 1)
    s = shape[free]
    for k in range(3):
        d.qvel[dof + k] = 1e-3 * s[:, k] / np.abs(s).max()
    steps = np.arange(1, int(duration / h) + 1)
    q = np.empty(len(steps))
    w = s / np.sum(s * s)
    for j in range(len(steps)):
        mujoco.mj_step(m, d)
        q[j] = np.sum(w * (d.qpos[qadr] - m.qpos0[qadr]))
    wn, z = measure_at(steps * h, q, 2 * np.pi * F_VIDEO)
    if series:
        return wn / (2 * np.pi), z, steps * h, q
    return wn / (2 * np.pi), z


def main():
    C = continuum_constant()
    E = RHO * (F_VIDEO / C) ** 2
    C_mj, X, shape = mujoco_sway_constant()
    S = (C / C_mj) ** 2
    w = 2 * np.pi * F_VIDEO
    print(f"continuum sway constant {C:.4f} Hz  ->  E = {E / 1e3:.2f} kPa at rho {RHO:g}")
    print(f"MuJoCo's P1 sway constant {C_mj:.4f} Hz  ->  S = {S:.4f}")
    print(f"video: f {F_VIDEO:.3f} Hz, zeta {ZETA_VIDEO:.4f}")
    for h in (2.5e-4, 1e-3, 2e-3):
        T, beta_star = backward_euler(w, ZETA_VIDEO, h)
        f0, z0 = run(E, 2 * ZETA_VIDEO / w, h, X, shape)
        f1, z1 = run(E * S * T, beta_star, h, X, shape)
        print(f"h {h * 1e3:.2f} ms  nominal: f {f0:.3f} Hz ({f0 / F_VIDEO - 1:+.2%}), zeta {z0:.4f}"
              f"   renormalized (E* = {S * T:.4f} E, beta* = {beta_star / (2 * ZETA_VIDEO / w):.3f} beta):"
              f" f {f1:.3f} Hz ({f1 / F_VIDEO - 1:+.2%}), zeta {z1:.4f}")


if __name__ == "__main__":
    main()
