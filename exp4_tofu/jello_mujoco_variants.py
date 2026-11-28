"""MuJoCo's other flex variants on the Jello cube, given the parameters the theory renormalizes.

jello_sim.py tests the discrete integrator on 10 cells a side.  Here, on the same cube with the
same material from the video (f = 8.01 Hz, zeta = 0.015, rho = 1080, nu = 0.45, E from the
continuum's sway constant C), three more variants:

  euler       dof="full", the same 10-cell mesh, integrator="Euler": symplectic Euler with the
              beta K damping taken explicitly.  S is jello_sim's (P1, equal vertex masses);
              T and beta* are symplectic Euler's; h must keep w_max^2 h (h + 2 beta*) <= 4.
  trilinear   dof="trilinear": 8 nodes for the whole cube, the base's 4 pinned; discrete.
  quadratic   dof="quadratic": 27 nodes, the base's 9 pinned; discrete.

For the interpolated two, S is not the P1 one.  MuJoCo's own mass matrix and the tangent of its
elastic force at rest (linearize) give the model's modes at E / rho = 1; the sway is the mode
with the largest overlap with the P1 sway on the 10-cell mesh, read at the flex's vertices, and
S = (C / C_model)^2.  T and beta* are backward Euler's, as for dof="full".  Each run starts in
the model's own sway, as jello_sim's does, and is read as jello_sim.measure_at reads it.  The
model's twist about the vertical, at E*, is set beside the continuum's (the video's other mode),
since an E* that fixes one mode need not fix the next.

  python jello_mujoco_variants.py          # the table; writes results/jello_mj_variants.json
"""

import json
import time
from functools import lru_cache

import mujoco
import numpy as np
from scipy.linalg import eigh

import common as c
import jello_model as jm
import jello_sim as js

ORDER = {"trilinear": 1, "quadratic": 2}
LABEL = {"euler": "Euler, 10 cells a side", "trilinear": "discrete, trilinear (8 nodes)",
         "quadratic": "discrete, quadratic (27 nodes)"}
H_RUNS = {"euler": (2.5e-5, 5e-5, 1e-4), "trilinear": (2.5e-4, 1e-3, 2e-3, 5e-3),
          "quadratic": (2.5e-4, 1e-3, 2e-3, 5e-3)}
H_FIGURE = {"euler": 1e-4, "trilinear": 1e-3, "quadratic": 1e-3}
OUT = c.RESULTS / "jello_mj_variants.json"


def xml(variant, E, beta, h, rho=js.RHO):
    """The cube of jello_sim.xml: the same 11 x 11 x 11 vertices, in the variant's dofs."""
    if variant == "euler":
        return js.xml(E, beta, h).replace('integrator="discrete"', 'integrator="Euler"')
    k = ORDER[variant]
    count = " ".join(str(js.CELLS + 1) for _ in range(3))
    spacing = " ".join(repr(float(s / js.CELLS)) for s in jm.SIDES)
    return f"""<mujoco>
  <option timestep="{float(h)!r}" gravity="0 0 0" integrator="discrete"/>
  <worldbody>
    <flexcomp name="cube" type="grid" dim="3" dof="{variant}" count="{count}" spacing="{spacing}"
              mass="{float(rho * np.prod(jm.SIDES))!r}" radius="0.0005">
      <contact contype="0" conaffinity="0" selfcollide="none"/>
      <elasticity young="{float(E)!r}" poisson="{js.NU!r}" damping="{float(beta)!r}"/>
      <pin gridrange="0 0 0 {k} {k} 0"/>
    </flexcomp>
  </worldbody>
</mujoco>"""


def linearize(m, eps=1e-8):
    """(M, K) of a model at rest: MuJoCo's mass matrix, and K = -d qfrc_passive / d qpos by
    central differences (the elastic force is geometrically nonlinear; its tangent at rest is
    what a small vibration sees)."""
    d = mujoco.MjData(m)
    K = np.zeros((m.nv, m.nv))
    for j in range(m.nv):
        for s in (1, -1):
            d.qpos[:] = m.qpos0
            d.qpos[j] += s * eps
            d.qvel[:] = 0
            mujoco.mj_forward(m, d)
            K[:, j] -= s * d.qfrc_passive / (2 * eps)
    d.qpos[:] = m.qpos0
    mujoco.mj_forward(m, d)
    M = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, M)
    return M, (K + K.T) / 2


def vertex_map(m, eps=1e-6):
    """B with flexvert displacement = B dq: the interpolation from the nodes, which is linear."""
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    X0 = d.flexvert_xpos.copy()
    B = np.zeros((X0.size, m.nv))
    for j in range(m.nv):
        d.qpos[:] = m.qpos0
        d.qpos[j] += eps
        mujoco.mj_forward(m, d)
        B[:, j] = (d.flexvert_xpos - X0).ravel() / eps
    return X0, B


def modes(m):
    """(f Hz, node vectors, M) of a model at rest, lowest first."""
    M, K = linearize(m)
    w2, V = eigh(K, M)
    return np.sqrt(np.abs(w2)) / (2 * np.pi) * np.sign(w2), V, M


def overlap(a, b):
    """|cos| of the angle between two displacement fields at the same vertices, equal weights."""
    return float(abs(a.ravel() @ b.ravel()) / np.linalg.norm(a) / np.linalg.norm(b))


@lru_cache(maxsize=None)
def continuum_modes():
    """(sway, twist): the continuum's f at E / rho = 1, each Richardson-extrapolated over two
    meshes as jello_sim.continuum_constant does the sway's."""
    a, b = jm.modes((16, 16, 15), js.NU), jm.modes((24, 24, 22), js.NU)
    out = []
    for kind in ("sway", "twist"):
        fa = min(f for f, k in a if k.startswith(kind))
        fb = min(f for f, k in b if k.startswith(kind))
        out.append(fb + (fb - fa) / ((24 / 16) ** 2 - 1))
    return tuple(out)


def continuum_constant():
    return continuum_modes()[0]


def twist_constant(m, f, V):
    """The f of a model's lowest twist about the vertical (jello_model.modes's test, at the vertices)."""
    X0, B = vertex_map(m)
    r = X0[:, :2] - X0[:, :2].mean(0)
    for j in np.argsort(f):
        u = (B @ V[:, j]).reshape(-1, 3)
        tw = np.sum(r[:, 0] * u[:, 1] - r[:, 1] * u[:, 0]) / np.sqrt(np.sum(r**2) * np.sum(u[:, :2]**2))
        if abs(tw) > 0.8:
            return float(f[j])
    return np.nan


@lru_cache(maxsize=None)
def p1_sway():
    return js.mujoco_sway_constant()


def interp_sway_constant(variant):
    """(C_model, overlap, node vector, the model's f, their overlaps, the span's overlap, the
    twist's f): the sway's f at E / rho = 1, chosen by its overlap with the P1 sway
    (jello_sim.mujoco_sway_constant) on the same vertices."""
    m = mujoco.MjModel.from_xml_string(xml(variant, 1.0, 1e-3, 1e-3, rho=1.0))
    f, V, M = modes(m)
    _, Xp, shape = p1_sway()
    X0, B = vertex_map(m)
    # the P1 mesh is shifted to start at 0; the flex is centred: match the vertices by position
    X0s = X0 - X0.min(0)
    order_p = np.lexsort(np.round(Xp / 1e-5).T)
    order_m = np.lexsort(np.round(X0s / 1e-5).T)
    assert np.allclose(Xp[order_p], X0s[order_m], atol=1e-9)
    ref = np.zeros_like(shape)
    ref[order_m] = shape[order_p]
    ovs = np.array([overlap((B @ V[:, j]).reshape(-1, 3), ref) for j in range(V.shape[1])])
    k = int(np.argmax(ovs))
    # the P1 sway leans a little toward the other sway (the tetrahedra's diagonal breaks the
    # mirror symmetry), so one mode's overlap understates what the dofs can represent: the
    # best approximation of the P1 sway by any node displacement says that
    fit = B @ np.linalg.lstsq(B, ref.ravel(), rcond=None)[0]
    return float(f[k]), float(ovs[k]), V[:, k], f, ovs, overlap(fit, ref), twist_constant(m, f, V)


def symplectic_euler(w, zeta, h):
    """theory.py's counterterms of MuJoCo's Euler: symplectic Euler, beta K damping explicit."""
    b = 2 * zeta / w
    return (1 - h * b * w**2 / 2 + h**2 * w**2 * (2 * b**2 * w**2 - 1) / 12,
            b + h**2 * b * w**2 / 12)


def w_max_full(E_ratio):
    """The highest frequency (rad/s) of the 10-cell P1 cube with equal masses, at E = E_ratio Pa
    and rho = RHO: MuJoCo's Euler is stable while w_max^2 h (h + 2 beta) <= 4."""
    import p1
    from scipy.sparse import diags
    from scipy.sparse.linalg import eigsh
    X, tets = p1.mesh_from_mujoco(None, js.xml(1.0, 1e-3, 1e-3))
    K, _ = p1.assemble(X, tets, E_ratio, js.NU)
    m = np.full(len(X), js.RHO * np.prod(jm.SIDES) / len(X))
    free = np.flatnonzero(np.repeat(X[:, 2] > X[:, 2].min() + 1e-9, 3))
    w2 = eigsh(K[free][:, free], k=1, M=diags(np.repeat(m, 3)[free]), which="LA",
               return_eigenvectors=False)[0]
    return float(np.sqrt(w2))


def h_max_euler(w_max, beta):
    return (-2 * beta + np.sqrt(4 * beta**2 + 16 / w_max**2)) / 2


def w_unstable_discrete(h, beta):
    """Backward Euler with beta K damping, for a mode w: x' = x + h v', v' = v - h w^2 (x' + beta v').
    Its step matrix has det 1 / D and trace (2 + h beta w^2) / D, D = 1 + h (h + beta) w^2: with
    beta < -h/2 (beta* = beta - h + ... is, once h passes about 2 beta), every mode above
    w^2 = 4 / (h (-2 beta - h)) grows.  Returns that w (rad/s), or inf."""
    return np.sqrt(4 / (h * (-2 * beta - h))) if -2 * beta - h > 0 else np.inf


def run_interp(variant, E, beta, h, vec, duration=1.5, series=False):
    """(f_n Hz, zeta) of an interpolated cube started in its own sway (node vector vec)."""
    m = mujoco.MjModel.from_xml_string(xml(variant, E, beta, h))
    d = mujoco.MjData(m)
    M = np.zeros((m.nv, m.nv))
    mujoco.mj_forward(m, d)
    mujoco.mj_fullM(m, d, M)
    v = 1e-3 * vec / np.abs(vec).max()
    d.qvel[:] = v
    w = M @ vec / (vec @ M @ vec)
    steps = np.arange(1, int(duration / h) + 1)
    q = np.empty(len(steps))
    for j in range(len(steps)):
        mujoco.mj_step(m, d)
        q[j] = w @ (d.qpos - m.qpos0)
    wn, z = js.measure_at(steps * h, q, 2 * np.pi * js.F_VIDEO)
    if series:
        return wn / (2 * np.pi), z, steps * h, q
    return wn / (2 * np.pi), z


def run_euler(E, beta, h, X, shape, duration=1.5, series=False):
    """jello_sim.run, copied, on the Euler model: started in the P1 sway, read on its shape."""
    m = mujoco.MjModel.from_xml_string(xml("euler", E, beta, h))
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
    if not np.all(np.isfinite(q)):
        return (np.nan, np.nan, steps * h, q) if series else (np.nan, np.nan)
    wn, z = js.measure_at(steps * h, q, 2 * np.pi * js.F_VIDEO)
    if series:
        return wn / (2 * np.pi), z, steps * h, q
    return wn / (2 * np.pi), z


class Setup:
    """Everything a variant's run needs, with nothing fitted: E, beta, S, the start."""

    def __init__(self, variant):
        self.variant = variant
        self.C = continuum_constant()
        self.E = js.RHO * (js.F_VIDEO / self.C) ** 2
        self.w = 2 * np.pi * js.F_VIDEO
        self.beta = 2 * js.ZETA_VIDEO / self.w
        if variant == "euler":
            self.C_model, self.X, self.shape = p1_sway()
            self.overlap = self.span_overlap = 1.0
            self.C_model_twist = np.nan
        else:
            (self.C_model, self.overlap, self.vec, self.f_all, self.ov_all,
             self.span_overlap, self.C_model_twist) = interp_sway_constant(variant)
        self.C_twist = continuum_modes()[1]
        self.S = (self.C / self.C_model) ** 2

    def twists(self, E_star):
        """(the model's twist at E*, the continuum's at E), Hz."""
        return self.C_model_twist * np.sqrt(E_star / js.RHO), self.C_twist * np.sqrt(self.E / js.RHO)

    def counterterms(self, h):
        if self.variant == "euler":
            return symplectic_euler(self.w, js.ZETA_VIDEO, h)
        return js.backward_euler(self.w, js.ZETA_VIDEO, h)

    def run(self, E, beta, h, duration=1.5, series=False):
        if self.variant == "euler":
            return run_euler(E, beta, h, self.X, self.shape, duration, series)
        return run_interp(self.variant, E, beta, h, self.vec, duration, series)

    def pair(self, h, duration=1.5, series=False):
        """The nominal run (E, beta) and the renormalized one (E S T(h), beta*)."""
        T, beta_star = self.counterterms(h)
        return (self.run(self.E, self.beta, h, duration, series),
                self.run(self.E * self.S * T, beta_star, h, duration, series), T, beta_star)


def main():
    rows = []
    for variant in ("euler", "trilinear", "quadratic"):
        s = Setup(variant)
        print(f"\n== {LABEL[variant]}:  C {s.C:.4f}  C_model {s.C_model:.4f}  S {s.S:.4f}"
              f"  overlap with the P1 sway {s.overlap:.4f} (the dofs' span {s.span_overlap:.4f})", flush=True)
        if variant != "euler":
            print("   model modes at E/rho = 1 (Hz, overlap):",
                  "  ".join(f"{f:.3f} ({o:.2f})" for f, o in zip(s.f_all[:6], s.ov_all[:6])))
            wmax1 = 2 * np.pi * s.f_all.max()             # at E / rho = 1
            ft, fc = s.twists(s.E * s.S)
            print(f"   twist: the model's at E* = S E {ft:.3f} Hz, the continuum's at E {fc:.3f} Hz")
        else:
            wm = w_max_full(s.E * s.S)
            print(f"   w_max at E* = S E: {wm:.0f} rad/s;  stable for h <= "
                  f"{h_max_euler(wm, s.beta) * 1e3:.4f} ms at beta", flush=True)
        for h in H_RUNS[variant]:
            t0 = time.time()
            (f0, z0), (f1, z1), T, bs = s.pair(h)
            sec = time.time() - t0
            wm = (w_max_full(s.E * s.S * T) if variant == "euler"
                  else wmax1 * np.sqrt(s.E * s.S * T / js.RHO))
            wu = np.inf if variant == "euler" else w_unstable_discrete(h, bs)
            if variant != "euler":
                print(f"   h {h * 1e3:.3f} ms  w_max at E* {wm:.0f} rad/s; backward Euler at beta* grows "
                      f"above w = {wu:.0f} rad/s: {'UNSTABLE' if wm > wu else 'stable'}")
            print(f"   h {h * 1e3:.3f} ms  nominal: f {f0:.4f} Hz ({f0 / js.F_VIDEO - 1:+.2%}), zeta {z0:.4f}"
                  f"   renormalized (E* = {s.S * T:.4f} E, beta* = {bs * 1e3:.4f} ms = "
                  f"{bs / s.beta:.3f} beta): f {f1:.4f} Hz ({f1 / js.F_VIDEO - 1:+.3%}), zeta {z1:.5f}"
                  f"   [{sec:.1f} s for both]", flush=True)
            rows.append({"variant": variant, "h": h, "C": s.C, "C_model": s.C_model, "S": s.S,
                         "overlap": s.overlap, "span_overlap": s.span_overlap, "E": s.E, "beta": s.beta,
                         "T": T, "E_ratio": s.S * T, "beta_star": bs, "twist_Hz": list(s.twists(s.E * s.S * T)),
                         "w_max": wm, "w_unstable": wu, "nominal": [f0, z0], "renormalized": [f1, z1],
                         "seconds_both": sec})
    OUT.write_text(json.dumps(rows, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
