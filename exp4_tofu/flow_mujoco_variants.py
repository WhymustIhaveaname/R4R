"""Experiment 4a's free vibration in MuJoCo's interpolated flexes: dof="trilinear" and "quadratic".

The 6 x 5 x 4 cm block of common.py floats free, as in mujoco_sim.py, but its displacement is
interpolated from one cell's nodes: 8 for trilinear, 27 for quadratic (cellcount 1 1 1; the
11 x 13 x 9 vertices at 5 mm only carry the interpolated surface).  There is no cell size to
refine, so the spatial factor S is one number per variant, from MuJoCo itself: its mass matrix
and the tangent of its elastic force at rest (jello_mujoco_variants.linearize) give the model's
modes at E0; the mode is the one whose interpolated field, read at the vertices, overlaps most
with the continuum's (common.MODE, 14.36 Hz), and S = (omega0 / omega_model)^2.

Each (variant, integrator, h) is then fitted as run_flow.py fits the full flex: E* and zeta*
such that the mode, projected in MuJoCo's own masses, has the continuum's frequency and
damping ratio.  The time factor and the damping counterterm are compared with theory.py's:
backward Euler for discrete, symplectic Euler with explicit damping for Euler.

  python flow_mujoco_variants.py          # writes results/flow_mj_interp.json
"""

import json
import time

import mujoco
import numpy as np
from scipy.linalg import eigh

import common as c
from jello_mujoco_variants import linearize, overlap, vertex_map

OUT = c.RESULTS / "flow_mj_interp.json"
VARIANTS = ("trilinear", "quadratic")
INTEGRATORS = ("discrete", "Euler")
H_DISCRETE = (0.25e-3, 0.5e-3, 1e-3, 2e-3, 3e-3, 4e-3)
EULER_FRACTIONS = (0.25, 0.5, 0.7, 0.85)          # of the stability limit at E* = S E0
VERT_SPACING = 0.005


def model_xml(variant, integrator, h):
    count = " ".join(str(int(round(s / VERT_SPACING)) + 1) for s in c.SIDES)
    sp = repr(float(VERT_SPACING))
    return f"""<mujoco>
  <option timestep="{float(h)!r}" gravity="0 0 0" integrator="{integrator}"/>
  <worldbody>
    <flexcomp name="block" type="grid" dim="3" dof="{variant}" count="{count}"
              spacing="{sp} {sp} {sp}" mass="{float(c.RHO * np.prod(c.SIDES))!r}" radius="0.0005">
      <contact contype="0" conaffinity="0" selfcollide="none"/>
      <elasticity young="{c.E0!r}" poisson="{c.NU!r}" damping="{c.damping_param("stiffness", c.ZETA0)!r}"/>
    </flexcomp>
  </worldbody>
</mujoco>"""


def spatial(variant):
    """The model's modes at E0: S, the chosen mode's overlap with the continuum's, the overlap of
    the best field the dofs can make, the node vector (scaled so that its field's projection on
    the continuum's shape at the vertices is that shape), the masses, w_max, and every mode's
    (f, overlap)."""
    m = mujoco.MjModel.from_xml_string(model_xml(variant, "Euler", 1e-3))
    M, K = linearize(m, eps=1e-9)
    w2, V = eigh(K, M)
    X0, B = vertex_map(m)
    phi = c.mode_shape(X0 - X0.mean(0))
    ovs = np.array([overlap((B @ V[:, j]).reshape(-1, 3), phi) for j in range(m.nv)])
    elastic = w2 > 1e-6 * w2.max()
    k = int(np.argmax(np.where(elastic, ovs, -1)))
    v = V[:, k]
    u = B @ v
    v = v * (phi.ravel() @ phi.ravel()) / (u @ phi.ravel())
    span = overlap(B @ np.linalg.lstsq(B, phi.ravel(), rcond=None)[0], phi)
    return {"S": c.omega0() ** 2 / w2[k], "overlap": ovs[k], "span_overlap": span, "vec": v,
            "M": M, "w_max": float(np.sqrt(w2.max())), "modes": list(zip(np.sqrt(np.abs(w2)) / 2 / np.pi, ovs)),
            "X_nodes": np.array([m.body_pos[b] for b in m.flex_nodebodyid])}


class MjInterpBlock:
    """mujoco_sim.MjBlock for an interpolated flex: the node dofs are the whole state."""

    def __init__(self, variant, integrator, h, sp):
        self.h = h
        self.m = mujoco.MjModel.from_xml_string(model_xml(variant, integrator, h))
        self.d = mujoco.MjData(self.m)
        assert self.m.nv == 3 * len(sp["X_nodes"]) and np.all(self.m.flex_nodebodyid > 0)
        self.k0 = self.m.flex_stiffness.copy()
        self.vec, M = sp["vec"], sp["M"]
        masses = np.diag(M)[::3]
        assert np.allclose(np.diag(M), np.repeat(masses, 3)) and np.allclose(M, np.diag(np.diag(M)))
        self.v0 = c.amplitude() * self.vec + c.seeded_noise(sp["X_nodes"], masses).ravel()
        self.w = M @ self.vec / (self.vec @ M @ self.vec)
        self.steps = c.sample_steps(h)

    def modal(self, E_ratio, zeta):
        m, d = self.m, self.d
        m.flex_stiffness[:] = self.k0 * E_ratio
        m.flex_damping[0] = c.damping_param("stiffness", zeta)
        mujoco.mj_resetData(m, d)
        d.qvel[:] = self.v0
        q = np.full(len(self.steps), np.nan)
        rest = np.full(len(self.steps), np.nan)
        j = 0
        for n in range(1, self.steps[-1] + 1):
            mujoco.mj_step(m, d)
            if n == self.steps[j]:
                u = d.qpos - m.qpos0
                q[j] = self.w @ u
                rest[j] = np.abs(u - q[j] * self.vec).max()
                if not np.isfinite(q[j]):
                    break
                j += 1
        if c.unstable(rest):
            q[:] = np.nan
        return q

    def measure(self, E_ratio, zeta):
        return c.measure(self.steps * self.h, self.modal(E_ratio, zeta))


SCHEME = {"discrete": "backward_euler_beta", "Euler": "symplectic_explicit_beta"}


def h_max_euler(sp):
    w = sp["w_max"] * np.sqrt(sp["S"])
    b = c.damping_param("stiffness", c.ZETA0)
    return (-2 * b + np.sqrt(4 * b * b + 16 / w**2)) / 2


def main():
    rows = []
    for variant in VARIANTS:
        t0 = time.time()
        sp = spatial(variant)
        print(f"\n== {variant}: S = E*/E at h -> 0 = {sp['S']:.5f}, overlap with the continuum's mode "
              f"{sp['overlap']:.4f} (the dofs' best field {sp['span_overlap']:.4f}), "
              f"w_max {sp['w_max']:.0f} rad/s at E0  [{time.time() - t0:.1f} s]", flush=True)
        print("   modes (Hz, overlap):", "  ".join(f"{f:.2f} ({o:.2f})" for f, o in sp["modes"][6:14]))
        for integ in INTEGRATORS:
            hs = H_DISCRETE if integ == "discrete" else [float(f"{f * h_max_euler(sp):.4g}") for f in EULER_FRACTIONS]
            if integ == "Euler":
                print(f"   Euler stable for h <= {h_max_euler(sp) * 1e3:.3f} ms at E* = S E0")
            for h in hs:
                t0 = time.time()
                e, z = c.time_prediction(SCHEME[integ], h)
                block = MjInterpBlock(variant, integ, h, sp)
                E, Z, misfit, nfev = c.fit(block.measure, [(np.log(sp["S"] * e), z)])
                if misfit > 1e-8:
                    E2, Z2, m2, n2 = c.fit(block.measure, [(np.log(sp["S"]), c.ZETA0)])
                    nfev += n2
                    if m2 < misfit:
                        E, Z, misfit = E2, Z2, m2
                sec = time.time() - t0
                print(f"   {integ:8s} h {h * 1e3:7.4f} ms  E*/E {E:.6f} (theory S T {sp['S'] * e:.6f}, "
                      f"{E / (sp['S'] * e) - 1:+.2e})  zeta* {Z:.6f} (theory {z:.6f})  misfit {misfit:.1e}"
                      f"  [{nfev} rollouts, {sec:.1f} s]", flush=True)
                rows.append({"variant": variant, "integrator": integ, "h": h, "E_ratio": E, "zeta": Z,
                             "misfit": misfit, "nfev": nfev, "S_theory": sp["S"], "time_theory": [e, z],
                             "overlap": sp["overlap"], "span_overlap": sp["span_overlap"],
                             "w_max": sp["w_max"], "seconds": sec})
    OUT.write_text(json.dumps(rows, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
