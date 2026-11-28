"""The block in MuJoCo 4.0.0: a 3D flexcomp grid with dof="full".

Each grid cell is cut into six tetrahedra along one body diagonal; the elements are linear and
the material is St. Venant-Kirchhoff, with stiffness-proportional damping beta.  MuJoCo gives
every vertex the same mass, total / count.  The stiffness, precomputed at compile time, is
linear in E, so a rollout rescales it instead of recompiling.  The block starts in its own
discrete version of the mode (p1.mode_vector), with MuJoCo's equal masses.  The damping must be nonzero at
compile time, or MuJoCo allocates no room for its Jacobian and a later nonzero value crashes.
"""

import mujoco
import numpy as np

import common as c
import p1

INTEGRATOR = {"mj_euler": "Euler", "mj_discrete": "discrete"}


def model_xml(dx, integrator, h):
    count = " ".join(str(int(round(s / dx)) + 1) for s in c.SIDES)
    return f"""<mujoco>
  <option timestep="{float(h)!r}" gravity="0 0 0" integrator="{integrator}"/>
  <worldbody>
    <flexcomp name="block" type="grid" dim="3" dof="full" count="{count}"
              spacing="{float(dx)!r} {float(dx)!r} {float(dx)!r}" mass="{float(c.RHO * np.prod(c.SIDES))!r}" radius="0.0005">
      <contact contype="0" conaffinity="0" selfcollide="none"/>
      <elasticity young="{c.E0!r}" poisson="{c.NU!r}" damping="{c.damping_param("stiffness", c.ZETA0)!r}"/>
    </flexcomp>
  </worldbody>
</mujoco>"""


class MjBlock:
    def __init__(self, arm, dx, h):
        self.arm, self.h = arm, h
        self.m = mujoco.MjModel.from_xml_string(model_xml(dx, INTEGRATOR[arm], h))
        self.d = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, self.d)
        self.X0 = self.d.flexvert_xpos.copy()
        self.k0 = self.m.flex_stiffness.copy()
        dof = self.m.body_dofadr[self.m.flex_vertbodyid]
        axes = np.stack([self.m.jnt_axis[self.m.dof_jntid[dof + k]] for k in range(3)], 1)
        assert np.allclose(axes, np.eye(3)[None]), "vertex dofs are not slides along x, y, z"
        self.dof = dof
        # Vertex positions are read from qpos: flexvert_xpos is computed before a step moves
        # them, so after mj_step it is one step old.
        self.qadr = np.stack([self.m.jnt_qposadr[self.m.dof_jntid[dof + k]] for k in range(3)], 1)
        self.shape = shape = p1.mode_vector(dx, "equal")
        masses = self.m.body_mass[self.m.flex_vertbodyid]
        self.v0 = c.amplitude() * shape + c.seeded_noise(self.X0, masses)
        self.w = c.modal_weights(shape, masses)
        self.steps = c.sample_steps(h)

    def modal(self, E_ratio, zeta):
        """q(t) at the sampled steps."""
        m, d = self.m, self.d
        m.flex_stiffness[:] = self.k0 * E_ratio
        m.flex_damping[0] = c.damping_param("stiffness", zeta)
        mujoco.mj_resetData(m, d)
        for k in range(3):
            d.qvel[self.dof + k] = self.v0[:, k]
        q = np.full(len(self.steps), np.nan)
        rest = np.full(len(self.steps), np.nan)
        j = 0
        for n in range(1, self.steps[-1] + 1):
            mujoco.mj_step(m, d)
            if n == self.steps[j]:
                u = d.qpos[self.qadr] - m.qpos0[self.qadr]
                q[j] = np.sum(self.w * u)
                rest[j] = np.abs(u - q[j] * self.shape).max()
                if not np.isfinite(q[j]):
                    break
                j += 1
        if c.unstable(rest):
            q[:] = np.nan
        return q

    def measure(self, E_ratio, zeta):
        return c.measure(self.steps * self.h, self.modal(E_ratio, zeta))


def indent_xml(dx, integrator, h, radius, solref, solimp):
    import indentation as ind
    count = [int(round(s / dx)) + 1 for s in ind.SIDES]
    top = ind.SIDES[2] / 2
    contact = f'condim="1" solref="{float(solref[0])!r} {float(solref[1])!r}" solimp="{" ".join(repr(float(x)) for x in solimp)}"'
    return f"""<mujoco>
  <option timestep="{float(h)!r}" gravity="0 0 0" integrator="{integrator}"/>
  <worldbody>
    <body name="ball" mocap="true" pos="0 0 {float(top + radius + FLEX_RADIUS)!r}">
      <geom name="ball" type="sphere" size="{float(radius)!r}" {contact}/>
    </body>
    <flexcomp name="slab" type="grid" dim="3" dof="full" count="{count[0]} {count[1]} {count[2]}"
              spacing="{float(dx)!r} {float(dx)!r} {float(dx)!r}"
              mass="{float(c.RHO * np.prod(ind.SIDES))!r}" radius="{FLEX_RADIUS!r}">
      <contact {contact} selfcollide="none"/>
      <elasticity young="{c.E0!r}" poisson="{c.NU!r}" damping="{c.damping_param("stiffness", c.ZETA0)!r}"/>
      <pin gridrange="0 0 0 {count[0] - 1} {count[1] - 1} 0"/>
    </flexcomp>
  </worldbody>
</mujoco>"""


FLEX_RADIUS = 1e-4   # m; flexcomp's default 5 mm would inflate the slab's surface by 5 mm


class MjIndent:
    """The slab of the ball press; the sphere is a mocap body, its contacts frictionless."""

    def __init__(self, arm, dx, h, radius, solref, solimp):
        self.h = h
        self.m = mujoco.MjModel.from_xml_string(indent_xml(dx, INTEGRATOR[arm], h, radius, solref, solimp))
        self.d = mujoco.MjData(self.m)
        self.ball = self.m.geom("ball").id
        self.z0 = float(self.m.body("ball").pos[2])

    def forces(self, depths=None):
        """The vertical force on the sphere (N) at each depth, averaged over its window."""
        import indentation as ind
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        depth, reads = ind.schedule(self.h, ind.DEPTHS if depths is None else depths)
        f6 = np.zeros(6)
        F = np.zeros(len(depth))
        for n, dep in enumerate(depth):
            d.mocap_pos[0, 2] = self.z0 - dep
            mujoco.mj_step(m, d)
            for i in range(d.ncon):
                con = d.contact[i]
                if self.ball in (con.geom[0], con.geom[1]):
                    mujoco.mj_contactForce(m, d, i, f6)
                    F[n] += f6[0] * abs(con.frame[2])        # the normals are radial
            if not np.isfinite(F[n]):
                return np.full(len(reads), np.nan)
        return np.array([F[r].mean() for r in reads])
