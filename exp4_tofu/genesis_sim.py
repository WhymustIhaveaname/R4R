"""The block in Genesis 1.5.0 (float64): FEM, explicit or implicit, on MuJoCo's own grid.

Genesis meshes a box with tetgen, which gives an unstructured mesh whose size the options
barely control.  Here the meshing call is replaced by one that returns MuJoCo's grid, the same
vertices and the same six tetrahedra per cell, so the two engines share the stiffness exactly
and differ in the mass: Genesis gives each vertex a quarter of the mass of each tetrahedron it
belongs to.  The material is the linear one, the only one both solvers accept.  The block
starts in its own discrete version of the mode (p1.mode_vector), with Genesis's masses.

E and the damping live in Genesis's arrays, so a rollout rewrites them rather than rebuilding:
the per-element Lame constants, and the explicit decay exp(-c h) or the implicit beta / h.  The
implicit damping must be nonzero at build time, or its terms are compiled out.
"""

import os

import numpy as np
import torch

import common as c
import p1

_INIT = False


def init():
    """Genesis on the GPU, or with R4R_GENESIS_BACKEND=cpu on the CPU.

    The cluster's CPU nodes have no OpenGL, and a scene's build opens an offscreen OpenGL
    context even when nothing is rendered, so on the CPU the rasterizer is built without it.
    """
    global _INIT
    if not _INIT:
        import genesis as gs
        cpu = os.environ.get("R4R_GENESIS_BACKEND") == "cpu"
        gs.init(backend=gs.cpu if cpu else gs.gpu, precision="64", logging_level="warning")
        if cpu:
            from genesis.vis.rasterizer import Rasterizer

            def build(self):
                self._is_built = True
            Rasterizer.build = build
        _INIT = True


def oriented(X, tets):
    """Genesis wants det [x0 - x3, x1 - x3, x2 - x3] < 0 for every element."""
    tets = tets.copy()
    D = np.stack([X[tets[:, a]] - X[tets[:, 3]] for a in range(3)], -1)
    flip = np.linalg.det(D) > 0
    tets[flip, 0], tets[flip, 1] = tets[flip, 1].copy(), tets[flip, 0].copy()
    return tets


def set_field(field, values):
    import quadrants as qd  # Genesis's kernel language
    f = field._unwrap() if isinstance(field, qd.Tensor) else field
    f.from_numpy(np.asarray(values, dtype=np.float64))


class GsBlock:
    def __init__(self, arm, dx, h):
        init()
        import genesis as gs
        import genesis.utils.element as eu
        self.arm, self.h = arm, h
        self.implicit = arm == "gs_fem_implicit"
        X, tets = p1.mesh_from_mujoco(dx)
        tets = oriented(X, tets)
        keep = (eu.mesh_to_elements, eu.split_all_surface_tets)

        def grid(surface, tet_cfg=None):
            # the box's eight corners first, as Genesis maps its surface mesh to the first vertices
            first = [int(np.argmin(np.linalg.norm(X - v, axis=1))) for v in np.asarray(surface.vertices)]
            order = first + [i for i in range(len(X)) if i not in set(first)]
            inv = np.empty(len(X), int)
            inv[order] = np.arange(len(X))
            self.order = np.array(order)
            return X[order], inv[tets]

        eu.mesh_to_elements = grid
        eu.split_all_surface_tets = lambda v, e: (v, e)
        try:
            self.scene = gs.Scene(
                sim_options=gs.options.SimOptions(dt=h, substeps=1, gravity=(0.0, 0.0, 0.0)),
                fem_options=gs.options.FEMOptions(
                    use_implicit_solver=self.implicit,
                    damping=0.0 if self.implicit else c.damping_param("mass", c.ZETA0),
                    damping_alpha=0.0,
                    damping_beta=c.damping_param("stiffness", c.ZETA0) if self.implicit else 0.0),
                show_viewer=False)
            self.ent = self.scene.add_entity(
                gs.morphs.Box(pos=(0.0, 0.0, 0.0), size=tuple(float(s) for s in c.SIDES)),
                gs.materials.FEM.Elastic(E=c.E0, nu=c.NU, rho=c.RHO, model="linear"))
            self.scene.build()
        finally:
            eu.mesh_to_elements, eu.split_all_surface_tets = keep
        from genesis.utils.misc import qd_to_numpy
        info = self.scene.sim.fem_solver.fem_info
        self.info = info
        self.mu0 = qd_to_numpy(info.elements.mu).copy()
        self.lam0 = qd_to_numpy(info.elements.lam).copy()
        masses = qd_to_numpy(info.vertices.mass).copy()
        self.X0 = self.ent.get_verts_pos().cpu().numpy().reshape(-1, 3).astype(np.float64)
        assert np.allclose(self.X0, X[self.order])
        shape = p1.mode_vector(dx, "volume")[self.order]
        self.v0 = c.amplitude() * shape + c.seeded_noise(self.X0, masses)
        w = c.modal_weights(shape, masses)
        dev = self.ent.get_verts_pos().device
        self.w_t = torch.as_tensor(w, dtype=torch.float64, device=dev)
        self.shape_t = torch.as_tensor(shape, dtype=torch.float64, device=dev)
        self.X0_t = torch.as_tensor(self.X0, dtype=torch.float64, device=dev)
        self.steps = c.sample_steps(h)

    def set_params(self, E_ratio, zeta):
        set_field(self.info.elements.mu, self.mu0 * E_ratio)
        set_field(self.info.elements.lam, self.lam0 * E_ratio)
        if self.implicit:
            b = c.damping_param("stiffness", zeta)
            set_field(self.info.damping_beta_over_dt, b / self.h)
            set_field(self.info.damping_beta_factor, b / self.h + 1.0)
        else:
            set_field(self.info.damping_decay, np.exp(-self.h * c.damping_param("mass", zeta)))

    def modal(self, E_ratio, zeta):
        self.set_params(E_ratio, zeta)
        self.scene.reset()
        self.ent.set_velocity(torch.as_tensor(self.v0, dtype=torch.float64, device=self.X0_t.device))
        q = np.full(len(self.steps), np.nan)
        rest = np.full(len(self.steps), np.nan)
        j = 0
        for n in range(1, self.steps[-1] + 1):
            self.scene.step()
            if n == self.steps[j]:
                u = self.ent.get_verts_pos().reshape(-1, 3) - self.X0_t
                qj = (self.w_t * u).sum()
                q[j] = float(qj)
                rest[j] = float((u - qj * self.shape_t).abs().max())
                if not np.isfinite(q[j]):
                    break
                j += 1
        if c.unstable(rest):
            q[:] = np.nan
        return q

    def measure(self, E_ratio, zeta):
        return c.measure(self.steps * self.h, self.modal(E_ratio, zeta))


class GsMpmBlock:
    """The block in Genesis's MPM: MLS-MPM with quadratic B-splines, eight particles per cell.

    MPM has no damping parameter, so only E* is fitted.  Its Lame constants are compiled into
    the kernels, but with no damping and no gravity a rollout with E* and step h is the rollout
    with E0 and step h sqrt(E* / E0), started at v0 / sqrt(E* / E0) and read at the same step
    indices: the step and the stress scale it multiplies are arrays, so a rollout rewrites them.
    The block is shifted so that its faces lie on grid nodes.
    """

    def __init__(self, arm, dx, h):
        init()
        import genesis as gs
        from genesis.utils.misc import qd_to_numpy
        self.arm, self.h, self.dx = arm, h, dx
        cells = np.round(c.SIDES / dx).astype(int)
        assert np.allclose(cells * dx, c.SIDES), "the block is not a whole number of cells"
        self.center = c.HALF - np.floor(c.HALF / dx + 1e-9) * dx
        margin = 8 * dx
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=h, substeps=1, gravity=(0.0, 0.0, 0.0)),
            mpm_options=gs.options.MPMOptions(
                grid_density=1.0 / dx, particle_size=dx / 2,
                lower_bound=tuple(float(x) for x in self.center - c.HALF - margin),
                upper_bound=tuple(float(x) for x in self.center + c.HALF + margin)),
            show_viewer=False)
        self.ent = self.scene.add_entity(
            gs.morphs.Box(pos=tuple(float(x) for x in self.center), size=tuple(float(s) for s in c.SIDES)),
            gs.materials.MPM.Elastic(E=c.E0, nu=c.NU, rho=c.RHO, sampler="regular"))
        self.scene.build()
        self.info = self.scene.sim.mpm_solver.mpm_info
        self.stress_scale0 = float(qd_to_numpy(self.info.stress_scale))
        pos = self.ent.get_particles_pos().reshape(-1, 3)
        self.X0_t = pos.to(torch.float64)
        X = self.X0_t.cpu().numpy() - self.center
        shape = c.mode_shape(X)     # MPM has no discrete modes to start in
        self.v0 = c.amplitude() * shape + c.seeded_noise(X, np.ones(len(X)))
        self.w_t = torch.as_tensor(c.modal_weights(shape, np.ones(len(X))), dtype=torch.float64,
                                   device=pos.device)
        self.shape_t = torch.as_tensor(shape, dtype=torch.float64, device=pos.device)
        self.steps = c.sample_steps(h)

    def modal(self, E_ratio, zeta=0.0):
        s = float(np.sqrt(E_ratio))
        set_field(self.info.substep_dt, self.h * s)
        set_field(self.info.stress_scale, self.stress_scale0 * s)
        self.scene.reset()
        self.ent.set_particles_vel(torch.as_tensor(self.v0 / s, dtype=torch.float64, device=self.X0_t.device))
        q = np.full(len(self.steps), np.nan)
        rest = np.full(len(self.steps), np.nan)
        j = 0
        for n in range(1, self.steps[-1] + 1):
            self.scene.step()
            if n == self.steps[j]:
                u = self.ent.get_particles_pos().reshape(-1, 3) - self.X0_t
                qj = (self.w_t * u).sum()
                q[j] = float(qj)
                rest[j] = float((u - qj * self.shape_t).abs().max())
                if not np.isfinite(q[j]):
                    break
                j += 1
        if c.unstable(rest):
            q[:] = np.nan
        return q

    def measure(self, E_ratio, zeta=0.0):
        return c.measure(self.steps * self.h, self.modal(E_ratio, zeta))


class GsPbdBlock:
    """The block in Genesis's PBD (an elastic PBD3DEntity) on MuJoCo's grid mesh, its particles of
    equal mass.  E* scales the edges' stretch compliance as 1 / E*, from the nominal 1 / (E dx)
    (pbd_linear.py); PBD has no damping parameter, so only E* is fitted."""

    def __init__(self, arm, dx, h, n_stretch=None):
        init()
        import genesis as gs
        from genesis.engine.mesh import Mesh
        from genesis.utils.misc import qd_to_numpy

        import pbd_linear as pl
        self.arm, self.h, self.dx = arm, h, dx
        X, tets = p1.mesh_from_mujoco(dx)
        keep = Mesh.tetrahedralize
        Mesh.tetrahedralize = lambda self_, cfg: (X.copy(), tets.copy())
        try:
            self.scene = gs.Scene(
                sim_options=gs.options.SimOptions(dt=h, substeps=1, gravity=(0.0, 0.0, 0.0)),
                pbd_options=gs.options.PBDOptions(
                    particle_size=dx / 2, lower_bound=(-1.0, -1.0, -1.0), upper_bound=(1.0, 1.0, 1.0),
                    max_stretch_solver_iterations=n_stretch or pl.N_STRETCH),
                show_viewer=False)
            self.ent = self.scene.add_entity(
                gs.morphs.Box(pos=(0.0, 0.0, 0.0), size=tuple(float(s) for s in c.SIDES)),
                gs.materials.PBD.Elastic(rho=c.RHO, stretch_compliance=pl.nominal_compliance(dx)))
            self.scene.build()
        finally:
            Mesh.tetrahedralize = keep
        self.info = self.scene.sim.pbd_solver.pbd_info
        self.alpha0 = qd_to_numpy(self.info.edges.stretch_compliance).copy()
        pos = self.ent.get_particles_pos().reshape(-1, 3)
        self.X0_t = pos.to(torch.float64)
        assert np.allclose(self.X0_t.cpu().numpy(), X)
        masses = np.full(len(X), c.RHO * np.prod(c.SIDES) / len(X))
        shape = p1.mode_vector(dx, "equal")
        self.v0 = c.amplitude() * shape + c.seeded_noise(X, masses)
        self.w_t = torch.as_tensor(c.modal_weights(shape, masses), dtype=torch.float64, device=pos.device)
        self.shape_t = torch.as_tensor(shape, dtype=torch.float64, device=pos.device)
        self.steps = c.sample_steps(h)

    def modal(self, E_ratio, zeta=0.0):
        set_field(self.info.edges.stretch_compliance, self.alpha0 / E_ratio)
        self.scene.reset()
        self.ent.set_particles_vel(torch.as_tensor(self.v0, dtype=torch.float64, device=self.X0_t.device))
        q = np.full(len(self.steps), np.nan)
        rest = np.full(len(self.steps), np.nan)
        j = 0
        for n in range(1, self.steps[-1] + 1):
            self.scene.step()
            if n == self.steps[j]:
                u = self.ent.get_particles_pos().reshape(-1, 3) - self.X0_t
                qj = (self.w_t * u).sum()
                q[j] = float(qj)
                rest[j] = float((u - qj * self.shape_t).abs().max())
                if not np.isfinite(q[j]):
                    break
                j += 1
        if c.unstable(rest):
            q[:] = np.nan
        return q

    def measure(self, E_ratio, zeta=0.0):
        return c.measure(self.steps * self.h, self.modal(E_ratio, zeta))


# The sphere is driven by resetting its position and velocity every step, but within a step the
# coupler pushes it back with the slab's reaction: a 1000 kg/m^3 sphere recoils far enough to
# lower the force by 20 % at 7 mm and 0.9 ms.  At 1e7 kg/m^3 it follows its schedule.
BALL_MJCF = """<mujoco><worldbody><body name="ball" pos="0 0 {z}">
  <joint name="z" type="slide" axis="0 0 1"/>
  <geom type="sphere" size="{r}" density="1e7"/>
</body></worldbody></mujoco>"""

# SAP collides the FEM tets with the triangles of a rigid geom's collision mesh, and Genesis meshes
# an MJCF sphere at two subdivisions of an icosahedron: 3.8 mm facets that lie 0.23 mm inside the
# sphere, as deep as the first depth read.  The implicit arm's sphere is a mesh at four
# subdivisions instead, 1 mm facets 0.014 mm inside.
BALL_MESH_MJCF = """<mujoco><asset><mesh name="ball" file="{obj}"/></asset>
<worldbody><body name="ball" pos="0 0 {z}">
  <joint name="z" type="slide" axis="0 0 1"/>
  <geom type="mesh" mesh="ball" density="1e7"/>
</body></worldbody></mujoco>"""
BALL_SUBDIVISIONS = 4


# The legacy coupler's reach beyond a rigid geom: it acts on a point at distance s outside with the
# weight exp(-s / coup_softness), 2 mm by default, as deep as the indentation.  A hundredth of a
# millimetre leaves its contact nearly rigid.
COUP_SOFTNESS = 1e-5   # m

# Genesis's material per arm, and the name p1_static and axisym_nl give it
GS_MODEL = {"gs_fem_explicit": ("stable_neohookean", "neohookean"),
            "gs_fem_implicit": ("linear_corotated", "corotated")}


class GsIndent:
    """The slab of experiment 4b in Genesis's FEM, on MuJoCo's grid, bottom vertices held.

    gs_fem_implicit couples through SAP (hydroelastic contact), gs_fem_explicit through the
    legacy coupler (a velocity projection).  The sphere hangs on a vertical slide joint, and
    each step its position and velocity are reset to the schedule, so it moves as prescribed.
    The legacy coupler reports a wrong force on the sphere, so the force is read from the slab:
    the reaction at its held vertices, the gradient there of the elastic energy of the slab's
    material (p1_static.Gradient), which balances the sphere's force when the slab is at rest.

    The material is large-deformation, and Genesis 1.5.0 has none for both solvers: the explicit
    one has stable neo-Hookean (not linear corotated, whose stress it does not update), the
    implicit one linear corotated (not stable neo-Hookean, whose Hessian it lacks).
    """

    def __init__(self, arm, dx, h, radius, E_ratio=1.0, hydroelastic_modulus=None, coup_softness=COUP_SOFTNESS):
        init()
        import tempfile

        import genesis as gs
        import genesis.utils.element as eu

        import indentation as ind
        from mujoco_sim import indent_xml
        self.h = h
        implicit = arm == "gs_fem_implicit"
        X, tets = p1.mesh_from_mujoco(dx, indent_xml(dx, "discrete", 1e-3, radius, (0.02, 1), (0.9, 0.95, 0.001, 0.5, 2)))
        tets = oriented(X, tets)
        keep = (eu.mesh_to_elements, eu.split_all_surface_tets)
        order_box = {}

        def grid(surface, tet_cfg=None):
            first = [int(np.argmin(np.linalg.norm(X - v, axis=1))) for v in np.asarray(surface.vertices)]
            order = first + [i for i in range(len(X)) if i not in set(first)]
            inv = np.empty(len(X), int)
            inv[order] = np.arange(len(X))
            order_box["order"] = np.array(order)
            return X[order], inv[tets]

        eu.mesh_to_elements = grid
        eu.split_all_surface_tets = lambda v, e: (v, e)
        top = ind.SIDES[2] / 2
        try:
            coupler = gs.options.SAPCouplerOptions() if implicit else gs.options.LegacyCouplerOptions()
            self.scene = gs.Scene(
                sim_options=gs.options.SimOptions(dt=h, substeps=1, gravity=(0.0, 0.0, 0.0)),
                fem_options=gs.options.FEMOptions(
                    use_implicit_solver=implicit,
                    damping=0.0 if implicit else c.damping_param("mass", c.ZETA0),
                    damping_alpha=0.0,
                    damping_beta=c.damping_param("stiffness", c.ZETA0) if implicit else 0.0,
                    max_vertex_constraints=int(np.sum(np.isclose(X[:, 2], X[:, 2].min())))),
                coupler_options=coupler, show_viewer=False)
            kw = {} if hydroelastic_modulus is None else {"hydroelastic_modulus": hydroelastic_modulus}
            self.slab = self.scene.add_entity(
                gs.morphs.Box(pos=(0.0, 0.0, 0.0), size=tuple(float(s) for s in ind.SIDES)),
                gs.materials.FEM.Elastic(E=c.E0 * E_ratio, nu=c.NU, rho=c.RHO, model=GS_MODEL[arm][0], **kw))
            path = tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False)
            # touching the slab: unlike MuJoCo's flex, Genesis's surface has no radius to clear
            if implicit:
                import trimesh
                obj = tempfile.NamedTemporaryFile("w", suffix=".obj", delete=False).name
                trimesh.creation.icosphere(subdivisions=BALL_SUBDIVISIONS, radius=radius).export(obj)
                path.write(BALL_MESH_MJCF.format(obj=obj, z=float(top + radius)))
            else:
                path.write(BALL_MJCF.format(z=float(top + radius), r=float(radius)))
            path.close()
            # Genesis decimates a collision mesh to 500 faces by default: 3 mm facets, 0.09 mm inside the sphere
            self.ball = self.scene.add_entity(gs.morphs.MJCF(file=path.name, decimate=False),
                                              material=gs.materials.Rigid(coup_softness=coup_softness))
            self.scene.build()
        finally:
            eu.mesh_to_elements, eu.split_all_surface_tets = keep
        order = order_box["order"]
        self.X0 = X[order]
        bottom = np.flatnonzero(np.isclose(self.X0[:, 2], self.X0[:, 2].min()))
        self.slab.set_vertex_constraints(bottom)
        inv = np.empty(len(X), int)
        inv[order] = np.arange(len(X))
        from p1_static import Gradient
        self.gradient = Gradient(self.X0, inv[tets], GS_MODEL[arm][1], c.E0 * E_ratio, c.NU)
        self.bottom_z = 3 * bottom + 2

    def forces(self, depths=None):
        import indentation as ind
        depth, reads = ind.schedule(self.h, ind.DEPTHS if depths is None else depths)
        read = set(np.concatenate(reads).tolist())
        self.scene.reset()
        F = np.zeros(len(depth))
        for n, dep in enumerate(depth):
            self.ball.set_dofs_position(np.array([-dep]))
            self.ball.set_dofs_velocity(np.array([-ind.RATE]))
            self.scene.step()
            if n in read:
                x = self.slab.get_verts_pos().reshape(-1, 3).cpu().numpy()
                F[n] = self.gradient(x)[self.bottom_z].sum()
                if not np.isfinite(F[n]):
                    return np.full(len(reads), np.nan)
        return np.array([F[r].mean() for r in reads])
