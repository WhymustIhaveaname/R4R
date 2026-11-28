"""Genesis 1.5.0's FEM through its IPC coupler: libuipc 0.0.30 (pyuipc) on the GPU, on MuJoCo's grid.

With `coupler_options=IPCCouplerOptions(...)` Genesis's own FEM solver skips its substep
(FEMSolver.substep_pre_coupling returns at once) and libuipc simulates every FEM entity
(ipc_coupler/coupler.py, _add_fem_entities_to_ipc):

  material     libuipc's StableNeoHookean, whatever `model` the Genesis material names: E and nu
               become the Lame constants mu_L and lambda_L + mu_L (Smith et al.'s SNH without the
               log term, p1_static's "neohookean"); the vertex masses are lumped by volume.
  integrator   libuipc's default "bdf1", backward Euler, solved by Newton's method to a velocity
               tolerance; Genesis does not expose the integrator.
  damping      none: FEMOptions' damping, damping_alpha and damping_beta are never passed to
               libuipc, whose constitutions have no damping.  The only dissipation is backward
               Euler's own.
  contact      IPC's log barrier within d_hat (1 cm by default), stiffness `contact_resistance`
               (1e9 Pa), friction on by default; the FEM body also contacts itself.
  hardware     CUDA only (libuipc's backend), double precision.

Genesis passes nothing of a vertex's initial velocity or of a fixed vertex to libuipc
(set_vertex_constraints raises), so both are written into the libuipc geometry here, between
the coupler's adding of the objects and the IPC world's init (`_HOOK`); the world's frame 0 then
holds them, and scene.reset() returns to it.  `_CONFIG` adds entries to libuipc's scene config
(the integrator, d_hat), which Genesis's options do not reach.  Neither changes what libuipc does
with what Genesis gives it.
"""

import numpy as np
import torch

import common as c
import p1
from genesis_sim import init, oriented

_HOOK = None      # f(coupler) run on the IPC scene before the world is initialized
_CONFIG = {}      # {("integrator", "type"): "bdf2", ...}: extra entries of libuipc's scene config

# Newton's method to a velocity tolerance, m/s (libuipc's default 0.05 m/s is ten times the
# mode's peak velocity, 4.5 mm/s), with its semi-implicit shortcut off and the linear solves tight
TOL = dict(newton_tolerance=1e-9, newton_semi_implicit_enable=False, linear_system_tolerance=1e-8)


def _patch():
    import genesis.engine.solvers  # noqa: F401  (before the couplers, or their imports circle)
    from genesis.engine.couplers.ipc_coupler import coupler as cp
    if getattr(cp, "_r4r_patched", False):
        return
    add = cp.IPCCoupler._add_objects_to_ipc
    config = cp.build_ipc_scene_config

    def add_objects(self):
        add(self)
        if _HOOK is not None:
            _HOOK(self)

    def scene_config(options, sim_options, dt):
        out = config(options, sim_options, dt)
        for keys, value in _CONFIG.items():
            d = out
            for k in keys[:-1]:
                d = d[k]
            d[keys[-1]] = value
        return out
    cp.IPCCoupler._add_objects_to_ipc = add_objects
    cp.build_ipc_scene_config = scene_config
    cp._r4r_patched = True


def fem_geometries(coupler):
    """The libuipc geometries of the FEM entities, by entity index."""
    from uipc.backend import SceneVisitor
    from uipc.geometry import SimplicialComplexSlot

    from genesis.engine.couplers.ipc_coupler.utils import read_ipc_geometry_metadata
    out = {}
    for slot in SceneVisitor(coupler._ipc_scene).geometries():
        if isinstance(slot, SimplicialComplexSlot):
            meta = read_ipc_geometry_metadata(slot.geometry())
            if meta is not None and meta[0] == "fem":
                out[meta[2]] = slot.geometry()
    return out


def grid_meshing(X, tets):
    """Replace Genesis's tetgen meshing of a Box by MuJoCo's grid (as genesis_sim.GsBlock does).

    Returns (enter, leave, order): order[i] is the grid vertex of Genesis's vertex i."""
    import genesis.utils.element as eu
    keep = (eu.mesh_to_elements, eu.split_all_surface_tets)
    box = {}

    def grid(surface, tet_cfg=None):
        first = [int(np.argmin(np.linalg.norm(X - v, axis=1))) for v in np.asarray(surface.vertices)]
        order = first + [i for i in range(len(X)) if i not in set(first)]
        inv = np.empty(len(X), int)
        inv[order] = np.arange(len(X))
        box["order"] = np.array(order)
        return X[order], inv[tets]

    def enter():
        eu.mesh_to_elements = grid
        eu.split_all_surface_tets = lambda v, e: (v, e)

    def leave():
        eu.mesh_to_elements, eu.split_all_surface_tets = keep
    return enter, leave, box


def release(scene):
    """Destroy a scene and the workspace directory the coupler made for libuipc in /tmp, which
    Genesis leaves behind (about 1 MB per build)."""
    import shutil
    engine = scene.sim.coupler._ipc_engine
    workspace = engine.workspace() if engine is not None else None
    scene.destroy()
    if workspace:
        shutil.rmtree(workspace, ignore_errors=True)


def build_body(X, tets, sides, E, nu, rho, h, velocity, fixed=None, tol=None, integrator="bdf1"):
    """A Genesis scene of one FEM box meshed as (X, tets), centred at the origin, simulated by
    libuipc with no contact and no gravity.  velocity(order) gives the initial velocity of every
    vertex in Genesis's order (order[i] the grid vertex of Genesis's vertex i), fixed(order), if
    given, a mask of the vertices libuipc holds.  Returns (scene, entity, order)."""
    global _HOOK
    import genesis as gs
    init()
    _patch()
    enter, leave, box = grid_meshing(X, tets)
    v0 = {}

    def hook(coupler):
        import uipc
        (geo,) = fem_geometries(coupler).values()
        assert np.allclose(geo.positions().view().reshape(-1, 3), X[box["order"]])
        uipc.view(geo.vertices().find(uipc.builtin.velocity))[:] = v0["v"][:, :, None]
        if fixed is not None:
            uipc.view(geo.vertices().find(uipc.builtin.is_fixed))[fixed(box["order"])] = 1
    enter()
    _HOOK = hook
    _CONFIG[("integrator", "type")] = integrator
    try:
        # Genesis's FEMOptions damping never reaches libuipc; it is left at its defaults
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=h, substeps=1, gravity=(0.0, 0.0, 0.0)),
            coupler_options=gs.options.IPCCouplerOptions(contact_enable=False, **dict(TOL if tol is None else tol)),
            show_viewer=False)
        # Genesis's meshing call runs at add_entity, so the order is known before the build
        ent = scene.add_entity(
            gs.morphs.Box(pos=(0.0, 0.0, 0.0), size=tuple(float(s) for s in sides)),
            gs.materials.FEM.Elastic(E=E, nu=nu, rho=rho, model="stable_neohookean"))
        v0["v"] = velocity(box["order"])
        scene.build()
    finally:
        leave()
        _HOOK = None
        _CONFIG.clear()
    return scene, ent, box["order"]


class IpcBlock:
    """The free block of experiment 4a, simulated by libuipc through Genesis's IPC coupler.

    libuipc fixes the material's constants and the step at the world's init, so every rollout
    builds its scene anew.  There is no damping parameter: a rollout takes E* alone, and the
    damping is the scheme's own, which the fit reports as zeta*.
    """

    def __init__(self, dx, h, integrator="bdf1", tol=None):
        self.dx, self.h, self.integrator = dx, h, integrator
        self.tol = dict(TOL if tol is None else tol)
        X, tets = p1.mesh_from_mujoco(dx)
        self.X, self.tets = X, oriented(X, tets)
        _, vol = p1.assemble(X, tets, c.E0, c.NU)
        self.masses_grid = p1.masses(X, tets, vol, "volume")
        self.shape_grid = p1.mode_vector(dx, "volume")
        self.steps = c.sample_steps(h)

    def _build(self, E_ratio):
        def velocity(order):
            return c.amplitude() * self.shape_grid[order] + c.seeded_noise(self.X[order], self.masses_grid[order])
        scene, ent, order = build_body(self.X, self.tets, c.SIDES, c.E0 * E_ratio, c.NU, c.RHO, self.h, velocity,
                                       tol=self.tol, integrator=self.integrator)
        dev = ent.get_verts_pos().device
        shape, masses = self.shape_grid[order], self.masses_grid[order]
        self.w_t = torch.as_tensor(c.modal_weights(shape, masses), dtype=torch.float64, device=dev)
        self.shape_t = torch.as_tensor(shape, dtype=torch.float64, device=dev)
        self.X0_t = torch.as_tensor(self.X[order], dtype=torch.float64, device=dev)
        return scene, ent

    def modal(self, E_ratio, zeta=0.0):
        """q(t) at the sampled steps; zeta is ignored (there is no damping parameter)."""
        scene, ent = self._build(E_ratio)
        q = np.full(len(self.steps), np.nan)
        rest = np.full(len(self.steps), np.nan)
        j = 0
        try:
            for n in range(1, self.steps[-1] + 1):
                scene.step()
                if n == self.steps[j]:
                    u = ent.get_verts_pos().reshape(-1, 3) - self.X0_t
                    qj = (self.w_t * u).sum()
                    q[j] = float(qj)
                    rest[j] = float((u - qj * self.shape_t).abs().max())
                    if not np.isfinite(q[j]):
                        break
                    j += 1
        finally:
            release(scene)
        if c.unstable(rest):
            q[:] = np.nan
        self.last_rest = rest
        return q

    def measure(self, E_ratio, zeta=0.0):
        return c.measure(self.steps * self.h, self.modal(E_ratio, zeta))


def backward_euler_undamped(h, w_star):
    """(omega, zeta) that undamped backward Euler at step h gives a mode of frequency w_star:
    its eigenvalue 1 / (1 - i w* h), read as exp((-zeta omega +- i omega_d) h)."""
    x = w_star * h
    s = np.log1p(x * x) / (2 * h)
    wd = np.arctan(x) / h
    wn = np.hypot(s, wd)
    return wn, s / wn


def backward_euler_time_factor(h, w0=None):
    """T(h) = E*/E*_{h->0} and the damping ratio of undamped backward Euler, exactly, for the
    mode to have the continuum's frequency w0: the root w* of backward_euler_undamped(h, w*) = w0."""
    from scipy.optimize import brentq
    w0 = c.omega0() if w0 is None else w0
    ws = brentq(lambda w: backward_euler_undamped(h, w)[0] - w0, 0.5 * w0, 50 * w0, xtol=1e-14, rtol=1e-15)
    return (ws / w0) ** 2, backward_euler_undamped(h, ws)[1]


# The sphere, a mesh of trimesh's icosphere: at five subdivisions its 0.5 mm facets lie at most
# 3.5 um inside the sphere, 1.4 % of the first depth.  It hangs on a vertical slide joint, and its
# density is that of genesis_sim's, also given to the material, which is where libuipc's affine
# body takes it from: the soft constraint holding it to its schedule, 100 m / h^2 (N/m), is then
# stiff against the slab, and the pull the coupler computes from Genesis's mass is libuipc's.
BALL_SUBDIVISIONS = 5
# IPC's barrier width, m.  libuipc's default, 1 cm, is four times the whole depth.  The barrier
# keeps the slab's surface a gap short of d_hat away, which deepens the indentation by that gap:
# at 1e-5 m the gap is 8 um and the force at the first depth 4 % high; at 2e-6 m the gap is 0.5 um.
D_HAT = 2e-6


class IpcIndent:
    """Experiment 4b's slab in libuipc through Genesis's IPC coupler, on MuJoCo's grid.

    The bottom vertices are fixed in libuipc (Genesis cannot pass a vertex constraint to it).
    The sphere is a rigid entity coupled as "two_way_soft_constraint": libuipc carries it as an
    affine body pulled to the pose Genesis gives it, which every step is reset to the schedule.
    The contact is IPC's barrier between the sphere's triangles and the slab's whole surface,
    frictionless.  The force is read twice: on the sphere, the pull of its soft constraint,
    which the coupler computes for Genesis and which balances the contact (the sphere moves
    steadily); and from the slab, the reaction at its fixed vertices, the gradient there of the
    elastic energy (p1_static.Gradient, "neohookean", libuipc's SNH), which balances it only
    when the slab is at rest: libuipc's slab rings, undamped but for backward Euler, after the
    sphere sets off.
    """

    def __init__(self, dx, h, radius, E_ratio=1.0, d_hat=D_HAT, tol=None, subdivisions=BALL_SUBDIVISIONS):
        global _HOOK
        init()
        _patch()
        import tempfile

        import genesis as gs
        import trimesh

        import indentation as ind
        from genesis_sim import BALL_MESH_MJCF
        from mujoco_sim import indent_xml
        from p1_static import Gradient
        self.h, self.radius = h, radius
        X, tets = p1.mesh_from_mujoco(dx, indent_xml(dx, "discrete", 1e-3, radius, (0.02, 1), (0.9, 0.95, 0.001, 0.5, 2)))
        tets = oriented(X, tets)
        enter, leave, box = grid_meshing(X, tets)
        self.top = ind.SIDES[2] / 2

        def hook(coupler):
            (geo,) = fem_geometries(coupler).values()
            pos = geo.positions().view().reshape(-1, 3)
            import uipc
            fixed = uipc.view(geo.vertices().find(uipc.builtin.is_fixed))
            fixed[np.isclose(pos[:, 2], pos[:, 2].min())] = 1
        tol = dict(TOL if tol is None else tol)
        enter()
        _HOOK = hook
        try:
            self.scene = gs.Scene(
                sim_options=gs.options.SimOptions(dt=h, substeps=1, gravity=(0.0, 0.0, 0.0)),
                coupler_options=gs.options.IPCCouplerOptions(
                    contact_d_hat=d_hat, contact_friction_enable=False, enable_rigid_ground_contact=False, **tol),
                show_viewer=False)
            self.slab = self.scene.add_entity(
                gs.morphs.Box(pos=(0.0, 0.0, 0.0), size=tuple(float(s) for s in ind.SIDES)),
                gs.materials.FEM.Elastic(E=c.E0 * E_ratio, nu=c.NU, rho=c.RHO, model="stable_neohookean",
                                         friction_mu=0.0))
            obj = tempfile.NamedTemporaryFile("w", suffix=".obj", delete=False).name
            trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius).export(obj)
            path = tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False)
            path.write(BALL_MESH_MJCF.format(obj=obj, z=float(self.top + radius)))
            path.close()
            self.ball = self.scene.add_entity(
                gs.morphs.MJCF(file=path.name, decimate=False),
                material=gs.materials.Rigid(rho=1e7, coup_type="two_way_soft_constraint", coup_friction=0.0))
            self.scene.build()
        finally:
            leave()
            _HOOK = None
            _CONFIG.clear()
        order = box["order"]
        self.X0 = X[order]
        inv = np.empty(len(X), int)
        inv[order] = np.arange(len(X))
        self.tets = inv[tets]
        bottom = np.flatnonzero(np.isclose(self.X0[:, 2], self.X0[:, 2].min()))
        self.bottom_z = 3 * bottom + 2
        self.gradient = Gradient(self.X0, self.tets, "neohookean", c.E0 * E_ratio, c.NU)
        top = np.isclose(self.X0[:, 2], self.X0[:, 2].max())
        from p1_contact import top_triangles
        tris = top_triangles(self.X0, self.tets, top)
        r = np.hypot(self.X0[:, 0], self.X0[:, 1])
        self.tris = tris[(r[tris] < 0.02 + 2 * dx).any(1)]     # those the sphere can reach
        self.interior = np.flatnonzero(~top & ~np.isin(np.arange(len(X)), bottom))
        coupler = self.scene.sim.coupler
        (self.link,) = coupler._abd_data_by_link.keys()
        self.coupler = coupler
        os_remove = __import__("os").remove
        os_remove(obj)
        os_remove(path.name)

    def gap(self, x, centre):
        """The distance from the slab's top surface to the sphere (m): what IPC's barrier keeps."""
        from p1_static import _closest_on_triangles
        t = self.tris
        p, _ = _closest_on_triangles(x[t[:, 0]], x[t[:, 1]], x[t[:, 2]], np.broadcast_to(centre, (len(t), 3)))
        return float(np.linalg.norm(p - centre, axis=1).min() - self.radius)

    def close(self):
        release(self.scene)

    def sphere_z(self):
        return float(self.coupler._abd_data_by_link[self.link][0].transform[2, 3])

    def forces(self, depths=None):
        """Per depth of `depths` (default indentation.DEPTHS), averaged over its read window:
        the force from the slab's fixed vertices, the soft constraint's pull on the sphere, the
        sphere's actual depth, the gap between the sphere and the slab's top surface, and the
        largest elastic force left on an interior vertex (zero at rest if p1_static's material
        is libuipc's)."""
        import indentation as ind
        depths = ind.DEPTHS if depths is None else np.asarray(depths)
        depth, reads = ind.schedule(self.h, depths)
        read = set(np.concatenate(reads).tolist())
        self.scene.reset()
        rec = np.full((len(depth), 5), np.nan)
        z0 = self.top + self.radius
        for n, dep in enumerate(depth):
            self.ball.set_dofs_position(np.array([-dep]))
            self.ball.set_dofs_velocity(np.array([0.0]))
            self.scene.step()
            if n in read:
                x = self.slab.get_verts_pos().reshape(-1, 3).cpu().numpy()
                g = self.gradient(x)
                zc = self.sphere_z()
                gap = self.gap(x, np.array([0.0, 0.0, zc]))
                pull = float(self.coupler._coupling_data.out_forces[0, 0, 2])
                inner = float(np.abs(g.reshape(-1, 3)[self.interior]).max())
                rec[n] = (g[self.bottom_z].sum(), pull, z0 - zc, gap, inner)
                if not np.isfinite(rec[n, 0]):
                    break
        self.last_x = self.slab.get_verts_pos().reshape(-1, 3).cpu().numpy()
        return np.array([np.nanmean(rec[r], 0) for r in reads])


class IpcJello:
    """The Jello cube of jello_sim.py, 4.9 x 4.7 x 4.5 cm, 10 cells a side, its base fixed, in
    libuipc through Genesis's IPC coupler.  It starts in the P1 sway of its own mesh, with
    libuipc's masses (lumped by volume), and its motion is projected on that sway."""

    def __init__(self):
        from scipy.sparse import diags
        from scipy.sparse.linalg import eigsh

        import jello_model as jm
        import jello_sim as js
        X, tets = p1.mesh_from_mujoco(None, js.xml(1.0, 1e-3, 1e-3))
        X = X - (X.max(0) + X.min(0)) / 2
        self.X, self.tets, self.sides = X, oriented(X, tets), jm.SIDES
        K, vol = p1.assemble(X, tets, 1.0, js.NU)
        m = np.zeros(len(X))
        np.add.at(m, tets.ravel(), np.repeat(vol / 4, 4))     # rho = 1, lumped by volume
        self.base = np.isclose(X[:, 2], X[:, 2].min())
        free = np.flatnonzero(np.repeat(~self.base, 3))
        w2, V = eigsh(K[free][:, free], k=4, M=diags(np.repeat(m, 3)[free]), sigma=0.0, which="LM")
        k = int(np.argmin(w2))
        self.C_p1 = float(np.sqrt(w2[k]) / (2 * np.pi))       # the sway's f at E / rho = 1
        shape = np.zeros((len(X), 3))
        shape.ravel()[free] = V[:, k]
        self.shape, self.masses = shape, m * js.RHO

    def run(self, E, h, duration=1.5):
        """(f_n Hz, zeta, t, q) of the cube started in its sway."""
        import jello_sim as js
        scene, ent, order = build_body(
            self.X, self.tets, self.sides, E, js.NU, js.RHO, h,
            lambda o: 1e-3 * self.shape[o] / np.abs(self.shape).max(), fixed=lambda o: self.base[o])
        dev = ent.get_verts_pos().device
        w = torch.as_tensor(c.modal_weights(self.shape[order], self.masses[order]), dtype=torch.float64, device=dev)
        X0 = torch.as_tensor(self.X[order], dtype=torch.float64, device=dev)
        steps = np.arange(1, int(duration / h) + 1)
        q = np.empty(len(steps))
        try:
            for j in range(len(steps)):
                scene.step()
                q[j] = float((w * (ent.get_verts_pos().reshape(-1, 3) - X0)).sum())
        finally:
            release(scene)
        wn, z = js.measure_at(steps * h, q, 2 * np.pi * js.F_VIDEO)
        return wn / (2 * np.pi), z, steps * h, q
