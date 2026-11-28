"""The ball press (experiment 4b) and Brookfield's calibration for Genesis's MPM and PBD.

As genesis_sim.GsIndent does for the FEM: the 10 x 10 x 4 cm slab of indentation.py centred at the
origin, its bottom held, no gravity; the sphere of BALL_MJCF (density 1e7, a vertical slide joint,
decimate=False, Rigid(coup_softness=COUP_SOFTNESS)) is reset every step to the schedule
indentation.schedule(h, depths) through the legacy coupler, and the force at each depth is the mean
over its read window.  E = c.E0, nu = c.NU, rho = c.RHO.

  gs_mpm  MPM.Elastic (Genesis's "corotation": fixed corotated, tau = 2 mu (F - R) F^T + lam J (J - 1) I)
          on a grid of dx with Genesis's regular sampler at dx / 2, the slab's faces on grid planes; the
          bottom held by a layer of fixed particles 0.5 to 1.5 cells below it (genesis_jello.JelloMpm).
          h = 0.02 dx, Genesis's suggestion.  The coupler acts on the grid nodes inside the sphere, so
          depth 0 is the sphere touching the particles' top face, as for the FEM.
          Reference: the continuum in fixed corotated (axisym_nl's LayerNL with that energy,
          mu sum (s_i - 1)^2 + lam / 2 (J - 1)^2, added here; axisym_nl's "corotated" is the FEM's
          linear corotated).
  gs_pbd  PBD.Elastic on MuJoCo's grid of dx, compliance pbd_linear.nominal_compliance(dx), particle
          size dx / 2, bottom particles fixed.  The coupler pushes a particle out to particle_size / 2
          from the geom, so the sphere geom has the radius R - particle_size / 2: the surface the
          particles meet is then a sphere of radius R, and depth 0 is that surface touching the top
          face, at every dx.  h = 0.5 h_max at E0, h_max the largest step at which zero compliance
          still reaches the slab's lowest elastic frequency at E0 (pbd_linear's definition, on the
          slab).  Reference: neo-Hookean (PBD has no continuum material; section 3 inverts
          Brookfield with neo-Hookean).

The force on the sphere is read from the sphere: every step its velocity is reset to the
schedule's, so the coupler's push over the step changes it by F h / m (m its mass), and F = m
(v_after - v_set) / h.  For MPM it is checked against an independent number, the vertical stress
of the particles summed over a horizontal section of the slab (in equilibrium every section
carries the sphere's force).

  python run_indent_particles.py --task K           # a Slurm array index: one (arm, dx) press
  python run_indent_particles.py --deep ARM         # 5 mm to 7 mm
  python run_indent_particles.py --brookfield ARM   # 5 mm to 7 mm at each tofu's E*
"""

import argparse
import json
import os
import time

import numpy as np
import torch

import common as c
import indentation as ind
from genesis_sim import BALL_MJCF, COUP_SOFTNESS, init, set_field

PARTS = c.RESULTS / "indent_parts"
DXS = (0.005, 0.01 / 3, 0.0025, 0.002)
ARMS = ("gs_mpm", "gs_pbd")
CUT_SIM, CUT_WALL, CUT_SPEED = 0.05, 5.0, 0.01
TOP = ind.SIDES[2] / 2
MODEL = {"gs_mpm": "fixed_corotated", "gs_pbd": "neohookean"}


# ---- the reference in MPM's material ---------------------------------------------------------

def register_fixed_corotated():
    import axisym_nl
    if "fixed_corotated" in axisym_nl.MODELS:
        return
    base = axisym_nl.density

    def density(model, mu, lam):
        if model != "fixed_corotated":
            return base(model, mu, lam)

        def psi(F2, ft):
            th = torch.atan2(F2[:, 1, 0] - F2[:, 0, 1], F2[:, 0, 0] + F2[:, 1, 1])
            cs, sn = torch.cos(th), torch.sin(th)
            Rt = torch.stack([torch.stack([cs, sn], 1), torch.stack([-sn, cs], 1)], 1)
            S = Rt @ F2
            S = 0.5 * (S + S.transpose(1, 2)) - torch.eye(2)
            J = (F2[:, 0, 0] * F2[:, 1, 1] - F2[:, 0, 1] * F2[:, 1, 0]) * ft
            return mu * ((S * S).sum((1, 2)) + (ft - 1) ** 2) + 0.5 * lam * (J - 1) ** 2
        return psi
    axisym_nl.density = density
    axisym_nl.MODELS = axisym_nl.MODELS + ("fixed_corotated",)


def reference(arm, depths=None):
    register_fixed_corotated()
    return ind.reference(MODEL[arm], None if depths is None else tuple(float(d) for d in depths))


# ---- the scenes -------------------------------------------------------------------------------

def ball_entity(gs, scene, radius, top):
    import tempfile
    path = tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False)
    path.write(BALL_MJCF.format(z=float(top + radius), r=float(radius)))
    path.close()
    ball = scene.add_entity(gs.morphs.MJCF(file=path.name, decimate=False),
                            material=gs.materials.Rigid(coup_softness=COUP_SOFTNESS))
    return ball, path.name


class ParticleIndent:
    """One press: build, then forces(depths) runs the schedule and reads the sphere."""

    def forces(self, depths=None, E_ratio=1.0, timed=True, section=False):
        """(force at each depth (N), speed, cut, extra): the force read from the sphere's velocity;
        with `section`, also the MPM section force at the same reads."""
        depths = ind.DEPTHS if depths is None else np.asarray(depths)
        depth, reads = ind.schedule(self.h, depths)
        read = set(np.concatenate(reads).tolist())
        self.set_E(E_ratio)
        dev = self.dev
        sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
        if timed:      # compile and warm up out of the clock: 10 steps, then start again
            self.scene.reset()
            self.hold()
            for dep in depth[:10]:
                self.ball.set_dofs_position(np.array([-dep]))
                self.ball.set_dofs_velocity(np.array([-ind.RATE]))
                self.scene.step()
            self.ball.get_dofs_velocity()
            sync()
        self.scene.reset()
        self.hold()
        F = np.zeros(len(depth))
        Fs = np.zeros(len(depth))
        v_set = np.array([-ind.RATE])
        t0 = time.perf_counter()
        for n, dep in enumerate(depth):
            self.ball.set_dofs_position(np.array([-dep]))
            self.ball.set_dofs_velocity(v_set)
            self.scene.step()
            if n in read:
                v = float(self.ball.get_dofs_velocity().reshape(-1)[0])
                F[n] = self.mass * (v - v_set[0]) / self.h
                if section:
                    Fs[n] = self.section_force()
                if not np.isfinite(F[n]):
                    return np.full(len(reads), np.nan), (n + 1) * self.h / (time.perf_counter() - t0), False, {}
            if timed and (n + 1) * self.h <= CUT_SIM + 1e-12:
                wall = time.perf_counter() - t0
                if abs((n + 1) * self.h - CUT_SIM) < 0.5 * self.h:
                    sync()
                    wall = time.perf_counter() - t0
                if wall > CUT_WALL or (abs((n + 1) * self.h - CUT_SIM) < 0.5 * self.h and (n + 1) * self.h / wall < CUT_SPEED):
                    return None, (n + 1) * self.h / wall, True, {}
        sync()
        wall = time.perf_counter() - t0
        speed = len(depth) * self.h / wall
        extra = {"wall_s": wall, "sim_s": len(depth) * self.h,
                 "speed_note": "the whole loop: schedule resets, steps and the reads in the windows"}
        if section:
            extra["section_force"] = [float(Fs[r].mean()) for r in reads]
        return np.array([F[r].mean() for r in reads]), speed, False, extra


class MpmIndent(ParticleIndent):
    def __init__(self, dx, h=None):
        init()
        import genesis as gs
        from genesis.utils.misc import qd_to_numpy
        self.dx, self.h = dx, h or 0.02 * dx
        p = dx / 2
        margin = 8 * dx
        half = ind.SIDES / 2
        lower = (-half[0] - margin, -half[1] - margin, -half[2] - margin)
        upper = (half[0] + margin, half[1] + margin, half[2] + ind.RADIUS * 2 + margin)
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.h, substeps=1, gravity=(0.0, 0.0, 0.0)),
            mpm_options=gs.options.MPMOptions(grid_density=1.0 / dx, particle_size=p,
                                              lower_bound=lower, upper_bound=upper),
            coupler_options=gs.options.LegacyCouplerOptions(), show_viewer=False)
        mat = gs.materials.MPM.Elastic(E=c.E0, nu=c.NU, rho=c.RHO, sampler="regular")
        self.slab = self.scene.add_entity(gs.morphs.Box(pos=(0.0, 0.0, 0.0), size=tuple(float(s) for s in ind.SIDES)), mat)
        self.base = self.scene.add_entity(
            gs.morphs.Box(pos=(0.0, 0.0, float(-half[2] - 0.75 * dx)),
                          size=(float(ind.SIDES[0] + 4 * dx), float(ind.SIDES[1] + 4 * dx), dx / 2)), mat)
        self.ball, self._xml = ball_entity(gs, self.scene, ind.RADIUS, TOP)
        self.scene.build()
        self.info = self.scene.sim.mpm_solver.mpm_info
        self.stress_scale0 = float(qd_to_numpy(self.info.stress_scale))
        self.dev = self.slab.get_particles_pos().device
        zs = self.base.get_particles_pos().reshape(-1, 3)[:, 2].cpu().numpy()
        assert np.all((zs >= -half[2] - 1.5 * dx) & (zs < -half[2] - 0.5 * dx))
        X = self.slab.get_particles_pos().reshape(-1, 3).cpu().numpy()
        assert abs(X[:, 2].max() + p / 2 - TOP) < 1e-9 and abs(X[:, 2].min() - p / 2 + half[2]) < 1e-9
        self.mass = float(self.ball.get_mass())
        # the section: the particle layer just below mid-height; in equilibrium it carries the force
        zl = np.unique(np.round(X[:, 2], 12))
        self.layer = torch.as_tensor(np.flatnonzero(np.isclose(X[:, 2], zl[len(zl) // 2 - 1])), device=self.dev)
        self.p = p
        self.slab_start = self.slab.particle_start
        self.mu = c.E0 / (2 * (1 + c.NU))
        self.lam = c.E0 * c.NU / ((1 + c.NU) * (1 - 2 * c.NU))
        self.E_ratio = 1.0

    def hold(self):
        self.base.set_free(torch.zeros(self.base.n_particles, dtype=torch.bool, device=self.dev))

    def set_E(self, E_ratio):
        """E is compiled into the kernels: E_ratio scales the stress per step (stress_scale), which
        the elastic force alone reads; the grid's momentum and the particles' masses are untouched."""
        self.E_ratio = E_ratio
        set_field(self.info.stress_scale, self.stress_scale0 * E_ratio)

    def section_force(self):
        """The vertical force across the layer's horizontal section: -sum P_zz p^2 over the layer, P = tau
        F^-T the first Piola-Kirchhoff stress of the particle's F in fixed corotated, p^2 the layer's area
        per particle at rest; positive in compression."""
        from genesis.utils.misc import qd_to_torch
        F = qd_to_torch(self.scene.sim.mpm_solver.mpm_state.particles.F)
        F = F.reshape(-1, 3, 3) if F.shape[-2:] == (3, 3) else F
        F = F[self.slab_start + self.layer].to(torch.float64)
        U, S, Vh = torch.linalg.svd(F)
        R = U @ Vh
        J = torch.linalg.det(F)
        eye = torch.eye(3, dtype=F.dtype, device=F.device)
        tau = 2 * self.mu * (F - R) @ F.transpose(1, 2) + (self.lam * J * (J - 1))[:, None, None] * eye
        P = tau @ torch.linalg.inv(F).transpose(1, 2)
        return float(-(P[:, 2, 2]).sum() * self.p**2 * self.E_ratio)


class PbdIndent(ParticleIndent):
    def __init__(self, dx, h):
        init()
        import genesis as gs
        from genesis.engine.mesh import Mesh
        from genesis.utils.misc import qd_to_numpy

        import p1
        import pbd_linear as pl
        from mujoco_sim import indent_xml
        self.dx, self.h = dx, h
        X, tets = p1.mesh_from_mujoco(dx, indent_xml(dx, "discrete", 1e-3, ind.RADIUS, (0.02, 1), (0.9, 0.95, 0.001, 0.5, 2)))
        self.r_p = dx / 4                              # particle_size / 2
        keep = Mesh.tetrahedralize
        Mesh.tetrahedralize = lambda self_, cfg: (X.copy(), tets.copy())
        try:
            self.scene = gs.Scene(
                sim_options=gs.options.SimOptions(dt=h, substeps=1, gravity=(0.0, 0.0, 0.0)),
                pbd_options=gs.options.PBDOptions(particle_size=dx / 2, lower_bound=(-1.0, -1.0, -1.0),
                                                  upper_bound=(1.0, 1.0, 1.0),
                                                  max_stretch_solver_iterations=pl.N_STRETCH),
                coupler_options=gs.options.LegacyCouplerOptions(), show_viewer=False)
            self.slab = self.scene.add_entity(
                gs.morphs.Box(pos=(0.0, 0.0, 0.0), size=tuple(float(s) for s in ind.SIDES)),
                gs.materials.PBD.Elastic(rho=c.RHO, stretch_compliance=pl.nominal_compliance(dx)))
            # the geom of radius R - r_p: the particles stop on the sphere of radius R
            self.ball, self._xml = ball_entity(gs, self.scene, ind.RADIUS - self.r_p, TOP + self.r_p)
            self.scene.build()
        finally:
            Mesh.tetrahedralize = keep
        self.info = self.scene.sim.pbd_solver.pbd_info
        self.alpha0 = qd_to_numpy(self.info.edges.stretch_compliance).copy()
        pos = self.slab.get_particles_pos().reshape(-1, 3).cpu().numpy()
        assert np.allclose(pos, X) and abs(X[:, 2].max() - TOP) < 1e-9
        self.dev = self.slab.get_particles_pos().device
        self.bottom = np.flatnonzero(np.isclose(X[:, 2], X[:, 2].min()))
        self.mass = float(self.ball.get_mass())

    def hold(self):
        self.slab.fix_particles(self.bottom)

    def set_E(self, E_ratio):
        set_field(self.info.edges.stretch_compliance, self.alpha0 / E_ratio)


def pbd_bottom_check(dx=0.005, depths=None, heavy=1e6):
    """The PBD read checked against the bottom reaction, in one run.

    PBD has no stress to sum over a section, and its fixed particles take no momentum.  So the
    bottom particles are made free but `heavy` times as massive (their inverse mass, which weighs
    every projection, nearly zero, so they hardly move), and every step they are put back at rest:
    the momentum they took over the step, M sum v_z, is the impulse of the slab on its base, and
    -M sum v_z / h the reaction, against the sphere's force read as in forces()."""
    from genesis.utils.misc import qd_to_numpy
    depths = ind.DEPTHS if depths is None else np.asarray(depths)
    h, _ = step("gs_pbd", dx)
    sim = PbdIndent(dx, h)
    mass = qd_to_numpy(sim.info.particles.mass).copy()
    M = mass[sim.bottom[0]] * heavy
    mass[sim.bottom] = M
    set_field(sim.info.particles.mass, mass)
    sim.hold = lambda: None
    X0 = sim.slab.get_particles_pos().reshape(-1, 3).clone()
    bottom_t = torch.as_tensor(sim.bottom, device=sim.dev)
    rest = X0[bottom_t]
    zero = torch.zeros_like(rest)
    depth, reads = ind.schedule(h, depths)
    read = set(np.concatenate(reads).tolist())
    sim.scene.reset()
    F = np.zeros(len(depth))
    B = np.zeros(len(depth))
    v_set = np.array([-ind.RATE])
    for n, dep in enumerate(depth):
        sim.slab.set_particles_pos(rest, sim.bottom)
        sim.slab.set_particles_vel(zero, sim.bottom)
        sim.ball.set_dofs_position(np.array([-dep]))
        sim.ball.set_dofs_velocity(v_set)
        sim.scene.step()
        if n in read:
            v = float(sim.ball.get_dofs_velocity().reshape(-1)[0])
            F[n] = sim.mass * (v - v_set[0]) / h
            vb = sim.slab.get_particles_vel().reshape(-1, 3)[bottom_t]
            B[n] = -float(vb[:, 2].sum()) * M / h
    return {"dx": dx, "h": h, "heavy": heavy, "depths": depths.tolist(),
            "sphere_read": [float(F[r].mean()) for r in reads], "bottom_reaction": [float(B[r].mean()) for r in reads]}


def pbd_h_max(dx):
    """pbd_linear's h_max on the slab: the largest step at which zero compliance still reaches the
    slab's lowest elastic frequency at E0 (P1 on MuJoCo's grid of dx, masses by volume), PBD's
    sweep linearized about rest with the bottom particles fixed (genesis_jello.pbd_sweep_matrix)."""
    from scipy.sparse import diags
    from scipy.sparse.linalg import eigsh

    import genesis_jello as gj
    import p1
    import pbd_linear as pl
    from mujoco_sim import indent_xml
    X, tets = p1.mesh_from_mujoco(dx, indent_xml(dx, "discrete", 1e-3, ind.RADIUS, (0.02, 1), (0.9, 0.95, 0.001, 0.5, 2)))
    K, vol = p1.assemble(X, tets, c.E0, c.NU)
    m = np.zeros(len(X))
    np.add.at(m, tets.ravel(), np.repeat(c.RHO * vol / 4, 4))
    fixed = np.isclose(X[:, 2], X[:, 2].min())
    free = np.flatnonzero(np.repeat(~fixed, 3))
    w2 = eigsh(K[free][:, free], k=1, M=diags(np.repeat(m, 3)[free]), sigma=0.0, which="LM",
               return_eigenvectors=False)[0]
    w0 = float(np.sqrt(w2))
    e = {tuple(sorted((t[a], t[b]))) for t in tets for a in range(4) for b in range(a + 1, 4)}
    e = np.array(sorted(e))
    nv = X[e[:, 0]] - X[e[:, 1]]
    nv /= np.linalg.norm(nv, axis=1)[:, None]
    net = (X, tets, e, nv, fixed, c.RHO * np.prod(ind.SIDES) / len(X))
    A = gj.pbd_sweep_matrix(net, 0.0)
    mu0 = float(eigsh(A, k=1, sigma=-1e-6, which="LM", return_eigenvectors=False)[0])
    p0 = (1 - mu0) ** pl.N_STRETCH
    return float(np.hypot(np.arccos(np.sqrt(p0)), np.log(p0) / 2) / w0), w0 / (2 * np.pi)


@__import__("functools").lru_cache(maxsize=None)
def step(arm, dx):
    """(h, info) of an arm at dx."""
    if arm == "gs_mpm":
        return float(f"{0.02 * dx:.4g}"), {"rule": "0.02 dx"}
    path = c.CACHE / f"indent_pbd_hmax_{dx * 1e3:.3f}mm.json"
    if path.exists():
        hm, f0 = json.loads(path.read_text())
    else:
        hm, f0 = pbd_h_max(dx)
        c.CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([hm, f0]))
    return float(f"{0.5 * hm:.4g}"), {"rule": "0.5 h_max", "h_max": hm, "f_lowest_E0": f0}


def build(arm, dx):
    h, _ = step(arm, dx)
    return MpmIndent(dx, h) if arm == "gs_mpm" else PbdIndent(dx, h)


def hardware():
    import platform
    import socket
    out = {"node": socket.gethostname(), "python": platform.python_version(), "slurm_job": os.environ.get("SLURM_JOB_ID")}
    if torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name()
    return out


def save(path, row):
    os.listdir(path.parent.parent)     # an NFS client may cache that the directory is missing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=1, default=float))


def tasks():
    return [(arm, dx) for dx in DXS for arm in ARMS]


def run_press(arm, dx):
    h, info = step(arm, dx)
    out = PARTS / f"{arm}_legacy_{dx * 1e3:.3f}mm_{h * 1e6:.0f}us.json"
    if out.exists():
        return out.name
    t0 = time.time()
    sim = build(arm, dx)
    build_s = time.time() - t0
    validate = dx == 0.005
    F, speed, cut, extra = sim.forces(timed=True, section=validate and arm == "gs_mpm")
    ref = reference(arm)
    row = {"arm": arm, "dx": dx, "h": h, "contact": "legacy", "depths": ind.DEPTHS.tolist(), "speed": speed,
           "cut": cut, "step": info, "reference_model": MODEL[arm], "build_seconds": build_s,
           "hardware": hardware(), "sphere_mass": sim.mass, "force_read": "sphere: m (v_after - v_set) / h"}
    if arm == "gs_pbd":
        row["depth_zero"] = ("geom radius R - particle_size/2, so the particles stop on a sphere of radius R; "
                             "depth 0: that sphere touching the top face")
        row["sphere_geom_radius"] = ind.RADIUS - sim.r_p
    if not cut:
        row["force"] = F.tolist()
        row["reference"] = ref.tolist()
        row["relative_error"] = (F / ref - 1).tolist()
        row.update(extra)
    row["seconds"] = time.time() - t0
    save(out, row)
    return out.name


def run_deep(arm):
    dx = 0.005
    h, info = step(arm, dx)
    out = c.RESULTS / f"gs_deep_{arm[3:]}_5mm.json"
    if out.exists():
        return out.name
    t0 = time.time()
    sim = build(arm, dx)
    F, speed, cut, extra = sim.forces(ind.DEEP, timed=True, section=arm == "gs_mpm")
    ref = reference(arm, ind.DEEP)
    row = {"arm": arm, "dx": dx, "h": h, "contact": "legacy", "depths": ind.DEEP.tolist(), "speed": speed,
           "cut": cut, "step": info, "reference_model": MODEL[arm], "hardware": hardware(), "sphere_mass": sim.mass}
    if not cut:
        row.update(force=F.tolist(), reference=ref.tolist(), relative_error=(F / ref - 1).tolist(), **extra)
    row["seconds"] = time.time() - t0
    save(out, row)
    return out.name


def run_brookfield(arm):
    """Each tofu's E from Brookfield in the arm's continuum material, E* = E F_ref(7 mm) / F_run(7 mm; E0)
    from the deep run, then the press to 7 mm at E* and at E."""
    import calibrate as cal
    deep = json.loads((c.RESULTS / f"gs_deep_{arm[3:]}_5mm.json").read_text())
    F_ref7 = float(reference(arm, ind.DEEP)[-1])
    F_run7 = float(deep["force"][-1])
    out = c.RESULTS / f"brookfield_{arm}.json"
    rows = json.loads(out.read_text())["tofus"] if out.exists() else {}
    sim = build(arm, 0.005)
    for tofu, grams in cal.BROOKFIELD_G.items():
        if tofu in rows:
            continue
        measured = grams * cal.G / 1000
        E = c.E0 * measured / F_ref7
        E_star = E * F_ref7 / F_run7
        r = {"grams": grams, "measured_N": measured, "E": E, "E_star": E_star}
        for key, ratio in (("E_star", E_star / c.E0), ("E", E / c.E0)):
            F, speed, cut, extra = sim.forces(ind.DEEP, E_ratio=ratio, timed=False)
            r[f"F7_at_{key}"] = float(F[-1])
            r[f"error_at_{key}"] = float(F[-1] / measured - 1)
            r[f"force_at_{key}"] = F.tolist()
        rows[tofu] = r
        save(out, {"arm": arm, "dx": 0.005, "h": deep["h"], "reference_model": MODEL[arm], "F_ref_7mm_E0": F_ref7,
                   "F_run_7mm_E0": F_run7, "tofus": rows})
        print(tofu, {k: v for k, v in r.items() if not k.startswith("force_")}, flush=True)
    return out.name


def pbd_brookfield_fit(tol=0.005):
    """PBD's Brookfield by the force itself: its force is not proportional to 1 / alpha, so the
    linear E* is no test.  The ceiling is the 7 mm force at zero compliance, where PBD's projection
    saturates (c = r / 2), timed; a tofu measured above it has no E*.  Below it, E* (one scale on the
    compliance, alpha = alpha_nom E / E*) is solved in log alpha by the Illinois method on
    log F(7 mm), bracketed by the nominal compliance and by alpha_nom / 1e4, until F(7 mm) is
    within `tol` of the measurement."""
    import calibrate as cal
    out = c.RESULTS / "brookfield_gs_pbd.json"
    data = json.loads(out.read_text())
    sim = build("gs_pbd", 0.005)
    runs = {}

    def F7(x):                                   # x = ln(alpha / alpha_nom)
        if x not in runs:
            F, speed, cut, extra = sim.forces(ind.DEEP, E_ratio=float(np.exp(-x)), timed=False)
            runs[x] = float(F[-1])
            print(f"  alpha/alpha_nom {np.exp(x):.6g}: F7 {runs[x]:.5f} N", flush=True)
        return runs[x]

    t0 = time.time()
    Fc, speed, cut, extra = sim.forces(ind.DEEP, E_ratio=float("inf"), timed=True)
    ceiling = {"alpha": 0.0, "F7": float(Fc[-1]), "force": Fc.tolist(), "speed": speed, "cut": cut,
               "hardware": hardware(), "seconds": time.time() - t0, **extra}
    print("ceiling", ceiling["F7"], "N, speed", speed, flush=True)
    x_lo, x_hi = np.log(1e-4), 0.0
    tofus = {}
    for tofu, grams in cal.BROOKFIELD_G.items():
        target = grams * cal.G / 1000
        r = {"measured_N": target}
        if target >= ceiling["F7"]:
            r.update(E_star_over_E=None, note=f"above the ceiling {ceiling['F7']:.4f} N: no compliance reaches it")
            tofus[tofu] = r
            continue
        a, b = x_lo, x_hi                         # F7(a) > target > F7(b)
        fa, fb = np.log(F7(a) / target), np.log(F7(b) / target)
        if fa < 0:
            r.update(E_star_over_E=None, note=f"F7 at alpha_nom/1e4 is {F7(a):.4f} N, below the measurement")
            tofus[tofu] = r
            continue
        side = 0
        for _ in range(30):
            x = b - fb * (b - a) / (fb - fa)
            fx = np.log(F7(x) / target)
            if abs(np.exp(fx) - 1) < tol:
                break
            if fx > 0:
                a, fa = x, fx
                if side == -1:
                    fb /= 2
                side = -1
            else:
                b, fb = x, fx
                if side == 1:
                    fa /= 2
                side = 1
        E_ratio = float(np.exp(-x))
        r.update(alpha_over_alpha_nom=float(np.exp(x)), E_star_over_E=E_ratio, E_star=c.E0 * E_ratio,
                 F7=runs[x], error=runs[x] / target - 1)
        tofus[tofu] = r
        print(tofu, r, flush=True)
    data["fitted"] = {"rule": "E* = E0 alpha_nom / alpha, alpha solved so that F(7 mm) is within 0.5 % of "
                              "Brookfield's; E* / E here is against E0, the run's nominal modulus",
                      "ceiling": ceiling, "tofus": tofus,
                      "runs": [{"alpha_over_alpha_nom": float(np.exp(x)), "F7": f} for x, f in sorted(runs.items())]}
    save(out, data)
    return out.name


BF_PARTS = c.RESULTS / "brookfield_gs_pbd_parts"


def pbd_fit_tofu(fraction, tofu, tol=0.005):
    """pbd_brookfield_fit for one tofu at h = fraction h_max, for jobs in parallel: the ceiling at
    alpha = 0 (timed), then, below it, alpha* by the Illinois method in log alpha on log F(7 mm)
    between alpha_nom / 1e4 and alpha_nom.  Saved as a part, merged by pbd_fit_merge."""
    import calibrate as cal
    _, info = step("gs_pbd", 0.005)
    h = float(f"{fraction * info['h_max']:.4g}")
    sim = PbdIndent(0.005, h)
    runs = {}

    def F7(x):
        if x not in runs:
            F, speed, cut, extra = sim.forces(ind.DEEP, E_ratio=float(np.exp(-x)), timed=False)
            runs[x] = float(F[-1])
            print(f"  alpha/alpha_nom {np.exp(x):.6g}: F7 {runs[x]:.5f} N", flush=True)
        return runs[x]

    t0 = time.time()
    timed = os.environ.get("R4R_UNTIMED") is None      # a fit off the cluster: its speed is not the one reported
    Fc, speed, cut, extra = sim.forces(ind.DEEP, E_ratio=float("inf"), timed=timed)
    ceiling = {"alpha": 0.0, "F7": float(Fc[-1]), "force": Fc.tolist(), "speed": speed, "cut": cut, "timed": timed,
               "hardware": hardware(), "seconds": time.time() - t0, **extra}
    print("ceiling", ceiling["F7"], "N, speed", speed, flush=True)
    target = cal.BROOKFIELD_G[tofu] * cal.G / 1000
    r = {"measured_N": target}
    if target >= ceiling["F7"]:
        r.update(E_star_over_E=None, note=f"above the ceiling {ceiling['F7']:.4f} N: no compliance reaches it")
    else:
        a, b = np.log(1e-4), 0.0
        fa, fb = np.log(F7(a) / target), np.log(F7(b) / target)
        side, x = 0, None
        if fa < 0:
            r.update(E_star_over_E=None, note=f"F7 at alpha_nom/1e4 is {F7(a):.4f} N, below the measurement")
        else:
            for _ in range(30):
                x = b - fb * (b - a) / (fb - fa)
                fx = np.log(F7(x) / target)
                if abs(np.exp(fx) - 1) < tol:
                    break
                if fx > 0:
                    a, fa = x, fx
                    if side == -1:
                        fb /= 2
                    side = -1
                else:
                    b, fb = x, fx
                    if side == 1:
                        fa /= 2
                    side = 1
            E_ratio = float(np.exp(-x))
            r.update(alpha_over_alpha_nom=float(np.exp(x)), E_star_over_E=E_ratio, E_star=c.E0 * E_ratio,
                     F7=runs[x], error=runs[x] / target - 1)
    print(tofu, r, flush=True)
    part = {"fraction": fraction, "h": h, "h_max": info["h_max"], "tofu": tofu, "result": r, "ceiling": ceiling,
            "runs": [{"alpha_over_alpha_nom": float(np.exp(x)), "F7": f} for x, f in sorted(runs.items())]}
    save(BF_PARTS / f"{fraction:g}_{tofu.replace(' ', '_')}.json", part)
    return part


def pbd_fit_merge(fraction, key):
    import calibrate as cal
    out = c.RESULTS / "brookfield_gs_pbd.json"
    data = json.loads(out.read_text())
    parts = [json.loads((BF_PARTS / f"{fraction:g}_{t.replace(' ', '_')}.json").read_text()) for t in cal.BROOKFIELD_G]
    ceilings = [p["ceiling"]["F7"] for p in parts]
    first = parts[0]
    data[key] = {"rule": "as 'fitted', at h = fraction h_max: E* = E0 alpha_nom / alpha, alpha solved so that "
                         "F(7 mm) is within 0.5 % of Brookfield's; E* / E against E0",
                 "fraction_of_h_max": fraction, "h": first["h"], "h_max": first["h_max"], "ceiling": first["ceiling"],
                 "ceiling_F7_all_jobs": ceilings,
                 "ceiling_ratio_to_half_h_max": first["ceiling"]["F7"] / data["fitted"]["ceiling"]["F7"],
                 "tofus": {p["tofu"]: p["result"] for p in parts},
                 "runs": {p["tofu"]: p["runs"] for p in parts}}
    save(out, data)
    return data[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int)
    ap.add_argument("--point", nargs=2)
    ap.add_argument("--deep")
    ap.add_argument("--brookfield")
    ap.add_argument("--reference", action="store_true", help="compute the continuum references (CPU)")
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--pbd-fit", action="store_true")
    ap.add_argument("--pbd-fit-tofu", nargs="+", metavar="FRACTION TOFU")
    ap.add_argument("--pbd-fit-merge", nargs=2, metavar=("FRACTION", "KEY"))
    args = ap.parse_args()
    if args.count:
        print(len(tasks()))
    elif args.reference:
        for arm in ARMS:
            print(arm, reference(arm), reference(arm, ind.DEEP), flush=True)
    elif args.deep:
        print("done", run_deep(args.deep), flush=True)
    elif args.pbd_fit_tofu:
        for tofu in args.pbd_fit_tofu[1:]:
            pbd_fit_tofu(float(args.pbd_fit_tofu[0]), tofu)
    elif args.pbd_fit_merge:
        print(json.dumps(pbd_fit_merge(float(args.pbd_fit_merge[0]), args.pbd_fit_merge[1]), indent=1, default=float)[:3000])
    elif args.pbd_fit:
        print("done", pbd_brookfield_fit(), flush=True)
    elif args.brookfield:
        print("done", run_brookfield(args.brookfield), flush=True)
    else:
        arm, dx = (args.point[0], float(args.point[1])) if args.point else tasks()[args.task]
        print("done", run_press(arm, dx), flush=True)


if __name__ == "__main__":
    main()
