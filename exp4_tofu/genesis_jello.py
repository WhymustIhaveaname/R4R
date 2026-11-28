"""Genesis 1.5.0 against the Jello video: FEM explicit and implicit, MPM and PBD, nothing fitted.

The cube of jello_sim.py, 4.9 x 4.7 x 4.5 cm on a fixed base, its E from the video's sway alone
(E = rho (f / C)^2, f = 8.01 Hz, C the continuum's sway constant, rho = 1080 kg/m^3, nu = 0.45),
is put in four of Genesis's methods.  Each starts in its own lowest sway and is read as
jello_sim.run reads MuJoCo (js.measure_at): the sway's frequency and damping, against the video's
8.01 Hz and 0.015.  Each runs twice, nominal (the material's E and damping as they are) and
renormalized (the parameters the theory gives; nothing is fitted to the video):

  gs_fem_explicit  MuJoCo's grid, 10 cells a side, masses lumped by volume.  E* = E S T(h), with
                   S = (C / C_P1)^2 from the P1 sway on this mesh with Genesis's masses, and T(h)
                   of symplectic Euler with the decay exp(-c h), c = 2 zeta w; zeta* = zeta.
  gs_fem_implicit  the same mesh; E* = E S T(h) and beta* = beta - h + h^2 beta w^2 / 12, backward
                   Euler's, as MuJoCo's discrete in jello_sim.py.
  gs_mpm           a 5 mm grid and Genesis's regular sampler at 2.5 mm, 20 x 19 x 18 particles.
                   MPM has no pinning: a slab of held particles (is_free = False) under the cube
                   zeroes the grid's velocity on and below the base plane, a sticky floor.  It
                   has no damping parameter either: E* and its own damping come from MLS-MPM's
                   step linearized about rest (mpm_linear.LinearMPM, with those nodes held), and
                   h is the step at which that damping is the video's.
  gs_pbd           MuJoCo's grid, equal masses, the base particles fixed.  No damping parameter:
                   E* (the edges' compliance) and its own damping come from the PBD step
                   linearized about rest (pbd_linear.py, with the pins), and h is the step at
                   which that damping is the video's.

  python genesis_jello.py gs_fem_explicit      # or gs_fem_implicit, gs_mpm, gs_pbd; --theory
"""

import json
import sys
import time

import numpy as np
import torch
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh

import common as c
import jello_model as jm
import jello_sim as js
import mpm_linear as ml
import p1
import pbd_linear as pl
from genesis_sim import init, oriented, set_field

SIDES = jm.SIDES
F_VIDEO, ZETA_VIDEO, RHO, NU = js.F_VIDEO, js.ZETA_VIDEO, js.RHO, js.NU
W = 2 * np.pi * F_VIDEO
DURATION = 2.555          # s: the video's sway (plot_jello.video_sway) is 2.545 s long, plus 0.01 as plot_jello
V_PEAK = 1e-3             # m/s, the start's peak velocity, as jello_sim.run: 20 um of sway, strain 4e-4
ARMS = ("gs_fem_explicit", "gs_fem_implicit", "gs_mpm", "gs_pbd")
H_IMPLICIT = 1e-3         # s, as MuJoCo's discrete in fig_jello
EXPLICIT_FRACTION = 0.5   # of the explicit FEM's stability limit 2 / w_max
MPM_DX, MPM_P = 0.005, 0.0025
OUT = c.RESULTS


def out_path(arm, ext="json"):
    return OUT / f"jello_{arm}.{ext}"


def material():
    """E (Pa) from the video's sway and the continuum's constant, cached (it takes a minute)."""
    path = c.CACHE / "jello_continuum.json"
    if path.exists():
        C = json.loads(path.read_text())["C"]
    else:
        C = js.continuum_constant()
        c.CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"C": C, "nu": NU}))
    return C, RHO * (F_VIDEO / C) ** 2


def fit_sway(t, q):
    f, z = js.measure_at(t, q, W)
    return f / (2 * np.pi), z


# ---- FEM ------------------------------------------------------------------------------------

def fem_mesh():
    """MuJoCo's grid of the cube (jello_sim.xml), centred on the origin, oriented for Genesis."""
    X, tets = p1.mesh_from_mujoco(None, js.xml(1.0, 1e-3, 1e-3))
    return X, oriented(X, tets)


def fem_sway(X, tets):
    """The P1 sway at E / rho = 1 with Genesis's masses (a quarter of each tetrahedron's): its f (Hz),
    its shape, the masses per unit density, and the highest frequency w_max (rad/s)."""
    K, vol = p1.assemble(X, tets, 1.0, NU)
    m = np.zeros(len(X))
    np.add.at(m, tets.ravel(), np.repeat(vol / 4, 4))
    free = np.flatnonzero(np.repeat(X[:, 2] > X[:, 2].min() + 1e-9, 3))
    Kf, Mf = K[free][:, free], diags(np.repeat(m, 3)[free])
    w2, V = eigsh(Kf, k=4, M=Mf, sigma=0.0, which="LM")
    j = int(np.argmin(w2))
    shape = np.zeros((len(X), 3))
    shape.ravel()[free] = V[:, j]
    w2max = eigsh(Kf, k=1, M=Mf, which="LA", return_eigenvectors=False)[0]
    return float(np.sqrt(w2[j]) / (2 * np.pi)), shape, m, float(np.sqrt(w2max))


def explicit_T(h):
    """Symplectic Euler then v <- exp(-c h) v, c = 2 zeta w (common.time_prediction)."""
    cc = 2 * ZETA_VIDEO * W
    return 1 + h * cc / 2 + h**2 * (2 * cc**2 - W**2) / 12


def fem_theory(arm):
    C, E = material()
    X, tets = fem_mesh()
    C_gs, shape, m, wmax1 = fem_sway(X, tets)
    S = (C / C_gs) ** 2
    if arm == "gs_fem_implicit":
        h = H_IMPLICIT
        T, beta_star = js.backward_euler(W, ZETA_VIDEO, h)
        nominal, renorm = 2 * ZETA_VIDEO / W, beta_star
        h_max = None
    else:
        h_max = 2 / (wmax1 * np.sqrt(E * S / RHO))
        h = float(f"{EXPLICIT_FRACTION * h_max:.2g}")
        T = explicit_T(h)
        nominal = renorm = 2 * ZETA_VIDEO * W
    return {"arm": arm, "E": E, "C": C, "C_P1": C_gs, "S": S, "T": T, "E_ratio": S * T, "h": h,
            "h_max": h_max, "damping_nominal": nominal, "damping_renormalized": renorm,
            "cells": [js.CELLS] * 3, "n_vertices": len(X), "n_tets": len(tets)}


class JelloFem:
    """Genesis's FEM (linear material) on MuJoCo's grid of the cube, as genesis_sim.GsBlock builds
    the tofu block, its base vertices held by hard vertex constraints."""

    def __init__(self, arm, h, E, damping):
        init()
        import genesis as gs
        import genesis.utils.element as eu
        from genesis.utils.misc import qd_to_numpy
        self.arm, self.h = arm, h
        self.implicit = arm == "gs_fem_implicit"
        X, tets = fem_mesh()
        n_bottom = int(np.sum(np.isclose(X[:, 2], X[:, 2].min())))
        keep = (eu.mesh_to_elements, eu.split_all_surface_tets)

        def grid(surface, tet_cfg=None):
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
                    damping=0.0 if self.implicit else damping,
                    damping_alpha=0.0,
                    damping_beta=abs(damping) if self.implicit else 0.0,   # nonzero, or compiled out
                    max_vertex_constraints=n_bottom),
                show_viewer=False)
            self.ent = self.scene.add_entity(
                gs.morphs.Box(pos=(0.0, 0.0, 0.0), size=tuple(float(s) for s in SIDES)),
                gs.materials.FEM.Elastic(E=E, nu=NU, rho=RHO, model="linear"))
            self.scene.build()
        finally:
            eu.mesh_to_elements, eu.split_all_surface_tets = keep
        self.info = self.scene.sim.fem_solver.fem_info
        self.mu0 = qd_to_numpy(self.info.elements.mu).copy()
        self.lam0 = qd_to_numpy(self.info.elements.lam).copy()
        masses = qd_to_numpy(self.info.vertices.mass).copy()
        self.X0 = self.ent.get_verts_pos().cpu().numpy().reshape(-1, 3).astype(np.float64)
        assert np.allclose(self.X0, X[self.order])
        self.bottom = np.flatnonzero(np.isclose(self.X0[:, 2], self.X0[:, 2].min()))
        self.ent.set_vertex_constraints(self.bottom)
        _, shape, m, _ = fem_sway(X, tets)
        shape = shape[self.order]
        assert np.allclose(masses, RHO * m[self.order])
        self.v0 = V_PEAK * shape / np.abs(shape).max()
        dev = self.ent.get_verts_pos().device
        self.w_t = torch.as_tensor(c.modal_weights(shape, masses), dtype=torch.float64, device=dev)
        self.X0_t = torch.as_tensor(self.X0, dtype=torch.float64, device=dev)

    def set_params(self, E_ratio, damping):
        set_field(self.info.elements.mu, self.mu0 * E_ratio)
        set_field(self.info.elements.lam, self.lam0 * E_ratio)
        if self.implicit:
            set_field(self.info.damping_beta_over_dt, damping / self.h)
            set_field(self.info.damping_beta_factor, damping / self.h + 1.0)
        else:
            set_field(self.info.damping_decay, np.exp(-self.h * damping))

    def run(self, E_ratio, damping, duration=DURATION):
        """(t, q, the base's largest displacement) of the cube started in its P1 sway."""
        self.set_params(E_ratio, damping)
        self.scene.reset()
        self.ent.set_velocity(torch.as_tensor(self.v0, dtype=torch.float64, device=self.X0_t.device))
        n = int(round(duration / self.h))
        q = np.empty(n)
        base = 0.0
        for j in range(n):
            self.scene.step()
            u = self.ent.get_verts_pos().reshape(-1, 3) - self.X0_t
            q[j] = float((self.w_t * u).sum())
            if j % 100 == 0:
                base = max(base, float(u[self.bottom].abs().max()))
        return np.arange(1, n + 1) * self.h, q, base


# ---- MPM ------------------------------------------------------------------------------------

def mpm_particles():
    """Genesis's regular sampler on the cube, its base on the grid plane z = 0 (utils/particle.py):
    round(side / p) particles a side, spread evenly from p / 2 to side - p / 2."""
    n = [max(1, round(s / MPM_P)) for s in SIDES]
    axes = [np.linspace(MPM_P / 2, s - MPM_P / 2, k) for s, k in zip(SIDES, n)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)


class JelloLinearMPM(ml.LinearMPM):
    """mpm_linear.LinearMPM on given particles, of volume p^3 each as Genesis gives them, with the
    grid nodes on and below the base plane z = 0 held: the held slab zeroes their velocity every
    step, so they drop out of the state (U, V)."""

    def __init__(self, X, p, dx, E, nu=NU, rho=RHO):
        from scipy.sparse import bmat, identity, kron
        vol = p**3
        m = rho * vol
        fx = X / dx
        base = np.floor(fx - 0.5).astype(int)
        f = fx - base
        wts = np.stack([0.5 * (1.5 - f) ** 2, 0.75 - (f - 1.0) ** 2, 0.5 * (f - 0.5) ** 2], 1)
        offs = np.stack(np.meshgrid(*[np.arange(3)] * 3, indexing="ij"), -1).reshape(-1, 3)
        node = base[:, None, :] + offs[None]
        w = wts[:, offs[:, 0], 0] * wts[:, offs[:, 1], 1] * wts[:, offs[:, 2], 2]
        d = (offs[None] - f[:, None, :]) * dx
        keys, inv = np.unique(node.reshape(-1, 3), axis=0, return_inverse=True)
        inv = inv.reshape(len(X), 27)
        free = keys[:, 2] > 0
        new = np.cumsum(free) - 1
        keep = free[inv]
        rows = new[inv][keep]
        cols = np.broadcast_to(np.arange(len(X))[:, None], inv.shape)[keep]
        ng, npart = int(free.sum()), len(X)
        self.nodes = keys[free] * dx
        self.X, self.dx, self.m, self.vol = X, dx, m, vol
        W = coo_matrix((w[keep], (rows, cols)), shape=(ng, npart)).tocsr()
        Qb = [coo_matrix(((w * d[:, :, b])[keep], (rows, cols)), shape=(ng, npart)).tocsr() for b in range(3)]
        self.W, self.Qb, self.ng, self.npart = W, Qb, ng, npart
        self.mass = m * np.asarray(W.sum(1)).ravel()
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        I3 = identity(3, format="csr")
        self.B = kron(I3, m * (W @ W.T) + m * 4 / dx**2 * sum(Q @ Q.T for Q in Qb)).tocsr()
        blocks = [[None] * 3 for _ in range(3)]
        for a in range(3):
            for e in range(3):
                acc = None
                for b in range(3):
                    terms = []
                    if e == a:
                        terms.append(mu * Qb[b] @ Qb[b].T)
                    if e == b:
                        terms.append(mu * Qb[b] @ Qb[a].T)
                    if a == b:
                        terms.append(lam * Qb[b] @ Qb[e].T)
                    for t in terms:
                        acc = t if acc is None else acc + t
                blocks[a][e] = acc
        self.K = vol * (4 / dx**2) ** 2 * bmat(blocks).tocsr()
        self.M = diags(np.tile(self.mass, 3))

    def sway(self, h, ratio, U0, sigma):
        """(w_n, zeta, lambda, U) of the step's mode nearest U0, U by component, real, peak 1."""
        lam, U = self.eigen(h, ratio, sigma, U0)
        U = U * np.exp(-1j * np.angle(U[np.argmax(np.abs(U))]))
        U = U.real / np.abs(U).max()
        sd, wd = -np.log(abs(lam)) / h, np.angle(lam) / h
        wn = np.hypot(sd, wd)
        return wn, sd / wn, lam, U


def sway_direction():
    """0 or 1: the axis along which the cube's lowest P1 sway moves."""
    X, tets = fem_mesh()
    _, shape, _, _ = fem_sway(X, tets)
    return int(np.argmax(np.abs(shape[:, :2]).sum(0)))


def mpm_mode(L, h, E_ratio=None, start=None, f_guess=F_VIDEO, r_start=1.0):
    """E*/E that brings the linearized step's sway to 8.01 Hz (by the secant method from r_start,
    or the given E_ratio), with its (w_n, zeta, lambda, U); the search starts from the shape
    `start` (by component) at the frequency f_guess."""
    if start is None:
        a = sway_direction()
        z = L.nodes[:, 2] / SIDES[2]
        guess = np.zeros((L.ng, 3))
        guess[:, a] = 1 - np.cos(np.pi * z / 2)
        start = guess.T.ravel()
    state = {"U": start, "lam": np.exp(2j * np.pi * f_guess * h)}

    def resid(r):
        wn, z, lam, U = L.sway(h, r, state["U"], state["lam"])
        state.update(U=U, lam=lam, out=(wn, z, lam, U))
        return np.log(wn / W)
    if E_ratio is not None:
        resid(E_ratio)
        return E_ratio, state["out"]
    r0, r1 = r_start, r_start * 1.005
    f0, f1 = resid(r0), resid(r1)
    for _ in range(30):
        r0, f0, r1 = r1, f1, r1 - f1 * (r1 - r0) / (f1 - f0)
        f1 = resid(r1)
        if abs(f1) < 1e-10:
            break
    return r1, state["out"]


MPM_PROBES = (1e-4, 3e-4)   # s: the steps at which the linearized damping is first computed


def mpm_theory():
    """E*/E and MLS-MPM's own zeta at the step whose zeta is the video's, cached on disk.

    zeta goes as a power of h (about h^-0.85 here), so the step is interpolated in log-log
    between two probes, and the linearized step is solved again there: its zeta is the one
    reported, not the target."""
    C, E = material()
    X = mpm_particles()
    L = JelloLinearMPM(X, MPM_P, MPM_DX, E)
    path = c.CACHE / f"jello_mpm_theory_{MPM_DX * 1e3:g}mm_{MPM_P * 1e3:g}mm.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        out = json.loads(str(z["out"]))
        return out, L, z["U"]
    probes, U, r = [], None, 1.0
    for h in MPM_PROBES:
        r, (wn, zeta, lam, U) = mpm_mode(L, h, start=U, r_start=r)
        probes.append((h, r, zeta))
        print(f"MPM probe h {h * 1e6:.1f} us: E*/E {r:.6f}, zeta {zeta:.6f}", flush=True)
    (h0, _, z0), (h1, _, z1) = probes
    slope = np.log(z1 / z0) / np.log(h1 / h0)
    h = float(f"{h0 * (ZETA_VIDEO / z0) ** (1 / slope):.3g}")
    r, (wn, zeta, lam, U) = mpm_mode(L, h, start=U, r_start=probes[0][1])
    _, (wn0, zeta0, _, _) = mpm_mode(L, h, E_ratio=1.0, start=U, f_guess=F_VIDEO / np.sqrt(r))
    out = {"arm": "gs_mpm", "E": E, "C": C, "h": h, "dx": MPM_DX, "particle_size": MPM_P,
           "n_particles": len(X), "particles_per_side": [int(round(s / MPM_P)) for s in SIDES],
           "n_grid_nodes_free": L.ng, "probes": probes, "zeta_slope": slope,
           "E_ratio": r, "zeta_theory": zeta, "f_theory": wn / (2 * np.pi),
           "f_theory_nominal": wn0 / (2 * np.pi), "zeta_theory_nominal": zeta0,
           "damping_nominal": None, "damping_renormalized": None,
           "cfl": h * np.sqrt(E * r * (1 - NU) / ((1 + NU) * (1 - 2 * NU)) / RHO) / MPM_DX}
    np.savez(path, out=json.dumps(out), U=U)
    return out, L, U


class JelloMpm:
    """The cube in Genesis's MPM, its base held by a slab of fixed particles below it.

    A particle held in place (is_free = False) zeroes the grid's momentum over its whole stencil,
    27 nodes.  The slab's particles lie between 1.5 and 0.5 cells below the base plane, so the
    nodes they zero are those on and below it: the cube's own particles are all free, and the
    grid holds the plane z = 0, as the FEM's pins hold its base vertices.  E is compiled into the
    kernels, so a rollout with E* is the one with E at the step h sqrt(E* / E) and v0 / sqrt(E* / E),
    as genesis_sim.GsMpmBlock does.
    """

    def __init__(self, h, E, U_nodes, L):
        init()
        import genesis as gs
        from genesis.utils.misc import qd_to_numpy
        self.h = h
        margin = 8 * MPM_DX
        mat = gs.materials.MPM.Elastic(E=E, nu=NU, rho=RHO, sampler="regular")
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=h, substeps=1, gravity=(0.0, 0.0, 0.0)),
            mpm_options=gs.options.MPMOptions(
                grid_density=1.0 / MPM_DX, particle_size=MPM_P,
                lower_bound=(-margin, -margin, -margin),
                upper_bound=tuple(float(s + margin) for s in SIDES)),
            show_viewer=False)
        self.ent = self.scene.add_entity(
            gs.morphs.Box(pos=tuple(float(s / 2) for s in SIDES), size=tuple(float(s) for s in SIDES)), mat)
        self.slab = self.scene.add_entity(
            gs.morphs.Box(pos=(float(SIDES[0] / 2), float(SIDES[1] / 2), -0.75 * MPM_DX),
                          size=(float(SIDES[0] + 4 * MPM_DX), float(SIDES[1] + 4 * MPM_DX), MPM_DX / 2)), mat)
        self.scene.build()
        self.hold()
        self.info = self.scene.sim.mpm_solver.mpm_info
        self.stress_scale0 = float(qd_to_numpy(self.info.stress_scale))
        pos = self.ent.get_particles_pos().reshape(-1, 3)
        self.X0_t = pos.to(torch.float64)
        X = self.X0_t.cpu().numpy()
        assert np.allclose(X, L.X), "Genesis sampled the cube elsewhere than the theory"
        self.slab_X0 = self.slab.get_particles_pos().reshape(-1, 3).to(torch.float64)
        zs = self.slab_X0[:, 2].cpu().numpy()
        assert np.all((zs >= -1.5 * MPM_DX) & (zs < -0.5 * MPM_DX))
        ng = L.ng
        shape = np.stack([L.W.T @ U_nodes[a * ng:(a + 1) * ng] for a in range(3)], 1)
        self.v0 = V_PEAK * shape / np.abs(shape).max()
        self.w_t = torch.as_tensor(c.modal_weights(shape, np.ones(len(X))), dtype=torch.float64, device=pos.device)

    def hold(self):
        n = self.slab.n_particles
        self.slab.set_free(torch.zeros(n, dtype=torch.bool, device=self.slab.get_particles_pos().device))

    def run(self, E_ratio, damping=None, duration=DURATION):
        s = float(np.sqrt(E_ratio))
        set_field(self.info.substep_dt, self.h * s)
        set_field(self.info.stress_scale, self.stress_scale0 * s)
        self.scene.reset()
        self.hold()
        self.ent.set_particles_vel(torch.as_tensor(self.v0 / s, dtype=torch.float64, device=self.X0_t.device))
        n = int(round(duration / self.h))
        q = np.empty(n)
        for j in range(n):
            self.scene.step()
            u = self.ent.get_particles_pos().reshape(-1, 3) - self.X0_t
            q[j] = float((self.w_t * u).sum())
        slab = float((self.slab.get_particles_pos().reshape(-1, 3).to(torch.float64) - self.slab_X0).abs().max())
        return np.arange(1, n + 1) * self.h, q, slab


# ---- PBD ------------------------------------------------------------------------------------

def pbd_network():
    """MuJoCo's grid of the cube, its edges and their directions, the base particles, the mass."""
    X, tets = p1.mesh_from_mujoco(None, js.xml(1.0, 1e-3, 1e-3))
    e = {tuple(sorted((t[a], t[b]))) for t in tets for a in range(4) for b in range(a + 1, 4)}
    e = np.array(sorted(e))
    n = X[e[:, 0]] - X[e[:, 1]]
    n /= np.linalg.norm(n, axis=1)[:, None]
    fixed = np.isclose(X[:, 2], X[:, 2].min())
    m = RHO * np.prod(SIDES) / len(X)
    return X, tets, e, n, fixed, m


def pbd_dx():
    """The edge length the nominal compliance 1 / (E dx) takes: the cell's geometric mean."""
    return float(np.prod(SIDES / js.CELLS) ** (1 / 3))


def pbd_sweep_matrix(net, a):
    """A, one Jacobi sweep about rest being u <- (I - A) u on the free particles' coordinates.

    An edge of two free particles moves each by r C / (2 + a), a = m alpha / h^2; an edge from a
    free particle to a fixed one (w = 0) moves the free one by r C / (1 + a)."""
    X, tets, e, n, fixed, m = net
    free = ~fixed
    idx = -np.ones(len(X), int)
    idx[free] = np.arange(free.sum())
    ff = free[e[:, 0]] & free[e[:, 1]]
    coef = np.where(ff, pl.RELAX / (2 + a), pl.RELAX / (1 + a))
    rows, cols, vals = [], [], []
    for a_ in range(3):
        for b in range(3):
            k = coef * n[:, a_] * n[:, b]
            for (i, j, s) in ((0, 0, 1), (1, 1, 1), (0, 1, -1), (1, 0, -1)):
                vi, vj = idx[e[:, i]], idx[e[:, j]]
                ok = (vi >= 0) & (vj >= 0)
                rows.append(3 * vi[ok] + a_)
                cols.append(3 * vj[ok] + b)
                vals.append(s * k[ok])
    nf = 3 * int(free.sum())
    return coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nf, nf)).tocsr()


def pbd_lowest(net, a):
    """The lowest eigenvalue mu of A, the sway, and its mode on all particles (zero on the pins)."""
    A = pbd_sweep_matrix(net, a)
    v0 = np.random.default_rng(0).standard_normal(A.shape[0])
    w, V = eigsh(A, k=3, sigma=-1e-6, which="LM", v0=v0, tol=1e-13)
    j = int(np.argmin(w))
    X, _, _, _, fixed, _ = net
    shape = np.zeros((len(X), 3))
    shape[~fixed] = V[:, j].reshape(-1, 3)
    return float(w[j]), shape


def pbd_e_star(net, h):
    """(E*/E, w_n, zeta, shape) at the compliance that gives 8.01 Hz, or None if none does."""
    from scipy.optimize import brentq
    m = net[5]
    a0 = 1 / (material()[1] * pbd_dx())

    def wz(alpha):
        mu, shape = pbd_lowest(net, m * alpha / h**2)
        return _pbd_mode(mu, h), shape

    top = _pbd_mode(pbd_lowest(net, 0.0)[0], h)
    if top is None or top[0] < W:
        return None
    lg = brentq(lambda lg: wz(a0 / 10**lg)[0][0] - W, -6.0, 12.0, xtol=1e-13)
    (wn, z), shape = wz(a0 / 10**lg)
    return 10**lg, wn, z, shape


def _pbd_mode(mu, h, n_s=pl.N_STRETCH):
    """pbd_linear.mode for a sweep eigenvalue mu already scaled by its compliance."""
    p = (1 - mu) ** n_s
    if not 0 < p < 1:
        return None
    wd = np.arccos(np.sqrt(p)) / h
    s = -np.log(p) / (2 * h)
    wn = np.hypot(wd, s)
    return wn, s / wn


def pbd_theory(h=None):
    from scipy.optimize import brentq
    C, E = material()
    net = pbd_network()
    mu0 = pbd_lowest(net, 0.0)[0]
    p0 = (1 - mu0) ** pl.N_STRETCH
    h_max = np.hypot(np.arccos(np.sqrt(p0)), np.log(p0) / 2) / W
    if h is None:
        h = brentq(lambda lh: pbd_e_star(net, np.exp(lh))[2] - ZETA_VIDEO,
                   np.log(0.05 * h_max), np.log(0.999 * h_max), xtol=1e-10)
        h = float(f"{np.exp(h):.4g}")
    r, wn, z, shape = pbd_e_star(net, h)
    a = net[5] / (E * pbd_dx() * h**2)
    out0 = _pbd_mode(pbd_lowest(net, a)[0], h)
    return {"arm": "gs_pbd", "E": E, "C": C, "h": h, "h_max": float(h_max), "cells": [js.CELLS] * 3,
            "n_particles": len(net[0]), "dx_nominal": pbd_dx(), "alpha_nominal": 1 / (E * pbd_dx()),
            "E_ratio": r, "f_theory": wn / (2 * np.pi), "zeta_theory": z,
            "f_theory_nominal": None if out0 is None else out0[0] / (2 * np.pi),
            "zeta_theory_nominal": None if out0 is None else out0[1],
            "damping_nominal": None, "damping_renormalized": None}, shape


class JelloPbd:
    """The cube as Genesis's elastic PBD on MuJoCo's grid, as genesis_sim.GsPbdBlock builds the tofu
    block, its base particles fixed (fix_particles, which sets their inverse mass to zero)."""

    def __init__(self, h, E, shape):
        init()
        import genesis as gs
        from genesis.engine.mesh import Mesh
        from genesis.utils.misc import qd_to_numpy
        self.h = h
        X, tets, _, _, fixed, m = pbd_network()
        keep = Mesh.tetrahedralize
        Mesh.tetrahedralize = lambda self_, cfg: (X.copy(), tets.copy())
        try:
            self.scene = gs.Scene(
                sim_options=gs.options.SimOptions(dt=h, substeps=1, gravity=(0.0, 0.0, 0.0)),
                pbd_options=gs.options.PBDOptions(
                    particle_size=float(SIDES.min() / js.CELLS / 2), lower_bound=(-1.0, -1.0, -1.0),
                    upper_bound=(1.0, 1.0, 1.0), max_stretch_solver_iterations=pl.N_STRETCH),
                show_viewer=False)
            self.ent = self.scene.add_entity(
                gs.morphs.Box(pos=(0.0, 0.0, 0.0), size=tuple(float(s) for s in SIDES)),
                gs.materials.PBD.Elastic(rho=RHO, stretch_compliance=1 / (E * pbd_dx())))
            self.scene.build()
        finally:
            Mesh.tetrahedralize = keep
        self.info = self.scene.sim.pbd_solver.pbd_info
        self.alpha0 = qd_to_numpy(self.info.edges.stretch_compliance).copy()
        pos = self.ent.get_particles_pos().reshape(-1, 3)
        self.X0_t = pos.to(torch.float64)
        assert np.allclose(self.X0_t.cpu().numpy(), X)
        assert np.allclose(qd_to_numpy(self.info.particles.mass), m)
        self.bottom = np.flatnonzero(fixed)
        self.v0 = V_PEAK * shape / np.abs(shape).max()
        self.w_t = torch.as_tensor(c.modal_weights(shape, np.full(len(X), m)), dtype=torch.float64,
                                   device=pos.device)

    def run(self, E_ratio, damping=None, duration=DURATION):
        set_field(self.info.edges.stretch_compliance, self.alpha0 / E_ratio)
        self.scene.reset()
        self.ent.set_particles_vel(torch.as_tensor(self.v0, dtype=torch.float64, device=self.X0_t.device))
        self.ent.fix_particles(self.bottom)
        n = int(round(duration / self.h))
        q = np.empty(n)
        for j in range(n):
            self.scene.step()
            u = self.ent.get_particles_pos().reshape(-1, 3) - self.X0_t
            q[j] = float((self.w_t * u).sum())
        base = float((self.ent.get_particles_pos().reshape(-1, 3) - self.X0_t)[self.bottom].abs().max())
        return np.arange(1, n + 1) * self.h, q, base


# ---- the runs -------------------------------------------------------------------------------

def theory(arm):
    if arm.startswith("gs_fem"):
        return fem_theory(arm), None
    if arm == "gs_mpm":
        out, L, U = mpm_theory()
        return out, (L, U)
    out, shape = pbd_theory()
    return out, shape


def main(arm, theory_only=False, duration=DURATION):
    t_start = time.time()
    th, extra = theory(arm)
    th["theory_seconds"] = time.time() - t_start
    print(json.dumps(th, indent=1), flush=True)
    if theory_only:
        return
    E, h = th["E"], th["h"]
    t0 = time.time()
    if arm.startswith("gs_fem"):
        sim = JelloFem(arm, h, E, th["damping_nominal"])
    elif arm == "gs_mpm":
        L, U = extra
        sim = JelloMpm(h, E, U, L)
    else:
        sim = JelloPbd(h, E, extra)
    th["build_seconds"] = time.time() - t0
    series = {}
    for name, (Er, damp) in {"nominal": (1.0, th["damping_nominal"]),
                             "renormalized": (th["E_ratio"], th["damping_renormalized"])}.items():
        t0 = time.time()
        t, q, base = sim.run(Er, damp, duration)
        f, z = fit_sway(t, q) if np.all(np.isfinite(q)) else (np.nan, np.nan)
        th[name] = {"f": f, "zeta": z, "seconds": time.time() - t0, "base_max_displacement": base,
                    "steps": len(t), "q_max": float(np.nanmax(np.abs(q)))}
        series[name] = q
        series["t"] = t
        print(f"{arm} {name}: f {f:.4f} Hz ({f / F_VIDEO - 1:+.2%}), zeta {z:.5f}"
              f"  ({time.time() - t0:.0f} s, base moved {base:.2e} m)", flush=True)
    th["total_seconds"] = time.time() - t_start
    out_path(arm).write_text(json.dumps(th, indent=1))
    np.savez(out_path(arm, "npz"), **series)
    print("wrote", out_path(arm), flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for a in args:
        assert a in ARMS, a
        main(a, theory_only="--theory" in sys.argv)
