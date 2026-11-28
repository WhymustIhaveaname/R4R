"""Speed against accuracy on the Jello cube, for Genesis 1.5.0's FEM (explicit, implicit), MPM and PBD.

The cube of jello_sim.py (4.9 x 4.7 x 4.5 cm, base fixed, rho 1080, nu 0.45, E from the video's
sway and the continuum's constant), with two knobs, n cells per side and the step h, as
speed_accuracy_mujoco.py turns them for MuJoCo:

  gs_fem_explicit  MuJoCo's grid of n^3 cells, six P1 tetrahedra each, masses lumped by volume;
                   h = 0.25, 0.5, 0.9 of the stability limit 2 / w_max at E*
  gs_fem_implicit  the same mesh, backward Euler; h = 0.25, 0.5, 1 ms
  gs_mpm           a grid of spacing (4.9 x 4.7 x 4.5 cm^3)^(1/3) / n, Genesis's regular sampler at
                   half of it, the base held by a slab of fixed particles (genesis_jello.JelloMpm);
                   h = 0.25, 0.5, 0.9 of the limit 2 / w_max of its linearized step at E*, and
                   Genesis's suggested 0.02 dx (s, dx in m)
  gs_pbd           MuJoCo's grid, equal masses, base particles fixed (genesis_jello.JelloPbd);
                   h = 0.25, 0.5, 0.9 of h_max, above which no compliance reaches the video's sway

The parameters are first renormalized for (n, h), nothing fitted.  FEM: S = (C / C_sway)^2 from the
mesh's own modal analysis with masses lumped by volume, T(h) and the damping counterterm from the
time theory, at the video's sway (8.01 Hz, zeta 0.015), as genesis_jello.py.  MPM: S from the
stiffness and masses of its step linearized about rest (h -> 0) and symplectic Euler's T(h).  PBD:
the compliance at which its linearized step sways at 8.01 Hz.  MPM and PBD damp as they do.

Each run is 1.5 s, started in the model's own sway or its own twist (the lowest elastic mode whose
motion is a rotation about the vertical, jello_model.modes's test) and read by
jello_sim.measure_at.  The two residuals, against the clip top_front_pluck (sway 8.010 Hz and
zeta 0.0151, twist 9.992 Hz and 0.0154):

  frequency  a stiffness multiplier lam on E* fitted to the two frequencies; 100 x the RMS of the
             two relative errors
  damping    at that lam, every damping parameter the engine offers fitted to the two damping
             ratios (FEM explicit: c; FEM implicit: Rayleigh alpha and beta; MPM, PBD: none); 100 x
             the RMS of the two relative errors

Each fit predicts from a model (f ~ sqrt(lam); zeta linear in the damping parameters; PBD's
frequencies from its linearized step), corrected by the last runs, and is refined with real runs
until it stops moving.  Both residuals come from one last pair of runs at the final parameters.

Speed = simulated seconds / wall seconds of the stepping loop of the renormalized sway run, after
10 warm-up steps (build and compile not counted); the readout, every 0.5 ms simulated, stays on
the GPU and is counted.  If the first 0.05 s simulated (or the first 5 s of wall time) run slower
than 1/100 of real time, the point stops and is saved with "cut": true and that speed.

  python speed_accuracy_genesis.py --count
  python speed_accuracy_genesis.py --task K        # one point; a Slurm array index
"""

import argparse
import json
import os
import platform
import socket
import time
from functools import lru_cache

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

import common as c
import genesis_jello as gj
import jello_model as jm
import jello_sim as js
import p1
import pbd_linear as pl
from genesis_sim import init, oriented, set_field

OUT = c.RESULTS / "speed_accuracy"
OUT_TIMING = c.RESULTS / "speed_accuracy_timing"     # --timing-only: the timed run alone
OUT_FIT = c.RESULTS / "speed_accuracy_fit"           # --no-cut: the fits, untimed (merged by --merge)
CUTOFF = True
CELLS = (3, 4, 5, 6, 8, 10, 12, 16, 20)
H_IMPLICIT = (0.25e-3, 0.5e-3, 1e-3)
FRACTIONS = (0.25, 0.5, 0.9)
METHODS = ("gs_fem_explicit", "gs_fem_implicit", "gs_mpm", "gs_pbd")
DURATION = 1.5
CUT_SIM, CUT_WALL, CUT_SPEED, WARMUP = 0.05, 5.0, 0.01, 10
READ_EVERY = 5e-4                          # s simulated between readouts
TARGET = {"f_sway": 8.010, "zeta_sway": 0.0151, "f_twist": 9.992, "zeta_twist": 0.0154}
SIDES, RHO, NU = jm.SIDES, js.RHO, js.NU
W, ZETA = 2 * np.pi * js.F_VIDEO, js.ZETA_VIDEO   # the renormalization's sway, as genesis_jello.py
GM = float(np.prod(SIDES) ** (1 / 3))      # m, the side of the cube of the same volume


@lru_cache(maxsize=None)
def continuum():
    """(C_sway, C_twist, E): the continuum's f at E / rho = 1 (jello_mujoco_variants.continuum_modes,
    cached by speed_accuracy_mujoco.py) and E from the video's sway."""
    path = c.CACHE / "jello_continuum_modes.json"
    if path.exists():
        cs, ct = json.loads(path.read_text())
    else:
        import jello_mujoco_variants as jv
        cs, ct = jv.continuum_modes()
        c.CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([cs, ct]))
    return cs, ct, RHO * (js.F_VIDEO / cs) ** 2


ZETA_STEP = 0.0151                         # MPM's and PBD's own sway damping at the step "zeta"
MPM_SUGGESTED = (0.5, 1.0)                 # of Genesis's suggested MPM substep, 0.02 dx (s, dx in m)


def tasks():
    """(method, n, k).  FEM: k indexes the steps (FRACTIONS of the limit, or H_IMPLICIT).  MPM: k = 0
    is the step at which its own sway damping is ZETA_STEP, k = 1, 2 the MPM_SUGGESTED steps.  PBD:
    k = 0, 1, 2 the FRACTIONS of h_max, k = 3 the step at which its own sway damping is ZETA_STEP."""
    out = []
    for method in METHODS:
        k_max = 3 if method == "gs_mpm" else 4 if method == "gs_pbd" else 3
        for n in CELLS:
            for k in range(k_max):
                out.append((method, n, k))
    return out


def step_rule(method, k):
    if method == "gs_fem_explicit":
        return f"{FRACTIONS[k]} x 2/w_max"
    if method == "gs_fem_implicit":
        return f"{H_IMPLICIT[k] * 1e3:g} ms"
    if method == "gs_mpm":
        return f"own sway zeta {ZETA_STEP}" if k == 0 else f"{MPM_SUGGESTED[k - 1]} x 0.02 dx"
    return f"{FRACTIONS[k]} x h_max" if k < 3 else f"own sway zeta {ZETA_STEP}"


class NoSuchStep(Exception):
    """The step a point asks for does not exist, or not where the scheme is stable."""


def mpm_zeta_step(n):
    """The step at which MPM's own sway damping is ZETA_STEP on n cells.

    Its damping goes as a power of h (about h^-0.4 to h^-0.9), so the step is interpolated in
    log-log between the two steps of the first sweep that bracket it (renormalized sway runs at
    0.25, 0.5, 0.9 of 2 / w_max and at 0.02 dx, results/speed_accuracy_fit), or extrapolated from
    the two smallest when the damping is below ZETA_STEP everywhere.  When it lies above a step the
    sweep found unstable, there is none.  Cached in results/cache, so the cluster reads the table."""
    path = c.CACHE / "jello_mpm_zeta_steps.json"
    table = json.loads(path.read_text()) if path.exists() else {}
    if str(n) not in table:
        rows = [json.loads(f.read_text()) for f in OUT_FIT.glob(f"gs_mpm_{n}cells_*.json")]
        rows = sorted([r for r in rows if r.get("step_index", 0) <= 3 and "renormalized" in r and r.get("lam_history")],
                      key=lambda r: r["h_s"])
        h = np.array([r["h_s"] for r in rows])
        z = np.array([r["renormalized"]["zeta_sway"] for r in rows])
        ok = np.array([np.isfinite(r.get("err_freq_pct", np.nan)) for r in rows])
        above = np.flatnonzero(z >= ZETA_STEP)
        j = int(above.max()) if len(above) else 0
        j = min(j, len(h) - 2)
        slope = np.log(z[j + 1] / z[j]) / np.log(h[j + 1] / h[j])
        hz = float(f"{h[j] * (ZETA_STEP / z[j]) ** (1 / slope):.4g}")
        unstable = h[~ok]
        table[str(n)] = {"h": hz, "stable_sweep_max_h": float(h[ok].max()),
                         "sweep_unstable_min_h": float(unstable.min()) if len(unstable) else None,
                         "exists": not (len(unstable) and hz > unstable.min() * 1.01),
                         "bracket": [float(h[j]), float(h[j + 1])], "zeta_bracket": [float(z[j]), float(z[j + 1])]}
        c.CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(table, indent=1))
    t = table[str(n)]
    if not t["exists"]:
        raise NoSuchStep(f"MPM on {n} cells reaches zeta {ZETA_STEP} only above {t['sweep_unstable_min_h'] * 1e6:.0f} us, "
                         f"where the sweep was unstable")
    return t["h"]


def classify(P, mass, u):
    """jello_model.modes's test on displacements u (N, 3) at points P with masses."""
    r = P[:, :2] - (mass[:, None] * P[:, :2]).sum(0) / mass.sum()
    twist = np.sum(mass * (r[:, 0] * u[:, 1] - r[:, 1] * u[:, 0])) / np.sqrt(
        np.sum(mass * (r**2).sum(1)) * np.sum(mass * (u[:, :2]**2).sum(1)))
    ux, uy, uz = np.sum(mass[:, None] * u**2, 0)
    if abs(twist) > 0.8:
        return "twist"
    return "sway x" if ux > max(uy, uz) else "sway y" if uy > uz else "vertical"


def twist_metric(P, mass, u):
    r = P[:, :2] - (mass[:, None] * P[:, :2]).sum(0) / mass.sum()
    return float(np.sum(mass * (r[:, 0] * u[:, 1] - r[:, 1] * u[:, 0])) / np.sqrt(
        np.sum(mass * (r**2).sum(1)) * np.sum(mass * (u[:, :2]**2).sum(1))))


def sway_and_twist(w2, V, P, mass, to_nodes, relaxed=False):
    """The lowest sway and twist among eigenpairs (w2, V): {kind: (w, shape (N, 3))}.

    With `relaxed` (PBD), the sway is the lowest mode and the twist the one of the next five
    that turns most about the vertical: the edge springs of MuJoCo's grid, its diagonals all one
    way, mix the two (the lowest mode turns with |twist| ~ 0.4, the twist reaches only 0.6 to 0.7,
    against jello_model's 0.8)."""
    order = np.argsort(w2)
    if relaxed:
        us = [to_nodes(V[:, j]) for j in order[:6]]
        tw = [abs(twist_metric(P, mass, u)) for u in us[1:]]
        j = 1 + int(np.argmax(tw))
        return {"sway": (float(np.sqrt(max(w2[order[0]], 0.0))), us[0]),
                "twist": (float(np.sqrt(max(w2[order[j]], 0.0))), us[j]), "twist_metric": max(tw),
                "sway_twist_metric": abs(twist_metric(P, mass, us[0]))}
    out = {}
    for j in order:
        u = to_nodes(V[:, j])
        kind = classify(P, mass, u)
        key = "sway" if kind.startswith("sway") else kind
        if key in ("sway", "twist") and key not in out:
            out[key] = (float(np.sqrt(max(w2[j], 0.0))), u)
    assert "sway" in out and "twist" in out, "no sway or no twist among the lowest modes"
    return out


def cube_xml(n):
    """jello_sim.xml with n cells a side, for its mesh alone."""
    count = " ".join(str(n + 1) for _ in range(3))
    spacing = " ".join(repr(float(s / n)) for s in SIDES)
    return f"""<mujoco><worldbody>
    <flexcomp name="cube" type="grid" dim="3" dof="full" count="{count}" spacing="{spacing}"
              mass="{float(RHO * np.prod(SIDES))!r}" radius="0.0005">
      <contact contype="0" conaffinity="0" selfcollide="none"/>
      <elasticity young="1e4" poisson="{NU!r}" damping="0.001"/>
      <pin gridrange="0 0 0 {n} {n} 0"/>
    </flexcomp></worldbody></mujoco>"""


@lru_cache(maxsize=None)
def mesh(n):
    X, tets = p1.mesh_from_mujoco(None, cube_xml(n))
    return X, oriented(X, tets)


# ---- theory: the renormalized parameters and the modes to start in -------------------------

def fem_theory(method, n, k):
    """FEM on n cells: E*/E, the damping, h, and the sway and twist at E / rho = 1, masses by volume."""
    cs, ct, E = continuum()
    X, tets = mesh(n)
    K, vol = p1.assemble(X, tets, 1.0, NU)
    m = np.zeros(len(X))
    np.add.at(m, tets.ravel(), np.repeat(vol / 4, 4))
    free = np.flatnonzero(np.repeat(X[:, 2] > X[:, 2].min() + 1e-9, 3))
    Kf, Mf = K[free][:, free], diags(np.repeat(m, 3)[free])
    w2, V = eigsh(Kf, k=min(12, len(free) - 2), M=Mf, sigma=0.0, which="LM")

    def to_nodes(v):
        u = np.zeros(3 * len(X))
        u[free] = v
        return u.reshape(-1, 3)
    modes = sway_and_twist(w2, V, X, m, to_nodes)
    w_max1 = float(np.sqrt(eigsh(Kf, k=1, M=Mf, which="LA", return_eigenvectors=False)[0]))
    S = (cs / (modes["sway"][0] / (2 * np.pi))) ** 2
    out = {"S": S, "f_sway_model_Erho1": modes["sway"][0] / (2 * np.pi),
           "f_twist_model_Erho1": modes["twist"][0] / (2 * np.pi), "w_max_Erho1": w_max1,
           "n_vertices": len(X), "n_elements": len(tets)}
    if method == "gs_fem_implicit":
        h = H_IMPLICIT[k]
        T, beta = js.backward_euler(W, ZETA, h)
        out.update(h=h, T=T, damping={"alpha": 0.0, "beta": beta}, damping_name="alpha (1/s), beta (s)")
    else:
        h = 1e-4
        for _ in range(20):
            T = gj.explicit_T(h)
            limit = 2 / (w_max1 * np.sqrt(E * S * T / RHO))
            h = FRACTIONS[k] * limit
        out.update(h=float(h), T=T, h_limit=float(limit), damping={"c": 2 * ZETA * W}, damping_name="c (1/s)")
    out["E_ratio"] = S * out["T"]
    out["shapes"] = {key: modes[key][1] for key in ("sway", "twist")}
    out["masses"] = RHO * m
    return out


def mpm_setup(n):
    dx = GM / n
    p = dx / 2
    counts = [max(1, round(s / p)) for s in SIDES]
    axes = [np.linspace(p / 2, s - p / 2, q) for s, q in zip(SIDES, counts)]
    X = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    return dx, p, X


SHAPE_CELLS = 16   # the P1 mesh whose sway and twist start MPM, which has no cheap discrete modes


@lru_cache(maxsize=None)
def p1_shapes():
    """The sway and twist of the P1 cube on SHAPE_CELLS cells, as trilinear interpolants."""
    from scipy.interpolate import RegularGridInterpolator
    th = fem_theory("gs_fem_implicit", SHAPE_CELLS, 0)
    X, _ = mesh(SHAPE_CELLS)
    idx = np.round((X - X.min(0)) / (SIDES / SHAPE_CELLS)).astype(int)
    axes = [np.linspace(-a / 2, a / 2, SHAPE_CELLS + 1) for a in SIDES]
    out = {}
    for key in ("sway", "twist"):
        g = np.zeros((SHAPE_CELLS + 1,) * 3 + (3,))
        g[idx[:, 0], idx[:, 1], idx[:, 2]] = th["shapes"][key]
        out[key] = RegularGridInterpolator(axes, g)
    return out


def mpm_theory(n, k):
    """MPM on a grid of GM / n.  Its step linearized about rest is costly to solve on fine grids,
    and its stiffness and masses alone (h -> 0) miss what the transfer does to the frequency (8 %
    at 10 cells), so S comes from the step itself, a run at E (run_point): the sway's f there
    gives E* = E (8.01 Hz / f)^2.  The stability limit is 2 / w_max of the linearized step's
    stiffness and masses at E.  The runs start in the P1 cube's sway and twist (p1_shapes) at the
    particles, as genesis_sim.GsMpmBlock starts in the continuum's mode."""
    cs, ct, E = continuum()
    dx, p, X = mpm_setup(n)
    L = gj.JelloLinearMPM(X, p, dx, 1.0, NU, 1.0)
    s = 1 / np.sqrt(np.tile(L.mass, 3))
    w_max1 = float(np.sqrt(eigsh(diags(s) @ L.K @ diags(s), k=1, which="LA", return_eigenvectors=False)[0]))
    limit = 2 / (w_max1 * np.sqrt(E / RHO))
    h = mpm_zeta_step(n) if k == 0 else MPM_SUGGESTED[k - 1] * 0.02 * dx
    shapes = {key: f(X - SIDES / 2) for key, f in p1_shapes().items()}
    return {"h": float(h), "h_limit": float(limit), "dx": dx, "particle_size": p, "n_particles": len(X),
            "n_grid_nodes_free": L.ng, "w_max_Erho1": w_max1, "E_ratio": None, "T": 1 - h**2 * W**2 / 12,
            "damping": {}, "damping_name": None, "shapes": shapes, "masses": np.ones(len(X)), "X": X,
            "f_guess": {"sway": js.F_VIDEO, "twist": ct * np.sqrt(continuum()[2] / RHO)}}


class PbdTheory:
    """Genesis's PBD step linearized about rest on n cells (genesis_jello.pbd_sweep_matrix)."""

    def __init__(self, n):
        X, tets = p1.mesh_from_mujoco(None, cube_xml(n))
        e = {tuple(sorted((t[a], t[b]))) for t in tets for a in range(4) for b in range(a + 1, 4)}
        e = np.array(sorted(e))
        nv = X[e[:, 0]] - X[e[:, 1]]
        nv /= np.linalg.norm(nv, axis=1)[:, None]
        fixed = np.isclose(X[:, 2], X[:, 2].min())
        self.m = RHO * np.prod(SIDES) / len(X)
        self.net = (X, tets, e, nv, fixed, self.m)
        self.X, self.tets, self.fixed = X, tets, fixed
        self.dx = GM / n
        self.alpha_nom = 1 / (continuum()[2] * self.dx)

    def modes(self, a, k=10):
        """{sway, twist: (mu, shape)} of A at a = m alpha / h^2."""
        A = gj.pbd_sweep_matrix(self.net, a)
        v0 = np.random.default_rng(0).standard_normal(A.shape[0])
        w, V = eigsh(A, k=min(k, A.shape[0] - 2), sigma=-1e-6, which="LM", v0=v0, tol=1e-12)
        free = ~self.fixed

        def to_nodes(v):
            u = np.zeros((len(self.X), 3))
            u[free] = v.reshape(-1, 3)
            return u
        return sway_and_twist(np.maximum(w, 0.0), V, self.X, np.full(len(self.X), self.m), to_nodes,
                              relaxed=True)

    def freqs(self, alpha, h):
        """(f_sway, zeta_sway, f_twist, zeta_twist) of the linearized step; NaN for a mode that does
        not oscillate."""
        md = self.modes(self.m * alpha / h**2)
        out = []
        for key in ("sway", "twist"):
            mu = md[key][0] ** 2                       # sway_and_twist returned sqrt(mu)
            r = gj._pbd_mode(mu, h)
            out += [np.nan, np.nan] if r is None else [r[0] / (2 * np.pi), r[1]]
        return out

    def h_max(self):
        mu0 = self.modes(0.0)["sway"][0] ** 2
        p0 = (1 - mu0) ** pl.N_STRETCH
        return float(np.hypot(np.arccos(np.sqrt(p0)), np.log(p0) / 2) / W)

    def alpha_star(self, h):
        """The compliance at which the linearized sway is at 8.01 Hz."""
        from scipy.optimize import brentq
        f = lambda lg: self.freqs(self.alpha_nom / 10**lg, h)[0] - js.F_VIDEO   # noqa: E731
        lg = brentq(f, -6.0, 12.0, xtol=1e-12)
        return self.alpha_nom / 10**lg


def pbd_theory(n, k):
    from scipy.optimize import brentq
    th = PbdTheory(n)
    hm = th.h_max()
    if k < 3:
        h = FRACTIONS[k] * hm
    else:   # the step at which the linearized sway, at 8.01 Hz, is damped by ZETA_STEP
        zeta = lambda h: th.freqs(th.alpha_star(h), h)[1] - ZETA_STEP   # noqa: E731
        if zeta(0.999 * hm) < 0:
            raise NoSuchStep(f"PBD on {n} cells damps the sway by at most {zeta(0.999 * hm) + ZETA_STEP:.4f} "
                             f"below h_max = {hm * 1e6:.0f} us")
        h = float(f"{brentq(zeta, 0.05 * hm, 0.999 * hm, xtol=1e-9):.4g}")
    a_star = th.alpha_star(h)
    fs, zs, ft, zt = th.freqs(a_star, h)
    md = th.modes(th.m * a_star / h**2)
    return {"h": float(h), "h_limit": hm, "E_ratio": th.alpha_nom / a_star, "dx_nominal": th.dx,
            "alpha_nominal": th.alpha_nom, "theory": {"f_sway": fs, "zeta_sway": zs, "f_twist": ft, "zeta_twist": zt},
            "damping": {}, "damping_name": None, "n_particles": len(th.X),
            "shapes": {key: md[key][1] for key in ("sway", "twist")},
            "masses": np.full(len(th.X), th.m), "pbd": th,
            "mode_twist_metrics": {"sway": md["sway_twist_metric"], "twist": md["twist_metric"]}}


# ---- Genesis models with one interface: set(E_ratio, damping), rollout(mode, timed) ---------

class Model:
    h = None

    def prepare(self, shapes, masses, device):
        self.v0, self.w = {}, {}
        for key, s in shapes.items():
            self.v0[key] = gj.V_PEAK * s / np.abs(s).max()
            self.w[key] = torch.as_tensor(c.modal_weights(s, masses), dtype=torch.float64, device=device)

    def rollout(self, mode, timed=False, duration=DURATION):
        """(t, q, wall s, simulated s, cut) of a run started in `mode`."""
        dev = self.X0_t.device
        sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
        if timed:
            self.start(mode)
            for _ in range(WARMUP):
                self.scene.step()
            sync()
        self.start(mode)
        steps = int(round(duration / self.h))
        every = max(1, int(round(READ_EVERY / self.h)))
        reads = np.arange(every, steps + 1, every)
        q = torch.empty(len(reads), dtype=torch.float64, device=dev)
        w = self.w[mode]
        sync()
        t0 = time.perf_counter()
        j, checked = 0, not (timed and CUTOFF)
        for s in range(1, steps + 1):
            try:
                self.scene.step()
            except Exception as err:           # Genesis stops on NaN: the run blew up
                print("rollout stopped:", str(err)[:200], flush=True)
                return reads * self.h, np.full(len(reads), np.nan), time.perf_counter() - t0, s * self.h, False
            if j < len(reads) and s == reads[j]:
                q[j] = (w * (self.pos() - self.X0_t)).sum()
                j += 1
            if not checked:
                wall = time.perf_counter() - t0
                if s * self.h >= CUT_SIM - 1e-12:
                    sync()
                    wall = time.perf_counter() - t0
                    checked = True
                    if s * self.h / wall < CUT_SPEED:
                        return None, None, wall, s * self.h, True
                elif wall > CUT_WALL:
                    return None, None, wall, s * self.h, True
        sync()
        wall = time.perf_counter() - t0
        return reads * self.h, q.cpu().numpy(), wall, steps * self.h, False


class FemModel(Model):
    def __init__(self, method, n, th, E):
        init()
        import genesis as gs
        import genesis.utils.element as eu
        from genesis.utils.misc import qd_to_numpy
        self.h = th["h"]
        self.implicit = method == "gs_fem_implicit"
        X, tets = mesh(n)
        n_bottom = int(np.sum(np.isclose(X[:, 2], X[:, 2].min())))
        keep = (eu.mesh_to_elements, eu.split_all_surface_tets)
        box = {}

        def grid(surface, tet_cfg=None):
            first = [int(np.argmin(np.linalg.norm(X - v, axis=1))) for v in np.asarray(surface.vertices)]
            order = first + [i for i in range(len(X)) if i not in set(first)]
            inv = np.empty(len(X), int)
            inv[order] = np.arange(len(X))
            box["order"] = np.array(order)
            return X[order], inv[tets]

        eu.mesh_to_elements = grid
        eu.split_all_surface_tets = lambda v, e: (v, e)
        try:
            self.scene = gs.Scene(
                sim_options=gs.options.SimOptions(dt=self.h, substeps=1, gravity=(0.0, 0.0, 0.0)),
                fem_options=gs.options.FEMOptions(
                    use_implicit_solver=self.implicit,
                    damping=0.0 if self.implicit else th["damping"]["c"],
                    damping_alpha=1e-6 if self.implicit else 0.0,          # nonzero, or compiled out
                    damping_beta=max(abs(th["damping"].get("beta", 0.0)), 1e-6) if self.implicit else 0.0,
                    max_vertex_constraints=n_bottom),
                show_viewer=False)
            self.ent = self.scene.add_entity(
                gs.morphs.Box(pos=(0.0, 0.0, 0.0), size=tuple(float(s) for s in SIDES)),
                gs.materials.FEM.Elastic(E=E, nu=NU, rho=RHO, model="linear"))
            self.scene.build()
        finally:
            eu.mesh_to_elements, eu.split_all_surface_tets = keep
        order = box["order"]
        self.info = self.scene.sim.fem_solver.fem_info
        self.mu0 = qd_to_numpy(self.info.elements.mu).copy()
        self.lam0 = qd_to_numpy(self.info.elements.lam).copy()
        masses = qd_to_numpy(self.info.vertices.mass).copy()
        X0 = self.ent.get_verts_pos().cpu().numpy().reshape(-1, 3).astype(np.float64)
        assert np.allclose(X0, X[order]) and np.allclose(masses, th["masses"][order])
        self.ent.set_vertex_constraints(np.flatnonzero(np.isclose(X0[:, 2], X0[:, 2].min())))
        dev = self.ent.get_verts_pos().device
        self.X0_t = torch.as_tensor(X0, dtype=torch.float64, device=dev)
        self.prepare({k: v[order] for k, v in th["shapes"].items()}, masses, dev)

    def set(self, E_ratio, damping):
        set_field(self.info.elements.mu, self.mu0 * E_ratio)
        set_field(self.info.elements.lam, self.lam0 * E_ratio)
        if self.implicit:
            set_field(self.info.damping_alpha_dt, damping["alpha"] * self.h)
            set_field(self.info.damping_beta_over_dt, damping["beta"] / self.h)
            set_field(self.info.damping_beta_factor, damping["beta"] / self.h + 1.0)
        else:
            set_field(self.info.damping_decay, np.exp(-self.h * damping["c"]))

    def start(self, mode):
        self.scene.reset()
        self.ent.set_velocity(torch.as_tensor(self.v0[mode], dtype=torch.float64, device=self.X0_t.device))

    def pos(self):
        return self.ent.get_verts_pos().reshape(-1, 3)


class MpmModel(Model):
    """genesis_jello.JelloMpm on n cells; E* by its rescaling of the step (exact, as GsMpmBlock)."""

    def __init__(self, n, th, E):
        init()
        import genesis as gs
        from genesis.utils.misc import qd_to_numpy
        self.h, dx, p = th["h"], th["dx"], th["particle_size"]
        margin = 8 * dx
        mat = gs.materials.MPM.Elastic(E=E, nu=NU, rho=RHO, sampler="regular")
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.h, substeps=1, gravity=(0.0, 0.0, 0.0)),
            mpm_options=gs.options.MPMOptions(
                grid_density=1.0 / dx, particle_size=p, lower_bound=(-margin, -margin, -margin),
                upper_bound=tuple(float(s + margin) for s in SIDES)),
            show_viewer=False)
        self.ent = self.scene.add_entity(
            gs.morphs.Box(pos=tuple(float(s / 2) for s in SIDES), size=tuple(float(s) for s in SIDES)), mat)
        self.slab = self.scene.add_entity(
            gs.morphs.Box(pos=(float(SIDES[0] / 2), float(SIDES[1] / 2), -0.75 * dx),
                          size=(float(SIDES[0] + 4 * dx), float(SIDES[1] + 4 * dx), dx / 2)), mat)
        self.scene.build()
        self.info = self.scene.sim.mpm_solver.mpm_info
        self.stress_scale0 = float(qd_to_numpy(self.info.stress_scale))
        pos = self.ent.get_particles_pos().reshape(-1, 3)
        self.X0_t = pos.to(torch.float64)
        assert np.allclose(self.X0_t.cpu().numpy(), th["X"]), "Genesis sampled the cube elsewhere"
        zs = self.slab.get_particles_pos().reshape(-1, 3)[:, 2].cpu().numpy()
        assert np.all((zs >= -1.5 * dx) & (zs < -0.5 * dx))
        self.prepare(th["shapes"], th["masses"], pos.device)
        self.s = 1.0

    def hold(self):
        n = self.slab.n_particles
        self.slab.set_free(torch.zeros(n, dtype=torch.bool, device=self.X0_t.device))

    def set(self, E_ratio, damping):
        self.s = float(np.sqrt(E_ratio))
        set_field(self.info.substep_dt, self.h * self.s)
        set_field(self.info.stress_scale, self.stress_scale0 * self.s)

    def start(self, mode):
        self.scene.reset()
        self.hold()
        self.ent.set_particles_vel(torch.as_tensor(self.v0[mode] / self.s, dtype=torch.float64,
                                                   device=self.X0_t.device))

    def pos(self):
        return self.ent.get_particles_pos().reshape(-1, 3)


class PbdModel(Model):
    def __init__(self, n, th):
        init()
        import genesis as gs
        from genesis.engine.mesh import Mesh
        from genesis.utils.misc import qd_to_numpy
        self.h = th["h"]
        pt = th["pbd"]
        X, tets = pt.X, pt.tets
        keep = Mesh.tetrahedralize
        Mesh.tetrahedralize = lambda self_, cfg: (X.copy(), tets.copy())
        try:
            self.scene = gs.Scene(
                sim_options=gs.options.SimOptions(dt=self.h, substeps=1, gravity=(0.0, 0.0, 0.0)),
                pbd_options=gs.options.PBDOptions(
                    particle_size=float(SIDES.min() / n / 2), lower_bound=(-1.0, -1.0, -1.0),
                    upper_bound=(1.0, 1.0, 1.0), max_stretch_solver_iterations=pl.N_STRETCH),
                show_viewer=False)
            self.ent = self.scene.add_entity(
                gs.morphs.Box(pos=(0.0, 0.0, 0.0), size=tuple(float(s) for s in SIDES)),
                gs.materials.PBD.Elastic(rho=RHO, stretch_compliance=pt.alpha_nom))
            self.scene.build()
        finally:
            Mesh.tetrahedralize = keep
        self.info = self.scene.sim.pbd_solver.pbd_info
        self.alpha0 = qd_to_numpy(self.info.edges.stretch_compliance).copy()
        pos = self.ent.get_particles_pos().reshape(-1, 3)
        self.X0_t = pos.to(torch.float64)
        assert np.allclose(self.X0_t.cpu().numpy(), X)
        assert np.allclose(qd_to_numpy(self.info.particles.mass), pt.m)
        self.bottom = np.flatnonzero(pt.fixed)
        self.prepare(th["shapes"], th["masses"], pos.device)

    def set(self, E_ratio, damping):
        set_field(self.info.edges.stretch_compliance, self.alpha0 / E_ratio)

    def start(self, mode):
        self.scene.reset()
        self.ent.set_particles_vel(torch.as_tensor(self.v0[mode], dtype=torch.float64, device=self.X0_t.device))
        self.ent.fix_particles(self.bottom)

    def pos(self):
        return self.ent.get_particles_pos().reshape(-1, 3)


# ---- one point ------------------------------------------------------------------------------

def measure(t, q, f_guess):
    if t is None or not np.all(np.isfinite(q)):
        return np.nan, np.nan
    try:
        w, z = js.measure_at(t, q, 2 * np.pi * f_guess)
    except RuntimeError:
        return np.nan, np.nan
    return w / (2 * np.pi), z


def hardware():
    import genesis as gs
    out = {"node": socket.gethostname(), "python": platform.python_version(),
           "genesis": getattr(gs, "__version__", "1.5.0"), "slurm_job": os.environ.get("SLURM_JOB_ID"),
           "backend": os.environ.get("R4R_GENESIS_BACKEND", "gpu"), "precision": "float64"}
    if torch.cuda.is_available() and out["backend"] != "cpu":
        out["gpu"] = torch.cuda.get_device_name()
    return out


def errors(meas):
    e = {k: 100 * (meas[k] / TARGET[k] - 1) for k in TARGET}
    return {"err_sway_f": e["f_sway"], "err_sway_zeta": e["zeta_sway"], "err_twist_f": e["f_twist"],
            "err_twist_zeta": e["zeta_twist"],
            "err_freq_pct": float(np.sqrt((e["f_sway"] ** 2 + e["f_twist"] ** 2) / 2)),
            "err_damp_pct": float(np.sqrt((e["zeta_sway"] ** 2 + e["zeta_twist"] ** 2) / 2))}


def run_point(method, n, k, force=False, timing_only=False):
    t_start = time.time()
    cs, ct, E = continuum()
    if method.startswith("gs_fem"):
        th = fem_theory(method, n, k)
    try:
        if method == "gs_mpm":
            th = mpm_theory(n, k)
        elif method == "gs_pbd":
            th = pbd_theory(n, k)
    except NoSuchStep as err:
        return f"no point: {err}"
    h = th["h"]
    out_dir = OUT_TIMING if timing_only else OUT if CUTOFF else OUT_FIT
    out_file = out_dir / f"{method}_{n}cells_{h * 1e6:.1f}us.json"
    if out_file.exists() and not force:
        return out_file.name
    timing = OUT_TIMING / out_file.name
    if not CUTOFF and timing.exists() and json.loads(timing.read_text())["cut"]:
        return f"{out_file.name} skipped: the cluster's timing cut it"
    f_twist_ref = ct * np.sqrt(E / RHO)
    row = {"method": method, "cells": n, "dx_m": th.get("dx", GM / n), "h_s": h, "step_index": k,
           "step_rule": step_rule(method, k),
           "E": E, "E_ratio": th["E_ratio"], "damping_name": th["damping_name"],
           "damping_value": dict(th["damping"]), "f_twist_ref": f_twist_ref, "targets": TARGET,
           "theory_seconds": time.time() - t_start}
    for key in ("S", "T", "h_limit", "f_sway_model_Erho1", "f_twist_model_Erho1", "w_max_Erho1", "n_vertices",
                "n_elements", "n_particles", "n_grid_nodes_free", "particle_size", "dx_nominal", "alpha_nominal",
                "theory", "mode_twist_metrics"):
        if key in th:
            row[key] = th[key]
    t0 = time.time()
    if method.startswith("gs_fem"):
        sim = FemModel(method, n, th, E)
    elif method == "gs_mpm":
        sim = MpmModel(n, th, E)
    else:
        sim = PbdModel(n, th)
    row["build_seconds"] = time.time() - t0
    row["hardware"] = hardware()
    if "theory" in th:
        guess = {"sway": th["theory"]["f_sway"], "twist": th["theory"]["f_twist"]}
    elif "f_guess" in th:
        guess = dict(th["f_guess"])
    else:
        scale = np.sqrt(E * th["E_ratio"] / RHO)
        guess = {"sway": th["f_sway_model_Erho1"] * scale, "twist": th["f_twist_model_Erho1"] * scale}
    n_runs = [0]
    if timing_only:
        # the stepping costs the same at any E (MPM rescales its step), so MPM is timed at E
        sim.set(1.0 if method == "gs_mpm" else th["E_ratio"], dict(th["damping"]))
        t, q, wall, sim_s, cut = sim.rollout("sway", timed=True)
        f, z = (np.nan, np.nan) if cut else measure(t, q, guess["sway"])
        row.update(wall_s=wall, sim_s=sim_s, speed=sim_s / wall, cut=cut, f_sway_timed_run=f,
                   zeta_sway_timed_run=z, seconds_total=time.time() - t_start)
        save(out_file, row)
        return out_file.name

    def run_both(E_ratio, damping, timed=False):
        sim.set(E_ratio, damping)
        meas, timing = {}, None
        for mode in ("sway", "twist"):
            t, q, wall, sim_s, cut = sim.rollout(mode, timed=timed and mode == "sway")
            n_runs[0] += 1
            if timed and mode == "sway":
                timing = (wall, sim_s, cut)
                if cut:
                    return None, timing
            f, z = measure(t, q, guess[mode])
            meas[f"f_{mode}"], meas[f"zeta_{mode}"] = f, z
            if np.isfinite(f):
                guess[mode] = f
        return meas, timing

    damping = dict(th["damping"])
    if method == "gs_mpm":
        # the step's own sway at E, which sets E* (mpm_theory); timed too, so a slow point stops here
        sim.set(1.0, damping)
        t, q, wall, sim_s, cut = sim.rollout("sway", timed=True)
        n_runs[0] += 1
        if cut:
            row.update(wall_s=wall, sim_s=sim_s, speed=sim_s / wall, cut=True, seconds_total=time.time() - t_start)
            save(out_file, row)
            return out_file.name
        f0, z0 = measure(t, q, guess["sway"])
        row["at_E"] = {"f_sway": f0, "zeta_sway": z0, "speed": sim_s / wall}
        th["E_ratio"] = row["E_ratio"] = (js.F_VIDEO / f0) ** 2
        row["S"] = th["E_ratio"] / th["T"]
        guess["sway"] = js.F_VIDEO
        guess["twist"] *= np.sqrt(th["E_ratio"])
    meas, (wall, sim_s, cut) = run_both(th["E_ratio"], damping, timed=True)
    row.update(wall_s=wall, sim_s=sim_s, speed=sim_s / wall, cut=cut)
    if cut:
        row["seconds_total"] = time.time() - t_start
        save(out_file, row)
        return out_file.name
    row["renormalized"] = meas
    print(f"{method} n {n} h {h * 1e6:.1f} us: speed {row['speed']:.3g}, renormalized {meas}", flush=True)

    # frequency: lam on E*, fitted to the two frequencies
    if method == "gs_pbd":
        pt = th["pbd"]

        def predict(lam):
            fs, _, ft, _ = pt.freqs(pt.alpha_nom / (th["E_ratio"] * lam), h)
            return np.array([fs, ft])
    else:
        def predict(lam):
            return np.sqrt(lam) * np.array([1.0, 1.0])
    targets_f = np.array([TARGET["f_sway"], TARGET["f_twist"]])
    lam, history = 1.0, []
    for _ in range(5):
        measured = np.array([meas["f_sway"], meas["f_twist"]])
        if not np.all(np.isfinite(measured)):
            break
        corr = measured / predict(lam)

        def obj(x):
            r = corr * predict(np.exp(x)) / targets_f - 1
            return float(np.sum(np.where(np.isfinite(r), r, 10.0) ** 2))
        x = minimize_scalar(obj, bounds=(np.log(lam) - 4, np.log(lam) + 4), method="bounded",
                            options={"xatol": 1e-9}).x
        history.append({"lam": lam, **meas})
        if abs(x - np.log(lam)) < 2e-6:
            break
        lam = float(np.exp(x))
        meas, _ = run_both(th["E_ratio"] * lam, damping)
    row["lam"] = lam
    row["lam_history"] = history

    # damping: at that lam, every damping parameter the engine offers, fitted to the two ratios
    if method.startswith("gs_fem") and np.all(np.isfinite(list(meas.values()))):
        targets_z = np.array([TARGET["zeta_sway"], TARGET["zeta_twist"]])
        dhist = []
        for _ in range(5):
            w = 2 * np.pi * np.array([meas["f_sway"], meas["f_twist"]])
            z = np.array([meas["zeta_sway"], meas["zeta_twist"]])
            dhist.append({**damping, **meas})
            if method == "gs_fem_explicit":
                d = z - damping["c"] / (2 * w)              # zeta = d + c / (2 w)
                a = 1 / (2 * w * targets_z)
                b = d / targets_z - 1
                new = {"c": float(-np.sum(a * b) / np.sum(a * a))}
                change = abs(new["c"] - damping["c"]) / (2 * ZETA * W)
            else:
                d = z - damping["alpha"] / (2 * w) - damping["beta"] * w / 2
                A = np.stack([1 / (2 * w), w / 2], 1)
                al, be = np.linalg.solve(A, targets_z - d)
                # backward Euler with beta K keeps a mode w only while 4 + h w^2 (2 beta + h) >= 0: a
                # beta below that for the mesh's highest mode blows up, so beta stops there (with a
                # margin) and alpha alone is fitted by least squares
                w_top = th["w_max_Erho1"] * np.sqrt(E * th["E_ratio"] * lam / RHO)
                be_min = -h / 2 - 0.9 * 2 / (h * w_top**2)
                if be < be_min:
                    be = be_min
                    a = 1 / (2 * w * targets_z)
                    b = (d + be * w / 2) / targets_z - 1
                    al = -np.sum(a * b) / np.sum(a * a)
                    row["beta_at_stability_bound"] = True
                new = {"alpha": float(al), "beta": float(be)}
                change = max(abs(al - damping["alpha"]) / (2 * ZETA * W), abs(be - damping["beta"]) / (2 * ZETA / W))
            if change < 1e-6:
                break
            damping = new
            meas, _ = run_both(th["E_ratio"] * lam, damping)
        row["damping_history"] = dhist
    row["damping_fit"] = damping
    row.update(meas)
    row.update(errors(meas) if np.all(np.isfinite(list(meas.values()))) else
               {"err_freq_pct": np.nan, "err_damp_pct": np.nan})
    row["err_pct"] = row["err_freq_pct"]
    row["n_runs"] = n_runs[0]
    row["seconds_total"] = time.time() - t_start
    print(f"{method} n {n} h {h * 1e6:.1f} us: lam {lam:.6f}, damping {damping}, final {meas}, "
          f"err_freq {row['err_freq_pct']:.4g} %, err_damp {row['err_damp_pct']:.4g} %", flush=True)
    save(out_file, row)
    return out_file.name


def save(path, row):
    os.listdir(path.parent.parent)     # an NFS client may cache that the directory is missing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=1, default=float))


def merge():
    """results/speed_accuracy/<point>.json from a fit made elsewhere (OUT_FIT) and the cluster's
    timing of that point (OUT_TIMING): speed, cut, wall_s, sim_s and hardware are the timing's."""
    done = 0
    for tp in sorted(OUT_TIMING.glob("gs_*.json")):
        if not tp.name.startswith(tuple(f"{m}_" for m in METHODS)):
            continue                                 # another agent's method in the same directory
        out = OUT / tp.name
        if out.exists() or (c.RESULTS / "speed_accuracy_extra" / tp.name).exists():
            continue                                 # done, or a step the spec no longer asks for
        tim = json.loads(tp.read_text())
        keys = ("speed", "cut", "wall_s", "sim_s", "hardware", "build_seconds")
        if tim["cut"]:
            row = tim
        elif (OUT_FIT / tp.name).exists():
            row = json.loads((OUT_FIT / tp.name).read_text())
            row["fit_hardware"] = row.get("hardware")
            row["fit_speed_untimed"] = row.get("speed")
            row.update({k: tim[k] for k in keys})
            row["timing_from"] = "speed_accuracy_timing"
        else:
            continue
        if not row["cut"]:
            row["finite"] = bool(np.isfinite(row.get("err_freq_pct", np.nan)) and np.isfinite(row.get("err_damp_pct", np.nan)))
            if not row["finite"]:   # a run blew up: the step is not stable, so the point is not one the spec asks for
                row["note"] = "unstable: a run of the fit blew up"
                out = c.RESULTS / "speed_accuracy_extra" / tp.name
        save(out, row)
        done += 1
    print("merged", done)


def main():
    global CUTOFF
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--task", type=int)
    ap.add_argument("--point", nargs=3, help="method n k")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--timing-only", action="store_true")
    ap.add_argument("--no-cut", action="store_true", help="fit without the cutoff, into OUT_FIT")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.count:
        print(len(tasks()))
        return
    if args.merge:
        merge()
        return
    CUTOFF = not args.no_cut
    point = args.point and (args.point[0], int(args.point[1]), int(args.point[2]))
    point = point or tasks()[args.task]
    print("done", run_point(*point, force=args.force, timing_only=args.timing_only), flush=True)


if __name__ == "__main__":
    main()
