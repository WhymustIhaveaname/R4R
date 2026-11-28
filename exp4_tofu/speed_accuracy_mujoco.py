"""Speed against accuracy on the Jello cube, for MuJoCo's four flex methods.

The cube of jello_sim.py (4.9 x 4.7 x 4.5 cm, base fixed, rho 1080, nu 0.45, E from the video's
sway and the continuum's constant), with two knobs: n cells per side and the step h.

  mj_discrete    dof="full", n^3 cells of six P1 tetrahedra, integrator="discrete"
  mj_euler       the same mesh, integrator="Euler" (symplectic, damping explicit)
  mj_trilinear   dof="trilinear", cellcount n n n (8-node cells), integrator="discrete"
  mj_quadratic   dof="quadratic", cellcount n n n (27-node cells), integrator="discrete"

The interpolated flexes get a vertex grid on their nodes (order n + 1 per side), so their mesh
refines with n and nothing else is stepped.

Each (method, n, h) is run with the parameters the theory renormalizes, nothing fitted:
  S       from the model's own modal analysis at E / rho = 1: MuJoCo's masses, and the tangent of
          its elastic force at rest by central differences, one colour of nodes at a time (nodes
          of a colour are too far apart to share a neighbour, so each difference gives whole
          columns of the sparse stiffness).  The sway is the lowest mode that moves mostly
          sideways, the twist the lowest whose motion is a rotation about the vertical
          (jello_model.modes's test, on the nodes with their masses); S = (C / C_sway)^2.
  T, beta*  backward Euler's (discrete) or symplectic Euler's (Euler) counterterms at the sway's
          w = 2 pi 8.01 Hz and zeta = 0.015.
Two runs of 1.5 s, started in the model's own sway and in its own twist; (f, zeta) of each by
jello_sim.measure_at.  Error (%) = 100 max(|f_sway / 8.01 - 1|, |zeta_sway / 0.015 - 1|,
|f_twist / f_twist_ref - 1|), f_twist_ref the continuum's twist at E; each term is also kept,
in percent.

Speed = simulated seconds / wall seconds of mj_step over the sway run, after 10 warm-up steps;
model building and the readout between steps are not counted.  If the first 0.05 s simulated
(or the first 5 s of wall time) run slower than 1/100 of real time, the point stops there and is
saved with "cut": true and that speed.

  python speed_accuracy_mujoco.py --count
  python speed_accuracy_mujoco.py --task K      # one point; a Slurm array index
"""

import argparse
import json
import os
import platform
import socket
import time
from functools import lru_cache

import mujoco
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh

import common as c
import jello_mujoco_variants as jv
import jello_model as jm
import jello_sim as js

OUT = c.RESULTS / "speed_accuracy"
METHODS = {"mj_discrete": ("full", "discrete", (3, 4, 5, 6, 8, 10, 12, 16, 20)),
           "mj_euler": ("full", "Euler", (3, 4, 5, 6, 8, 10, 12, 16, 20)),
           "mj_trilinear": ("trilinear", "discrete", (1, 2, 3, 4, 6, 8)),
           "mj_quadratic": ("quadratic", "discrete", (1, 2, 3, 4, 6, 8))}
H_DISCRETE = (0.25e-3, 0.5e-3, 1e-3)
EULER_FRACTIONS = (0.25, 0.5, 0.9)
ORDER = {"full": 1, "trilinear": 1, "quadratic": 2}
DURATION = 1.5
CUT_SIM, CUT_WALL, CUT_SPEED = 0.05, 5.0, 0.01
WARMUP = 10


def xml(dof, n, integrator, E, beta, h, rho=js.RHO):
    k = ORDER[dof] * n
    count = " ".join(str(k + 1) for _ in range(3))
    spacing = " ".join(repr(float(s / k)) for s in jm.SIDES)
    cells = "" if dof == "full" else f' cellcount="{n} {n} {n}"'
    return f"""<mujoco>
  <option timestep="{float(h)!r}" gravity="0 0 0" integrator="{integrator}"/>
  <worldbody>
    <flexcomp name="cube" type="grid" dim="3" dof="{dof}"{cells} count="{count}" spacing="{spacing}"
              mass="{float(rho * np.prod(jm.SIDES))!r}" radius="0.0005">
      <contact contype="0" conaffinity="0" selfcollide="none"/>
      <elasticity young="{float(E)!r}" poisson="{js.NU!r}" damping="{float(beta)!r}"/>
      <pin gridrange="0 0 0 {k} {k} 0"/>
    </flexcomp>
  </worldbody>
</mujoco>"""


def node_grid(m):
    """Per dof body: its grid index (3 ints) and position; the dofs are 3 slides per body."""
    bodies = m.dof_bodyid[::3]
    assert np.all(m.dof_bodyid[1::3] == bodies) and np.all(m.dof_bodyid[2::3] == bodies)
    axes = np.stack([m.jnt_axis[m.dof_jntid[np.arange(k, m.nv, 3)]] for k in range(3)], 1)
    assert np.allclose(axes, np.eye(3)[None]), "dofs are not slides along x, y, z"
    P = m.body_pos[bodies].copy()
    step = np.array([np.diff(u).min() if len(u := np.unique(np.round(P[:, a], 12))) > 1 else 1.0
                     for a in range(3)])                  # one free layer: any step
    idx = np.round((P - P.min(0)) / step).astype(int)
    return idx, P


def stiffness(m, reach, eps=1e-7):
    """K = -d qfrc_passive / d qpos at rest, sparse, by central differences on colours of nodes
    2 reach + 1 apart in some axis, so that no row sees two perturbed nodes."""
    idx, _ = node_grid(m)
    p = 2 * reach + 1
    shape = idx.max(0) + 1
    lookup = -np.ones(shape, int)
    lookup[tuple(idx.T)] = np.arange(len(idx))
    d = mujoco.MjData(m)
    rows, cols, vals = [], [], []
    for color in np.ndindex(p, p, p):
        members = np.flatnonzero(np.all(idx % p == color, 1))
        if not len(members):
            continue
        # for each node, the member of this colour within reach of it, if any
        near = idx + (np.array(color) - idx + reach) % p - reach
        ok = np.all((near >= 0) & (near < shape), 1)
        partner = np.full(len(idx), -1)
        partner[ok] = lookup[tuple(near[ok].T)]
        have = np.flatnonzero(partner >= 0)
        for a in range(3):
            df = np.zeros(m.nv)
            for s in (1, -1):
                d.qpos[:] = m.qpos0
                d.qpos[3 * members + a] += s * eps
                d.qvel[:] = 0
                mujoco.mj_forward(m, d)
                df += s * d.qfrc_passive
            for b in range(3):
                rows.append(3 * have + b)
                cols.append(3 * partner[have] + a)
                vals.append(-df[3 * have + b] / (2 * eps))
    K = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(m.nv, m.nv)).tocsc()
    K.eliminate_zeros()
    return (K + K.T) / 2


def classify(P, mass, u):
    """jello_model.modes's test on node displacements u (N, 3) with node masses."""
    r = P[:, :2] - (mass[:, None] * P[:, :2]).sum(0) / mass.sum()
    twist = np.sum(mass * (r[:, 0] * u[:, 1] - r[:, 1] * u[:, 0])) / np.sqrt(
        np.sum(mass * (r**2).sum(1)) * np.sum(mass * (u[:, :2]**2).sum(1)))
    ux, uy, uz = np.sum(mass[:, None] * u**2, 0)
    if abs(twist) > 0.8:
        return "twist"
    return "sway x" if ux > max(uy, uz) else "sway y" if uy > uz else "vertical"


def modal(dof, n):
    """The model's sway and twist at E / rho = 1: (f_sway, f_twist, w_max, sway, twist, masses)."""
    m = mujoco.MjModel.from_xml_string(xml(dof, n, "Euler", 1.0, 1e-3, 1e-3, rho=1.0))
    K = stiffness(m, ORDER[dof] if dof == "quadratic" else 1)
    mass = m.body_mass[m.dof_bodyid]
    _, P = node_grid(m)
    if m.nv <= 600:
        w2, V = eigh(K.toarray(), np.diag(mass))
        wmax2 = w2.max()
    else:
        w2, V = eigsh(K, k=16, M=diags(mass), sigma=0.0, which="LM")
        s = 1 / np.sqrt(mass)
        wmax2 = eigsh(diags(s) @ K @ diags(s), k=1, which="LA", return_eigenvectors=False)[0]
    order = np.argsort(w2)
    f_sway = f_twist = None
    for j in order:
        kind = classify(P, mass[::3], V[:, j].reshape(-1, 3))
        f = np.sqrt(max(w2[j], 0)) / (2 * np.pi)
        if f_sway is None and kind.startswith("sway"):
            f_sway, sway = f, V[:, j]
        if f_twist is None and kind == "twist":
            f_twist, twist = f, V[:, j]
    assert f_sway is not None and f_twist is not None, "no sway or no twist among the lowest modes"
    return f_sway, f_twist, float(np.sqrt(wmax2)), sway, twist, mass


def tasks():
    out = []
    for method, (dof, integ, ns) in METHODS.items():
        for n in ns:
            for k in range(3):
                out.append((method, n, k))
    return out


def euler_step(w_max1, E, S, frac):
    """(frac of Euler's stability limit, the limit): w_max^2 h (h + 2 beta*) = 4 at E* = E S T(h),
    w_max1 being the model's highest w at E / rho = 1."""
    w = 2 * np.pi * js.F_VIDEO
    h = 1e-4
    for _ in range(20):
        T, bs = jv.symplectic_euler(w, js.ZETA_VIDEO, h)
        h = jv.h_max_euler(w_max1 * np.sqrt(E * S * T / js.RHO), bs)
    return frac * h, h


@lru_cache(maxsize=None)
def continuum():
    """(C_sway, C_twist) at E / rho = 1 (jello_mujoco_variants.continuum_modes), cached on disk:
    the finer mesh takes a minute, and every Slurm task needs it."""
    path = c.CACHE / "jello_continuum_modes.json"
    if path.exists():
        return tuple(json.loads(path.read_text()))
    out = jv.continuum_modes()
    c.CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(out)))
    return out


def hardware():
    cpu = "?"
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"node": socket.gethostname(), "cpu": cpu, "mujoco": mujoco.__version__,
            "python": platform.python_version(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "slurm_cpus": os.environ.get("SLURM_CPUS_PER_TASK")}


def run(m, vec, h, duration, timed):
    """Step from rest with velocity along vec; q(t) its projection in MuJoCo's masses.  With
    timed, mj_step alone is timed and the run stops early if it is slower than the cutoff.
    Returns (t, q, wall seconds of mj_step, simulated seconds, cut)."""
    d = mujoco.MjData(m)
    mass = m.body_mass[m.dof_bodyid]
    w = mass * vec / (vec @ (mass * vec))
    v0 = 1e-3 * vec / np.abs(vec).max()
    if timed:
        t0 = time.perf_counter()
        d.qvel[:] = v0
        for _ in range(WARMUP):
            mujoco.mj_step(m, d)
            if time.perf_counter() - t0 > CUT_WALL:
                break
        mujoco.mj_resetData(m, d)
    d.qvel[:] = v0
    steps = int(round(duration / h))
    q = np.empty(steps)
    wall = 0.0
    for j in range(steps):
        t0 = time.perf_counter()
        mujoco.mj_step(m, d)
        wall += time.perf_counter() - t0
        q[j] = w @ (d.qpos - m.qpos0)
        if timed and ((j + 1) * h < CUT_SIM + 1e-12) and (
                wall > CUT_WALL or (abs((j + 1) * h - CUT_SIM) < 0.5 * h and (j + 1) * h / wall < CUT_SPEED)):
            return np.arange(1, j + 2) * h, q[:j + 1], wall, (j + 1) * h, True
    return np.arange(1, steps + 1) * h, q, wall, steps * h, False


def measure(t, q, w0):
    """jello_sim.measure_at, or NaN where the run blew up or the fit fails."""
    if not np.all(np.isfinite(q)):
        return np.nan, np.nan
    try:
        return js.measure_at(t, q, w0)
    except RuntimeError:
        return np.nan, np.nan


def run_point(method, n, k):
    dof, integ, _ = METHODS[method]
    t_start = time.time()
    C, C_tw = continuum()
    E = js.RHO * (js.F_VIDEO / C) ** 2
    f_twist_ref = C_tw * np.sqrt(E / js.RHO)
    w = 2 * np.pi * js.F_VIDEO
    f_sway1, f_twist1, w_max1, sway, twist, _ = modal(dof, n)
    S = (C / f_sway1) ** 2
    if integ == "discrete":
        h = H_DISCRETE[k]
        T, beta = js.backward_euler(w, js.ZETA_VIDEO, h)
        h_limit = None
    else:
        h, h_limit = euler_step(w_max1, E, S, EULER_FRACTIONS[k])
        T, beta = jv.symplectic_euler(w, js.ZETA_VIDEO, h)
    E_star = E * S * T
    m = mujoco.MjModel.from_xml_string(xml(dof, n, integ, E_star, beta if beta != 0 else 1e-9, h))
    t, q, wall, sim, cut = run(m, sway, h, DURATION, timed=True)
    row = {"method": method, "cells": n, "dx_m": float(np.mean(jm.SIDES) / n), "h_s": h,
           "speed": sim / wall, "cut": cut, "E_ratio": S * T, "damping_name": "beta (s)",
           "damping_value": beta, "S": S, "T": T, "f_twist_ref": f_twist_ref,
           "hardware": hardware(), "wall_s": wall, "sim_s": sim, "nv": int(m.nv),
           "f_sway_model_Erho1": f_sway1, "f_twist_model_Erho1": f_twist1, "w_max_Erho1": w_max1,
           "euler_h_limit_s": h_limit}
    if not cut:
        w_tw = 2 * np.pi * f_twist1 * np.sqrt(E_star / js.RHO)
        t2, q2, _, _, _ = run(m, twist, h, DURATION, timed=False)
        (f_s, z_s), (f_t, z_t) = (measure(t, q, w), measure(t2, q2, w_tw))
        f_s, f_t = f_s / (2 * np.pi), f_t / (2 * np.pi)
        e = [100 * abs(f_s / js.F_VIDEO - 1), 100 * abs(z_s / js.ZETA_VIDEO - 1), 100 * abs(f_t / f_twist_ref - 1)]
        row.update({"f_sway": f_s, "zeta_sway": z_s, "f_twist": f_t, "zeta_twist": z_t,
                    "err_sway_f": e[0], "err_sway_zeta": e[1], "err_twist_f": e[2], "err_pct": max(e),
                    "finite": bool(np.all(np.isfinite(q)) and np.all(np.isfinite(q2)))})
    else:
        row.update({k_: None for k_ in ("f_sway", "zeta_sway", "f_twist", "zeta_twist", "err_sway_f",
                                        "err_sway_zeta", "err_twist_f", "err_pct")})
    row["seconds_total"] = time.time() - t_start
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{method}_{n}cells_{h * 1e6:.1f}us.json"
    path.write_text(json.dumps(row, indent=1))
    return path, row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--task", type=int)
    ap.add_argument("--method")
    ap.add_argument("--n", type=int)
    ap.add_argument("--k", type=int)
    args = ap.parse_args()
    if args.count:
        print(len(tasks()))
        return
    todo = [tasks()[args.task]] if args.task is not None else [(args.method, args.n, args.k)]
    for method, n, k in todo:
        path, row = run_point(method, n, k)
        print(f"{path.name}: speed {row['speed']:.4g}  cut {row['cut']}  err {row['err_pct']}  "
              f"E*/E {row['E_ratio']:.4f}  beta* {row['damping_value']:.3e}  "
              f"[{row['seconds_total']:.1f} s]", flush=True)


if __name__ == "__main__":
    main()
