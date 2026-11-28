"""Fit E* and zeta* on the (h, dx) grid for one engine's schemes; each point is a parts file.

  uv run python run_flow.py --engine mujoco --jobs 30
  uv run python run_flow.py --merge
"""

import argparse
import json
import time
from multiprocessing import Pool

import numpy as np

import common as c

PARTS = c.RESULTS / "flow_parts"
DXS = (0.01, 0.005, 0.01 / 3, 0.0025, 0.002)
H_IMPLICIT = (0.25e-3, 0.5e-3, 0.75e-3, 1e-3) + tuple(round(0.5e-3 * k, 6) for k in range(3, 11))  # to 5 ms
EXPLICIT_FRACTIONS = (0.25, 0.4, 0.55, 0.7, 0.85)                  # of the stability limit

# arm: (engine, time scheme, vertex masses, damping model, cell sizes)
ARMS = {
    "mj_euler": ("mujoco", "symplectic_explicit_beta", "equal", "stiffness", DXS[:3]),
    "mj_discrete": ("mujoco", "backward_euler_beta", "equal", "stiffness", DXS),
    "gs_fem_explicit": ("genesis", "symplectic_exp_mass", "volume", "mass", DXS),
    "gs_fem_implicit": ("genesis", "backward_euler_beta", "volume", "stiffness", DXS),
    "gs_mpm": ("genesis", "symplectic_none", None, "none", DXS),
}
MPM_FRACTIONS = (0.25, 0.5, 1.0, 2.0)   # of Genesis's suggested substep, 0.02 dx (s, dx in m)


def theory_table():
    """Per cell size and mass lumping: E*/E at h -> 0 and the highest discrete frequency."""
    path = c.CACHE / "p1_theory.json"
    if path.exists():
        return json.loads(path.read_text())
    import p1
    from scipy.sparse import diags
    from scipy.sparse.linalg import eigsh
    table = {}
    for dx in DXS:
        X, tets = p1.mesh_from_mujoco(dx)
        K, vol = p1.assemble(X, tets, c.E0, c.NU)
        for kind in ("equal", "volume"):
            w, overlap = p1.discrete_mode(X, tets, kind)
            m = np.repeat(p1.masses(X, tets, vol, kind), 3)
            w2max = eigsh(K, k=1, M=diags(m), which="LA", return_eigenvectors=False)[0]
            table[f"{dx:.6f}_{kind}"] = {"S": (c.omega0() / w) ** 2, "overlap": overlap,
                                         "w_max": float(np.sqrt(w2max))}
    c.CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table, indent=1))
    return table


def h_max(arm, dx, table):
    """The largest stable step of an explicit scheme, at E* = S E0 (so w_max grows by sqrt S)."""
    _, scheme, kind, _, _ = ARMS[arm]
    t = table[f"{dx:.6f}_{kind}"]
    w = t["w_max"] * np.sqrt(t["S"])
    if scheme == "symplectic_explicit_beta":   # w^2 h (h + 2 beta) <= 4
        b = c.damping_param("stiffness", c.ZETA0)
        return (-2 * b + np.sqrt(4 * b * b + 16 / w**2)) / 2
    return 2 / w                                # symplectic Euler, damping by a factor


def steps_for(arm, dx, table):
    scheme = ARMS[arm][1]
    if scheme.startswith("backward"):
        return list(H_IMPLICIT)
    if scheme == "symplectic_none":
        return [float(f"{f * 0.02 * dx:.4g}") for f in MPM_FRACTIONS]
    hm = h_max(arm, dx, table)
    return [float(f"{f * hm:.4g}") for f in EXPLICIT_FRACTIONS]


def make_block(arm, dx, h):
    if ARMS[arm][0] == "mujoco":
        from mujoco_sim import MjBlock
        return MjBlock(arm, dx, h)
    if arm == "gs_mpm":
        from genesis_sim import GsMpmBlock
        return GsMpmBlock(arm, dx, h)
    from genesis_sim import GsBlock
    return GsBlock(arm, dx, h)


def run_point(task):
    arm, dx, h = task
    out = PARTS / f"{arm}_{dx * 1e3:.3f}mm_{h * 1e6:.1f}us.json"
    if out.exists():
        return out.name
    t0 = time.time()
    table = theory_table()
    _, scheme, kind, damping, _ = ARMS[arm]
    S = table[f"{dx:.6f}_{kind}"]["S"] if kind else 1.0
    e, z = c.time_prediction(scheme, h)
    block = make_block(arm, dx, h)
    free = damping != "none"
    start = (np.log(S * e), z)
    # MPM's linearized step (mpm_linear.py) gives its E* to 1e-6: from there the fit needs about
    # ten rollouts, not the forty of a start at 1, which at 2 mm cells are a day on the GPU
    theory = c.RESULTS / "mpm_theory_parts" / f"{dx * 1e3:.3f}mm_{h * 1e6:.1f}us.json"
    if arm == "gs_mpm" and theory.exists():
        start = (np.log(json.loads(theory.read_text())["E_ratio"]), z)
    E, Z, misfit, nfev = c.fit(block.measure, [start], zeta_free=free)
    if misfit > 1e-8:
        E2, Z2, m2, n2 = c.fit(block.measure, [(np.log(S), c.ZETA0)], zeta_free=free)
        nfev += n2
        if m2 < misfit:
            E, Z, misfit = E2, Z2, m2
    t = block.steps * h
    q_true = c.truth_q(t)
    err = {}
    for name, (Er, Zr) in {"nominal": (1.0, c.ZETA0), "fitted": (E, Z if free else 0.0)}.items():
        q = block.modal(Er, Zr)
        err[name] = float(np.sqrt(np.mean((q - q_true) ** 2)) / c.amplitude() * c.omega0())
    row = {"arm": arm, "dx": dx, "h": h, "E_ratio": E, "zeta": Z, "misfit": misfit, "nfev": nfev,
           "S_theory": S, "time_theory": [e, z], "q_rmse": err, "seconds": time.time() - t0}
    import os
    os.listdir(PARTS.parent)   # an NFS client may cache that the directory is missing; a listing refreshes it
    PARTS.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, indent=1))
    return out.name


def merge():
    rows = [json.loads(p.read_text()) for p in sorted(PARTS.glob("*.json"))]
    (c.RESULTS / "flow.json").write_text(json.dumps(rows, indent=1))
    print("merged", len(rows), "points")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=("mujoco", "genesis"))
    ap.add_argument("--arm", action="append")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--count", action="store_true", help="print the number of tasks")
    ap.add_argument("--task", type=int, help="run only this task (a Slurm array index)")
    args = ap.parse_args()
    if args.merge:
        merge()
        return
    try:   # before any work: on NFS, many array tasks creating it at once can race
        PARTS.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        pass
    table = theory_table()
    arms = args.arm or [a for a, v in ARMS.items() if v[0] == args.engine]
    tasks = [(a, dx, h) for a in arms for dx in ARMS[a][4] for h in steps_for(a, dx, table)]
    # the costliest first: fine cells and small steps
    tasks.sort(key=lambda t: -(1 / t[1]) ** 3 / t[2])
    if args.count:
        print(len(tasks))
        return
    if args.task is not None:
        print("done", run_point(tasks[args.task]), flush=True)
        return
    with Pool(args.jobs, maxtasksperchild=1) as pool:
        for name in pool.imap_unordered(run_point, tasks):
            print("done", name, flush=True)


if __name__ == "__main__":
    main()
