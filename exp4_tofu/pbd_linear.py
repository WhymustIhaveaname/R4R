"""Experiment 4a's theory for Genesis's PBD: one step, linearized about rest.

Genesis 1.5.0's PBD elastic body (PBD3DEntity, solvers/pbd_solver.py) gives every particle the
same mass m and, each step of h, moves it by its velocity, then projects in Jacobi sweeps the
stretch constraint of every edge of its tetrahedra n_s times and the volume constraint of every
tetrahedron n_v times, and takes the velocity as the displacement over the step over h:

  stretch   dp_i = -r C / (w_i + w_j + alpha / h^2) n,   C = |x_i - x_j| - l,   w = 1 / m
  volume    dp_k = s w_k g_k,   s = -r_v C_V / (sum w_k + alpha_v / h^2),   w_k = |g_k|^2 / m

with r = 0.1 the relaxation, alpha the compliance, g_k the gradient of the volume.  Two things
follow from the code:

  - the volume correction carries |g_k|^2 once too often (m^4, about 1e-10 at 1 cm cells): the
    volume constraint does nothing, and the body is a network of springs along its edges, whose
    Poisson ratio is fixed by the network (Cauchy), not 0.45;
  - no multiplier is accumulated, so as the sweeps go on each constraint is met whatever its
    compliance: the compliance alone is no stiffness, the step's projections are.

About rest a sweep is linear, u <- P u, P = I - c L, c = r / (2 + m alpha / h^2), L the stiffness
of the network with unit springs.  A step is u' = P^n_s (2 u - u_prev), so a mode of L with
eigenvalue mu has per step lambda^2 - 2 p lambda + p = 0, p = (1 - c mu)^n_s: |lambda|^2 = p and
arg lambda = arccos(sqrt p).  Its frequency and damping follow; E* is the compliance that gives the
continuum's frequency, and none does once p at alpha = 0 is not low enough.

  python pbd_linear.py
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

import common as c
import p1

RELAX = 0.1      # Genesis's default stretch_relaxation
N_STRETCH = 4    # PBDOptions.max_stretch_solver_iterations, by default


def network(dx):
    """MuJoCo's grid mesh, its edges and the stiffness L of unit springs along them."""
    X, tets = p1.mesh_from_mujoco(dx)
    e = {tuple(sorted((t[a], t[b]))) for t in tets for a in range(4) for b in range(a + 1, 4)}
    e = np.array(sorted(e))
    n = (X[e[:, 0]] - X[e[:, 1]])
    n /= np.linalg.norm(n, axis=1)[:, None]
    rows, cols, vals = [], [], []
    for a in range(3):
        for b in range(3):
            k = n[:, a] * n[:, b]
            for (i, j, s) in ((0, 0, 1), (1, 1, 1), (0, 1, -1), (1, 0, -1)):
                rows.append(3 * e[:, i] + a)
                cols.append(3 * e[:, j] + b)
                vals.append(s * k)
    L = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(3 * len(X), 3 * len(X))).tocsr()
    return X, tets, e, L


def lowest_elastic(dx):
    """The network's lowest elastic eigenvalue, past the six rigid modes, and its mode."""
    X, tets, e, L = network(dx)
    v0 = np.random.default_rng(0).standard_normal(L.shape[0])
    w, V = eigsh(L, k=16, sigma=-1e-3, which="LM", v0=v0, ncv=64, tol=1e-12)
    order = np.argsort(w)
    w, V = w[order], V[:, order]
    rigid = int((np.abs(w) < 1e-8).sum())
    assert rigid == 6, w[:8]
    return w[6], V[:, 6].reshape(-1, 3), X


def mode(mu, alpha, h, m, n_s=N_STRETCH, r=RELAX):
    """(omega_n rad/s, zeta) of a mode of eigenvalue mu, or None if the step does not oscillate."""
    cc = r / (2 + m * alpha / h**2)
    p = (1 - cc * mu) ** n_s
    if not 0 < p < 1:
        return None
    wd = np.arccos(np.sqrt(p)) / h
    s = -np.log(p) / (2 * h)
    wn = np.hypot(wd, s)
    return wn, s / wn


def nominal_compliance(dx):
    """The compliance of an edge spring E dx, as nominal: E* scales it by 1 / E*."""
    return 1 / (c.E0 * dx)


def e_star(dx, h, n_s=N_STRETCH, r=RELAX, mu=None, m=None):
    """E* / E (the compliance that gives the continuum's frequency) and PBD's damping there, or
    (nan, nan) when even zero compliance is too soft."""
    from scipy.optimize import brentq
    if mu is None:
        mu, _, X = lowest_elastic(dx)
        m = c.RHO * np.prod(c.SIDES) / len(X)
    w0 = c.omega0()
    top = mode(mu, 0.0, h, m, n_s, r)
    if top is None or top[0] < w0:
        return np.nan, np.nan
    a0 = nominal_compliance(dx)
    f = lambda lg: mode(mu, a0 / 10**lg, h, m, n_s, r)[0] - w0      # noqa: E731
    lo, hi = -6.0, 12.0
    lg = brentq(f, lo, hi, xtol=1e-12)
    return 10**lg, mode(mu, a0 / 10**lg, h, m, n_s, r)[1]


def volume_map(X, tets, m, alpha_v, h, r=RELAX):
    """One Jacobi sweep of Genesis's volume constraints, linearized: u <- u - Bw^T D Bv u."""
    from scipy.sparse import diags
    P = X[tets]
    g = np.stack([np.cross(P[:, 3] - P[:, 1], P[:, 2] - P[:, 1]), np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 0]),
                  np.cross(P[:, 3] - P[:, 0], P[:, 1] - P[:, 0]), np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])], 1) / 6
    w = (g**2).sum(2) / m                                   # Genesis's w_k = |g_k|^2 / m
    rows = np.repeat(np.arange(len(tets)), 12)
    cols = (3 * tets[:, :, None] + np.arange(3)).reshape(-1, 12).ravel()
    Bv = coo_matrix((g.reshape(-1, 12).ravel(), (rows, cols)), shape=(len(tets), 3 * len(X))).tocsr()
    Bw = coo_matrix(((w[:, :, None] * g).reshape(-1, 12).ravel(), (rows, cols)), shape=(len(tets), 3 * len(X))).tocsr()
    D = diags(r / (w.sum(1) + alpha_v / h**2))
    return Bw.T @ D @ Bv


def simulate(dx, h, E_ratio, n_s=N_STRETCH, r=RELAX):
    """Genesis's PBD step, linearized, run from the start GsPbdBlock gives and read as it reads:
    (omega_n, zeta) from common.measure."""
    from scipy.sparse import identity
    X, tets, e, L = network(dx)
    m = c.RHO * np.prod(c.SIDES) / len(X)
    cc = r / (2 + m * nominal_compliance(dx) / E_ratio / h**2)
    Ps = identity(3 * len(X), format="csr") - cc * L
    Vm = volume_map(X, tets, m, 0.0, h)                     # GsPbdBlock leaves volume_compliance at 0
    masses = np.full(len(X), m)
    shape = p1.mode_vector(dx, "equal")
    v0 = (c.amplitude() * shape + c.seeded_noise(X, masses)).ravel()
    w = c.modal_weights(shape, masses).ravel()
    steps = c.sample_steps(h)
    u = np.zeros_like(v0)
    vel = v0.copy()
    q = np.empty(len(steps))
    j = 0
    for n in range(1, steps[-1] + 1):
        x = u + h * vel
        for _ in range(n_s):
            x = Ps @ x
        x = x - Vm @ x
        vel = (x - u) / h
        u = x
        if n == steps[j]:
            q[j] = w @ u
            j += 1
    return c.measure(steps * h, q)


DXS = (0.01, 0.005, 0.01 / 3, 0.0025, 0.002)
FRACTIONS = (0.1, 0.25, 0.5, 0.9)      # of h_max
PARTS = c.RESULTS / "pbd_parts"


def h_max(dx, n_s=N_STRETCH, r=RELAX):
    """The largest step at which zero compliance still reaches the continuum's frequency: at
    alpha = 0, p = (1 - r mu / 2)^n_s does not depend on h, so omega_n goes as 1 / h."""
    mu, _, _ = lowest_elastic(dx)
    p0 = (1 - r * mu / 2) ** n_s
    return np.hypot(np.arccos(np.sqrt(p0)), np.log(p0) / 2) / c.omega0()


def tasks():
    return [(dx, float(f"{f * h_max(dx):.4g}"), f) for dx in DXS for f in FRACTIONS]


def theory(dx, h, E_guess):
    """E*/E and the damping from the linearized step, read as Genesis's runs are read."""
    from scipy.optimize import brentq
    w0 = c.omega0()
    f = lambda lg: np.log(simulate(dx, h, np.exp(lg))[0] / w0)    # noqa: E731
    lo, hi = np.log(E_guess) - 0.05, np.log(E_guess) + 0.05
    while f(lo) > 0:
        lo -= 0.1
    while f(hi) < 0:
        hi += 0.1
    lg = brentq(f, lo, hi, xtol=1e-10)
    return float(np.exp(lg)), float(simulate(dx, h, np.exp(lg))[1])


def run_task(i):
    import json
    import time
    dx, h, frac = tasks()[i]
    out = PARTS / f"{dx * 1e3:.3f}mm_{h * 1e6:.2f}us.json"
    if out.exists():
        return out.name
    t0 = time.time()
    mu, _, X = lowest_elastic(dx)
    m = c.RHO * np.prod(c.SIDES) / len(X)
    E_mode, z_mode = e_star(dx, h, mu=mu, m=m)
    E_th, z_th = theory(dx, h, E_mode)
    from genesis_sim import GsPbdBlock
    block = GsPbdBlock("gs_pbd", dx, h)
    E, Z, misfit, nfev = c.fit(block.measure, [(np.log(E_th), 0.0)], zeta_free=False)
    row = {"arm": "gs_pbd", "dx": dx, "h": h, "fraction": frac, "E_ratio": E, "zeta": Z, "misfit": misfit,
           "nfev": nfev, "E_theory": E_th, "zeta_theory": z_th, "E_mode": E_mode, "zeta_mode": z_mode,
           "seconds": time.time() - t0}
    PARTS.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, indent=1))
    return out.name


def merge():
    import json
    rows = [json.loads(p.read_text()) for p in sorted(PARTS.glob("*.json"))]
    (c.RESULTS / "pbd.json").write_text(json.dumps(rows, indent=1))
    print("merged", len(rows))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int)
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.count:
        print(len(tasks()))
    elif args.merge:
        merge()
    elif args.task is not None:
        print("done", run_task(args.task), flush=True)
    else:
        for dx in DXS:
            print(f"dx {dx * 1e3:.2f} mm: h_max {h_max(dx) * 1e3:.4f} ms")
if False:
    for dx in (0.01, 0.005, 0.01 / 3, 0.0025, 0.002):
        mu, _, X = lowest_elastic(dx)
        m = c.RHO * np.prod(c.SIDES) / len(X)
        line = []
        for h in (5e-5, 1e-4, 2e-4, 5e-4, 1e-3):
            E, z = e_star(dx, h, mu=mu, m=m)
            top = mode(mu, 0.0, h, m)
            line.append(f"h {h * 1e3:.2f} ms: E*/E {E:9.4g} zeta {z:.4f} (max f {top[0] / 2 / np.pi if top else 0:.1f} Hz)")
        print(f"dx {dx * 1e3:.2f} mm, mu {mu:.4g}, m {m:.3g} kg:\n   " + "\n   ".join(line), flush=True)
