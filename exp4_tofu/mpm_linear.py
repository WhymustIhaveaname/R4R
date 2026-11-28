"""The theory of Genesis's MPM on the free block: its step, linearized, and its lowest mode.

Genesis's MPM is MLS-MPM: particles carry v, the affine velocity C and the deformation gradient
F; each step scatters m v + (s tau + m C)(x_i - x_p) to the grid with quadratic B-spline weights
w_ip (s = -h V_p 4 / dx^2, tau the Kirchhoff stress of F advanced by I + h C), divides by the
node's mass, and gathers v = sum w v_i and C = 4/dx^2 sum w v_i (x_i - x_p)^T back.

At the experiment's amplitude (strain 1e-3) the particles stay where they started, to first
order, so the weights are constants and a step is linear.  The particles' v and C are then the
gather of the grid's last velocity V, and their F - I the gradient of the grid's summed
displacement U, so the state is (U, V) on the grid alone:

  U' = U + h V,    V' = T V - h M^-1 K U',

T = M^-1 m (W W^T + 4/dx^2 Q Q^T) the transfer there and back (W the weights, Q the weights
times x_i - x_p), and K = V_p (4/dx^2)^2 Q L Q^T the stiffness, L linear elasticity, to which
Genesis's corotated stress reduces.  T is where MPM loses energy: it keeps a grid field only as
far as its particles' affine fields can carry it.

The eigenvalue lambda of the step nearest the continuum's lowest elastic mode gives that mode's
frequency and damping: lambda = exp((-s + i w_d) h), w_n = |s + i w_d|, zeta = s / w_n, as
common.measure fits them.  E* is the E that brings w_n to the continuum's 14.4 Hz.
"""

import os
from functools import lru_cache

# The solves are small; on a busy machine more BLAS threads only contend.  This holds when
# the module is imported before numpy, as when it is run.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from scipy.sparse import bmat, coo_matrix, diags, identity  # noqa: E402
from scipy.sparse.linalg import splu  # noqa: E402

import common as c  # noqa: E402


def particles(dx):
    """Genesis's regular sampler: particles of size dx / 2 at (i + 1/2) dx / 2 in a block whose
    faces lie on grid nodes, here with the block's corner at the grid's origin."""
    n = np.round(2 * c.SIDES / dx).astype(int)
    axes = [(np.arange(k) + 0.5) * dx / 2 for k in n]
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)


class LinearMPM:
    def __init__(self, dx, E=c.E0, nu=c.NU, rho=c.RHO):
        X = particles(dx)
        vol = (dx / 2) ** 3
        m = rho * vol
        fx = X / dx
        base = np.floor(fx - 0.5).astype(int)
        f = fx - base
        wts = np.stack([0.5 * (1.5 - f) ** 2, 0.75 - (f - 1.0) ** 2, 0.5 * (f - 0.5) ** 2], 1)  # (p, 3, d)
        offs = np.stack(np.meshgrid(*[np.arange(3)] * 3, indexing="ij"), -1).reshape(-1, 3)    # 27
        node = base[:, None, :] + offs[None]                                                     # (p, 27, 3)
        w = wts[:, offs[:, 0], 0] * wts[:, offs[:, 1], 1] * wts[:, offs[:, 2], 2]               # (p, 27)
        d = (offs[None] - f[:, None, :]) * dx                                                   # x_i - x_p
        keys, inv = np.unique(node.reshape(-1, 3), axis=0, return_inverse=True)
        inv = inv.reshape(len(X), 27)
        self.nodes = keys * dx
        self.X, self.dx, self.m, self.vol = X, dx, m, vol
        ng, npart = len(keys), len(X)
        rows = inv.ravel()
        cols = np.repeat(np.arange(npart), 27)
        W = coo_matrix((w.ravel(), (rows, cols)), shape=(ng, npart)).tocsr()
        self.mass = m * np.asarray(W.sum(1)).ravel()
        # Q maps a particle's 3 x 3 matrix A to the nodes: (Q A)_i = sum_p w_ip A_p d_ip, per component
        # a of the node's vector: sum_b A_p[a, b] d_ip[b].  Built per (a, b).
        Qb = [coo_matrix(((w * d[:, :, b]).ravel(), (rows, cols)), shape=(ng, npart)).tocsr() for b in range(3)]
        self.W, self.Qb = W, Qb
        self.ng, self.npart = ng, npart
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        # Node vectors are stacked by component: [all x, all y, all z].
        I3 = identity(3, format="csr")
        from scipy.sparse import kron
        Wm = m * (W @ W.T)
        Qsum = sum(Q @ Q.T for Q in Qb)
        self.B = kron(I3, Wm + m * 4 / dx**2 * Qsum).tocsr()        # M T, symmetric
        # K = vol (4/dx^2)^2 Q L Q^T: the gradient G_p[a, b] = sum_i w d_b u_i[a] = (Qb^T u_a)_p, then
        # tau = 2 mu sym(G) + lam tr(G) I, and the force on node component a is sum_b Qb tau[a, b].
        blocks = [[None] * 3 for _ in range(3)]
        for a in range(3):
            for e in range(3):          # coupling node component a <- node component e
                acc = None
                for b in range(3):
                    # tau[a, b] gets mu G[a, b] + mu G[b, a] + lam delta_ab tr G
                    terms = []
                    if e == a:
                        terms.append(mu * Qb[b] @ Qb[b].T)                    # mu G[a, b], G[a,b] from u_a, Q_b
                    if e == b:
                        terms.append(mu * Qb[b] @ Qb[a].T)                    # mu G[b, a]: from u_b, Q_a
                    if a == b:
                        terms.append(lam * Qb[b] @ Qb[e].T)                   # lam tr G: G[e, e] from u_e, Q_e
                    for t in terms:
                        acc = t if acc is None else acc + t
                blocks[a][e] = acc
        self.K = vol * (4 / dx**2) ** 2 * bmat(blocks).tocsr()
        self.M = diags(np.tile(self.mass, 3))

    def eigen(self, h, ratio, sigma, U0, tol=1e-13):
        """The eigenvalue of the step at E = ratio E0 nearest the mode U0, by Rayleigh quotient
        iteration from sigma, with its U part.

        From U' = lambda U and V = (lambda - 1) U / h:
          lambda^2 M U + lambda (h^2 K - M - B) U + B U = 0,
        solved as the companion pencil A z = lambda Bm z, z = (U, lambda U)."""
        n = 3 * self.ng
        M, B, K = self.M, self.B, self.K
        Id = identity(n, format="csr")
        A = bmat([[None, Id], [-B, -(ratio * h * h * K - M - B)]]).tocsc()
        Bm = bmat([[Id, None], [None, M]]).tocsc()
        z = np.concatenate([U0, sigma * U0]).astype(complex)
        for _ in range(40):
            z = splu((A - sigma * Bm).tocsc()).solve(Bm @ z)
            z /= np.linalg.norm(z)
            new = np.vdot(z, A @ z) / np.vdot(z, Bm @ z)
            done = abs(new - sigma) < tol
            sigma = new
            if done:
                break
        else:
            raise RuntimeError("the Rayleigh quotient iteration did not converge")
        return sigma, z[:n]

    def mode(self, h, ratio, shape_nodes, sigma=None):
        """(w_n, zeta, lambda, overlap) of the step's mode that starts from shape_nodes."""
        s = shape_nodes.T.ravel()                         # by component, as the state
        if sigma is None:
            sigma = np.exp(1j * c.omega0() * np.sqrt(ratio) * h)
        lam, U = self.eigen(h, ratio, sigma, s)
        mw = np.tile(self.mass, 3)
        overlap = abs(np.vdot(U * mw, s)) / np.sqrt(np.vdot(U * mw, U).real * np.vdot(s * mw, s).real)
        sd = -np.log(abs(lam)) / h
        wd = np.angle(lam) / h
        wn = np.hypot(sd, wd)
        return wn, sd / wn, lam, overlap


def simulate(L, h, ratio, v0, weights, steps):
    """The amplitude q of the mode in the particles' displacement, at the given step indices,
    as Genesis's run measures it: the particles start at v0 with C = 0 and F = I, so the first
    step only scatters m v0; then U' = U + h V, V' = T V - h M^-1 K U', and the particles have
    moved by W^T (U + h V)."""
    mw = np.tile(L.mass, 3)
    Tm = L.B.multiply(1 / mw[:, None]).tocsr()
    Km = (ratio * h * L.K).multiply(1 / mw[:, None]).tocsr()
    Wt = L.W.T.tocsr()
    V = np.concatenate([L.m * (L.W @ v0[:, a]) for a in range(3)]) / mw
    U = np.zeros_like(V)
    q = np.full(len(steps), np.nan)
    j = 0
    ng = L.ng
    for n in range(1, steps[-1] + 1):
        if n > 1:
            U = U + h * V
            V = Tm @ V - Km @ U
        if n == steps[j]:
            D = U + h * V
            u = np.stack([Wt @ D[a * ng:(a + 1) * ng] for a in range(3)], 1)
            q[j] = np.sum(weights * u)
            j += 1
    return q


def measured(dx, h):
    """(E*/E, zeta, misfit) of the linearized MPM through common.fit, as run_flow fits Genesis."""
    L = LinearMPM(dx)
    X = L.X - c.HALF
    shape = c.mode_shape(X)
    v0 = c.amplitude() * shape + c.seeded_noise(X, np.ones(len(X)))
    weights = c.modal_weights(shape, np.ones(len(X)))
    steps = c.sample_steps(h)
    t = steps * h
    E, Z, misfit, _ = c.fit(lambda r, z: c.measure(t, simulate(L, h, r, v0, weights, steps)), [(0.0, 0.0)],
                            zeta_free=False)
    return E, Z, misfit


@lru_cache(maxsize=None)
def prediction(dx, h):
    """(E*/E, zeta_MPM, overlap) of the linearized MPM, cached in memory.  E enters K alone."""
    L = LinearMPM(dx)
    shape = c.mode_shape(L.nodes - c.HALF)
    last = {}

    def resid(ratio):
        wn, z, lam, ov = L.mode(h, ratio, shape, last.get("lam"))
        last.update(lam=lam, z=z, ov=ov)
        return np.log(wn / c.omega0())

    r0, r1 = 1.0, 1.02                                    # the secant method, warm-started
    f0, f1 = resid(r0), resid(r1)
    for _ in range(30):
        r2 = r1 - f1 * (r1 - r0) / (f1 - f0)
        r0, f0, r1, f1 = r1, f1, r2, resid(r2)
        if abs(f1) < 1e-12:
            break
    return r1, last["z"], last["ov"]


PARTS = c.RESULTS / "mpm_theory_parts"


def tasks():
    """Every (dx, h) of the MPM sweep, the finest cells first: they take longest."""
    import run_flow as rf
    out = [(dx, h) for dx in rf.DXS for h in rf.steps_for("gs_mpm", dx, None)]
    return sorted(out, key=lambda t: t[0])


def run_point(dx, h):
    import json
    import time
    out = PARTS / f"{dx * 1e3:.3f}mm_{h * 1e6:.1f}us.json"
    if out.exists():
        return out.name
    t0 = time.time()
    E, z, misfit = measured(dx, h)
    os.listdir(PARTS.parent)   # an NFS client may cache that the directory is missing
    PARTS.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"dx": dx, "h": h, "E_ratio": E, "zeta": z, "misfit": misfit,
                               "seconds": time.time() - t0}, indent=1))
    return out.name


def merge():
    import json
    rows = [json.loads(q.read_text()) for q in sorted(PARTS.glob("*.json"))]
    (c.RESULTS / "mpm_theory.json").write_text(json.dumps(rows, indent=1))
    print("merged", len(rows))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--task", type=int)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.count:
        print(len(tasks()))
    elif args.merge:
        merge()
    else:
        print("done", run_point(*tasks()[args.task]), flush=True)
