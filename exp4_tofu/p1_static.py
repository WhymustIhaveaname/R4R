"""Experiment 4b's spatial discretization alone, at large deformation: the static P1 slab.

What p1.py's modes are to the vibration, this is to the indentation: the engines' own tetrahedra
on MuJoCo's grid, in the engine's own material, pressed by the rigid frictionless sphere with no
time step and no compliance in the contact.  It is solved for each depth as axisym_nl.py solves
the continuum: Newton's method on the energy, with a line search, a Levenberg shift where the
Hessian is not positive, no element allowed to invert, and the contact as an augmented
Lagrangian.  Each tetrahedron's Hessian is twelve forward derivatives of its gradient, from torch.

The contact is that of the engine:

  "surface"  MuJoCo: the sphere collides with the elements, so the whole piecewise-linear top
             surface stays out of it; held at a lattice of points on each top triangle
  "vertex"   Genesis's legacy coupler: the top vertices alone stay out
  "winkler"  Genesis's SAP: the whole surface, through a compliant layer (WINKLER)
"""

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.sparse import coo_matrix, csr_matrix, diags  # noqa: E402
from scipy.sparse.linalg import splu  # noqa: E402

import common as c  # noqa: E402
import indentation as ind  # noqa: E402
from p1_contact import lattice, slab_mesh, top_triangles  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

# SAP's hydroelastic contact of a rigid geom with an FEM body (sap_coupler.RigidFEMContactHandler):
# each contact carries its area times the FEM's pressure where the geom's triangle cuts the tet,
# the pressure growing from the surface at the FEM's hydroelastic modulus, 1e7 Pa, over its
# deepest vertex, 2 cm in the slab.  SAP's own stiffness, the harmonic mean of that gradient and
# its hydroelastic stiffness, only linearizes the contact within a step; at rest the force is the
# FEM's pressure.
WINKLER = 1e7 / 0.02   # Pa/m


def polar_rotation(F, iterations=8):
    """The rotation of F's polar decomposition by the scaled Newton iteration, as Genesis takes it."""
    X = F
    for _ in range(iterations):
        Y = torch.linalg.inv(X).transpose(1, 2)
        z = torch.sqrt(torch.sqrt((Y * Y).sum((1, 2)) / (X * X).sum((1, 2))))[:, None, None]
        X = 0.5 * (z * X + Y / z)
    return X


def density(model, mu, lam):
    eye = torch.eye(3)

    def strain_energy(e):
        tr = e[:, 0, 0] + e[:, 1, 1] + e[:, 2, 2]
        return mu * (e * e).sum((1, 2)) + 0.5 * lam * tr * tr

    def psi(F):
        if model == "stvk":
            return strain_energy(0.5 * (F.transpose(1, 2) @ F - eye))
        if model == "corotated":
            S = polar_rotation(F).transpose(1, 2) @ F
            return strain_energy(0.5 * (S + S.transpose(1, 2)) - eye)
        if model == "neohookean":
            lam_s = lam + mu
            alpha = 1 + mu / lam_s
            return 0.5 * (mu * ((F * F).sum((1, 2)) - 3) + lam_s * (torch.linalg.det(F) - alpha) ** 2)
        raise ValueError(model)
    return psi


class P1Static:
    def __init__(self, dx, model, kind, radius=ind.RADIUS, E=c.E0, nu=c.NU, points_per_edge=8):
        X, tets = slab_mesh(dx)
        self.X, self.tets, self.dx, self.R, self.kind = X, tets, dx, radius, kind
        z = X[:, 2]
        self.H = z.max()
        bottom = np.isclose(z, z.min())
        top = np.isclose(z, z.max())
        n = 3 * len(X)
        self.free = np.flatnonzero(np.repeat(~bottom, 3))
        self.dofs = (3 * tets[:, :, None] + np.arange(3)).reshape(-1, 12)
        Dm = np.stack([X[tets[:, k]] - X[tets[:, 0]] for k in (1, 2, 3)], 2)
        self.Bm = torch.tensor(np.linalg.inv(Dm))
        self.vol = torch.tensor(np.abs(np.linalg.det(Dm)) / 6)
        mu = E / (2 * (1 + nu))
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        psi = density(model, mu, lam)

        def energies(x):
            x = x.reshape(-1, 4, 3)
            Ds = torch.stack([x[:, k] - x[:, 0] for k in (1, 2, 3)], 2)
            return self.vol * psi(Ds @ self.Bm)
        self._energies = energies
        self._rc = (np.repeat(self.dofs, 12, 1).ravel(), np.tile(self.dofs, (1, 12)).ravel())

        # The points the contact holds out of the sphere, as a sparse map from vertex positions
        r = np.hypot(X[:, 0], X[:, 1])
        reach = 1.6 * np.sqrt(radius * ind.DEPTHS.max()) + 2 * dx
        if kind == "vertex":
            v = np.flatnonzero(top & (r < reach))
            self.Wc = csr_matrix((np.ones(len(v)), (np.arange(len(v)), v)), shape=(len(v), len(X)))
        else:
            tris = top_triangles(X, tets, top)
            tris = tris[(r[tris] < reach + 1.5 * dx).all(1)]
            bary = lattice(points_per_edge)
            rows = np.repeat(np.arange(len(tris) * len(bary)), 3)
            cols = np.repeat(tris, len(bary), 0).ravel()
            self.Wc = csr_matrix((np.tile(bary, (len(tris), 1)).ravel(), (rows, cols)),
                                 shape=(len(tris) * len(bary), len(X)))
        # a penalty per point a thousand times the slab's stiffness there, about E dx; or, for
        # SAP's hydroelastic contact, a compliant layer: the pressure grows as WINKLER (Pa/m) times
        # the penetration, each lattice point carrying its triangle's area over the points
        self.winkler = kind == "winkler"
        if self.winkler:
            area = 0.5 * np.linalg.norm(np.cross(X[tris[:, 1]] - X[tris[:, 0]], X[tris[:, 2]] - X[tris[:, 0]]), axis=1)
            self.kappa = np.repeat(area / len(bary), len(bary)) * WINKLER
        else:
            self.kappa = np.full(self.Wc.shape[0], 1e3 * E * dx)
        self.mult = np.zeros(self.Wc.shape[0])
        self.u = np.zeros(n)
        self._delta = 0.0

    def _elastic(self, u, order):
        x = torch.tensor((self.X.ravel() + u)[self.dofs])
        if order == 0:
            return float(self._energies(x).sum())
        n = 3 * len(self.X)

        def gradient(x):
            return torch.func.grad(lambda y: self._energies(y).sum())(x)
        g = np.zeros(n)
        np.add.at(g, self.dofs, gradient(x).numpy())
        He = np.empty((len(self.tets), 12, 12))
        for k in range(12):
            e = torch.zeros_like(x)
            e[:, k] = 1
            He[:, :, k] = torch.func.jvp(gradient, (x,), (e,))[1].numpy()
        return g, coo_matrix((He.ravel(), self._rc), shape=(n, n)).tocsr()

    def min_jacobian(self, u):
        x = (self.X.ravel() + u)[self.dofs].reshape(-1, 4, 3)
        Ds = np.stack([x[:, k] - x[:, 0] for k in (1, 2, 3)], 2)
        return (np.linalg.det(Ds @ self.Bm.numpy())).min()

    def _contact(self, u, centre, order):
        x = self.Wc @ (self.X + u.reshape(-1, 3))
        v = x - centre
        dist = np.linalg.norm(v, axis=1)
        phi = dist - self.R
        p = np.maximum(self.mult - self.kappa * phi, 0)
        energy = ((p**2 - self.mult**2) / (2 * self.kappa)).sum()
        if order == 0:
            return energy
        nrm = v / dist[:, None]
        gpts = -p[:, None] * nrm                       # gradient at the points
        g = (self.Wc.T @ gpts).ravel()
        on = p > 0
        Hp = (self.kappa * on)[:, None, None] * nrm[:, :, None] * nrm[:, None, :] \
            - (p / dist)[:, None, None] * (np.eye(3) - nrm[:, :, None] * nrm[:, None, :])
        # H = W^T Hp W, by component pairs
        n = 3 * len(self.X)
        act = np.flatnonzero(on | (self.mult > 0))
        Wa = self.Wc[act].tocoo()
        blocks = []
        rows, cols, vals = [], [], []
        for a in range(3):
            for b in range(3):
                Wab = (Wa.T @ diags(Hp[act, a, b]) @ Wa).tocoo()
                rows.append(3 * Wab.row + a)
                cols.append(3 * Wab.col + b)
                vals.append(Wab.data)
        del blocks
        H = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
        return energy, g, H, p, nrm, phi

    def _total(self, u, centre):
        if self.min_jacobian(u) < 0.1:     # a crushed or inverted element (see axisym_nl._total)
            return np.inf
        return self._elastic(u, 0) + self._contact(u, centre, 0)

    def _newton(self, centre, tol):
        n = 3 * len(self.X)
        shift = 0.0
        for it in range(300):
            g, K = self._elastic(self.u, 2)
            _, gc, Hc, *_ = self._contact(self.u, centre, 1)
            g = g + gc
            A = (K + Hc)[self.free][:, self.free].tocsc()
            gf = g[self.free]
            if np.abs(gf).max() < tol:
                return it
            D = diags(np.abs(A.diagonal()))
            while True:
                step = np.zeros(n)
                # the symmetric ordering fills the factor 25 % less than the default and is 1.5 to 2
                # times faster, if the pivots stay on the diagonal
                step[self.free] = -splu((A + shift * D).tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.1,
                                        options=dict(SymmetricMode=True)).solve(gf)
                slope = gf @ step[self.free]
                if slope < 0:
                    break
                shift = max(1e-8, 10 * shift)
            e0, t = self._total(self.u, centre), 1.0
            near = np.abs(gf).max() < 1e3 * tol
            while True:
                e1 = self._total(self.u + t * step, centre)
                if e1 <= e0 + 1e-4 * t * slope or (near and e1 - e0 <= 1e-13 * abs(e0)):
                    break
                t /= 2
                if t < 1e-12:
                    raise RuntimeError("the line search failed")
            self.u += t * step
            # the shift is for a Hessian that is not positive; a short step only means the
            # contact set is changing, which the line search handles
            shift = shift / 10 if shift > 1e-8 else 0.0
            if near and np.abs(t * step).max() < 1e-15:
                return it
        raise RuntimeError("Newton did not converge")

    def indent(self, delta, substeps=2):
        """Force (N) on the sphere pressed delta into the top, continuing from the last call."""
        for d in np.linspace(self._delta, delta, substeps + 1)[1:]:
            centre = np.array([0.0, 0.0, self.H + self.R - d])
            if getattr(self, "_last", None) is not None:      # extrapolate from the last two depths
                (d1, u1), (d2, u2) = self._last
                guess = self.u + (u2 - u1) * (d - d2) / (d2 - d1)
                if self.min_jacobian(guess) >= 0.1:
                    self.u = guess
            for _ in range(50):
                self._newton(centre, tol=1e-7 * max(1.0, self.mult.sum()))   # N; the contact set flips below
                *_, p, nrm, phi = self._contact(self.u, centre, 1)
                if self.winkler:                    # a compliant contact: the penalty is the law
                    break
                self.mult = p
                if -phi.min() < 1e-7 * d:
                    break
            else:
                raise RuntimeError("the contact did not settle")
            prev = getattr(self, "_last", None)
            self._last = ((prev[1] if prev else (self._delta, np.zeros_like(self.u))), (d, self.u.copy()))
        self._delta = delta
        return float((p * -nrm[:, 2]).sum())


class Gradient:
    """The gradient of a P1 mesh's elastic energy at given vertex positions: the force (N) each
    vertex must be held with, as an engine with the same material sees it."""

    def __init__(self, X, tets, model, E=c.E0, nu=c.NU):
        Dm = np.stack([X[tets[:, k]] - X[tets[:, 0]] for k in (1, 2, 3)], 2)
        Bm = torch.tensor(np.linalg.inv(Dm))
        vol = torch.tensor(np.abs(np.linalg.det(Dm)) / 6)
        psi = density(model, E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu)))
        self.dofs = (3 * tets[:, :, None] + np.arange(3)).reshape(-1, 12)
        self.n = 3 * len(X)

        def total(x):
            x = x.reshape(-1, 4, 3)
            Ds = torch.stack([x[:, k] - x[:, 0] for k in (1, 2, 3)], 2)
            return (vol * psi(Ds @ Bm)).sum()
        self._grad = torch.func.grad(total)

    def __call__(self, x):
        g = np.zeros(self.n)
        np.add.at(g, self.dofs, self._grad(torch.tensor(np.asarray(x).ravel()[self.dofs])).numpy())
        return g


def static_forces(dx, model, kind, deep=False):
    """The static P1 slab's force (N) at indentation.DEPTHS, or at DEEP, cached."""
    depths = ind.DEEP if deep else ind.DEPTHS
    path = c.CACHE / f"p1_static_{model}_{kind}_{ind.RADIUS * 1e3:.1f}mm_{dx:.6f}{'_deep' if deep else ''}.npy"
    if path.exists():
        return np.load(path)
    P = P1Static(dx, model, kind)
    F = []
    for d in depths:
        F.append(P.indent(d))
        print(f"  {d * 1e3:.2f} mm: {F[-1]:.5f} N", flush=True)
    F = np.array(F)
    c.CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, F)
    return F


def _closest_on_triangles(a, b, c, p):
    """The points of triangles abc nearest p, and their barycentric weights (Ericson's method)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp = p - b
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    cp = p - c
    d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    n = len(a)
    w = np.zeros((n, 3))
    done = np.zeros(n, bool)

    def put(mask, wa, wb, wc):
        m = mask & ~done
        w[m, 0], w[m, 1], w[m, 2] = wa[m], wb[m], wc[m]
        done[m] = True
    one, zero = np.ones(n), np.zeros(n)
    put((d1 <= 0) & (d2 <= 0), one, zero, zero)
    put((d3 >= 0) & (d4 <= d3), zero, one, zero)
    put((d6 >= 0) & (d5 <= d6), zero, zero, one)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = d1 / (d1 - d3)
        put((vc <= 0) & (d1 >= 0) & (d3 <= 0), 1 - t, t, zero)
        t = d2 / (d2 - d6)
        put((vb <= 0) & (d2 >= 0) & (d6 <= 0), 1 - t, zero, t)
        t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        put((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), zero, 1 - t, t)
        den = va + vb + vc
        v_, w_ = vb / den, vc / den
        put(np.ones(n, bool), 1 - v_ - w_, v_, w_)
    return w[:, :1] * a + w[:, 1:2] * b + w[:, 2:] * c, w


if __name__ == "__main__":
    import sys
    import time
    dx, model, kind = float(sys.argv[1]), sys.argv[2], sys.argv[3]
    deep = len(sys.argv) > 4 and sys.argv[4] == "deep"
    t0 = time.time()
    F = static_forces(dx, model, kind, deep)
    print(f"dx {dx * 1e3:.3f} mm {model} {kind}:", " ".join(f"{f:.5f}" for f in F), f"({time.time() - t0:.0f} s)")
