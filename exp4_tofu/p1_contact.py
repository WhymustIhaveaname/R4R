"""Experiment 4b's spatial discretization alone: the static indentation of the P1 slab.

The same linear tetrahedra as MuJoCo's flex and Genesis's FEM, on MuJoCo's grid, bottom face
fixed, pressed by a rigid frictionless sphere, with no time step and no contact compliance:
the analogue for the force of what p1.py is for the frequency.  The contact is linearized as in
axisym.py, the continuum's reference: a vertex of the top face may not rise above the sphere,
u_z >= delta - gap, and the sphere pushes only downward.

The vertices that can touch are few, so the problem is reduced to them: their compliance C
(K^-1 restricted to their vertical displacements, from one factorization of K) and a small
nonnegative least-squares problem for the contact forces p >= 0,

  minimize 1/2 p.C.p - b.p,  b_i = delta - gap_i,

whose optimality conditions are the Signorini ones: C p >= b, p >= 0, p.(C p - b) = 0.

That is contact at the vertices only, as Genesis's legacy coupler has it.  MuJoCo collides the
sphere with the elements instead, one contact per element at its deepest point, so the whole
piecewise-linear surface is kept out of the sphere, not only its vertices.  kind="surface" is
that constraint: it holds at a lattice of points on every top triangle, each a barycentric
combination of the triangle's vertices.  There are more points than vertices, so it is solved
for the vertices' deflections w instead, with a stiff penalty, by Newton on the active set:

  minimize 1/2 w.C^-1.w + kappa/2 sum_i max(0, b_i - (W w)_i)^2,

kappa a million times the slab's stiffness, so the points sink into the sphere by 1e-6 of the
depth at most.
"""

import time

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import nnls
from scipy.sparse.linalg import splu

import common as c
import indentation as ind
import p1


def slab_mesh(dx):
    from mujoco_sim import indent_xml
    return p1.mesh_from_mujoco(dx, indent_xml(dx, "discrete", 1e-3, 0.01, (0.02, 1.0),
                                              (0.9, 0.95, 0.001, 0.5, 2)))


def top_triangles(X, tets, top):
    """The faces of the tetrahedra that lie in the top face, once each."""
    faces = set()
    for t in tets:
        on = [v for v in t if top[v]]
        if len(on) == 3:
            faces.add(tuple(sorted(on)))
    return np.array(sorted(faces))


def lattice(n):
    """Barycentric coordinates of a triangle's points (i, j, k) / n, i + j + k = n."""
    return np.array([(i, j, n - i - j) for i in range(n + 1) for j in range(n + 1 - i)], float) / n


class P1Indent:
    def __init__(self, dx, radius, depths=ind.DEPTHS, E=c.E0, kind="vertex", points_per_edge=4):
        t0 = time.time()
        X, tets = slab_mesh(dx)
        K, _ = p1.assemble(X, tets, E, c.NU)
        z = X[:, 2]
        bottom = np.isclose(z, z.min())
        top = np.isclose(z, z.max())
        r = np.hypot(X[:, 0], X[:, 1])
        reach = 1.6 * np.sqrt(radius * depths.max()) + 2 * dx   # beyond the widest contact
        if kind == "vertex":
            cand = np.flatnonzero(top & (r < min(reach, 0.999 * radius)))
        else:   # every corner of a triangle the sphere can reach
            cand = np.flatnonzero(top & (r < min(reach, radius) + 1.5 * dx))
        free = np.flatnonzero(np.repeat(~bottom, 3))
        pos = -np.ones(3 * len(X), int)
        pos[free] = np.arange(len(free))
        lu = splu(K[free][:, free].tocsc())
        rhs = np.zeros((len(free), len(cand)))
        rhs[pos[3 * cand + 2], np.arange(len(cand))] = -1.0      # a unit downward force
        sol = lu.solve(rhs)
        self.C = -sol[pos[3 * cand + 2]]                          # downward displacement per unit force
        self.C = (self.C + self.C.T) / 2
        self.gap = radius - np.sqrt(np.maximum(radius**2 - r[cand] ** 2, 0))
        self.cand, self.X, self.radius, self.depths, self.kind = cand, X, radius, depths, kind
        if kind == "surface":
            local = -np.ones(len(X), int)
            local[cand] = np.arange(len(cand))
            tris = top_triangles(X, tets, top)
            tris = tris[(local[tris] >= 0).all(1)]
            bary = lattice(points_per_edge)
            pts = np.einsum("pk,tkd->tpd", bary, X[tris]).reshape(-1, 3)
            rows = np.repeat(np.arange(len(pts)), 3)
            cols = local[np.repeat(tris, len(bary), 0)].ravel()
            from scipy.sparse import csr_matrix
            self.W = csr_matrix((np.tile(bary, (len(tris), 1)).ravel(), (rows, cols)),
                                shape=(len(pts), len(cand)))
            rp = np.hypot(pts[:, 0], pts[:, 1])
            keep = rp < 0.999 * radius
            self.W, rp = self.W[keep], rp[keep]
            self.gap_pts = radius - np.sqrt(radius**2 - rp**2)
            self.Kc = np.linalg.inv(self.C)
            self.Kc = (self.Kc + self.Kc.T) / 2
        self.seconds = time.time() - t0

    def solve(self, delta):
        """(total force, contact forces) at depth delta."""
        if self.kind == "surface":
            return self.solve_surface(delta)
        L = cholesky(self.C, lower=True)
        b = delta - self.gap
        p, _ = nnls(L.T, solve_triangular(L, b, lower=True))
        w = self.C @ p
        assert np.all(w >= b - 1e-9 * delta), "a vertex is above the sphere"
        return p.sum(), p

    def solve_surface(self, delta):
        """(total force, vertex forces) with the whole top surface kept out of the sphere."""
        kappa = 1e6 * np.abs(np.diag(self.Kc)).max()
        b = delta - self.gap_pts
        W, K = self.W, self.Kc

        def energy(w):
            g = np.maximum(b - W @ w, 0)
            return 0.5 * w @ K @ w + 0.5 * kappa * g @ g

        w = np.zeros(K.shape[0])
        for _ in range(200):                  # Newton with backtracking: the energy is convex
            gap = np.maximum(b - W @ w, 0)
            grad = K @ w - kappa * (W.T @ gap)
            WA = W[gap > 0]
            step = -np.linalg.solve(K + kappa * (WA.T @ WA).toarray(), grad)
            e0, t = energy(w), 1.0
            while energy(w + t * step) > e0 + 1e-4 * t * (grad @ step) and t > 1e-12:
                t /= 2
            w = w + t * step
            if np.abs(t * step).max() < 1e-12 * delta:
                break
        else:
            raise RuntimeError("Newton did not converge")
        f = self.Kc @ w                                   # downward vertex forces
        assert (b - W @ w).max() < 1e-5 * delta, "the surface sinks into the sphere"
        return f.sum(), f

    def forces(self):
        return np.array([self.solve(d)[0] for d in self.depths])


def p1_forces(dx, radius, kind="vertex"):
    """The P1 slab's force (N) at ind.DEPTHS, cached; at 2 mm cells it takes 15 minutes and 10 GB.

    The surface constraint is sampled at 16 points per triangle edge, which leaves the force
    within about 2 % of its limit at 5 mm cells and finer."""
    path = c.CACHE / f"p1_indent_{kind}_{radius * 1e3:.0f}mm_{dx:.6f}.npy"
    if path.exists():
        return np.load(path)
    F = P1Indent(dx, radius, kind=kind, points_per_edge=16).forces()
    c.CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, F)
    return F


if __name__ == "__main__":
    # small strain: the linear slab against the linear continuum, at Brookfield's sphere
    ref = ind.reference("linear")
    for kind in ("vertex", "surface"):
        for dx in (0.01, 0.005, 0.01 / 3, 0.0025, 0.002):
            F = p1_forces(dx, ind.RADIUS, kind)
            print(f"{kind:7s}  dx {dx * 1e3:5.2f} mm   F/ref " + " ".join(f"{x:.3f}" for x in F / ref), flush=True)
