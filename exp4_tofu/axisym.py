"""The reference for indentation: a rigid frictionless sphere pressed into an elastic cylinder.

Linear elasticity, axisymmetric, on a tensor grid that is fine under the contact and coarse
far from it, each cell cut into two linear triangles.  The bottom face is bonded.  The contact
is Signorini's: a top node either touches the sphere, with a compressive reaction, or stays
below it.  An active-set iteration finds which, and the force is the sum of the reactions.
This converges far below the engines' errors as the grid is refined, and it does not rely on
Hertz's half-space: the layer's finite thickness is in it.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def graded(length, fine, fine_until, growth=1.08):
    """Nodes from 0: spacing `fine` up to `fine_until`, then growing by `growth` to `length`."""
    x = list(np.arange(0.0, fine_until + 1e-12, fine))
    s = fine
    while x[-1] < length - 1e-12:
        s *= growth
        x.append(min(x[-1] + s, length))
    return np.array(x)


class Layer:
    def __init__(self, radius, height, E, nu, fine, fine_until, growth=1.08):
        r = graded(radius, fine, fine_until, growth)
        d = graded(height, fine, fine_until, growth)      # depth below the top face
        z = height - d[::-1]
        self.r, self.z, self.H = r, z, height
        R, Z = np.meshgrid(r, z, indexing="ij")
        self.X = np.stack([R.ravel(), Z.ravel()], 1)
        nr, nz = len(r), len(z)
        idx = np.arange(nr * nz).reshape(nr, nz)
        tris = []
        for i in range(nr - 1):
            for j in range(nz - 1):
                a, b, c, e = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
                tris += [(a, b, c), (a, c, e)] if (i + j) % 2 == 0 else [(a, b, e), (b, c, e)]
        self.tris = np.array(tris)
        self.K = self._stiffness(E, nu)
        self.top = idx[:, -1]
        self.bottom = idx[:, 0]
        self.axis = idx[0, :]

    def _stiffness(self, E, nu):
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        D = np.zeros((4, 4))           # (e_r, e_z, e_theta, g_rz)
        D[:3, :3] = lam
        D[[0, 1, 2], [0, 1, 2]] += 2 * mu
        D[3, 3] = mu
        P = self.X[self.tris]          # (ne, 3, 2)
        r_, z_ = P[..., 0], P[..., 1]
        b = np.stack([z_[:, 1] - z_[:, 2], z_[:, 2] - z_[:, 0], z_[:, 0] - z_[:, 1]], 1)
        c = np.stack([r_[:, 2] - r_[:, 1], r_[:, 0] - r_[:, 2], r_[:, 1] - r_[:, 0]], 1)
        A2 = r_[:, 0] * b[:, 0] + r_[:, 1] * b[:, 1] + r_[:, 2] * b[:, 2]   # twice the area
        rc = r_.mean(1)
        ne = len(self.tris)
        B = np.zeros((ne, 4, 6))
        for i in range(3):
            B[:, 0, 2 * i] = b[:, i] / A2
            B[:, 1, 2 * i + 1] = c[:, i] / A2
            B[:, 2, 2 * i] = 1 / (3 * rc)
            B[:, 3, 2 * i] = c[:, i] / A2
            B[:, 3, 2 * i + 1] = b[:, i] / A2
        w = 2 * np.pi * rc * np.abs(A2) / 2
        Ke = np.einsum("eki,kl,elj,e->eij", B, D, B, w)
        dofs = (2 * self.tris[:, :, None] + np.arange(2)).reshape(ne, 6)
        n = 2 * len(self.X)
        return coo_matrix((Ke.ravel(), (np.repeat(dofs, 6, 1).ravel(), np.tile(dofs, (1, 6)).ravel())),
                          shape=(n, n)).tocsr()

    def indent(self, R, delta, max_iter=60):
        """Force (N) and contact radius (m) of a sphere of radius R pressed delta into the top."""
        n = 2 * len(self.X)
        rt = self.X[self.top, 0]
        gap = -delta + R - np.sqrt(np.maximum(R * R - rt * rt, 0))   # allowed u_z at each top node
        gap[rt >= R] = np.inf
        fixed = set(2 * self.bottom) | set(2 * self.bottom + 1) | set(2 * self.axis)
        active = np.zeros(len(self.top), bool)
        active[rt < np.sqrt(R * delta)] = True          # Hertz's contact radius, to start
        for _ in range(max_iter):
            pres = {int(2 * self.top[k] + 1): gap[k] for k in np.flatnonzero(active)}
            known = np.array(sorted(fixed | set(pres)))
            values = np.array([pres.get(int(i), 0.0) for i in known])
            free = np.setdiff1d(np.arange(n), known)
            u = np.zeros(n)
            u[known] = values
            Kff = self.K[free][:, free].tocsc()
            u[free] = splu(Kff).solve(-self.K[free][:, known] @ values)
            react = self.K @ u                           # force the constraints exert
            rz = react[2 * self.top + 1]
            uz = u[2 * self.top + 1]
            release = active & (rz > 0)                  # a contact that pulls the surface up
            add = (~active) & np.isfinite(gap) & (uz > gap + 1e-12 * delta)   # a free node above the sphere
            if not release.any() and not add.any():
                break
            active = (active & ~release) | add
        else:
            raise RuntimeError("the contact set did not settle")
        force = -rz[active].sum()
        a = rt[active].max() if active.any() else 0.0
        return float(force), float(a)


def hertz(E, nu, R, delta):
    return 4 / 3 * E / (1 - nu * nu) * np.sqrt(R) * delta ** 1.5


def dimitriadis_bonded(E, nu, R, delta, H):
    """Hertz with the correction for a layer bonded to a rigid base (Dimitriadis et al. 2002)."""
    chi = np.sqrt(R * delta) / H
    a0 = -(1.2876 - 1.4678 * nu + 1.3442 * nu**2) / (1 - nu)
    b0 = (0.6387 - 1.0277 * nu + 1.5164 * nu**2) / (1 - nu)
    p = np.pi
    s = (1 - 2 * a0 / p * chi + 4 * a0**2 / p**2 * chi**2
         - 8 / p**3 * (a0**3 + 4 * p**2 / 15 * b0) * chi**3
         + 16 * a0 / p**4 * (a0**3 + 3 * p**2 / 5 * b0) * chi**4)
    return hertz(E, nu, R, delta) * s


if __name__ == "__main__":
    E, R = 1.13e4, 0.01
    for nu in (0.45, 0.3):
        for H, Rc in ((0.20, 0.20), (0.04, 0.027)):
            for fine in (2e-4, 1e-4, 5e-5):
                lay = Layer(Rc, H, E, nu, fine, 0.008)
                for delta in (0.5e-3, 1e-3):
                    F, a = lay.indent(R, delta)
                    print(f"nu {nu} H {H*100:.0f} cm  fine {fine*1e3:.2f} mm  nodes {len(lay.X)}  delta {delta*1e3:.1f} mm:"
                          f"  F {F*1e3:.4f} mN  / Hertz {F / hertz(E, nu, R, delta):.4f}"
                          f"  / Dimitriadis {F / dimitriadis_bonded(E, nu, R, delta, H):.4f}"
                          f"  a {a*1e3:.2f} mm (Hertz {np.sqrt(R*delta)*1e3:.2f})", flush=True)
