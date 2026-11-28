"""The exact modes of a free isotropic elastic box, by the Ritz method.

The displacement is expanded in products of Legendre polynomials P_l(x/a) P_m(y/b) P_n(z/c)
with l + m + n <= N (Visscher et al., J. Acoust. Soc. Am. 90, 2154 (1991), with Legendre
instead of monomials for conditioning).  Every surface is traction free, which the Ritz method
imposes naturally.  The frequencies converge exponentially in N, so this is the continuum
reference that the simulators' meshes approach.
"""

import numpy as np
from numpy.polynomial import legendre
from scipy.linalg import eigh


def legendre_tables(n_max):
    """S[a][b][p, q] = integral over [-1, 1] of d^a P_p * d^b P_q."""
    xs, ws = legendre.leggauss(n_max + 2)
    vals = np.zeros((2, n_max + 1, len(xs)))
    for p in range(n_max + 1):
        c = np.zeros(p + 1)
        c[p] = 1
        vals[0, p] = legendre.legval(xs, c)
        vals[1, p] = legendre.legval(xs, legendre.legder(c))
    return np.einsum("apx,bqx,x->abpq", vals, vals, ws)


class FreeBox:
    """A box of half sides (a, b, c), centered at the origin, every face free."""

    def __init__(self, half_sides, E, nu, rho, order=14):
        self.half = np.asarray(half_sides, float)
        self.rho = rho
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        self.exps = np.array([(l, m, n) for l in range(order + 1) for m in range(order + 1 - l)
                              for n in range(order + 1 - l - m)])
        S = legendre_tables(order)
        P = len(self.exps)
        vol_jac = np.prod(self.half)
        e = self.exps

        def integral(dj, dl):
            """int d_j psi_p d_l psi_q dV, dj and dl being axis indices or None."""
            out = np.full((P, P), vol_jac)
            for d in range(3):
                a = int(d == dj)
                b = int(d == dl)
                out = out * S[a, b][e[:, d][:, None], e[:, d][None, :]]
            scale = (1 / self.half[dj] if dj is not None else 1) * (1 / self.half[dl] if dl is not None else 1)
            return out * scale

        mass = rho * integral(None, None)
        G = [[integral(j, l) for l in range(3)] for j in range(3)]
        K = np.zeros((3, P, 3, P))
        for i in range(3):
            for k in range(3):
                for j in range(3):
                    for l in range(3):
                        C = lam * (i == j) * (k == l) + mu * ((i == k) * (j == l) + (i == l) * (j == k))
                        if C:
                            K[i, :, k, :] += C * G[j][l]
        K = K.reshape(3 * P, 3 * P)
        M = np.kron(np.eye(3), mass)
        w2, vecs = eigh(K, M)
        self.omega = np.sqrt(np.clip(w2, 0, None))
        self.modes = vecs  # M-orthonormal: vecs.T M vecs = I

    def basis(self, points):
        """psi_p at points (n, 3) -> (n, P)."""
        xi = np.asarray(points, float) / self.half
        n_max = self.exps.max()
        out = np.ones((len(xi), len(self.exps)))
        for d in range(3):
            table = np.stack([legendre.legval(xi[:, d], np.eye(n_max + 1)[p]) for p in range(n_max + 1)], 1)
            out = out * table[:, self.exps[:, d]]
        return out

    def displacement(self, points, k):
        """Mode k's displacement at points: (n, 3)."""
        B = self.basis(points)
        P = len(self.exps)
        coeff = self.modes[:, k].reshape(3, P)
        return B @ coeff.T


if __name__ == "__main__":
    # convergence check: a 6 x 5 x 4 cm block of a soft gel
    for N in (8, 10, 12, 14, 16):
        box = FreeBox((0.03, 0.025, 0.02), E=1e4, nu=0.45, rho=1000.0, order=N)
        f = box.omega[6:12] / (2 * np.pi)
        print(N, " ".join(f"{x:.6f}" for x in f), "| rigid", f"{box.omega[:6].max() / 2 / np.pi:.1e}")
