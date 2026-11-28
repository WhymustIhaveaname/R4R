"""The reference for deep indentation: axisym.py's problem with the exact kinematics.

A rigid frictionless sphere pressed into an elastic cylinder whose bottom face is bonded, as in
axisym.py, but the strain is the full one and the sphere touches the deformed surface.  At the
7 mm of Brookfield's test under a 12.7 mm sphere both matter by about 10 %, more than the
discretization errors to be measured, so each engine is measured against the continuum limit of
its own material:

  "stvk"        Saint Venant-Kirchhoff, which MuJoCo's flex is (its P1 force matches to 1e-14)
  "corotated"   Genesis's linear corotated
  "neohookean"  Genesis's stable neo-Hookean (explicit only: its implicit solver has no Hessian)
  "linear"      Genesis's linear, small-strain elasticity in the deformation gradient

The cylinder is meshed as in axisym.py.  Each triangle's in-plane deformation gradient is
constant, and its hoop stretch is its centroid's r over the centroid's reference r.  Newton's
method with a line search minimizes the energy, with each triangle's Hessian from torch's
automatic differentiation.  The contact is an augmented Lagrangian on the top nodes: a node's
distance from the sphere's centre may not be less than the radius.
"""

import numpy as np
import torch
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu

from axisym import graded

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)   # the tensors are small; more threads only contend
MODELS = ("stvk", "corotated", "neohookean", "linear")
JMIN = 0.1


def density(model, mu, lam):
    """The energy density of a batch: in-plane deformation gradients (n, 2, 2), hoop stretches (n,)."""
    def strain_energy(e2, et):                    # mu |e|^2 + lam/2 (tr e)^2, e = diag(e2, et)
        tr = e2[:, 0, 0] + e2[:, 1, 1] + et
        return mu * ((e2 * e2).sum((1, 2)) + et * et) + 0.5 * lam * tr * tr

    def sym(A):
        return 0.5 * (A + A.transpose(1, 2))

    def psi(F2, ft):
        eye = torch.eye(2)
        if model == "stvk":
            return strain_energy(0.5 * (F2.transpose(1, 2) @ F2 - eye), 0.5 * (ft * ft - 1))
        if model == "linear":
            return strain_energy(sym(F2) - eye, ft - 1)
        if model == "corotated":                  # the polar rotation of a 2 x 2 block
            th = torch.atan2(F2[:, 1, 0] - F2[:, 0, 1], F2[:, 0, 0] + F2[:, 1, 1])
            c, s = torch.cos(th), torch.sin(th)
            Rt = torch.stack([torch.stack([c, s], 1), torch.stack([-s, c], 1)], 1)
            return strain_energy(sym(Rt @ F2) - eye, ft - 1)
        if model == "neohookean":
            lam_s = lam + mu
            alpha = 1 + mu / lam_s
            J = (F2[:, 0, 0] * F2[:, 1, 1] - F2[:, 0, 1] * F2[:, 1, 0]) * ft
            return 0.5 * (mu * ((F2 * F2).sum((1, 2)) + ft * ft - 3) + lam_s * (J - alpha) ** 2)
        raise ValueError(model)
    return psi


class LayerNL:
    def __init__(self, radius, height, E, nu, model, fine, fine_until, growth=1.08, bottom="bonded"):
        r = graded(radius, fine, fine_until, growth)
        d = graded(height, fine, fine_until, growth)
        z = height - d[::-1]
        self.H, self.model = height, model
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
        self.top = idx[:, -1]
        n = 2 * len(self.X)
        # the bottom face bonded, or free to slide on a frictionless table ("slip")
        held = [2 * idx[:, 0] + 1] + ([2 * idx[:, 0]] if bottom == "bonded" else [])
        fixed = np.unique(np.concatenate(held + [2 * idx[0, :]]))
        self.free = np.setdiff1d(np.arange(n), fixed)
        self.dofs = (2 * self.tris[:, :, None] + np.arange(2)).reshape(-1, 6)

        P = self.X[self.tris]
        Dm = np.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]], 2)
        area = 0.5 * np.abs(np.linalg.det(Dm))
        rc = P[:, :, 0].mean(1)
        self.Bm = torch.tensor(np.linalg.inv(Dm))
        self.rc = torch.tensor(rc)
        self.w = torch.tensor(2 * np.pi * rc * area)
        mu = E / (2 * (1 + nu))
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        psi = density(model, mu, lam)

        def energies(x):                          # x: (n, 6), each triangle's nodes
            x = x.reshape(-1, 3, 2)
            F2 = torch.stack([x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]], 2) @ self.Bm
            return self.w * psi(F2, x[:, :, 0].mean(1) / self.rc)
        self._energies = energies

        rt = r
        mid = np.concatenate([[0], (rt[1:] + rt[:-1]) / 2, [rt[-1]]])
        self.top_area = np.pi * (mid[1:] ** 2 - mid[:-1] ** 2)     # each top node's ring
        self.kappa = 1e2 * E / fine * self.top_area                 # penalty per node, N/m
        self.u = np.zeros(n)
        self.mult = np.zeros(len(self.top))
        rows = np.repeat(self.dofs, 6, 1).ravel()
        cols = np.tile(self.dofs, (1, 6)).ravel()
        self._rc_index = (rows, cols)

    def _elastic(self, u, order):
        """The elastic energy, or (its gradient, Hessian); a triangle's Hessian is six forward
        derivatives of its gradient, taken for all triangles at once."""
        x = torch.tensor((self.X.ravel() + u)[self.dofs])
        if order == 0:
            return float(self._energies(x).sum())
        n = 2 * len(self.X)

        def gradient(x):
            return torch.func.grad(lambda y: self._energies(y).sum())(x)
        ge = gradient(x)
        g = np.zeros(n)
        np.add.at(g, self.dofs, ge.numpy())
        if order == 1:
            return g
        He = np.empty((len(self.tris), 6, 6))
        for k in range(6):
            e = torch.zeros_like(x)
            e[:, k] = 1
            He[:, :, k] = torch.func.jvp(gradient, (x,), (e,))[1].numpy()
        K = coo_matrix((He.ravel(), self._rc_index), shape=(n, n)).tocsr()
        return g, K

    def _contact(self, u, centre, R, order):
        """The augmented Lagrangian of the contact: energy, or (energy, gradient, Hessian blocks)."""
        x = self.X[self.top] + u.reshape(-1, 2)[self.top]
        v = x - centre
        dist = np.linalg.norm(v, axis=1)
        phi = dist - R
        p = np.maximum(self.mult - self.kappa * phi, 0)            # the contact pressure force
        energy = ((p**2 - self.mult**2) / (2 * self.kappa)).sum()
        if order == 0:
            return energy
        nrm = v / dist[:, None]
        g = -p[:, None] * nrm
        on = p > 0
        Hb = (self.kappa * on)[:, None, None] * nrm[:, :, None] * nrm[:, None, :] \
            - (p / dist)[:, None, None] * (np.eye(2) - nrm[:, :, None] * nrm[:, None, :])
        return energy, g, Hb, p, nrm, phi

    def min_jacobian(self, u):
        """The smallest volume ratio J = det F over the triangles."""
        x = (self.X.ravel() + u)[self.dofs].reshape(-1, 3, 2)
        F2 = np.stack([x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]], 2) @ self.Bm.numpy()
        J = np.linalg.det(F2) * x[:, :, 0].mean(1) / self.rc.numpy()
        return J.min()

    def _total(self, u, centre, R):
        """The energy; infinite where a triangle is crushed to a tenth of its volume, so that the
        solution stays on the branch that starts undeformed: Saint Venant-Kirchhoff has inverted
        states of lower energy, and neo-Hookean and corotated no barrier against them.  The
        deepest indentation here leaves every triangle above three quarters of its volume."""
        if self.min_jacobian(u) < JMIN:
            return np.inf
        return self._elastic(u, 0) + self._contact(u, centre, R, 0)

    def _newton(self, centre, R, tol):
        """Newton with a line search, and a Levenberg shift where the Hessian is not positive:
        Saint Venant-Kirchhoff softens in compression past a stretch of 1/sqrt(3), which a
        deep indentation nears under the sphere."""
        n = 2 * len(self.X)
        tdofs = (2 * self.top[:, None] + np.arange(2))
        shift = 0.0
        for it in range(300):
            g, K = self._elastic(self.u, 2)
            _, gc, Hb, *_ = self._contact(self.u, centre, R, 1)
            np.add.at(g, tdofs, gc)
            Hc = coo_matrix((Hb.ravel(), (np.repeat(tdofs, 2, 1).ravel(), np.tile(tdofs, (1, 2)).ravel())),
                            shape=(n, n)).tocsr()
            A = (K + Hc)[self.free][:, self.free].tocsc()
            gf = g[self.free]
            if np.abs(gf).max() < tol:
                return it
            D = diags(np.abs(A.diagonal()))
            while True:                       # a descent direction, shifting as needed
                step = np.zeros(n)
                step[self.free] = -splu((A + shift * D).tocsc()).solve(gf)
                slope = gf @ step[self.free]
                if slope < 0:
                    break
                shift = max(1e-8, 10 * shift)
            e0, t = self._total(self.u, centre, R), 1.0
            near = np.abs(gf).max() < 1e3 * tol
            while True:          # Armijo; near the solution, a change within the energy's rounding
                e1 = self._total(self.u + t * step, centre, R)
                if e1 <= e0 + 1e-4 * t * slope or (near and e1 - e0 <= 1e-13 * abs(e0)):
                    break
                t /= 2
                if t < 1e-12:
                    raise RuntimeError("the line search failed")
            self.u += t * step
            # the shift is for a Hessian that is not positive; a short step only means the
            # contact set is changing, which the line search handles
            shift = shift / 10 if shift > 1e-8 else 0.0
            if near and np.abs(t * step).max() < 1e-15:     # m: the displacement no longer changes
                return it
        raise RuntimeError("Newton did not converge")

    def indent(self, R, delta, substeps=4):
        """Force (N) on a sphere of radius R pressed delta into the top, continuing from the last call."""
        centre_of = lambda d: np.array([0.0, self.H + R - d])
        d0 = getattr(self, "_delta", 0.0)
        for d in np.linspace(d0, delta, substeps + 1)[1:]:
            centre = centre_of(d)
            if getattr(self, "_last", None) is not None:      # extrapolate from the last two depths
                (d1, u1), (d2, u2) = self._last
                guess = self.u + (u2 - u1) * (d - d2) / (d2 - d1)
                if self.min_jacobian(guess) >= JMIN:
                    self.u = guess
            for _ in range(50):                                     # the multipliers' updates
                self._newton(centre, R, tol=1e-8 * max(1.0, abs(self.mult).sum()))   # N; the contact set flips below
                *_, p, nrm, phi = self._contact(self.u, centre, R, 1)
                self.mult = p
                if -phi.min() < 1e-7 * d:
                    break
            else:
                raise RuntimeError("the contact did not settle")
            prev = getattr(self, "_last", None)
            self._last = ((prev[1] if prev else (d0, np.zeros_like(self.u))), (d, self.u.copy()))
        self._delta = delta
        return float((p * -nrm[:, 1]).sum())


if __name__ == "__main__":
    import time
    import axisym
    E, nu, R = 1.13e4, 0.45, 0.0127
    H, Rc = 0.04, np.sqrt(0.1 * 0.1 / np.pi)
    depths = np.array([0.5, 1, 2, 3.5, 5, 7]) * 1e-3

    def curve(model, fine, E=E, nu=nu, Rc=Rc, H=H, depths=depths):
        t0 = time.time()
        lay = LayerNL(Rc, H, E, nu, model, fine, 0.015)
        F = np.array([lay.indent(R, d) for d in depths])
        return F, time.time() - t0

    lin = axisym.Layer(Rc, H, E, nu, 1e-4, 0.015)
    print("linear, linearized contact (axisym.py):", " ".join(f"{lin.indent(R, d)[0]:.5f}" for d in depths[:2]), flush=True)
    for fine in (4e-4, 2e-4, 1e-4):
        F, t = curve("stvk", fine)
        print(f"stvk  fine {fine * 1e3:.1f} mm:", " ".join(f"{f:.5f}" for f in F), f"({t:.0f} s)", flush=True)
    for model in ("corotated", "neohookean", "linear"):
        F, t = curve(model, 2e-4)
        print(f"{model:10s} fine 0.2 mm:", " ".join(f"{f:.5f}" for f in F), f"({t:.0f} s)", flush=True)
    # A neo-Hookean half-space, nearly incompressible, against Mu et al. 2026, Eq. 16
    nu_h = 0.49
    dr = np.array([0.1, 0.25, 0.5, 0.75, 1.0])
    F, t = curve("neohookean", 2e-4, nu=nu_h, Rc=30 * R, H=30 * R, depths=dr * R)
    FH = 4 / 3 * E / (1 - nu_h**2) * np.sqrt(R) * (dr * R) ** 1.5
    print("half-space, neo-Hookean, nu 0.49: delta/R", dr, flush=True)
    print("  F/F_H        ", " ".join(f"{x:.4f}" for x in F / FH), f"({t:.0f} s)")
    print("  Eq. 16       ", " ".join(f"{x:.4f}" for x in 1 - 11 / 60 * dr - dr**2 / 350))
