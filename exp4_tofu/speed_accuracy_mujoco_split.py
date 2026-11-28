"""Two residuals per speed-accuracy point: the frequencies' and the damping's, fitted apart.

Targets, all from the top-front pluck: sway 8.010 Hz, zeta 0.0151; twist 9.992 Hz, zeta 0.0154.  The parameters start renormalized (E* = E S T(h), beta* from the
time theory at zeta 0.0151), and two multipliers are fitted, each to its own pair of numbers:

  frequency   lam on E*, to (f_sway, f_twist): err_freq_pct = 100 RMS of their relative errors
  damping     mu on beta*, at that lam, to (zeta_sway, zeta_twist): err_damp_pct likewise

Each mode's (f, zeta) depends a little on both multipliers through the step, so the two fits
are solved together: lam is the best for the frequencies at the final mu, and mu the best for
the damping at the final lam.  The start is that pair for the exact single-mode model of the
step: each run is one mode of the model, whose step is a 2 x 2 linear map in (x, v) for backward
Euler or symplectic Euler with beta K damping, so its (f, zeta) follow from that map's
eigenvalue (step_mode).  Newton steps on actual runs, with a forward-difference Jacobian from
actual runs, then move it until the step is below 1e-6.  The errors stored are those of the
runs at the final (lam, mu).

Kept from before: speed, cut and hardware (the cluster's timing); err_rms4_pct and its "rms4"
entry where a JSON has them (an earlier, superseded fit of both multipliers to all four numbers);
err_max_pct (the first definition: max of the sway's f and zeta errors and the twist's f error
against the continuum's twist, from the cluster's runs, read from --orig or kept if present).
err_pct = err_freq_pct.

  python speed_accuracy_mujoco_split.py --jobs 24 [--orig DIR]   # DIR: the cluster's first JSONs
"""

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import minimize_scalar

import jello_mujoco_variants as jv
import jello_sim as js
import speed_accuracy_mujoco as sa

TARGET = {"f_sway": 8.010, "zeta_sway": 0.0151, "f_twist": 9.992, "zeta_twist": 0.0154}
TGT = np.array(list(TARGET.values()))
TOL = 1e-6
MAX_IT = 5
FD_STEP = 1e-3
ORIG = None


def step_mode(w, beta, h, integrator):
    """(f Hz, zeta) of one mode of natural frequency w (rad/s) under MuJoCo's step."""
    if integrator == "discrete":                 # backward Euler, beta K implicit
        D = 1 + h * beta * w**2 + h**2 * w**2
        A = np.array([[1 - h**2 * w**2 / D, h / D], [-h * w**2 / D, 1 / D]])
    else:                                        # symplectic Euler, beta K explicit
        A = np.array([[1 - h**2 * w**2, h * (1 - h * beta * w**2)], [-h * w**2, 1 - h * beta * w**2]])
    z = np.linalg.eigvals(A)
    z = z[np.argmax(z.imag)]
    s = np.log(z) / h
    return abs(s) / (2 * np.pi), -s.real / abs(s)


class Point:
    """One (method, n, h): its model at the renormalized parameters, its sway and twist."""

    def __init__(self, row):
        self.row = row
        dof, self.integ, _ = sa.METHODS[row["method"]]
        n, h = row["cells"], row["h_s"]
        self.h = h
        C, _ = sa.continuum()
        self.E = js.RHO * (TARGET["f_sway"] / C) ** 2
        f_s1, f_t1, _, self.sway, self.twist, _ = sa.modal(dof, n)
        self.w1 = 2 * np.pi * np.array([f_s1, f_t1])        # at E / rho = 1
        S = (C / f_s1) ** 2
        w = 2 * np.pi * TARGET["f_sway"]
        if self.integ == "discrete":
            T, beta = js.backward_euler(w, TARGET["zeta_sway"], h)
        else:
            T, beta = jv.symplectic_euler(w, TARGET["zeta_sway"], h)
        self.S, self.T, self.E_star, self.beta_star = S, T, self.E * S * T, beta
        self.m = mujoco.MjModel.from_xml_string(sa.xml(dof, n, self.integ, self.E_star, beta, h))
        self.k0 = self.m.flex_stiffness.copy()
        self.runs = 0

    def measure(self, lam, mu):
        """(f_sway, zeta_sway, f_twist, zeta_twist) of MuJoCo's runs at lam E*, mu beta*."""
        self.m.flex_stiffness[:] = self.k0 * lam
        self.m.flex_damping[0] = mu * self.beta_star
        out = []
        for vec, w0 in ((self.sway, 2 * np.pi * TARGET["f_sway"]),
                        (self.twist, self.w1[1] * np.sqrt(lam * self.E_star / js.RHO))):
            t, q, _, _, _ = sa.run(self.m, vec, self.h, sa.DURATION, timed=False)
            wn, z = sa.measure(t, q, w0)
            out += [wn / (2 * np.pi), z]
            self.runs += 1
        return np.array(out)

    def predict(self, lam, mu):
        """The same four numbers for the exact single-mode model of the step."""
        out = []
        for w1 in self.w1:
            out += list(step_mode(w1 * np.sqrt(lam * self.E_star / js.RHO), mu * self.beta_star, self.h, self.integ))
        return np.array(out)

    def jacobian(self, x, r):
        """d(relative errors)/d(lam, mu) from actual runs, forward differences."""
        J = np.empty((4, 2))
        for k in range(2):
            dx = np.zeros(2)
            dx[k] = FD_STEP * x[k]
            J[:, k] = (self.measure(*(x + dx)) / TGT - 1 - r) / dx[k]
        return J


def block_step(J, r):
    """The Newton step to (lam best for the frequencies given mu, mu best for the damping given
    lam), linearized: rows 0, 2 are the frequencies, 1, 3 the damping ratios."""
    F, Z = [0, 2], [1, 3]
    a = J[F, 0]
    b = J[Z, 1]
    A = np.array([[a @ J[F, 0], a @ J[F, 1]], [b @ J[Z, 0], b @ J[Z, 1]]])
    rhs = -np.array([a @ r[F], b @ r[Z]])
    return np.linalg.solve(A, rhs), np.array([a @ r[F], b @ r[Z]])


class SplitPoint(Point):
    def start_split(self):
        """The pair for the single-mode model, by alternating the two one-dimensional fits."""
        lam, mu = 1.0, 1.0
        for _ in range(100):
            lam1 = np.exp(minimize_scalar(lambda u: np.sum((self.predict(np.exp(u), mu)[[0, 2]] / TGT[[0, 2]] - 1) ** 2),
                                          bracket=(-0.5, 0.5), tol=1e-14).x)
            mu1 = minimize_scalar(lambda v: np.sum((self.predict(lam1, v)[[1, 3]] / TGT[[1, 3]] - 1) ** 2),
                                  bracket=(0.0, 2.0), tol=1e-14).x
            done = abs(lam1 / lam - 1) < 1e-13 and abs(mu1 / mu - 1) < 1e-13
            lam, mu = lam1, mu1
            if done:
                break
        return np.array([lam, mu])

    def fit_split(self):
        x = self.start_split()
        r = self.measure(*x) / TGT - 1
        history = [{"x": x.tolist(), "r": r.tolist()}]
        J = self.jacobian(x, r)
        for _ in range(MAX_IT):
            step, g = block_step(J, r)
            if np.max(np.abs(step / x)) < TOL:
                break
            x = x + step
            r = self.measure(*x) / TGT - 1
            history.append({"x": x.tolist(), "r": r.tolist(), "step_rel": float(np.max(np.abs(step / x)))})
        _, g = block_step(J, r)
        return x, r, history, g


def do(path):
    row = json.loads(path.read_text())
    if row["cut"]:
        for k in ("err_freq_pct", "err_damp_pct", "err_rms4_pct", "err_max_pct", "err_twist_zeta"):
            row[k] = None
        path.write_text(json.dumps(row, indent=1))
        return path.name, None
    t0 = time.time()
    if "rms4" in row:                         # the superseded four-number fit, kept as data
        row["err_rms4_pct"] = row["rms4"]["err_pct"]
    if ORIG is not None:
        orig = json.loads((ORIG / path.name).read_text())
        assert orig["speed"] == row["speed"] and orig["method"] == row["method"]
        row["err_max_pct"] = orig["err_pct"]
        row["err_max_definition"] = ("100 max(|f_sway/8.01 - 1|, |zeta_sway/0.015 - 1|, |f_twist/10.087 - 1|) at "
                                     "the renormalized parameters (zeta 0.015), from the cluster's runs")
    p = SplitPoint(row)
    x, r, history, g = p.fit_split()
    e = 100 * r
    row.update({"lam": float(x[0]), "mu": float(x[1]), "E_fit_ratio": float(x[0] * p.S * p.T),
                "damping_fit": float(x[1] * p.beta_star),
                "f_sway": float((1 + r[0]) * TGT[0]), "zeta_sway": float((1 + r[1]) * TGT[1]),
                "f_twist": float((1 + r[2]) * TGT[2]), "zeta_twist": float((1 + r[3]) * TGT[3]),
                "err_sway_f": float(e[0]), "err_sway_zeta": float(e[1]), "err_twist_f": float(e[2]),
                "err_twist_zeta": float(e[3]),
                "err_freq_pct": float(np.sqrt((e[0] ** 2 + e[2] ** 2) / 2)),
                "err_damp_pct": float(np.sqrt((e[1] ** 2 + e[3] ** 2) / 2))})
    row["err_pct"] = row["err_freq_pct"]
    row["error_definition"] = ("err_pct = err_freq_pct: 100 RMS of the relative errors of f_sway and f_twist with lam "
                               "fitted to them; err_damp_pct: 100 RMS of those of zeta_sway and zeta_twist with mu "
                               "fitted to them at that lam; both from the runs at the final (lam, mu)")
    row["fit_split"] = {"start_single_mode": history[0]["x"], "iterations": history, "runs": p.runs,
                        "stationarity": g.tolist(), "seconds": time.time() - t0}
    path.write_text(json.dumps(row, indent=1))
    return path.name, row


def init(orig):
    global ORIG
    ORIG = Path(orig) if orig else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--orig", help="the cluster's first JSONs, for err_max_pct")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    paths = sorted(sa.OUT.glob("mj_*.json"))
    if args.only:
        paths = [p for p in paths if p.name in args.only]
    rows = {p: json.loads(p.read_text()) for p in paths}
    paths.sort(key=lambda p: -1 / max(rows[p]["speed"], 1e-9) if not rows[p]["cut"] else 0)
    with Pool(args.jobs, initializer=init, initargs=(args.orig,), maxtasksperchild=1) as pool:
        for name, row in pool.imap_unordered(do, paths):
            if row is None:
                print(name, "cut", flush=True)
            else:
                print(f"{name}: freq {row['err_freq_pct']:.4f}%  damp {row['err_damp_pct']:.4f}%  "
                      f"lam {row['lam']:.5f}  mu {row['mu']:.5f}  errs {row['err_sway_f']:+.3f} {row['err_sway_zeta']:+.3f} "
                      f"{row['err_twist_f']:+.3f} {row['err_twist_zeta']:+.3f}  runs {row['fit_split']['runs']}  "
                      f"[{row['fit_split']['seconds']:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
