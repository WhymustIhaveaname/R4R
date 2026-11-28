"""Experiment 4a: the free vibration of a soft block, against the continuum.

A 6 x 5 x 4 cm block of a tofu-like gel floats free, with no gravity and no contact.  It starts
undeformed, with the velocity field of the continuum's lowest elastic mode.  The continuum
answer is then one damped oscillation of that mode shape (ritz.py gives the shape and the
frequency exactly).  Each scheme runs on a grid of cell size dx and step h.  Its block's motion is
projected on that mode, and the fit finds the Young's modulus E* and the damping that give the
projection the continuum's frequency and damping ratio.

The damping models differ between engines: MuJoCo's flex and Genesis's implicit FEM damp in
proportion to the stiffness (beta K), Genesis's explicit FEM in proportion to the mass (a factor
exp(-c h) on the velocity each step), and Genesis's MPM not at all.  For a single mode they are
the same thing, a damping ratio zeta, so every scheme is fitted and plotted in zeta.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).parent
RESULTS = HERE / "results"
CACHE = RESULTS / "cache"

SIDES = np.array([0.06, 0.05, 0.04])  # m
HALF = SIDES / 2
E0 = 1.13e4      # Pa; silken tofu, mu = 3.9 kPa (Pattison et al. 2015) at nu = 0.45
NU = 0.45
RHO = 1000.0     # kg / m^3
ZETA0 = 0.10     # damping ratio of the mode, as Pattison's silken tofu at 14 Hz
MODE = 6         # the lowest elastic mode, after the six rigid ones: u_z ~ x y, 14.36 Hz
U_AMP = 5e-5     # m, peak displacement: 0.1 % strain, so every material law is linear
WINDOW = 0.3     # s, four periods; the fit and the residual both use it
RITZ_ORDER = 12
OBS_SPACING = 0.01  # the 1 cm lattice, on which U_AMP is the peak
N_TIME_SAMPLES = 600


@lru_cache(maxsize=None)
def continuum():
    """The reference block's Ritz solution, cached on disk (it takes about 15 s)."""
    from ritz import FreeBox
    key = f"ritz_{'_'.join(f'{x:g}' for x in (*HALF, E0, NU, RHO))}_{RITZ_ORDER}.npz"
    path = CACHE / key
    if path.exists():
        box = FreeBox.__new__(FreeBox)
        z = np.load(path)
        box.half, box.rho, box.exps, box.omega, box.modes = (z["half"], float(z["rho"]), z["exps"],
                                                             z["omega"], z["modes"])
        return box
    box = FreeBox(HALF, E0, NU, RHO, order=RITZ_ORDER)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, half=box.half, rho=box.rho, exps=box.exps, omega=box.omega, modes=box.modes)
    return box


def omega0():
    return float(continuum().omega[MODE])


def lattice(spacing):
    """Grid points of the block with the given spacing, x fastest last (matches MuJoCo's grid)."""
    axes = [np.linspace(-a, a, int(round(2 * a / spacing)) + 1) for a in HALF]
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)


def mode_shape(points):
    return continuum().displacement(np.asarray(points, float), MODE)


@lru_cache(maxsize=None)
def amplitude():
    """Scale of the initial velocity field v0 = A phi that gives a peak displacement of U_AMP."""
    phi = mode_shape(lattice(OBS_SPACING))
    return U_AMP * omega0() / np.abs(phi).max()


def sample_steps(h):
    """The step indices observed in the window: about N_TIME_SAMPLES of them, evenly spaced."""
    n = int(round(WINDOW / h))
    every = max(1, n // N_TIME_SAMPLES)
    return np.arange(every, n + 1, every)


def modal_weights(shape, masses):
    """w such that q = sum_i w_i . u_i is the amplitude of `shape` in a displacement field u.

    The projection uses the engine's own masses, in which the scheme's modes are orthogonal.
    With the continuum's shape at the points, q is in the units of the continuum's
    q(t) = A exp(-zeta w t) sin(w_d t) / w_d; p1.mode_vector is scaled to keep those units.
    """
    w = masses[:, None] * shape
    return w / np.sum(w * shape)


def seeded_noise(points, masses):
    """A small random velocity on every vertex or particle, a thousandth of the mode's peak.

    It puts every elastic mode of the scheme in motion, so that a step past the scheme's
    stability limit blows up as it would in use; a pure mode would start the unstable ones at
    round-off and hide the limit.  The projection on the mode filters the noise out of q.  Its
    momentum and angular momentum are removed, or the free block would drift away.
    """
    rng = np.random.default_rng(0)
    v = 1e-3 * U_AMP * omega0() * rng.standard_normal((len(points), 3))
    m = masses[:, None]
    v -= (m * v).sum(0) / m.sum()
    r = points - (m * points).sum(0) / m.sum()
    L = (m * np.cross(r, v)).sum(0)
    inertia = (m[:, 0, None, None] * ((r * r).sum(1)[:, None, None] * np.eye(3)
                                       - r[:, :, None] * r[:, None, :])).sum(0)
    v -= np.cross(np.linalg.solve(inertia, L), r)
    return v


def unstable(rest):
    """Whether the motion outside the mode grew over the window, rest(t) being its peak.

    The projection keeps the other modes out of q even when they blow up, as they do past a
    scheme's stability limit, so the rest of the motion is watched separately: damped, it
    decays from the seeded noise; growing, the scheme is unstable at this step.  The first and
    the last quarters are compared, since an engine may report the positions a step late.
    """
    k = max(1, len(rest) // 4)
    last = np.max(rest[-k:])
    return not np.isfinite(last) or last > 2 * np.max(rest[:k])


def truth_q(t):
    w, z = omega0(), ZETA0
    wd = w * np.sqrt(1 - z * z)
    return amplitude() * np.exp(-z * w * t) * np.sin(wd * t) / wd


def measure(t, q):
    """(omega, zeta) of a damped oscillation q(t) = a exp(-zeta w t) sin(w_d t + p).

    Amplitude and phase are free: only the frequency and the damping are the behaviour, as a
    camera on a wobbling block would see them.
    """
    import warnings
    from scipy.optimize import OptimizeWarning, curve_fit
    if not np.all(np.isfinite(q)) or np.abs(q).max() > 1e3 * amplitude():
        return np.nan, np.nan
    w0 = omega0()

    def model(t, a, s, wd, ph):
        return a * np.exp(-s * t) * np.sin(wd * t + ph)

    p0 = [amplitude() / w0, ZETA0 * w0, w0, 0.0]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            p, _ = curve_fit(model, t, q, p0=p0, maxfev=20000)
    except RuntimeError:
        return np.nan, np.nan
    a, s, wd, ph = p
    wn = np.hypot(s, wd)
    return float(wn), float(s / wn)


# ---- the parameters: x = (ln(E*/E0), zeta*) -------------------------------------------------

def damping_param(arm_model, zeta):
    """The engine's damping parameter that gives the continuum mode the damping ratio zeta."""
    w = omega0()
    if arm_model == "stiffness":   # beta K: zeta = beta w / 2
        return 2 * zeta / w
    if arm_model == "mass":        # c M: zeta = c / (2 w)
        return 2 * zeta * w
    if arm_model == "none":
        return 0.0
    raise ValueError(arm_model)


def time_prediction(scheme, h):
    """First- and second-order counterterms of the time step, for one mode (theory.py derives them).

    Returns (E*/E, zeta*) at the reference zeta.  scheme is one of
      'symplectic_explicit_beta'  symplectic Euler, beta K damping explicit (MuJoCo Euler)
      'backward_euler_beta'       backward Euler, beta K damping (MuJoCo discrete, Genesis implicit)
      'symplectic_exp_mass'       symplectic Euler, then v <- exp(-c h) v (Genesis explicit FEM)
      'symplectic_none'           symplectic Euler, undamped (MPM's time stepping)
    """
    w, z = omega0(), ZETA0
    if scheme == "symplectic_explicit_beta":
        b = 2 * z / w
        e = 1 - h * b * w**2 / 2 + h**2 * w**2 * (2 * b**2 * w**2 - 1) / 12
        bs = b + h**2 * b * w**2 / 12
        return e, bs * w / 2
    if scheme == "backward_euler_beta":
        b = 2 * z / w
        e = 1 + h * b * w**2 / 2 + h**2 * w**2 * (2 * b**2 * w**2 - 1) / 12
        bs = b - h + h**2 * b * w**2 / 12
        return e, bs * w / 2
    if scheme == "symplectic_exp_mass":
        c = 2 * z * w
        return 1 + h * c / 2 + h**2 * (2 * c**2 - w**2) / 12, z
    if scheme == "symplectic_none":
        return 1 - h**2 * w**2 / 12, 0.0
    raise ValueError(scheme)


def fit(measure_at, starts, zeta_free=True):
    """Solve omega(E*, zeta*) = omega0 and zeta(E*, zeta*) = zeta0; returns (E*/E0, zeta*, misfit, nfev).

    measure_at(E_ratio, zeta) runs the scheme and returns its (omega, zeta).  Two equations in
    two unknowns, both smooth in float64: a Gauss-Newton solve needs about ten rollouts.  The
    misfit is |(ln(omega / omega0), zeta - zeta0)|; it is not zero where no value of the
    parameters reproduces the continuum, as when the scheme goes unstable first.  A scheme with
    no damping parameter (zeta_free=False) fits E* to the frequency alone and returns its own,
    numerical damping as zeta*.
    """
    best = None
    for x0 in starts:
        x0 = np.asarray(x0, float) if zeta_free else np.asarray(x0[:1], float)

        def fun(x):
            w, z = measure_at(np.exp(x[0]), x[1] if zeta_free else 0.0)
            if not np.isfinite(w):
                return np.full(len(x), 10.0)
            r = [np.log(w / omega0())]
            return np.array(r + [z - ZETA0] if zeta_free else r)

        res = least_squares(fun, x0, method="lm", diff_step=1e-5, x_scale=0.01,
                            xtol=1e-12, ftol=1e-14, max_nfev=40)
        misfit = float(np.linalg.norm(res.fun))
        if best is None or misfit < best[2]:
            E = float(np.exp(res.x[0]))
            Z = float(res.x[1]) if zeta_free else measure_at(E, 0.0)[1]
            best = (E, Z, misfit, res.nfev)
    return best
