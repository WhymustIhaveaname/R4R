"""System identification shared by every experiment and both engines.

Having one optimizer is the point: a fit that behaves differently in MuJoCo and in Isaac cannot
be compared, and the difference is then an artifact of the fitter rather than of the physics.
The only thing an engine supplies is how to roll out a trajectory, plus optionally a way to
evaluate several parameter vectors at once.
"""

import threading

import numpy as np
from scipy.optimize import least_squares

DIVERGED_RESIDUAL = 10.0  # residual reported for a rollout that blew up


N_RANDOM_STARTS = 2


class _Gang:
    """Lets several fits share one batched rollout.

    Each start runs in its own thread and calls scipy exactly as it would on its own; this
    only collects the rollout requests that are outstanding at the same moment and issues
    them as a single call.  The optimizer, its tolerances and its iterates are untouched, so
    the answer is the serial answer - what changes is that a simulator which costs the same
    for eight worlds as for sixty-four is asked once instead of seven times.
    The rollouts run on the thread that serves the gang, never on a fit's thread: Isaac only
    steps from the thread that created it, and hangs when asked from another.
    """

    def __init__(self, batch, n_fits):
        self.batch, self.active = batch, n_fits
        self.cv = threading.Condition()
        self.pending, self.results = {}, {}

    def submit(self, rows):
        me = threading.get_ident()
        with self.cv:
            self.pending[me] = rows
            self.cv.notify_all()
            while me not in self.results:
                self.cv.wait()
            return self.results.pop(me)

    def retire(self):
        with self.cv:
            self.active -= 1
            self.cv.notify_all()

    def serve(self):
        """Issue one batched rollout whenever every live fit is waiting; return when all are done."""
        while True:
            with self.cv:
                while self.active and len(self.pending) < self.active:
                    self.cv.wait()
                if not self.active:
                    return
                order, requests = list(self.pending), dict(self.pending)
                self.pending.clear()
            out = self.batch([r for t in order for r in requests[t]])
            with self.cv:
                i = 0
                for t in order:
                    k = len(requests[t])
                    self.results[t], i = out[i:i + k], i + k
                self.cv.notify_all()


def start_points(h, predicted, prev, rng, with_delay, param0):
    """The starting points every fit uses, whichever engine it runs in.

    `predicted` is the scheme's own first-order prediction.  Inflated inertia keeps explicit PD
    stable when h K_D / J is no longer small, and the random restarts cover the poor local
    optima the landscape has at coarse control rates.
    """
    j0, kp0, kd0 = param0
    starts = [np.append(param0, 0.0) if with_delay else np.asarray(param0),
              np.asarray(predicted, float)]
    starts += [np.array([j0 + k * h * kd0, kp0, kd0, h / 2][:3 + with_delay])
               for k in (1, 2, 4)]
    for _ in range(N_RANDOM_STARTS):
        jitter = np.asarray(param0) * np.exp(rng.normal(0, 0.5, 3))
        starts.append(np.append(jitter, rng.uniform(0, h)) if with_delay else jitter)
    return starts + ([np.asarray(prev, float)] if prev is not None else [])


def identify(rollout, gt, h, starts, param0, with_delay=False, batch=None,
             inertia_floor=None, fixed=()):
    """Least-squares fit of (J, K_P, K_D) in log space, plus the command delay in units of h.

    `rollout(J, kp, kd, delay)` returns the simulated angles, already concatenated over every
    command, so that one parameter set is fitted to all of them at once; None means it diverged.
    `batch(params)` is optional: given several parameter vectors it returns one trajectory each,
    which lets a simulator that steps many copies at once evaluate the Jacobian in a single
    rollout.  It changes what a Jacobian costs, not what the fit is.
    `inertia_floor` switches the first parameter from log space to a linear coordinate in units
    of h K_D / 2, bounded so the total inertia stays above zero.  A joint that carries no rotor
    inertia of its own has nothing to take a logarithm of, and its counterterm may be negative.
    `fixed` holds indices into (J, K_P, K_D, delay) that are held at their measured value
    instead of being fitted.  An operator whose counterterm the theory says is zero is not free
    to help: it is not orthogonal to the others, so letting it absorb higher-order residual
    drags the ones that do flow off their predicted slopes.
    `starts` are parameter vectors (J, K_P, K_D) or (J, K_P, K_D, delay); returns the best fit.
    """
    gt = np.asarray(gt)
    n = gt.size          # a multi-joint ground truth is 2-D
    unit = h / 2 * param0[2]                  # the predicted rotor inertia
    linear = inertia_floor is not None
    bounds = (-np.inf, np.inf)
    if linear:
        lo = [-0.9 * inertia_floor / unit, -2.0, -2.0] + ([-5.0] if with_delay else [])
        hi = [0.9 * inertia_floor / unit, 2.0, 2.0] + ([5.0] if with_delay else [])
        bounds = (np.array(lo), np.array(hi))

    ndim = 4 if with_delay else 3
    free = [k for k in range(ndim) if k not in fixed]
    full_bounds = bounds                        # in all ndim coordinates, for clipping starts
    if fixed:                                   # the bounds follow the free coordinates
        bounds = ((bounds[0][free], bounds[1][free]) if linear else bounds)

    def unpack(x):
        full = np.zeros(ndim)
        full[free] = x
        head = ((full[0] * unit,) if linear else (param0[0] * np.exp(full[0]),))
        return (*head, *(np.asarray(param0[1:3]) * np.exp(full[1:3])), *(full[3:] * h))

    def pack(p):
        p = np.asarray(p, dtype=float)
        first = p[0] / unit if linear else np.log(p[0] / param0[0])
        x = np.append(first, np.log(p[1:3] / np.asarray(param0[1:3])))
        x = np.append(x, (p[3] if len(p) > 3 else 0.0) / h) if with_delay else x
        # clip in all ndim coordinates, before dropping the held ones
        x = np.clip(x, full_bounds[0] * 0.99, full_bounds[1] * 0.99) if linear else x
        return x[free]

    def clean(theta):
        """A rollout that blew up is reported as a large residual, never as NaN."""
        if theta is None:
            return np.full(n, DIVERGED_RESIDUAL), False
        theta = np.asarray(theta)
        if not np.all(np.isfinite(theta)) or np.max(np.abs(theta)) > 1e3:
            return np.full(n, DIVERGED_RESIDUAL), False
        return (theta - gt).ravel() / np.sqrt(n), True

    # The K_P and K_D columns are nearly parallel, so the Gauss-Newton step hangs on their small
    # difference.  At 1e-4 the float32 noise of a batched simulator swamps that difference, and
    # the fit stalls far from the minimum; 1e-2 keeps it above the noise.
    eps = 1e-2

    def solve(p, roll):
        """One start, with `roll(params) -> trajectories` however they get evaluated."""
        def residual(x):
            return clean(roll([unpack(x)])[0])[0]

        def jac(x):
            steps = np.eye(len(x)) * eps
            half = len(steps)
            raw = roll([unpack(x + d) for d in steps] + [unpack(x - d) for d in steps])
            rows, ok = zip(*(clean(t) for t in raw))
            J = (np.asarray(rows[:half]) - np.asarray(rows[half:])) / (2 * eps)
            J[[k for k in range(half) if not (ok[k] and ok[half + k])]] = 0.0
            return J.T

        x0 = pack(p)
        r0 = residual(x0)
        if np.all(r0 == r0[0]) and r0[0] > 0:  # diverged start
            return (np.inf, np.array(unpack(x0)), x0, None)
        res = least_squares(residual, x0, method="trf", jac=jac if batch else "2-point",
                            bounds=bounds, xtol=1e-12, ftol=1e-14, gtol=1e-14, max_nfev=400)
        return (np.sqrt(2 * res.cost), np.array(unpack(res.x)), res.x, res.jac)

    fits = []
    if batch is None:
        serial = lambda rows: [rollout(*row) for row in rows]
        fits = [solve(p, serial) for p in starts]
    else:
        # Every start advances at once, so a simulator that charges the same for eight worlds
        # as for sixty-four is asked once per round instead of once per start.
        gang = _Gang(batch, len(starts))
        fits = [None] * len(starts)

        def worker(k, p):
            try:
                fits[k] = solve(p, gang.submit)
            finally:
                gang.retire()

        threads = [threading.Thread(target=worker, args=(k, p), daemon=True)
                   for k, p in enumerate(starts)]
        for t in threads:
            t.start()
        gang.serve()
        for t in threads:
            t.join()
    rmse, params, x, jac_out = min(fits, key=lambda f: f[0])
    cond = np.linalg.cond(jac_out) if jac_out is not None else np.inf
    spread = [f[1].tolist() for f in fits if np.isfinite(f[0])]
    return {"rmse": rmse, "params": params.tolist(), "cond": cond,
            "start_rmse": [f[0] for f in fits], "start_params": spread}


def cross_seed(rows, problem, param0, key="params", rmse_key="rmse", with_delay=True):
    """Refit every grid point from the answers the other chains found, and keep the lowest.

    Each chain runs at one nominal control rate and walks up in h, starting every fit from its
    own previous answer.  If the fit has several minima, chains can settle in different ones,
    and the flow then scatters by control rate for a reason that is the fitter's, not the
    simulator's.  So each row is refitted from its own answer and from the best-scoring answer
    of the other chains near its h, and the lower of the two minima is kept.
    `problem(row)` returns (gt, rollout, batch) for the row's grid point.  Rows are updated in
    place; a row whose fit moved gets "reseeded": the RMSE it had.  Returns that count.
    """
    chains = {}
    for r in rows:
        chains.setdefault(r["f_ctrl_nominal"], []).append(r)
    moved = 0
    for r in rows:
        gt, rollout, batch = problem(r)
        seeds = [min(c, key=lambda o: abs(o["h"] - r["h"]))[key]
                 for f, c in chains.items() if f != r["f_ctrl_nominal"]]
        if not seeds:
            continue
        score = [np.inf if t is None else float(np.sqrt(np.mean((np.asarray(t) - gt) ** 2)))
                 for t in batch(seeds)]
        best = np.asarray(seeds[int(np.argmin(score))], float)
        fit = identify(rollout, gt, r["h"], [np.asarray(r[key], float), best], param0,
                       with_delay=with_delay, batch=batch)
        if fit["rmse"] < r[rmse_key] * (1 - 1e-3):
            r["reseeded"] = r[rmse_key]
            r[key], r[rmse_key] = fit["params"], fit["rmse"]
            moved += 1
    return moved
