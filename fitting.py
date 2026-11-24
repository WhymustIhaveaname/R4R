"""System identification shared by every experiment and every engine.

Having one optimizer is the point: a fit that behaves differently in MuJoCo and in Isaac cannot
be compared, and the difference is then an artifact of the fitter rather than of the physics.
The only thing an engine supplies is how to roll out a trajectory, plus optionally a way to
evaluate several parameter vectors at once.

The fit is CMA-ES, which uses no derivatives: it samples a population around its current
guess, ranks the samples by error, and moves and reshapes its sampling distribution towards
the better ones.  A finite-difference Jacobian needs a step that clears each engine's noise -
float64 MuJoCo and float32 Isaac want different ones, and one chosen too small once stalled
the Isaac fits - while a ranking needs no such choice, so every engine runs the same fitter
with the same settings.
"""

import cma
import numpy as np

DIVERGED_RESIDUAL = 10.0  # error reported for a rollout that blew up

N_RANDOM_STARTS = 2
# 32 rather than CMA-ES's default of 8: a batched simulator steps 32 copies for the price of 8,
# and the larger population needs 70 generations instead of 150.  The parameters come out the
# same to 1e-7, well below the 1e-5 the smallest flows in the figures need.
POPSIZE = 32
SIGMA0 = 0.1          # initial spread, in the coordinates identify() fits in
TOLX, TOLFUN = 1e-7, 1e-12
MAX_EVALS = 20000


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


def fit_steps(gt, h, starts, param0, with_delay=False, inertia_floor=None, fixed=()):
    """Fit (J, K_P, K_D), and the command delay if asked, by CMA-ES on the trajectory RMSE.

    A generator: it yields the parameter rows it wants rolled out, is sent back one trajectory
    per row (None where the rollout blew up), and returns the fit.  Written this way, several
    fits can have their rows rolled out together; identify() drives a single one.
    The coordinates are log J, log K_P, log K_D and the delay in units of h.
    `inertia_floor` switches the first parameter from log space to a linear coordinate in units
    of h K_D / 2, bounded so the total inertia stays above zero.  A joint that carries no rotor
    inertia of its own has nothing to take a logarithm of, and its counterterm may be negative.
    `fixed` holds indices into (J, K_P, K_D, delay) that are held at their measured value
    instead of being fitted.  An operator whose counterterm the theory says is zero is not free
    to help: it is not orthogonal to the others, so letting it absorb higher-order residual
    drags the ones that do flow off their predicted slopes.
    `starts` are parameter vectors (J, K_P, K_D) or (J, K_P, K_D, delay); the search starts
    from whichever of them has the lowest error.
    """
    gt = np.asarray(gt)
    unit = h / 2 * param0[2]                  # the predicted rotor inertia
    linear = inertia_floor is not None
    ndim = 4 if with_delay else 3
    free = [k for k in range(ndim) if k not in fixed]
    lo = np.array([-0.9 * inertia_floor / unit, -2.0, -2.0, -5.0])[:ndim] if linear else None
    hi = -lo if linear else None

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
        x = np.clip(x, lo * 0.99, hi * 0.99) if linear else x
        return x[free]

    def error(theta):
        """Trajectory RMSE; a rollout that blew up scores a large error, never NaN."""
        if theta is None:
            return DIVERGED_RESIDUAL
        theta = np.asarray(theta)
        if not np.all(np.isfinite(theta)) or np.max(np.abs(theta)) > 1e3:
            return DIVERGED_RESIDUAL
        return float(np.sqrt(np.mean((theta - gt) ** 2)))

    xs = [pack(p) for p in starts]
    start_error = [error(t) for t in (yield [unpack(x) for x in xs])]
    # Its own random stream, not numpy's global one, so that fits run side by side cannot
    # change each other's samples.
    rng = np.random.default_rng(1)
    options = {"popsize": POPSIZE, "tolx": TOLX, "tolfun": TOLFUN, "maxfevals": MAX_EVALS,
               "randn": lambda *shape: rng.standard_normal(shape), "verbose": -9}
    if linear:
        options["bounds"] = [list(lo[free]), list(hi[free])]
    es = cma.CMAEvolutionStrategy(xs[int(np.argmin(start_error))], SIGMA0, options)
    while not es.stop():
        population = es.ask()
        es.tell(population, [error(t) for t in (yield [unpack(x) for x in population])])
    return {"rmse": float(es.result.fbest), "params": list(map(float, unpack(es.result.xbest))),
            "evaluations": int(es.result.evaluations), "start_rmse": start_error}


def drive(fit, run):
    """Run one fit_steps generator to the end, with `run(rows) -> trajectories`."""
    try:
        rows = next(fit)
        while True:
            rows = fit.send(run(rows))
    except StopIteration as done:
        return done.value


def identify(rollout, gt, h, starts, param0, with_delay=False, batch=None,
             inertia_floor=None, fixed=()):
    """fit_steps for a single fit.  `rollout(J, kp, kd, delay)` returns the simulated angles,
    already concatenated over every command, so that one parameter set is fitted to all of them
    at once; None means it diverged.  `batch(params)`, if given, returns one trajectory per
    parameter vector from a single rollout; it changes what a generation costs, not the fit."""
    return drive(fit_steps(gt, h, starts, param0, with_delay, inertia_floor, fixed),
                 batch or (lambda rows: [rollout(*row) for row in rows]))


def cross_seed(rows, problem, param0, key="params", rmse_key="rmse", with_delay=True,
               on_row=lambda: None):
    """Score every grid point at the answers the other chains found, and refit where one wins.

    Each chain runs at one nominal control rate and walks up in h, starting every fit from its
    own previous answer.  If the fit has several minima, chains can settle in different ones,
    and the flow then scatters by control rate for a reason that is the fitter's, not the
    simulator's.  So each row is scored at the answers of the other chains nearest its h; where
    one of them does better than the row's own fit, the row is refitted from both and the lower
    minimum is kept.
    `problem(row)` returns (gt, rollout, batch) for the row's grid point.  Rows are updated in
    place; a row whose fit moved gets "reseeded": the RMSE it had, and every row gets
    "cross_seeded", so that `on_row`, called after each, can save the work.  Returns the count.
    """
    chains = {}
    for r in rows:
        chains.setdefault(r["f_ctrl_nominal"], []).append(r)
    moved = 0
    for r in rows:
        seeds = [min(c, key=lambda o: abs(o["h"] - r["h"]))[key]
                 for f, c in chains.items() if f != r["f_ctrl_nominal"]]
        if r.get("cross_seeded") or not seeds:
            continue
        gt, rollout, batch = problem(r)
        run = batch or (lambda rows: [rollout(*row) for row in rows])
        score = [np.inf if t is None else float(np.sqrt(np.mean((np.asarray(t) - gt) ** 2)))
                 for t in run(seeds)]
        if min(score) < r[rmse_key]:
            best = np.asarray(seeds[int(np.argmin(score))], float)
            fit = identify(rollout, gt, r["h"], [np.asarray(r[key], float), best], param0,
                           with_delay=with_delay, batch=batch)
            if fit["rmse"] < r[rmse_key] * (1 - 1e-3):
                r["reseeded"] = r[rmse_key]
                r[key], r[rmse_key] = fit["params"], fit["rmse"]
                moved += 1
        r["cross_seeded"] = True
        on_row()
    return moved
