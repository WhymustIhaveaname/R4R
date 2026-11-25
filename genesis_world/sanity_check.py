"""Checks that the Genesis pendulum is MuJoCo's, and that the fast stepping changes nothing.

  1. the joint's inertia is the rod's, with no rotor inertia added by default;
  2. each scheme matches its MuJoCo counterpart, environment by environment, with parameters of
     its own in each (some below the rod's inertia, so a negative armature) and a delay;
  3. approximate_implicitfast against implicitfast;
  4. the fast stepping (zero-copy views, unused passes off) against the public API with
     Genesis's default options, bit for bit;
  5. a scene rebuilt for another step and back gives the first trajectory bit for bit;
  6. a diverging environment comes back as None and leaves the others bit for bit as they are;
  7. the cost of a step.
With --fit, also the first two grid points of one flow chain, against MuJoCo's.

Usage:  uv run --extra genesis python sanity_check.py [--fit]
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pendulum_genesis as pg  # noqa: E402
import common as mj  # noqa: E402

N = 32
STEPS = ((50, 4), (100, 10), (10, 400))       # 200 Hz, 1 kHz, 4 kHz physics
rng = np.random.default_rng(0)
params = np.column_stack([mj.J0 * np.exp(0.3 * rng.standard_normal(N)),
                          mj.KP0 * np.exp(0.1 * rng.standard_normal(N)),
                          mj.KD0 * np.exp(0.1 * rng.standard_normal(N)),
                          rng.uniform(-1, 1, N) * 2.5e-3])
print(f"{np.sum(params[:, 0] < mj.J0)} of {N} environments have J < J0")


def max_diff(a, b):
    return max(float(np.max(np.abs(x - y))) for x, y in zip(a, b))


def bitwise(a, b):
    return all(np.array_equal(x, y) for x, y in zip(a, b))


sims = {arm: pg.GenesisPendulum(arm, N) for arm in pg.ARMS}
sims["euler_explicit"]._build(1 / 200)
print(f"1. J: genesis {sims['euler_explicit'].mass()[0]:.10f}  analytic {mj.J0:.10f}")

runs = {}
for f_ctrl, dec in STEPS:
    refs = mj.ref_sequence(f_ctrl)
    for arm in pg.ARMS:
        runs[arm, f_ctrl, dec] = got = sims[arm].rollout_batch(refs, f_ctrl, dec, params)
        ref = mj.Sim(pg.THEORY_ARM[arm])
        want = [ref.rollout(refs, f_ctrl, dec, *p) for p in params]
        print(f"2. {arm} at {f_ctrl * dec:4d} Hz: max |genesis - mujoco {pg.THEORY_ARM[arm]}| = "
              f"{max_diff(got, want):.1e} rad")
    print(f"3. approximate_implicitfast against implicitfast at {f_ctrl * dec:4d} Hz: "
          f"{max_diff(*(runs[a, f_ctrl, dec] for a in pg.ARMS[1:])):.1e} rad")

f_ctrl, dec = STEPS[0]
refs = mj.ref_sequence(f_ctrl)
for arm in pg.ARMS:
    api = pg.GenesisPendulum(arm, N, fast=False).rollout_batch(refs, f_ctrl, dec, params)
    print(f"4. {arm}: fast against the public API: "
          f"{'bit for bit' if bitwise(api, runs[arm, f_ctrl, dec]) else 'DIFFERENT'}")
    again = sims[arm].rollout_batch(refs, f_ctrl, dec, params)  # last built for STEPS[-1]
    print(f"5. {arm}: rebuilt for {STEPS[-1][0] * STEPS[-1][1]} Hz and back: "
          f"{'bit for bit' if bitwise(again, runs[arm, f_ctrl, dec]) else 'DIFFERENT'}")

bad = params.copy()
bad[0, 0] = 1e-5                                # explicit Euler at 200 Hz blows up
got = sims["euler_explicit"].rollout_batch(refs, f_ctrl, dec, bad)
print(f"6. diverged: {got[0] is None}; the others "
      f"{'bit for bit' if bitwise(got[1:], runs['euler_explicit', f_ctrl, dec][1:])
         else 'DIFFERENT'}")

for arm in pg.ARMS:
    sims[arm].rollout_batch(refs, f_ctrl, dec, params)
    t = time.perf_counter()
    sims[arm].rollout_batch(refs, f_ctrl, dec, params)
    print(f"7. {arm}: {(time.perf_counter() - t) / (len(refs) * dec) * 1e3:.3f} ms per step "
          f"of {N} environments")

if "--fit" in sys.argv:
    import json
    import run_flow as rf
    rf.ARMS = pg.ARMS
    rf.WITH_DELAY = True
    rf.first_order_prediction_delay = pg.first_order_prediction_delay
    rf.H_TARGET = rf.H_TARGET[:2]
    mujoco = json.loads((Path(mj.__file__).parent / "results" /
                         "flow_mujoco_delay.json").read_text())
    for arm in ("euler_explicit", "implicitfast_native"):
        rf.make_sim = lambda a: pg.GenesisPendulum(a, N)
        t = time.perf_counter()
        rows = rf.chain((arm, 50))
        print(f"   {(time.perf_counter() - t) / len(rows):.0f} s per fit")
        for r in rows:
            m = next(o for o in mujoco if o["arm"] == arm and o["f_ctrl_nominal"] == 50
                     and o["dec"] == r["dec"] and o["f_ctrl"] == r["f_ctrl"])
            rel = np.abs(np.array(r["params"]) / np.array(m["params"]) - 1)
            print(f"   {arm} h={r['h'] * 1e3:.2f} ms: rmse genesis {r['rmse']:.3e} mujoco "
                  f"{m['rmse']:.3e}; parameters differ by at most {rel.max():.1e}")
