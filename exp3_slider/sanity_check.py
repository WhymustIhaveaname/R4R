"""Checks the ground truth against MuJoCo, and measures the largest Lyapunov exponent.

  1. at penetrating states, the ground truth's contact force against MuJoCo's;
  2. MuJoCo converges to the ground truth as h -> 0, in every scheme;
  3. the largest Lyapunov exponent of the ground truth, by Benettin's method: a neighbour at
     distance D0 is followed for one command period, its distance measured and reset to D0, and
     the exponent is the mean log growth per second.  The distance is taken on the state scaled
     to (theta, theta' / W, x / L, x' / (L W)), W = 2 pi F_REF; the exponent does not depend on
     the scaling.

Usage:  uv run python sanity_check.py
"""

import mujoco
import numpy as np

import common as c

F_CTRL = 50

# 1. penetrating states: sliding either way along the face, and at rest
m = mujoco.MjModel.from_xml_string(c.model_xml())
d = mujoco.MjData(m)
for qpos, qvel in (([c.THETA_C + 0.03, 0.0], [0.0, 0.0]), ([c.THETA_C + 0.05, 0.01], [2.0, -0.3]),
                   ([c.THETA_C + 0.05, 0.01], [-2.0, 0.3])):
    d.qpos[:], d.qvel[:] = qpos, qvel
    mujoco.mj_forward(m, d)
    (th, x), (om, v) = qpos, qvel
    a_th, a_x = -c.MGL2 * np.sin(th) / c.J0, -c.K_S * x / c.M_S  # no control
    q_th, q_x, dist = c.Contact().force(th, om, x, v, a_th, a_x, c.J0)
    print(f"1. penetration {-dist * 1e3:5.2f} mm, velocity {qvel}: generalized force "
          f"ground truth ({q_th:+.10f}, {q_x:+.10f}), MuJoCo ({d.qfrc_constraint[0]:+.10f}, "
          f"{d.qfrc_constraint[1]:+.10f})")

# 2. convergence
refs = c.ref_sequence(F_CTRL, t_end=c.T_SHOW)
for dec in (20, 200, 2000):
    gt = c.ground_truth(refs, F_CTRL, dec)
    for arm in c.ARMS:
        err = np.abs(c.mujoco_rollout(refs, F_CTRL, dec, arm) - gt).max(axis=0)
        print(f"2. {arm} at {F_CTRL * dec:6d} Hz: max |MuJoCo - ground truth| "
              f"theta {err[0]:.2e} rad, x {err[1]:.2e} m")

# 3. largest Lyapunov exponent
T_LYAP, D0 = 60.0, 1e-7
W = 2 * np.pi * c.F_REF
scale = np.array([1.0, 1 / W, 1 / c.L, 1 / (c.L * W)])
refs = c.ref_sequence(F_CTRL, t_end=T_LYAP)
rng = np.random.default_rng(0)
y = np.zeros(4)
u = rng.standard_normal(4)
z = y + D0 * u / np.linalg.norm(u) / scale
logs = []
for k, r in enumerate(refs):
    y_next = c.ground_truth([r], F_CTRL, 1, state0=y, full=True)[-1]
    z_next = c.ground_truth([r], F_CTRL, 1, state0=z, full=True)[-1]
    sep = (z_next - y_next) * scale
    dist = np.linalg.norm(sep)
    logs.append(np.log(dist / D0))
    y, z = y_next, y_next + sep / dist * D0 / scale
logs = np.array(logs)
for t in (15, 30, 60):
    n = int(t * F_CTRL)
    print(f"3. largest Lyapunov exponent over {t:2d} s: {logs[:n].sum() / t:+.2f} /s")
lam = logs.sum() / T_LYAP
print(f"   a perturbation shrinks by e every {-1 / lam * 1e3:.0f} ms; over the {c.T_FIT:g} s window, "
      f"by {np.exp(lam * c.T_FIT):.1e}")
