"""PD-controlled pendulum striking a passive slider: model, ground truth, and the parameters.

The pendulum is experiment 1's: a uniform rod, 1 lb and 1 ft long, hinged at one end, with
the ANYmal-C gains, driven by a sine as in experiment 1.  A massless ball at its tip strikes a
block.  The block rides a frictionless rail (a prismatic joint) and is held by a spring on that
joint, with no damper: it rings at its own frequency, irrational against the drive's.  The
contact's normal is the rail, and its friction is 0.01, the least Genesis allows, so the force
acts almost entirely along the rail.  The strikes damp the block, so the motion locks onto the
drive's period after about 1.5 s (the largest Lyapunov exponent is -2.4 /s).

The contact follows MuJoCo's soft-constraint model with the elliptic friction cone: the
reference acceleration is aref = -b v - k d r along the normal, of the penetration r, and
-b v along the face, with the impedance d(r) from solimp.  The impedance starts near zero at
first touch, so the contact force grows smoothly from zero instead of jumping; a force that
jumps at first touch would make every engine's error depend on where in its step the touch falls.
"""

import functools

import mujoco
import numpy as np
from scipy.integrate import solve_ivp

# The pendulum of experiment 1 (SI units).
M = 0.45359237  # 1 lb
L = 0.3048  # 1 ft
G = 9.81
J0 = M * L**2 / 3
MGL2 = M * G * L / 2
KP0 = 80.0
KD0 = 2.0

# The striker and the slider.
R_BALL = 0.03  # m, the ball at the tip
THETA_C = 0.3  # rad, the angle at which the ball touches the block at rest
M_S = 0.15  # kg, the block; about the rod's inertia seen at its tip, J0 / L^2
F_S = 2.0 * (1 + 5**0.5) / 2  # Hz, the block's frequency on its spring: the drive's times the golden ratio
K_S = M_S * (2 * np.pi * F_S) ** 2  # N/m
X_FACE = -L * np.sin(THETA_C) - R_BALL  # m, the block's struck face at rest
BLOCK_HALF = 0.035  # m, half the side of the block, a cube
BLOCK_Z = -L * np.cos(0.35)  # m, the block's centre height, near where the ball strikes

# The contact: a time constant of 12 ms, which no physics step of the grid clamps (both engines
# raise it to 2h when it is shorter), MuJoCo's default damping ratio, and an impedance that
# rises from its floor at first touch to 0.95 over 2 cm of penetration.  The strikes come at up
# to 1.8 m/s, so the rise takes at least 11 ms, two of the coarsest steps, and the ball sinks in
# at most 16 mm.
SOLREF = (0.012, 1.0)
SOLIMP = (0.0001, 0.95, 0.02, 0.5, 2.0)
# Genesis raises every contact's friction to at least 0.01, so no contact is frictionless in it.
MU = 0.01
IMPRATIO = 1.0  # MuJoCo's default; Genesis's, with the elliptic cone, is 100

# Command: theta_ref(t) = A sin(2 pi f t), sampled at f_ctrl and held, as in experiment 1.
A_REF = 0.4  # rad
F_REF = 2.0  # Hz
T_FIT = 4.0  # s, the identification window, as in experiment 1
T_SHOW = 4.5  # s, the preview, a little longer than the window

ARMS = ("euler_explicit", "rk4_explicit", "implicitfast_native")
ARM_LABEL = {"euler_explicit": "Euler + explicit PD",
             "rk4_explicit": "RK4 + explicit PD (torque held)",
             "implicitfast_native": "implicitfast + native PD"}
# What each scheme fits, as indices into (J, K_P, K_D, d, lam): the parameters the theory gives
# a counterterm (see first_order_prediction); the others stay at their measured values.
FITTED = {"euler_explicit": (0, 3, 4), "rk4_explicit": (0, 2), "implicitfast_native": (0, 3, 4)}
# The contact's damping rate, which sets its counterterm.
B = 2 / (SOLIMP[1] * SOLREF[0])
PARAM0 = np.array([J0, KP0, KD0])
# A trajectory is (theta, x / L): the block's displacement as the angle it subtends at the hinge,
# so that the fit weighs the two coordinates alike.
OBSERVE = np.array([1.0, 1.0 / L])
ARM_INTEGRATOR = {"euler_explicit": mujoco.mjtIntegrator.mjINT_EULER,
                  "rk4_explicit": mujoco.mjtIntegrator.mjINT_RK4,
                  "implicitfast_native": mujoco.mjtIntegrator.mjINT_IMPLICITFAST}


def ref_sequence(f_ctrl, t_end=T_FIT):
    t = np.arange(int(round(t_end * f_ctrl))) / f_ctrl
    return A_REF * np.sin(2 * np.pi * F_REF * t)


def model_xml(arm="euler_explicit"):
    actuator = ('<position name="pd" joint="hinge" kp="80" kv="2"/>' if arm == "implicitfast_native"
                else '<motor name="tau" joint="hinge"/>')
    sol = (f'solref="{SOLREF[0]} {SOLREF[1]}" solimp="{" ".join(str(v) for v in SOLIMP)}" '
           f'condim="3" friction="{MU} 0 0" contype="1" conaffinity="1"')
    hb = BLOCK_HALF
    return f"""
<mujoco model="pendulum_slider">
  <compiler inertiafromgeom="false"/>
  <option timestep="0.005" gravity="0 0 -{G}" cone="elliptic" impratio="{IMPRATIO}"/>
  <visual><global offwidth="720" offheight="720"/></visual>
  <worldbody>
    <light pos="0 -1 1" dir="0 1 -1"/>
    <geom type="cylinder" fromto="0 -0.03 0 0 0.03 0" size="0.012" rgba=".3 .3 .3 1"
          contype="0" conaffinity="0"/>
    <body name="rod">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <inertial pos="0 0 {-L / 2}" mass="{M}" diaginertia="{M * L**2 / 12} {M * L**2 / 12} 1e-8"/>
      <geom type="capsule" fromto="0 0 0 0 0 {-L}" size="0.008" rgba=".85 .45 .15 1"
            contype="0" conaffinity="0"/>
      <geom name="ball" type="sphere" pos="0 0 {-L}" size="{R_BALL}" rgba=".85 .45 .15 1" {sol}/>
    </body>
    <body name="slider" pos="{X_FACE - hb} 0 {BLOCK_Z}">
      <joint name="rail" type="slide" axis="1 0 0" stiffness="{K_S}"/>
      <inertial pos="0 0 0" mass="{M_S}" diaginertia="1e-4 1e-4 1e-4"/>
      <geom name="block" type="box" size="{hb} {hb} {hb}" rgba=".16 .47 .84 1" {sol}/>
    </body>
  </worldbody>
  <actuator>{actuator}</actuator>
</mujoco>"""


def _impedance(r):
    """MuJoCo's impedance d of a penetration depth r >= 0 (getimpedance in engine_core_constraint)."""
    dmin, dmax, width, mid, power = SOLIMP
    x = r / width
    if x >= 1:
        return dmax
    if x <= mid:
        y = x**power / mid ** (power - 1)
    else:
        y = 1 - (1 - x) ** power / (1 - mid) ** (power - 1)
    return dmin + y * (dmax - dmin)


@functools.cache
def _invweight():
    """MuJoCo's diagonal approximation of the contact's inverse inertia, sum of body_invweight0."""
    m = mujoco.MjModel.from_xml_string(model_xml())
    return m.body_invweight0[m.body("rod").id, 0] + m.body_invweight0[m.body("slider").id, 0]


class Contact:
    """The soft contact between the ball and the block, as MuJoCo resolves it.

    The penetration's distance is dist = (x_ball - R_BALL) - (X_FACE + x), with the Jacobian
    jn = d dist / d(theta, x) = (-L cos theta, -1).  The face's direction in the plane of the
    motion is z, along which the contact point, midway between the two surfaces at x_p, moves
    with the rod: jt = (-x_p, 0).  The force f = (f_n, f_t) minimizes
    1/2 f' (A + R) f + f' (Jc a0 - aref) over the cone f_n >= 0, |f_t| <= MU f_n, with a0 the
    unconstrained acceleration, A = Jc M^-1 Jc' exact, R_n = (1 - d) / d * Ahat with MuJoCo's
    constant estimate Ahat of A, and R_t = R_n / IMPRATIO.  Like MuJoCo, it leaves out the
    Jdot qdot term.  (The third row, along y, has a zero Jacobian and carries no force.)
    """

    def __init__(self):
        self.ahat = _invweight()
        tc, zeta = SOLREF
        dmax = SOLIMP[1]
        self.k = 1 / (dmax**2 * tc**2 * zeta**2)
        self.b = 2 / (dmax * tc)

    def force(self, th, om, x, v, a_th, a_x, J):
        """The generalized contact force (Q_theta, Q_x), and dist."""
        dist = (-L * np.sin(th) - R_BALL) - (X_FACE + x)
        if dist >= 0:
            return 0.0, 0.0, dist
        x_p = (-L * np.sin(th) - R_BALL + X_FACE + x) / 2
        jc = np.array([[-L * np.cos(th), -1.0], [-x_p, 0.0]])
        d = _impedance(-dist)
        aref = -self.b * (jc @ [om, v]) - np.array([self.k * d * dist, 0.0])
        H = jc @ np.diag([1 / J, 1 / M_S]) @ jc.T
        r_n = max(mujoco.mjMINVAL, (1 - d) / d * self.ahat)
        H += np.diag([r_n, r_n / IMPRATIO])
        g = jc @ [a_th, a_x] - aref
        f = -np.linalg.solve(H, g)
        if not (f[0] >= 0 and abs(f[1]) <= MU * f[0]):
            # on an edge of the cone, f = s (1, +-MU) with s >= 0, or at its apex
            f, best = np.zeros(2), 0.0
            for e in (np.array([1.0, MU]), np.array([1.0, -MU])):
                s = -(e @ g) / (e @ H @ e)
                if s > 0 and 0.5 * s * (e @ g) < best:
                    f, best = s * e, 0.5 * s * (e @ g)
        q = jc.T @ f
        return q[0], q[1], dist


def ground_truth(refs, f_ctrl, dec, J=J0, kp=KP0, kd=KD0, m_s=M_S, k_s=K_S,
                 state0=(0.0, 0.0, 0.0, 0.0), rtol=1e-12, full=False):
    """Continuous-time motion under the held command, sampled at the simulator's physics steps.

    Returns (theta, x) at t = h, 2h, ..., len(refs) * dec * h with h = 1 / (f_ctrl * dec), as an
    array of shape (n, 2); with full=True, the whole state (theta, theta', x, x'), shape (n, 4).
    """
    assert m_s == M_S, "the contact's Ahat is MuJoCo's for the nominal block"
    contact = Contact()
    h = 1.0 / (f_ctrl * dec)
    rows = slice(None) if full else [0, 2]
    out = np.empty((len(refs) * dec, 4 if full else 2))
    y = np.array(state0, dtype=float)
    for k, r in enumerate(refs):
        t0 = k / f_ctrl
        t_eval = t0 + h * np.arange(1, dec + 1)
        t_eval[-1] = (k + 1) / f_ctrl

        def rhs(t, s):
            th, om, x, v = s
            a_th = (-MGL2 * np.sin(th) + kp * (r - th) - kd * om) / J
            a_x = -k_s * x / m_s
            q_th, q_x, _ = contact.force(th, om, x, v, a_th, a_x, J)
            return [om, a_th + q_th / J, v, a_x + q_x / m_s]

        sol = solve_ivp(rhs, (t0, (k + 1) / f_ctrl), y, method="DOP853",
                        t_eval=t_eval, rtol=rtol, atol=rtol)
        out[k * dec:(k + 1) * dec] = sol.y[rows].T
        y = sol.y[:, -1]
    return out


def mujoco_rollout(refs, f_ctrl, dec, arm="euler_explicit", kp=KP0, kd=KD0):
    """MuJoCo at the nominal parameters: (theta, x) after each physics step, shape (n, 2)."""
    return Sim(arm).rollout(refs, f_ctrl, dec, J0, kp, kd)


def first_order_prediction(arm, h):
    """(J*, K_P*, K_D*, d*, lam*) to first order in h; probe_counterterms.py derives and checks them.

    euler_explicit: the PD damping lags, so J* = J0 + h/2 K_D; the command acts h/2 early, so
       d* = h/2; and the contact's damping lags, so its stiffness, damping and regularization
       are all scaled by lam* = 1 - h b / 2.
    rk4_explicit: the torque is held over a step, a delay of h/2 on both feedback terms, as in
       experiment 1; the contact is recomputed at every stage, so it needs no counterterm.
    implicitfast_native: implicit damping, J* = J0 - h/2 K_D and d* = h/2 as in experiment 1.
       The contact damping lags as in Euler, which lam* = 1 - h b / 2 absorbs.  But the contact
       is solved with M, not M + h K_D, so it also sees the rod's acceleration (1 + h K_D / J)
       times too large, a first-order term that no parameter absorbs.
    """
    if arm == "euler_explicit":
        return np.array([J0 + h / 2 * KD0, KP0, KD0, h / 2, 1 - h * B / 2])
    if arm == "rk4_explicit":
        return np.array([J0 + h / 2 * KD0, KP0, KD0 + h / 2 * KP0, 0.0, 1.0])
    return np.array([J0 - h / 2 * KD0, KP0, KD0, h / 2, 1 - h * B / 2])


def contact_params(lam):
    """lam applied in MuJoCo's parameters: solref (timeconst / lam, dampratio sqrt(lam)) and a
    factor on the inverse weights, which scales b, k d and R by lam."""
    return (SOLREF[0] / lam, SOLREF[1] * np.sqrt(lam)), lam


def delayed_command(refs, f_ctrl, dec, delay):
    """Average of the held command r(t - delay) over each physics step [t_n, t_n + h), as
    experiment 1's: r(t) = refs[floor(t f_ctrl)], extended by the first value before 0."""
    T = 1.0 / f_ctrl
    h = T / dec
    cum = np.concatenate([[0.0], np.cumsum(refs) * T])

    def integral(t):
        k = np.clip(np.floor(t / T).astype(int), 0, len(refs) - 1)
        return np.where(t < 0, refs[0] * t, cum[k] + refs[k] * (t - k * T))

    t0 = np.arange(len(refs) * dec) * h - delay
    return (integral(t0 + h) - integral(t0)) / h


class Sim:
    """MuJoCo for one scheme; the parameters (J, K_P, K_D, d, lam) are set per rollout."""

    def __init__(self, arm="euler_explicit"):
        self.arm = arm
        self.model = mujoco.MjModel.from_xml_string(model_xml(arm))
        self.model.opt.integrator = ARM_INTEGRATOR[arm]
        self.data = mujoco.MjData(self.model)
        self.invweight0 = self.model.body_invweight0[:, 0].copy()

    def rollout(self, refs, f_ctrl, dec, J, kp, kd, delay=0.0, lam=1.0):
        """(theta, x) after each physics step, shape (n, 2); None if the rollout diverged."""
        m, d = self.model, self.data
        native = self.arm == "implicitfast_native"
        m.opt.timestep = 1.0 / (f_ctrl * dec)
        m.dof_armature[0] = J - J0  # may be negative; the total inertia J must stay positive
        if native:
            m.actuator_gainprm[0, 0] = kp
            m.actuator_biasprm[0, 1] = -kp
            m.actuator_biasprm[0, 2] = -kd
        solref, scale = contact_params(lam)
        m.geom_solref[:] = solref
        m.body_invweight0[:, 0] = self.invweight0 * scale
        mujoco.mj_resetData(m, d)
        cmd = np.repeat(refs, dec) if delay == 0 else delayed_command(refs, f_ctrl, dec, delay)
        out = np.empty((len(cmd), 2))
        for i, r in enumerate(cmd):
            d.ctrl[0] = r if native else kp * (r - d.qpos[0]) - kd * d.qvel[0]
            mujoco.mj_step(m, d)
            out[i] = d.qpos
        diverged = (d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0
                    or not np.all(np.abs(out) < 1e3))
        return None if diverged else out

    def rollout_many(self, jobs):
        """One trajectory per (refs, f_ctrl, dec, params) job."""
        return [self.rollout(refs, f_ctrl, dec, *p) for refs, f_ctrl, dec, p in jobs]
