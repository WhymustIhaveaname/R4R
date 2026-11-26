"""First-order counterterms of the three schemes, checked by the order of the error.

The true motion obeys M q'' = F(q, q') + Jc' f, where F holds the PD torque, gravity and the
block's spring, M = diag(J, m_s), and the contact force f minimizes
1/2 f' (A + R) f + f' (Jc a0 - aref) over the friction cone.  Equivalently, Jc a + R f - aref
lies in the cone's normal cone at f, with a the acceleration and aref = -b Jc q' - k d r.
Scaling aref and R by a factor lam > 0 leaves a normal cone as it is, so a step whose condition
reads lam (Jc a) + R f - aref is the true one with aref and R divided by lam.

Euler + explicit PD.  The step is v' = v + h M^-1 [F(q, v) + Jc' f],  q' = q + h v'.  The
velocity it carries lags the true one by h/2 q''.  In the torque this makes
    J* = J0 + h/2 K_D,   d* = h/2      (as in experiment 1).
In aref it adds h/2 b Jc a, so the condition reads (1 - h b / 2) Jc a + R f - aref:
    lam* = 1 - h b / 2,   b = 2 / (dmax timeconst),
applied as timeconst / lam, dampratio sqrt(lam), invweight lam.

RK4 + explicit PD.  The torque is computed at the start of the step and held through the
stages; gravity, the spring and the contact are recomputed at every stage, to fourth order.
Holding the torque delays both PD terms by h/2, as in experiment 1:
    J* = J0 + h/2 K_D,   K_D* = K_D + h/2 K_P.
The contact sees the acceleration and the force the step integrates, so it needs no
counterterm, and neither does the command (d* = 0).

implicitfast + native PD.  The step is (M + h D) a = F(q, v) + Jc' f, D = diag(K_D, 0).  The
velocity lags as in Euler, so
    J* = J0 - h/2 K_D,   d* = h/2      (as in experiment 1),
and the lagging contact damping gives (1 - h b / 2) Jc a again.  But f is solved with M, not
M + h D: the solver's acceleration is M^-1 (F + Jc' f) = a + h M^-1 D a, whose rod component
is (1 + h K_D / J) times the step's.  The extra term h K_D / J Jc_theta a_theta is not a
multiple of Jc a, so lam cannot absorb it, and no other parameter enters the contact's
condition alone.  This scheme has a first-order error that no counterterm removes.

In none of the three does K_P, the block's mass or its spring get a counterterm.

Two checks, at h = 0.5, 0.25 and 0.125 ms, measured parameters against renormalized:
  A. no contact, the block released from 3 cm: the counterterms make the error second order;
  B. contact: so they do in Euler and RK4.  In implicitfast the error stays first order
     whatever lam is: lam = 1 - h b / 2, 1 - h (b - K_D / J) / 2, and 1 are shown.

Usage:  uv run python probe_counterterms.py
"""

import mujoco
import numpy as np

import common as c

F_CTRL, T_RUN = 50, 1.5
DECS = (40, 80, 160)


def rollout(arm, refs, dec, p, contact=True, x0=0.0):
    J, kp, kd, delay, lam = p
    h = 1.0 / (F_CTRL * dec)
    native = arm == "implicitfast_native"
    m = mujoco.MjModel.from_xml_string(c.model_xml(arm))
    m.opt.integrator = c.ARM_INTEGRATOR[arm]
    m.opt.timestep = h
    if not contact:
        m.geom_contype[:] = 0
        m.geom_conaffinity[:] = 0
    m.dof_armature[0] = J - c.J0
    solref, scale = c.contact_params(lam)
    m.geom_solref[:] = solref
    m.body_invweight0[:, 0] *= scale
    if native:
        m.actuator_gainprm[0, 0] = kp
        m.actuator_biasprm[0, 1] = -kp
        m.actuator_biasprm[0, 2] = -kd
    d = mujoco.MjData(m)
    d.qpos[1] = x0
    if arm != "rk4_explicit":  # the carried velocity is the backward difference: u - h/2 q''
        d.qvel[1] = h / 2 * c.K_S * x0 / c.M_S
    out = []
    for r in c.delayed_command(refs, F_CTRL, dec, delay):
        d.ctrl[0] = r if native else kp * (r - d.qpos[0]) - kd * d.qvel[0]
        mujoco.mj_step(m, d)
        out.append(d.qpos.copy())
    return np.array(out)


def renormalized(arm, h, lam=None):
    p = c.first_order_prediction(arm, h)
    if lam is not None:
        p[4] = lam
    return p


def check(label, cases, contact=True, x0=0.0):
    print(label)
    refs = c.ref_sequence(F_CTRL, t_end=T_RUN)
    gts = {dec: c.ground_truth(refs, F_CTRL, dec, state0=(0, 0, x0, 0)) for dec in DECS}
    nominal = np.array([c.J0, c.KP0, c.KD0, 0.0, 1.0])
    for name, arm, lam in cases:
        print(f"  {name}")
        prev = None
        for dec in DECS:
            h = 1.0 / (F_CTRL * dec)
            lam_h = None if lam is None else lam(h)
            errs = [np.abs(rollout(arm, refs, dec, p, contact, x0) - gts[dec]).max(axis=0)
                    for p in (nominal, renormalized(arm, h, lam_h))]
            ratio = "" if prev is None else f"   ratio {prev[0] / errs[1][0]:.1f}, {prev[1] / errs[1][1]:.1f}"
            print(f"    h = {h * 1e3:.4f} ms: measured θ {errs[0][0]:.2e} x {errs[0][1]:.2e};"
                  f"  renormalized θ {errs[1][0]:.2e} x {errs[1][1]:.2e}{ratio}", flush=True)
            prev = errs[1]


lag = lambda h: 1 - h * c.B / 2
cases_a = [(c.ARM_LABEL[a], a, None if a != "implicitfast_native" else lag) for a in c.ARMS]
cases_b = [(c.ARM_LABEL["euler_explicit"], "euler_explicit", None),
           (c.ARM_LABEL["rk4_explicit"], "rk4_explicit", None)]
cases_b += [(f"{c.ARM_LABEL['implicitfast_native']}, lam = {name}", "implicitfast_native", lam)
            for name, lam in (("1 - h b / 2", lag),
                              ("1 - h (b - K_D / J) / 2", lambda h: 1 - h * (c.B - c.KD0 / c.J0) / 2),
                              ("1", lambda h: 1.0))]
force = c.Contact.force
c.Contact.force = lambda self, *a: (0.0, 0.0, 1.0)
check("A. no contact, block released from 3 cm", cases_a, contact=False, x0=0.03)
c.Contact.force = force
check("B. contact", cases_b)
