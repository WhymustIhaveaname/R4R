"""Experiment 1's pendulum in SuperDex Physics (Meta's Mochi engine).

The simulator has the rollout signature of exp1_pendulum/common.py's Sim, so run_flow.py's sweep,
starting points and fitter run on it unchanged.  Set SUPERDEX_PRECISION=fp64 before the import
for the double-precision engine.

Mochi integrates implicitly.  The three schemes:
  backward_euler_explicit  backward Euler, with the PD torque computed from the state at the
      start of the step and held over it (set_external_forces_on_dofs), as in MuJoCo's
      euler_explicit.
  backward_euler_native  backward Euler, with Mochi's own PD (the pose controller): spring and
      damper are both evaluated at the end of the step.
  bdf2_native  BDF2, with the same PD.
The pose controller damps q' - (r - r_old) / dt, so a held command that jumps would also kick
the damper.  The target is therefore reset every step (reset_articulated_target_pose), which
leaves the plain PD K_P (r - q) - K_D q' that the other engines run.

Mochi's joint inertia (armature) cannot be negative, so a fitted J below the link's own inertia
cannot be run; the link can be made lighter than the rod, with the armature carrying the rest.
"""

import sys
from pathlib import Path

import numpy as np
import superdex.physics as sdp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "exp1_pendulum"))

import common as mj  # noqa: E402

ARMS = ("backward_euler_explicit", "backward_euler_native", "bdf2_native")
ARM_LABEL = {"backward_euler_explicit": "backward Euler + explicit PD",
             "backward_euler_native": "backward Euler + native PD",
             "bdf2_native": "BDF2 + native PD"}
ARM_METHOD = {"backward_euler_explicit": sdp.IntegrationMethod.BACKWARD_EULER,
              "backward_euler_native": sdp.IntegrationMethod.BACKWARD_EULER,
              "bdf2_native": sdp.IntegrationMethod.BDF2}
REAL = np.float64 if sdp.uses_double_precision() else np.float32

_initialized = False


def _initialize():
    """Once per process; a forked worker has to do it itself."""
    global _initialized
    if not _initialized:
        sdp.initialize(num_worker_threads=0)
        _initialized = True


def solver_params(arm, tight):
    """The scheme's integrator; `tight` iterates Newton until its step stalls at round-off.

    Mochi's default stops after at most 4 Newton iterations or at a weighted residual of 1e-3,
    which leaves the step map a few 1e-6 rad away from the backward Euler it solves.
    """
    sp = sdp.SolverParams()
    sp.integration_method = ARM_METHOD[arm]
    if tight:
        nl = sp.non_linear_solver
        nl.abs_tol, nl.rel_tol, nl.max_iter = 0.0, 0.0, 50
        sp.non_linear_solver = nl
    return sp


class SuperDexPendulum:
    """`n` pendulums in one scene, one parameter set each.

    The copies sit at the same place; they have no shapes, so they cannot touch, and each is
    its own island: copy i is bitwise the same as a lone pendulum.
    """

    def __init__(self, arm, n=1, tight=True, link_inertia=None):
        """`link_inertia=None` builds the rod itself: mass M at L/2 plus M L^2/12 about it."""
        _initialize()
        self.arm, self.n = arm, n
        self.scene = sdp.create_scene(f"pendulum_{arm}")
        self.scene.set_gravity([0.0, -mj.G, 0.0])
        self.scene.set_solver_params(solver_params(arm, tight))
        self.rod = link_inertia is None
        self.link_inertia = mj.J0 if self.rod else link_inertia
        self.actors = [self._pendulum(f"p{i}") for i in range(n)]
        self._dof = np.array([0], dtype=np.int32)
        self._buf = sdp.DynamicArrayReal(1)
        self.start = self.scene.capture_state()

    def _pendulum(self, name):
        """A revolute joint about z, hanging along -y, whose link keeps the rod's gravity torque.

        The link is a point mass m at distance c with m c = M L / 2, so its inertia about the
        hinge is m c^2 = self.link_inertia; the armature makes up the rest of J.
        """
        if self.rod:
            mass, c, i_cm = mj.M, mj.L / 2, mj.M * mj.L**2 / 12
        else:
            c = self.link_inertia / (mj.M * mj.L / 2)
            mass, i_cm = mj.M * mj.L / 2 / c, 1e-12
        link = sdp.ArticulatedLinkParams(
            name="rod", parent_link=-1, has_gravity=True, mass=mass,
            center_of_mass=[0.0, -c, 0.0], moment_of_inertia=[i_cm, 0, 0, 1e-12, 0, i_cm])
        params = sdp.ArticulatedActorParams(name=name)
        params.world_from_root = sdp.TransformRT(translation=[0.0, 1.0, 0.0])
        params.joints = [sdp.ArticulatedJointParams(
            name="hinge", type=sdp.ArticulatedJointType.REVOLUTE, axis=[0, 0, 1])]
        params.links = [link]
        actor = self.scene.create_articulated_actor(params)
        if self.arm != "backward_euler_explicit":
            pc = sdp.PoseControllerParams(1)
            pc.joint_tracking[0] = sdp.PoseTrackingParams(stiffness=mj.KP0, damping=mj.KD0)
            actor.add_articulated_pose_controller(pc)
        return actor

    def _write(self, p):
        """Armature and gains per copy; returns which rows can run, and the rows that do run.

        A row whose J is below the link's own cannot be run.  Its copy runs the measured
        parameters instead, since an inertia near zero would blow up the step, and the row is
        reported as failed.
        """
        ok = p[:, 0] >= self.link_inertia
        run = np.where(ok[:, None], p, np.append(mj.PARAM0, 0.0))
        for a, (J, kp, kd) in zip(self.actors, run[:, :3]):
            a.set_articulated_joint_inertia_params([J - self.link_inertia])
            if self.arm != "backward_euler_explicit":
                pc = sdp.PoseControllerParams(1)
                pc.joint_tracking[0] = sdp.PoseTrackingParams(stiffness=kp, damping=kd)
                a.set_articulated_pose_controller_params(pc)
        return ok, run

    def rollout(self, refs, f_ctrl, dec, J, kp, kd, delay=0.0, strict=True):
        """Theta after each physics step, for one parameter set; None if it diverged."""
        out = self.rollout_batch(refs, f_ctrl, dec, [(J, kp, kd, delay)])
        return None if (strict and out[0] is None) else out[0]

    def rollout_batch(self, refs, f_ctrl, dec, params):
        """One trajectory per parameter row; a diverged row is None.

        More rows than copies run as several rollouts.  When Newton diverges, Mochi silently
        puts that copy back where the step started and stops it; such a copy is found by that
        signature and reported as failed.  If no copy shows it, the rollout is redone one row
        at a time.
        """
        asked = np.atleast_2d(np.asarray(params, dtype=float))
        if asked.shape[1] == 3:
            asked = np.hstack([asked, np.zeros((len(asked), 1))])
        if len(asked) > self.n:
            return [t for i in range(0, len(asked), self.n)
                    for t in self.rollout_batch(refs, f_ctrl, dec, asked[i:i + self.n])]
        k = len(asked)
        ok, p = self._write(np.concatenate([asked, np.repeat(asked[:1], self.n - k, axis=0)]))
        h = 1.0 / (f_ctrl * dec)
        cmd = np.stack([np.repeat(refs, dec) if d == 0
                        else mj.delayed_command(refs, f_ctrl, dec, d) for d in p[:, 3]])
        self.scene.restore_state(self.start, release_immediately=False)
        out = np.empty((self.n, cmd.shape[1]))
        prev = np.zeros(self.n)
        unexplained = False
        for j in range(cmd.shape[1]):
            for i, a in enumerate(self.actors):
                if self.arm == "backward_euler_explicit":
                    q, v = self._state(a)
                    tau = p[i, 1] * (cmd[i, j] - q) - p[i, 2] * v
                    a.set_external_forces_on_dofs(self._dof, np.array([tau], dtype=REAL))
                else:
                    a.reset_articulated_target_pose(np.array([cmd[i, j]], dtype=REAL))
            self.scene.step(h)
            for i, a in enumerate(self.actors):
                out[i, j] = self._state(a)[0]
            if (self.scene.get_solver_stats().convergence_status
                    == sdp.ConvergenceStatus.DIVERGED):
                v = np.array([self._state(a)[1] for a in self.actors])
                hit = (v == 0) & (out[:, j] == prev) & (prev != 0)
                ok &= ~hit
                unexplained |= not hit.any()
            prev = out[:, j]
        if unexplained and k > 1:
            return [self.rollout_batch(refs, f_ctrl, dec, asked[i:i + 1])[0] for i in range(k)]
        ok &= (not unexplained) & np.all(np.isfinite(out), axis=1) & (np.max(np.abs(out), axis=1) < 1e3)
        return [out[i] if ok[i] else None for i in range(k)]

    def _state(self, actor):
        actor.get_articulated_pose(self._buf)
        q = float(self._buf[0])
        actor.get_articulated_joint_velocities(self._buf)
        return q, float(self._buf[0])


# The link models.  The rod is the pendulum as built; a lever is a point mass whose inertia
# about the hinge is the given share of J, with the rod's gravity torque, the armature carrying
# the rest of J.  Mochi steps a link in maximal coordinates, which adds a term of order
# h J_link q'^3 that no joint parameter absorbs; the levers shrink it.
LINKS = {"rod": None, "quarter": mj.J0 / 4, "hundredth": mj.J0 / 100}


def first_order_prediction_delay(arm, h):
    """(J*, K_P*, K_D*, d*) to first order in h, from exp1_pendulum/derive_second_order.py.

    backward_euler_explicit has semi-implicit Euler's counterterms.  In backward_euler_native,
    the spring and damper act at the end of the step, which adds h K_P of damping.  Both
    advance the position with the new velocity, so a held command acts h/2 early.  BDF2 has no
    first-order shift of its own, but it differences across the step where the command jumps,
    so a held command acts h/2 late.
    """
    if arm == "backward_euler_explicit":
        return mj.J0 + h / 2 * mj.KD0, mj.KP0, mj.KD0, h / 2
    if arm == "backward_euler_native":
        return mj.J0 - h / 2 * mj.KD0, mj.KP0, mj.KD0 - h * mj.KP0, h / 2
    return mj.J0, mj.KP0, mj.KD0, -h / 2
