"""The experiment-1 pendulum in Isaac Sim (PhysX), built to the same inertial specification.

Run inside the Isaac Sim venv; `SimulationApp` must be constructed before any other import.
"""

import numpy as np

# Same uniform rod and gains as the MuJoCo experiments (SI units).
M = 0.45359237  # 1 lb
L = 0.3048  # 1 ft
G = 9.81
J0 = M * L**2 / 3
KP0, KD0 = 80.0, 2.0
F_CTRL = 50

ARMS = ("A", "B", "C")
ARM_LABEL = {
    "A": "A: TGS + explicit PD",
    "B": "B: TGS + PhysX joint drive",
    "C": "C: PGS + PhysX joint drive",
}


# Each scheme's own first-order prediction, from exp1_pendulum/derive_second_order.py.  An
# explicitly held torque makes PhysX behave as semi-implicit Euler; its own drive is an implicit
# spring whose damping is K_D + delta K_P, run at delta = h / 32 under TGS and delta = h under PGS.
TGS_ITERS = 32
SUBSTEPS = {"A": 1, "B": TGS_ITERS, "C": 1}
HELD_TORQUE = {"A": True, "B": False, "C": False}


def first_order_prediction(arm, h, with_delay=True):
    """(J, K_P, K_D[, d]) to first order in h, J being the total inertia."""
    d = h / SUBSTEPS[arm]
    sign = +1 if HELD_TORQUE[arm] else -1
    p = np.array([J0 + sign * d / 2 * KD0, KP0, KD0 - (0.0 if HELD_TORQUE[arm] else d * KP0)])
    if with_delay:
        return np.append(p, d / 2)
    p[2] += sign * d / 2 * KP0  # without a delay operator the advance lands on K_D
    return p


def build_scene(n_envs, arm, spacing=1.0):
    """One pendulum per environment, all identical; returns the articulation view."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
    import omni.usd
    from isaacsim.core.prims import Articulation

    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(G)
    px = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene"))
    px.CreateSolverTypeAttr().Set("PGS" if arm == "C" else "TGS")
    px.CreateEnableGPUDynamicsAttr().Set(False)  # 1-2 DOF; the CPU pipeline is faster here
    px.CreateEnableStabilizationAttr().Set(False)

    for i in range(n_envs):
        root = f"/World/env{i}"
        UsdGeom.Xform.Define(stage, root)
        UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(root))
        PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(root))

        # PhysX needs a base link fixed to the world; the rod then hinges off it.
        base = f"{root}/base"
        bx = UsdGeom.Xform.Define(stage, base)
        bx.AddTranslateOp().Set(Gf.Vec3d(i * spacing, 0.0, 0.0))
        bprim = stage.GetPrimAtPath(base)
        UsdPhysics.RigidBodyAPI.Apply(bprim)
        bmass = UsdPhysics.MassAPI.Apply(bprim)
        bmass.CreateMassAttr().Set(1e-6)
        bmass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(1e-9, 1e-9, 1e-9))
        anchor = UsdPhysics.FixedJoint.Define(stage, f"{root}/anchor")
        anchor.CreateBody0Rel().SetTargets([])
        anchor.CreateBody1Rel().SetTargets([base])
        anchor.CreateLocalPos0Attr().Set(Gf.Vec3f(i * spacing, 0.0, 0.0))
        anchor.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

        link = f"{root}/link"
        xform = UsdGeom.Xform.Define(stage, link)
        xform.AddTranslateOp().Set(Gf.Vec3d(i * spacing, 0.0, 0.0))
        prim = stage.GetPrimAtPath(link)
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr().Set(M)
        mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, -L / 2))
        mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(M * L**2 / 12, M * L**2 / 12, 1e-9))
        mass.CreatePrincipalAxesAttr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

        rod = UsdGeom.Capsule.Define(stage, f"{link}/geom")
        rod.CreateAxisAttr().Set("Z")
        rod.CreateHeightAttr().Set(L)
        rod.CreateRadiusAttr().Set(0.008)
        rod.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -L / 2))
        rod.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)  # visual only, no collision

        joint = UsdPhysics.RevoluteJoint.Define(stage, f"{root}/hinge")
        joint.CreateBody0Rel().SetTargets([base])
        joint.CreateBody1Rel().SetTargets([link])
        joint.CreateAxisAttr().Set("Y")
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        jp = joint.GetPrim()
        PhysxSchema.PhysxJointAPI.Apply(jp)
        drive = UsdPhysics.DriveAPI.Apply(jp, "angular")
        drive.CreateTypeAttr().Set("force")
        drive.CreateMaxForceAttr().Set(1e6)
        drive.CreateStiffnessAttr().Set(0.0)
        drive.CreateDampingAttr().Set(0.0)
        drive.CreateTargetPositionAttr().Set(0.0)

    return Articulation(prim_paths_expr="/World/env.*", name="pendulums")


def delayed_command(refs, f_ctrl, dec, delays):
    """Per-environment command: average of the held r(t - delay) over each physics step.

    `delays` is one value per environment (seconds, may be negative); returns
    (n_envs, len(refs) * dec). delay = 0 reproduces the held command exactly.
    """
    T = 1.0 / f_ctrl
    h = T / dec
    cum = np.concatenate([[0.0], np.cumsum(refs) * T])

    def integral(t):
        k = np.clip(np.floor(t / T).astype(int), 0, len(refs) - 1)
        return np.where(t < 0, refs[0] * t, cum[k] + refs[k] * (t - k * T))

    t0 = np.arange(len(refs) * dec)[None, :] * h - np.asarray(delays, float)[:, None]
    return (integral(t0 + h) - integral(t0)) / h


def set_step(world, h):
    """Set the physics step to h, which must be 1 / (an integer).

    Isaac stores the step as int(1 / dt) steps per second, and int() truncates: 1 / (1 / 210)
    is 209.99999999999997, so asking for 1 / 210 s runs at 209 Hz.  Asking for a step a little
    shorter keeps the integer rate exact.
    """
    rate = round(1.0 / h)
    if abs(rate * h - 1.0) > 1e-9:
        raise ValueError(f"Isaac can only step at an integer rate, not 1 / {1.0 / h} s")
    dt = 1.0 / (rate + 0.5)
    world.set_simulation_dt(physics_dt=dt, rendering_dt=dt)
    assert world.get_physics_dt() == 1.0 / rate


class IsaacPendulum:
    """Steps n_envs copies of the pendulum, one parameter set per environment."""

    def __init__(self, world, arm, n_envs):
        self.world, self.arm, self.n = world, arm, n_envs
        self.view = build_scene(n_envs, arm)
        world.scene.add(self.view)
        world.reset()
        self.view.initialize()
        self.stale = True           # reset once more before the first rollout

    def _apply(self, params):
        """Write the parameters, after any world.reset(), which restores USD values.

        Fewer rows than environments is allowed: the rest are padded with the first row and
        their output is discarded, so a short batch costs no more than a full one.
        """
        p = np.atleast_2d(np.asarray(params, dtype=np.float32))
        if len(p) < self.n:
            p = np.concatenate([p, np.repeat(p[:1], self.n - len(p), axis=0)])
        self.view.set_armatures(p[:, :1])
        if self.arm == "A":  # torques come from Python, so the PhysX drive stays off
            z = np.zeros((self.n, 1), np.float32)
            self.view.set_gains(kps=z, kds=z)
        else:
            self.view.set_gains(kps=p[:, 1:2], kds=p[:, 2:3])
        return p

    def set_dt(self, h):
        set_step(self.world, h)

    def rollout(self, refs, dec, params, f_ctrl=F_CTRL, every_step=False):
        """Joint angle after each control step, shape (len(params), len(refs)), or after each
        physics step with `every_step`, shape (len(params), len(refs) * dec).

        `params` is rows of (armature, K_P, K_D[, delay]); every row runs the same held command,
        delayed by its own delay.
        """
        p = np.atleast_2d(np.asarray(params, dtype=float))
        delays = p[:, 3] if p.shape[1] > 3 else np.zeros(len(p))
        out = self.rollout_cmd(delayed_command(refs, f_ctrl, dec, delays), p)
        return out if every_step else out[:, dec - 1::dec]

    def rollout_cmd(self, cmd, params):
        """Joint angle after each physics step, shape (len(params), cmd.shape[1]).

        Row i of `params`, (armature, K_P, K_D[, ...]), runs the per-step command cmd[i].  More
        rows than environments run as several rollouts.
        """
        cmd, p = np.atleast_2d(cmd), np.atleast_2d(np.asarray(params, dtype=float))
        if len(p) > self.n:
            return np.concatenate([self.rollout_cmd(cmd[i:i + self.n], p[i:i + self.n])
                                   for i in range(0, len(p), self.n)])
        k = len(p)
        if self.stale:              # a new physics step leaves the tensor views to rebuild
            self.world.reset()
            self.stale = False
        p = self._apply(p)
        cmd = np.concatenate([cmd, np.repeat(cmd[:1], self.n - k, axis=0)]).astype(np.float32)
        zeros = np.zeros((self.n, 1), dtype=np.float32)
        self.view.set_joint_positions(zeros)
        self.view.set_joint_velocities(zeros)
        out = np.empty((self.n, cmd.shape[1]))
        for j in range(cmd.shape[1]):
            target = cmd[:, j:j + 1]
            if self.arm == "A":
                q = self.view.get_joint_positions()
                v = self.view.get_joint_velocities()
                self.view.set_joint_efforts(
                    (p[:, 1:2] * (target - q) - p[:, 2:3] * v).astype(np.float32))
            else:
                self.view.set_joint_position_targets(target)
            self.world.step(render=False)
            out[:, j] = self.view.get_joint_positions().reshape(self.n)
        return out[:k]


class IsaacSim:
    """The pendulum with MuJoCo's rollout signature (exp1_pendulum/common.py, Sim), so that
    MuJoCo's sweeps run on it unchanged: total inertia J, the angle after every physics step,
    and None for a rollout that blew up."""

    def __init__(self, world, arm, n_envs):
        self.pend, self.h = IsaacPendulum(world, arm, n_envs), None

    def rollout(self, refs, f_ctrl, dec, J, kp, kd, delay=0.0, strict=True):
        return self.rollout_many([(refs, f_ctrl, dec, (J, kp, kd, delay))])[0]

    def rollout_batch(self, refs, f_ctrl, dec, rows):
        return self.rollout_many([(refs, f_ctrl, dec, row) for row in rows])

    def rollout_many(self, jobs):
        """(refs, f_ctrl, dec, (J, K_P, K_D[, delay])) jobs at one physics step, each with its
        own command, all in the same rollout: one environment per job."""
        steps = {f_ctrl * dec for _, f_ctrl, dec, _ in jobs}
        if len(steps) != 1:
            raise ValueError(f"one rollout steps at one rate, not at {sorted(steps)} Hz")
        h = 1.0 / steps.pop()
        if h != self.h:
            set_step(self.pend.world, h)
            self.h, self.pend.stale = h, True
        P = np.array([np.append(row, [0.0] * (4 - len(row))) for *_, row in jobs], float)
        cmd = np.stack([delayed_command(refs, f_ctrl, dec, [d])[0]
                        for (refs, f_ctrl, dec, _), d in zip(jobs, P[:, 3])])
        P[:, 0] -= J0                                    # Isaac takes the rotor inertia
        out = self.pend.rollout_cmd(cmd, P)
        ok = np.all(np.isfinite(out), axis=1) & (np.max(np.abs(out), axis=1) < 1e3)
        return [out[k] if ok[k] else None for k in range(len(P))]
