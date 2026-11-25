"""The experiment-1 pendulum stepped by Genesis (genesis-world 1.5.0), as shipped: FP32.

Genesis's rigid solver is modelled on MuJoCo's: it reads the same MJCF, steps generalized
coordinates, and two of its integrators are MuJoCo's.  The three schemes, named as MuJoCo's
are where they are MuJoCo's:
  euler_explicit  Euler (semi-implicit), with the PD torque computed from the state at the start
      of each physics step.
  implicitfast_native  implicitfast, with Genesis's own PD (control_dofs_position).
  approximate_implicitfast_native  approximate_implicitfast, Genesis's default, with its own PD.
The two implicit integrators both solve (J + h K_D) a = f; they differ only in how the
constraint solve sees the damping, and this pendulum has no constraint, so they differ only by
the order of the floating-point operations.  Nothing about the system is restated here: the
model XML, the reference solution and the identification come from exp1_pendulum/common.py.

`rollout` mirrors `common.Sim.rollout`, so exp1_pendulum/run_flow.py can sweep either engine
without knowing which one it has.
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "exp1_pendulum"))

import common as mj  # noqa: E402

ARMS = ("euler_explicit", "approximate_implicitfast_native", "implicitfast_native")
ARM_LABEL = {"euler_explicit": "Euler + explicit PD",
             "approximate_implicitfast_native": "approximate_implicitfast + native PD",
             "implicitfast_native": "implicitfast + native PD"}
# MuJoCo's scheme with the same step map
THEORY_ARM = {arm: "implicitfast_native" if "implicitfast" in arm else arm for arm in ARMS}


def first_order_prediction_delay(arm, h):
    return mj.first_order_prediction_delay(THEORY_ARM[arm], h)


_gs = None


def _genesis():
    """Genesis initializes once per process.  The CPU backend runs this 1-DOF model four times
    faster than the GPU, whose kernel launches cost more than the physics."""
    global _gs
    if _gs is None:
        import genesis as gs
        torch.set_num_threads(1)
        gs.init(backend=gs.cpu, precision="32", logging_level="warning")
        _gs = gs
    return _gs


class GenesisPendulum:
    """`n_envs` pendulums at one physics step, each with its own parameters and command.

    The step is fixed when a scene is built, so a rollout at another step builds a new scene
    (0.2 s once the kernels are compiled; the result is bit for bit that of a fresh process).
    The model has no contact, no joint limit and no constraint, so their passes are switched
    off; this leaves every trajectory bit for bit as it is and halves the cost of a step.  The
    state is read and the controls written through zero-copy views of Genesis's arrays rather
    than its getters and setters, which leaves the trajectories bit for bit as they are too.
    """

    def __init__(self, arm, n_envs=32, fast=True):
        self.arm, self.n, self.fast = arm, n_envs, fast
        self.h = self.scene = None

    def _build(self, h):
        gs = _genesis()
        if self.scene is not None:
            self.scene.destroy()
        integrator = {"euler_explicit": gs.integrator.Euler,
                      "approximate_implicitfast_native": gs.integrator.approximate_implicitfast,
                      "implicitfast_native": gs.integrator.implicitfast}[self.arm]
        unused = (dict(enable_collision=False, enable_joint_limit=False, disable_constraint=True)
                  if self.fast else {})
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=h, substeps=1, gravity=(0.0, 0.0, -mj.G)),
            rigid_options=gs.options.RigidOptions(integrator=integrator, batch_dofs_info=True,
                                                  **unused),
            show_viewer=False)
        # Without default_armature=None, Genesis adds a rotor inertia of 0.1 kg m^2, seven times
        # the rod's, to every joint the file leaves without one.
        self.entity = self.scene.add_entity(gs.morphs.MJCF(
            file=mj.model_xml(THEORY_ARM[self.arm]), default_armature=None))
        self.scene.build(n_envs=self.n)
        self.h = h

    def rollout(self, refs, f_ctrl, dec, J, kp, kd, delay=0.0, strict=True):
        """Theta after each physics step, for one parameter set; None if it diverged."""
        out = self.rollout_batch(refs, f_ctrl, dec, [(J, kp, kd, delay)])
        return None if (strict and out[0] is None) else out[0]

    def rollout_batch(self, refs, f_ctrl, dec, params):
        """One trajectory per parameter row, all stepped together.  A diverged row is None."""
        return self.rollout_many([(refs, f_ctrl, dec, p) for p in params])

    def rollout_many(self, jobs):
        """One trajectory per (refs, f_ctrl, dec, params) job, all at one physics step, each
        with its own command.  A diverged job is None."""
        if len(jobs) > self.n:
            return [t for i in range(0, len(jobs), self.n)
                    for t in self.rollout_many(jobs[i:i + self.n])]
        steps = {f_ctrl * dec for _, f_ctrl, dec, _ in jobs}
        if len(steps) > 1:
            raise ValueError(f"jobs at {len(steps)} physics steps; a scene has one")
        h = 1.0 / steps.pop()
        if h != self.h:
            self._build(h)
        p = np.array([np.append(row, [0.0] * (4 - len(row))) for *_, row in jobs], dtype=float)
        p = np.concatenate([p, np.repeat(p[:1], self.n - len(p), axis=0)])
        cmds = [np.repeat(refs, dec) if d == 0 else mj.delayed_command(refs, f_ctrl, dec, d)
                for (refs, f_ctrl, dec, _), d in zip(jobs, p[:, 3])]
        nstep = max(len(c) for c in cmds)
        cmd = np.zeros((nstep, self.n), dtype=np.float32)
        for k, c in enumerate(cmds):
            cmd[:len(c), k] = c
        cmd[:, len(cmds):] = cmd[:, :1]
        theta, bad = (self._step_fast if self.fast else self._step_api)(torch.from_numpy(cmd), p)
        out = theta.numpy().astype(float)
        trajectories = [out[:len(c), k] for k, c in enumerate(cmds)]
        return [None if bad[k] or not (np.all(np.isfinite(t)) and np.max(np.abs(t)) < 1e3)
                else t for k, t in enumerate(trajectories)]

    def _set_params(self, p):
        envs = torch.arange(self.n)
        col = lambda x: torch.tensor(x, dtype=torch.float32)[:, None]
        self.scene.reset()
        self.entity.set_dofs_armature(col(p[:, 0] - mj.J0), envs_idx=envs)  # J may be below J0
        if self.arm != "euler_explicit":
            self.entity.set_dofs_kp(col(p[:, 1]), envs_idx=envs)
            self.entity.set_dofs_kv(col(p[:, 2]), envs_idx=envs)
        return (torch.tensor(p[:, 1], dtype=torch.float32),
                torch.tensor(p[:, 2], dtype=torch.float32))

    def _step_fast(self, cmd, p):
        from genesis.utils.misc import qd_to_torch
        kp, kd = self._set_params(p)
        solver, sim = self.scene.sim.rigid_solver, self.scene.sim
        if self.arm == "euler_explicit":
            self.entity.control_dofs_force(torch.zeros(self.n, 1))
        else:
            self.entity.control_dofs_position(torch.zeros(self.n, 1))
        dofs = solver.dyn_state.dofs
        view = lambda field: qd_to_torch(field, transpose=True, copy=False)[:, 0]
        pos, vel = view(dofs.pos), view(dofs.vel)
        ctrl = view(dofs.ctrl_force if self.arm == "euler_explicit" else dofs.ctrl_pos)
        out = torch.empty_like(cmd)
        bad = torch.zeros(self.n, dtype=torch.bool)
        for i in range(len(cmd)):
            if self.arm == "euler_explicit":
                ctrl.copy_(kp * (cmd[i] - pos) - kd * vel)
            else:
                ctrl.copy_(cmd[i])
            sim.step()
            out[i] = pos
            # A non-finite step freezes its environment and flags it, and the next step would
            # raise; the flag is cleared and the environment written off instead.
            if sim._has_errno_flags:
                flagged = solver.get_error_envs_mask()
                bad |= flagged
                solver.clear_errno(torch.nonzero(flagged)[:, 0])
        return out, bad

    def _step_api(self, cmd, p):
        """The same rollout through Genesis's public getters, setters and Scene.step only; the
        sanity check compares the two."""
        kp, kd = self._set_params(p)
        out = torch.empty_like(cmd)
        for i in range(len(cmd)):
            if self.arm == "euler_explicit":
                q = self.entity.get_dofs_position()[:, 0]
                v = self.entity.get_dofs_velocity()[:, 0]
                self.entity.control_dofs_force((kp * (cmd[i] - q) - kd * v)[:, None])
            else:
                self.entity.control_dofs_position(cmd[i][:, None])
            self.scene.step(update_visualizer=False)
            out[i] = self.entity.get_dofs_position()[:, 0]
        return out, torch.zeros(self.n, dtype=torch.bool)

    def mass(self):
        """The joint's total inertia as Genesis assembles it, per environment."""
        return self.entity.get_mass_mat()[:, 0, 0].numpy().astype(float)
