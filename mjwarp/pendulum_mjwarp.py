"""The experiment-1 pendulum stepped by MuJoCo Warp.

MuJoCo Warp is MuJoCo reimplemented in NVIDIA Warp, and it is the solver Newton runs on, so
this is the same discretization reached through a different implementation.  Nothing about the
system is restated here: the model XML, the integrators, the reference solution and the
identification all come from exp1_pendulum/common.py, and only the stepping is new.

`rollout` deliberately mirrors `common.Sim.rollout`, so exp1_pendulum/run_flow.py can sweep
either engine without knowing which one it has.
"""

import sys
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "exp1_pendulum"))

import common as mj  # noqa: E402


# Reading the state back to Python every step costs more than the physics does, so the control
# law and the recording both run on the device and only the finished trajectory is read back.
# The step counter lives on the device too, so the whole loop can be captured as a CUDA graph.
# Without that the kernel launches cost sixteen times more than the physics for a 1-DOF model.
@wp.kernel
def _explicit_pd(qpos: wp.array2d(dtype=wp.float32), qvel: wp.array2d(dtype=wp.float32),
                 cmd: wp.array2d(dtype=wp.float32), gain: wp.array2d(dtype=wp.float32),
                 idx: wp.array(dtype=wp.int32), ctrl: wp.array2d(dtype=wp.float32)):
    w = wp.tid()
    ctrl[w, 0] = gain[w, 0] * (cmd[w, idx[0]] - qpos[w, 0]) - gain[w, 1] * qvel[w, 0]


@wp.kernel
def _held_command(cmd: wp.array2d(dtype=wp.float32), idx: wp.array(dtype=wp.int32),
                  ctrl: wp.array2d(dtype=wp.float32)):
    w = wp.tid()
    ctrl[w, 0] = cmd[w, idx[0]]


@wp.kernel
def _record(qpos: wp.array2d(dtype=wp.float32), idx: wp.array(dtype=wp.int32),
            out: wp.array2d(dtype=wp.float32)):
    w = wp.tid()
    out[w, idx[0]] = qpos[w, 0]


@wp.kernel
def _advance(idx: wp.array(dtype=wp.int32)):
    idx[0] = idx[0] + 1


class WarpPendulum:
    """One pendulum per world; a batch of parameter sets is stepped in a single pass.

    `nworld` worlds are allocated once.  A rollout that is given fewer parameter sets pads the
    rest with the first one and discards their output, which costs nothing extra on the GPU.
    Each world also has its own time step and command, so jobs at different physics steps
    share a rollout too (`mixed_steps`): the GPU charges by the step, not by the world.
    """

    mixed_steps = True

    CHUNK = 100     # steps per captured graph; the graph is replayed to reach the full length

    def __init__(self, arm, nworld=1, nstep_max=16500):
        self.arm, self.n, self.nmax = arm, nworld, nstep_max
        self.mjm = mujoco.MjModel.from_xml_string(mj.model_xml(arm))
        self.mjm.opt.integrator = mj.ARM_INTEGRATOR[arm]
        self.mjd = mujoco.MjData(self.mjm)
        batched = {"dof_armature": nworld}
        # the native position actuator carries K_P and K_D in its gains
        if arm == "implicitfast_native":
            batched |= {"actuator_gainprm": nworld, "actuator_biasprm": nworld}
        self.m = mjw.put_model(self.mjm, batch_sizes=batched)
        # MuJoCo Warp reads the step as timestep[world % len(timestep)], so one per world
        self.m.opt.timestep = wp.zeros(nworld, dtype=self.m.opt.timestep.dtype)
        self.d = mjw.put_data(self.mjm, self.mjd, nworld=nworld)
        self.cmd = wp.zeros((nworld, nstep_max + self.CHUNK), dtype=wp.float32)
        self.gain = wp.zeros((nworld, 2), dtype=wp.float32)
        self.out = wp.zeros((nworld, nstep_max + self.CHUNK), dtype=wp.float32)
        self.idx = wp.zeros(1, dtype=wp.int32)
        self.graph = None

    def _write(self, params):
        """params: (k, 4) rows of (J, K_P, K_D, delay); returns the padded (n, 4) array."""
        p = np.atleast_2d(np.asarray(params, dtype=float))
        if len(p) > self.n:
            raise ValueError(f"{len(p)} parameter sets but only {self.n} worlds; the fitter "
                             f"runs a whole generation at once, so nworld must be at least "
                             f"fitting.POPSIZE")
        if p.shape[1] == 3:
            p = np.hstack([p, np.zeros((len(p), 1))])
        if len(p) < self.n:
            p = np.concatenate([p, np.repeat(p[:1], self.n - len(p), axis=0)])
        self.m.dof_armature.assign((p[:, :1] - mj.J0).astype(np.float32))
        if self.arm == "implicitfast_native":
            gain = self.m.actuator_gainprm.numpy().copy()
            bias = self.m.actuator_biasprm.numpy().copy()
            gain[:, 0, 0] = p[:, 1]
            bias[:, 0, 1], bias[:, 0, 2] = -p[:, 1], -p[:, 2]
            self.m.actuator_gainprm.assign(gain)
            self.m.actuator_biasprm.assign(bias)
        return p

    def rollout(self, refs, f_ctrl, dec, J, kp, kd, delay=0.0, strict=True):
        """Theta after each physics step, for one parameter set; None if it diverged."""
        out = self.rollout_batch(refs, f_ctrl, dec, [(J, kp, kd, delay)])
        return None if (strict and out[0] is None) else out[0]

    def rollout_batch(self, refs, f_ctrl, dec, params):
        """One trajectory per parameter row, all stepped together.  A diverged row is None."""
        return self.rollout_many([(refs, f_ctrl, dec, p) for p in params])

    def rollout_many(self, jobs):
        """One trajectory per (refs, f_ctrl, dec, params) job, each world at its own physics step
        with its own command, all stepped together until the longest is done.  A diverged job
        is None."""
        if len(jobs) > self.n:
            return [t for i in range(0, len(jobs), self.n)
                    for t in self.rollout_many(jobs[i:i + self.n])]
        p = self._write([row for *_, row in jobs])
        h = np.array([1.0 / (f_ctrl * dec) for _, f_ctrl, dec, _ in jobs])
        self.m.opt.timestep.assign(np.append(h, np.full(self.n - len(h), h[0])).astype(np.float32))
        zeros = np.zeros((self.n, 1), dtype=np.float32)
        self.d.qpos.assign(zeros)
        self.d.qvel.assign(zeros)
        # The command builder from common.py is reused per job, so the two engines see exactly
        # the same sequence of held commands.
        cmds = [np.repeat(refs, dec) if d == 0 else mj.delayed_command(refs, f_ctrl, dec, d)
                for (refs, f_ctrl, dec, _), d in zip(jobs, p[:, 3])]
        nstep = max(len(c) for c in cmds)
        if nstep > self.nmax:
            raise ValueError(f"{nstep} steps exceeds the {self.nmax} this instance allocated")
        padded = np.zeros((self.n, self.cmd.shape[1]), dtype=np.float32)
        for k, c in enumerate(cmds):
            padded[k, :len(c)] = c
        self.cmd.assign(padded)
        self.gain.assign(p[:, 1:3].astype(np.float32))
        self.idx.zero_()
        for _ in range((nstep + self.CHUNK - 1) // self.CHUNK):
            wp.capture_launch(self._captured())
        out = self.out.numpy()
        trajectories = [out[k, :len(c)].astype(float) for k, c in enumerate(cmds)]
        return [t if np.all(np.isfinite(t)) and np.max(np.abs(t)) < 1e3 else None
                for t in trajectories]

    def _captured(self):
        """A graph of CHUNK steps; the step index advances on the device so one graph serves
        every time step and every command."""
        if self.graph is None:
            for _ in range(3):                       # make sure every kernel is compiled
                self._one_step()
            wp.synchronize()
            with wp.ScopedCapture() as cap:
                for _ in range(self.CHUNK):
                    self._one_step()
            self.graph = cap.graph
        return self.graph

    def _one_step(self):
        if self.arm == "implicitfast_native":
            wp.launch(_held_command, dim=self.n, inputs=[self.cmd, self.idx, self.d.ctrl])
        else:
            wp.launch(_explicit_pd, dim=self.n,
                      inputs=[self.d.qpos, self.d.qvel, self.cmd, self.gain, self.idx,
                              self.d.ctrl])
        mjw.step(self.m, self.d)
        wp.launch(_record, dim=self.n, inputs=[self.d.qpos, self.idx, self.out])
        wp.launch(_advance, dim=1, inputs=[self.idx])
