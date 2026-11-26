"""The slider stepped by Genesis (genesis-world 1.5.0), FP32, in the two schemes it shares with MuJoCo:
Euler with an explicit PD, and implicitfast with Genesis's own PD (control_dofs_position).

Genesis reads the same MJCF as MuJoCo, but it resolves this contact differently unless told
otherwise, so each difference is set to MuJoCo's here:
  - the elliptic cone, whose contact Genesis resolves its own way ('signorini') unless asked for
    MuJoCo's ('convex'), with MuJoCo's impratio of 1 instead of Genesis's 100;
  - MuJoCo's inverse weights, which set the contact's regularization R: Genesis computes its own
    (1 / (3 m) for the block where MuJoCo has 1 / m) and recomputes them whenever an armature
    changes, and it has no setter for them, so they are written into its arrays directly;
  - no default rotor inertia (default_armature=None), as in experiment 1.
The friction is common.MU = 0.01 in both engines, the least Genesis allows.  Set up this way,
Genesis follows MuJoCo to float32 round-off in both schemes.  Its implicitfast, like MuJoCo's,
solves the contact with M and only then makes the damping implicit, (M + h D) a' = M a.  Its
default, approximate_implicitfast, solves the contact with M + h D, and so differs from both.

The contact's parameters are one per geom, not per environment, so each environment holds its
own copy of the model (a heterogeneous entity of identical variants), and its geoms carry its
own contact parameters; this leaves every trajectory bit for bit as it is.

`rollout_many` mirrors common.Sim's, so run_flow.py sweeps either engine.
"""

import numpy as np
import torch

import common as c

_gs = None


def _genesis():
    """Genesis initializes once per process, on the CPU, as experiment 1's sweep does."""
    global _gs
    if _gs is None:
        import genesis as gs
        torch.set_num_threads(1)
        gs.init(backend=gs.cpu, precision="32", logging_level="warning")
        _gs = gs
    return _gs


class GenesisSlider:
    """`n_envs` sliders at one physics step, each with its own parameters and command."""

    def __init__(self, arm="euler_explicit", n_envs=32):
        self.arm, self.n = arm, n_envs
        self.native = arm == "implicitfast_native"
        self.h = self.scene = None
        self.invweight0 = None

    def _build(self, h):
        import mujoco
        from genesis.utils.misc import qd_to_torch
        gs = _genesis()
        if self.scene is not None:
            self.scene.destroy()
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=h, substeps=1, gravity=(0.0, 0.0, -c.G)),
            rigid_options=gs.options.RigidOptions(
                integrator=(gs.integrator.implicitfast if self.native else gs.integrator.Euler),
                batch_dofs_info=True, batch_links_info=True,
                friction_cone=gs.friction_cone.elliptic, impratio=c.IMPRATIO,
                contact_resolution=gs.contact_resolution.convex),
            show_viewer=False)
        self.entity = self.scene.add_entity(
            [gs.morphs.MJCF(file=c.model_xml(self.arm), default_armature=None)
             for _ in range(self.n)])
        self.scene.build(n_envs=self.n)
        solver = self.scene.sim.rigid_solver
        # The ball of every variant, then the block of every variant; variant k is environment k.
        links = [g.link.name for g in self.entity.geoms]
        assert links == ["rod"] * self.n + ["slider"] * self.n, links
        self.sol_params = solver.get_sol_params().clone()
        self.invweight = qd_to_torch(solver.dyn_info.links.invweight, transpose=True, copy=False)
        dofs = solver.dyn_state.dofs
        view = lambda field: qd_to_torch(field, transpose=True, copy=False)
        self.pos, self.vel = view(dofs.pos), view(dofs.vel)
        self.ctrl = view(dofs.ctrl_pos if self.native else dofs.ctrl_force)
        m = mujoco.MjModel.from_xml_string(c.model_xml(self.arm))
        self.invweight0 = torch.tensor(m.body_invweight0[1:, 0], dtype=self.invweight.dtype)
        self.h = h

    def _set_params(self, p):
        envs = torch.arange(self.n)
        self.scene.reset()
        armature = np.zeros((self.n, 2))
        armature[:, 0] = p[:, 0] - c.J0   # J may be below J0
        self.entity.set_dofs_armature(torch.tensor(armature, dtype=torch.float32), envs_idx=envs)
        if self.native:
            col = lambda x: torch.tensor(x, dtype=torch.float32)[:, None]
            self.entity.set_dofs_kp(col(p[:, 1]), dofs_idx_local=[0], envs_idx=envs)
            self.entity.set_dofs_kv(col(p[:, 2]), dofs_idx_local=[0], envs_idx=envs)
        solref, scale = zip(*(c.contact_params(lam) for lam in p[:, 4]))
        sp = self.sol_params.clone()
        sp[:, :2] = torch.tensor(np.tile(np.array(solref), (2, 1)), dtype=sp.dtype)
        self.scene.sim.rigid_solver.set_sol_params(sp, geoms_idx=torch.arange(len(sp)))
        # after the armature, which makes Genesis recompute its own
        self.invweight[:, 1:, 0] = self.invweight0 * torch.tensor(scale, dtype=sp.dtype)[:, None]
        return (torch.tensor(p[:, 1], dtype=torch.float32),
                torch.tensor(p[:, 2], dtype=torch.float32))

    def rollout_many(self, jobs):
        """One trajectory (theta, x) per (refs, f_ctrl, dec, params) job, all at one physics
        step, each with its own command and parameters (J, K_P, K_D, d, lam).  A diverged job
        is None."""
        if len(jobs) > self.n:
            return [t for i in range(0, len(jobs), self.n)
                    for t in self.rollout_many(jobs[i:i + self.n])]
        steps = {f_ctrl * dec for _, f_ctrl, dec, _ in jobs}
        if len(steps) > 1:
            raise ValueError(f"jobs at {len(steps)} physics steps; a scene has one")
        h = 1.0 / steps.pop()
        if h != self.h:
            self._build(h)
        p = np.array([np.append(row, [0.0, 1.0][len(row) - 3:]) for *_, row in jobs], dtype=float)
        p = np.concatenate([p, np.repeat(p[:1], self.n - len(p), axis=0)])
        cmds = [np.repeat(refs, dec) if d == 0 else c.delayed_command(refs, f_ctrl, dec, d)
                for (refs, f_ctrl, dec, _), d in zip(jobs, p[:, 3])]
        nstep = max(len(x) for x in cmds)
        cmd = np.zeros((nstep, self.n), dtype=np.float32)
        for k, x in enumerate(cmds):
            cmd[:len(x), k] = x
        cmd[:, len(cmds):] = cmd[:, :1]
        cmd = torch.from_numpy(cmd)
        kp, kd = self._set_params(p)
        solver, sim = self.scene.sim.rigid_solver, self.scene.sim
        self.entity.control_dofs_force(torch.zeros(self.n, 2))
        if self.native:
            self.entity.control_dofs_position(torch.zeros(self.n, 1), dofs_idx_local=[0])
        out = torch.empty((nstep, self.n, 2))
        bad = torch.zeros(self.n, dtype=torch.bool)
        for i in range(nstep):
            if self.native:
                self.ctrl[:, 0] = cmd[i]
            else:
                self.ctrl[:, 0] = kp * (cmd[i] - self.pos[:, 0]) - kd * self.vel[:, 0]
                self.ctrl[:, 1] = 0.0
            sim.step()
            out[i] = self.pos
            # A non-finite step freezes its environment and flags it, and the next step would
            # raise; the flag is cleared and the environment written off instead.
            if sim._has_errno_flags:
                flagged = solver.get_error_envs_mask()
                bad |= flagged
                solver.clear_errno(torch.nonzero(flagged)[:, 0])
        out = out.numpy().astype(float)
        trajectories = [out[:len(x), k] for k, x in enumerate(cmds)]
        return [None if bad[k] or not (np.all(np.isfinite(t)) and np.max(np.abs(t)) < 1e3)
                else t for k, t in enumerate(trajectories)]
