# Experiment 4: tofu and jelly (notes, 2026-11-25)

Notes taken before any code. Versions: MuJoCo 4.0.0 and Genesis 1.5.0, the latest on 2026-11-25.

## Goal

A soft solid, such as tofu or jelly, handled by a robot.

1. Elastic response: it jiggles and bounces.
2. Breakage under load: a knife presses into it, or a gripper squeezes it. The simulator only has to say when it breaks. At the onset it gives a "broken" signal, and the episode stops. Nothing after the break is simulated.

## A "broken" signal needs no fracture model

The episode stops at the onset. So the simulator needs only three things: the stress field up to the onset, a failure criterion, and a calibrated threshold.

Candidate criteria:

- Global: the contact force on the knife or the gripper exceeds $F_c$. This is cheap, and a force gauge on real tofu calibrates it.
- Local: the largest principal stress, or stretch from $C = F^\mathsf{T} F$, averaged over a material length $\ell$, exceeds $\sigma_c$. $C$ does not change under a rigid rotation. RFW-S used $\|F - I\|_F$, which a rigid rotation does change.
- Plastic: the accumulated plastic strain exceeds $\varepsilon_c$. RFW-S starts its damage at 0.15.

The R4R question: the stress under a sharp edge is singular, so its local maximum grows as $\Delta x$ shrinks. The threshold then flows with $\Delta x$ and $h$: it is an effective parameter. Does the onset (its time, the knife depth, the force) agree across $\Delta x$ and $h$ once the threshold is renormalized? This is the consistency criterion of experiments 1 to 3.

## Possible reference solutions

- Hertz contact of a sphere on an elastic half-space: $F = \tfrac{4}{3} E^\ast \sqrt{R}\, \delta^{3/2}$, and the contact time of an impact.
- Elastic indentation by a flat punch or a wedge.
- The vibration modes of an elastic cube or cylinder, for the jiggle frequency.
- A converged run of FEM at a fine resolution.
- Real tofu: the force or the stress at which it breaks under a knife. TODO: find measured values.

## Methods

### Genesis 1.5.0

- MPM, `gs.materials.MPM`: Elastic, ElastoPlastic (von Mises, `von_mises_yield_stress`), Snow, Sand, Liquid, Muscle.
  - Explicit only: the substep must be at most $0.02\,\Delta x$ (s, with $\Delta x$ in m), or the solver warns. Two PRs, #3721 and #3779, add an implicit MPM; both were open on 2026-11-25.
  - The deformation gradient $F$ of each particle is in the solver state. TODO: find how to read it.
- FEM, `gs.materials.FEM.Elastic`: linear, stable_neohookean or linear_corotated, explicit or implicit (`FEMOptions(use_implicit_solver=True)`). It couples to rigid bodies through the legacy, SAP or IPC coupler. IPC needs libuipc, which is not installed here.
- No fracture or damage model. RFW-S added its own (see References).

### MuJoCo 4.0.0

- A 3D flexcomp with `<elasticity young poisson damping>` (Saint Venant–Kirchhoff). `dof` is full, trilinear or quadratic.
- Flex elasticity needs `integrator="discrete"` (new in 4.0) or Euler. implicitfast raises an error (checked 2026-11-25).
- Plasticity, `<elasticity yield>`: von Mises, 2D and 3D only, new in 4.0 (`model/plastic/clay.xml`). It deforms permanently, but nothing separates.
- The $F$ of each element: from `d.flexvert_xpos` and the rest shape.
- Fracture workaround: pre-cut pieces held by `connect` or `weld` equalities. Set `d.eq_active[i] = 0` when the force exceeds a threshold. The crack path is fixed in advance.
- MJX has no flex. MuJoCo Warp has part of it: no `dof="quadratic"`, no collision of a flex with an SDF or a height field.

## Findings so far (2026-11-25)

MuJoCo 4.0.0, a subagent's tests (to check again):

- A 10 cm cube, E 10 kPa, dropped from 0.3 m:
  - `discrete` at 2 ms is stable but dissipative: the coefficient of restitution is 0.49, against 0.72 with Euler at 0.05 ms.
  - Euler diverges at E 30 kPa, at 2 ms and at 0.5 ms.
  - MuJoCo warns of an ill-conditioned $M + K$ at E 3 MPa.
- A flex of 1000 vertices runs at about 1/3.3 of real time at 2 ms.
- Pre-cut pieces held by 16 connects broke at 0.78 N, within 6 ms.

## References

- RFW-S, arXiv 2611.23056.
  - Built on Genesis 1.3.3: PBD and MPM.
  - Tofu grasp, sim-to-real on a Piper arm.
  - MPM tofu: E 28 kPa, ν 0.32, ρ 950 kg/m³, yield stress 5 kPa, 6460 particles.
  - Its own damage extension lowers the shear and tensile strength over an accumulated plastic strain of 0.15 to 0.45.
  - Its integrity proxy $\|F - I\|$ flags all 4000 of its grasps.
- Wolper et al., CD-MPM: continuum damage material point methods for dynamic fracture animation, ACM TOG 2019.
- Huang et al., DefGraspSim: physics-based simulation of grasp outcomes for 3D deformable objects, arXiv 2203.11274.
- Balaji et al., OopsieVerse: a safety benchmark with damage-aware simulation for robot manipulation, arXiv 2606.31993.
- Huang et al., PlasticineLab: a soft-body manipulation benchmark with differentiable physics, ICLR 2021.
- MuJoCo docs: XML reference (flex elasticity, flex contact), Modeling (deformable objects), Changelog 4.0.0, GitHub Discussion #3773.
- Genesis: PRs #3721 and #3779 (implicit MPM).
