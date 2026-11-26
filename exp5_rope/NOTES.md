# Experiment 5: rope, shoelace, cable (notes, 2026-11-25)

Notes taken before any code. Versions: MuJoCo 4.0.0 and Genesis 1.5.0, the latest on 2026-11-25.

## Goal

A deformable linear object (DLO): a rope, a shoelace, a cable, an electric wire.

1. Draping and hanging.
2. Tying a knot, and the knot holding: self-contact with friction, and no tunneling.
3. Fast swinging and whipping, with friction against itself and against other objects.

## Possible reference solutions

- A hanging inextensible chain:
  - static shape: the catenary;
  - normal modes: $\omega_n = \tfrac{j_{0,n}}{2} \sqrt{g / L}$, with $j_{0,n}$ the zeros of $J_0$ (2.405, 5.520, ...). This is a frequency test, as in experiments 1 to 3.
- A stiff rod under gravity: the elastica of a cantilever.
- The capstan equation: a rope over a cylinder holds $T_\mathrm{out} = T_\mathrm{in}\, e^{\mu \theta}$. This tests the friction.
- The pull-out force of a tight overhand knot: Jawed et al., PRL 2015.

The R4R question: the effective bending and twist stiffness flow with the segment length $\ell$ and the step $h$.

## Methods

### MuJoCo 4.0.0

- MuJoCo 4.0 removed `<composite>` and the `mujoco.elasticity.cable` plugin. Checked on 2026-11-25: the first gives a schema error, the second "plugin not found".
- Now: `<flexcomp dim="1">` with `<elasticity>`, a discrete elastic rod (bend and twist).
  - `<elasticity>` does not resist stretch: add `<edge equality="true">`.
  - The integrator must be `discrete` or Euler.
- The default edge `solref` is too soft.
  - A 1 m, 50 g rope stretches about 25% under its own weight.
  - With `solref="-1e8 -3000"`, it stretches 0.03% under a 0.2 kg end load at 2 ms.
  - Under a 1 kg end load, it stretches more than 400% at 2 ms, with linesearch warnings, and 0.14% at 0.5 ms.
- Self-contact: `selfcollide="auto"`. The elements collide as capsules, with friction (condim 3).
- No continuous collision detection: speculative contacts do not support flex, and the IPC flag is for 2D only and frictionless.
- A subagent's tests (to check again):
  - Two crossing ropes, radius 4 mm:
    - at 2 ms, they stop each other up to 4 m/s and tunnel at 8 m/s;
    - at 0.5 ms with contact `solref` 0.001, they stop each other up to 16 m/s and tunnel at 32 m/s.
  - An overhand knot, radius 3 mm:
    - at 2 ms, the rope tunnels through itself;
    - at 0.5 ms with friction 0.5, the knot holds at 9.7 N, but penetrates 2.7 to 3.2 mm.
  - A whip driven by a 2 Hz square wave:
    - stable at 2, 1 and 0.5 ms;
    - the tip reaches 46 to 59 m/s;
    - the stretch of the tip edge peaks at 306%, 156% and 37%.
- MJX has no flex. MuJoCo Warp has part of it. Before 2026-10-07 it collided a 1D flex only as spheres at the vertices (mujoco_warp issue #1707).

### Genesis 1.5.0

- No rope, rod or strand material. Genesis loads MJCF by compiling it with MuJoCo and then reads only the bodies, joints, geoms and equalities. It drops plugins and flex.
- Plan A: a chain of capsules with joints, as MuJoCo's old composite cable was.
  - Generate the MJCF: N capsule bodies. Each joint is two bending hinges and one twist hinge, with stiffness $k_b = EI/\ell$ and $k_t = GJ/\ell$, and damping.
  - Genesis applies the stiffness of every dof of a joint that is not free (`func_torque_and_passive_force` in `engine/solvers/rigid/abd/forward_dynamics.py`).
  - `RigidOptions.enable_self_collision` is True by default, and `enable_adjacent_collision` is False.
  - The chain is exactly inextensible.
  - Limits:
    - no continuous collision detection;
    - a closed loop (both ends held) needs an equality;
    - the mass matrix is dense in N.
- Plan B: a PBD cloth strip or a thin tube of tetrahedra. It has no twist. Not recommended.
- Plan C: a tube of FEM elements through the IPC coupler. It has no penetration and has friction, so it is the most robust for knots. It is slow, and libuipc is not installed here.
- DLO-Lab (arXiv 2606.04206) built a solver of discrete elastic rods in Taichi and integrated it into Genesis.
  - It has rigid coupling through SDFs and MPM coupling.
  - Its self-collision is PBD-style, with Coulomb friction.
  - It has an unknotting task.
  - Project page: https://dlo-lab-26.github.io/. TODO: check whether the code is released.

## References

- Bergou, Wardetzky, Robinson, Audoly and Grinspun, Discrete elastic rods, ACM TOG 2008.
- Bergou, Audoly, Vouga, Wardetzky and Grinspun, Discrete viscous threads, ACM TOG 2010.
- Li, Kaufman and Jiang, Codimensional incremental potential contact (C-IPC), ACM TOG 2021.
- Tong, Choi, Joo and Jawed, A fully implicit method for robust frictional contact handling in elastic rods, Extreme Mechanics Letters 58 (2023).
- Jawed, Dieleman, Audoly and Reis, Untangling the mechanics and topology in the frictional response of long overhand elastic knots, PRL 115, 118302 (2015). From memory: TODO check.
- Naughton et al., Elastica: a compliant mechanics environment for soft robotic control (PyElastica, Cosserat rods), RA-L 2021.
- DLO-Lab: benchmarking deformable linear object manipulations with differentiable physics, arXiv 2606.04206.
- Seita et al., DeformableRavens, ICRA 2021.
- Lin et al., SoftGym, CoRL 2020.
- MuJoCo docs: XML reference (flex elasticity, flex contact), Modeling (preventing tunneling, tuning deformables), Changelog 4.0.0.
- mujoco_warp issues #1707 and #1362.
- Genesis issue #1605: a MuJoCo cable in Genesis (2025).
