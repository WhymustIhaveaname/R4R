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

## Two discretizations

Every effective parameter flows with two discretizations, not one: the time step $h$, and the spatial resolution $\Delta x$ (the element size of FEM and flex, the grid spacing and the particles per cell of MPM). The consistency check is therefore on a two-dimensional grid of $(h, \Delta x)$. Derive how each parameter flows along both axes, then test whether the fitted points collapse onto the theory or scatter.

## Plan (2026-11-26)

0. Renormalization, theory and numerical experiments, with no real data. This is the R4R core.
   - References: the vibration modes of an elastic block (analytic, or a converged fine FEM), Hertz indentation, and compression up to the failure criterion.
   - Schemes: MuJoCo flex (`discrete`, Euler); Genesis FEM (explicit, implicit); Genesis MPM (explicit).
   - Fit $E^\star$, the damping and the failure threshold on the $(h, \Delta x)$ grid.
1. Calibrate the material from the literature (no experiment of our own).
   - For tofu (silken, soft, firm) and for gelatin, agar and konjac gels: $E$ or $G$, $\nu$, $\rho$, the loss tangent, and the fracture stress and strain under compression, indentation and wire cutting. Record the scatter as well.
   - The simulation must reproduce the published compression and indentation curves.
2. Check the vibration against video (no experiment of our own).
   - Find real videos of tofu or jelly wobbling online, with a known frame rate and a scale reference (see External data).
   - Identify the frequency and the damping, hence $E/\rho$ and the loss, and compare them with step 1. The two sources are independent.
   - Fracture is rarely checked by others; getting the vibration right is already worth it.
3. The drop test, by hand, only after steps 0 to 2: several heights, filmed at 240 fps. It checks the bounce and the onset of breakage on data the calibration never saw.

## External data, cross-checked (2026-11-26)

Three agents searched; two others re-checked every claim from the downloaded files and the papers. Only what survived is here.

Videos:

- Visual Vibration Tomography: Feng, Ogren, Daraio and Bouman, CVPR 2022, arXiv 2104.02735. Repo `berthyf96/visvib-tomography`; the videos are in a public Caltech Box folder, with no license.
  - A real Jello cube, 4.9×4.7×4.5 cm, with a clay inclusion of 2.2×2.9×1.4 cm, plucked and twisted.
  - Raw BGR24, 276×288, about 3 s per clip.
  - Capture at 4000 fps, confirmed by the 120 Hz mains flicker (one period every 33.1 frames).
  - Our check found modes at 8.23 and 10.40 Hz, with ζ = 0.020 to 0.022. The paper's ζ 0.0175 at 12.5 Hz and 0.020 at 15.5 Hz are not at this cube's modes.
  - E = 9 kPa and ρ = 1270 kg/m³ in the paper are simulation settings. The supplement's "measured" 1500 kg/m³ is implausible for a gel.
  - The model must include the clay.
  - Best for both stiffness and damping.
- "Chambord Jelly Wobble HD", YouTube `_2byiY4dJDI`.
  - A clean free decay: side view, fixed camera, 17 periods.
  - ζ = 0.0101 (0.007 to 0.014 per cycle; it depends on the amplitude).
  - The capture rate and the size are unknown: damping only.
- Waitukaitis et al., Nature Physics 2017, Supplementary Video 3.
  - A PAAm hydrogel sphere: D 14.9 mm, m 1.75 g, E 50±4 kPa, filmed at 5000 fps.
  - The Hertz contact time, 7.4 ms, matches the video.
  - For impact. Not food, and not open access.
- Communications Physics 2022 (s42005-022-01114-8), Movie 1.
  - A PAAm bead: D 16 mm, G 6 to 8 kPa; CC BY.
  - The video's time scale does not match the 1.21 m/s of its caption. Ask the authors for the raw data.
- Photron, coffee jelly drop, YouTube `1JEsIzAaaQM`: 4000 fps on screen. Size and material unknown: qualitative only.
- No usable video of tofu exists online. For tofu vibration, use the dynamic moduli below, or film it ourselves later.

Literature, checked against the papers:

- Tofu, dynamic: Pattison, McGarry, Weaver and Paulsen, IEEE TBME 62:443 (2015), doi 10.1109/TBME.2014.2357771, Table I.
  - Mori-nu silken soft / firm / extra firm at 14 Hz: μ_r = 3.88±0.64 / 11.8±1.6 / 21.1±3.5 kPa.
  - Damping ratio 0.10 / 0.11 / 0.12, i.e. tan δ ≈ 0.20 to 0.24. ν is assumed to be 0.45.
  - Cylinders 28.5 mm across and 7.5 to 10 mm high, at about 1% strain.
- Tofu, dynamic, independent: Dunne et al., Food Research International 205:115876 (2025), Table 3.
  - Firm (365 Whole Foods) / extra firm (House Foods), at 0.22 to 4.64 rad/s: G′ = 5.67 / 11.0 kPa, tan δ = 0.229 / 0.241.
- Tofu fracture: Kohyama and Nishinari, Nippon Shokuhin Kogyo Gakkaishi 39(8):715 (1992), doi 10.3136/nskkk1962.39.715.
  - Kinugoshi (silken) tofu, unconfined compression at 1 mm/s.
  - Nominal fracture stress 7.1 to 8.4 kPa; engineering fracture strain 0.44 to 0.56.
  - Nearly the same from 5 to 30 °C.
- Tofu under a ball probe: Brookfield application note "Tofu Texture Analysis" (CTX analyzer).
  - A 25.4 mm sphere (TA43), 7 mm deep at 2 mm/s, twice. The tofu is out of its package, on the table. The block's size is not given.
  - Peak force of the first press: 173 / 278 / 289.5 / 479 / 183 g for Gourmet Soft / House Firm / Ichiban Firm / Ichiban Regular / Longlife Soft.
- Deep indentation: Mu, Linghu, Liu, Leng, Gao and Hsia, J. Mech. Phys. Solids 212:106572 (2026), doi 10.1016/j.jmps.2026.106572. In `papers/`, with its supplementary files, the three appendix figures.
  - A rigid sphere on a neo-Hookean half-space: $F = F_H\,[1 - \tfrac{11}{60}\,\delta/R - \tfrac{1}{350}(\delta/R)^2 + \dots]$ (Eq. 16), checked against FEA up to δ/R = 2.5.
  - Tofu is among its experiments (Fig. 7, F/F_H only). Neither the article nor its supplement gives the tofu's sphere, sample or modulus.
- Tofu under a ball, in its jar: Huang, Kaletunc, St. Martin, Feller and McHale, Plant Breeding 133:67–73 (2014), doi 10.1111/pbr.12122. In `papers/`.
  - Lab-made CaSO₄ tofu, pressed to failure in the glass jar it set in, by a 19.05 mm ball at 1.00 mm/s. The jar's size is not given.
  - Only BLUPs, deviations from the mean, are reported: no absolute force.
- Tofu compression: Mullin et al., Food Research International 34:669–677 (2001), doi 10.1016/S0963-9969(01)00087-4. In `papers/`.
  - Cylinders 22 mm across (printed "22 cm") and 2 cm high, compressed to half their height by a 57 mm plate at 10 mm/min. Hardness (N) and firmness (N/mm) in Tables 6 and 7.
  - Its penetration test uses an "8 mm cell", not a ball, though Huang et al. cite it for their ball.
- Khoder et al., J. Food Sci. 89:2843–2856 (2024), doi 10.1111/1750-3841.17067. In `papers/`. Breaking force and penetration distance of acid-set tofu; neither the probe nor the sample size is given.
- Gelatin: Gravelle and Marangoni, Data in Brief 38:107410 (2021). 250 Bloom, about 8 °C, at 2 / 4 / 6 / 8%:
  - E = 3.2–4.0 / 10.6–14.6 / 22.9–34.2 / 40.4–50.2 kPa;
  - true fracture stress 7–10 / 21–30 / 33–51 / 52–63 kPa;
  - Hencky fracture strain 1.1 to 1.7.
- Gelatin resonance: Okamoto, Clayton and Bayly, Phys. Med. Biol. 56(19) (2011).
  - 2.8% gelatin with 48.6% glycerol; a 5.5 mm layer resonates in shear at 83±4 and 167±9 Hz.
  - G′ = 0.86 to 0.97 kPa; loss factor 0.06 to 0.14.
- Gelatin vibration against concentration: Schwarz et al., Exp. Mech. 60:1067 (2020).
  - Clamped discs, 3 mm thick and 9.7 mm across: the (0,1) mode is at 144 / 223 / 486 Hz for 0.10 / 0.20 / 0.40 g/ml.
  - Their AFM moduli are about 3 times below the vibration-based ones.
- Jell-O on a rheometer: Ata et al., Gels 12:295 (2026). G′ ≈ 168 Pa and G″ ≈ 6 Pa, at 10 rad/s and 20 °C. This is a tenth of the cube's assumed E.
- Caution: the brand, the frequency, the strain, the temperature and the curing time each change these values, by up to 10 times. A secant modulus from compression is about 10 times the small-strain G′.

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
