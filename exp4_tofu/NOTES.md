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
- FEM, `gs.materials.FEM.Elastic`: linear, stable_neohookean or linear_corotated, explicit or implicit (`FEMOptions(use_implicit_solver=True)`). It couples to rigid bodies through the legacy, SAP or IPC coupler. IPC needs libuipc (`pyuipc` 0.0.30, installed in the venv); with it libuipc simulates the FEM body itself: its own stable neo-Hookean whatever `model` is given, backward Euler by Newton, no damping parameter (Genesis's damping options never reach it), contact by a log barrier with d_hat = 1 cm by default, no vertex constraints and no initial velocity through Genesis (`genesis_ipc.py` writes both into libuipc's geometry).
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

## Findings (2026-11-27)

Step 0a, the free vibration (`run_flow.py`, `plot_flow.py`):

- The two discretizations separate: $E^\star = E\,S(\Delta x)\,T(h)$, with $S$ from the P1 modes (`p1.py`) and $T$ from one mode's time stepping (`theory.py`). Over 150 runs of MuJoCo and Genesis's FEM, $E^\star / (E\,T(h))$ is within 7e-4 of $S(\Delta x)$, and $E^\star / E^\star_{h \to 0}$ within 2e-6 of $T(h)$.
- $\zeta^\star$ has no spatial counterterm (to 1e-8 in Genesis, 3e-6 in MuJoCo).
- Equal vertex masses (MuJoCo) make $S - 1$ first order in $\Delta x$ (0.064 per mm); masses lumped by volume (Genesis) make it second order.
- MuJoCo's `discrete` is backward Euler with a metric $M + hD + h^2K$ clamped per term. Its departure from backward Euler is linear in the amplitude ($-2.3\times10^{-5}$ in $\omega$ at 0.05 mm, 1 ms), not a law of the step. Above $h = 2\beta$ = 4.4 ms no $E^\star, \zeta^\star$ gives the mode.
- `flexvert_xpos` lags `qpos` by one step after `mj_step`; read the vertices from `qpos`.
- Genesis's MPM is linear at this amplitude: its step, linearized about the particles at rest (`mpm_linear.py`), run through the same measurement, gives Genesis's $E^\star$ and damping to five digits. Its damping is that of the transfer to the grid and back, about $8.5\times10^{3}\,\Delta x^4 / h$ (s, m), which no parameter removes.
- Genesis's PBD (`pbd_linear.py`, `fig_flow_pbd`), on MuJoCo's grid with equal masses: each step projects every edge's stretch constraint in 4 Jacobi sweeps with relaxation 0.1, $dp = -rC/(w_i + w_j + \alpha/h^2)\,n$, and no multiplier is kept, so the compliance $\alpha$ alone is no stiffness. Its volume constraint does nothing: the correction carries $|\nabla C|^2$ once too often ($10^{-10}$ m$^4$ at 1 cm cells). The body is a network of edge springs.
- About rest a PBD step is $u' = P^{4}(2u - u_\mathrm{prev})$, $P = I - cL$, $c = r/(2 + m\alpha/h^2)$, $L$ the network of unit springs: a mode of $L$ has $|\lambda|^2 = p$, $\arg\lambda = \arccos\sqrt p$, $p = (1 - c\mu)^4$. Run through the same measurement, this linear step gives Genesis's $E^\star$ to 2e-5 and its damping to 2e-3 (relative) at all 20 points, 10 to 2 mm, $h$ from 0.1 to 0.9 $h_\max$.
- Zero compliance is as stiff as PBD gets: $E^\star$ exists only below $h_\max$ = 1.03, 0.58, 0.41, 0.31, 0.25 ms at 10, 5, 3.3, 2.5, 2 mm. $E^\star/E$ depends on $h/h_\max$ almost alone: 2.4 to 2.5 at 0.1 $h_\max$ (the edge springs $E\,\Delta x$ are too soft for the continuum's mode), 13 at 0.9. PBD's own damping, $\zeta \approx \omega h/2$, has no parameter and stays below 0.043 (10 mm) and 0.011 (2 mm): it cannot give tofu's 0.10 at the right frequency.
- Genesis 1.5.0 on the CPU runs one thread per process; the cluster's CPU nodes need the X and GL libraries copied in.

Step 0b, Brookfield's ball (`indentation.py`, `axisym_nl.py`, `p1_static.py`):

- MuJoCo's flex force is exactly that of Saint Venant-Kirchhoff on the P1 tetrahedra (1e-14).
- At 7 mm under the 12.7 mm sphere both the strain and the moving contact matter by about 10 %: the reference must be large-deformation, in each engine's own material.
- Saint Venant-Kirchhoff's stress in compression has a maximum (at a stretch of $1/\sqrt{3}$ in uniaxial compression). Under the sphere the most compressed element, on the axis 2 mm deep, reaches it at $\delta \approx 2.9$ mm ($\delta/R = 0.23$; 0.2 mm cells, 3.0 mm at 0.4 mm): its own stiffness falls to zero, and the solution jumps to a crushed element. MuJoCo's flex cannot reproduce Brookfield's 7 mm, whatever $E^\star$. Linear corotated and neo-Hookean reach 7 mm.
- Genesis 1.5.0 has no large-deformation material for both FEM solvers: explicit has stable neo-Hookean, implicit linear corotated.
- MuJoCo collides the sphere with whole elements, so its contact holds the piecewise-linear surface, not only the vertices; at coarse cells that is much stiffer than vertex contact. The static error of P1 contact depends on $\Delta x / a$, $a = \sqrt{R\delta}$, not on $\Delta x$ alone.
- MuJoCo's default contact (`solref` 0.02) is soft: at 10 mm cells it takes up 70 % of a 0.25 mm indentation.
- MuJoCo's CG solver gives Newton's contact forces to 1e-4 here, five times faster.
- The step does not matter: the indentation is quasi-static, and MuJoCo's contact stiffness at rest is free of $h$ (the factor $1/(1 + hB + h^2K)$ of $a_\mathrm{ref}$ cancels against the same factor in the inverse contact mass $A$). Four steps from 0.25 to 2 ms fall within 1 % of each other.
- Each MuJoCo contact at rest is a spring $k = K d^2 / (1 - d)\, m_v$, $K = 1/(d_\max t_c)^2$, $m_v = \rho V / N$ (from MuJoCo's source; the MuJoCo ball-press code has since been removed). It goes as $\Delta x^3 / t_c^2$ while the slab under it is about $E\,\Delta x$ stiff, so at a fixed `solref` the contact softens against the slab as $\Delta x^2$: with the default 0.02 s the force at 2.5 mm falls to 1.07, 0.54, 0.34 of the continuum's at 10, 5, 3.3 mm cells. There is no continuum limit unless $t_c$ is renormalized: $t_c^\star \propto \Delta x^{3/2}$ keeps the ratio. Under the discrete integrator MuJoCo 4.0 bounds the spring by what the step can resolve (refsafe in `mj_makeImpedance`): $K \to \min(K, 1/(h^2 d))$. At $h$ = 1 ms the bound already holds for the renormalized $t_c^\star$ = 0.38 and 0.25 ms (3.3 and 2.5 mm cells): 0.15 and 0.06 of the unbounded spring. Once $t_c$ is below about $h$ the contact goes as $\Delta x^3/h^2$ and softens against the slab again: a renormalized $t_c^\star$ needs $h \lesssim t_c^\star$, the step shrinking as $\Delta x^{3/2}$.
- MuJoCo's force follows the static P1 slab held by those springs at rest (statics with MuJoCo's contact, since removed: one contact per top triangle, at its point nearest the sphere's centre, its Hessian by differences so that Newton converges under stiff springs): hard contact to 0.9 % past the first depth, renormalized to 0.7 %, default to 6 % (10 to 2 mm). At the first depth, 0.25 mm, the runs lag the static slab, as the sphere moves at 2 mm/s: −1.5 to −3 % (hard, the largest step the most), −11 % (default at 10 mm). At 2 mm the default contact leaves 5 to 23 % of the continuum's force: what is lost with refinement is lost in the contact, not in the material.
- The force on the sphere is the vertical part of the contacts' forces. Their normals are radial, tilted from the vertical by up to 15°: summed as magnitudes they read 3 to 5 % high, the more at depth and in fine cells. The MuJoCo runs before 2026-11-27 15:00 summed magnitudes (kept in `results/indent_parts_normal_sum/`) and were run again.
- Genesis's legacy coupler acts on a point at distance $s$ outside a rigid geom with the weight $e^{-s/\mathrm{coup\_softness}}$, 2 mm by default, as deep as the indentation; 4b sets 1e-5 m. Then the explicit FEM follows the static P1 slab with vertex contact, in stable neo-Hookean, to a few %.
- The sphere is driven by resetting its state every step; within the step the coupler pushes it back with the slab's reaction. A sphere of tofu's density recoils enough to lower the force by up to 20 % (0.9 ms, 7 mm), in proportion to $h\,\delta$; at 1e7 kg/m³ it does not.
- Genesis's implicit FEM needs the SAP coupler, which collides the FEM's surface tets with the triangles of the sphere's collision mesh, and Genesis meshes an MJCF sphere at two icosahedral subdivisions: 3.8 mm facets, 0.23 mm inside the sphere. With a mesh at four subdivisions and the heavy sphere the force is free of $h$. Each SAP contact carries its area times the FEM's hydroelastic pressure where the sphere cuts the tet, a field growing from the surface at the modulus (1e7 Pa) over the deepest vertex (2 cm): a compliant layer of 5e8 Pa/m. SAP's own stiffness, the harmonic mean of that and its `hydroelastic_stiffness`, only linearizes the contact within a step. Genesis decimates a collision mesh to 500 faces by default (`decimate`, implied by `convexify`), so the sphere of four subdivisions was 3 mm facets lying 0.09 mm inside it, a third of the first depth; with `decimate=False` it keeps its 5120. With the decimated sphere SAP's force left the static P1 slab on the layer by up to 8 % (12 % at the first depth), the more as the cells shrank; with the full sphere, by 1.3 % at 2.5 mm cells (runs of 2026-11-27 17:30; the decimated ones kept in `results/indent_parts_decimated_sphere/`).
- Genesis's explicit FEM follows the static P1 slab with vertex contact, in stable neo-Hookean, to 4 % at every cell size from 10 to 2 mm (6 % at 2 mm and the first depth).
- Tofu breaks in compression at an engineering strain of 0.44 to 0.56 (Kohyama), a Green strain of 0.34 to 0.40 in magnitude. Saint Venant-Kirchhoff's stress peaks at a stretch of $1/\sqrt{3}$, a Green strain of 0.33: MuJoCo's flex gives way before tofu would break.

Step 3, the drop test, needs a camera; what it needs:

- A block of silken tofu, its size measured (about 10 x 8 x 4 cm), drained, from a known brand, weighed.
- Drops from 2, 5, 10 and 20 cm onto a flat rigid plate, three each, the block flat; and from higher until it cracks.
- Filmed side-on at 240 fps, the camera level with the plate, a ruler in the plane of the block, steady light (not mains-flickering LEDs, or note the frequency).
- What is read: the rebound height, the contact time, the largest squash, and the height at which it first cracks.

Step 1, the tofu's modulus from the literature (`calibrate.py`):

- Brookfield's peak forces at 7 mm, through neo-Hookean on a 4 cm block of the slab's area, bonded: $E$ = 13.3 / 14.0 / 21.3 / 22.2 / 36.7 kPa for Gourmet Soft / Longlife Soft / House Firm / Ichiban Firm / Ichiban Regular. The block's thickness (3 to 5 cm), its support (bonded or sliding) and $\nu$ (0.45 or 0.49) move each by about 10 %.
- The soft tofus agree with Pattison's silken soft at 14 Hz, 11.3 kPa ($\nu$ = 0.45); the firm ones fall between Dunne's firm (16.4 kPa from $G'$) and Pattison's silken firm (34.2).
- Kohyama's fracture point of silken tofu, 7.1 to 8.4 kPa at 44 to 56 % strain, gives 4.5 to 9.6 kPa in neo-Hookean: at half its height tofu is 1.5 to 2.5 times softer than neo-Hookean with its small-strain modulus. One $E$ does not fit both.
- Brookfield's test in Genesis's explicit FEM (5 mm cells, 0.25 ms) to 7 mm follows the statics of its mesh to 2 %; the mesh is 12 to 19 % stiffer than the continuum. Given $E^\star = E\,F_\mathrm{ref}/F_\mathrm{mesh}$, each tofu's $E$ from its measured force, the five forces are within 2 % of the measured ones (14 % above with $E$ as it is); the deviation is the same for all five, the run's force over the statics' at 7 mm. MuJoCo's force at 7 mm has no value that converges with the mesh (Saint Venant-Kirchhoff, above); MuJoCo was never run past 2.5 mm.
- Genesis's MPM and PBD in the press (`run_indent_particles.py`), through the legacy coupler, the force read from the sphere's velocity change over the step (it is reset every step), checked against a mid-height stress section (MPM, within 2 %) and the bottom's reaction (PBD, within 2.3 %). Neither has statics: the continuum is the only reference.
  - MPM's `MPM.Elastic` is fixed corotated, not the FEM's linear corotated; its continuum is computed in that material. At 0.02 dx only 5 mm cells run above 0.01x (0.016x). At E it is 18 to 69 % under the continuum, 52 % at 7 mm. Brookfield: $E^\star$ = 2.09 E brings the five forces to 1.8 to 3.0 % under the measured ones (52 to 53 % under with E).
  - PBD (0.5 $h_\max$, the sphere's geom shrunk by particle_size/2 so the particles stop on radius R) is 28 to 83 % under the continuum at the nominal compliance $1/(E\Delta x)$, and goes to about -63 % as the cells shrink, not to 0. Its force is not proportional to 1/compliance, and at zero compliance it saturates at a ceiling that goes as $1/h^2$ (the coupler's force is $m\delta/h^2$, and the push-back $\delta$ no longer depends on h): 2.55 N at 0.5 $h_\max$, 10.2 N at 0.25 $h_\max$ (5 mm, 7 mm). At 0.25 $h_\max$ one compliance per tofu meets each of the five measured forces to 0.3 %, at $E^\star$ = 5.8 to 8.9 times the tofu's E, not one ratio.

Step 2, the Jello cube of Visual Vibration Tomography (`video_modes.py`, `jello_model.py`, `jello_sim.py`, `plot_jello.py`; the videos in `/data/datasets/vvt_jello_cube`):

- The three clips run at 4000 fps to 0.05 %, by the lights' 120 Hz flicker.
- Optical flow, then a matrix pencil on the motion's largest components: two sways at 8.0 to 8.3 Hz, the twist at 10.0 to 10.4 Hz, $\zeta$ = 0.015 to 0.022. The twist video's modes are 3.5 % above the plucks'.
- On a fixed base, the P1 cube puts the twist 1.25 times above the sway with no clay or with the clay at the top, as observed; then both modes give $E/\rho$ = 10.4 to 11.5 m²/s², $E$ = 11.8 kPa at 1080 kg/m³. With the clay at the bottom it would be 5.2 to 7.
- $\zeta$ agrees with Jell-O's $\tan\delta / 2$ = 0.018 on a rheometer (Ata et al.); $E$ with 4 % gelatin (Gravelle and Marangoni, 10.6 to 14.6 kPa).
- `video_jello.py` overlays MuJoCo's cube on the video: the cube's motion magnified 20 times by the video's own optical flow, MuJoCo's edges through the camera of the clicked corners (DLT, 0.6 px rms). The two sways of the P1 cube hold 88 % of the motion from 4 to 20 Hz.
- MuJoCo, `discrete`, 10 cells a side, given $E^\star = E\,S\,T(h)$ and $\beta^\star$ from step 0a's theory with $S$ from the P1 sway on its own mesh: the sway's frequency and damping match the video's to 1e-4 at 0.25 and 1 ms, with nothing fitted. Nominal $E, \beta$ at 1 ms give $\zeta$ = 0.041. At 2 ms > $2\beta$ no parameters do.
- The top-front pluck alone (`video_modes.py`'s motion, matrix pencil): sway 8.010 Hz, $\zeta$ 0.0151; twist 9.992 Hz, $\zeta$ 0.0154. The speed-accuracy figure's targets.
- Backward Euler with $\beta^\star < -h/2$ amplifies every mode with $\omega^2 h(-2\beta^\star - h) > 4$. Jello's $2\beta$ is 1.19 ms, so `jello_sim.py`'s 2 ms run is garbage (f 8.84 Hz, $\zeta$ 0.103), and the quadratic flex blows up at 5 ms ($\omega_\max$ 909 rad/s against a threshold of 459) while the trilinear one (356 rad/s) does not.
- MuJoCo Euler, full dof (`jello_mujoco_variants.py`): at 0.1 ms (limit 0.151 ms) the renormalized sway is 8.010 Hz, $\zeta$ 0.0150, nothing fitted; there $T$ and $\beta^\star/\beta$ differ from 1 by 1e-4, so it is $S$ that matters.
- MuJoCo Euler, ball press (`run_indent.py`, `mj_euler_*`): only 5 mm cells are above 0.01x real time (h 0.045 ms, 0.017x); at every depth its force is within 0.1 % of `discrete`'s at 1 ms on the same cells, both with MuJoCo's default contact (49 to 70 % below the continuum) and with the hard one (65 to 176 % above).
- MuJoCo `dof="trilinear"` (8 nodes): MuJoCo puts the flex's whole mass on the unpinned nodes. Jello: $S$ = 2.22, sway matched, but its twist is at 5.9 Hz against the video's 10.0. Tofu block (one cell): $S$ = 6.23, exactly 9 times the quadratic's: its corner-lumped masses overstate a bilinear field's kinetic energy 9-fold.
- MuJoCo `dof="quadratic"` (27 nodes): Jello $S$ = 0.929, sway matched, twist 9.45 Hz. Tofu block: $S$ = 0.693; its lowest elastic mode is the continuum's second.
- Genesis through IPC (`genesis_ipc.py`, `run_ipc.py`): $E^\star = E\,S\,T(h)$ with $T$ undamped backward Euler's (each step multiplies the mode by $1/(1 - i\omega h)$) to 3e-7 relative, over 65 runs; its only damping is backward Euler's, $\zeta \approx \omega h/2$: the tofu's 0.10 at $h$ = 2.225 ms, Jello's 0.015 at 0.596 ms, where its sway matches the video with nothing fitted. Newton's tolerance must be tightened: at Genesis's default $\omega$ is 3e-4 low.
- Speed against accuracy on the Jello cube (`plot_speed_accuracy.py`, `fig_speed_accuracy`; data `results/speed_accuracy/`, from `speed_accuracy_{mujoco,genesis,ipc}.py`): cells per side 1 to 20, three or four steps each; speed is simulated over wall time of the stepping loop on the cluster (one EPYC 7763 core for MuJoCo, one B300 for Genesis), points below 0.01x real time not run. Two residuals, each with its own fit, because one fit to all four numbers lets coarse meshes win by accident (their low twist brings $\zeta_\mathrm{twist}/\zeta_\mathrm{sway}$ nearer the video's 1.02):
  - frequency: $E^\star$ alone fitted to the sway and twist of the same model's continuum (8.010, 10.087 Hz), not to the video's twist (9.992 Hz), since the model without the clay is itself 0.47 % off and a mesh error can cancel that. Genesis's FEM (explicit, implicit, IPC; masses by volume) is near 0.15 % at 1 to 4x real time; MuJoCo's full dof, with equal vertex masses, 2 to 5.5 %; quadratic 0.36 % at 0.02x; MPM reaches 0.05 % only at 0.012x (16 cells); PBD's edge springs put the twist 11 to 12 Hz, 2.5 to 9 %.
  - damping: every damping parameter the engine offers fitted to the two damping ratios. Only Genesis implicit, with both Rayleigh coefficients, meets both (below 1e-5 %), except at h = 1 ms with 8 or more cells, where the fit's beta hits backward Euler's stability bound $4 + h\omega_\max^2(2\beta + h) \ge 0$ (1.5 to 7.5 %). MuJoCo's beta K and Genesis explicit's mass-proportional c give $\zeta \propto \omega^{\pm 1}$: 5 to 10 % and 12 %. MPM, PBD and IPC have no parameter: 10 % to thousands of percent, set by h and dx. MPM and PBD were also run at the step whose own sway damping is the video's 0.0151: MPM's twist is then damped twice as much (72 to 86 %); PBD's has such a step only up to 10 cells (0.600 ms).

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
