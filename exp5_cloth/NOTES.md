# Experiment 5: cloth, rag, towel (notes, 2026-11-28)

Notes taken before any code. Versions: MuJoCo 4.0.0 and Genesis 1.5.0 (libuipc 0.0.30), as in experiment 4; MuJoCo 3.1.0 only to reproduce Blanco-Mulero's fit. Every external fact below was checked on 2026-11-28 in the downloaded paper, data or source; where a value is our own reading or calculation, it says so.

## Goal

A robot handles a rag or a towel. Two things matter.

1. Dynamics: the cloth swings, flings, drapes and settles. How do its effective parameters flow with the time step $h$ and the mesh size $\Delta x$?
2. Contact with the hand: a gripper pinches the cloth, lifts it off a table, drags it, and holds it while it swings. A policy sees the forces at the wrist and at the fingers, and whether the grip holds. If these change with $h$ and $\Delta x$, a policy trained at one resolution acts differently at another.

## Plan (2026-11-28)

0. Renormalization, theory and numerical experiments, with no real data. This is the R4R core.
   - 0a. Dynamics. References with known answers:
     - A hanging sheet, its top edge clamped, with no bending: the swing modes of a hanging chain, $\omega_n = \tfrac{j_{0,n}}{2}\sqrt{g/L}$ = 1.20241, 2.76004, 4.32686 $\sqrt{g/L}$ (Verbin 2015), for any material. It tests the mass lumping, the inextensibility, and the numerical damping of a mode that gravity restores.
       - MuJoCo's flexcomp gives every vertex the same mass, the edge and corner vertices too. As in experiment 4, $S - 1$ should then be first order in $\Delta x$.
       - Bhat et al. 2003 (p. 2): "implicit integration introduced damping which could not be eliminated by optimizing cloth parameters", so they used an explicit higher-order scheme. Experiment 4 found no $E^\star, \zeta^\star$ above $h = 2\beta$. Is there a counterterm for this damping?
     - A clamped strip, no gravity, small amplitude: $\omega_1 = 1.8751^2\sqrt{D/\rho_A L^4}$ (Timoshenko 1937; Leissa 1969, $D = Et^3/12(1-\nu^2)$ for a wide strip). It tests the bending operator.
       - Turn the strip against the mesh: $\varphi$ = 0°, 22.5°, 45°, 90°, 135°.
       - MuJoCo's bending is not isotropic (measured on 2026-11-28 with the installed 4.0.0, a cylinder bent along $\varphi$). On `type="grid"`, where every diagonal runs along +x+y, the energy is $1 + \sin 2\varphi + \sin^2 2\varphi$ times Kirchhoff's: 3 at 45°, 2.2 at 22.5°, 1 at 0°, 90° and 135°. With alternating diagonals it is $1 + \tfrac12\sin^2 2\varphi$, at most 1.5. A dome is isotropic on both. The factor does not shrink as the mesh refines (n = 16, 32, 64: 2.875, 2.937, 2.969 at 45°).
       - So $\omega_1$ changes by up to $\sqrt 3$ with the direction of the fold. A scalar $D^\star$ cannot absorb this: a real cloth folds in several directions at once. Measure the same factor for an unstructured mesh, and choose the mesh before any fit.
       - Run with the stretch on (`elastic2d="both"`) and with inextensible edges (`"bend"` and edge equalities). Inextensible edges on an irregular mesh lock at any resolution (Chen, Kry and Vouga 2019). The difference is the locking.
     - Peirce's heavy cantilever: a static large deflection, the exact elastica solved by shooting. At the chord angle 41.5° of ASTM D1388 the exact solution gives $c = l/2$; Peirce's approximate formula gives $0.509\,l$ (Lammens et al. 2014; our shooting gives $l/c$ = 2.0004).
     - The drape test of Garcia-Camacho et al. 2024: the cloth folded to a rectangle on a disc of 60 % of its short edge, its projected area from above. It is static, and the cloths of step 2 have published values for it.
     - Converged fine runs of the motions in step 2 (Coltraro's shake and twist, Blanco-Mulero's fling): the nonlinear references. A reference may take at most 1000 times its simulated time on the cluster (a 15 s shake: 4.2 h). The finest mesh and step within that is the reference; its difference from the next coarser run is its error.
   - Fit $D^\star$, $Y^\star$ and the damping on the $(h, \Delta x)$ grid. Test whether they separate as $S(\Delta x)\,T(h)$, as in experiment 4. $T(h)$ for one mode under backward Euler and $\beta K$ is already in `exp4_tofu/theory.py`.
   - Then run the nonlinear motions with the renormalized parameters at a coarse $(h, \Delta x)$, against the converged runs. Do parameters from linear modes carry over to large motion?
   - Air drag is per vertex in both engines, so it flows with $\Delta x$:
     - MuJoCo applies the inertia-box fluid model to every vertex body, whose box has side $2b$, with $b$ = `inertiabox` (5 mm by default, whatever the `radius`): $\tfrac12\rho_\mathrm{air}(2b)^2|v|v$ along each axis, plus a viscous term $3\pi\eta\,(2b)\,v$. The total drag goes as the number of vertices, does not depend on the area of the cloth, and does not depend on its normal. A counterterm $b \propto \Delta x$ keeps the quadratic drag; the viscous term would need $b \propto \Delta x^2$, so one $b$ cannot keep both. Nothing makes it depend on the normal.
     - Its size on the 402 vertices of Blanco-Mulero's mesh at 1 m/s (measured 2026-11-28): 0.024 N quadratic and 0.0007 N viscous in air. A plate of that area broadside, with $C_d \approx 1.2$, would feel about 0.23 N (our estimate); edge-on, almost nothing.
     - Genesis's PBD cloth: $v \leftarrow v - c|v|v\,h/m$ per particle, isotropic. Same counting.
   - 0b. Contact with the hand. References:
     - Pinch and pull: a strip between two rigid pads of area $A_p$, squeezed by a normal force $N$ (force control) or to a gap (position control), then pulled in its plane. Coulomb's law: the grip holds up to $T = 2\mu N$. Measure the holding force, the creep below it, and the time to slip under a constant load (the quantity of Donaire et al. 2020).
     - A slow lift off a table: a strip lying flat, lifted by one end. The static reference is a heavy elastica, solved by shooting as Peirce's cantilever: the wrist reads $\lambda g z$ plus a correction over the bending length $(D/\rho_A g)^{1/3}$, with $\lambda = \rho_A w$. Keep the lift slow: for a straight strip on a table no kink forms (Singh and Hanna 2017), so the $\lambda v^2$ of a chain picked up from a pile does not apply.
     - Drag over a table: the force is $\mu$ times the weight on the table. Over the edge of the table, the capstan law $T_2 = T_1 e^{\mu\theta}$ applies.
   - Contact theory from the source, as experiment 4 did for MuJoCo's contact spring (source facts in Methods):
     - The pad holds about $A_p/\Delta x^2$ contacts, and each is a spring $k = K d^2/(1 - d)\, m$ with $m$ the vertex mass. So the stiffness of the pinch, about $\rho_A A_p K$, may not depend on $\Delta x$. The tofu's contact did. Check it, with the duplicate contacts at shared vertices counted.
     - MuJoCo's friction creeps below $\mu N$. Under Euler and implicitfast, with the elliptic cone, the steady creep velocity is $v = \tfrac{1-d}{d}\,\hat A\,|f_t| / (\mathrm{impratio}\cdot B_t)$, $B_t = 2/(d_w \tau)$, with $\hat A = 1/m$ for a vertex; `discrete` adds a factor in $h$. A light cloth therefore creeps fast. Find which counterterms (`impratio`, `solreffriction`) hold the creep fixed across $(h, \Delta x)$, and where refsafe stops them.
     - Genesis's IPC creeps at $v = \epsilon_v(1 - \sqrt{1 - T/\mu\lambda})$, with $\epsilon_v$ = 0.01 m/s by default, independent of $h$. Its contact stiffness is clamped by libuipc to a range that depends on $h$, the mean vertex mass, the scene size and $\hat d$: it flows with $h$ and $\Delta x$ without notice.
   - 0c. A policy in the loop. Run fixed scripts on the $(h, \Delta x)$ grid, with the nominal and with the renormalized parameters.
     - Open loop: the lift, drag and fold primitives of Garcia-Camacho et al. 2024, with a grasp through contact.
     - Closed loop: lift until the wrist force reaches a target, then stop.
     - Score the shape of the cloth against the converged run, the wrist force, and whether the grip holds (FabricBench's criteria: the grasp point rises at least half the commanded height and stays within 30 mm of the TCP). The claim to test: the same accuracy at a coarser, cheaper resolution.
   - Schemes:
     - MuJoCo flex, `dim="2"`: `discrete` and Euler; `elastic2d` "both", or "bend" with edge equalities; with and without the `ipc` flag (0a and the gripper only: under `ipc` the table is frictionless).
     - Meshes for MuJoCo: `type="grid"` (one diagonal), alternating diagonals through `type="direct"`, and an unstructured mesh through `type="mesh"`.
     - Genesis PBD cloth: 0a only. Its coupling to rigid bodies has no friction, so it cannot hold a cloth by friction. It remeshes the cloth isotropically to its particle size, so the mesh is not ours to choose.
     - Genesis FEM cloth through the IPC coupler (GPU): 0a and 0b.
1. The valid range of each parameter, from the literature.
   - The cloths of step 2 come with their own measurements: Coltraro et al. give the areal density, the drape ratio, the friction on paper and the elasticity of their four fabrics; the Household Cloth Object Set gives the mass and the drape ratio of the three rags of Blanco-Mulero et al.
   - $\rho_A$ is weighed, so it is fixed.
   - $D$ from the drape ratio, by simulating the same drape procedure; $Y$ from the elasticity under 0.5 kg; $\mu$ against paper from the incline. Each is widened to a range by the scatter of cotton, linen and terry in textile testing (below), which also gives the friction against silicone pads and against a table.
2. Fit to public data. Fit every parameter the engine offers to the motion. The fit is good if it lands inside the ranges of step 1; report what is left after the best fit. Then the R4R test: the fits at different $(h, \Delta x)$ fall on the flow of step 0.
   - Dynamics, first: Coltraro et al. 2025, free hanging and self-collisions. Drive the simulation by the two tracked upper corners; fit $D^\star$, $Y^\star$, the damping and the drag to every marker. The hitting takes are left out: their corners have gaps of up to 3.7 s.
   - Dynamics, second: Blanco-Mulero et al. 2024. They fitted MuJoCo 3.1.0 at 100 Hz, ran the same parameters at 10 and 1000 Hz, and concluded that "all engines need to be fine-tuned for the specific simulation frequency". Their study changes $h$ only; the mesh stays at 402 vertices.
     - First reproduce their fit in MuJoCo 3.1.0, which runs their model unchanged (checked, below). The robot paths have to be rebuilt from their quintic via-points, as their code does.
     - Their fit is outside the valid range: the masses are 30 to 200 times the real ones, while the fixed `viscosity` 0.3 alone damps the real cloth at 140 to 230 /s (below). Refit with the real mass and air, and see whether the other parameters then land in range.
     - The claim of R4R: fit once at 100 Hz, carry the parameters to 10 and 1000 Hz and to finer and coarser meshes by the flow of step 0, and do as well against the point clouds as a fresh fit at each, which we run ourselves.
   - Contact with a table: Coltraro's lay on a polished table and on a tablecloth (positions only), driven by corners 1 and 5; fit $\mu$ against each table. One take (cotton, full lay, low friction) loses corner 5 for 1.9 s.
   - Contact with a gripper: no public data gives the clamping force $N$ together with the holding force, so $T = 2\mu N$ cannot be tested directly. What can be tested:
     - Donaire et al. 2020: the load at which the grip of a towel, a T-shirt and silk stops holding for 20 s, and the times to slip just above it. The sim's creep law predicts a time to slip.
     - Herz and Kyosev 2024: the holding forces of 16 fabrics in one silicone gripper, with the static friction of each fabric on the same silicone. At a fixed $N$, Coulomb's law predicts that the holding force is in proportion to $\mu$.
     - G.O.G. 2024: peak pull-out forces of five garments, for comparison only.

## External data (checked 2026-11-28)

### Dynamics

- Coltraro, Borràs, Alberich-Carramiñana and Torras, Tracking cloth deformation: a novel dataset for closing the sim-to-real gap for robotic cloth manipulation learning, IJRR 44(9):1431 (2025). Local: `/data/datasets/coltraro2025_cloth_mocap/tracking_dataset/` (unpacked). License: CC BY-NC-SA 4.0 in the zip's License.txt; Zenodo says CC BY-SA 4.0.
  - 120 CSV files of OptiTrack markers, 120 Hz, in metres. 20 markers on an A2 cloth (0.42 × 0.594 m), 12 on an A3 (0.297 × 0.420 m), spaced equally.
  - Free hanging (64): shake or twist, slow or fast, held by two upper corners with a hanger or by hand, about 15 s. Twist speeds 0.08 to 0.31 m/s (Table 1).
  - Laid on a table (16, A2): from two corners about 10 cm up, partly or fully, on a polished table (low friction) or on a tablecloth (high friction), about 5 s.
  - Hit by a 75 cm stick (4, A2), about 40 s. Self-collisions (36, A2), about 7 s.
  - Y is up. Markers 1 to 5 are the top row of an A2 cloth (4 rows of 5), 1 to 4 of an A3 (3 rows of 4); the upper corners are 1 and 5 (A2), 1 and 4 (A3).
  - Gaps, our count: 0.3 % of the values in the free hanging, 0.8 % in the lays, 0.1 % in the self-collisions, 19 % in the hitting (48 % in denim). The upper corners have no gap in the free hanging and the self-collisions; in the lays at most 28 frames, except 230 frames of corner 5 in cotton, full lay, low friction; in the hitting up to 441 frames.
  - Fabrics (Fig. 1, measured by the method of Garcia-Camacho et al. 2024): density in kg/m², drape stiffness in %, friction, elasticity (short, long, diagonal):
    - polyester 0.1042, 7.6, 0.48, (3.3, 3.5, 11.3 %)
    - stiff cotton 0.3046, 43.7, 0.68, (0.8, 0.6, 1.0 %)
    - wool 0.1804, 12.9, 0.59, (5.0, 2.4, 9.6 %)
    - denim 0.3046, 20.3, 0.8, (10.0, 3.8, 11.3 %)
- Blanco-Mulero, Barbany, Alcan, Colomé, Torras and Kyrki, Benchmarking the sim-to-real gap in cloth manipulation, RA-L 9(3):2981 (2024), arXiv 2310.09543. Data: Zenodo 13823986, CC BY 4.0, unpacked in `/data/datasets/cloth_sim2real_benchmark/data/`. Code: `/data/datasets/cloth_sim2real_benchmark/code/` (GPL-3.0, commit 178a9b9, not run).
  - Only point clouds: 4950 `.ply` files (float64 xyz in metres, RGB), 16k to 92k points each, 30 fps. No robot poses, no gripper state, no forces.
  - Three rags of 0.5 × 0.7 m, called cotton, linen and chequered in the data. The paper's "towel rag" is the cotton rag, by elimination.
  - Two motions by two Frankas holding two corners of a 0.5 m edge: dynamic (3 trials each, 130 frames, 4.3 s) and quasi-static, lowered onto a table and dragged (3 trials each, 420 frames, 14 s).
  - The end-effector paths exist only as the quintic via-points in `manipulation_utils.py`. The frame of first contact with the table is in `bcm/utils.py` (dynamic: 89 to 99).
  - Their MuJoCo model (`bcm/assets/mujoco3_cloth.xml`): MuJoCo 3.1, flexcomp from `rag.obj` (402 vertices, 0.50 × 0.63 m), radius 0.01, edge equalities, `mujoco.elasticity.shell` plugin, `implicitfast`, $h$ = 10 ms, `viscosity` 0.3. Each corner is one vertex welded to a mocap body: no contact grasp.
  - Fitted per cloth and motion on trial 0 (`bcm/conf/envs/mujoco3.yaml`): mass 2.29 to 4.75 kg (dynamic) and 8.23 to 11.68 kg (quasi-static), against 0.05 to 0.08 kg real; Young's modulus 1.08e4 to 8.52e4 Pa. The two motions of one cloth get different parameters. The search space notes that the real mass "does not stabilise" and that GPa moduli "don't work at all".
  - `viscosity` 0.3 is fixed, not fitted (`density` 0), about 17,000 times that of air. Through the per-vertex fluid model it drags the 402 vertices with 11.4 N at 1 m/s, the same in 3.1.0 and 4.0.0. That is a damping rate of 140 to 230 /s for the real 0.05 to 0.08 kg, and 1.0 to 5.0 /s for the fitted 2.29 to 11.68 kg (measured 2026-11-28). This may be why the fitted masses are so large.
  - At 10 Hz MuJoCo diverged (Chamfer 235 m); at 100 and 1000 Hz, 0.083 and 0.087 m (our reading of the vector Fig. 5c). Section IV-C: "keeping the same fine-tuned simulation parameters as in Section IV-B".
  - Their code pins `mujoco==3.1.0` (`bcm.yml`). MuJoCo 4.0.0 cannot run the model unchanged: the shell plugin was removed in 3.3.3 (changelog, commit 7edbdd0a), and since 3.13.0 flex elasticity under `implicit` or `implicitfast` raises an error. MuJoCo 3.1.0 installs with uv on Python 3.11 and runs the plugin (checked 2026-11-28).
  - On `rag.obj`, 4.0.0's `elastic2d="bend"` with the changelog's migration $E' = \tfrac34(1-\nu)E$ gives the bending energy of 3.1.0's plugin to within 2 % for a cylinder at $\varphi$ = 0°, 45° and 90°. On this mesh the energy at 45° is 1.2 times that at 90°.
- Household Cloth Object Set: Garcia-Camacho, Borràs, Calli, Norton and Alenyà, RA-L 7(3):5866 (2022), arXiv 2111.01527. Local: `/data/datasets/household_cloth_object_set/`. Table II: towel rag 0.05 kg, linen rag 0.05 kg, chequered rag 0.08 kg (three of them), all 0.5 × 0.7 m. No fibre composition and no product names are given.
- Drape ratios of these rags (`TextileCharacterization_repo/code/results/HCOS/HCOS_stiffness_data.csv`, no license file): towel 21.4 %, linen 16.2 %, chequered 15.5 %, folded to 35 × 50 cm on a 21 cm disc. Their friction and elasticity are published only as the range of the whole set.

### Material, from textile testing

Bending rigidity in µN·m per unit width (1 gf·cm²/cm = 98.07 µN·m; 1 mg·cm = 0.0981 µN·m).

- Plain cotton:
  - $\rho_A$ 69 to 200 g/m² (Fabric-101; Clyde et al. 2017, poplin 103.6; Utkun 2021; Kan 2013; Herz and Kyosev 2024). Thickness 0.18 to 0.98 mm, depending on the pressure.
  - KES B 0.022 to 0.125 gf·cm²/cm, 2.2 to 12 µN·m (Utkun 2021 Table 7; Kan 2013). Cantilever 4.8 to 9.6 µN·m (Szymańska 2025; Tania 2021, our conversion).
  - Cusick drape 45 to 55 % (Özdil 2014).
- Linen: $\rho_A$ 117 to 240 g/m² (Fabric-101; Ukey 2023; Vasile 2024; Truncytė 2006). Cusick 76 to 82 % (Ukey 2023). No open KES B or cantilever value found.
- Cotton terry: $\rho_A$ 318 to 626 g/m², thickness 2.6 to 6.8 mm (Krasteva 2015). Cantilever 7.0 to 20.5 µN·m, the mean of warp and weft (Krasteva Table 5; warp alone 6.9 to 28.8); 1.9 to 2.9 µN·m (Mutlu Ala 2017). The two disagree 3 to 7 times.
- Datasets of fitted parameters, downloaded:
  - Wang, O'Brien and Ramamoorthi 2011: 10 fabrics, none plain cotton, linen or terry. `/data/datasets/wang2011_cloth_measurements/`, academic license. The fitted tables are ARCSim 0.2.1's `materials/*.json`, in `/data/datasets/arcsim_0.2.1_materials/`.
  - Feng et al. 2022: 618 fabrics, density and bending, no names. `/data/datasets/feng2022_drape_tester_618fabrics/`.
  - Fabric-101, Gong et al. 2026: 103 fabrics, including plain cotton C01 and C14, linen L01 and L04, terry E-KN06; raw curves; the units of its fitted parameters are not stated; CC BY-NC 4.0. `/data/datasets/fabric101_gong2026/repo`.
  - Bouman et al. 2013: 30 fabrics hanging in a fan's wind, with lab-measured bending and areal density. `/data/datasets/bouman2013_mit_fabric_properties/`.
- Hysteresis: Miguel et al. 2013 (cotton warp, Dahl-type internal friction). Damping of a freely swinging cotton, linen or terry: not found.
- Air: Wilk and Skuta 2009, small flags in a wind tunnel at 4.3 to 13.4 m/s, $c_x$ 0.13 to 0.55, falling with the speed.

### Friction

- Fabric on silicone gripper pads: static 0.86 to 1.28 for 16 fabrics; plain cotton 0.95 to 1.19, plain linen 0.91 (Herz and Kyosev 2024 Table 2, sled at 100 mm/min).
- Plain cotton on itself: static 0.74 to 1.14, kinetic 0.39 to 0.70 (Sülar 2013 Table 4).
- On a table: cotton twill on glass 0.27 and PMMA 0.21, linen on glass 0.31 and PMMA 0.15, dynamic (Truncytė 2006 Table 2); two towels on a melamine-faced chipboard table 0.28 and 0.24, and on a rough tablecloth 1.23 and 1.20, static (Petrik 2016 Table III).
- Terry on terry: static 0.33 to 0.51, dynamic 0.86 to 1.02, as printed (Krasteva 2015 Table 4).

### Contact with a gripper

- Donaire, Borràs, Alenyà and Torras, A versatile gripper for cloth manipulation, RA-L 5(4):6520 (2020), Table II. A hand-held gripper, Dynamixel in position control; a bag clamped 15 cm from the edge, loaded in steps from 7.5 to 250 g, held 20 s, 3 trials. With friction on, a 60 g towel held up to +147.5 g, then slipped after 10.66 s (+185 g) and 1.33 s (+202 g), and at once at +250 g. The clamping force is not given.
- Herz and Kyosev, CDATP 5(2):222 (2024), Table 2: a pneumatic Schmalz OFG silicone finger gripper, 100 mm discs of 16 fabrics, the largest force before the first slip, 1.71 to 2.87 N. The clamping force is not given.
- Lee, Chen, Chen and Rojas, G.O.G., RA-L 9(3):2415 (2024), arXiv 2401.10702: a garment edge in a jig, pulled out by a UR5 through a Robotiq FT 300; peak force before failure, capped at 30 N. Read from the bar chart: 9.5 to 27 N for five garments, single edge, high-friction mode. No $N$, speed or time series.
- FabricBench, arXiv 2611.31577: grasp, hold and release criteria, and a creep law $v = \epsilon(1 - \sqrt{1-\rho})$ in its solver. Code not released (2026-11-28). No comparison with real forces.

### Checked and not usable

- Forces in cloth datasets: the ICRA 2024 Cloth Competition (Zenodo 14621179, 83 GB) and RGBench (HF, 6.7 GB) record no force; RH20T's cloth wiping presses down, it does not drag. The force-time curve of a cloth lifted off a table exists only for a pneumatic gripper, whose air flow dominates (Mykhailyshyn 2025).
- Runia et al. 2020, real flags in wind: the fabric's properties are unknown ("the cloth's internal parameters are unknown").
- Miguel et al. 2012 and Clyde et al. 2017: their measured curves are not released.

## Methods

### MuJoCo 4.0.0 (installed package and source at tag 4.0.0, commit ac60eae, in `/data/src/mujoco-4.0.0`)

- Bending (`elastic2d` bend or both) is the quadratic cotangent bending of Wardetzky and Bergou, with Garg's correction for a curved rest shape (`user_flexcomp.cc`, `engine_passive.c`). Per interior edge a constant 4 × 4 stencil $D\cos\bar\theta_e/(A_1 + A_2)\, c\,c^\mathsf{T}$, $D = Et^3/12(1-\nu^2)$. Its Hessian is constant.
- Stretching is Saint Venant-Kirchhoff on the triangles, through squared edge lengths, with $\mu$ and the plane-stress $\lambda^\ast = E\nu/(1-\nu^2)$. 4.0 fixed both constants.
- `type="grid"` cuts every quad along the same diagonal. Other meshes: `type="mesh"`, `gmsh` or `direct`.
- Every flexcomp vertex has mass `mass`/N and an inertia box of side 2 × `inertiabox`, independent of `radius`. Fluid forces act on these vertex bodies (`mj_fluid`, `mj_inertiaBoxFluidModel`).
- Flex-geom contacts: a plane gets one per vertex; other geoms one per triangle and geom pair (sphere 1, capsule up to 5, box up to 50, mesh, cylinder and ellipsoid 1). Contacts at shared vertices are not merged. Default flex condim 3, friction 1, radius 5 mm.
- Under `discrete`, refsafe bounds a contact's stiffness by $K \le 1/(h^2 d)$ (and changes $B$). Friction rows have $K = 0$; their $B = 2/(d_w \tau)$, with $\tau$ from `solreffriction` or `solref`. The regularizer uses $\hat A$ from `body_invweight0`, $1/m$ for a vertex; the friction rows divide it by `impratio`.
- The creep formula of 0b was checked against a ball on a slide joint to 7 digits, for Euler, implicitfast and `discrete` (2026-11-28).
- The `ipc` flag (experimental, `discrete` only, 2D flex): continuous collision detection, so a flex cannot tunnel. It solves flex with flex, self-contact, and flex with geoms welded to the world, all frictionless. Moving bodies, such as a gripper, keep the usual contacts with friction. Gap 1 mm, band 3 mm, tolerance 0.05 m/s, fixed in `engine_ipc.c`.
- Flex elasticity needs `discrete` or Euler: since 3.13.0, `implicit` and `implicitfast` raise an error (changelog). Blanco-Mulero's model ran under `implicitfast` in 3.1.0. 4.0 also changed the 2D elastic constants (changelog, commit 7ba52b533): bending used $\tfrac34(1-\nu)D$ before.

### Genesis 1.5.0 (installed source) and libuipc v0.0.30 (`/data/src/libuipc`, commit 6c288d9)

- PBD cloth (`gs.materials.PBD.Cloth`): remeshed isotropically to `particle_size` (1 cm by default); equal particle masses. Each substep: 4 Jacobi stretch iterations, 1 bending iteration (Müller's dihedral constraint, flat at rest), then collision. No multiplier is kept, as experiment 4 found. Collision with rigid bodies pushes particles out along the SDF with no friction ("friction left out", `legacy_coupler.py`).
- FEM cloth (`gs.materials.FEM.Cloth`) through the IPC coupler, GPU only:
  - libuipc's StrainLimitingBaraffWitkinShell, not neo-Hookean. `thickness` is the half thickness, so $\rho_A = 2\rho t$.
  - `bending_stiffness` goes to libuipc as $\kappa$ of a dihedral energy; libuipc's convention is $\kappa = D/2$, and Genesis does not convert.
  - Defaults: $\hat d$ = 0.01 m, $\epsilon_v$ = 0.01 m/s, `contact_resistance` 1e9. libuipc clamps the contact stiffness to $[\kappa_{lb}, 100\kappa_{lb}]$, where $\kappa_{lb}$ grows with the mean vertex mass and as $1/h^2$, and depends on the scene size and $\hat d$ (`global_contact_manager.cu`). In one test scene, 1e9 became about 2e4 at $h$ = 10 ms and 2e6 at 1 ms. The warning is hidden by Genesis's log level.
  - Rigid bodies enter as ABD, with friction, so fingers can pinch the cloth. Cloth vertices cannot be pinned. No air drag.

## References

- Verbin, Eur. J. Phys. 36:015005 (2015), arXiv 1412.1846: the hanging chain.
- Timoshenko, Vibration problems in engineering, 2nd ed. (1937), pp. 344–345. Leissa, Vibration of plates, NASA SP-160 (1969).
- Peirce, The "handle" of cloth as a measurable quantity, J. Textile Inst. Trans. 21(9):T377–T416 (1930). Lammens et al., Textile Res. J. 84(12):1307 (2014).
- Singh and Hanna, J. Mech. Phys. Solids 106:46 (2017), arXiv 1611.03332: a rope lifted off a plane.
- Grinspun, Hirani, Desbrun and Schröder, Discrete shells, SCA 2003. Bergou, Wardetzky, Harmon, Zorin and Grinspun, A quadratic bending model for inextensible surfaces, SGP 2006.
- Chen, Kry and Vouga, Locking-free simulation of isometric thin plates, arXiv 1911.05204 (2019).
- Li, Kaufman and Jiang, Codimensional incremental potential contact, ACM TOG 40(4) (2021): with 0.01 times the bending stiffness, locking stays severe at 246K nodes (supplement, Fig. 3).
- Chen, Suri, Wu, Cheng, Abeysirigoonawardena, Vouga, Levin and Pai, SpringTime, Graphics Interface 2026, arXiv 2507.21288: locking is larger on coarse meshes and for folds across the mesh edges.
- Bhat, Twigg, Hodgins, Khosla, Popović and Seitz, Estimating cloth simulation parameters from video, SCA 2003, pp. 37–51.
- Garcia-Camacho, Longhini, Welle, Alenyà, Kragic and Borràs, Standardization of cloth objects and its relevance in robotic manipulation, ICRA 2024, pp. 8298–8304, arXiv 2403.04608.
- Miguel et al., Data-driven estimation of cloth simulation models, CGF 31(2) (2012); Modeling and estimation of internal friction in cloth, ACM TOG (2013).
- Clyde, Teran and Tamstorf, Modeling and data-driven parameter estimation for woven fabrics, SCA 2017.
- The PDFs are in `papers/`.
