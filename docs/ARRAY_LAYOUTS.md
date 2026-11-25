# Array Layouts

The unsteady solvers have two paths through each time step. With `run(use_arrays=False)` (the default), each time step builds `Airplane`, `Wing`, `WingCrossSection`, and `Panel` objects and the solver reads its geometry from them. With `run(use_arrays=True)`, each time step's geometry, loads, and wake are computed as arrays, and the objects a user reads are built from those arrays only when something reads them. This document describes how the array path lays out its data. Read [Axes, Points, and Frames](AXES_POINTS_AND_FRAMES.md) first: every array's name follows its conventions.

Every stage inside the time step loop reads arrays, and the array path builds, copies, and writes no `Airplane`, `Wing`, `WingCrossSection`, `Panel`, or `SteadyProblem` during the run (a test counts them). The objects a user reads are written or built after the run, as the section [Objects after the run](#objects-after-the-run) describes. On the full FAAV and pterosaur free flight flap driven examples it runs about 1.8 to 1.9 times faster than the object path (8.5 s against about 15.2 s, and 5.7 s against 10.5 s, measured 2026-11-24), and holds about 3.6 times less memory per time step.

## What runs inside the time step loop, and what doesn't

Only what runs at every time step (or every free flight coupling trial) has an array version: the movements' per time step parameters, meshing, the solver's per Panel geometry, the loads, the wake, the torsional springs, and the free flight coupling's geometry rebuilds. Everything outside the loop stays on objects with no array version: problem setup, the steady solvers, trim, convergence, output, plotting, and serialization. A new feature only needs an array version if it runs inside the loop.

## The four kinds of data

| Kind | Built | Holds | Code |
|---|---|---|---|
| Plans | once per run, at the start of `run` | everything fixed by the problem's structure | `MeshPlan` (`_mesh_arrays.py`), `LayoutPlan` (`_step_geometry.py`) |
| Step parameters | once per geometry build (each time step, and each coupling trial) | the values the movements produce that can change between time steps | `WingCrossSectionStepParameters`, `WingStepParameters`, `AirplaneStepParameters` (`_step_parameters.py`) |
| Step geometry | from the step parameters | each Wing's Panel corners, and every per Panel quantity in global order | `_step_geometry.py` |
| Solver state | per aerodynamic evaluation | the solver's existing attributes (`stackCpp_GP1_CgP1`, the influence matrices, the strengths, the wake stacks), unchanged in name and meaning | the solver |

Plans and step parameters are immutable: NamedTuples whose arrays are read only. Step geometry is immutable once built, so a free flight coupling trial builds new geometry rather than copying an Airplane.

The plans are run scoped: the solver builds them at the start of a run with `use_arrays=True` and clears them when the run ends, so they are never saved in a `.psz` file. The solver's step geometries are run scoped the same way.

## Step parameters

The movement classes evaluate their motion in private `_generate_*_parameters_at_time_step` methods, and their object builders (`generate_*_at_time_step`) build the objects from those values, so both paths read the motion from one place.

- A `WingCrossSectionStepParameters` holds `Lp_Wcsp_Lpp` and `angles_Wcsp_to_Wcs_ixyz` (including any structural deformation).
- A `WingStepParameters` holds the Wing's `Ler_Gs_Cgs` (after the rotation point offset adjustment) and `angles_Gs_to_Wn_ixyz` (for a flap driven Wing, with the recorded flap angle), and its WingCrossSections' values stacked as `stackLp_Wcsp_Lpp` and `stackAngles_Wcsp_to_Wcs_ixyz`, (K, 3) with one row per WingCrossSection.
- An `AirplaneStepParameters` holds the Airplane's `Cg_GP1_CgP1` and one `WingStepParameters` per Wing movement.

Everything else (each WingCrossSection's `Airfoil`, chord, panel counts and spacings, control surface settings; each Wing's symmetry settings and chordwise panels; each Airplane's weight and reference dimensions) is fixed for the run and is read from the movements' base objects.

A movement's Wing movements correspond one to one to its base Airplane's Wings, which are already symmetry processed: a type 5 Wing has already been split into a type 1 Wing and its type 3 reflection, each with its own movement. So `AirplaneStepParameters.wings[i]` describes the Airplane's `i`th Wing.

The coupled problems (aeroelastic, flap driven, free flight, and their combinations) find each time step's geometry during the run, from the structural deformation, the flap angles, or the body's motion, so the solver can't get it from the movement. Each one records, for every time step, the `AirplaneStepParameters` its first Airplane is built from, with `_CoupledUnsteadyProblem._record_airplane_step_parameters`, and the coupled solver reads them with `get_airplane_step_parameters`:

| Problem | Records |
|---|---|
| `AeroelasticUnsteadyProblem` (and the flap driven ones) | time step 0 when it is created, then each later time step in `_build_recorded_airplane`, which builds the deformed Airplane from the same parameters |
| `FreeFlightUnsteadyProblem` | time step 0 when it is created, then, on the array path, each next time step's prescribed parameters in `initialize_next_problem` (a coupling trial changes only the `OperatingPoint`, so it meshes the same parameters) |

On the array path, the coupled problems record each time step's parameters without building its Airplane, and build the time step from them when it is first read (see [Objects after the run](#objects-after-the-run)). The record isn't saved in a `.psz` file (saving builds every time step first), and a loaded problem has `None` in its place.

## The global Panel order

The solvers' per Panel arrays (and the `LayoutPlan`) use one global order:

1. Airplanes in order.
2. Within an Airplane, its Wings in order, after symmetry processing.
3. Within a Wing, row major: chordwise row outer (leading edge first), spanwise column inner.

So Panel `(row, column)` of the Wing at position `w` in the global Wing order sits at index `wing_offsets[w] + row * num_spanwise_panels + column`, where `wing_offsets` and each Wing's `(num_chordwise_panels, num_spanwise_panels)` come from the `LayoutPlan`.

A Wing's spanwise column order is `mesh_wing`'s Panel grid order, which depends on its symmetry type:

| Symmetry type | Spanwise column order |
|---|---|
| 1 | the Wing's wing sections root to tip, left to right |
| 2 and 3 (mirror only) | reversed: tip to root, with each Panel's outer corners on its left |
| 4 | the mirrored half (reversed, tip to root) to the left of the main half (root to tip), in one grid |

The torsional springs' strips run root to tip, so for a mirror only Wing the spanwise axis is flipped at the springs' boundary.

## Edge flags

The `LayoutPlan`'s `is_leading_edge`, `is_trailing_edge`, `is_left_edge`, and `is_right_edge` are global (num_panels,) arrays of bools, one entry per Panel in global order: the Panel's Wing's first chordwise row, last chordwise row, first spanwise column, and last spanwise column. They never change during a run, so they are computed once.

The `LayoutPlan` also holds each Panel's neighbors in its Wing, `right_neighbors`, `front_neighbors`, `left_neighbors`, and `back_neighbors` (global indices: one column to the right or left, one row of num_spanwise_panels in front or behind, or the Panel's own index on that edge, which the edge flags mark), and each Airplane's Panels' (start, stop) range, `airplane_panel_ranges`. The loads use them to find neighbors' strengths and to total each Airplane.

## Mesh plans

Each Wing's `MeshPlan` holds its symmetry type (1 to 4), its chordwise and spanwise panel counts (the spanwise count doubled for a type 4 Wing, whose grid holds both halves), the normalized chordwise coordinates of the Panels' edges (from 0.0 at the leading edge to 1.0 at the trailing edge, uniform or cosine spaced exactly as `mesh_wing` spaces them), and, for each Panel column of the main half, its wing section and the normalized spanwise coordinates of its inner and outer edges within that wing section.

It also holds everything else about the Wing's mesh that is fixed for the run:

- `gridMclPoints_Wcs_Lp`, (K, num_chordwise_panels + 1, 3), with one entry per `WingCrossSection`: its mean camber line points at the chordwise coordinates (in its own axes, relative to its leading point), with its control surface deflected and scaled by its chord. A `WingCrossSection`'s `Airfoil`, chord, and control surface never change during a run, so no `Airfoil` work happens inside the loop.
- `gridMirroredMclPoints_Wcs_Lp`, the same points for a type 4 Wing's mirrored half when some asymmetric control surface has a non zero deflection (so it deflects the other way on that half), and otherwise `None`.
- The symmetry plane, `symmetryNormal_G` and `symmetryPoint_G_Cg` (`None` for a type 1 Wing).

A type 5 Wing needs nothing extra: its Airplane has already split it into a type 1 Wing and its type 3 reflection, each with its own movement, step parameters, and `MeshPlan` (the reflection's `WingCrossSection`s already carry its reversed asymmetric deflections).

## Meshing

`mesh_wing_arrays(plan, parameters)` meshes one Wing from its `MeshPlan` and its `WingStepParameters`, returning a `WingMesh`: four read only (num_chordwise_panels, num_spanwise_panels, 3) arrays, `gridFlpp_G_Cg`, `gridFrpp_G_Cg`, `gridBlpp_G_Cg`, and `gridBrpp_G_Cg`, the Panels' corners (in geometry axes, relative to the CG) in the Wing's Panel grid order (see the column order table above), plus the Wing's `span` and `WnZ_G`, computed as `Wing.span` and `Wing.WnZ_G` compute them, which the step geometry needs for the standard mean chord. `wing_mesh_from_wing(wing)` reads the same `WingMesh` from a Wing that is already built. It does the same operations as `mesh_wing`, in the same order, for every `WingCrossSection` and Panel column at once, and its corners equal `mesh_wing`'s bit for bit on the development machine (its tests allow 1.0e-12 m, since batched NumPy calls aren't guaranteed to round identically on every platform).

It first calls `_step_parameters.validate_wing_step_parameters`, which raises the error the object path's constructors would raise for the same values (non finite values, angles outside [-90, 90] degrees, a non zero root `WingCrossSection` position or angles, or a non positive spanwise offset), with the same type and message. It also raises if the values change the Wing's symmetry type, which the plans assume is fixed. The object path has no such check (its Airplane would process the Wing as its new type).

**The mesh cache.** The solver keeps each Wing's last mesh for the run (`_wing_mesh_cache`, run scoped like the plans), keyed by `_step_parameters.wing_step_parameters_key`: each parameter array's shape, dtype, and bytes, so a key only matches bit for bit the same values (-0.0 doesn't match 0.0). `_wing_meshes_from_parameters` returns the cached `WingMesh` when a Wing's key matches its last one, and meshes it otherwise. The Airplane's parameters are validated at every build, hit or miss. Within a free flight time step, the coupling trials mesh every Wing again with the same parameters (a Wing's deformation and flap angle advance once per time step, before the trials, and its mesh is in the first Airplane's geometry axes, which the trial's body state doesn't change), so about 85% of the FAAV's meshings are hits. It relies on the wings being loosely coupled: with strongly coupled wings, whose angles change within the trials, every build would be a miss (`docs/MASTER_TODO.md` item 8).

## Step geometry

`build_step_geometry` turns every Wing's `WingMesh` (and its Airplane's `Cg_GP1_CgP1`) into a `StepGeometry`, every per Panel quantity in global Panel order: the corners and collocation points (in the first Airplane's geometry axes, relative to its CG), unit normals, areas, each Wing's standard mean chord, each Airplane's CG at this time step (`stackAirplaneCg_GP1_CgP1`, from its first Wing's, which a formation's moving Airplanes change every time step), and the bound ring vortex corners (whose trailing edge back corners need the previous time step's geometry and the freestream velocity). It replaces `SteadyProblem`'s per Panel transformation, `set_derived_GP1_caches`, `Wing.standard_mean_chord`, and `_initialize_panel_vortices_at`. `bound_r_c0s`, `seed_points`, `wake_vortex_ages`, and `katz_geometry` derive the rest of what `_collapse_geometry` and `_collapse_geometry_katz_data` write.

Everything is bit for bit the object path's except the norms (in the unit normals, areas, projected areas, and Katz lengths), which are vectorized and round about one unit in the last place differently from the object path's one row at a time `np.linalg.norm`. Whole runs agree to about 1e-15 relative. A free wake can amplify that exponentially on a chaotic problem (a flapping Wing's rolled up wake grew it to about 3e-7 over 42 time steps), as any perturbation would.

Where each time step's meshes come from:

| Solver | Meshes |
|---|---|
| The plain solver | `mesh_wing_arrays` from the movement's `AirplaneStepParameters` |
| The coupled solvers (aeroelastic, flap driven, free flight) | `mesh_wing_arrays` from the `AirplaneStepParameters` the problem recorded (see Step parameters) |

Both go through `_wing_meshes_from_parameters`, which first checks each `AirplaneStepParameters` with `validate_airplane_step_parameters`. `wing_mesh_from_wing` reads a `WingMesh` from a Wing that is already built. No solver calls it, and the tests use it to give the array functions exactly the object path's geometry.

The solver builds the first time step's geometry at the start of the run, and each later time step's one time step early (its wake shedding reads the next time step's ring vortex corners). It keeps only the previous, current, and next time steps' geometry.

## Reuse across coupling trials

A free flight time step's coupling trials each solve the next time step with a different trial body state. Two things the trials share are reused, both bit for bit:

- **The step geometry (array path).** Every trial builds the next time step's geometry from the same WingMeshes (the mesh cache returns the same objects), the same CGs, and the same previous geometry, so everything `build_step_geometry` finds is the same except the trailing edge Panels' ring vortices' back corners, which follow the trial's freestream velocity. `_build_step_geometry_at` keeps the last full build (`_last_step_geometry_build`, run scoped) and, when the time step, the WingMeshes (by identity), the CGs (by bytes), and the previous geometry (by identity) are the same, calls `rebuild_step_geometry_for_freestream`, which keeps the build's arrays and finds only those corners again, with `build_step_geometry`'s own formulas (`_set_trailing_edge_back_corners`).
- **The Wing Wing influences (both paths).** The influence matrix's columns each come from one bound ring vortex alone. `_calculate_wing_wing_influences` keeps the last calculation (`_WingWingInfluenceCache`, run scoped) and, when the collocation points, unit normals, strengths, and nu are bit for bit the same, recomputes only the columns of the rings whose corners or core radii changed (on the FAAV example, 58 trailing edge rings of 324) and copies the rest. It makes the full calculation with an image surface (whose reflection follows the trial's body state) and when the last calculation counted unexpected singularities (so the logged counts are always the full calculation's). The influences are read only, since the cache keeps them.

The wake can't be reused this way: the solver works in the body's axes, so each trial's body state moves every wake corner relative to the Wings.

## Loads

Both force methods already work on the solver's per Panel arrays, except for the parts that walked the Panels, which `_loads_arrays.py` replaces with the same operations in the same order, so they are bit for bit the object path's:

| Function | Replaces |
|---|---|
| `effective_line_vortex_strengths` | the Kutta-Joukowski method's Panel loop (`_effective_line_vortex_strengths_from_panels` on the object path) |
| `chordwise_vorticity_densities`, `spanwise_vorticity_densities`, `chordwise_vorticity_differences` | the Katz method's three Panel loops |
| `step_loads` | `process_solver_loads`: each Panel's loads in wind axes, each Airplane's totals and coefficients, plus each Wing's totals (for the flap drives' fluid torque) |
| `strip_leading_edge_points`, `strip_leading_edge_moments` | the aeroelastic solver's `_populate_slep_positions` and its strip leading edge moments |

`step_loads` takes each Airplane's moments about that Airplane's CG at the current time step, which the solver reads from the current `StepGeometry`'s `stackAirplaneCg_GP1_CgP1`. Each Airplane's reference dimensions are fixed for the run, so they come from the first time step's Airplanes.

The solver keeps the current evaluation's `StepLoads` (run scoped, like the plans), and on the array path the coupled problems read their loads from it:

- the free flight problems' `_assemble_interval_loads` (for the snapshot, each coupling trial, and the prescribed phase) reads the first Airplane's total force and moment from `stackAirplaneForces_W` and `stackAirplaneMoments_W_CgP1`;
- the flap driven problems' `_calculate_flap_fluid_torque` reads each Wing's total force and moment from `stackWingForces_GP1` and `stackWingMoments_GP1_CgP1` (summed over the Wing's Panels in the same order as the object path's Panel loop), and the Wing's transformation from its recorded step parameters with `_mesh_arrays._T_pas_G_Cg_to_Wn_Ler`.

Both give exactly the object path's values on the same time step. On the array path, `external_loads_fn` gets an `Airplane` built from the time step's parameters with the loads written onto it (`_build_loaded_airplane`), only when a function is given.

The run loop keeps each time step's official `StepLoads` (`_step_loads_history`), taken right after the time step's solve, before a coupled problem's trials overwrite the solver's. `_finalize_loads` averages them, and they are written onto the objects after the run.

## Ground effect

Ground effect has no array code of its own. Both paths add the image surface's contribution inside the influence and induced velocity functions they share (`_calculate_wing_wing_influences`, `_calculate_wake_wing_influences`, `_calculate_induced_velocity`, and the Katz method's `_calculate_chordwise_induced_velocity` and `_calculate_wake_induced_velocity`): each reflects the solver's bound and wake ring vortex stacks with the current `OperatingPoint`'s `surfaceReflect_T_act_GP1_CgP1` and calls the same kernels. No array function calls a kernel directly. A free flight coupling trial sets the current `OperatingPoint` to its own, so each trial reflects across the ground at its own height. A new function that computes an induced velocity must reflect the same way, or call one of these.

## Wake

Each Wing's wake is a grid of wake ring vortex points, (R + 1, N + 1, 3) for R wake rows and N spanwise Panels, whose cells are its wake ring vortices. On the object path each grid is stored on the next time step's Wing (`gridWrvp_GP1_CgP1`). On the array path the solver keeps them itself, run scoped and keyed by time step (only the current and next time steps'), and doesn't set them on any Wing during the run. After the run, each time step's grids are rebuilt from the solver's per time step wake ring vortex corners with `wake_grids_from_ring_vortices` (the corners are four shifted slices of the grid, so the grid is put back bit for bit, with no extra storage).

`_wake_arrays.py` replaces the two Python loops of the wake update, with the same operations in the same order, so the results are bit for bit the object path's:

| Function | Replaces |
|---|---|
| `new_wake_rows` | the loop over the trailing edge Panels that builds each Wing's new row |
| `first_wake_grid`, `next_wake_grid` | the rest of `_populate_next_wake_grids_on_wings` (the object path's per Wing loop): the first time step's two rows, and afterwards the convected grid behind the new row, truncated to `max_wake_rows` |
| `wake_ring_vortices` | `_populate_next_airplanes_wake_vortices`' loop over every Wing's wake rows: the next time step's wake ring vortex corners, strengths, and initial core radii |

The velocities that convect the grids (the kernels' induced velocities, the freestream, and the body rate) stay in the solver, unchanged.

## Objects after the run

The array path writes the objects a user reads once, after the run, in `_write_array_results_to_objects`, which hands each time step's kept loads and wake ring vortex corners to `_store_step_results`:

| Solver | After the run |
|---|---|
| The plain solver | Its movement built every time step's Airplanes before the run, so `_write_step_results` writes each time step's loads and wake point grids onto them, exactly as the object path writes them during the run. |
| The coupled solvers | The problem keeps each unbuilt time step as an `_ArrayStep` (its `OperatingPoint`, and after the run its loads and wake corners). `steady_problems` and `get_steady_problem(step)` build a time step the first time it is read, with `_array_step_airplane` (by default the object path's builder, `_airplane_from_parameters`, on the recorded step parameters, and for a `FreeFlightUnsteadyProblem` the movement's own Airplane, as the object path uses), write its loads and wake point grids onto it, and keep it. `get_operating_point(step)` reads a time step's `OperatingPoint` without building it. |

Either way, the solver's `panels`, `current_airplanes`, and `current_operating_point` are the last time step's afterwards, as on the object path. Saving builds every time step first, so a `.psz` file holds exactly what an object path run's does. A test compares everything a user can read (every public attribute and property reachable from the solver, across every time step's objects) on both paths.

Reading every time step of a coupled array path run builds about what the object path holds anyway (about 1,600 kB per time step on the FAAV), so a run that is never read pays nothing for it.

## Springs

The torsional spring solve already works on arrays, except for what it read from the current time step's Wings: each Panel's area (for its mass) and each strip's span width (the length of its first chordwise row Panel's front leg). `_springs_arrays.py` computes them from the time step's `StepGeometry`, with the same formulas and operation order:

| Function | Replaces |
|---|---|
| `wing_panel_areas_and_strip_widths` | the Panel reads in `_calculate_mass_matrix` (`Panel.area`) and `_calculate_strip_inertias` (`Panel.frontLeg_G`) |
| `panel_mass_matrix` | `_calculate_mass_matrix` |
| `strip_inertias` | `_calculate_strip_inertias` |

The areas come from the step geometry's vectorized norms, so they round about one unit in the last place differently from `Panel.area`. The strip widths are bit for bit, and the masses and inertias are bit for bit given the same areas. A mirror only Wing's Panel grid runs tip to root, so its strip widths are read from its last column to its first, and `_calculate_wing_deformation` flips its mass matrix, as on the object path. A strip's width enters the springs only through its flapping axis inertia, which scales the prescribed flapping acceleration.

On the array path, `_calculate_wing_deformation` reads each Wing's counts, chords, and mirroring, which are fixed for the run, from its movement's base Wing, rather than from the current time step's Wing.

## Error checking

The array path raises the same errors as the object path. The object path checks most per time step values implicitly, in the constructors of the objects it builds, so each array function that replaces object building has a validation function that repeats those checks, and a test that it raises exactly where building the objects raises. The step parameter checks are `validate_wing_step_parameters` and `validate_airplane_step_parameters` in `_step_parameters.py`. `FreeFlightMovement` and `FreeFlightAeroelasticMovement`, which no longer build their Airplanes when they are created (their `airplanes` properties build them the first time they are read), run `_validate_prescribed_motion` instead: it checks every prescribed time step with `validate_airplane_step_parameters`, then every Wing's symmetry type at every time step, so invalid prescribed motion still raises when the movement is created.

## The copy dimension and the array module

For GPU batches (Part B of `experimental/plan_arrays_and_gpu.md`), the array functions will accept an optional leading copy axis, (B, N, 3) rather than (N, 3), and broadcast over it. Plans have no copy axis, since every copy shares the problem's structure. The functions Part B runs on the GPU will take an array module (`xp`, NumPy or CuPy). Index arithmetic from the plans stays NumPy and is moved to the device once per run.

## Conventions every array function follows

- Its inputs and outputs are named by [Axes, Points, and Frames](AXES_POINTS_AND_FRAMES.md), with the `stack` prefix for (N, 3) global arrays, `grid` for (M, N, 3) per Wing arrays, and `list` for per time step lists.
- Every array it creates has an explicit `dtype`.
- It is a pure function: its inputs are its arguments, and its outputs are its return value.
- It is unit tested against the object code it replaces, on every relevant fixture and on random geometry.
