# Array Layouts

The unsteady solvers have two paths through each time step. With `run(use_arrays=False)` (the default), each time step builds `Airplane`, `Wing`, `WingCrossSection`, and `Panel` objects and the solver reads its geometry from them. With `run(use_arrays=True)`, each time step's geometry, loads, and wake are computed as arrays, and the objects a user reads are built from those arrays only when something reads them. This document describes how the array path lays out its data. Read [Axes, Points, and Frames](AXES_POINTS_AND_FRAMES.md) first: every array's name follows its conventions.

The array path is being built in stages. Until it is complete, `use_arrays=True` builds the plans below and runs the stages that aren't converted yet on the objects.

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

## Step geometry

`build_step_geometry` turns every Wing's `WingMesh` (and its Airplane's `Cg_GP1_CgP1`) into a `StepGeometry`, every per Panel quantity in global Panel order: the corners and collocation points (in the first Airplane's geometry axes, relative to its CG), unit normals, areas, each Wing's standard mean chord, and the bound ring vortex corners (whose trailing edge back corners need the previous time step's geometry and the freestream velocity). It replaces `SteadyProblem`'s per Panel transformation, `set_derived_GP1_caches`, `Wing.standard_mean_chord`, and `_initialize_panel_vortices_at`. `bound_r_c0s`, `seed_points`, `wake_vortex_ages`, and `katz_geometry` derive the rest of what `_collapse_geometry` and `_collapse_geometry_katz_data` write.

Everything is bit for bit the object path's except the norms (in the unit normals, areas, projected areas, and Katz lengths), which are vectorized and round about one unit in the last place differently from the object path's one row at a time `np.linalg.norm`. Whole runs agree to about 1e-15 relative. A free wake can amplify that exponentially on a chaotic problem (a flapping Wing's rolled up wake grew it to about 3e-7 over 42 time steps), as any perturbation would.

Where each time step's meshes come from:

| Solver | Meshes |
|---|---|
| The plain solver | `mesh_wing_arrays` from the movement's `WingStepParameters` (validated first with `validate_airplane_step_parameters`) |
| The coupled solvers (aeroelastic, flap driven, free flight) | `wing_mesh_from_wing` from the Wings the problem built during the run, until step 6 of the arrays rewrite gives their step parameters the run's deformation, flap angles, and trial states |

The solver builds the first time step's geometry at the start of the run, and each later time step's one time step early (its wake shedding reads the next time step's ring vortex corners). It keeps only the previous, current, and next time steps' geometry.

## Loads

Both force methods already work on the solver's per Panel arrays, except for the parts that walked the Panels, which `_loads_arrays.py` replaces with the same operations in the same order, so they are bit for bit the object path's:

| Function | Replaces |
|---|---|
| `effective_line_vortex_strengths` | the Kutta-Joukowski method's Panel loop (`_effective_line_vortex_strengths_from_panels` on the object path) |
| `chordwise_vorticity_densities`, `spanwise_vorticity_densities`, `chordwise_vorticity_differences` | the Katz method's three Panel loops |
| `step_loads` | `process_solver_loads`: each Panel's loads in wind axes, each Airplane's totals and coefficients, plus each Wing's totals (for the flap drives in step 6) |
| `strip_leading_edge_points`, `strip_leading_edge_moments` | the aeroelastic solver's `_populate_slep_positions` and its strip leading edge moments |

The solver keeps the current evaluation's `StepLoads` (run scoped, like the plans). Until step 7 of the arrays rewrite, `apply_step_loads_to_objects` also writes them onto the Panels and Airplanes, which output and the coupled problems still read.

## Error checking

The array path raises the same errors as the object path. The object path checks most per time step values implicitly, in the constructors of the objects it builds, so each array function that replaces object building has a validation function that repeats those checks, and a test that it raises exactly where building the objects raises. The step parameter checks are `validate_wing_step_parameters` and `validate_airplane_step_parameters` in `_step_parameters.py`. `FreeFlightAeroelasticMovement`, which no longer builds its Airplanes when it is created, runs `validate_airplane_step_parameters` on every prescribed time step instead, so invalid prescribed motion still raises when the movement is created.

## The copy dimension and the array module

For GPU batches (Part B of `experimental/plan_arrays_and_gpu.md`), the array functions will accept an optional leading copy axis, (B, N, 3) rather than (N, 3), and broadcast over it. Plans have no copy axis, since every copy shares the problem's structure. The functions Part B runs on the GPU will take an array module (`xp`, NumPy or CuPy). Index arithmetic from the plans stays NumPy and is moved to the device once per run.

## Conventions every array function follows

- Its inputs and outputs are named by [Axes, Points, and Frames](AXES_POINTS_AND_FRAMES.md), with the `stack` prefix for (N, 3) global arrays, `grid` for (M, N, 3) per Wing arrays, and `list` for per time step lists.
- Every array it creates has an explicit `dtype`.
- It is a pure function: its inputs are its arguments, and its outputs are its return value.
- It is unit tested against the object code it replaces, on every relevant fixture and on random geometry.
