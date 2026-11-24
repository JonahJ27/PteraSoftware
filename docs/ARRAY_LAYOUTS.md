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

The plans are run scoped: the solver builds them at the start of a run with `use_arrays=True` and clears them when the run ends, so they are never saved in a `.psz` file.

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

## Mesh plans

Each Wing's `MeshPlan` holds its symmetry type (1 to 4), its chordwise and spanwise panel counts (the spanwise count doubled for a type 4 Wing, whose grid holds both halves), the normalized chordwise coordinates of the Panels' edges (from 0.0 at the leading edge to 1.0 at the trailing edge, uniform or cosine spaced exactly as `mesh_wing` spaces them), and, for each Panel column of the main half, its wing section and the normalized spanwise coordinates of its inner and outer edges within that wing section.

## The copy dimension and the array module

For GPU batches (Part B of `experimental/plan_arrays_and_gpu.md`), the array functions will accept an optional leading copy axis, (B, N, 3) rather than (N, 3), and broadcast over it. Plans have no copy axis, since every copy shares the problem's structure. The functions Part B runs on the GPU will take an array module (`xp`, NumPy or CuPy). Index arithmetic from the plans stays NumPy and is moved to the device once per run.

## Conventions every array function follows

- Its inputs and outputs are named by [Axes, Points, and Frames](AXES_POINTS_AND_FRAMES.md), with the `stack` prefix for (N, 3) global arrays, `grid` for (M, N, 3) per Wing arrays, and `list` for per time step lists.
- Every array it creates has an explicit `dtype`.
- It is a pure function: its inputs are its arguments, and its outputs are its return value.
- It is unit tested against the object code it replaces, on every relevant fixture and on random geometry.
