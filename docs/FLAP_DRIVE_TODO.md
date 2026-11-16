# Flap Drive TODO

This document lists the known issues, compromises, and major reworks for the flap drive feature and the free flight aeroelastic code it builds on. Each item says what the problem is, why it matters, what the fix is, and where the work would be. [Flap Drives](FLAP_DRIVE.md) describes how the feature works today, and some items here point to its sections for the full design.

Items are grouped by area and roughly ordered by priority within each group. Priority is a judgment of how much the item affects whether results can be trusted:

- **High**: results can be wrong, or a modeling choice limits what the tool can answer.
- **Medium**: results are approximate in a known way, or the code is fragile.
- **Low**: cleanup, documentation, or convenience.

## Summary

| # | Item | Area | Priority |
|---|---|---|---|
| 1 | The flap and the torsional springs are coupled to the fluid loosely | Coupling | High |
| 2 | The free flight rigid body doesn't feel the motors' reaction torques, the wings' inertial loads, or the moving CG | Free flight | High |
| 3 | The strip mass and inertia estimate is three times too large | Inertia | High |
| 4 | A measured wing inertia and the torsional springs' strip inertia can disagree | Inertia | High |
| 5 | The free flight aeroelastic code is unverified work in progress | Free flight | High |
| 6 | There are no integration tests or regression results for flap driven runs | Testing | High |
| 7 | The example airframe is untrimmed and uses placeholder mass properties | Free flight | Medium |
| 8 | The flap trajectory is a precomputed table, not a controller | Control | Medium |
| 9 | Cycle averaged results need a fixed period | Averaging | Medium |
| 10 | The period hand-off reaches into private caches of the core movement classes | Code health | Medium |
| 11 | The feedforward's aerodynamic torque is lagged, and no blade element theory model exists yet | Control | Medium |
| 12 | MotorSaturationController can't be rewound | Control | Medium |
| 13 | FreeFlightAeroelasticMovement builds every Airplane up front at the base flap angle | Free flight | Medium |
| 14 | The startup transient is handled by zeroing the fluid torque | Coupling | Medium |
| 15 | The torsional spring model differs from the blade element theory code's deformation model | Structure | Medium |
| 16 | Flap angles beyond 90 degrees are clipped for the geometry only | Geometry | Low |
| 17 | A saved problem can't be rerun | Serialization | Low |
| 18 | The controller doesn't see the OperatingPoint | Control | Low |
| 19 | The "quasi_square" trajectory style isn't ported | Trajectories | Low |
| 20 | Documentation and test coverage gaps | Docs | Low |
| 21 | The torsional spring ODE is stiff for light or narrow strips, and is integrated explicitly | Structure | Medium |
| 22 | A wing can't be driven by more than one FlapDrive | Control | Low |
| 23 | The flap examples duplicate their wing and motor building code | Code health | Low |
| 24 | The free wake is sensitive to the root spacing and passes through the V-tail | Geometry | Medium |

## Coupling

### 1. The flap and the torsional springs are coupled to the fluid loosely

**Priority:** High

**Problem.** The UVLM solves each time step once. The fluid torque from step `s` is held constant over `[t_s, t_(s+1)]` while the drive moves the wing, so the fluid's reaction to that motion acts one step late. The torsional springs have the same lag. This is true in both `FlapDrivenAeroelasticUnsteadyProblem` and `FreeFlightFlapDrivenAeroelasticUnsteadyProblem` (in free flight, the body is strongly coupled, but the flap and the springs are not).

**Why it matters.**

- A phase error in the aerodynamic load of about `360 * delta_time / period` degrees, about 7 degrees at the examples' 50 steps per wingbeat.
- A possible instability. The added mass part of the fluid torque resists the wing's acceleration, and applied a step late it acts as a delayed restoring torque. When the added mass is large relative to `I_w + I_a / J^2`, the flap overshoots and can diverge. A heavy geared motor hides this. A light wing or a direct drive exposes it.

**Fix.** Sub-iterate each time step on the fluid torque at its end until the flap motion and the loads agree, with Aitken relaxation, as the free flight body already does. Integrate the springs inside the same loop so their lag goes too. In free flight, append one entry per drive to the body's iterated state vector, so one loop drives the body, the flap, and the loads to agreement. The full design, what it needs, and its cost are in [Flap Drives](FLAP_DRIVE.md), "Planned: Strongly Coupling the Flap".

**Needs first.** Item 12 (rewinding the saturation controller). For the non free flight problem, a solver interface for evaluating a step without committing it, which changes shared solver code. The free flight solver already has one (`freeze_substep`, `evaluate_trial_aero_loads`, `restore_substep`).

**Cheaper stopgaps.** Extrapolating the torque over the step (cuts the phase error, not the instability), and treating the added mass implicitly with an `I_added` on `FlapDrive` (fixes the instability, not the circulatory lag). Both are described in the same section.

**Where.** `_FlapDriveCoupling._advance_flap_drives` and `initialize_next_problem` in `flap_driven_aeroelastic_unsteady_problem.py`, `FreeFlightAeroelasticUnsteadyProblem._advance_strongly_coupled` in `problems.py`, the free flight solver's substep interface, `_fixed_point_relaxation.py`.

### 14. The startup transient is handled by zeroing the fluid torque

**Priority:** Medium

**Problem.** For the first `step_discards` steps (default 5), the drives see zero fluid torque, as the springs ignore the loads. The flap then follows the actuator alone, and when the torque switches on it arrives as a step change.

**Why it matters.** The step change kicks the controller, and the first wingbeat's actuator torques and saturation are not representative. The right number of steps to discard depends on the time step and the geometry, and nothing checks it.

**Fix.** Ramp the fluid torque in over the discarded steps instead of switching it on, or start the drives only after a prescribed wake development phase, which free flight already has (`prescribed_num_steps`). Document how to choose `step_discards`.

**Where.** `_advance_flap_drives`.

## Free Flight

### 2. The free flight rigid body doesn't feel the motors' reaction torques, the wings' inertial loads, or the moving CG

**Priority:** High

**Problem.** MuJoCo models the vehicle as one rigid body with the constant `mass` and `I_BP1_CgP1` given to the problem. The flap drives' dynamics never reach it. It feels the aerodynamic loads, gravity, and `external_loads_fn`, but not:

- The motors' reaction torques on the airframe, and a spinning rotor's gyroscopic torque.
- The wings' inertial loads from swinging their mass back and forth.
- The movement of the CG and the change in inertia as the wings flap.

**Why it matters.** A real vehicle shakes at the flap frequency, and an unbalanced pair of motors (one saturating, or different gains) rolls or yaws it. Neither shows up. For a heavy wing relative to the body, the body's attitude oscillation can be wrong.

**Fix.** Two levels, described in [Flap Drives](FLAP_DRIVE.md), "Planned: Actuator and Wing Inertia in Free Flight":

- Apply the reaction loads to the body. Turn the drives' `phi`, `phi'`, `phi''`, `q'`, `q''`, and `F_a` into forces and moments on the airframe and add them to the interval loads. New inputs: each wing's mass and center of mass offset, each motor's shaft axis. Moderate effort, free flight side only.
- Model the wings and rotors as MuJoCo bodies on hinge joints. MuJoCo then handles every coupling, and replaces `solve_ivp` as the drive integrator. The crank linkage needs either a closed loop constraint or a control callback that applies the transmission. Large effort, but it is where a controller that varies the flap to reach body states belongs (item 8), and it gives item 1's strong coupling almost for free.

**Where.** `_assemble_interval_loads` in `problems.py` for the first level. `_mujoco_model.py`, the sub-iteration's state vector and weights, serialization, and rendering for the second.

### 5. The free flight aeroelastic code is unverified work in progress

**Priority:** High

**Problem.** `FreeFlightAeroelasticUnsteadyProblem`, `FreeFlightAeroelasticMovement`, and `FreeFlightAeroelasticUnsteadyRingVortexLatticeMethodSolver` were committed as "BELIEVED WORKING, NEED TO DOUBLE CHECK". The flap driven free flight problem inherits from them unchanged. Known signs that they need review:

- mypy reports errors in them: `free_flight_movement.py` uses `np` without importing it (line 397), assigns an `AirplaneMovement` to a variable typed `AeroelasticAirplaneMovement`, and passes `deformationAngles_Wcsp_to_Wcs_ixyz` to a method whose type doesn't accept it. `problems.py` passes a `FreeFlightAeroelasticMovement` where `AeroelasticMovement` is expected, and `AeroelasticUnsteadyProblem.__init__` accepts it through an `isinstance` check against both types.
- `FreeFlightAeroelasticUnsteadyProblem.__slots__` declares `_wing_density`, `_spring_constant_rad`, and `_damping_constant_rad` again, which its parent already declares.
- Its `__init__` has commented out arguments, and the file isn't formatted. Running the formatters on `problems.py` rewrites about 70 lines of it, so they are being kept off that file for now.
- Its docstrings still describe the non aeroelastic free flight classes in places.
- There are no unit tests for any of the three classes. The only tests that exercise them are the flap driven free flight tests.

**Why it matters.** Every free flight result, with or without flap drives, depends on this code.

**Fix.** Review it against `FreeFlightUnsteadyProblem` and `AeroelasticUnsteadyProblem` (with Jonah), fix the type errors and the duplicate slots, give the free flight aeroelastic movement a proper base class or protocol shared with `AeroelasticMovement` instead of the `isinstance` pair, format the file, update the docstrings, and add unit tests and an integration test.

**Where.** `problems.py`, `movements/free_flight_movement.py`, `free_flight_aeroelastic_unsteady_ring_vortex_lattice_method.py`, `output.py`.

### 7. The example airframe is untrimmed and uses placeholder mass properties

**Priority:** Medium

**Problem.** `free_flight_flap_driven_aeroelastic_unsteady_crank.py` uses a 30 kg mass and an inertia matrix scaled from another example, a V-tail sized by eye, and an initial state that isn't trimmed. In its run with the earlier wing, the vehicle sped up from 10 to 15.3 m/s over one second of free flight. It hasn't been run in full with the model 5 wing, 3 Hz, and 20 m/s. Its comment on `PRESCRIBED_NUM_STEPS` still says the prescribed start is one wingbeat, but at 3 Hz its 50 steps are 1.5 wingbeats.

**Why it matters.** The example demonstrates the coupling, not a flyable vehicle. Its trajectory says nothing about the robot.

**Fix.** Use the robot's measured mass, CG, and inertia (including the wings and motors, per item 2), and move them into a JSON file in `examples/flap_drive_parameters/` like the wing and powertrain. Fix the wingbeat comment, or set `PRESCRIBED_NUM_STEPS` from the flap period. Find a trimmed initial state for a flapping vehicle, which needs a cycle averaged trim (the existing `trim.py` finds trim for prescribed motion and would need to work with flap drives). Size the tail for static stability.

**Where.** The example, `trim.py`.

### 13. FreeFlightAeroelasticMovement builds every Airplane up front at the base flap angle

**Priority:** Medium

**Problem.** `FreeFlightAeroelasticMovement.__init__` generates the `Airplane` for every time step when it is constructed, before a flap driven problem records any flap angles. So `movement.airplanes` holds wings at the base flap angle with no deformation, which is not the simulated geometry. The solver uses the per step `SteadyProblem`s instead, and `_build_relaxation_weights` reads `airplanes[0][0]` only for reference lengths.

**Why it matters.** It costs construction time and memory, and anything that reads `movement.airplanes` for a flap driven or aeroelastic run gets the wrong geometry with no warning.

**Fix.** Generate only what is needed up front (the first time step, for the symmetry checks and reference lengths), or generate them lazily, as `AeroelasticMovement` does. Check that `output.py` never reads `movement.airplanes` for these runs.

**Where.** `movements/free_flight_movement.py`, `output.py`.

## Inertia and Structure

### 3. The strip mass and inertia estimate is three times too large

**Priority:** High

**Problem.** `AeroelasticUnsteadyProblem._calculate_mass_matrix` returns each panel's mass repeated over three components, and `_calculate_strip_inertias` sums all three, so every strip's mass and both of its inertias are three times too large. The flapping axis inertia formula, `(1 / 12) * M * (L^2 + W^2) + M * d^2`, also looks wrong for an axis along the chord, where the `L^2` (span) term shouldn't appear in the same way.

**Why it matters.** The torsional springs' inertial forcing, and so the deformation, uses these inertias in every aeroelastic run. A flap drive uses the same estimate for its wing inertia when no `wing_flap_inertia` is given, so the motor's reflected load is wrong too.

**Fix.** Fix the mass matrix to one component or sum one, re-derive the strip inertias about the flap axis and the torsional axis, and add unit tests against a hand computed plate. This changes existing aeroelastic results, so it needs its own change and regenerated validation outputs.

**Where.** `problems.py`, `_calculate_mass_matrix` and `_calculate_strip_inertias`.

### 4. A measured wing inertia and the torsional springs' strip inertia can disagree

**Priority:** High

**Problem.** A `FlapDrive`'s `wing_flap_inertia`, such as the examples' 1.411 kg\*m^2 per wing (the model 5 leading edge tube's, from `model_5_wing.json`, and twice that for the shared motor example's one drive), is used only in the drive's equation of motion. The torsional springs always use the strip inertias from `wing_density`. The two can describe wings of different mass, and the UVLM wing has no leading edge spar mass at all.

**Why it matters.** The drive's motion can be right while the deformation is computed for a different mass distribution, or the reverse.

**Fix.** Make one mass model the source of truth: either derive `wing_density` (and a separate spar mass per strip) from the measured properties, or give the structural model per strip masses that sum to the measured inertia. Warn when the two disagree by more than a tolerance.

**Where.** `flap_drive.py` (`inverse_dynamics_controller`'s inertia choice), `problems.py` (strip inertias).

### 21. The torsional spring ODE is stiff for light or narrow strips, and is integrated explicitly

**Priority:** Medium

**Problem.** `_spring_numerical_ode` integrates each strip's `I * theta'' + c * theta' + k * theta = M` with `solve_ivp`'s default explicit method (RK45). Its fastest mode decays at about `c / I`, so a strip with a small torsional inertia `I` (a narrow strip, a short chord near the tip or the root, or a low `wing_density`) and the examples' `damping_constant_rad` of 1000 N\*m\*s/rad is very stiff, and RK45 needs a tiny step to stay stable. With the blade element theory code's model 5 wing (about 23 strips per side, a 5 millimeter root chord, a 32 millimeter tip chord, and a `wing_density` of 0.888 kg/m^2), the flap driven examples spend almost all their time here: a profile of 8 time steps showed 4.3 million RK45 steps over 350 strip integrations, about 170 seconds for 9 time steps instead of about 1 minute for 150 with the earlier wing.

**Why it matters.** It makes realistic wing geometries impractical to run, and the cost doesn't come from the aerodynamics.

**Fix.** Integrate the spring ODE with an implicit method (`method="Radau"` or `"LSODA"` in `solve_ivp`), or solve it exactly over the time step, since it is linear with constant coefficients and only the inertial forcing varies. Choose physical damping values for the examples instead of the placeholder 1000.0, which interacts with item 3's inertia estimate.

**Where.** `problems.py`, `_spring_numerical_ode`.

### 15. The torsional spring model differs from the blade element theory code's deformation model

**Priority:** Medium

**Problem.** The examples' deformation comes from Ptera Software's per strip torsional spring model, not the blade element theory code's torsional deformation, so deformation results won't match it.

**Why it matters.** Comparisons between the two codes have to separate the aerodynamic difference from the structural one.

**Fix.** Decide which structural model is the reference, and either port the other or document the expected differences with a side by side case.

## Control and Trajectories

### 8. The flap trajectory is a precomputed table, not a controller

**Priority:** Medium

**Problem.** A `FlapReference` is a table of the actuator's desired motion built before the run. Nothing can change the flap trajectory during a simulation in response to the vehicle's state.

**Why it matters.** The goal of a MuJoCo simulation in which a flight controller varies the flap (frequency, amplitude, bias) to reach body states needs the reference to be produced while the simulation runs.

**Fix.** Replace the table with a reference source protocol that `FlapReference` implements, and add a dynamic source, such as a phase oscillator whose frequency, amplitude, and bias a flight controller sets each step from the body state. The source needs snapshot and restore if it runs inside a strongly coupled loop. This interacts with item 9 (a changing frequency has no fixed period) and is most natural with item 2's MuJoCo bodies.

**Where.** `flap_drive.py`, `_FlapDriveCoupling`.

### 11. The feedforward's aerodynamic torque is lagged, and no blade element theory model exists yet

**Priority:** Medium

**Problem.** By default, the controller's feedforward cancels the previous step's UVLM torque at the wing's actual motion. The blade element theory code's controller instead predicts the torque at the desired motion. The hook for that exists (`feedforward_aerodynamic` accepts a model that receives `AerodynamicModelInputs`, including the `OperatingPoint`), but no model is implemented.

**Why it matters.** When the motor lags, as while it saturates, the lagged feedforward asks for less torque than the blade element theory one, so the closed loop behaves differently from the blade element theory code's.

**Fix.** Port `aerodynamicBladeElementTheoryCalc` as an `AerodynamicModelInputs` model, check it against the MATLAB code on identical inputs, and compare its torque with the UVLM's `flap_fluid_torques`.

**Where.** A new module, used through `feedforward_aerodynamic`.

### 12. MotorSaturationController can't be rewound

**Priority:** Medium

**Problem.** The saturation controller's leaky buckets treat a time earlier than one already seen as the latest time seen. That is fine for an adaptive integrator's retried steps (an error of at most one integrator step), but not for re-integrating a whole time step from its start.

**Why it matters.** It blocks item 1's sub-iteration and item 8's snapshot and restore: every trial iteration would advance the buckets.

**Fix.** Add `save_state` and `restore_state` to `MotorSaturationController`, and a way for the problem to call them on the actuator functions' state (for example, an optional protocol the actuator can implement).

**Where.** `motor_saturation.py`, `_FlapDriveCoupling`.

### 18. The controller doesn't see the OperatingPoint

**Priority:** Low

**Problem.** The aerodynamic model receives the `OperatingPoint`, but the actuator function's `FlapDriveInputs` doesn't, so the controller can't schedule its gains on airspeed or body rates.

**Fix.** Add an `operating_point` field to `FlapDriveInputs`.

**Where.** `flap_drive.py`.

### 19. The "quasi_square" trajectory style isn't ported

**Priority:** Low

**Problem.** `generate_flap_trajectory` supports `"sine"` and `"square"`, but not the blade element theory code's `"quasi_square"`.

**Fix.** Port it, with its analytical derivatives.

**Where.** `flap_drive.py`.

### 22. A wing can't be driven by more than one FlapDrive

**Priority:** Low

**Problem.** A `FlapDrive` can flap several wings (one shared motor), and each wing can have its own `FlapDrive` (one motor per wing), but a wing can belong to only one `FlapDrive`. The problem rejects a wing that two drives claim, because each drive has its own actuator state and transmission, and a wing can only have one flap angle.

**Why it matters.** A design with two motors on one wing, for redundancy or extra torque, can't be modeled as two drives.

**Fix.** For now, model two motors on one wing as one `FlapDrive` whose actuator function returns the sum of both motors' torques, with their combined rotor inertia, as long as they share the same transmission. If the motors need separate transmissions or states, a drive would need several actuators coupled through one flap angle, which is a different equation of motion.

**Where.** `flap_drive.py`, `_FlapDriveCoupling._prepare_flap_drives`.

## Averaging and Code Health

### 9. Cycle averaged results need a fixed period

**Priority:** Medium

**Problem.** The solver averages loads over the last `lcm_period` of the movement, and a flap driven run gets its period from the `FlapReference`. A reference with `period=None`, such as one whose frequency changes, gives inaccurate averages, and if nothing else moves the run is treated as static.

**Why it matters.** Item 8's dynamic trajectories have no fixed period, so their averaged results would be meaningless.

**Fix.** Average over the recorded wingbeats instead: detect each wingbeat from the flap angle history (for example, its upward zero crossings) and average over the last whole one. This changes the shared solver's averaging.

**Where.** The solver's averaging and the output labels.

### 10. The period hand-off reaches into private caches of the core movement classes

**Priority:** Medium

**Problem.** A flap driven problem hands each reference's period to its wings' movements, then clears the cached periods of the movement and its `AirplaneMovement`s by assigning their private attributes (`_clear_movement_period_caches`), because the core movement classes have no public way to invalidate them. A test (`test_period_cache_names_exist`) fails if those names change.

**Why it matters.** It couples the flap code to private implementation details of shared classes.

**Fix.** Add a public cache invalidation method to `CoreMovement` and `CoreAirplaneMovement`, or let a movement take extra periods at construction.

**Where.** `_core.py`, `flap_driven_aeroelastic_unsteady_problem.py`.

### 23. The flap examples duplicate their wing and motor building code

**Priority:** Low

**Problem.** The four flap examples load their hardware parameters from the JSON files in `examples/flap_drive_parameters/`, but each still has its own copy of the code that turns the model 5 outline into `WingCrossSection`s and computes the wing density, the motor's line fits, and `make_motor`. That is about 150 lines repeated in each example.

**Why it matters.** The examples are long and hard to read, and a fix to the wing or motor code has to be made four times.

**Fix.** Move the wing building code and `make_motor` into one helper module next to the examples (or into the package, if a general wing from edge points is wanted), and have each example call it. Keep each example's flight condition and drive setup in the script.

**Where.** `examples/`.

### 16. Flap angles beyond 90 degrees are clipped for the geometry only

**Priority:** Low

**Problem.** A `Wing` only allows flap angles in [-90, 90] degrees. If a drive goes beyond, the geometry is clipped with a warning, but the actuator's state and the histories are not.

**Fix.** Decide whether exceeding the range should be an error, a hard stop in the dynamics, or allowed by lifting the `Wing`'s limit.

**Where.** `_advance_flap_drives`, `geometry/wing.py`.

### 24. The free wake is sensitive to the root spacing and passes through the V-tail

**Priority:** Medium

**Problem.** To flap separately, the two halves of the main wing are split into two Wings whose roots are off the plane of symmetry. Each root edge then carries a bound vortex leg, and the two legs spin in opposite directions. Bound vortex legs have no core, so a free wake vertex near both sees a nearly singular velocity. With the roots 2 mm apart (an earlier version of the model 5 examples), the newest root wake vertices moved up to 208 m/s faster than the rest of the wake at a 20 m/s airspeed, ended up meters out of place, and dragged the first wingbeat's wake corners out to 12 m from the centerline. With the CAD model's 410 mm root spacing (`root_offset_y_m` = 0.205 in `model_5_wing.json`), the worst vertex moves 23 m/s faster than the rest of the wake and the wake stays within the span. The remaining hot spot is where the main wing's inner wake passes through the V-tail, whose bound legs also have no core.

**Why it matters.** The wing and tail loads come from the same velocity field, so a wake vertex thrown near a surface corrupts the loads, not only the picture. Runs of the model 5 examples made with the 2 mm root spacing shouldn't be trusted near the root.

**Fix.** Keep the roots at the real body's spacing, never a token gap. For the tail, give bound vortex legs a small core when they act on wake points, or truncate or cut the wake where it meets a downstream surface. A prescribed wake (`prescribed_wake=True`) avoids both, at the cost of the wake's rollup.

**Where.** The examples and `model_5_wing.json`. A core on bound legs would be in `unsteady_ring_vortex_lattice_method.py`, where `_current_bound_r_c0s` is zero except for the trailing legs.

### 17. A saved problem can't be rerun

**Priority:** Low

**Problem.** The actuator function, the passive element functions, and an aerodynamic model are saved as inert placeholders, so a loaded flap driven problem can be inspected and plotted but not rerun unless the functions are passed back to `load`.

**Fix.** Document the pattern for rerunning a loaded problem, or support saving named, registered actuator functions.

**Where.** `_serialization.py`.

## Testing and Documentation

### 6. There are no integration tests or regression results for flap driven runs

**Priority:** High

**Problem.** The flap drive feature has unit tests, but no integration tests in `tests/integration/`, and no saved expected results. When the trajectory handling was reworked, no before and after comparison was possible, because the old code was untracked. Every later refactor (the y and z only movement arguments, the separate controller and physics torques, the mixin) is covered only by unit tests and short runs.

**Why it matters.** A change that shifts results slightly would go unnoticed.

**Fix.** Add integration tests for `FlapDrivenAeroelasticUnsteadyProblem` and `FreeFlightFlapDrivenAeroelasticUnsteadyProblem` that solve a small case and compare the flap histories and the cycle averaged loads with stored values. Rerun the four flap examples and store their outputs in `docs/examples_expected_output/`. Add the cross check against the MATLAB `singleWingDynamics` on identical inputs that the original plan called for.

Also add a test that one shared drive and one drive per wing agree. A one-off script checked this: on a symmetric airplane, a shared drive with twice the actuator inertia, twice the wing inertia, and twice the actuator force matched two per-wing drives. The flap angles agreed to 1e-8 rad in a prescribed run and 1e-5 rad in free flight, the shared drive's fluid torque was the sum of the per-wing torques, and the left and right per-wing drives matched each other. That comparison isn't in the test suite yet.

**Where.** `tests/integration/`, `docs/examples_expected_output/`.

### 20. Documentation and test coverage gaps

**Priority:** Low

- `CLASSES_AND_IMMUTABILITY.md` doesn't describe the flap drive classes, and `test_slots.py` doesn't cover them.
- `CLAUDE.md`'s architecture list doesn't list `flap_drive.py`, `crank_linkage.py`, `motor_saturation.py`, `flap_driven_aeroelastic_unsteady_problem.py`, `free_flight_flap_driven_aeroelastic_unsteady_problem.py`, or `movements/flap_driven_aeroelastic_wing_movement.py`.
- The free flight crank example, the shared motor example, and the scotch yoke example have no pages on the documentation website and no expected output. The crank example's stored output predates the model 5 wing.
- `_advance_flap_drives`'s docstring still has a note to check that it works with one drive per wing and with one drive for both wings. That has been checked (see item 6), so the note can be removed.
- The development environment has pyvista 0.49.0 installed while `requirements_dev.txt` pins 0.48.4, so the `pinned-versions` hook fails.
