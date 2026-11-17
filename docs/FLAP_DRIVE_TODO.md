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
| 21 | The torsional spring constants are placeholders, and the analytical spring solver has a light strip limit | Structure | Medium |
| 22 | A wing can't be driven by more than one FlapDrive | Control | Low |
| 23 | The flap examples duplicate their wing and motor building code | Code health | Low |
| 24 | The free wake is sensitive to the root spacing and passes through the V-tail | Geometry | Medium |
| 25 | The flap drives ignore the wings' weight, and the feedforward has no gravity term | Control | Medium |
| 26 | The wings' leading edge and membrane mass properties are computed, not measured | Inertia | Medium |
| 27 | The flap examples' discretization hasn't been checked for convergence, nor their spring constants for sensitivity | Geometry | Medium |

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

### 7. The example airframe is untrimmed and uses placeholder mass properties

**Priority:** Medium

**Problem.** `free_flight_flap_driven_aeroelastic_unsteady_crank.py` uses a 30 kg mass and an inertia matrix scaled from another example, a V-tail sized by eye, and an initial state that isn't trimmed. In its run with the earlier wing, the vehicle sped up from 10 to 15.3 m/s over one second of free flight. It hasn't been run in full with the model 5 wing, 3.5 Hz, and 20 m/s.

**Why it matters.** The example demonstrates the coupling, not a flyable vehicle. Its trajectory says nothing about the robot.

**Fix.** Build `mass` and `I_BP1_CgP1` from parts instead of placeholders, in a `vehicle.json` in `examples/flap_drive_parameters/` like the wing and powertrain: the body (fuselage, motors, and gearboxes, with their CG and inertia from the CAD model) plus, for each wing, both of its mass parts from `pterosaur_model_5_wing.json` (the leading edge tube as a line mass along the leading edge points, and the membrane as an areal mass over the outline), placed at the 0.205 m root offset and the mean flap angle, with the parallel axis terms. The rigid body sees the whole wing's mass, spar and membrane, unlike the torsional springs (item 4). The inertia stays constant during a run (item 2). Find a trimmed initial state for a flapping vehicle, which needs a cycle averaged trim (the existing `trim.py` finds trim for prescribed motion and would need to work with flap drives). Size the tail for static stability.

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

### 4. The spar and membrane mass split is set by hand in the examples

**Priority:** Medium

**Problem.** The model 5 wing's mass has two parts in `pterosaur_model_5_wing.json`: the leading edge tube (0.882 kg, 1.411 kg\*m^2 about the shoulder) and the membrane (0.0546 kg, 0.0500 kg\*m^2, from the blade element theory code's measured areal density). The examples give each `FlapDrive` both parts' inertia (1.461 kg\*m^2 per wing, twice that for the shared motor) and give the torsional springs only the membrane (`wing_density` = 0.0550 kg/m^2), because the strips twist about the leading edge, where the tube sits. Nothing in the package knows about the split: the problem only takes one `wing_density`, and a `FlapDrive` without `wing_flap_inertia` estimates its inertia from that density, which leaves out the spar. The free flight example also multiplies the density by 10 (a user setting).

**Why it matters.** The split is correct only as long as each example passes both values by hand. A drive that relies on the estimate would miss most of the wing's inertia (about 97% of it for model 5). The torsional springs also triple the membrane's mass (item 3).

**Fix.** Let the problem take the spar separately (for example, a spar mass per unit span, or a callable that returns a strip's spar inertia about its leading edge), use spar plus membrane for the estimate of a drive's wing inertia, and use the membrane alone, plus the spar's small twist inertia about its own axis, for the springs. Fix item 3 first.

**Where.** `flap_driven_aeroelastic_unsteady_problem.py` (`_calculate_wings_flap_inertia`), `problems.py` (strip inertias), the examples.

### 26. The wings' leading edge and membrane mass properties are computed, not measured

**Priority:** Medium

**Problem.** Every wing's mass and inertia in `examples/flap_drive_parameters/` comes from the blade element theory code's `precomputeWingGeometry`, which models the wing from a few assumed dimensions rather than the built hardware:

| Wing | Leading edge | Membrane |
|---|---|---|
| `pterosaur_model_5_wing.json` | 0.882 kg, 1.411 kg\*m^2 | 0.0550 kg/m^2, 0.0546 kg, 0.0500 kg\*m^2 |
| `faav_small_wing.json` | 0.00484 kg, 0.000298 kg\*m^2 | 0.17 kg/m^2, 0.0140 kg, 0.000706 kg\*m^2 |
| `faav_large_wing.json` | 0.00596 kg, 0.000558 kg\*m^2 | 0.17 kg/m^2, 0.0212 kg, 0.00164 kg\*m^2 |

The inertias are about the flap axis at the shoulder. The assumptions behind them are uncertain:

- The pterosaur's leading edge is modeled as a uniform 36 mm tube with a 2.5 mm wall. The real spar tapers from a 52.4 mm outer diameter with a 1.95 mm wall at the root to 28.1 mm with a 3.05 mm wall at the tip, and the BET code's comments give its measured weight as 0.858 kg without the 7.5 cm shoulder offset. The tube's inertia depends on where along the span its mass sits, which the taper moves inboard.
- The FAAV's leading edge is modeled as a flat 5 mm by 1.5 mm carbon plate (BET_plotter's FAAV parameters). The FAAV's actual spar hasn't been checked.
- The FAAV membrane density is 0.17 kg/m^2 from BET_plotter, but runSim labels 0.0550 kg/m^2 as approximately the FAAV membrane's density, a factor of 3 apart.
- The membrane is a 0 thickness sheet over the outline, summed in 1 cm strips as point masses. Grommets, edge reinforcements, attachments, and the shoulder hardware aren't included. The BET code's strips also stop one strip short of the tip.
- None of the wings' centers of mass are stored, though item 25 (gravity) needs them.

**Why it matters.** The leading edge and membrane inertias set each FlapDrive's wing inertia, which the motor must accelerate (`wing_flap_inertia`), and the membrane's areal density sets the torsional springs' strip inertias (`wing_density`). An error here goes straight into the motor's required torque and its saturation, and into the wing's twist.

**Fix.** For each wing (the pterosaur model 5 wing and both FAAV wings):

1. Weigh the leading edge and the membrane separately, or the whole wing and one part.
2. Measure the whole wing's inertia about the flap axis, for example with a bifilar or compound pendulum about the shoulder axis, or take it from a CAD model with the materials' measured densities.
3. Find each part's spanwise center of mass (for example by balancing it).
4. Compare all three with the JSON files, and update the files (adding the centers of mass).
5. If the pterosaur's taper matters, replace the uniform tube with the tapered one in the inertia calculation.

Item 4 (passing the split into the package) and item 7 (building the vehicle's mass properties from parts) both rely on these numbers.

**Where.** `examples/flap_drive_parameters/` (the three wing files), and the examples that read them.

### 21. The torsional spring constants are placeholders, and the analytical spring solver has a light strip limit

**Priority:** Medium

**Background.** `_spring_numerical_ode` integrates each strip's `I * theta'' + c * theta' + k * theta = M` with `solve_ivp`'s default explicit method (RK45). Its fastest mode decays at about `c / I`, so a strip with a small torsional inertia `I` and the examples' `damping_constant_rad` of 1000 N\*m\*s/rad is very stiff, and RK45 needs a tiny step to stay stable. The model 5 wing took about 170 seconds for 9 time steps, and the FAAV small wing (strip inertias down to 3e-9 kg\*m^2) never finished.

**Done.** `AeroelasticUnsteadyProblem` has a `use_analytical_spring_solver` flag (False by default, so prescribed motion keeps `solve_ivp`, and True by default in both flap driven problems) that solves each strip's ODE exactly with `_spring_analytical_ode`. That solve treats the forcing as linear between `_SPRING_FORCING_SAMPLES` (100) evenly spaced samples per time step and advances the strip across each piece with one matrix exponential (`_spring_piece_map`). It costs about the same for any strip inertia. `_FLAP_ACCELERATION_SAMPLES` is defined as `_SPRING_FORCING_SAMPLES`, so the FlapDrives sample their accelerations at the same points and the solve is exact for them. The flap examples now run 120 time steps in about 15 to 25 s, and they set their time step from `STEPS_PER_FLAP` = 40.

**Remaining problems.**

- The placeholder spring and damping constants: k = c = 1000.0 in every example, which interacts with item 3's inertia estimate. With them, the FAAV wing's twist is only around 1e-8 rad, so the wing is effectively rigid.
- The light strip limit: the matrix exponential's rounding error grows roughly in proportion to `c / I`. With k = c = 1000.0 and a 0.005 s time step, the relative error in theta is about 6e-10 for I = 1e-7 kg\*m^2 and 8e-8 for I = 1e-9 kg\*m^2 (the FAAV's lightest strips are about 3e-9 kg\*m^2). It reaches the numerical solver's 1e-6 tolerance for strips near 1e-10 kg\*m^2.
- The smooth forcing error: for a smooth prescribed flap, the straight lines miss the forcing by about (omega \* h)^2 / 8 of its amplitude, with h the time step / 99. That is about 3e-7 at 40 time steps per flap and 1.3e-6 at 20, at any flap frequency, so keep 20 or more time steps per flap.

**Fix.**

- Choose physical spring and damping values for the examples.
- If a wing has strips lighter than about 1e-10 kg\*m^2, build `_spring_piece_map` from the closed form two-rate solution for strongly overdamped strips, which keeps the error at rounding level (a version of this was written and tested, then removed as unnecessary for current wings).
- Optionally, size `_SPRING_FORCING_SAMPLES` from the FlapDrives' period and the time step, so a coarse time step can't push the sampling error above the solver tolerance.

**Where.** `problems.py` (`_spring_analytical_ode`, `_spring_piece_map`, `_SPRING_FORCING_SAMPLES`), and the examples.

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

### 25. The flap drives ignore the wings' weight, and the feedforward has no gravity term

**Priority:** Medium

**Problem.** A wing's weight acts at its center of mass, which is outboard of the shoulder, so gravity puts a torque on the wing about the flap axis: about -m_w * g * r_cg * cos(phi) for a level body, where m_w is the wing's mass, r_cg is its center of mass's spanwise distance from the shoulder, and phi is the flap angle. Neither the flap drive's equation of motion nor its feedforward includes this torque. `single_wing_dynamics` sums only the fluid torque, the passive torques, and the inertial terms. The feedforward has inertial, damping, passive, and aerodynamic terms, but no gravity term. The blade element theory code's `singleWingDynamics` and its controllers leave gravity out too. In free flight, MuJoCo applies gravity to the rigid body, but not to the drives.

**Why it matters.** For the pterosaur wing, the weight torque is roughly 10 N\*m per wing (the 0.88 kg leading edge tube has its center of mass about 1.1 m out), which the motor must carry through the downstroke and which helps it on the upstroke. That is a sizable fraction of the motor's torque reflected through the gearbox and the linkage. Leaving it out makes the downstroke look easier than it is and the upstroke look harder. For the FAAV wing, the torque is small (a few hundredths of a N\*m), but not zero.

**Fix.**

1. Add the weight torque to the wing side torque in `single_wing_dynamics`, so the wings feel it. It enters like the passive wing torque, multiplied by J when it is reflected to the actuator.
2. Add a `feedforward_gravity` switch to `FlapDrive`, with a `feedforward_gravity_force` term in `FlapDriveInputs`, that cancels -J * T_gravity at the actual or desired flap angle, like the passive term.
3. The torque needs each wing's mass and center of mass, which a `FlapDrive` doesn't have. Add them as `FlapDrive` arguments (summed over the drive's wings, like `wing_flap_inertia`), or derive them from the leading edge and membrane parts once item 4 moves that split into the package. The parameter JSON files would need each part's center of mass (the blade element theory code's `precomputeWingGeometry` already computes `leading_edge_COM` and `membrane_COM`).
4. Take the gravity direction from the body's attitude: from the `OperatingPoint` in a prescribed run, and from MuJoCo's state in free flight (see item 18 for the controller's access to the `OperatingPoint`). For a mirrored wing, the sign follows the same flip as its fluid torque.

**Where.** `flap_drive.py` (`single_wing_dynamics`, `FlapDrive`, `FlapDriveInputs`, the inverse dynamics controller), `_FlapDriveCoupling`, and the parameter JSON files.

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

**Problem.** To flap separately, the two halves of the main wing are split into two Wings whose roots are off the plane of symmetry. Each root edge then carries a bound vortex leg, and the two legs spin in opposite directions. Bound vortex legs have no core, so a free wake vertex near both sees a nearly singular velocity. With the roots 2 mm apart (an earlier version of the model 5 examples), the newest root wake vertices moved up to 208 m/s faster than the rest of the wake at a 20 m/s airspeed, ended up meters out of place, and dragged the first wingbeat's wake corners out to 12 m from the centerline. With the CAD model's 410 mm root spacing (`root_offset_y_m` = 0.205 in `pterosaur_model_5_wing.json`), the worst vertex moves 23 m/s faster than the rest of the wake and the wake stays within the span. The remaining hot spot is where the main wing's inner wake passes through the V-tail, whose bound legs also have no core.

**Why it matters.** The wing and tail loads come from the same velocity field, so a wake vertex thrown near a surface corrupts the loads, not only the picture. Runs of the model 5 examples made with the 2 mm root spacing shouldn't be trusted near the root.

**Fix.** Keep the roots at the real body's spacing, never a token gap. For the tail, give bound vortex legs a small core when they act on wake points, or truncate or cut the wake where it meets a downstream surface. A prescribed wake (`prescribed_wake=True`) avoids both, at the cost of the wake's rollup.

**Where.** The examples and `pterosaur_model_5_wing.json`. A core on bound legs would be in `unsteady_ring_vortex_lattice_method.py`, where `_current_bound_r_c0s` is zero except for the trailing legs.

### 27. The flap examples' discretization hasn't been checked for convergence, nor their spring constants for sensitivity

**Priority:** Medium

**Problem.** The time step and the mesh of every flap example were chosen by hand, and no run has checked that refining them leaves the results unchanged. The two wings' meshes are equally fine relative to their spans (strips about 4.6 percent of the half span wide, with 6 chordwise panels), but the pterosaur's has weak spots that the FAAV's doesn't:

| | Pterosaur model 5 | FAAV small |
|---|---|---|
| Panels per wing | 25 strips x 6 chordwise = 150 | 23 strips x 6 chordwise = 138 |
| Strip width (spanwise_panel_width_m) | 100 mm target, 36 to 114 mm actual | 20 mm target, 2 to 30 mm actual |
| Panel chordwise length | 0.8 to 146 mm, median 71 mm | 5 to 36 mm, median 36 mm |
| Panel aspect ratio (span / chord) | median 1.3, 19 panels above 5, worst 313 | median 0.5, worst 4 |
| Wake row length (air speed * time step) at 40 time steps per flap | 125 mm at 4 Hz and 20 m/s, about 1.8 times the trailing edge panels | 30 mm at 5 Hz and 6 m/s, about 0.8 times the trailing edge panels |

- At the pterosaur's root, the model 5 chord grows from 5 mm (the 5 mm root_chord_m stand-in for the shoulder's zero chord) to 835 mm only 0.22 m outboard, so the first strip's panels are 0.8 mm long and 107 mm wide, and the trailing edge is steeply swept across the first two strips.
- At the pterosaur's tip, the chord shrinks to 66 mm, so the last four strips' panels have aspect ratios of 4 to 8.
- The pterosaur's wake rows are coarser than the panels that shed them, while the FAAV's are well matched.

**Why it matters.** Every result the examples show (the lift, the fluid torque that loads the motor, whether the motor saturates, the twist, and in free flight the vehicle's motion) depends on the discretization. Without a convergence check, a difference between two designs could be a meshing artifact.

**Fix.** For each wing (`pterosaur_model_5_wing.json`, `faav_small_wing.json`, and `faav_large_wing.json`), run the prescribed flap driven example with that wing and refine one parameter at a time:

- STEPS_PER_FLAP (for example 40, 70, and 100), which also sets the wake row length.
- num_chordwise_panels (for example 6, 9, and 12).
- spanwise_panel_width_m in the wing's meshing settings (for example the current width, half of it, and a quarter of it).
- For the pterosaur, also the root treatment: root_chord_m, and whether the first two strips need more panels or a different station layout near the shoulder.
- The number of flaps before averaging (for example 3 and 5), so the startup transient is out of the averages.
- The torsional spring_constant_rad and damping_constant_rad. Unlike the settings above, these are physical properties, not discretization, so this is a sensitivity sweep rather than a convergence check: every example uses the placeholder k = c = 1000.0 (item 21), which leaves the twist tiny (around 1e-8 rad on the FAAV wing), so the wings are effectively rigid. Sweep each over a few decades (for example 1, 10, 100, and 1000) on its own and together, to see where the twist starts to change the lift, the motor torque, and the saturation, and check the discretization again at the stiffness and damping finally chosen, because a flexible wing may need a finer time step than a rigid one.

Compare the cycle averaged lift and thrust, the peak and cycle averaged motor torque, whether and where the motor saturates, and the tip twist. For the discretization settings, keep the coarsest setting at which each changes by less than a few percent from the finest, record the chosen settings (for example in each wing's meshing settings), and update the examples. For the spring constant and damping, record how sensitive each result is, and replace the placeholders with values measured on the wings (for example from the tip twist in flapping video, or a static twist and ring-down test) or tuned to match them. Free flight then needs one confirming run at the chosen settings, because its sub-iteration and body motion add their own sensitivity. ps.convergence.analyze_unsteady_convergence can't do this yet, because it rejects aeroelastic problems, so the sweep is manual (or that function could be extended to flap driven problems).

**Where.** The flap examples, the `meshing` settings in `examples/flap_drive_parameters/`, and possibly `convergence.py`.

### 17. A saved problem can't be rerun

**Priority:** Low

**Problem.** The actuator function, the passive element functions, and an aerodynamic model are saved as inert placeholders, so a loaded flap driven problem can be inspected and plotted but not rerun unless the functions are passed back to `load`.

**Fix.** Document the pattern for rerunning a loaded problem, or support saving named, registered actuator functions.

**Where.** `_serialization.py`.

## Testing and Documentation

### 20. Documentation and test coverage gaps

**Priority:** Low

- `CLASSES_AND_IMMUTABILITY.md` doesn't describe the flap drive classes, and `test_slots.py` doesn't cover them.
- `CLAUDE.md`'s architecture list doesn't list `flap_drive.py`, `crank_linkage.py`, `motor_saturation.py`, `flap_driven_aeroelastic_unsteady_problem.py`, `free_flight_flap_driven_aeroelastic_unsteady_problem.py`, or `movements/flap_driven_aeroelastic_wing_movement.py`.
- The flap examples have no pages on the documentation website. Their outputs are stored in `docs/examples_expected_output/` (the three prescribed flap examples save their wing kinematics, motor dynamics, and motor operating point figures, and the free flight example saves its drawings, animation, and result plots).
- `_advance_flap_drives`'s docstring still has a note to check that it works with one drive per wing and with one drive for both wings. That has been checked (`test_shared_drive_matches_per_wing_drives` in `tests/integration/test_flap_driven_aeroelastic_unsteady_ring_vortex_lattice_method.py`), so the note can be removed.
- The development environment has pyvista 0.49.0 installed while `requirements_dev.txt` pins 0.48.4, so the `pinned-versions` hook fails.
- The free flight aeroelastic code (`FreeFlightAeroelasticUnsteadyProblem`, `FreeFlightAeroelasticMovement`, and `FreeFlightAeroelasticUnsteadyRingVortexLatticeMethodSolver`) is now covered by `tests/unit/test_free_flight_aeroelastic_unsteady_problem.py` and `tests/integration/test_free_flight_aeroelastic_unsteady_ring_vortex_lattice_method.py`, but still has code cleanup left, none of which changes results:
    - mypy errors: `movements/free_flight_movement.py` uses `np` without importing it, assigns an `AirplaneMovement` to a variable typed `AeroelasticAirplaneMovement`, and passes `deformationAngles_Wcsp_to_Wcs_ixyz` to a method whose type doesn't accept it. `problems.py` passes a `FreeFlightAeroelasticMovement` where `AeroelasticMovement` is expected, and `AeroelasticUnsteadyProblem.__init__` accepts it through an `isinstance` check against both types. A base class or protocol shared by the two movements would replace that pair.
    - `FreeFlightAeroelasticUnsteadyProblem.__slots__` declares `_wing_density`, `_spring_constant_rad`, and `_damping_constant_rad` again, which its parent already declares.
    - Its `__init__` has commented out arguments, and that part of `problems.py` isn't formatted (running the formatters on `problems.py` rewrites about 70 lines of it, so they are being kept off that file for now).
    - Its docstrings still describe the non aeroelastic free flight classes in places.
