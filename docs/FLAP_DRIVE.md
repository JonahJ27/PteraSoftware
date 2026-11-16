# Flap Drives

This document describes the flap drive feature: a model of the actuator, transmission, and controller that flap a `Wing`, coupled to the aeroelastic unsteady ring vortex lattice method. It covers what changed in the existing codebase, how the pieces work, how they plug into the solver, and exactly what happens at each time step.

## Purpose

In the plain `AeroelasticUnsteadyProblem`, a `Wing`'s flap angle is prescribed: it is a function of time set by the `AeroelasticWingMovement`'s oscillation parameters, and nothing the solver computes can change it. A flap drive replaces that with a closed loop. Each time step, the aerodynamic torque about the flap axis is fed to a model of the actuator (a motor with a torque-speed line, a muscle, or anything else), which is integrated through its transmission (a gearbox, a crank linkage, or a direct drive). The resulting flap angle sets the next time step's geometry. If the actuator can't supply the torque the desired trajectory needs, the wing falls behind it.

## Changes to Existing Files

The feature was built to modify existing classes as little as possible. Everything else lives in new files.

| File | Change |
|------|--------|
| `pterasoftware/movements/aeroelastic_wing_movement.py` | `AeroelasticWingMovement.generate_wing_at_time_step` gained one optional keyword, `overrideAngleX_Gs_to_Wn_ixyz=None`. When given, it replaces the x component of the prescribed `angles_Gs_to_Wn_ixyz` (the flap angle, in degrees) before the rotation point offset adjustment. When `None`, behavior is unchanged. |
| `pterasoftware/_serialization.py` | Imports and registers `FlapTransmission`, `CrankLinkage`, `FlapReference`, `FlapDrive`, `FlapDrivenAeroelasticWingMovement`, `FlapDrivenAeroelasticUnsteadyProblem`, and `FreeFlightFlapDrivenAeroelasticUnsteadyProblem` in `_CLASS_REGISTRY` and `_PUBLIC_SAVEABLE_CLASSES`. `_FORMAT_VERSION` went from 31 to 38. |
| `pterasoftware/__init__.py` | Eagerly imports `crank_linkage`, `flap_drive`, `motor_saturation`, `flap_driven_aeroelastic_unsteady_problem`, and `free_flight_flap_driven_aeroelastic_unsteady_problem`. |
| `pterasoftware/movements/__init__.py` | Imports `flap_driven_aeroelastic_wing_movement`. |
| `pterasoftware/problems.py` | `FreeFlightAeroelasticUnsteadyProblem.__init__` gained a `step_discards=5` keyword, which it passes to `AeroelasticUnsteadyProblem`. Before, it always used the default of 5. |
| `docs/website/examples.md` | Adds the new example to the solvers toctree. |

The solvers and every other existing class are unchanged.

## New Files

| File | Contents |
|------|----------|
| `pterasoftware/flap_drive.py` | The `Transmission` interface, `FlapTransmission`, `FlapReference`, `AerodynamicModelInputs`, `FlapDriveInputs`, `FlapDriveAccelerations`, and `FlapDrive`. Nothing in it knows about cranks. |
| `pterasoftware/crank_linkage.py` | `CrankLinkage`, a motor, gearbox, and crank linkage powertrain with analytical kinematics. It implements the `Transmission` interface, so it is passed to a `FlapDrive` directly. |
| `pterasoftware/motor_saturation.py` | `MotorSaturationController` and `SaturationInfo`, a port of the BET code's `motorSaturationController` |
| `pterasoftware/movements/flap_driven_aeroelastic_wing_movement.py` | `FlapDrivenAeroelasticWingMovement` |
| `pterasoftware/flap_driven_aeroelastic_unsteady_problem.py` | `FlapDrivenAeroelasticUnsteadyProblem`, and the `_FlapDriveCoupling` mixin that holds the flap drive logic it shares with the free flight problem |
| `pterasoftware/free_flight_flap_driven_aeroelastic_unsteady_problem.py` | `FreeFlightFlapDrivenAeroelasticUnsteadyProblem` |
| `examples/flap_driven_aeroelastic_unsteady_crank.py` | The BET code's model 5 wing (4.3 m span), each half driven by its own motor 9, 15:1 gearbox, and crank linkage, at 2 Hz, 10 m/s, and 10 degrees angle of attack |
| `examples/flap_driven_aeroelastic_unsteady_crank_shared_motor.py` | The crank example with one motor, gearbox, and crank linkage driving both halves of the wing through one `FlapDrive` (`wing_indices=(0, 1)`, twice one wing's inertia) |
| `examples/flap_driven_aeroelastic_unsteady.py` | The same wing driven through a scotch yoke given as user functions wrapped in a `FlapTransmission` |
| `examples/free_flight_flap_driven_aeroelastic_unsteady_crank.py` | The same wing, motors, and crank linkages on a vehicle with a V-tail that flies freely after a prescribed start |
| `examples/flap_drive_parameters/model_5_wing.json`, `mad_m13_kv90_powertrain.json`, and `crank_linkage.json` | The hardware parameters that the four flap examples load: the model 5 wing's outline, mass, and meshing settings, the motor, motor controller, gearbox, and feedback gains, and the crank linkage's geometry |
| `docs/website/examples/flap_driven_aeroelastic_unsteady_crank.md` and `docs/examples_expected_output/flap_driven_aeroelastic_unsteady_crank/flap_drive.png` | The example's documentation page and its plot |
| `tests/unit/test_flap_drive.py`, `tests/unit/test_crank_linkage.py`, `tests/unit/test_flap_trajectories.py`, `tests/unit/test_motor_saturation.py`, `tests/unit/test_flap_driven_aeroelastic_unsteady_problem.py`, `tests/unit/test_free_flight_flap_driven_aeroelastic_unsteady_problem.py`, `tests/unit/fixtures/flap_drive_fixtures.py` | Unit tests and fixtures |

Run the example from the repository root with `python -m examples.flap_driven_aeroelastic_unsteady_crank`.

## The Pieces

### The `Transmission` interface

`FlapDrive` is independent of the powertrain. The actuator has one generalized coordinate, the actuator position `q` (a motor shaft angle, a muscle length, or the flap angle itself), and `FlapDrive` only asks its transmission for these members:

| Member | Returns | Used for |
|--------|---------|----------|
| `evaluate_angle(q)` | The flap angle `phi`, the x component of the `Wing`'s `angles_Gs_to_Wn_ixyz` (rad) | The Wings' geometry at each time step |
| `evaluate_jacobian(q)` | `J = d(phi)/dq` | The equation of motion, at every evaluation |
| `evaluate_hessian(q)` | `H = d^2(phi)/dq^2` | The equation of motion, at every evaluation |
| `convert_trajectory(times, phi, phi', phi'')` | The actuator's position, velocity, and acceleration (the inverse kinematics) | Building a `FlapReference`, once, before the simulation. Required, so a wing trajectory can always be converted into an actuator trajectory |

`flap_drive.Transmission` is a `typing.Protocol`, so any class with those members works without subclassing anything. `FlapDrive` checks only that the members exist.

| Transmission | Use |
|--------------|-----|
| `FlapTransmission.identity()` | Direct drive or a muscle acting on the flap angle. `J = 1`, `H = 0`. |
| `FlapTransmission.geared(ratio)` | A constant reduction. `J = 1 / ratio`, `H = 0`. |
| `FlapTransmission(angle, jacobian, hessian, trajectory)` | Any transmission you can write the functions for, without writing a class. |
| `FlapTransmission.from_forward_kinematics(angle, trajectory)` | A convenience that finds `J` and `H` by central differences. |
| `FlapTransmission.from_positions(angle, inverse_position)` | Only the position maps in both directions. `J` and `H` come from central differences in `q`, and the actuator's velocity and acceleration from `numpy.gradient` in time, like `convertWingTrajToMotorNumerical`. |
| `numerical_trajectory_function(inverse_position)` | Not a transmission, but a helper that turns a positions-only inverse kinematics into a full trajectory function, for use with your own analytical `J` and `H`. |
| `crank_linkage.CrankLinkage(..., gear_ratio, derivative_method)` | A motor, a gearbox, and the crank linkage from the BET code. |
| Your own class | Any other powertrain. |

### `CrankLinkage`

A motor turns a gearbox, the gearbox turns the gearbox crank, and a rigid link joins it to a wing crank on the Wing. The constructor takes the linkage geometry, the `gear_ratio`, and the `derivative_method` for trajectory conversion (`"analytical"` or `"numerical"`), and the object can report all of them back.

It works in two coordinates. `wing_angle`, `wing_angle_jacobian`, `wing_angle_hessian`, `crank_angles`, and `crank_angle_derivatives` use the crank angle (`theta_r`), and they match `calcCrankshaftFK`, `calcCrankshaftJacFK`, `calcCrankshaftHessFK`, `calcCrankshaftIK`, `calcCrankshaftJacIK`, and `calcCrankshaftHessIK` to round off. The `Transmission` members use the motor angle, dividing by the gear ratio once per derivative, like the BET code's `J_motor` and `H_motor`. `angle_range()` samples one turn of the gearbox crank for the stroke's ends, as `runSingleSimulation` does.

`convert_trajectory` picks the inverse kinematics branch from the sign of the wing's velocity and unwraps the crank angle, as `convertWingTrajToMotorAnalytical` does. That assumes a trajectory consistent with a crank turning through full revolutions, meaning one that reaches the full stroke at every turnaround. A trajectory that turns around short of the full stroke would make the gearbox crank jump, so it raises a `ValueError`. With the analytical method, the motor velocity and acceleration within 10 samples of each turnaround are filled by interpolation, because there the inverse kinematics derivatives diverge and the acceleration's two terms nearly cancel. The numerical method differentiates the motor position with `numpy.gradient`, like `convertWingTrajToMotorNumerical`.

### `FlapReference`

The desired actuator trajectory, as a table of position, velocity, and acceleration at increasing times, sampled every 1e-4 s like the BET code's. It is built once, before the simulation runs, so the controller never has to invert a singular linkage while the simulation runs. There are two ways to get one, mirroring the BET code's trajectory options:

- `FlapReference.constant_velocity(start, velocity, duration, period=...)`: a motor turning at a constant speed, the BET code's `use_faav_traj` path. It needs no inverse kinematics.
- `FlapReference.from_flap_trajectory(transmission, *generate_flap_trajectory(...), period=...)`: one of the BET code's `generateFlapTrajCustom` styles, `"sine"` or `"square"` (its `"quasi_square"` style isn't ported), with the stroke amplitude and dihedral from `CrankLinkage.angle_range` (sampled as `runSingleSimulation` does), clipped to the stroke, and repeated for the requested cycles. The conversion goes through the transmission's `convert_trajectory`.

#### The only definition of the flap trajectory

A `FlapReference` is the only place a `FlapDrive`'s trajectory is defined. A `FlapDrive` can't be built without one, and a `FlapDrivenAeroelasticWingMovement` can't prescribe any flap motion, so there is no second source of the trajectory to disagree with it. Earlier versions let the movement's flap oscillation stand in for a missing reference, and let it set the cycle averaging window even when the drive followed its own reference. Both paths are gone.

#### The period and cycle averaged results

Every `FlapReference` states its `period`, a required keyword argument: the flap trajectory's period in seconds (one wingbeat), or `None` if it doesn't repeat with a fixed period. It must be the flap's period, not the actuator's. For a motor driving a crank linkage through a gearbox of ratio `N`, one wingbeat is one turn of the crank, which is `N` turns of the motor. For a trajectory from `generate_flap_trajectory`, it is `1 / flapping_frequency`.

The period exists for the solver's cycle averaged results. The solver averages loads over the last full period of all the motions (their least common multiple), and it takes the periods from the movement. A `FlapDrivenAeroelasticWingMovement` prescribes no flap motion, so the `FlapDrivenAeroelasticUnsteadyProblem` hands each drive's reference period to its `Wing`s' movements before the parent class reads the movement's periods. The movement's `lcm_period`, `static`, and the problem's `first_averaging_step` then work as they would for a prescribed flap of that period.

A reference with `period=None`, such as one whose flapping frequency changes, gives inaccurate cycle averaged results, because no fixed window covers whole wingbeats. If nothing else in the movement moves, the simulation is also treated as static: the movement's `static` is `True`, the solver reports the final time step's loads as its averages, and the output labels the geometry as static. The problem logs a warning that says so. This is a known limitation of a dynamic trajectory, and averaging over the recorded wingbeats would fix it, but that needs a change to the shared solver, which this feature avoids.

The hand-off clears the period caches of the movement and its `AirplaneMovement`s, which have no public way to invalidate them. That reaches into private attributes of the core movement classes without changing them, and `test_period_cache_names_exist` fails if those names change.

A crank turning one way reverses the flap only at its dead centers. A trajectory that turns around short of the full stroke (any style with an amplitude smaller than the stroke) raises an error that points to the turnaround, rather than producing a motor trajectory that jumps between inverse kinematics solutions. Both styles cover the full requested amplitude; the BET code's `"square"` covers only half of it. A full stroke `"square"` does convert, but its instant reversal at each dead center needs an unbounded motor speed there, so its motor trajectory has very large spikes at the turnarounds.

It can be read two ways: interpolated linearly in time (`evaluate`), like the BET code's `griddedInterpolant`, or at the table entry whose position is nearest the actuator's actual position (`evaluate_at_position`), like the BET code's `use_closest_pos` `argmin`. Which one the feedforward uses is the `FlapDrive`'s explicit choice, through its `use_position_lookup`, not the reference's. It is a required keyword argument with no default, so every drive states its choice: `False` reads by time, like `use_closest_pos = false`, and `True` matches `use_closest_pos = true`, which the crank and scotch yoke examples use. It is only meaningful for a reference whose positions are monotonic, as they are for a turning crank, so setting it for a table that moves back and forth raises an error. The reference only reports `positions_are_monotonic`.

### `FlapDrive`

One actuator, its transmission, and the `Wing`s it flaps. It holds:

- `wing_indices`: which of the first `Airplane`'s `Wing`s it drives. Give each `Wing` its own drive for one motor per wing, or put both halves in one drive to share a motor.
- `actuator`: a callable that takes a `FlapDriveInputs` record and returns the actuator's generalized force. The controller and the limits go inside it, typically a `MotorSaturationController` (see below).
- `actuator_inertia` (rotor inertia) and `actuator_damping`.
- `reference`, the `FlapReference` it follows, the only definition of its flap trajectory. It is a required keyword argument with no default. Whether the position lookup works with it is checked at construction.
- Optional `initial_actuator_position` and `initial_actuator_velocity`. They default to the reference's first sample, as the BET code starts the motor on its desired trajectory.
- Optional `passive_wing_torque_function(phi, phi_dot)` and `passive_actuator_force_function(q, q_dot)`, for springs like the BET code's `spring_params`.
- Optional `wing_flap_inertia`, a user given (for example measured) inertia of its `Wing`s about the flap axis. It is the combined inertia of all of the drive's `Wing`s: one `Wing`'s inertia for a per-wing drive, and twice one half's inertia for a drive that flaps both halves of a symmetric `Wing` (see `examples/flap_driven_aeroelastic_unsteady_crank_shared_motor.py`). A shared drive's actuator also carries the sum of its `Wing`s' fluid torques, so size it for all of them. Without it, the problem estimates the inertia from its `wing_density` and the `Wing`s' panel geometry, by summing the same strip flapping-axis inertias that its torsional springs use.

  **Warning:** a user given `wing_flap_inertia` is used only in the drive's equation of motion. The torsional springs that deform the `Wing`s always use the strip inertias from `wing_density`, including their inertial forcing from the flap acceleration. If the two disagree, the drive and the structure model `Wing`s of different mass. If the user given value is the true inertia, the drive moves the wing correctly, but the wing deformation is computed for the wrong mass and is likely invalid. To keep them consistent, leave `wing_flap_inertia` unset and choose `wing_density` so that the estimate is the true inertia. The estimate scales linearly with `wing_density`, and it carries the triple counted mass described under Known Limitations.

The equation of motion generalizes `singleWingDynamics`:

```
(I_a + J^2 * I_w) * q'' = F_a - c * q' + F_p + J * (T_fluid + T_p) - J * I_w * H * q'^2
phi'' = H * q'^2 + J * q''
```

Here `I_a` is the actuator inertia, `I_w` the wing inertia, `c` the damping, `F_a` the actuator function's output, `F_p` and `T_p` the passive force and torque, and `T_fluid` the fluid torque on the wings. `J^2 * I_w` is the wing inertia reflected to the actuator, and it sits on the left side, so it is treated implicitly. The equation multiplies by `J` instead of dividing by a mechanical advantage, so it stays finite at the linkage's dead centers where `J = 0`. Note the sign convention: `T_fluid` is the torque the fluid exerts on the wing, positive in the direction of increasing flap angle, so it enters with `+J`. The BET code negates its aerodynamic torque because it carries a reaction torque instead. The passive elements follow the same convention: `F_p` and `T_p` are the force and torque applied to the actuator and the wing, so a restoring spring is written `-k * (q - q_0)` or `-k * (phi - phi_0)` and the equation adds it. The BET code writes its springs as reaction torques, `+k * (...)`, and subtracts them, which gives the same motion, so a spring copied from it needs its sign flipped.

`FlapDriveAccelerations` also reports `transmission_torque`, the torque the linkage applies to the wing, `I_w * phi'' - T_fluid - T_p`. It equals `wing_torques.total` in the BET code, with the same sign, because that code's inertial term is `+I_w * phi''` and its aerodynamic and spring terms are reaction torques, `-T_fluid` and `-T_p`. Like it, it never gets divided by `J`, so it is the right quantity for sizing the linkage.

### `MotorSaturationController`

A port of the BET code's `motorSaturationController`, limit for limit. A torque command passes through the torque-speed curve (driving torque only, clamped at 0.0, so braking above the no-load speed is allowed), the brake current cap (braking only), the gearbox limit at the motor, the motor's peak and continuous torque, and the motor controller's peak and continuous current, in that order. The motor and the controller each have a time stamped leaky bucket for their peak allowance, with the same expiry time and `peak_duration / 2` lock hysteresis, so the result depends on the time passed in and not on how many times the integrator evaluates it. It reproduces the lock windows given in `MOTOR_LIMIT_SIM_README.md` (0.1001-0.1501 s, 0.2003-0.2502 s, and so on, under a constant 14 N*m demand).

It keeps state, so each motor needs its own instance, and a reused instance must be reset with `reset_state`. Use it inside the actuator callable:

```python
torque_applied, info = saturation_controller.clip_torque(torque_desired, inputs.actuator_velocity, inputs.time)
```

`info` (and `last_info`) report which limits were active, the largest torque possible, and the state of both allowances. As in the BET code, the integrator's trial evaluations are not undone: a time earlier than one already seen is treated as the latest one, and a rejected trial at a later time can start a lock slightly early.

### `FlapDriveInputs`

The record the actuator callable receives. Besides the time and actuator state, it carries the actual flap angle and rate, `J`, `H`, the wing inertia, the desired actuator position, velocity, and acceleration read from the reference, `feedforward_force`, and `feedforward_fluid_torque`. The feedforward force is the inverse dynamics (`singleWingInverseDynamicsController`): the force that would produce the desired acceleration, computed from the same equation of motion as the forward dynamics so the two can't drift apart.

#### The controller and the dynamics get the fluid torque separately

`FlapDriveInputs` is what the controller knows, so it doesn't carry the true fluid torque on the wing. The UVLM torque enters the drive through two separate arguments:

| Call | Argument | Role |
|---|---|---|
| `inverse_dynamics_controller(time, q, q', measured_fluid_torque, ...)` | `measured_fluid_torque` | The controller's measurement. It is used only for the feedforward, and only when `feedforward_aerodynamic` is `True`. |
| `single_wing_dynamics(inputs, fluid_torque)` | `fluid_torque` | The torque that actually acts on the wing in the equation of motion. |

In a simulation `_evaluate_flap_drive` passes the same UVLM torque to both. Because they are separate arguments, a BET model passed as `feedforward_aerodynamic` changes only `feedforward_fluid_torque` (the controller's estimate) and the feedforward. It never changes what the wing feels, and any mismatch shows up as tracking error for the feedback to correct.

**Important difference from the BET feedforward:** the fluid torque in `feedforward_force` is the previous time step's UVLM torque at the wing's actual motion, held constant over the step. `singleWingInverseDynamicsController` instead predicts the aerodynamic torque at the desired motion, by calling `aerodynamicBladeElementTheoryCalc` with the current wing angle and the desired wing velocity `wing_velo_des = J_motor * des_motor_velo`. The UVLM can't cheaply evaluate the loads for a motion the wing isn't making, so that prediction isn't available here. The difference matters most when the motor lags its reference, as while it is saturated: the BET feedforward asks for the torque to overcome the aerodynamic load at the desired, faster flap rate, while this one only asks for the load at the actual, slower rate from one step ago, so it is less aggressive, and it lags where the load changes quickly, near the turnarounds. To use a predicted torque, pass your own model as the `FlapDrive`'s `feedforward_aerodynamic` (see below).

Each term of the feedforward can be switched on or off on the `FlapDrive`, and `FlapDriveInputs` reports each one, so they can be plotted. They sum to `feedforward_force`, and a term that is off is 0.0:

| `FlapDrive` switch (default) | Term (at the motor) | Input field | BET counterpart |
|---|---|---|---|
| `feedforward_inertial` (`True`) | `(I_a + J^2 I_w) q''_des + J I_w H q'_des^2` | `feedforward_inertial_force` | `motor_torque_des_due_to_wing`, `torque_inertial_wing_speed_motor` |
| `feedforward_damping` (`True`) | `c q'_des` | `feedforward_damping_force` | `motor_damping_const * des_motor_velo` |
| `feedforward_passive` (`True`) | `-F_p - J T_p`, at the actual position and the desired velocity | `feedforward_passive_force` | `motor_spring_torque`, `tau_spring_wing_motor` |
| `feedforward_aerodynamic` (`True`) | `-J T_aero` | `feedforward_aerodynamic_force` | `tau_aero_motor` |

`feedforward_aerodynamic` also chooses the aerodynamic model. `True` uses the lagged UVLM torque, `False` leaves the term out, and a callable `model(aerodynamic_inputs)` that returns the torque on the wing (N*m, positive toward increasing flap angle) is your own model, such as a BET model. The switches only change the feedforward. The equation of motion that is integrated always has every term.

#### `AerodynamicModelInputs`: what an aerodynamic model receives

A model passed as `feedforward_aerodynamic` receives one `AerodynamicModelInputs` record, so fields can be added later without breaking it:

| Field | Meaning |
|---|---|
| `time` | The simulation time (s) |
| `wing_indices` | The drive's `Wing`s, so one model can serve several drives |
| `angleXRad_Gs_to_Wn_ixyz`, `angleDerivativeXRad_Gs_to_Wn_ixyz` | The actual flap angle (rad) and rate (rad/s) |
| `desiredAngleDerivativeXRad_Gs_to_Wn_ixyz` | The desired flap rate `J * q'_des` at the actual position, which the BET's `singleWingInverseDynamicsController` evaluates its model at (`wing_velo_des`) |
| `desiredAngleSecondDerivativeXRad_Gs_to_Wn_ixyz` | The desired flap acceleration `H * q'_des^2 + J * q''_des` |
| `actuator_position`, `actuator_velocity`, `desired_actuator_velocity`, `desired_actuator_acceleration`, `jacobian`, `hessian` | The actuator's actual and desired state, and the transmission's derivatives |
| `measured_fluid_torque` | The previous step's UVLM torque, which a model can use as a correction or ignore |
| `operating_point` | The `OperatingPoint` of the time step the measured torque comes from: freestream speed, angles of attack and sideslip, density, and body rates. In free flight it is the vehicle's actual state from the rigid body dynamics, so a model sees the airspeed change as the vehicle flies. It is `None` when `inverse_dynamics_controller` is called outside a simulation. |

Like the measured torque, the `OperatingPoint` is held constant over each time step.

A controller like the BET one is then just:

```python
torque = inputs.feedforward_force + kp * (inputs.desired_actuator_position - inputs.actuator_position) + kd * (inputs.desired_actuator_velocity - inputs.actuator_velocity)
```

followed by whatever clipping the motor needs.

### `AeroelasticWingMovement` and the flap override

`AeroelasticWingMovement` is the existing movement for aeroelastic runs. Its `generate_wing_at_time_step(step, delta_time)` builds each time step's `Wing` from prescribed oscillations, `base + amp * f(2 * pi * t / period + phase)`, for the leading edge position (`ampLer_Gs_Cgs`, ...) and for each angle in `angles_Gs_to_Wn_ixyz` (`ampAngles_Gs_to_Wn_ixyz`, ...), whose x component is the flap angle. It also applies the solver's structural deformation to each `WingCrossSection` through `deformationAngles_Wcsp_to_Wcs_ixyz`, and its `spacingAnglesSecondDerivative_Gs_to_Wn_ixyz` gives the torsional spring model the prescribed flap acceleration for its inertial forcing.

The only change to it is one optional keyword on `generate_wing_at_time_step`, `overrideAngleX_Gs_to_Wn_ixyz` (degrees). When it is given, it replaces the prescribed flap angle for that time step. It is applied after the angles are computed and before the rotation point offset adjustment, so the `Wing` still pivots about the shoulder at the overridden angle. When it is `None`, nothing changes, so existing aeroelastic simulations are unaffected.

With a flap drive, the `Wing`s are moved by `FlapDrivenAeroelasticWingMovement`s, which prescribe no flap motion, and the drive's angle is passed through the override at every time step. The rest of the movement still applies:

| Part of the movement's trajectory | With a flap drive |
|---|---|
| Flap angle, the x component of `angles_Gs_to_Wn_ixyz` | Not prescribed. The drive's angle at each time step |
| Flap acceleration for the torsional spring's inertial forcing | Not prescribed. The drive's actual flap acceleration over each time step |
| The other two angle components (y and z) | Still prescribed, unchanged |
| Leading edge position oscillation (`ampLer_Gs_Cgs`, ...) | Still prescribed, unchanged |
| Rotation point offset | Still applied, about the overridden flap angle |
| Structural deformation | Still applied, unchanged |

#### Warning: the override only changes the geometry, not the flap acceleration

Overriding the flap angle changes the flap's acceleration too, but `overrideAngleX_Gs_to_Wn_ixyz` doesn't know that. It only sets the angle that the `Wing` is built at. The torsional springs, which deform the `Wing`, are forced by `strip inertia * phi''(t)`, and that acceleration reaches them by a separate path, `_generate_inertial_moment_function` on the problem, not through the movement's override.

The two paths are:

| Quantity | Where it is used | Where it comes from in a plain `AeroelasticUnsteadyProblem` | Where it comes from in a `FlapDrivenAeroelasticUnsteadyProblem` |
|---|---|---|---|
| Flap angle `phi(t_(s+1))` | The next time step's `Wing` geometry | The prescribed oscillation, or `overrideAngleX_Gs_to_Wn_ixyz` if given | The drive's ODE solution at `t_(s+1)`, recorded on the movement and passed as the override |
| Flap acceleration `phi''(t)` over `[t_s, t_(s+1)]` | The torsional springs' inertial forcing, so the deformation | Always the prescribed motion (the analytical second derivative of the spacing), even if an override is given | The same ODE solution, `H * q'^2 + J * q''`, sampled across the time step and recorded on the movement |

So in a `FlapDrivenAeroelasticUnsteadyProblem` the angle and the acceleration always come from the same `solve_ivp` result, and they are consistent. If the actuator saturates and the wing decelerates early, the springs feel that deceleration over the time step and the geometry shows the lagging angle at its end. This is why `_advance_flap_drives` must run before the parent's `initialize_next_problem`: both quantities must be recorded before the springs are integrated and the next `Wing` is built.

Passing `overrideAngleX_Gs_to_Wn_ixyz` yourself, outside a `FlapDrivenAeroelasticUnsteadyProblem`, breaks this. The `Wing` would be built at your angle while the springs are still forced by the prescribed acceleration, so the deformation would be computed for a motion the `Wing` isn't doing, with no error or warning. Don't drive the flap through the override directly. Use a `FlapDrive`, which keeps the angle and the acceleration together.

### `FlapDrivenAeroelasticWingMovement`

A subclass of `AeroelasticWingMovement` that can't prescribe a flap motion, because the drive's `FlapReference` is the only definition of the flap trajectory. It is built so that a flap motion can't even be written down: its angular oscillation parameters take only the y and z components, two values each, where `AeroelasticWingMovement` takes three.

| `AeroelasticWingMovement` parameter (x, y, z) | `FlapDrivenAeroelasticWingMovement` parameter (y, z) |
|---|---|
| `ampAngles_Gs_to_Wn_ixyz` | `ampAnglesYZ_Gs_to_Wn_ixyz` |
| `periodAngles_Gs_to_Wn_ixyz` | `periodAnglesYZ_Gs_to_Wn_ixyz` |
| `spacingAngles_Gs_to_Wn_ixyz` | `spacingAnglesYZ_Gs_to_Wn_ixyz` |
| `phaseAngles_Gs_to_Wn_ixyz` | `phaseAnglesYZ_Gs_to_Wn_ixyz` |
| `spacingAnglesSecondDerivative_Gs_to_Wn_ixyz` | `spacingAnglesSecondDerivativeYZ_Gs_to_Wn_ixyz` |

Its initialization method puts a static x component in front of the y and z values (an amplitude, period, and phase of 0.0, with `"sine"` spacing and no second derivative) and passes the three components to `AeroelasticWingMovement`. So the inherited three component attributes, such as `ampAngles_Gs_to_Wn_ixyz`, still exist, and their x component is always 0.0. Passing three values, or the old three component keyword names, raises an error. The leading edge position, the `WingCrossSection`s' motion, and the rotation point offset take the same parameters as before.

It holds the flap state of one simulation, which the problem writes:

- The flap angle at each time step. When it generates a `Wing`, it passes the recorded angle as the override. A step with no recorded angle uses the base `Wing`'s flap angle, but the problem records the first step's angle before any `Wing` is built, so in a simulation every step uses the drive's angle.
- The flap acceleration over the latest time step, for the torsional springs.
- The flap period, `flap_period`, set from the drive's reference with `set_flap_period`. Its `all_periods` and `max_period` include it, so the movement's cycle averaging sees it. `reset_flap_state` clears all three.

`from_aeroelastic_wing_movement(existing)` copies an existing movement's parameters, passing only the y and z components of its angular motion and discarding the x (flap) component, so an existing script only needs to swap its movements and give its drives a reference.

### `FlapDrivenAeroelasticUnsteadyProblem`

A subclass of `AeroelasticUnsteadyProblem` that takes `flap_drives`. It is solved by the unchanged `AeroelasticUnsteadyRingVortexLatticeMethodSolver`. At construction it validates the drives, checks that each drive's `FlapReference` covers the simulation, warns about a reference with no period, starts each drive at its initial state, records that starting flap angle so the first time step's `Wing`s use it, hands each reference's period to its `Wing`s' movements, estimates or takes each drive's wing inertia, and seeds the histories. It overrides two parent methods:

- `initialize_next_problem`, to advance the drives before calling the parent.
- `_generate_inertial_moment_function`, so the torsional spring model's inertial forcing uses the wing's actual flap acceleration.

The results are stored per drive and per time step in `flap_actuator_positions`, `flap_actuator_velocities`, `flap_actuator_forces`, `flap_fluid_torques`, `flap_transmission_torques`, `flap_angles_rad`, and `flap_angle_derivatives_rad`. All are saved and loaded with the problem. The actuator callable comes back from a saved file as an inert placeholder, so a loaded problem can be inspected but not rerun unless you pass the function to `load`.

#### The `_FlapDriveCoupling` mixin

The flap drive logic lives in a mixin, `_FlapDriveCoupling`, that both this problem and `FreeFlightFlapDrivenAeroelasticUnsteadyProblem` inherit, so it is written once. Both parent problems declare `__slots__`, and Python won't let a class have two parents that do, so the mixin has no slots and no `__init__`. Each problem declares the flap drive slots (`_FLAP_DRIVE_SLOTS`) itself, and its `__init__` calls the mixin's `_prepare_flap_drives` before its parent's `__init__` and `_initialize_flap_drives` after it. The mixin's `initialize_next_problem` advances the drives and then calls the next class's, which is `AeroelasticUnsteadyProblem`'s here and `FreeFlightAeroelasticUnsteadyProblem`'s in free flight.

### `FreeFlightFlapDrivenAeroelasticUnsteadyProblem`

A subclass of `FreeFlightAeroelasticUnsteadyProblem` that takes `flap_drives`, through the same mixin. It takes the free flight parameters (`mass`, `I_BP1_CgP1`, `external_loads_fn`, `k_max`, `integrator`, `extra_xml`, `mujoco_assets`) plus `flap_drives` and `step_discards`, and it has the same flap histories. It is solved by the unchanged `FreeFlightAeroelasticUnsteadyRingVortexLatticeMethodSolver`, and its movement is a `FreeFlightAeroelasticMovement` whose driven `Wing`s are moved by `FlapDrivenAeroelasticWingMovement`s.

Each time step has three stages, all inside `initialize_next_problem(solver, s)`:

```
solver solves step s (body state x_s, flap angle phi_s)
  -> FreeFlightFlapDrivenAeroelasticUnsteadyProblem.initialize_next_problem(step = s)
       1. _advance_flap_drives: fluid torque and OperatingPoint from step s, integrate each drive to t_(s+1), record phi_(s+1)
       2. FreeFlightAeroelasticUnsteadyProblem.initialize_next_problem (unchanged):
            a. torsional spring ODEs over [t_s, t_(s+1)], forced by the drives' actual phi''(t)
            b. generate_airplane_at_time_step(s + 1) at phi_(s+1) and the new deformation
            c. prescribed phase: the body coasts at its initial state
               free flight phase: the strongly coupled sub-iteration drives the loads on that Airplane
               and the rigid body state to agreement, with the geometry held fixed
solver sheds the wake and solves step s + 1
```

The flap drives and the torsional springs run once per time step, before the sub-iteration. So the rigid body is coupled strongly, the flap and the deformation loosely (one step late, as in the non free flight problem), and no flap state, including a `MotorSaturationController`'s buckets, has to be saved and restored inside the sub-iteration. The `OperatingPoint` an aerodynamic model receives is the accepted body state at step `s`, never a trial state from the sub-iteration.

#### What the rigid body dynamics don't include

MuJoCo models the vehicle as one rigid body with the constant `mass` and `I_BP1_CgP1` given to the problem. The flap drives' dynamics don't reach it. The body feels the aerodynamic loads on the flapping wings, gravity, and `external_loads_fn`, but not:

- The actuators' reaction torques on the airframe. Each motor's stator pushes back on the airframe with the motor's torque, and a spinning rotor adds a gyroscopic torque when the body rotates. For a left and right pair of drives the reactions largely cancel, but not when one saturates or the two differ.
- The flapping wings' inertial loads on the airframe. Swinging the wings' mass back and forth shakes a real vehicle at the flap frequency.
- The changing mass distribution. The CG and inertia stay fixed while the wings flap. Give `mass` and `I_BP1_CgP1` that include the wings and the motors, for example at the mean flap angle.

"Planned: Actuator and Wing Inertia in Free Flight" below describes how these could be added.

## How It Plugs Into the Solver

The aeroelastic solver already builds geometry one step at a time. After solving each step it calls the problem's `initialize_next_problem`, which builds the next step's `Airplane` before the wake is shed. The flap drive inserts itself at the start of that call:

```
solver solves step s (flap angle phi_s, deformation from step s - 1)
  -> FlapDrivenAeroelasticUnsteadyProblem.initialize_next_problem(step = s)
       1. _advance_flap_drives: fluid torque from step s's loads, integrate each drive to t_(s+1), record phi_(s+1) on the movements
       2. AeroelasticUnsteadyProblem.initialize_next_problem (unchanged)
            a. torsional spring ODEs over [t_s, t_(s+1)], inertial forcing = strip inertia * actual phi''(t)
            b. generate_airplane_at_time_step(s + 1): the movement applies phi_(s+1) through the override
solver sheds the wake and solves step s + 1
```

The flap torque about the shoulder is computed from loads the solver already stores on every `Panel`: `T = sum(moments_GP1_CgP1) - p x sum(forces_GP1)`, taken along the flap axis, with `p` the rotation point. For a mirror-only `Wing` the axis is reflected and the sign flipped. The main and reflected wings of the example agree to about 6e-8 relative, so a shared drive adds their torques instead of cancelling them.

## Time Steps

There are four clocks.

**The UVLM time step, `delta_time`.** This is the outer step, set on the `AeroelasticMovement`. The aerodynamic loads, the geometry, and the wake all advance once per `delta_time`.

**The drive's internal steps.** Within each UVLM step, each drive's state `(q, q')` is integrated from `t_s` to `t_(s+1)` with `scipy.integrate.solve_ivp` (RK45, relative tolerance 1e-6, absolute tolerance 1e-9). The solver picks its own sub-steps, usually several to many per UVLM step, and calls your actuator function at every one of them and at its rejected trial steps. Saturation and controller behavior therefore resolve within a step, but your actuator function must depend only on its input, as `ode45` requires in the BET code. Integration restarts from the last state at every step.

**The fluid torque's lag.** The fluid torque used over `[t_s, t_(s+1)]` comes from step `s`'s solution and is held constant over the interval. It is one step old when it acts, the same lag the existing structural deformation has. This is a loose (explicit) coupling. Keep `delta_time` small relative to the flap period. The example uses 50 steps per cycle. Very stiff controller gains or a very light wing can make this lag matter, and it's worth halving `delta_time` to check convergence. The planned fix is described in "Planned: Strongly Coupling the Flap" below.

**The reference's samples.** The `FlapReference` table is sampled every 1e-4 s, independent of the UVLM step, and interpolated linearly in time between samples. The inertial forcing passed to the torsional spring model is sampled at 21 points per UVLM step from the drive's dense output and interpolated linearly.

Two special cases:

- **Startup.** For the first `step_discards` steps (default 5), the drives see zero fluid torque, just as the structural model ignores the UVLM's startup transient. The flap follows the actuator's own dynamics during that time.
- **The first step.** The step-0 geometry is built at the transmission's flap angle at each drive's initial actuator position, before any dynamics run. The initial actuator position defaults to the reference's first position.

The last time step doesn't advance the drives, so every history list has exactly `num_steps` entries, and entry `i` describes time `i * delta_time`. The exception is `flap_fluid_torques[i]`, which is the torque that acted over the interval ending at step `i`.

## What Is Analytical

All derivatives are analytical: the crank kinematics and inverse kinematics with their derivatives, the generated flap trajectories' first and second derivatives, the equation of motion, and the feedforward. The approximations are:

- The `FlapReference` table and its linear interpolation, in time and in position.
- The 21-point linear interpolation of the flap acceleration for the torsional spring forcing.
- The ODE integration of the drive.
- The fluid torque held constant over each step.
- In the example, the stroke amplitude found by sampling the forward kinematics at 20,000 points, as `runSingleSimulation` does.

## Planned: Strongly Coupling the Flap

This is not implemented. It records the design for removing the one step lag, so it can be built later.

### The problem

The UVLM solves each step once. The fluid torque `T_s` from step `s` is then held constant over `[t_s, t_(s+1)]` while the drive moves the wing, so the fluid's reaction to that motion only arrives in `T_(s+1)`, one step later. This has two effects:

- A phase error. The aerodynamic load is late by about `360 * delta_time / period` degrees. With the examples' 50 steps per wingbeat, that is about 7 degrees.
- A possible instability. Part of the fluid torque resists the wing's acceleration (added mass, roughly `T ~ -I_added * phi''`). Applied a step late, it acts as a delayed restoring torque. If `I_added` is large relative to the wing's inertia plus the actuator's reflected inertia, `I_w + I_a / J^2`, the flap overshoots and can diverge. A heavy geared motor makes this unlikely, while a light wing or a direct drive makes it likely.

A smaller `delta_time` shrinks both effects but doesn't remove them, and the cost of a UVLM step grows with the wake.

### The fix: sub-iterate each step

Each time step becomes a fixed-point iteration that drives the flap motion and the fluid torque to agreement before the step is committed. This is the same approach `FreeFlightUnsteadyProblem` already uses for the rigid body (see [Strong Coupling](STRONG_COUPLING.md)).

```
save the drive's state at t_s (the last history entries) and the actuator function's state
guess the torque at t_(s+1): T_trial = T_s, or the extrapolation 2 * T_s - T_(s-1)
repeat k = 0, 1, 2, ... up to k_max:
    restore the drive's and the actuator function's state at t_s
    integrate the drive over [t_s, t_(s+1)] with T(t) = linear from T_s to T_trial
        -> phi_trial at t_(s+1), and phi''_trial(t) over the interval
    integrate the torsional springs over the same interval with phi''_trial(t)
    build a trial Airplane at phi_trial and the trial deformation
    solve the UVLM at step s + 1 on the trial Airplane, without committing it
        -> T_new, the flap torque on the trial Airplane
    residual = T_new - T_trial
    if the residual passes the convergence test: stop
    relax: T_trial = T_trial + omega_k * residual, with an Aitken omega_k
commit the converged step: record phi, phi'', and the histories, and build the real Airplane at s + 1
```

The unknown iterated on is the fluid torque at the end of the step, one scalar per drive. Once it is known, the drive's motion over the step follows from integrating the drive, so nothing else needs a guess.

### What it needs

- **A solver that can evaluate a step without committing it.** `FreeFlightAeroelasticUnsteadyRingVortexLatticeMethodSolver` already has `freeze_substep`, `evaluate_trial_aero_loads`, and `restore_substep` for the body's sub-iteration. Its transient next-step `SteadyProblem` is built over a copy of the next `Airplane`, so building it over a new trial `Airplane` at each iteration is a small extension. `AeroelasticUnsteadyRingVortexLatticeMethodSolver` has no such interface and would need one, which changes shared solver code. So the free-flight flap-driven problem is the natural first home for this.
- **A restartable actuator function.** The drive itself has no state, and each iteration restarts from the last history entries. The actuator function can hold state, though. `MotorSaturationController` keeps its leaky buckets and treats a time earlier than one it has already seen as the latest time seen. Restarting at `t_s` after integrating to `t_(s+1)` would therefore be wrong by a whole step, not by one integrator step. It needs a `save_state` and `restore_state` pair, called around every iteration. Logs the example's actuator function keeps, such as `max_torque_possible_log`, are keyed by time, so the accepted iteration overwrites the trial ones.
- **The springs in the same loop.** The trial `Airplane` carries both the flap angle and the deformation, so the torsional springs should be integrated inside the loop. That removes the deformation's lag too.
- **A convergence test and relaxation.** Reuse `_fixed_point_relaxation.py`: its weighted norm, its convergence test with a relative and an absolute tolerance, and its Aitken relaxation factor. Scale the torque residual by a reference torque, such as the drive's peak transmission torque, so one tolerance works across vehicle sizes.
- **In free flight, one loop for the body and the flap.** The body's sub-iteration already iterates the 12-vector of the rigid body state. Append one entry per drive (the trial torque) to that vector and give it a weight in `_build_relaxation_weights`. Then a single loop drives the body, the flap, and the loads to agreement, and each iteration costs one UVLM solve, as now.

### Cost

Each iteration costs one extra UVLM solve. With Aitken relaxation, expect 2 to 5 iterations per step, so a run takes about 2 to 5 times as long. In free flight with the merged loop, the body's iteration already pays for these solves, so the extra cost is only the iterations the flap adds.

### Cheaper alternatives, if strong coupling is too slow

- **Extrapolate the torque.** Use `T(t) = T_s + (T_s - T_(s-1)) * (t - t_s) / delta_time` over the interval instead of a constant. This needs no extra solves and cuts most of the phase error for a smooth periodic load, but it doesn't fix the added mass instability, and it amplifies the UVLM's step to step noise.
- **Treat added mass implicitly.** Give `FlapDrive` an added flap inertia `I_added`, use `I_w + I_added` in the effective inertia, and pass `T_s + I_added * phi''_s` as the fluid torque. This is the standard fix for the added mass instability in loosely coupled fluid structure solvers, and it needs no extra solves. `I_added` can come from thin plate theory (`(pi / 8) * rho * c^2 * r^2` per strip, summed along the span) or from two UVLM solves at different flap accelerations at the start of a run. The circulatory part of the torque is still one step late.

## Planned: Actuator and Wing Inertia in Free Flight

This is not implemented. It records two ways to make the free flight rigid body feel the flap drives' dynamics, so they can be built later.

### Apply the reaction loads to the body

Keep the drives integrated by `solve_ivp` and the body as one MuJoCo body. Each step, turn the drives' results into the loads the moving parts put on the airframe, and add them to the interval loads in `_assemble_interval_loads`, next to the aerodynamic loads and `external_loads_fn`:

- For each wing, with mass `m_w`, center of mass `r_w`, and inertia `I_w` relative to the airframe: the force `-m_w * a_w`, where `a_w` is the wing's center of mass acceleration from `phi''` and `phi'^2` about the shoulder, and the moment `-(r_w x m_w * a_w) - I_w * phi'' * e_flap` plus the gyroscopic terms from the body's rotation.
- For each motor: the stator's reaction `-F_a` about the motor's shaft, plus the rotor's gyroscopic torque `I_a * q' * (e_shaft x omega_body)`. The linkage's pin loads balance the wing's inertia, which the wing's reaction already counts.

Every input already comes out of the drives (`phi`, `phi'`, `phi''`, `q'`, `q''`, `F_a`). The new inputs are each wing's mass and center of mass offset, and each motor's shaft axis. This captures the airframe shaking at the flap frequency and an unbalanced motor reaction. It doesn't capture the moving CG or the changing inertia, which is a small error when the wings are light relative to the body, and the loads are one step late, like the fluid torque.

### Model the wings and rotors as MuJoCo bodies

Change the generated MuJoCo model from one body to a tree: the airframe with a free joint, each rotor on a hinge about its shaft, and each wing on a hinge about its flap axis at the shoulder. The motor torque acts on the rotor joint and the UVLM flap torque on the wing joint, and MuJoCo then handles the reaction torques, the inertial loads, the moving CG, and the gyroscopic effects. MuJoCo would replace `solve_ivp` as the drive integrator, so this is where a controller that varies the flap to reach body states would live.

The crank linkage is the hard part. MuJoCo's polynomial joint coupling only goes to fourth order and can't wrap a continuously turning crank. Either model the linkage's bars as bodies with a closed loop `connect` constraint, or keep `CrankLinkage` and apply the transmission through a MuJoCo control callback at every MuJoCo substep: wing joint torque `T_transmission`, rotor joint torque `F_a - J * T_transmission`.

Because the free flight sub-iteration already saves and restores the MuJoCo state for every trial, the wing and rotor joints would be sub-iterated with the body automatically, which also gives the flap strong coupling (see "Planned: Strongly Coupling the Flap" above). Only the `MotorSaturationController`'s buckets would need their own save and restore. This changes `_mujoco_model.py`, the state vector and weights of the sub-iteration, how the geometry gets its flap angle, serialization, and rendering.

## Known Limitations

[Flap Drive TODO](FLAP_DRIVE_TODO.md) lists every known issue and planned rework, with priorities and fixes. The main ones are:

- `AeroelasticUnsteadyProblem._calculate_mass_matrix` copies each panel's mass into three components, and `_calculate_strip_inertias` sums all three, so strip masses and inertias are three times too large. The flap drive uses that estimate only when `wing_flap_inertia` isn't given. This bug is in existing code and is not fixed.
- The flap is coupled to the fluid loosely, so the fluid torque acts one step late, which causes a phase error and can make a light wing's flap unstable. See "Planned: Strongly Coupling the Flap" above.
- The feedforward's aerodynamic torque is the previous step's UVLM torque at the actual motion, not a prediction at the desired motion as in the BET code's `singleWingInverseDynamicsController`. See `FlapDriveInputs` above.
- The example's torsional spring model is Ptera Software's strip model, not the BET code's torsional deformation, so deformation results won't match.
- A `FlapReference` whose trajectory doesn't repeat with a fixed period (`period=None`) gives inaccurate cycle averaged results, and if nothing else moves, the simulation is treated as static. The problem warns about it. See "The period and cycle averaged results" above.
- In `FreeFlightFlapDrivenAeroelasticUnsteadyProblem`, the rigid body doesn't feel the motors' reaction torques, the wings' inertial loads, or the moving CG, and its mass and inertia are constant. See "What the rigid body dynamics don't include" above.
- `CLASSES_AND_IMMUTABILITY.md` doesn't describe the new classes yet, and the slot tests in `test_slots.py` don't cover them.
