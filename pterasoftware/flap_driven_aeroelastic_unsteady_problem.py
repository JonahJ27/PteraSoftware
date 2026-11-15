"""Contains the FlapDrivenAeroelasticUnsteadyProblem class."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.integrate import solve_ivp

from . import (
    _logging,
    _transformations,
)
from . import flap_drive as flap_drive_mod
from . import geometry, problems
from .movements import aeroelastic_movement as aeroelastic_movement_mod
from .movements import aeroelastic_wing_movement as aeroelastic_wing_movement_mod
from .movements import (
    flap_driven_aeroelastic_wing_movement as flap_driven_wing_movement_mod,
)

if TYPE_CHECKING:
    from ._coupled_unsteady_ring_vortex_lattice_method import (
        CoupledUnsteadyRingVortexLatticeMethodSolver,
    )

_logger = _logging.get_logger("flap_driven_aeroelastic_unsteady_problem")

# These are the tolerances for integrating a FlapDrive's actuator over one time step.
# They match the strictness of the torsional spring integration in problems.py, for the
# same reason: the integration is re-seeded from the previous state at every time step,
# so its local errors compound across a simulation.
_FLAP_ODE_RELATIVE_TOLERANCE = 1e-6
_FLAP_ODE_ABSOLUTE_TOLERANCE = 1e-9

# This is the number of points at which a time step's flap angular acceleration is
# sampled for the torsional spring's inertial forcing. The spring integration then reads
# the samples by linear interpolation, instead of calling the user's actuator function
# on every evaluation.
_FLAP_ACCELERATION_SAMPLES = 21


def _evaluate_flap_drive(
    flap_drive: flap_drive_mod.FlapDrive,
    time: float,
    actuator_position: float,
    actuator_velocity: float,
    fluid_torque: float,
    wing_flap_inertia: float,
) -> tuple[flap_drive_mod.FlapDriveInputs, flap_drive_mod.FlapDriveAccelerations]:
    """Evaluates a FlapDrive's equation of motion at one instant.

    :param flap_drive: The FlapDrive to evaluate.
    :param time: The simulation time (s).
    :param actuator_position: The actuator position, in the units of the drive's
        transmission.
    :param actuator_velocity: The actuator velocity, in the units of the drive's
        transmission per second.
    :param fluid_torque: The fluid torque on the drive's Wings about their flap axis
        (N*m).
    :param wing_flap_inertia: The drive's Wings' combined inertia about the flap axis
        (kg*m^2).
    :return: A tuple of the FlapDriveInputs passed to the actuator function and the
        resulting FlapDriveAccelerations.
    """
    inputs = flap_drive.inverse_dynamics_controller(
        time,
        actuator_position,
        actuator_velocity,
        fluid_torque,
        wing_flap_inertia,
    )
    return inputs, flap_drive.single_wing_dynamics(inputs)


def _clear_movement_period_caches(
    movement: aeroelastic_movement_mod.AeroelasticMovement,
) -> None:
    """Clears the cached periods of a movement and its AirplaneMovements.

    The movement classes compute their periods once and cache them. A
    FlapDrivenAeroelasticUnsteadyProblem sets its FlapDrivenAeroelasticWingMovements'
    flap periods after the movement exists, and the movement may have read and cached
    its periods before then, so the caches are cleared and recomputed on their next
    read. This reaches into the private caches of the core movement classes, which have
    no public way to invalidate them, so a test checks that these names still exist.

    :param movement: The AeroelasticMovement whose cached periods to clear.
    :return: None
    """
    for airplane_movement in movement.airplane_movements:
        airplane_movement._all_periods = None
        airplane_movement._max_period = None
    movement._lcm_period = None
    movement._max_period = None
    movement._min_period = None
    movement._static = None


def _make_interpolated_function(
    times: np.ndarray, values: np.ndarray
) -> Callable[[float], float]:
    """Returns a function that reads sampled values by linear interpolation in time.

    :param times: A (N,) ndarray of floats holding the sample times, in increasing
        order.
    :param values: A (N,) ndarray of floats holding the value at each sample time.
    :return: A callable that accepts a time and returns the interpolated value.
    """
    sample_times = np.array(times, dtype=float)
    sample_values = np.array(values, dtype=float)

    def interpolated(time: float) -> float:
        return float(np.interp(time, sample_times, sample_values))

    return interpolated


class FlapDrivenAeroelasticUnsteadyProblem(problems.AeroelasticUnsteadyProblem):
    """A subclass of AeroelasticUnsteadyProblem whose Wings' flap angles come from
    FlapDrives instead of from the movement's prescribed oscillation.

    Inherits the structural model and the coupling to the unsteady solver from
    AeroelasticUnsteadyProblem without modification. It is solved by the same
    AeroelasticUnsteadyRingVortexLatticeMethodSolver.

    **What changes each time step:**

    After the solver finishes a time step, this problem sums the aerodynamic torque on
    each FlapDrive's Wings about their flap axis, and integrates the FlapDrive's
    actuator over the next time step with that torque held constant. The resulting flap
    angle sets the Wings' flap angle at the next time step, and the resulting flap
    angular acceleration is the flap acceleration in each strip's inertial forcing.
    Wings that belong to no FlapDrive keep their prescribed flap motion. The aerodynamic
    torque is one time step old when it acts, the same lag that the structural
    deformation has.

    Each FlapDrive follows its FlapReference, a table of its actuator's desired motion
    and the only definition of its flap trajectory. The FlapDrive's Wings are moved by
    FlapDrivenAeroelasticWingMovements, which prescribe no flap motion. The first time
    step's Wings start at the transmission's flap angle at each drive's initial actuator
    position.

    **Cycle averaged results:**

    The solver averages loads over the last full period of all the motions, which it
    takes from the movement. Because the FlapDrivenAeroelasticWingMovements prescribe no
    flap motion, this problem gives each FlapDrive's Wings' movements the period of the
    FlapDrive's reference before the parent class reads the movement's periods, so the
    averages cover whole wingbeats. A reference whose period is None, such as one whose
    flapping frequency changes, gives inaccurate cycle averaged results, and if nothing
    else in the movement moves, the simulation is treated as static: it reports the
    final time step's loads and labels its geometry as static. This problem logs a
    warning in that case.

    The Wings' inertia about the flap axis comes from the same wing density and strip
    model as the torsional spring, unless the FlapDrive supplies a measured value.

    **Flap drive histories:**

    Each of the lists below holds one list per FlapDrive, in the order of flap_drives,
    and each of those holds one entry per time step, starting at the first. An entry
    describes the drive at the instant of its time step: flap_actuator_positions,
    flap_actuator_velocities, and flap_actuator_accelerations hold the actuator's state
    and acceleration, flap_angles_rad, flap_angle_derivatives_rad, and
    flap_angle_second_derivatives_rad hold the Wings' flap angle (rad) and its first and
    second time derivatives (rad/s and rad/s^2), flap_actuator_forces holds the actuator
    function's output, and flap_transmission_torques holds the torque (N*m) that the
    transmission applies to the Wings. flap_fluid_torques holds the fluid torque (N*m)
    that acted on the Wings over the interval that ends at the time step, which is the
    torque from the previous time step's solution, and it is 0.0 at the first time step
    and during the discarded time steps.
    """

    __slots__ = (
        "_flap_drives",
        "_flap_wing_flap_inertias",
        "flap_actuator_positions",
        "flap_actuator_velocities",
        "flap_actuator_accelerations",
        "flap_actuator_forces",
        "flap_fluid_torques",
        "flap_transmission_torques",
        "flap_angles_rad",
        "flap_angle_derivatives_rad",
        "flap_angle_second_derivatives_rad",
    )

    def __init__(
        self,
        movement: aeroelastic_movement_mod.AeroelasticMovement,
        wing_density: float,
        spring_constant_rad: float,
        damping_constant_rad: float,
        flap_drives: Sequence[flap_drive_mod.FlapDrive],
        step_discards: int = 5,
    ) -> None:
        """The initialization method.

        :param movement: An AeroelasticMovement containing the prescribed motion and
            aerodynamic setup. The Wings that a FlapDrive flaps must be moved by
            FlapDrivenAeroelasticWingMovements in its first AeroelasticAirplaneMovement.
            Their recorded flap states are cleared and their flap periods are set from
            the FlapDrives' references, so use a new movement for each problem.
        :param wing_density: The mass per unit span area of the wing (kg/m^2). Used to
            distribute wing mass across Panels for inertial calculations.
        :param spring_constant_rad: The torsional spring stiffness for the spring-mass-
            damper model (N*m/rad). Controls the restoring moment opposing deformation.
        :param damping_constant_rad: The torsional damping coefficient (N*m*s/rad).
            Controls the viscous damping in the spring-mass-damper system.
        :param flap_drives: A sequence of FlapDrives. Can be a tuple or list. It must be
            non empty. Each FlapDrive's wing_indices index the first Airplane's Wings,
            every one of those Wings must be moved by a
            FlapDrivenAeroelasticWingMovement, and no Wing can belong to two FlapDrives.
            Each FlapDrive's reference must cover the whole simulation.
        :param step_discards: The number of initial time steps to discard for numerical
            stability (there are inconsistent startup effects from the UVLM solver).
            During these time steps, the solver will run but the results will not be
            applied to the deformation of the Wings, and the FlapDrives act as if the
            fluid applied no torque. It must be a non negative int. The default is 5.
        :return: None
        """
        if not isinstance(movement, aeroelastic_movement_mod.AeroelasticMovement):
            raise TypeError("movement must be an AeroelasticMovement.")

        if isinstance(flap_drives, (str, bytes)) or not isinstance(
            flap_drives, (Sequence, np.ndarray)
        ):
            raise TypeError("flap_drives must be a tuple or list of FlapDrives.")
        if len(flap_drives) == 0:
            raise ValueError("flap_drives can't be empty.")
        for position, flap_drive in enumerate(flap_drives):
            if not isinstance(flap_drive, flap_drive_mod.FlapDrive):
                raise TypeError(f"flap_drives[{position}] must be a FlapDrive.")
        validated_flap_drives = tuple(flap_drives)

        # Check which Wings each FlapDrive moves.
        first_wing_movements = movement.airplane_movements[0].wing_movements
        claimed_wing_indices: set[int] = set()
        for position, flap_drive in enumerate(validated_flap_drives):
            for wing_index in flap_drive.wing_indices:
                if wing_index >= len(first_wing_movements):
                    raise ValueError(
                        f"flap_drives[{position}] has a wing index of {wing_index}, but "
                        f"the first Airplane only has {len(first_wing_movements)} "
                        "Wings."
                    )
                if wing_index in claimed_wing_indices:
                    raise ValueError(
                        f"Wing {wing_index} belongs to more than one FlapDrive."
                    )
                claimed_wing_indices.add(wing_index)
                if not isinstance(
                    first_wing_movements[wing_index],
                    flap_driven_wing_movement_mod.FlapDrivenAeroelasticWingMovement,
                ):
                    raise TypeError(
                        f"Wing {wing_index} belongs to flap_drives[{position}], so it "
                        "must be moved by a FlapDrivenAeroelasticWingMovement."
                    )

        # Clear any flap state left on the movements by an earlier problem, so that this
        # problem's records start at the first time step.
        for wing_movement in first_wing_movements:
            if isinstance(
                wing_movement,
                flap_driven_wing_movement_mod.FlapDrivenAeroelasticWingMovement,
            ):
                wing_movement.reset_flap_state()

        # Check each drive's reference, find its initial state, and record its starting
        # flap angle and its flap period on its Wings' movements. This happens before
        # the parent creates the first time step's Airplane, so that Airplane starts at
        # the drives' angles, and before the parent reads the movement's periods for the
        # cycle averaging.
        final_time = (movement.num_steps - 1) * movement.delta_time
        initial_states: list[tuple[float, float]] = []
        for position, flap_drive in enumerate(validated_flap_drives):
            reference = flap_drive.reference
            if reference.times[-1] < final_time:
                raise ValueError(
                    f"The reference of flap_drives[{position}] ends at "
                    f"{reference.times[-1]} s, before the simulation does."
                )
            if reference.period is None:
                _logger.warning(
                    f"The reference of flap_drives[{position}] has no period, so the "
                    "cycle averaged results are inaccurate, because no fixed window "
                    "covers whole wingbeats. If nothing else in the movement moves, "
                    "the simulation is also treated as static: it reports the final "
                    "time step's loads and labels its geometry as static."
                )

            # Start at the drive's own initial state, or else where its reference
            # starts.
            initial_position = flap_drive.initial_actuator_position
            if initial_position is None:
                initial_position = float(reference.positions[0])
            initial_velocity = flap_drive.initial_actuator_velocity
            if initial_velocity is None:
                initial_velocity = float(reference.velocities[0])
            initial_states.append((initial_position, initial_velocity))

            initial_angle_deg = float(
                np.rad2deg(flap_drive.transmission.evaluate_angle(initial_position))
            )
            for wing_index in flap_drive.wing_indices:
                drive_wing_movement = cast(
                    flap_driven_wing_movement_mod.FlapDrivenAeroelasticWingMovement,
                    first_wing_movements[wing_index],
                )
                drive_wing_movement.record_flap_step(0, initial_angle_deg, None)
                drive_wing_movement.set_flap_period(reference.period)

        _clear_movement_period_caches(movement)

        super().__init__(
            movement=movement,
            wing_density=wing_density,
            spring_constant_rad=spring_constant_rad,
            damping_constant_rad=damping_constant_rad,
            step_discards=step_discards,
        )

        self._flap_drives = validated_flap_drives

        initial_airplane = self._steady_problems[0].airplanes[0]
        wing_flap_inertias: list[float] = []
        self.flap_actuator_positions: list[list[float]] = []
        self.flap_actuator_velocities: list[list[float]] = []
        self.flap_actuator_accelerations: list[list[float]] = []
        self.flap_actuator_forces: list[list[float]] = []
        self.flap_fluid_torques: list[list[float]] = []
        self.flap_transmission_torques: list[list[float]] = []
        self.flap_angles_rad: list[list[float]] = []
        self.flap_angle_derivatives_rad: list[list[float]] = []
        self.flap_angle_second_derivatives_rad: list[list[float]] = []

        for flap_drive, (initial_position, initial_velocity) in zip(
            validated_flap_drives, initial_states
        ):
            # Use the drive's measured Wing inertia if it has one, and otherwise
            # estimate it from the wing density and the Wings' geometry. This is chosen
            # once, here, and every evaluation of the drive then receives it.
            wing_flap_inertia = flap_drive.wing_flap_inertia
            if wing_flap_inertia is None:
                wing_flap_inertia = self._calculate_wings_flap_inertia(
                    [initial_airplane.wings[i] for i in flap_drive.wing_indices]
                )
            wing_flap_inertias.append(wing_flap_inertia)

            # Seed each history with the drive's state at the first time step, before
            # the fluid has acted on it.
            initial_inputs, initial_accelerations = _evaluate_flap_drive(
                flap_drive,
                0.0,
                initial_position,
                initial_velocity,
                0.0,
                wing_flap_inertia,
            )
            self.flap_actuator_positions.append([initial_inputs.actuator_position])
            self.flap_actuator_velocities.append([initial_inputs.actuator_velocity])
            self.flap_actuator_accelerations.append(
                [initial_accelerations.actuator_acceleration]
            )
            self.flap_actuator_forces.append([initial_accelerations.actuator_force])
            self.flap_fluid_torques.append([0.0])
            self.flap_transmission_torques.append(
                [initial_accelerations.transmission_torque]
            )
            self.flap_angles_rad.append([initial_inputs.angleXRad_Gs_to_Wn_ixyz])
            self.flap_angle_derivatives_rad.append(
                [initial_inputs.angleDerivativeXRad_Gs_to_Wn_ixyz]
            )
            self.flap_angle_second_derivatives_rad.append(
                [initial_accelerations.angleSecondDerivativeXRad_Gs_to_Wn_ixyz]
            )

        self._flap_wing_flap_inertias = tuple(wing_flap_inertias)

    # --- Immutable: read only properties ---
    @property
    def flap_drives(self) -> tuple[flap_drive_mod.FlapDrive, ...]:
        return self._flap_drives

    def _calculate_wings_flap_inertia(self, wings: list[geometry.wing.Wing]) -> float:
        """Estimates the combined inertia of Wings about their flap axis.

        It sums the strip flapping-axis inertias that the torsional spring model uses,
        so the drive and the structure agree on the Wings' mass distribution.

        :param wings: The Wings to include.
        :return: The combined inertia, in kilogram square meters.
        """
        total = 0.0
        for wing in wings:
            num_spanwise_panels = wing.num_spanwise_panels
            assert num_spanwise_panels is not None
            mass_matrix = self._calculate_mass_matrix(wing)
            # A mirror-meshed Wing's Panel grid runs tip to root spanwise, but the strip
            # inertias run root to tip.
            if wing.mirror_only:
                mass_matrix = mass_matrix[:, ::-1, :]
            _, flapping_axis_inertias = self._calculate_strip_inertias(
                wing=wing,
                num_spanwise_panels=num_spanwise_panels,
                mass_matrix=mass_matrix,
            )
            total += float(np.sum(flapping_axis_inertias))
        return total

    def _calculate_flap_fluid_torque(
        self,
        solver: CoupledUnsteadyRingVortexLatticeMethodSolver,
        wing_index: int,
    ) -> float:
        """Finds the aerodynamic torque on a Wing about its flap axis.

        The torque is taken about the Wing's rotation point and along the flap axis, and
        its sign is positive in the direction of increasing flap angle. It follows from
        the moment transport theorem: the sum of the Panels' moments about the first
        Airplane's CG, minus the cross product of the rotation point's position with the
        sum of the Panels' forces.

        The flap axis is the x axis of the Wing's geometry axes (after accounting for
        symmetry). For a mirror-only Wing, those axes are a reflection of the
        Airplane's, so the axis in the Airplane's geometry axes is reflected too, and a
        positive flap angle turns the Wing the opposite way about it.

        :param solver: The solver that has just solved the current time step.
        :param wing_index: The index of the Wing in the first Airplane's Wings.
        :return: The torque, in Newton meters.
        """
        wing = solver.current_airplanes[0].wings[wing_index]
        assert wing.panels is not None
        wing_movement = self._aeroelastic_movement.airplane_movements[0].wing_movements[
            wing_index
        ]

        panels = np.ravel(wing.panels)
        totalForces_GP1 = np.zeros(3, dtype=float)
        totalMoments_GP1_CgP1 = np.zeros(3, dtype=float)
        for panel in panels:
            assert panel.forces_GP1 is not None
            assert panel.moments_GP1_CgP1 is not None
            totalForces_GP1 += panel.forces_GP1
            totalMoments_GP1_CgP1 += panel.moments_GP1_CgP1

        # The rotation point sits at the rotation point offset in wing axes, relative to
        # the leading edge root point, however far the Wing has rotated.
        T_pas_Wn_Ler_to_G_Cg = wing.T_pas_Wn_Ler_to_G_Cg
        assert T_pas_Wn_Ler_to_G_Cg is not None
        rotationPoint_G_Cg = _transformations.apply_T_to_vectors(
            T_pas_Wn_Ler_to_G_Cg,
            np.array(wing_movement.rotationPointOffset_Gs_Ler, dtype=float),
            is_position=True,
        )

        torques_G = totalMoments_GP1_CgP1 - np.cross(
            rotationPoint_G_Cg, totalForces_GP1
        )

        flapAxis_G = np.array([1.0, 0.0, 0.0], dtype=float)
        sign = 1.0
        if wing.mirror_only:
            assert wing.symmetryNormal_G is not None
            normal_G = np.array(wing.symmetryNormal_G, dtype=float)
            flapAxis_G = flapAxis_G - 2.0 * np.dot(normal_G, flapAxis_G) * normal_G
            sign = -1.0

        return float(sign * np.dot(torques_G, flapAxis_G))

    def initialize_next_problem(
        self,
        solver: CoupledUnsteadyRingVortexLatticeMethodSolver,
        step: int,
    ) -> None:
        # There is no next time step to prepare after the final one.
        if step >= self.num_steps - 1:
            return

        # The flap angles of the next time step must be recorded on the movements before
        # the parent generates the next time step's Airplane from them.
        self._advance_flap_drives(solver, step)

        super().initialize_next_problem(solver, step)

    def _advance_flap_drives(
        self,
        solver: CoupledUnsteadyRingVortexLatticeMethodSolver,
        step: int,
    ) -> None:
        """Integrates every FlapDrive over the interval from the current time step to
        the next one, and records the results.

        :param solver: The solver that has just solved the current time step.
        :param step: The current time step index.
        :return: None
        """
        dt = self.delta_time
        next_step = step + 1
        start_time = step * dt
        end_time = next_step * dt

        for drive_position, flap_drive in enumerate(self._flap_drives):
            # Drive the flap without the startup transient's large aerodynamic loads for
            # the first step_discards steps.
            fluid_torque = 0.0
            if next_step > self.step_discards:
                fluid_torque = sum(
                    self._calculate_flap_fluid_torque(solver, wing_index)
                    for wing_index in flap_drive.wing_indices
                )

            wing_flap_inertia = self._flap_wing_flap_inertias[drive_position]
            position0 = self.flap_actuator_positions[drive_position][-1]
            velocity0 = self.flap_actuator_velocities[drive_position][-1]

            def evaluate(
                time: float,
                state: np.ndarray,
                flap_drive: flap_drive_mod.FlapDrive = flap_drive,
                fluid_torque: float = fluid_torque,
                wing_flap_inertia: float = wing_flap_inertia,
            ) -> tuple[
                flap_drive_mod.FlapDriveInputs, flap_drive_mod.FlapDriveAccelerations
            ]:
                return _evaluate_flap_drive(
                    flap_drive,
                    float(time),
                    float(state[0]),
                    float(state[1]),
                    fluid_torque,
                    wing_flap_inertia,
                )

            def derivative(time: float, state: np.ndarray) -> np.ndarray:
                _, accelerations = evaluate(time, state)
                return np.array(
                    [state[1], accelerations.actuator_acceleration], dtype=float
                )

            solution = solve_ivp(
                derivative,
                (start_time, end_time),
                np.array([position0, velocity0], dtype=float),
                rtol=_FLAP_ODE_RELATIVE_TOLERANCE,
                atol=_FLAP_ODE_ABSOLUTE_TOLERANCE,
                dense_output=True,
            )
            if not solution.success:
                raise RuntimeError(
                    f"Integrating flap_drives[{drive_position}] over the time step from "
                    f"{start_time} s to {end_time} s failed: {solution.message}"
                )
            # Sample the Wings' flap angular acceleration across the interval, for the
            # torsional spring's inertial forcing.
            sample_times = np.linspace(start_time, end_time, _FLAP_ACCELERATION_SAMPLES)
            dense_solution = solution.sol
            assert dense_solution is not None
            sample_accelerations = np.array(
                [
                    evaluate(sample_time, dense_solution(sample_time))[
                        1
                    ].angleSecondDerivativeXRad_Gs_to_Wn_ixyz
                    for sample_time in sample_times
                ],
                dtype=float,
            )

            end_inputs, end_accelerations = evaluate(end_time, solution.y[:, -1])

            angle_deg = float(np.rad2deg(end_inputs.angleXRad_Gs_to_Wn_ixyz))
            if abs(angle_deg) > 90.0:
                _logger.warning(
                    f"flap_drives[{drive_position}] drove its flap angle to "
                    f"{angle_deg:.2f} degrees at time step {next_step}, so it was "
                    "limited to the 90 degrees that a Wing allows."
                )
                angle_deg = float(np.clip(angle_deg, -90.0, 90.0))

            acceleration_function = _make_interpolated_function(
                sample_times, sample_accelerations
            )
            for wing_index in flap_drive.wing_indices:
                cast(
                    flap_driven_wing_movement_mod.FlapDrivenAeroelasticWingMovement,
                    self._aeroelastic_movement.airplane_movements[0].wing_movements[
                        wing_index
                    ],
                ).record_flap_step(next_step, angle_deg, acceleration_function)

            self.flap_actuator_positions[drive_position].append(
                end_inputs.actuator_position
            )
            self.flap_actuator_velocities[drive_position].append(
                end_inputs.actuator_velocity
            )
            self.flap_actuator_accelerations[drive_position].append(
                end_accelerations.actuator_acceleration
            )
            self.flap_actuator_forces[drive_position].append(
                end_accelerations.actuator_force
            )
            self.flap_fluid_torques[drive_position].append(fluid_torque)
            self.flap_transmission_torques[drive_position].append(
                end_accelerations.transmission_torque
            )
            self.flap_angles_rad[drive_position].append(
                end_inputs.angleXRad_Gs_to_Wn_ixyz
            )
            self.flap_angle_derivatives_rad[drive_position].append(
                end_inputs.angleDerivativeXRad_Gs_to_Wn_ixyz
            )
            self.flap_angle_second_derivatives_rad[drive_position].append(
                end_accelerations.angleSecondDerivativeXRad_Gs_to_Wn_ixyz
            )

    @staticmethod
    def _generate_inertial_moment_function(
        flapping_axis_inertia: float,
        wing_movement: aeroelastic_wing_movement_mod.AeroelasticWingMovement,
    ) -> Callable[[float], float]:
        """Generates the inertial moment function of a strip from the Wing's actual flap
        acceleration, when a FlapDrive has set one.

        For a Wing moved by a FlapDrivenAeroelasticWingMovement with a recorded
        acceleration, the inertial moment is the strip's flapping-axis inertia times the
        Wing's flap angular acceleration over the time step. Otherwise, the Wing's
        prescribed flap motion sets it, as in the parent class.

        :param flapping_axis_inertia: The strip's rotational inertia about the flapping
            axis (kg*m^2).
        :param wing_movement: The AeroelasticWingMovement whose flap motion is used.
        :return: A callable that accepts a time (s) and returns the inertial moment
            (N*m).
        """
        if isinstance(
            wing_movement,
            flap_driven_wing_movement_mod.FlapDrivenAeroelasticWingMovement,
        ):
            acceleration_function = wing_movement.angleSecondDerivativeXRad_function
            if acceleration_function is not None:
                return lambda time: flapping_axis_inertia * acceleration_function(time)

        return problems.AeroelasticUnsteadyProblem._generate_inertial_moment_function(
            flapping_axis_inertia, wing_movement
        )
