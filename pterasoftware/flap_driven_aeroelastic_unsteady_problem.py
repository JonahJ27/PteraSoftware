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
    # The UVLM's fluid torque plays two separate roles. The controller gets it as a
    # measurement, which it uses for its feedforward only when the drive's
    # feedforward_aerodynamic is True (a model passed there replaces it). The dynamics
    # get it as the torque that actually acts on the Wings. It reaches the dynamics only
    # through the second call, so the controller's estimate can't change it.
    inputs = flap_drive.inverse_dynamics_controller(
        time,
        actuator_position,
        actuator_velocity,
        measured_fluid_torque=fluid_torque,
        computed_wing_flap_inertia=wing_flap_inertia,
    )
    return inputs, flap_drive.single_wing_dynamics(inputs, fluid_torque=fluid_torque)


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

    **The order of one time step:**

    The solver solves time step s, then calls initialize_next_problem(solver, s) to
    build time step s + 1. That method first calls _advance_flap_drives, which, for each
    FlapDrive, sums the fluid torque on its Wings from time step s's solution,
    integrates the actuator's state from time s * delta_time to (s + 1) * delta_time
    with that torque held constant, and records the flap angle at s + 1 and the flap
    acceleration over the interval on its Wings' movements. It then calls the parent's
    initialize_next_problem, which integrates the torsional springs over the same
    interval (their inertial forcing reads the recorded flap acceleration through
    _generate_inertial_moment_function) and generates time step s + 1's Airplane (its
    Wings' flap angles come from the recorded flap angles through the movements'
    override). The solver then solves time step s + 1 on that Airplane.

    **Loose coupling and its one time step lag:**

    The fluid and the FlapDrives are coupled loosely (explicitly): the UVLM solves each
    time step once, and the FlapDrives see its result only on the next interval. Over
    the interval from s to s + 1, the actuator, the controller, and the motor's limits
    are integrated continuously, but the fluid torque is frozen at its value at s, so
    the fluid's response to the flap's motion during the interval only arrives one time
    step later. This acts like a delay of one time step in the aerodynamic load, which
    is a phase error of 360.0 * delta_time / period degrees per wingbeat. It is also the
    source of a possible instability: the part of the fluid torque that resists the
    Wings' acceleration (the added mass) is applied a step late, so for light Wings with
    a large added mass relative to their own inertia, the flap can oscillate and
    diverge. A smaller delta_time shrinks both effects. Removing the lag needs the UVLM
    to be solved more than once per time step, which this class doesn't do yet. The
    planned fix, a fixed-point sub-iteration of each time step on the fluid torque, is
    described in docs/FLAP_DRIVE.md under "Planned: Strongly Coupling the Flap".

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

        # Sum the Wing's Panels' forces, and their moments about the first Airplane's
        # CG, which the solver has just computed for the current time step. These are
        # the total aerodynamic force and moment on the Wing, before the moment is moved
        # to the shoulder below.
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

        # Move the moment from the CG to the rotation point (the shoulder) with the
        # moment transport theorem, M_p = M_CG - r_p x F, where r_p is the rotation
        # point's position relative to the CG.
        torques_G = totalMoments_GP1_CgP1 - np.cross(
            rotationPoint_G_Cg, totalForces_GP1
        )

        # Keep only the component about the flap axis, which is the torque that turns
        # the Wing about its shoulder and loads the FlapDrive. The other two components
        # are carried by the shoulder's bearing.
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
        """Builds the next time step's SteadyProblem, after first advancing the
        FlapDrives to it.

        The solver calls this once per time step, after it has solved time step step, so
        the Panels of the solver's current Airplane hold that time step's loads. See the
        class docstring for the order of the work.

        :param solver: The solver that has just solved time step step.
        :param step: The index of the time step that was just solved.
        :return: None
        """
        # There is no next time step to prepare after the final one.
        if step >= self.num_steps - 1:
            return

        # Advance the FlapDrives first. The parent's initialize_next_problem reads what
        # this records: it builds the next Airplane at the recorded flap angles, and its
        # torsional springs use the recorded flap acceleration as their inertial
        # forcing. Calling them in the other order would build the next Airplane at the
        # base Wing's flap angle.
        self._advance_flap_drives(solver, step)

        # Integrate the torsional springs over the same interval and generate the next
        # time step's Airplane, unchanged from AeroelasticUnsteadyProblem.
        super().initialize_next_problem(solver, step)

    def _advance_flap_drives(
        self,
        solver: CoupledUnsteadyRingVortexLatticeMethodSolver,
        step: int,
    ) -> None:
        """Integrates every FlapDrive over the interval from the current time step to
        the next one, and records the results.

        Each FlapDrive is independent of the others (each has its own actuator state and
        its own Wings), so they are integrated one after another. For each one, this
        finds the fluid torque on its Wings, integrates its actuator's state across the
        interval with that torque held constant, and then writes three things: the flap
        angle at the next time step and the flap acceleration over the interval, on its
        Wings' movements, where the parent class reads them, and one new entry in each
        of this problem's flap histories.

        :param solver: The solver that has just solved the current time step.
        :param step: The current time step index.
        :return: None
        """
        # The interval being integrated runs from the time step the solver just solved
        # to the one being built. It is the same interval the parent's torsional springs
        # integrate over, so the flap acceleration recorded here covers exactly the
        # times at which the springs read it.
        dt = self.delta_time
        next_step = step + 1
        start_time = step * dt
        end_time = next_step * dt

        for drive_position, flap_drive in enumerate(self._flap_drives):
            # Find the fluid torque that acts on the drive's Wings over the interval. It
            # is the torque from the solution of the time step just solved, summed over
            # the drive's Wings, because they share one actuator. It is held constant
            # over the whole interval: this is the one time step lag of the loose
            # coupling that the class docstring describes. For the first step_discards
            # time steps it is 0.0 instead, so the UVLM's startup transient, whose loads
            # are large and unphysical while the wake forms, doesn't kick the actuator.
            fluid_torque = 0.0
            if next_step > self.step_discards:
                fluid_torque = sum(
                    self._calculate_flap_fluid_torque(solver, wing_index)
                    for wing_index in flap_drive.wing_indices
                )

            # The integration starts from the actuator's state at the end of the
            # previous interval, which is the last entry of the histories. Nothing else
            # stores the actuator's state between time steps: the FlapDrive itself is
            # stateless.
            wing_flap_inertia = self._flap_wing_flap_inertias[drive_position]
            position0 = self.flap_actuator_positions[drive_position][-1]
            velocity0 = self.flap_actuator_velocities[drive_position][-1]

            # This evaluates the drive at one instant: the controller builds its
            # FlapDriveInputs (the actual state, the reference, and the feedforward),
            # the user's actuator function returns its force, and the equation of motion
            # gives the accelerations. The drive, the torque, and the inertia are bound
            # as default arguments so that each loop iteration's function keeps its own
            # values instead of reading the loop variables after they change.
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

            # The state is (actuator position, actuator velocity), so its time
            # derivative is (actuator velocity, actuator acceleration). The acceleration
            # comes from the full controller and equation of motion at this instant, so
            # the feedback, the feedforward, and the motor's limits all act continuously
            # within the interval, not once per time step.
            def derivative(time: float, state: np.ndarray) -> np.ndarray:
                _, accelerations = evaluate(time, state)
                return np.array(
                    [state[1], accelerations.actuator_acceleration], dtype=float
                )

            # Integrate across the interval with an adaptive step size. The integrator
            # takes as many internal steps as the actuator's dynamics need, which can be
            # many when the motor saturates or the linkage nears a dead center. The
            # dense output is a continuous solution across the interval, which the
            # sampling below reads.
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
            # torsional spring's inertial forcing. The springs need the acceleration at
            # whatever times their own integrator chooses, so it is sampled at evenly
            # spaced times from the dense solution and later read by linear
            # interpolation. This also keeps the springs from calling the user's
            # actuator function, which may log or keep its own records.
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

            # Evaluate the drive once more at the end of the interval, at the
            # integrator's final state, which is the drive's state at the next time
            # step. This gives the flap angle the next Airplane is built at, and every
            # value the histories record for that time step.
            end_inputs, end_accelerations = evaluate(end_time, solution.y[:, -1])

            # A Wing only allows flap angles in [-90.0, 90.0] degrees, so an angle
            # beyond that is limited for the geometry, with a warning. The actuator's
            # own state isn't changed, so the histories still show where the drive
            # actually went.
            angle_deg = float(np.rad2deg(end_inputs.angleXRad_Gs_to_Wn_ixyz))
            if abs(angle_deg) > 90.0:
                _logger.warning(
                    f"flap_drives[{drive_position}] drove its flap angle to "
                    f"{angle_deg:.2f} degrees at time step {next_step}, so it was "
                    "limited to the 90 degrees that a Wing allows."
                )
                angle_deg = float(np.clip(angle_deg, -90.0, 90.0))

            # Hand the results to the parent class through the drive's Wings' movements.
            # When the parent generates the next Airplane, each movement replaces its
            # Wing's flap angle with the recorded one, and when the parent integrates
            # the torsional springs, _generate_inertial_moment_function reads the
            # recorded acceleration. All of the drive's Wings get the same values,
            # because they share one actuator.
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

            # Record the drive at the next time step. The actuator position and velocity
            # entries are also the next interval's initial state. The fluid torque entry
            # is the torque that acted over this interval, not one computed at the next
            # time step, which the solver hasn't solved yet.
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
        # The parent's torsional spring model calls this once per strip per time step.
        # For a driven Wing, the acceleration is the one _advance_flap_drives just
        # recorded for this interval, so the strip's inertial load follows the flap's
        # actual motion, including any lag or saturation of the actuator.
        if isinstance(
            wing_movement,
            flap_driven_wing_movement_mod.FlapDrivenAeroelasticWingMovement,
        ):
            acceleration_function = wing_movement.angleSecondDerivativeXRad_function
            if acceleration_function is not None:
                return lambda time: flapping_axis_inertia * acceleration_function(time)

        # A Wing that no FlapDrive moves keeps the parent's behavior, which uses its
        # prescribed flap motion.
        return problems.AeroelasticUnsteadyProblem._generate_inertial_moment_function(
            flapping_axis_inertia, wing_movement
        )
