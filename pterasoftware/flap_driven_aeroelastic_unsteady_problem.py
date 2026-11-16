"""Contains the FlapDrivenAeroelasticUnsteadyProblem class.
ENTIRE SCRIPT VALIDATED CORRECT
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.integrate import solve_ivp

from . import (
    _core,
    _logging,
    _transformations,
)
from . import flap_drive as flap_drive_mod
from . import geometry
from . import operating_point as operating_point_mod
from . import problems
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
    operating_point: operating_point_mod.OperatingPoint | None,
) -> tuple[flap_drive_mod.FlapDriveInputs, flap_drive_mod.FlapDriveAccelerations]:
    """Evaluates a FlapDrive's equation of motion at one instant.

    VALIDATED CORRECT

    This is the one place a FlapDrive is evaluated, so every caller sees the same
    controller and dynamics: the integrator's right hand side and the acceleration
    sampling in _advance_flap_drives, the evaluation at the end of each time step, and
    the seeding of the histories at the first time step. It first runs the controller
    (inverse_dynamics_controller), which reads the reference and builds the
    FlapDriveInputs, and then the dynamics (single_wing_dynamics), which call the user's
    actuator function with those inputs and solve the equation of motion. It returns
    both, because the histories record values from each.

    :param flap_drive: The FlapDrive to evaluate.
    :param time: The simulation time (s).
    :param actuator_position: The actuator position, in the units of the drive's
        transmission.
    :param actuator_velocity: The actuator velocity, in the units of the drive's
        transmission per second.
    :param fluid_torque: The fluid torque on the drive's Wings about their flap axis
        (N*m). Only used in inverse_dynamics_controller if no other model is given
    :param wing_flap_inertia: The drive's Wings' combined inertia about the flap axis
        (kg*m^2).
    :param operating_point: The OperatingPoint of the time step the fluid torque comes
        from, which the controller passes to an aerodynamic model given as the drive's
        feedforward_aerodynamic, or None.
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
        operating_point=operating_point,
    )
    return inputs, flap_drive.single_wing_dynamics(inputs, fluid_torque=fluid_torque)


def _clear_movement_period_caches(
    movement: _core.CoreMovement,
) -> None:
    """Clears the cached periods of a movement and its AirplaneMovements.

    VALIDATED CORRECT

    The movement classes compute their periods once and cache them. A
    FlapDrivenAeroelasticUnsteadyProblem sets its FlapDrivenAeroelasticWingMovements'
    flap periods after the movement exists, and the movement may have read and cached
    its periods before then, so the caches are cleared and recomputed on their next
    read. This reaches into the private caches of the core movement classes, which have
    no public way to invalidate them, so a test checks that these names still exist.

    :param movement: The movement whose cached periods to clear, an AeroelasticMovement
        or a FreeFlightAeroelasticMovement.
    :return: None
    """
    # Each AirplaneMovement caches the periods of its WingMovements (all_periods) and
    # their longest one (max_period). Setting them to None makes the next read recompute
    # them, which now includes the flap periods the problem just set.
    for airplane_movement in movement.airplane_movements:
        airplane_movement._all_periods = None
        airplane_movement._max_period = None

    # The movement caches the values derived from those: the least common multiple of
    # all the periods (lcm_period, the window of the cycle averaged results), the
    # longest and shortest periods, and whether anything moves at all (static). These
    # are what the solver and the output read.
    movement._lcm_period = None
    movement._max_period = None
    movement._min_period = None
    movement._static = None


def _make_interpolated_function(
    times: np.ndarray, values: np.ndarray
) -> Callable[[float], float]:
    """Returns a function that reads sampled values by linear interpolation in time.

    VALIDATED CORRECT

    :param times: A (N,) ndarray of floats holding the sample times, in increasing
        order.
    :param values: A (N,) ndarray of floats holding the value at each sample time.
    :return: A callable that accepts a time and returns the interpolated value.
    """
    # Copy the samples, so that the function keeps its own values even if the caller
    # changes its arrays later. The function is kept on a movement for a whole time
    # step, and a deep copy of the movement shares it.
    sample_times = np.array(times, dtype=float)
    sample_values = np.array(values, dtype=float)

    # np.interp holds the first and last values constant outside the sampled range, so a
    # time a little before or after the time step, as an integrator may ask for, gets
    # the nearest end's value instead of an extrapolation.
    def interpolated(time: float) -> float:
        return float(np.interp(time, sample_times, sample_values))

    return interpolated


# These are the slots that hold a problem's FlapDrives and their histories. Both
# FlapDrivenAeroelasticUnsteadyProblem and
# FreeFlightFlapDrivenAeroelasticUnsteadyProblem declare them, because the
# _FlapDriveCoupling mixin they share can't: its other parent already declares slots,
# and Python doesn't allow a class whose two parents both declare slots. Defining them
# once here keeps the two problems' slots identical.
_FLAP_DRIVE_SLOTS = (
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

# For type checking, the mixin is treated as an AeroelasticUnsteadyProblem, so that its
# methods can use the attributes the problems it is mixed into provide (such as
# step_discards, delta_time, and _steady_problems) without the type checker reporting
# them as missing. At run time it derives from object, so that it adds no slots and no
# layout of its own, and can sit next to either parent problem.
if TYPE_CHECKING:
    _FlapDriveCouplingBase = problems.AeroelasticUnsteadyProblem
else:
    _FlapDriveCouplingBase = object


class _FlapDriveCoupling(_FlapDriveCouplingBase):
    """A mixin that adds FlapDrives to an AeroelasticUnsteadyProblem.

    VALIDATED CORRECT

    **What a mixin is:**

    A mixin is a class that isn't used on its own. It only supplies methods, and a
    concrete class lists it as its first parent, next to the class it really derives
    from, to gain those methods. This one supplies everything a problem needs to drive
    its Wings' flap angles with FlapDrives instead of a prescribed oscillation. It is
    private, so users never build one. They build one of the two problems that use it.

    **Why it exists:**

    Two problems need the same flap drive logic, but they derive from different
    parents:

    - FlapDrivenAeroelasticUnsteadyProblem(_FlapDriveCoupling,
      AeroelasticUnsteadyProblem), for a prescribed flight condition.

    - FreeFlightFlapDrivenAeroelasticUnsteadyProblem(_FlapDriveCoupling,
      FreeFlightAeroelasticUnsteadyProblem), for free flight, where MuJoCo integrates the
      vehicle's rigid body dynamics.

    Both parents derive from AeroelasticUnsteadyProblem, so the flap drive logic works
    the same on top of either. Putting it in this mixin means it is written, tested, and
    fixed once, instead of being copied into both problems and drifting apart. The
    alternative of the free flight problem deriving from both
    FlapDrivenAeroelasticUnsteadyProblem and FreeFlightAeroelasticUnsteadyProblem isn't
    possible, because a class can't have two parents that both declare slots (Python
    raises a "layout conflict" TypeError), and both of those declare them.

    **What it provides:**

    - _prepare_flap_drives, the setup that must happen before the parent's __init__:
      it validates the FlapDrives and the Wings they move, checks each reference, warns
      about a reference with no period, records each drive's starting flap angle on its
      Wings' movements (so the parent builds the first time step's Airplane at that
      angle), hands each reference's period to the movements, and clears their cached
      periods.

    - _initialize_flap_drives, the setup that must happen after the parent's __init__,
      once the first time step's Airplane exists: it chooses each drive's Wing inertia
      and seeds the flap histories.

    - initialize_next_problem, which the solver calls after each time step. It advances
      the FlapDrives (_advance_flap_drives) and then calls super(), the parent's
      initialize_next_problem, which integrates the torsional springs and builds the
      next Airplane. In the free flight problem, the parent also runs the rigid body's
      strongly coupled sub-iteration.

    - _advance_flap_drives, _calculate_flap_fluid_torque, and
      _calculate_wings_flap_inertia, the per time step work, and the flap_drives
      property.

    - _generate_inertial_moment_function, which overrides the parent's so that the
      torsional springs are forced by the Wings' actual flap acceleration.

    **How the parts fit together:**

    The mixin has no slots and no __init__. A concrete problem built on it must:

    - Declare __slots__ = _FLAP_DRIVE_SLOTS, which hold the FlapDrives and their
      histories, because the mixin can't declare them (see above).

    - In its __init__, call _prepare_flap_drives, then its parent's __init__ (through
      super().__init__, which skips the mixin, since the mixin has none), then
      _initialize_flap_drives.

    Because the mixin is listed first, Python's method resolution order checks it before
    the other parent. So its initialize_next_problem and
    _generate_inertial_moment_function replace the parent's, and its super() calls
    reach the parent's. For example, the free flight problem's order is
    FreeFlightFlapDrivenAeroelasticUnsteadyProblem, _FlapDriveCoupling,
    FreeFlightAeroelasticUnsteadyProblem, AeroelasticUnsteadyProblem, and so on.

    For type checking only, the mixin is treated as deriving from
    AeroelasticUnsteadyProblem and having the flap drive slots, so that its methods may
    use the parent's attributes (such as step_discards and delta_time) and assign the
    flap histories. At run time it derives from object and has no slots.

    See FlapDrivenAeroelasticUnsteadyProblem for what happens each time step.
    """

    # At run time the mixin has no slots, because the concrete classes declare the flap
    # drive slots. Type checking is told it has them, so that the methods below may
    # assign them.
    if TYPE_CHECKING:
        __slots__ = _FLAP_DRIVE_SLOTS
    else:
        __slots__ = ()

    # These are the flap drive slots' types, for type checking.
    _flap_drives: tuple[flap_drive_mod.FlapDrive, ...]
    _flap_wing_flap_inertias: tuple[float, ...]
    flap_actuator_positions: list[list[float]]
    flap_actuator_velocities: list[list[float]]
    flap_actuator_accelerations: list[list[float]]
    flap_actuator_forces: list[list[float]]
    flap_fluid_torques: list[list[float]]
    flap_transmission_torques: list[list[float]]
    flap_angles_rad: list[list[float]]
    flap_angle_derivatives_rad: list[list[float]]
    flap_angle_second_derivatives_rad: list[list[float]]

    @staticmethod
    def _prepare_flap_drives(
        movement: _core.CoreMovement,
        flap_drives: Sequence[flap_drive_mod.FlapDrive],
    ) -> tuple[tuple[flap_drive_mod.FlapDrive, ...], list[tuple[float, float]]]:
        """Validates the FlapDrives and prepares the movement for them, before the
        parent's __init__ runs.

        VALIDATED CORRECT

        It checks the FlapDrives and the Wings they move, clears any flap state left on
        the movements, checks that each reference covers the simulation, warns about a
        reference with no period, records each drive's starting flap angle and flap
        period on its Wings' movements, and clears the movement's period caches. It must
        run before the parent's __init__, which builds the first time step's Airplane
        from the recorded angles and reads the movement's periods.

        :param movement: The problem's movement. Its first AirplaneMovement's
            WingMovements are the ones the FlapDrives' wing_indices index.
        :param flap_drives: The FlapDrives, as given to the problem.
        :return: A tuple of the validated FlapDrives and each one's initial actuator
            position and velocity.
        """
        # Check that flap_drives is a non empty sequence of FlapDrives. A str is a
        # sequence too, so it is rejected explicitly. The drives are stored as a tuple,
        # so that the problem's list can't be changed through the caller's.
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

        # Check which Wings each FlapDrive moves. A FlapDrive only knows its Wings'
        # indices, so they are checked here, against the first Airplane: each index must
        # exist, no Wing can belong to two FlapDrives (it can only have one flap angle),
        # and each driven Wing must be moved by a FlapDrivenAeroelasticWingMovement,
        # which is what holds the recorded flap angles and accelerations. Wings that no
        # FlapDrive moves can have any movement.
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
            # The reference must last until the final time step, because the controller
            # reads it at every instant, and a reference that ended early would leave
            # the drive with no desired motion.
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

            # Convert the initial actuator position to the flap angle through the
            # transmission. The transmission works in radians, and the geometry in
            # degrees.
            initial_angle_deg = float(
                np.rad2deg(flap_drive.transmission.evaluate_angle(initial_position))
            )

            # Record that angle as each of the drive's Wings' flap angle at time step 0,
            # so the parent builds the first Airplane at it. There is no flap
            # acceleration to record yet, since no time step has been integrated, so it
            # is None. Also give each Wing's movement the reference's period, so the
            # movement's periods, and so the cycle averaging, include the flap.
            for wing_index in flap_drive.wing_indices:
                drive_wing_movement = cast(
                    flap_driven_wing_movement_mod.FlapDrivenAeroelasticWingMovement,
                    first_wing_movements[wing_index],
                )
                drive_wing_movement.record_flap_step(0, initial_angle_deg, None)
                drive_wing_movement.set_flap_period(reference.period)

        # The movement may have read and cached its periods before the flap periods were
        # set, for example while it was constructed, so clear those caches. The parent's
        # __init__ then reads the periods fresh.
        _clear_movement_period_caches(movement)

        return validated_flap_drives, initial_states

    def _initialize_flap_drives(
        self,
        validated_flap_drives: tuple[flap_drive_mod.FlapDrive, ...],
        initial_states: list[tuple[float, float]],
    ) -> None:
        """Chooses each FlapDrive's Wing inertia and seeds its histories with its state
        at the first time step, after the parent's __init__ has built that time step.

        VALIDATED CORRECT

        :param validated_flap_drives: The FlapDrives that _prepare_flap_drives returned.
        :param initial_states: Each FlapDrive's initial actuator position and velocity,
            which _prepare_flap_drives returned.
        :return: None
        """
        self._flap_drives = validated_flap_drives

        # The parent's __init__ has built the first time step, so its Airplane (whose
        # Wings give the inertia estimate) and its OperatingPoint (for an aerodynamic
        # model given as feedforward_aerodynamic) are available now.
        initial_airplane = self._steady_problems[0].airplanes[0]
        initial_operating_point = self._steady_problems[0].operating_point
        wing_flap_inertias: list[float] = []

        # Each history is a list with one inner list per FlapDrive, in the order of
        # flap_drives. Each inner list gets one entry per time step: the first here, and
        # one more each time _advance_flap_drives advances the drives.
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
            # the fluid has acted on it, so the fluid torque passed here is 0.0. The
            # evaluation gives the actuator force and accelerations at that state, as
            # the later entries record them. _advance_flap_drives starts its first
            # integration from these actuator position and velocity entries.
            initial_inputs, initial_accelerations = _evaluate_flap_drive(
                flap_drive,
                0.0,
                initial_position,
                initial_velocity,
                0.0,
                wing_flap_inertia,
                initial_operating_point,
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

        # Keep the chosen inertias, one per FlapDrive, for every later evaluation.
        self._flap_wing_flap_inertias = tuple(wing_flap_inertias)

    # --- Immutable: read only properties ---
    @property
    def flap_drives(self) -> tuple[flap_drive_mod.FlapDrive, ...]:
        """The FlapDrives that flap this problem's Wings.

        VALIDATED CORRECT

        :return: A tuple of the FlapDrives, in the order given to the problem, which is
            also the order of the flap histories' inner lists.
        """
        return self._flap_drives

    def _calculate_wings_flap_inertia(self, wings: list[geometry.wing.Wing]) -> float:
        """Estimates the combined inertia of Wings about their flap axis.

        VALIDATED CORRECT

        It sums the strip flapping-axis inertias that the torsional spring model uses,
        so the drive and the structure agree on the Wings' mass distribution.

        :param wings: The Wings to include.
        :return: The combined inertia, in kilogram square meters.
        """
        # This is only used when a FlapDrive has no wing_flap_inertia of its own. It
        # inherits the strip model's known error (it is three times too large), see
        # docs/FLAP_DRIVE_TODO.md.
        total = 0.0
        for wing in wings:
            num_spanwise_panels = wing.num_spanwise_panels
            assert num_spanwise_panels is not None

            # Spread the Wing's mass over its Panels by area, at the problem's
            # wing_density, the same way the torsional spring model does.
            mass_matrix = self._calculate_mass_matrix(wing)
            # A mirror-meshed Wing's Panel grid runs tip to root spanwise, but the strip
            # inertias run root to tip.
            if wing.mirror_only:
                mass_matrix = mass_matrix[:, ::-1, :]

            # Each strip's inertia about the flap axis includes the parallel axis term
            # for its distance from the axis, so their sum is the Wing's inertia about
            # the flap axis. The torsional inertias, the first return value, aren't
            # needed here.
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

        VALIDATED CORRECT
        
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
        # Read the Wing as the solver just solved it, at the current time step's flap
        # angle and deformation, with its Panels' loads. Its movement holds the rotation
        # point offset, which locates the shoulder.
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

        # A mirror-only Wing (the reflected half of a symmetric Wing) has its geometry
        # axes reflected about the symmetry plane. So its flap axis, expressed in the
        # Airplane's geometry axes, is the x axis reflected about that plane (v - 2 * (n
        # . v) * n for the plane's unit normal n), and, because a reflection reverses
        # the sense of rotation, a positive flap angle turns it the opposite way about
        # that axis, which the sign accounts for. For the usual symmetry plane (normal
        # along y), the axis stays the x axis and only the sign flips, so the left and
        # right Wings' torques have the same sign when they flap symmetrically.
        if wing.mirror_only:
            assert wing.symmetryNormal_G is not None
            normal_G = np.array(wing.symmetryNormal_G, dtype=float)
            flapAxis_G = flapAxis_G - 2.0 * np.dot(normal_G, flapAxis_G) * normal_G
            sign = -1.0

        # Project the torque onto the flap axis, so that a positive value pushes the
        # flap angle up.
        return float(sign * np.dot(torques_G, flapAxis_G))

    def initialize_next_problem(
        self,
        solver: CoupledUnsteadyRingVortexLatticeMethodSolver,
        step: int,
    ) -> None:
        """Builds the next time step's SteadyProblem, after first advancing the
        FlapDrives to it.

        VALIDATED CORRECT

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

        VALIDATED CORRECT

        Each FlapDrive is independent of the others (each has its own actuator state and
        its own Wings), so they are integrated one after another. For each one, this
        finds the fluid torque on its Wings, integrates its actuator's state across the
        interval with that torque held constant, and then writes three things: the flap
        angle at the next time step and the flap acceleration over the interval, on its
        Wings' movements, where the parent class reads them, and one new entry in each
        of this problem's flap histories.

        This function is called in the FlapDrivenAeroelasticUnsteadyProblem
        and FreeFlightFlapDrivenAeroelasticUnsteadyProblem class 
        which uses the same coreproblem class

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

        # Get the OperatingPoint of the time step just solved, for any aerodynamic model
        # given as a FlapDrive's feedforward_aerodynamic. Like the fluid torque, it is
        # held constant over the interval. In free flight it is the vehicle's state from
        # the rigid body dynamics at the time step just solved, because the drives
        # advance before the body's sub-iteration for the next time step starts.
        operating_point = solver.current_operating_point

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
            #VALIDATED CORRECT
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
                    operating_point,
                )

            # The state is (actuator position, actuator velocity), so its time
            # derivative is (actuator velocity, actuator acceleration). The acceleration
            # comes from the full controller and equation of motion at this instant, so
            # the feedback, the feedforward, and the motor's limits all act continuously
            # within the interval, not once per time step.
            #VALIDATED CORRECT
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
            #VALIDATED CORRECT
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
            #VALIDATED CORRECT
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

        VALIDATED CORRECT

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


class FlapDrivenAeroelasticUnsteadyProblem(
    _FlapDriveCoupling, problems.AeroelasticUnsteadyProblem
):
    """A subclass of AeroelasticUnsteadyProblem whose Wings' flap angles come from
    FlapDrives instead of from the movement's prescribed oscillation.

    VALIDATED CORRECT

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

    __slots__ = _FLAP_DRIVE_SLOTS

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

        # Validate the FlapDrives and record their starting flap angles and periods on
        # the movements, before the parent builds the first time step's Airplane.
        validated_flap_drives, initial_states = self._prepare_flap_drives(
            movement, flap_drives
        )

        super().__init__(
            movement=movement,
            wing_density=wing_density,
            spring_constant_rad=spring_constant_rad,
            damping_constant_rad=damping_constant_rad,
            step_discards=step_discards,
        )

        # Seed the FlapDrives' histories from the first time step.
        self._initialize_flap_drives(validated_flap_drives, initial_states)
