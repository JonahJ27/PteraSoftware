"""Contains the FlapDrive class, the Transmission interface that it drives Wings
through, the FlapTransmission and FlapReference classes, the records that FlapDrive
consumes and returns, and helpers for building transmissions and trajectories.

ENTIRE SCRIPT VALIDATED CORRECT

FlapDrive is independent of the powertrain. It only needs the flap angle, its Jacobian,
and its Hessian as functions of the actuator position, and optionally a way to convert a
flap angle trajectory into an actuator trajectory. Any object that provides those
satisfies the Transmission interface: a FlapTransmission built from functions, a
crank_linkage.CrankLinkage, or a class of your own.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, NamedTuple, Protocol, runtime_checkable

import numpy as np

from . import _parameter_validation

# This is the default central difference step for FlapTransmission's
# from_forward_kinematics constructor. It is in the same units as the actuator position.
_DEFAULT_FINITE_DIFFERENCE_STEP = 1e-4

# This is the default spacing (s) of the samples of a FlapReference's table and of a
# generated flap trajectory.
_DEFAULT_REFERENCE_TIME_STEP = 1e-4

# This is the default number of samples over one actuator period that
# FlapTransmission.angle_range uses to find the extremes of the flap stroke.
_DEFAULT_ANGLE_RANGE_SAMPLES = 10000

# These are the flap trajectory styles that generate_flap_trajectory supports.
_FLAP_TRAJECTORY_STYLES = ("sine", "square")


def _make_sample_times(duration: float | int, time_step: float | int) -> np.ndarray:
    """Returns sample times from 0.0 to a duration, inclusive.

    VALIDATED CORRECT

    :param duration: The last time, in seconds. It must be a positive number.
    :param time_step: The spacing of the samples, in seconds. It must be a positive
        number.
    :return: A (N,) ndarray of floats holding the sample times. The last time is the
        duration rounded down to a multiple of the time step.
    """
    duration = _parameter_validation.number_in_range_return_float(
        duration, "duration", 0.0, False
    )
    time_step = _parameter_validation.number_in_range_return_float(
        time_step, "time_step", 0.0, False
    )
    # Count the whole time steps that fit in the duration, like the blade element theory
    # code's 0:dt:T. The small tolerance keeps a duration that is a multiple of the time
    # step from losing its last sample to round off.
    num_intervals = int(np.floor(duration / time_step + 1e-9))
    if num_intervals < 1:
        raise ValueError("duration must be at least one time_step long.")
    return np.arange(num_intervals + 1, dtype=float) * time_step


def generate_flap_trajectory(
    style: str,
    flapping_frequency: float | int,
    wing_beat_amplitude_rad: float | int,
    dihedral_rad: float | int,
    min_angle_rad: float | int,
    max_angle_rad: float | int,
    num_cycles: int = 1,
    time_step: float | int = _DEFAULT_REFERENCE_TIME_STEP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates a desired flap angle trajectory and its analytical time derivatives.

    VALIDATED CORRECT

    It follows the flap trajectory generator of the blade element theory motor limit
    simulation. One cycle is built from time 0.0 to the period, offset by the dihedral,
    clipped to the stroke limits, and then repeated for the requested number of cycles.

    **Styles:**

    "sine" is half the wing beat amplitude times the sine of the flapping angular
    frequency times time, so it starts on an upstroke. "square" has a constant velocity
    of the wing beat amplitude divided by the half stroke time in each half stroke,
    starting on a downstroke, so it covers the full wing beat amplitude, and it is
    centered by subtracting its mean over the cycle. Its acceleration is 0.0 except at
    the reversals, where it is infinite and is not represented. Unlike the blade element
    theory code's square style, which covers half of the wing beat amplitude, both
    styles here cover the full amplitude.

    Wherever the clipping to the stroke limits changes the flap angle, the flap angle's
    derivatives are set to 0.0.

    :param style: The trajectory's style. It must be "sine" or "square".
    :param flapping_frequency: The flapping frequency, in hertz. It must be a positive
        number (int or float).
    :param wing_beat_amplitude_rad: The peak to peak stroke amplitude, in radians. It
        must be a positive number (int or float).
    :param dihedral_rad: The flap angle about which the stroke is centered, in radians.
        It must be a number (int or float).
    :param min_angle_rad: The smallest flap angle that the transmission can reach, in
        radians. It must be a number (int or float).
    :param max_angle_rad: The largest flap angle that the transmission can reach, in
        radians. It must be a number (int or float) greater than min_angle_rad.
    :param num_cycles: The number of cycles to generate. It must be a positive int. The
        default is 1.
    :param time_step: The spacing of the samples, in seconds. It must be a positive
        number (int or float). The default is 1e-4.
    :return: A tuple of four (N,) ndarrays of floats holding the sample times (s), the
        flap angle (rad), its time derivative (rad/s), and its second time derivative
        (rad/s^2).
    """
    # This follows the blade element theory code's generateFlapTrajCustom. Its inputs
    # are flapping_freq, wing_beat_amplitude, dihedral, style, max_wing_ang, and
    # min_wing_ang there. Its "quasi_square" style isn't ported.
    if style not in _FLAP_TRAJECTORY_STYLES:
        raise ValueError(
            f"style must be one of {_FLAP_TRAJECTORY_STYLES}, but it is {style}."
        )
    flapping_frequency = _parameter_validation.number_in_range_return_float(
        flapping_frequency, "flapping_frequency", 0.0, False
    )
    amplitude = _parameter_validation.number_in_range_return_float(
        wing_beat_amplitude_rad, "wing_beat_amplitude_rad", 0.0, False
    )
    dihedral = _parameter_validation.number_in_range_return_float(
        dihedral_rad, "dihedral_rad"
    )
    min_angle = _parameter_validation.number_in_range_return_float(
        min_angle_rad, "min_angle_rad"
    )
    max_angle = _parameter_validation.number_in_range_return_float(
        max_angle_rad, "max_angle_rad"
    )
    if not max_angle > min_angle:
        raise ValueError("max_angle_rad must be greater than min_angle_rad.")
    num_cycles = _parameter_validation.int_in_range_return_int(
        num_cycles, "num_cycles", 1, True
    )

    # Set up the timing of one cycle (T, half_stroke_time, and t = 0:dt:T there).
    period = 1.0 / flapping_frequency
    half_stroke_time = period / 2.0
    times = _make_sample_times(period, time_step)

    if style == "sine":
        # This is a standard sine trajectory with half the wing beat amplitude, so its
        # peak to peak size is the wing beat amplitude (omega_flap there). Unlike the
        # blade element theory code, which finds phi_ddot by differencing phi_dot, the
        # acceleration here is the sine's analytical second derivative.
        omega_rad = 2.0 * np.pi * flapping_frequency
        angles = amplitude / 2.0 * np.sin(omega_rad * times)
        rates = omega_rad * amplitude / 2.0 * np.cos(omega_rad * times)
        accelerations = -(omega_rad**2) * amplitude / 2.0 * np.sin(omega_rad * times)
    else:
        # This is a square waveform: a constant velocity on the downstroke and the
        # opposite one on the upstroke. Its acceleration is 0.0 except at the reversals,
        # where it is infinite and is left out. Each half stroke travels the full wing
        # beat amplitude, so the rate is the amplitude divided by the half stroke time.
        # The blade element theory code's generateFlapTrajCustom divides by two more
        # (half_stroke_amplitude = wing_beat_amplitude/2 there), which covers only half
        # of the requested amplitude.
        rate = amplitude / half_stroke_time
        rates = np.where(times <= half_stroke_time, -rate, rate)
        accelerations = np.zeros(times.shape, dtype=float)
        angles = np.where(
            times <= half_stroke_time,
            -rate * times,
            -rate * half_stroke_time + rate * (times - half_stroke_time),
        )

        # Center the stroke around zero.
        angles = angles - np.mean(angles)

    # Apply the dihedral offset first, since the limits are absolute angles.
    angles = angles + dihedral

    # Clip the flap angle to the limits, and find where the clipping changed it
    # (phi_clipped and clipped_idx there).
    clipped = (angles < min_angle) | (angles > max_angle)
    angles = np.clip(angles, min_angle, max_angle)

    # Keep the clipped samples consistent by giving them zero velocity and acceleration.
    rates = np.where(clipped, 0.0, rates)
    accelerations = np.where(clipped, 0.0, accelerations)

    # Repeat the cycle, dropping each cycle's last sample except the final one's, since
    # it coincides with the next cycle's first sample. This follows the blade element
    # theory code's runSingleSimulation, except that it offsets each cycle by the period
    # rather than by the period plus one time step.
    cycle_times = []
    cycle_angles = []
    cycle_rates = []
    cycle_accelerations = []
    for cycle in range(num_cycles):
        end = None if cycle == num_cycles - 1 else -1
        cycle_times.append(times[:end] + cycle * period)
        cycle_angles.append(angles[:end])
        cycle_rates.append(rates[:end])
        cycle_accelerations.append(accelerations[:end])
    return (
        np.concatenate(cycle_times),
        np.concatenate(cycle_angles),
        np.concatenate(cycle_rates),
        np.concatenate(cycle_accelerations),
    )


def _identity_angle(actuator_position: float) -> float:
    """Returns the actuator position as the flap angle.

    VALIDATED CORRECT

    :param actuator_position: The actuator position (rad).
    :return: The same value, as the flap angle (rad).
    """
    return actuator_position


def _identity_jacobian(actuator_position: float) -> float:
    """Returns the Jacobian of a direct drive transmission.

    VALIDATED CORRECT

    :param actuator_position: The actuator position (rad).
    :return: 1.0
    """
    return 1.0


def _identity_hessian(actuator_position: float) -> float:
    """Returns the Hessian of a direct drive transmission.

    VALIDATED CORRECT

    :param actuator_position: The actuator position (rad).
    :return: 0.0
    """
    return 0.0


def _identity_trajectory(
    times: np.ndarray,
    angles_rad: np.ndarray,
    angle_derivatives_rad: np.ndarray,
    angle_second_derivatives_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns a flap angle trajectory unchanged, as the actuator trajectory of a direct
    drive.

    VALIDATED CORRECT

    :param times: The times of the samples (s). It is unused.
    :param angles_rad: The flap angles (rad).
    :param angle_derivatives_rad: The flap angles' time derivatives (rad/s).
    :param angle_second_derivatives_rad: The flap angles' second time derivatives
        (rad/s^2).
    :return: Copies of the three trajectory inputs.
    """
    return (
        np.array(angles_rad, dtype=float),
        np.array(angle_derivatives_rad, dtype=float),
        np.array(angle_second_derivatives_rad, dtype=float),
    )


def _call_scalar_function(
    function: Callable[..., float], name: str, *arguments: Any
) -> float:
    """Calls a user supplied function and validates that it returned a finite number.

    VALIDATED CORRECT

    :param function: The function to call.
    :param name: The name of the function, used in error messages.
    :param arguments: The arguments to pass to the function. They are scalars, except
        for the actuator function, which receives a FlapDriveInputs.
    :return: The function's return value, converted to a float.
    """
    # Convert the result to a float, so that a numpy scalar or a 0D array is accepted,
    # and reject a nan or infinite value before it reaches the equation of motion.
    value = float(function(*arguments))
    if not np.isfinite(value):
        raise ValueError(
            f"{name} returned {value}, but it must return a finite number."
        )
    return value


def numerical_trajectory_function(
    inverse_position_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """VALIDATED CORRECT.

    Builds a transmission's inverse kinematics from a function that gives only the
    actuator position, finding the velocity and acceleration by differentiating it in
    time.

    This is the blade element theory code's convertWingTrajToMotorNumerical approach.
    Use it when you can write the actuator position for a flap angle, but not its
    derivatives. Pass the result as a FlapTransmission's trajectory_function, or use
    FlapTransmission.from_positions, which also finds the Jacobian and Hessian
    numerically.

    The velocity and acceleration come from numpy.gradient applied twice, against the
    actual sample times, so they are second order central differences inside the
    trajectory and one sided differences at its ends. Their accuracy depends on the
    sample spacing, and a jump in the actuator position becomes a spike in both.

    :param inverse_position_function: A callable that accepts two (N,) ndarrays of
        floats, the flap angles (rad) and their time derivatives (rad/s), and returns a
        (N,) ndarray of floats holding the actuator position that produces each flap
        angle. The time derivatives are there so that it can pick the solution for the
        stroke direction when a flap angle has more than one actuator position, as a
        crank's does. It is responsible for making the positions continuous, for example
        by unwrapping a rotary actuator's angle, because a jump between samples would be
        differentiated into a spike.
    :return: A trajectory function with the signature that a FlapTransmission's
        trajectory_function needs.
    """
    if not callable(inverse_position_function):
        raise TypeError("inverse_position_function must be a callable.")

    def trajectory_function(
        times: np.ndarray,
        flap_angles_rad: np.ndarray,
        flap_angle_derivatives_rad: np.ndarray,
        flap_angle_second_derivatives_rad: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Get the actuator positions from the user's inverse kinematics, then
        # differentiate them in time twice (motor_pos_vec_des, then gradient for
        # theta_dot and theta_ddot, in the blade element theory code). The flap angle's
        # second derivative isn't needed.
        positions = np.asarray(
            inverse_position_function(flap_angles_rad, flap_angle_derivatives_rad),
            dtype=float,
        )
        if positions.shape != np.shape(times):
            raise ValueError(
                "inverse_position_function must return a 1D array with one actuator "
                "position per sample."
            )
        velocities = np.asarray(np.gradient(positions, times), dtype=float)
        accelerations = np.asarray(np.gradient(velocities, times), dtype=float)
        return positions, velocities, accelerations

    return trajectory_function


@runtime_checkable
class Transmission(Protocol):
    """The interface that a FlapDrive uses to drive its Wings, whatever the powertrain.

    VALIDATED CORRECT

    **This is an abstract interface, not a class to instantiate.**

    Its methods have no bodies (each is just "..."), and Transmission() is never
    called. It only lists the members that a powertrain must have, so it plays the role
    of a C++ abstract base class with pure virtual methods, or a Java interface. Unlike
    those, it is a typing.Protocol, so it is satisfied structurally: a class conforms by
    having these members with these signatures, without inheriting from Transmission or
    even importing it. That is what keeps FlapDrive independent of the powertrain.

    It is checked two ways. mypy checks the full signatures wherever a powertrain is
    passed as a Transmission. At runtime, @runtime_checkable lets FlapDrive and
    FlapReference.from_flap_trajectory call isinstance(transmission, Transmission),
    but that only confirms that the members exist, not their signatures or return
    values.

    A transmission maps the actuator position q (a motor shaft angle, a muscle length,
    or the flap angle itself) to the Wings' flap angle phi, the x component of the
    Wing's angles_Gs_to_Wn_ixyz (angles describing the orientation of the wing axes
    relative to the geometry axes, after accounting for symmetry, using an intrinsic
    xy'z" sequence). A FlapDrive needs only these members, so any class that has them
    can be passed as a FlapDrive's transmission, without subclassing anything:

    evaluate_angle(q) returns phi (rad). evaluate_jacobian(q) returns J = d(phi)/dq.
    evaluate_hessian(q) returns H = d^2(phi)/dq^2. These are the forward kinematics and
    its derivatives, which the FlapDrive's equation of motion uses at every evaluation.

    convert_trajectory(times, phi, phi', phi'') returns the actuator's position,
    velocity, and acceleration that produce a flap angle trajectory (the inverse
    kinematics). It is used once, before the simulation, to build a FlapReference. It is
    required, so that a wing trajectory can always be turned into an actuator
    trajectory, whatever the powertrain.

    FlapTransmission and crank_linkage.CrankLinkage both satisfy this interface.
    """

    def evaluate_angle(self, actuator_position: float | int) -> float: ...

    def evaluate_jacobian(self, actuator_position: float | int) -> float: ...

    def evaluate_hessian(self, actuator_position: float | int) -> float: ...

    def convert_trajectory(
        self,
        times: np.ndarray,
        flap_angles_rad: np.ndarray,
        flap_angle_derivatives_rad: np.ndarray,
        flap_angle_second_derivatives_rad: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


class FlapTransmission:
    """A class used to build a Transmission from functions.

    VALIDATED CORRECT

    **A convenience for plugging in your own transmission without writing a class:**

    FlapDrive accepts any object that conforms to the Transmission protocol. The usual
    way to describe a powertrain is a class of its own that implements the protocol's
    members, as crank_linkage.CrankLinkage does. FlapTransmission is a shortcut around
    that: hand it your transmission's flap angle, Jacobian, and Hessian as functions of
    the actuator position, and optionally its inverse kinematics, and it wraps them in
    an object that conforms to the protocol, which a FlapDrive then uses like any other
    transmission. Your functions can come from anywhere, such as a formula you want to
    try, a fit to measured data, or the methods of another class. It also has presets
    for the two simplest cases, a direct drive (identity) and a gearbox (geared).

    FlapDrive never requires it, and it adds nothing to a powertrain that already has
    its own class. Because it only holds functions, it doesn't keep your transmission's
    parameters as attributes, and a saved problem stores its functions as inert
    placeholders rather than reloadable code. For a transmission you want to keep,
    inspect, or save, write a class that conforms to the protocol instead.

    The actuator has one generalized coordinate, the actuator position. It is a shaft
    angle (rad) for a rotary motor, a length (m) for a muscle or linear actuator, or the
    flap angle itself (rad) for a direct drive. The transmission maps that position to
    the flap angle, which is the x component of the Wing's angles_Gs_to_Wn_ixyz (angles
    describing the orientation of the wing axes relative to the geometry axes, after
    accounting for symmetry, using an intrinsic xy'z" sequence), and supplies the first
    and second derivatives of that map. Those derivatives are the Jacobian and Hessian
    of the transmission.

    A linkage is a nonlinear transmission, so its angle, Jacobian, and Hessian are all
    functions of the actuator position. A gearbox is linear, so its Jacobian is constant
    and its Hessian is zero. A direct drive or a muscle that acts on the flap angle
    directly has a Jacobian of 1.0 and a Hessian of 0.0.

    The flap angle is absolute, meaning it includes the base Wing's flap angle, not just
    the change from it. The actuator position is unbounded, so a transmission that
    repeats every revolution must wrap its own argument.

    **Constructors:**

    There are five ways to build a FlapTransmission. The initialization method,
    FlapTransmission(angle_function, jacobian_function, hessian_function,
    trajectory_function), takes the functions directly. identity, geared,
    from_forward_kinematics, and from_positions are alternate constructors: each builds
    those functions for a common case and passes them to the initialization method, so
    each returns a new FlapTransmission. They are called on the class, as in
    FlapTransmission.geared(15.0), because there is no FlapTransmission yet to call them
    on.

    They play the role of a C++ class's overloaded constructors. Python can't overload a
    method by its argument types, so each alternate constructor is a classmethod with
    its own name instead, like the C++ named constructor idiom of static factory
    methods. A classmethod receives the class as its first argument, cls, and cls(...)
    calls the initialization method.
    """

    __slots__ = (
        "_angle_function",
        "_jacobian_function",
        "_hessian_function",
        "_trajectory_function",
    )

    def __init__(
        self,
        angle_function: Callable[[float], float],
        jacobian_function: Callable[[float], float],
        hessian_function: Callable[[float], float],
        trajectory_function: Callable[
            [np.ndarray, np.ndarray, np.ndarray, np.ndarray],
            tuple[np.ndarray, np.ndarray, np.ndarray],
        ],
    ) -> None:
        """The initialization method.

        :param angle_function: A callable that accepts the actuator position and returns
            the flap angle. The flap angle is the x component of the Wing's
            angles_Gs_to_Wn_ixyz, and its units are in radians. The actuator position
            can be in any units, as long as they match the other functions.
        :param jacobian_function: A callable that accepts the actuator position and
            returns the first derivative of the flap angle with respect to it. The units
            are radians per actuator position unit.
        :param hessian_function: A callable that accepts the actuator position and
            returns the second derivative of the flap angle with respect to it. The
            units are radians per actuator position unit squared.
        :param trajectory_function: A callable that converts a flap angle trajectory
            into the actuator trajectory that produces it (the inverse kinematics). It
            is required, so that a wing trajectory can always be converted into an
            actuator trajectory. It accepts four (N,) ndarrays of floats holding the
            sample times (s), which are strictly increasing, followed by the flap angle
            (rad), its time derivative (rad/s), and its second time derivative (rad/s^2)
            at those times. It returns three (N,) ndarrays of floats holding the
            actuator position, velocity, and acceleration. A linkage has more than one
            actuator position for each flap angle, and it is singular where the flap
            angle turns around, so this function is responsible for choosing the branch,
            for unwrapping the actuator position so that it is continuous, and for
            avoiding division by a zero Jacobian.
        :return: None
        """
        # The forward kinematics and its two derivatives drive the equation of motion,
        # and the inverse kinematics converts a wing trajectory into an actuator
        # trajectory. All four are required.
        for name, function in (
            ("angle_function", angle_function),
            ("jacobian_function", jacobian_function),
            ("hessian_function", hessian_function),
            ("trajectory_function", trajectory_function),
        ):
            if not callable(function):
                raise TypeError(f"{name} must be a callable.")

        self._angle_function = angle_function
        self._jacobian_function = jacobian_function
        self._hessian_function = hessian_function
        self._trajectory_function = trajectory_function

    @classmethod
    def identity(cls) -> FlapTransmission:
        """Returns a direct drive transmission, where the actuator position is the flap
        angle.

        VALIDATED CORRECT

        This is an alternate constructor (see the class docstring). It has a Jacobian of
        1.0 and a Hessian of 0.0, and it is the right choice for muscle models and other
        actuators that act on the flap angle directly.

        :return: A FlapTransmission with a flap angle equal to the actuator position.
        """
        # The actuator moves the flap angle directly, so the blade element theory code's
        # J_motor is 1.0, H_motor is 0.0, and the inverse kinematics returns the
        # trajectory unchanged.
        return cls(
            _identity_angle,
            _identity_jacobian,
            _identity_hessian,
            _identity_trajectory,
        )

    @classmethod
    def geared(cls, gear_ratio: float | int) -> FlapTransmission:
        """Returns a gearbox transmission with a constant reduction.

        VALIDATED CORRECT

        This is an alternate constructor (see the class docstring). The flap angle is
        the actuator position divided by the gear ratio. It has a constant Jacobian of
        1.0 divided by the gear ratio and a Hessian of 0.0.

        :param gear_ratio: The reduction from the actuator position to the flap angle.
            It must be a nonzero number (int or float) and will be converted internally
            to a float. A negative value reverses the direction of the flap. It is
            unitless.
        :return: A FlapTransmission with a flap angle equal to the actuator position
            divided by the gear ratio.
        """
        gear_ratio = _parameter_validation.number_in_range_return_float(
            gear_ratio, "gear_ratio"
        )
        if gear_ratio == 0.0:
            raise ValueError("gear_ratio can't be zero.")

        # A gearbox divides the actuator's motion by the gear ratio, so its Jacobian is
        # the constant 1.0 / gear_ratio and its Hessian is 0.0, since a gearbox's
        # mechanical advantage doesn't change.
        def angle_function(actuator_position: float) -> float:
            return actuator_position / gear_ratio

        def jacobian_function(actuator_position: float) -> float:
            return 1.0 / gear_ratio

        # The inverse kinematics multiplies the flap trajectory by the gear ratio, like
        # the blade element theory code's motor_pos_curr = red_pos_curr *
        # gearbox_reduction.
        def trajectory_function(
            times: np.ndarray,
            angles_rad: np.ndarray,
            angle_derivatives_rad: np.ndarray,
            angle_second_derivatives_rad: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            return (
                angles_rad * gear_ratio,
                angle_derivatives_rad * gear_ratio,
                angle_second_derivatives_rad * gear_ratio,
            )

        return cls(
            angle_function, jacobian_function, _identity_hessian, trajectory_function
        )

    @classmethod
    def from_forward_kinematics(
        cls,
        angle_function: Callable[[float], float],
        trajectory_function: Callable[
            [np.ndarray, np.ndarray, np.ndarray, np.ndarray],
            tuple[np.ndarray, np.ndarray, np.ndarray],
        ],
        step_size: float | int = _DEFAULT_FINITE_DIFFERENCE_STEP,
    ) -> FlapTransmission:
        """Returns a transmission whose Jacobian and Hessian are found by finite
        differences of the flap angle function.

        VALIDATED CORRECT

        This is an alternate constructor (see the class docstring). It is a convenience
        for linkages that have a forward kinematics function but no analytical
        derivatives. Central differences are used, so the error scales with the square
        of the step size.

        :param angle_function: A callable that accepts the actuator position and returns
            the flap angle. The flap angle is the x component of the Wing's
            angles_Gs_to_Wn_ixyz, and its units are in radians.
        :param trajectory_function: The inverse kinematics that the FlapTransmission's
            initialization method describes. It is required.
        :param step_size: The central difference step, in the same units as the actuator
            position. It must be a positive number (int or float) and will be converted
            internally to a float. Too small a value amplifies round off error, and too
            large a value amplifies truncation error. The default is 1e-4.
        :return: A FlapTransmission with the given flap angle function and numerically
            differentiated Jacobian and Hessian.
        """
        if not callable(angle_function):
            raise TypeError("angle_function must be a callable.")
        step_size = _parameter_validation.number_in_range_return_float(
            step_size, "step_size", 0.0, False
        )

        # The first derivative is the central difference of the forward kinematics,
        # whose error scales with the square of the step size.
        def jacobian_function(actuator_position: float) -> float:
            return (
                angle_function(actuator_position + step_size)
                - angle_function(actuator_position - step_size)
            ) / (2.0 * step_size)

        # The second derivative is the central second difference, which divides by the
        # square of the step size, so it is far more sensitive to round off.
        def hessian_function(actuator_position: float) -> float:
            return (
                angle_function(actuator_position + step_size)
                - 2.0 * angle_function(actuator_position)
                + angle_function(actuator_position - step_size)
            ) / step_size**2

        return cls(
            angle_function, jacobian_function, hessian_function, trajectory_function
        )

    @classmethod
    def from_positions(
        cls,
        angle_function: Callable[[float], float],
        inverse_position_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
        step_size: float | int = _DEFAULT_FINITE_DIFFERENCE_STEP,
    ) -> FlapTransmission:
        """Returns a transmission built only from its position level kinematics in both
        directions, with every derivative found by finite differences.

        VALIDATED CORRECT

        This is an alternate constructor (see the class docstring). Give it the forward
        kinematics, the flap angle as a function of the actuator position, and the
        inverse kinematics, the actuator position as a function of the flap angle. The
        Jacobian and Hessian come from central differences of the forward kinematics in
        the actuator position, as in from_forward_kinematics, and the actuator's
        velocity and acceleration in a converted trajectory come from differences of the
        inverse kinematics in time, as numerical_trajectory_function describes.

        :param angle_function: A callable that accepts the actuator position and returns
            the flap angle. The flap angle is the x component of the Wing's
            angles_Gs_to_Wn_ixyz, and its units are in radians.
        :param inverse_position_function: A callable that accepts the flap angles and
            their time derivatives as two (N,) ndarrays and returns the actuator
            positions, as numerical_trajectory_function describes.
        :param step_size: The central difference step for the Jacobian and Hessian, as
            from_forward_kinematics describes. The default is 1e-4.
        :return: A FlapTransmission whose derivatives are all found numerically.
        """
        return cls.from_forward_kinematics(
            angle_function,
            numerical_trajectory_function(inverse_position_function),
            step_size,
        )

    def convert_trajectory(
        self,
        times: np.ndarray,
        flap_angles_rad: np.ndarray,
        flap_angle_derivatives_rad: np.ndarray,
        flap_angle_second_derivatives_rad: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Converts a flap angle trajectory into the actuator trajectory that produces
        it.

        VALIDATED CORRECT

        :param times: A (N,) ndarray of floats holding the times of the samples. They
            must be finite and strictly increasing. The units are in seconds.
        :param flap_angles_rad: A (N,) ndarray of floats holding the flap angles (the x
            component of the Wing's angles_Gs_to_Wn_ixyz) at those times. The units are
            in radians.
        :param flap_angle_derivatives_rad: A (N,) ndarray of floats holding the flap
            angles' time derivatives at the same times. The units are in radians per
            second.
        :param flap_angle_second_derivatives_rad: A (N,) ndarray of floats holding the
            flap angles' second time derivatives at the same times. The units are in
            radians per second squared.
        :return: A tuple of three (N,) ndarrays of floats holding the actuator position,
            velocity, and acceleration at the same times, in the units of this
            FlapTransmission.
        """
        # Check that the four arrays line up sample for sample, and that the times
        # increase, before handing them to the trajectory function.
        inputs = []
        for name, value in (
            ("times", times),
            ("flap_angles_rad", flap_angles_rad),
            ("flap_angle_derivatives_rad", flap_angle_derivatives_rad),
            ("flap_angle_second_derivatives_rad", flap_angle_second_derivatives_rad),
        ):
            array = np.asarray(value, dtype=float)
            if array.ndim != 1 or array.shape != np.shape(times):
                raise ValueError(f"{name} must be a 1D array with the same length.")
            inputs.append(array)
        if not np.isfinite(inputs[0]).all() or not (np.diff(inputs[0]) > 0.0).all():
            raise ValueError("times must be finite and strictly increasing.")

        converted = self._trajectory_function(
            inputs[0], inputs[1], inputs[2], inputs[3]
        )

        # Check what the trajectory function returned, since it may be a user's.
        if len(converted) != 3:
            raise ValueError("trajectory_function must return three ndarrays.")
        outputs = [np.array(item, dtype=float) for item in converted]
        for output in outputs:
            if output.shape != inputs[0].shape or not np.isfinite(output).all():
                raise ValueError(
                    "trajectory_function must return three finite ndarrays with the "
                    "same shape as its inputs."
                )
        return outputs[0], outputs[1], outputs[2]

    def angle_range(
        self,
        actuator_period: float | int,
        num_samples: int = _DEFAULT_ANGLE_RANGE_SAMPLES,
    ) -> tuple[float, float]:
        """Returns the smallest and largest flap angles over one period of the actuator.

        VALIDATED CORRECT

        The extremes are found by sampling the forward kinematics over the period.

        :param actuator_period: The actuator travel over which the flap angle repeats,
            in the units of the FlapTransmission. For a rotary actuator that turns a
            periodic linkage through a gearbox, it is 2.0 * pi times the gear ratio. It
            must be a positive number (int or float).
        :param num_samples: The number of samples over the period. It must be an int of
            at least 2. The default is 10000.
        :return: A tuple of the smallest and largest flap angles, in radians.
        """
        actuator_period = _parameter_validation.number_in_range_return_float(
            actuator_period, "actuator_period", 0.0, False
        )
        num_samples = _parameter_validation.int_in_range_return_int(
            num_samples, "num_samples", 2, True
        )
        # Sample the forward kinematics over one period and keep the extremes, like the
        # blade element theory code's loop over red_angs = linspace(0, 2*pi, 10000) in
        # runSingleSimulation. Its wing_beat_amplitude is the difference of the two, and
        # its dihedral is their average.
        angles = [
            self.evaluate_angle(position)
            for position in np.linspace(0.0, actuator_period, num_samples)
        ]
        return float(min(angles)), float(max(angles))

    def evaluate_angle(self, actuator_position: float | int) -> float:
        """Returns the flap angle at an actuator position.

        :param actuator_position: The actuator position. Its units must match the
            functions that define this FlapTransmission.
        :return: The flap angle (the x component of the Wing's angles_Gs_to_Wn_ixyz), in
            radians.
        """
        return _call_scalar_function(
            self._angle_function, "angle_function", actuator_position
        )

    def evaluate_jacobian(self, actuator_position: float | int) -> float:
        """Returns the first derivative of the flap angle with respect to the actuator
        position.

        :param actuator_position: The actuator position. Its units must match the
            functions that define this FlapTransmission.
        :return: The Jacobian, in radians per actuator position unit.
        """
        return _call_scalar_function(
            self._jacobian_function, "jacobian_function", actuator_position
        )

    def evaluate_hessian(self, actuator_position: float | int) -> float:
        """Returns the second derivative of the flap angle with respect to the actuator
        position.

        :param actuator_position: The actuator position. Its units must match the
            functions that define this FlapTransmission.
        :return: The Hessian, in radians per actuator position unit squared.
        """
        return _call_scalar_function(
            self._hessian_function, "hessian_function", actuator_position
        )


class FlapReference:
    """A class used to contain the desired actuator trajectory that a FlapDrive tracks.

    VALIDATED CORRECT

    The reference is a table of the actuator position, velocity, and acceleration at
    increasing times, in the actuator's own coordinate, whatever the powertrain. That
    coordinate is whatever the transmission takes as its actuator position: a motor's
    shaft angle (rad) behind a gearbox, crank linkage, scotch yoke, or any other
    mechanism, a muscle's or linear actuator's length (m), or the flap angle itself
    (rad) for a direct drive. The table knows nothing about the mechanism. It only holds
    the motion that the actuator should follow.

    There are two ways to build it. One is to write the actuator's motion directly, as
    constant_velocity does for a motor turning at a constant speed. The other is to
    start from a desired flap angle trajectory and convert it once, before the
    simulation, through the transmission's inverse kinematics (its convert_trajectory),
    as from_flap_trajectory does. All of the powertrain specific work happens in that
    conversion, so the controller can work entirely in actuator coordinates while the
    simulation runs. For a nonlinear transmission, such as a crank linkage, the
    conversion is also where the inverse kinematics picks a solution for each stroke
    direction, unwraps the actuator angle so that it is continuous, and handles the
    turnarounds where the linkage is singular, so the controller never has to invert a
    singular linkage while the simulation runs. For a linear one, such as a direct drive
    or a gearbox, the conversion just scales the trajectory.

    The table can be read two ways. evaluate interpolates it linearly in time.
    evaluate_at_position instead returns the velocity and acceleration at the table
    entry whose position is nearest a given actuator position. Which one a controller
    uses is not the reference's decision: FlapDrive's use_position_lookup chooses it for
    the feedforward force. The reference only reports, through positions_are_monotonic,
    whether a position lookup is possible. It is only meaningful when the position table
    is monotonic, as it is for an actuator that keeps moving one way, such as a
    continuously rotating motor driving a crank or a scotch yoke. It is ambiguous for an
    actuator that moves back and forth, such as a direct drive, a geared drive that
    reverses, or a muscle, because one position then appears in both directions of the
    stroke, so evaluate_at_position raises a ValueError for such a table.

    **The only definition of the flap trajectory:**

    A FlapReference is the only place a FlapDrive's trajectory is defined. A FlapDrive
    can't exist without one, and a FlapDrivenAeroelasticWingMovement can't prescribe any
    flap motion, so there is no second source of the trajectory to disagree with this
    one.

    **Period and cycle averaged results:**

    The solver averages loads over the last full period of all the motions, and it takes
    the periods from the movement. A FlapDrivenAeroelasticWingMovement prescribes no
    flap motion, so the FlapDrivenAeroelasticUnsteadyProblem gives each FlapDrive's
    Wings the period of its reference instead. A reference whose flap trajectory repeats
    with a fixed period, such as a sine of constant frequency, sets period to that
    value, and the cycle averaged results then cover whole wingbeats. A reference whose
    flap trajectory doesn't repeat with a fixed period, such as one whose flapping
    frequency changes, sets period to None. Its cycle averaged results are then
    inaccurate, because no fixed window covers whole wingbeats, and if nothing else in
    the movement moves, the simulation is treated as static: it reports the final time
    step's loads and labels its geometry as static. The problem logs a warning saying
    so.
    """

    __slots__ = (
        "_times",
        "_positions",
        "_velocities",
        "_accelerations",
        "_positions_are_monotonic",
        "_period",
    )

    def __init__(
        self,
        times: np.ndarray | Sequence[float],
        positions: np.ndarray | Sequence[float],
        velocities: np.ndarray | Sequence[float],
        accelerations: np.ndarray | Sequence[float],
        *,
        period: float | int | None,
    ) -> None:
        """The initialization method.

        :param times: An array-like object of floats with shape (N,) holding the times
            at which the table is sampled. Can be a tuple, list, or ndarray. It must
            have at least two values, and they must be finite and strictly increasing.
            The units are in seconds.
        :param positions: An array-like object of floats with shape (N,) holding the
            desired actuator position at each time. Can be a tuple, list, or ndarray.
            The units are those of the transmission.
        :param velocities: An array-like object of floats with shape (N,) holding the
            desired actuator velocity at each time. Can be a tuple, list, or ndarray.
            The units are those of the transmission per second.
        :param accelerations: An array-like object of floats with shape (N,) holding the
            desired actuator acceleration at each time. Can be a tuple, list, or
            ndarray. The units are those of the transmission per second squared.
        :param period: The period (s) of the flap trajectory that this reference
            produces, meaning one wingbeat, or None if the flap trajectory doesn't
            repeat with a fixed period. If not None, it must be a positive number (int
            or float) and will be converted internally to a float. It sets the window
            over which the solver averages loads, so it must be the flap's period, not
            the actuator's: for a motor driving a crank linkage through a gearbox, one
            wingbeat is one turn of the crank, which is gear_ratio turns of the motor.
            If None, the cycle averaged results are inaccurate, as the class docstring
            describes, and the FlapDrivenAeroelasticUnsteadyProblem logs a warning. It
            is a required keyword argument with no default, so that every reference
            states whether it is periodic.
        :return: None
        """
        # The table holds the blade element theory code's motor_pos_vec_des,
        # motor_velo_vec_des, and motor_acc_vec_des against time_vec. Copy each column
        # and check that the columns line up and the times increase.
        arrays = []
        for name, value in (
            ("times", times),
            ("positions", positions),
            ("velocities", velocities),
            ("accelerations", accelerations),
        ):
            array = np.array(value, dtype=float)
            if array.ndim != 1:
                raise ValueError(f"{name} must be a 1D array-like.")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} can't contain nan, inf, or -inf.")
            arrays.append(array)
        if arrays[0].size < 2:
            raise ValueError("times must have at least two values.")
        for name, array in zip(
            ("positions", "velocities", "accelerations"), arrays[1:]
        ):
            if array.shape != arrays[0].shape:
                raise ValueError(f"{name} must have the same length as times.")
        if not (np.diff(arrays[0]) > 0.0).all():
            raise ValueError("times must be strictly increasing.")

        # Record whether the positions are monotonic. A position lookup finds the
        # nearest table entry to an actuator position, which is only unambiguous if each
        # position appears once. That is the case for a crank that turns one way, but
        # not for an actuator that reverses.
        position_steps = np.diff(arrays[1])
        positions_are_monotonic = bool(
            (position_steps >= 0.0).all() or (position_steps <= 0.0).all()
        )

        # Make the table read only, so that it can't be changed after construction.
        for array in arrays:
            array.flags.writeable = False
        self._times = arrays[0]
        self._positions = arrays[1]
        self._velocities = arrays[2]
        self._accelerations = arrays[3]
        self._positions_are_monotonic = positions_are_monotonic

        if period is not None:
            period = _parameter_validation.number_in_range_return_float(
                period, "period", 0.0, False
            )
        self._period = period

    @classmethod
    def from_flap_trajectory(
        cls,
        transmission: Transmission,
        times: np.ndarray | Sequence[float],
        flap_angles_rad: np.ndarray | Sequence[float],
        flap_angle_derivatives_rad: np.ndarray | Sequence[float],
        flap_angle_second_derivatives_rad: np.ndarray | Sequence[float],
        *,
        period: float | int | None,
    ) -> FlapReference:
        """Returns the reference that produces a flap angle trajectory.

        :param transmission: The Transmission to convert the trajectory through, such as
            a FlapTransmission or a crank_linkage.CrankLinkage.
        :param times: An array-like object of floats with shape (N,) holding the times
            of the trajectory's samples, as FlapReference's initialization method
            requires. The units are in seconds.
        :param flap_angles_rad: An array-like object of floats with shape (N,) holding
            the flap angle (the x component of the Wing's angles_Gs_to_Wn_ixyz) at each
            time. The units are in radians.
        :param flap_angle_derivatives_rad: An array-like object of floats with shape
            (N,) holding the flap angle's time derivative at each time. The units are in
            radians per second.
        :param flap_angle_second_derivatives_rad: An array-like object of floats with
            shape (N,) holding the flap angle's second time derivative at each time. The
            units are in radians per second squared.
        :param period: The flap trajectory's period (s), or None if it doesn't repeat
            with a fixed period. See the initialization method. For a trajectory from
            generate_flap_trajectory, it is 1.0 / flapping_frequency.
        :return: A FlapReference holding the actuator trajectory that produces the flap
            angle trajectory.
        """
        if not isinstance(transmission, Transmission):
            raise TypeError("transmission must satisfy the Transmission interface.")

        # Convert the flap trajectory into the actuator trajectory through the
        # transmission's inverse kinematics, the blade element theory code's
        # convertWingTrajToMotorAnalytical step. The initialization method then checks
        # what it returned.
        positions, velocities, accelerations = transmission.convert_trajectory(
            np.array(times, dtype=float),
            np.array(flap_angles_rad, dtype=float),
            np.array(flap_angle_derivatives_rad, dtype=float),
            np.array(flap_angle_second_derivatives_rad, dtype=float),
        )
        return cls(times, positions, velocities, accelerations, period=period)

    @property
    def times(self) -> np.ndarray:
        return self._times

    @property
    def positions(self) -> np.ndarray:
        return self._positions

    @property
    def velocities(self) -> np.ndarray:
        return self._velocities

    @property
    def accelerations(self) -> np.ndarray:
        return self._accelerations

    @property
    def period(self) -> float | None:
        """The period of the flap trajectory that this reference produces.

        :return: The period (s), or None if the flap trajectory doesn't repeat with a
            fixed period.
        """
        return self._period

    @property
    def positions_are_monotonic(self) -> bool:
        """Whether the position table only ever increases or only ever decreases, which
        is what evaluate_at_position needs.

        :return: True if the positions are monotonic, and False otherwise.
        """
        return self._positions_are_monotonic

    def evaluate(self, time: float | int) -> tuple[float, float, float]:
        """Returns the desired actuator state at a time by linear interpolation.

        :param time: The time to evaluate at, in seconds. A time outside the table holds
            the nearest end value.
        :return: A tuple of the desired actuator position, velocity, and acceleration,
            in the units of the transmission.
        """
        # Interpolate each column linearly in time, like the blade element theory code's
        # griddedInterpolant objects motor_pos_interp, motor_velo_interp, and
        # motor_acc_interp.
        return (
            float(np.interp(time, self._times, self._positions)),
            float(np.interp(time, self._times, self._velocities)),
            float(np.interp(time, self._times, self._accelerations)),
        )

    def evaluate_at_position(self, position: float | int) -> tuple[float, float]:
        """Returns the desired actuator velocity and acceleration at the table entry
        whose position is nearest an actuator position.

        :param position: The actuator position, in the units of the transmission.
        :return: A tuple of the desired actuator velocity and acceleration at the
            nearest table entry, in the units of the transmission.
        """
        if not self._positions_are_monotonic:
            raise ValueError(
                "This FlapReference's positions are not monotonic, so it can't look up "
                "velocity and acceleration by position, because a position that "
                "appears in both directions of the stroke has no single velocity or "
                "acceleration."
            )
        # Take the entry whose position is nearest the actuator's, like the blade
        # element theory code's [~, idx_closest] = min(abs(motor_pos_vec_des -
        # motor_pos)) when use_closest_pos is true.
        index = int(np.argmin(np.abs(self._positions - position)))
        return float(self._velocities[index]), float(self._accelerations[index])

    @classmethod
    def constant_velocity(
        cls,
        start_position: float | int,
        velocity: float | int,
        duration: float | int,
        time_step: float | int = _DEFAULT_REFERENCE_TIME_STEP,
        *,
        period: float | int | None,
    ) -> FlapReference:
        """Returns the reference of an actuator that moves at a constant velocity.

        For a rotary motor this is a constant shaft speed, and the flap trajectory it
        produces is whatever the transmission's forward kinematics gives. It needs no
        inverse kinematics.

        :param start_position: The actuator position at time 0.0, in the units of the
            transmission. It must be a number (int or float).
        :param velocity: The actuator velocity, in the units of the transmission per
            second. It must be a number (int or float).
        :param duration: The time the reference covers, in seconds. It must be a
            positive number (int or float).
        :param time_step: The spacing of the reference's samples, in seconds. It must be
            a positive number (int or float). The default is 1e-4.
        :param period: The period (s) of the flap trajectory that the transmission
            produces from this motion, or None if it doesn't repeat with a fixed period.
            See the initialization method. For a motor driving a crank linkage through a
            gearbox of ratio N, it is 2.0 * pi * N / velocity.
        :return: A FlapReference with the actuator moving at a constant velocity.
        """
        start_position = _parameter_validation.number_in_range_return_float(
            start_position, "start_position"
        )
        velocity = _parameter_validation.number_in_range_return_float(
            velocity, "velocity"
        )
        # The motor turns at a constant speed, like the blade element theory code's
        # use_faav_traj path: motor_pos_des = t * motor_velo_des + motor_starting_pos,
        # with motor_accel_des = 0.0.
        times = _make_sample_times(duration, time_step)
        return cls(
            times,
            start_position + velocity * times,
            np.full(times.shape, velocity, dtype=float),
            np.zeros(times.shape, dtype=float),
            period=period,
        )


class FlapDriveInputs(NamedTuple):
    """The state that a FlapDrive passes to its actuator function each time it is
    evaluated.

    VALIDATED CORRECT

    A record is used so that fields can be added later without breaking existing
    actuator functions. Every field is a float.

    time is the simulation time (s). actuator_position and actuator_velocity are the
    actuator's generalized coordinate and its time derivative, in the units of the
    transmission. angleXRad_Gs_to_Wn_ixyz and angleDerivativeXRad_Gs_to_Wn_ixyz are the
    actual flap angle (rad) and its time derivative (rad/s), which follow from the
    actuator state and the transmission. jacobian and hessian are the transmission's
    derivatives of the flap angle at the actuator position. fluid_torque is the torque
    (N*m) that the fluid exerts on the Wings about their flap axis, positive in the
    direction of increasing flap angle. wing_flap_inertia is the Wings' combined inertia
    about the flap axis (kg*m^2).

    The desired fields come from the FlapReference, in the units of the transmission.
    They are zero when the FlapDrive has no reference. desired_actuator_position,
    desired_actuator_velocity, and desired_actuator_acceleration are interpolated at the
    current time, which is what a feedback term should compare the actual state against.
    feedforward_force is the generalized force that the actuator would need to apply to
    follow the reference given the current fluid torque, which already accounts for the
    Wings' reflected inertia, the transmission's Jacobian and Hessian, damping, and the
    passive elements. Which reference state it uses is set by the FlapDrive's
    use_position_lookup.

    **Important, the feedforward's fluid torque is not predicted:**

    feedforward_force uses fluid_torque, the torque from the previous time step's UVLM
    solution, at the Wings' actual motion, held constant over the step. The blade
    element theory code's singleWingInverseDynamicsController instead evaluates the
    aerodynamic torque at the desired motion: it calls aerodynamicBladeElementTheoryCalc
    with the current wing angle and the desired wing velocity, wing_velo_des = J_motor *
    des_motor_velo. That isn't possible here, because the UVLM can't cheaply evaluate the
    loads for a motion the Wings aren't actually making. The difference matters most
    when the actuator lags its reference, for example while it is saturated: there the
    blade element theory feedforward asks for the torque to overcome the aerodynamic
    load at the desired, faster flap rate, while this one only asks for the load at the
    actual, slower rate from one step ago, so it is less aggressive, and it also lags
    where the load changes quickly, as near the turnarounds. To predict it instead, pass
    your own aerodynamic model as the FlapDrive's feedforward_aerodynamic, which is then
    evaluated at the actual flap angle and the desired flap rate, as in the blade
    element theory code.

    **The feedforward's terms:**

    feedforward_force is the sum of feedforward_inertial_force,
    feedforward_damping_force, feedforward_passive_force, and
    feedforward_aerodynamic_force, which are the terms that the FlapDrive's
    feedforward_inertial, feedforward_damping, feedforward_passive, and
    feedforward_aerodynamic switches control. A term that is switched off is 0.0, and
    all of them are 0.0 when the FlapDrive has no reference.
    """

    time: float
    actuator_position: float
    actuator_velocity: float
    angleXRad_Gs_to_Wn_ixyz: float
    angleDerivativeXRad_Gs_to_Wn_ixyz: float
    jacobian: float
    hessian: float
    fluid_torque: float
    wing_flap_inertia: float
    desired_actuator_position: float
    desired_actuator_velocity: float
    desired_actuator_acceleration: float
    feedforward_force: float
    feedforward_inertial_force: float
    feedforward_damping_force: float
    feedforward_passive_force: float
    feedforward_aerodynamic_force: float


class FlapDriveAccelerations(NamedTuple):
    """The accelerations and loads that FlapDrive.single_wing_dynamics returns.

    VALIDATED CORRECT

    actuator_acceleration is the second time derivative of the actuator position, in the
    units of the transmission per second squared.
    angleSecondDerivativeXRad_Gs_to_Wn_ixyz is the Wing's resulting flap angle
    acceleration (rad/s^2). actuator_force is the generalized force that the actuator
    function returned (N*m for a rotary actuator, N for a linear one).
    transmission_torque is the torque (N*m) that the transmission applies to the Wing
    about its flap axis, which is what the linkage must supply. It stays finite where
    the Jacobian goes to zero, unlike the equivalent torque at the actuator.
    """

    actuator_acceleration: float
    angleSecondDerivativeXRad_Gs_to_Wn_ixyz: float
    actuator_force: float
    transmission_torque: float


class FlapDrive:
    """A class used to contain the flapping dynamics and the controller architecture of
    the Wings that one actuator drives.

    VALIDATED CORRECT

    A FlapDrive is everything a FlapDrivenAeroelasticUnsteadyProblem needs to advance
    its Wings' flap angle from one time step to the next. Instead of following a
    prescribed flap motion, the Wings flap wherever the actuator's dynamics take them,
    so the aerodynamic load, the Wings' inertia, and the actuator's limits all shape the
    motion. It brings together:

    - The actuator (actuator, actuator_inertia, actuator_damping): a motor, a muscle,
      or anything else whose force a function of its state can describe.

    - The transmission, any object that conforms to the Transmission protocol, such as
      a crank_linkage.CrankLinkage or a FlapTransmission. It converts the actuator's
      position into the flap angle and carries the Wings' loads back to the actuator,
      through its Jacobian and Hessian.

    - The Wings it flaps (wing_indices) and their inertia about the flap axis, plus
      optional passive elements, such as springs and dampers, at the actuator and at
      the Wings' shoulder.

    - The control architecture: the reference to track (a FlapReference), the
      feedforward that the inverse dynamics give for it, which terms the feedforward
      includes (the feedforward_* switches), and whether it reads the reference by
      position (use_position_lookup).

    **Dynamics and control:**

    The class splits the work like the blade element theory code does. The dynamics
    are single_wing_dynamics (singleWingDynamics there), which solves the equation of
    motion below for the actuator's and the Wings' accelerations. The control
    architecture is inverse_dynamics_controller (singleWingInverseDynamicsController
    there), which gathers the actual and desired states and computes the feedforward
    force. Both are built from the same terms, _equation_of_motion_terms, so they can't
    drift apart. The control law itself, meaning the feedback gains, and any motor,
    controller, or gearbox limits, such as a MotorSaturationController, lives in the
    actuator function that the user supplies, which receives a FlapDriveInputs and
    returns the actuator's force.

    **How a simulation uses it:**

    At each time step, the FlapDrivenAeroelasticUnsteadyProblem takes the fluid torque
    on the Wings from the latest UVLM solution and integrates the actuator's state over
    the step. At every evaluation, inverse_dynamics_controller builds a FlapDriveInputs,
    the actuator function returns a force, and single_wing_dynamics returns the
    accelerations. The flap angle and flap acceleration at the end of the step then set
    the Wings' geometry and the torsional springs' inertial forcing for the next step.
    A FlapDrive holds no state of its own between evaluations. The problem keeps the
    actuator's state and the histories.

    **Equation of motion:**

    With the actuator position q, the flap angle phi, the Jacobian J = d(phi)/dq, the
    Hessian H = d^2(phi)/dq^2, the actuator inertia I_a, the Wings' inertia about the
    flap axis I_w, and the damping coefficient c, the actuator accelerates according to

    (I_a + J^2 * I_w) * q'' = F_a - c * q' + F_p + J * (T_fluid + T_p) - J * I_w * H *
    q'^2

    where F_a is the actuator function's output, F_p is the optional passive actuator
    force, T_fluid is the fluid torque on the Wings about the flap axis, and T_p is the
    optional passive Wing torque. F_a, F_p, T_fluid, and T_p all follow one sign
    convention: each is the force or torque applied to the body it acts on, positive in
    the direction of increasing actuator position or flap angle. So a restoring spring
    is written with a minus sign, as -k * (q - q_0) or -k * (phi - phi_0), and the
    equation adds it. The blade element theory code's singleWingDynamics instead writes
    its springs as reaction torques, +k * (q - q_0) and +k * (phi - phi_0), and
    subtracts them, which gives the same motion. The term J^2 * I_w is the Wings'
    inertia reflected to the actuator. The flap angle acceleration then follows from phi'' = H * q'^2 + J *
    q''. This form multiplies by J instead of dividing by a mechanical advantage, so it
    stays finite where the linkage is singular and J goes to zero.

    All Wings in one FlapDrive share a single actuator state, flap angle, and fluid
    torque (the sum over the Wings). Give each Wing its own FlapDrive to flap them
    independently.
    """

    __slots__ = (
        "_wing_indices",
        "_transmission",
        "_actuator",
        "_actuator_inertia",
        "_actuator_damping",
        "_initial_actuator_position",
        "_initial_actuator_velocity",
        "_passive_wing_torque_function",
        "_passive_actuator_force_function",
        "_wing_flap_inertia",
        "_reference",
        "_use_position_lookup",
        "_feedforward_inertial",
        "_feedforward_damping",
        "_feedforward_passive",
        "_feedforward_aerodynamic",
    )

    def __init__(
        self,
        wing_indices: Sequence[int],
        transmission: Transmission,
        actuator: Callable[[FlapDriveInputs], float],
        actuator_inertia: float | int,
        actuator_damping: float | int = 0.0,
        initial_actuator_position: float | int | None = None,
        initial_actuator_velocity: float | int | None = None,
        passive_wing_torque_function: Callable[[float, float], float] | None = None,
        passive_actuator_force_function: Callable[[float, float], float] | None = None,
        wing_flap_inertia: float | int | None = None,
        *,
        reference: FlapReference,
        use_position_lookup: bool | np.bool,
        feedforward_inertial: bool | np.bool = True,
        feedforward_damping: bool | np.bool = True,
        feedforward_passive: bool | np.bool = True,
        feedforward_aerodynamic: (
            bool | np.bool | Callable[[float, float], float]
        ) = True,
    ) -> None:
        """The initialization method.

        :param wing_indices: A sequence of ints with shape (N,) holding the indices of
            the Wings that this FlapDrive flaps, in the first Airplane's wings. Can be a
            tuple, list, or ndarray. It must be non empty, and every value must be a non
            negative int with no duplicates. A Wing can belong to at most one FlapDrive,
            which the FlapDrivenAeroelasticUnsteadyProblem checks.
        :param transmission: The Transmission that maps the actuator position to the
            Wings' flap angle. It can be any object with the members that the
            Transmission interface lists, such as a FlapTransmission or a
            crank_linkage.CrankLinkage.
        :param actuator: A callable that accepts a FlapDriveInputs and returns the
            generalized force that the actuator applies (N*m for a rotary actuator, N
            for a linear one). Put the torque-speed limit, any controller, and any
            muscle model inside it. It must return a finite number. It is evaluated many
            times per time step, including at trial points that the integrator then
            rejects, so any state it keeps must depend only on the time it is given and
            must tolerate repeated or slightly earlier times, as the time stamped limits
            in a MotorSaturationController do.
        :param actuator_inertia: The actuator's own inertia about its coordinate (kg*m^2
            for a rotary actuator, kg for a linear one). For a motor this is the rotor
            inertia. It must be a non negative number (int or float) and will be
            converted internally to a float. The Wings' reflected inertia is added to it
            automatically, and the sum must be positive wherever the drive operates.
        :param actuator_damping: The viscous damping coefficient that acts on the
            actuator's velocity (N*m*s/rad for a rotary actuator, N*s/m for a linear
            one). It must be a non negative number (int or float) and will be converted
            internally to a float. The default is 0.0.
        :param initial_actuator_position: None, or the actuator position at the first
            time step, in the units of the transmission. If not None, it must be a
            number (int or float) and will be converted internally to a float. If None,
            the FlapDrivenAeroelasticUnsteadyProblem starts the actuator at its
            reference's first position. The Wings start at the transmission's flap angle
            at this position. The default is None.
        :param initial_actuator_velocity: None, or the actuator velocity at the first
            time step, in the units of the transmission per second. If not None, it must
            be a number (int or float) and will be converted internally to a float. If
            None, it is the reference's first velocity. The default is None.
        :param passive_wing_torque_function: None, or a callable that accepts the flap
            angle (rad) and its time derivative (rad/s) and returns the torque (N*m)
            that a passive element, such as a spring or damper at the shoulder, applies
            to the Wings about the flap axis. A positive value acts in the direction of
            increasing flap angle, so a restoring spring returns -k * (angle - offset).
            It must return a finite number. The default is None, which applies no
            passive Wing torque.
        :param passive_actuator_force_function: None, or a callable that accepts the
            actuator position and velocity and returns the generalized force that a
            passive element, such as a spring on the motor shaft, applies to the
            actuator. A positive value acts in the direction of increasing actuator
            position, so a restoring spring returns -k * (position - offset). It must
            return a finite number. The default is None, which applies no passive
            actuator force.
        :param wing_flap_inertia: None, or the Wings' combined inertia about the flap
            axis (kg*m^2). If not None, it must be a positive number (int or float) and
            will be converted internally to a float, and it replaces the inertia that
            the FlapDrivenAeroelasticUnsteadyProblem estimates from its wing density and
            the Wings' geometry. Use it when the inertia is measured. WARNING: it is
            used only in this FlapDrive's equation of motion. The problem's torsional
            springs, which deform the Wings, always use strip inertias from the
            problem's wing density. If this value disagrees with that estimate, the
            drive and the structure model Wings of different mass, and if this value is
            the true inertia, the computed wing deformation is likely invalid. Choose
            the wing density so that the two agree. The default is None.
        :param reference: The FlapReference that this FlapDrive follows, the only
            definition of its flap trajectory. It must cover the whole simulation. Build
            one with FlapReference.constant_velocity, or with
            FlapReference.from_flap_trajectory and generate_flap_trajectory. Its period
            sets the window of the cycle averaged results. It is a required keyword
            argument with no default.
        :param use_position_lookup: Determines which desired velocity and acceleration
            the feedforward force uses: the ones at the reference entry whose position
            is nearest the actuator's actual position (True), like the blade element
            theory code's use_closest_pos, or the ones at the current time (False). It
            is the controller's choice, so it lives here rather than on the reference.
            The desired state that a feedback term compares against is always read at
            the current time. It is an explicit choice: nothing turns the position
            lookup on automatically. It is meaningful only for a reference whose
            positions are monotonic, as they are for a motor that keeps turning one way,
            such as one driving a crank linkage, and if True with a reference whose
            positions aren't monotonic, a ValueError is raised. Can be a bool or a numpy
            bool. It is a required keyword argument with no default, so that every
            FlapDrive states which way its feedforward reads the reference.
        :param feedforward_inertial: Determines whether the feedforward force includes
            the inertial term, (I_a + J^2 * I_w) * q''_desired + J * I_w * H *
            q'_desired^2, the force to accelerate the actuator and the Wings, including
            the centripetal part that a linkage needs even at a constant speed. Can be a
            bool or a numpy bool. The default is True.
        :param feedforward_damping: Determines whether the feedforward force includes
            the damping term, c * q'_desired. Can be a bool or a numpy bool. The default
            is True.
        :param feedforward_passive: Determines whether the feedforward force includes
            the passive term, -F_p - J * T_p, which cancels the passive actuator force
            and the passive Wing torque, such as springs. They are evaluated at the
            actual position and the desired velocity. Can be a bool or a numpy bool. The
            default is True.
        :param feedforward_aerodynamic: Determines the aerodynamic term of the
            feedforward force, -J * T_aero, which cancels the aerodynamic torque on the
            Wings. If True, T_aero is the fluid torque given to
            inverse_dynamics_controller, which in a simulation is the previous time
            step's UVLM torque at the Wings' actual motion. If False, there is no
            aerodynamic term. If a callable, it is your own aerodynamic model: it
            accepts the actual flap angle (rad) and the desired flap rate, J *
            q'_desired (rad/s), and returns the predicted aerodynamic torque on the
            Wings about the flap axis (N*m), positive in the direction of increasing
            flap angle, like the blade element theory code's
            singleWingInverseDynamicsController, which evaluates its aerodynamic model
            at the desired wing velocity. It must return a finite number. The default is
            True.
        :return: None
        """
        # Validate the Wings that this drive flaps. Whether those indices exist on the
        # Airplane is checked by the problem, which knows the Airplane.
        if isinstance(wing_indices, (str, bytes)) or not isinstance(
            wing_indices, (Sequence, np.ndarray)
        ):
            raise TypeError("wing_indices must be a tuple, list, or ndarray of ints.")
        if len(wing_indices) == 0:
            raise ValueError("wing_indices can't be empty.")
        validated_wing_indices = tuple(
            _parameter_validation.int_in_range_return_int(
                int(index) if isinstance(index, np.integer) else index,
                f"wing_indices[{position}]",
                0,
                True,
            )
            for position, index in enumerate(wing_indices)
        )
        if len(set(validated_wing_indices)) != len(validated_wing_indices):
            raise ValueError("wing_indices can't contain duplicates.")
        self._wing_indices = validated_wing_indices

        # Accept any powertrain that provides the forward kinematics and its
        # derivatives. The check only confirms that the members exist.
        if not isinstance(transmission, Transmission):
            raise TypeError(
                "transmission must satisfy the Transmission interface, meaning it must "
                "have evaluate_angle, evaluate_jacobian, evaluate_hessian, and "
                "convert_trajectory."
            )
        self._transmission = transmission

        if not callable(actuator):
            raise TypeError("actuator must be a callable.")
        self._actuator = actuator

        # The actuator inertia is the blade element theory code's rotor_inertia, and the
        # damping is its motor_damping_const.
        self._actuator_inertia = _parameter_validation.number_in_range_return_float(
            actuator_inertia, "actuator_inertia", 0.0, True
        )
        self._actuator_damping = _parameter_validation.number_in_range_return_float(
            actuator_damping, "actuator_damping", 0.0, True
        )
        # A None initial state is filled in by the problem from the reference's first
        # sample, like the blade element theory code's initial_state =
        # [motor_pos_vec_des(1); motor_velo_vec_des(1)].
        if initial_actuator_position is not None:
            initial_actuator_position = (
                _parameter_validation.number_in_range_return_float(
                    initial_actuator_position, "initial_actuator_position"
                )
            )
        self._initial_actuator_position = initial_actuator_position
        if initial_actuator_velocity is not None:
            initial_actuator_velocity = (
                _parameter_validation.number_in_range_return_float(
                    initial_actuator_velocity, "initial_actuator_velocity"
                )
            )
        self._initial_actuator_velocity = initial_actuator_velocity

        # The passive elements stand in for the blade element theory code's
        # spring_params: the wing spring and the motor spring.
        if passive_wing_torque_function is not None and not callable(
            passive_wing_torque_function
        ):
            raise TypeError("passive_wing_torque_function must be a callable or None.")
        self._passive_wing_torque_function = passive_wing_torque_function

        if passive_actuator_force_function is not None and not callable(
            passive_actuator_force_function
        ):
            raise TypeError(
                "passive_actuator_force_function must be a callable or None."
            )
        self._passive_actuator_force_function = passive_actuator_force_function

        if wing_flap_inertia is not None:
            wing_flap_inertia = _parameter_validation.number_in_range_return_float(
                wing_flap_inertia, "wing_flap_inertia", 0.0, False
            )
        self._wing_flap_inertia = wing_flap_inertia

        if not isinstance(reference, FlapReference):
            raise TypeError("reference must be a FlapReference.")
        self._reference = reference

        self._use_position_lookup = _parameter_validation.boolLike_return_bool(
            use_position_lookup, "use_position_lookup"
        )

        # Catch a position lookup on a table that can't support one now, rather than
        # when the simulation first evaluates the drive.
        if self._use_position_lookup and not reference.positions_are_monotonic:
            raise ValueError(
                "use_position_lookup can't be True when the reference's positions are "
                "not monotonic, because a position that appears in both directions of "
                "the stroke has no single velocity or acceleration."
            )

        # Store the switches for each term of the feedforward force.
        self._feedforward_inertial = _parameter_validation.boolLike_return_bool(
            feedforward_inertial, "feedforward_inertial"
        )
        self._feedforward_damping = _parameter_validation.boolLike_return_bool(
            feedforward_damping, "feedforward_damping"
        )
        self._feedforward_passive = _parameter_validation.boolLike_return_bool(
            feedforward_passive, "feedforward_passive"
        )
        if callable(feedforward_aerodynamic):
            self._feedforward_aerodynamic: bool | Callable[[float, float], float] = (
                feedforward_aerodynamic
            )
        else:
            self._feedforward_aerodynamic = _parameter_validation.boolLike_return_bool(
                feedforward_aerodynamic, "feedforward_aerodynamic"
            )

    # --- Immutable: read only properties ---
    @property
    def wing_indices(self) -> tuple[int, ...]:
        return self._wing_indices

    @property
    def transmission(self) -> Transmission:
        return self._transmission

    @property
    def actuator(self) -> Callable[[FlapDriveInputs], float]:
        return self._actuator

    @property
    def actuator_inertia(self) -> float:
        return self._actuator_inertia

    @property
    def actuator_damping(self) -> float:
        return self._actuator_damping

    @property
    def initial_actuator_position(self) -> float | None:
        return self._initial_actuator_position

    @property
    def initial_actuator_velocity(self) -> float | None:
        return self._initial_actuator_velocity

    @property
    def reference(self) -> FlapReference:
        return self._reference

    @property
    def passive_wing_torque_function(self) -> Callable[[float, float], float] | None:
        return self._passive_wing_torque_function

    @property
    def passive_actuator_force_function(self) -> Callable[[float, float], float] | None:
        return self._passive_actuator_force_function

    @property
    def wing_flap_inertia(self) -> float | None:
        return self._wing_flap_inertia

    @property
    def use_position_lookup(self) -> bool:
        return self._use_position_lookup

    @property
    def feedforward_inertial(self) -> bool:
        return self._feedforward_inertial

    @property
    def feedforward_damping(self) -> bool:
        return self._feedforward_damping

    @property
    def feedforward_passive(self) -> bool:
        return self._feedforward_passive

    @property
    def feedforward_aerodynamic(self) -> bool | Callable[[float, float], float]:
        return self._feedforward_aerodynamic

    def _equation_of_motion_terms(
        self,
        actuator_position: float,
        actuator_velocity: float,
        jacobian: float,
        hessian: float,
        angle_rad: float,
        fluid_torque: float,
        wing_flap_inertia: float,
    ) -> tuple[float, float, float]:
        """Splits the equation of motion into the terms that don't depend on the
        actuator's force.

        Both single_wing_dynamics and inverse_dynamics_controller are built from these
        terms, as the blade element theory code's singleWingDynamics and
        singleWingInverseDynamicsController share the same torques.

        VALIDATED CORRECT

        The equation of motion in the class docstring can be written as
        effective_inertia * q'' = F_a + bias, where F_a is the actuator function's
        output. Both the forward dynamics and the feedforward force are built from these
        terms, so they can't drift apart.

        :param actuator_position: The actuator position.
        :param actuator_velocity: The actuator velocity that the velocity dependent terms
            use.
        :param jacobian: The transmission's Jacobian at the actuator position.
        :param hessian: The transmission's Hessian at the actuator position.
        :param angle_rad: The flap angle at the actuator position (rad).
        :param fluid_torque: The fluid torque on the Wings about the flap axis (N*m).
        :param wing_flap_inertia: The Wings' inertia about the flap axis (kg*m^2).
        :return: A tuple of the effective inertia, the bias force, and the passive Wing
            torque (N*m).
        """
        # This mirrors the blade element theory code's singleWingDynamics, rearranged so
        # that every torque at the wing is multiplied by J instead of divided by the
        # mechanical advantage mech_adv = 1 / J. The two are equal wherever J isn't 0.0,
        # and multiplying keeps every term finite at the linkage's dead centers.

        # Get the torque that the passive element at the shoulder applies to the Wings.
        # It is evaluated at the flap angle and the flap angle's rate, J * q' (the wing
        # spring, wing_spring_torque there, with the opposite sign, since this one is
        # the torque on the Wings rather than a reaction torque). VALIDATED CORRECT
        passive_wing_torque = 0.0
        if self._passive_wing_torque_function is not None:
            passive_wing_torque = _call_scalar_function(
                self._passive_wing_torque_function,
                "passive_wing_torque_function",
                angle_rad,
                jacobian * actuator_velocity,
            )

        # Get the force that the passive element on the actuator applies to it (the
        # motor spring, motor_spring_torque there). VALIDATED CORRECT
        passive_actuator_force = 0.0
        if self._passive_actuator_force_function is not None:
            passive_actuator_force = _call_scalar_function(
                self._passive_actuator_force_function,
                "passive_actuator_force_function",
                actuator_position,
                actuator_velocity,
            )

        # The Wings' inertia reflected to the actuator joins the actuator's own inertia
        # on the left side of the equation of motion, so it is treated implicitly. This
        # is the blade element theory code's denominator I_eff +
        # torque_inertial_wing_accel_motor, where the second term is I_w / mech_adv^2,
        # which equals J^2 * I_w. VALIDATED CORRECT
        effective_inertia = self._actuator_inertia + jacobian**2 * wing_flap_inertia
        if not effective_inertia > 0.0:
            raise ValueError(
                "The actuator's effective inertia (actuator_inertia plus the Wings' "
                "reflected inertia) must be positive, but it is "
                f"{effective_inertia}. Increase actuator_inertia if the Jacobian can "
                "reach zero."
            )

        # Collect every term that doesn't depend on the actuator's force. In the blade
        # element theory code's notation they are:
        #
        # -motor_damping_const * motor_velo, for the damping.
        #
        # -motor_spring_torque, for the passive actuator force. Here it is +F_p, because
        # F_p is the force the passive element applies to the actuator, while that
        # code's motor_spring_torque is a reaction torque.
        #
        # -tau_aero_motor and -tau_spring_wing_motor, the fluid and passive Wing torques
        # carried to the actuator. Here they are + J * (fluid + passive) because these
        # torques act on the Wings, while that code's are reaction torques.
        #
        # -torque_inertial_wing_speed_motor, the centripetal term. The Wings'
        # acceleration has a part H * q'^2 that comes from the linkage alone, even at a
        # constant motor speed, and the Wings' inertia resists it.
        bias = (
            -self._actuator_damping * actuator_velocity  # VALIDATED CORRECT
            + passive_actuator_force  # VALIDATED CORRECT
            + jacobian * (fluid_torque + passive_wing_torque)  # VALIDATED CORRECT
            # VALIDATED CORRECT
            - jacobian * wing_flap_inertia * hessian * actuator_velocity**2
        )
        return effective_inertia, bias, passive_wing_torque

    def inverse_dynamics_controller(
        self,
        time: float | int,
        actuator_position: float | int,
        actuator_velocity: float | int,
        fluid_torque: float | int,
        computed_wing_flap_inertia: float | int | None = None,
    ) -> FlapDriveInputs:
        """Builds the record that is passed to the actuator function, including the
        feedforward force from the inverse dynamics.

        VALIDATED CORRECT

        It is named after the blade element theory code's
        singleWingInverseDynamicsController, which computes the same feedforward. Unlike
        that function, it doesn't apply feedback gains or limits, which belong to the
        actuator function.

        It evaluates the transmission at the actuator position to fill in the actual
        flap angle, the flap angle's time derivative, the Jacobian, and the Hessian. It
        also fills in the desired actuator state from this FlapDrive's reference, and
        the feedforward force. The feedforward force uses fluid_torque as given, which
        in a simulation is the previous time step's fluid torque at the actual motion,
        not a prediction at the desired motion as in the blade element theory code. See
        FlapDriveInputs.

        :param time: The simulation time (s).
        :param actuator_position: The actuator position, in the units of the
            transmission.
        :param actuator_velocity: The actuator velocity, in the units of the
            transmission per second.
        :param fluid_torque: The torque (N*m) that the fluid exerts on the Wings about
            their flap axis, positive in the direction of increasing flap angle.
        :param computed_wing_flap_inertia: None, or the Wings' combined inertia about
            the flap axis (kg*m^2), as estimated by the caller. It is ignored if this
            FlapDrive has a wing_flap_inertia, and it is required if it does not. If not
            None, it must be a positive number (int or float). The default is None.
        :return: A FlapDriveInputs holding the state at this instant.
        """
        # Choose the Wings' inertia about the flap axis. If the user gave one when
        # constructing this FlapDrive (wing_flap_inertia, for example a measured value),
        # use it. Otherwise, use the one the caller computed. In a simulation, the
        # caller is the FlapDrivenAeroelasticUnsteadyProblem, which estimates it from
        # its wing_density and the Wings' panel geometry, the same strip model its
        # torsional springs use. It doesn't come from the aerodynamics. If neither is
        # available, there is no inertia to use, so it's an error. The user given value
        # plays the role of the blade element theory code's
        # inertialBladeElementTheoryCalc, whose leading edge Ixx plus membrane inertia
        # is the wing's inertia about the flap axis.
        #
        # WARNING: A USER GIVEN INERTIA IS USED ONLY HERE, IN THE DRIVE'S EQUATION OF
        # MOTION. THE TORSIONAL SPRINGS THAT DEFORM THE WINGS ALWAYS USE THE STRIP
        # INERTIAS FROM THE PROBLEM'S wing_density. If the two disagree, the drive and
        # the structure model wings of different mass. If the user given value is the
        # true inertia, the drive's motion is right but the wing deformation, which
        # depends on the strips' inertial loads, is computed for the wrong mass and is
        # likely invalid. Set wing_density so that its estimate matches the user given
        # value, or omit the user given value so that both use the same estimate.
        # However, the UVLM doesn't account for leading edge inertia, so you can use
        # this to adjust for that.
        if self._wing_flap_inertia is not None:
            wing_flap_inertia = self._wing_flap_inertia
        elif computed_wing_flap_inertia is None:
            raise ValueError(
                "Either this FlapDrive's wing_flap_inertia or "
                "computed_wing_flap_inertia must be provided."
            )
        else:
            wing_flap_inertia = _parameter_validation.number_in_range_return_float(
                computed_wing_flap_inertia, "computed_wing_flap_inertia", 0.0, False
            )

        position = float(actuator_position)
        velocity = float(actuator_velocity)
        fluid_torque = float(fluid_torque)

        # Evaluate the transmission at the actuator's position: the flap angle, J, and H
        # (wing_angle_curr, J_motor, and H_motor in the blade element theory code).
        jacobian = self._transmission.evaluate_jacobian(position)
        hessian = self._transmission.evaluate_hessian(position)
        angle_rad = self._transmission.evaluate_angle(position)

        feedforward_inertial_force = 0.0
        feedforward_damping_force = 0.0
        feedforward_passive_force = 0.0
        feedforward_aerodynamic_force = 0.0
        reference = self._reference
        # Read the desired state at the current time, which the feedback compares the
        # actual state against (des_motor_pos_traj and des_motor_velo_traj in the blade
        # element theory code).
        desired_position, desired_velocity, desired_acceleration = reference.evaluate(
            time
        )

        # The feedforward uses either the desired velocity and acceleration at the
        # current time, or the ones at the table entry nearest the actuator's position,
        # as this drive's use_position_lookup selects, like the blade element theory
        # code's use_closest_pos (motor_ang_velo_des and motor_ang_accel_des there).
        lookup_velocity = desired_velocity
        lookup_acceleration = desired_acceleration
        if self._use_position_lookup:
            lookup_velocity, lookup_acceleration = reference.evaluate_at_position(
                position
            )
        # The inverse dynamics use the actual Jacobian and Hessian, which depend on
        # where the actuator is, together with the desired velocity and acceleration.
        # This is the blade element theory code's singleWingInverseDynamicsController,
        # which evaluates J and H at the current motor position and the wing's speed and
        # acceleration at the desired motor speed and acceleration. It solves the
        # equation of motion for the force that gives the desired acceleration,
        # effective_inertia * q''_desired = F + bias, one term at a time, so that each
        # term can be switched on or off with this drive's feedforward_* switches. The
        # terms sum to effective_inertia * q''_desired - bias when all are on.
        desired_flap_rate = jacobian * lookup_velocity

        # The inertial term: the actuator's own inertia times its desired acceleration,
        # plus the Wings' inertia times their desired flap acceleration, carried to the
        # actuator by the Jacobian. The desired flap acceleration, phi'' = H * q'^2 + J
        # * q'', includes H * q'^2, which a linkage needs even at a constant desired
        # actuator speed. Expanded, this is (I_a + J^2 * I_w) * q'' + J * I_w * H *
        # q'^2, the blade element theory code's motor_torque_des_due_to_wing plus
        # torque_inertial_wing_speed_motor.
        #
        # VALIDATED CORRECT
        if self._feedforward_inertial:
            desired_flap_acceleration = (
                hessian * lookup_velocity**2 + jacobian * lookup_acceleration
            )
            feedforward_inertial_force = (
                self._actuator_inertia * lookup_acceleration
                + jacobian * wing_flap_inertia * desired_flap_acceleration
            )

        # The damping term, at the desired velocity (motor_damping_const *
        # des_motor_velo there).
        #
        # VALIDATED CORRECT
        if self._feedforward_damping:
            feedforward_damping_force = self._actuator_damping * lookup_velocity

        # The passive term: whatever the passive elements push with, the actuator must
        # cancel. They are evaluated at the actual position and the desired velocity,
        # like the other terms (motor_spring_torque and tau_spring_wing_motor there).
        #
        # VALIDATED CORRECT
        if self._feedforward_passive:
            passive_actuator_force = 0.0
            if self._passive_actuator_force_function is not None:
                passive_actuator_force = _call_scalar_function(
                    self._passive_actuator_force_function,
                    "passive_actuator_force_function",
                    position,
                    lookup_velocity,
                )
            passive_wing_torque = 0.0
            if self._passive_wing_torque_function is not None:
                passive_wing_torque = _call_scalar_function(
                    self._passive_wing_torque_function,
                    "passive_wing_torque_function",
                    angle_rad,
                    desired_flap_rate,
                )
            feedforward_passive_force = (
                -passive_actuator_force - jacobian * passive_wing_torque
            )

        # The aerodynamic term: the actuator must cancel the aerodynamic torque on the
        # Wings, carried through the transmission (tau_aero_motor there). By default it
        # is the fluid torque as given, which in a simulation is the previous time
        # step's UVLM torque at the Wings' actual motion. The blade element theory code
        # instead predicts it at the desired wing velocity
        # (aerodynamicBladeElementTheoryCalc with wing_velo_des), which the UVLM can't
        # do cheaply. A user model passed as feedforward_aerodynamic does that
        # prediction instead, at the actual flap angle and the desired flap rate.
        #
        # VALIDATED CORRECT
        if self._feedforward_aerodynamic is not False:
            if callable(self._feedforward_aerodynamic):
                feedforward_fluid_torque = _call_scalar_function(
                    self._feedforward_aerodynamic,
                    "feedforward_aerodynamic",
                    angle_rad,
                    desired_flap_rate,
                )
            else:
                feedforward_fluid_torque = fluid_torque
            feedforward_aerodynamic_force = -jacobian * feedforward_fluid_torque

        feedforward_force = (
            feedforward_inertial_force
            + feedforward_damping_force
            + feedforward_passive_force
            + feedforward_aerodynamic_force
        )

        # The flap angle's rate is J * q' (wing_velo_curr there).
        return FlapDriveInputs(
            time=float(time),
            actuator_position=position,
            actuator_velocity=velocity,
            angleXRad_Gs_to_Wn_ixyz=angle_rad,
            angleDerivativeXRad_Gs_to_Wn_ixyz=jacobian * velocity,
            jacobian=jacobian,
            hessian=hessian,
            fluid_torque=fluid_torque,
            wing_flap_inertia=wing_flap_inertia,
            desired_actuator_position=desired_position,
            desired_actuator_velocity=desired_velocity,
            desired_actuator_acceleration=desired_acceleration,
            feedforward_force=feedforward_force,
            feedforward_inertial_force=feedforward_inertial_force,
            feedforward_damping_force=feedforward_damping_force,
            feedforward_passive_force=feedforward_passive_force,
            feedforward_aerodynamic_force=feedforward_aerodynamic_force,
        )

    def single_wing_dynamics(self, inputs: FlapDriveInputs) -> FlapDriveAccelerations:
        """Calculates the actuator and flap angle accelerations from the equation of
        motion in the class docstring.

        It is named after the blade element theory code's singleWingDynamics, which
        solves the same equation of motion.

        VALIDATED CORRECT

        :param inputs: The FlapDriveInputs holding the drive's state at this instant.
        :return: A FlapDriveAccelerations holding the actuator acceleration, the flap
            angle acceleration, the actuator's generalized force, and the torque that
            the transmission applies to the Wings.
        """
        # Ask the actuator function for its force. In the blade element theory code this
        # is the controller's torque_des clipped by motor_sat_controller.clipTorque,
        # giving torque_applied.
        actuator_force = _call_scalar_function(self._actuator, "actuator", inputs)
        # VALIDATED CORRECT
        effective_inertia, bias, passive_wing_torque = self._equation_of_motion_terms(
            inputs.actuator_position,
            inputs.actuator_velocity,
            inputs.jacobian,
            inputs.hessian,
            inputs.angleXRad_Gs_to_Wn_ixyz,
            inputs.fluid_torque,
            inputs.wing_flap_inertia,
        )
        # Solve for the actuator's acceleration (motor_accel in the blade element theory
        # code's singleWingDynamics).  VALIDATED CORRECT
        actuator_acceleration = (actuator_force + bias) / effective_inertia

        # Get the Wings' flap acceleration from the actuator's motion through the
        # transmission (wing_accel_curr = H_motor * curr_motor_velo^2 + J_motor *
        # motor_accel there).  VALIDATED CORRECT
        angleSecondDerivativeXRad_Gs_to_Wn_ixyz = (
            inputs.hessian * inputs.actuator_velocity**2
            + inputs.jacobian * actuator_acceleration
        )

        # The Wings obey I_w * phi'' = T_transmission + T_fluid + T_passive, so the
        # torque that the transmission supplies follows directly. Unlike the torque at
        # the actuator, it never gets divided by the Jacobian. It is what to size the
        # linkage from. It equals the blade element theory code's wing_torques.total,
        # with the same sign: that code's tau_inertial_wing is +I_w * phi'', and its
        # tau_aero_wing and wing_spring_torque are reaction torques, -T_fluid and
        # -T_passive here, so their sum is this same expression. VALIDATED CORRECT
        transmission_torque = (
            inputs.wing_flap_inertia * angleSecondDerivativeXRad_Gs_to_Wn_ixyz
            - inputs.fluid_torque
            - passive_wing_torque
        )

        return FlapDriveAccelerations(
            actuator_acceleration=actuator_acceleration,
            angleSecondDerivativeXRad_Gs_to_Wn_ixyz=angleSecondDerivativeXRad_Gs_to_Wn_ixyz,
            actuator_force=actuator_force,
            transmission_torque=transmission_torque,
        )
