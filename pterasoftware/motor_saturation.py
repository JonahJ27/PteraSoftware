"""Contains the MotorSaturationController class, which limits the torque that a motor,
its controller, and its gearbox can apply.

SCRIPT VALIDATED CORRECT
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from . import _parameter_validation


class SaturationInfo(NamedTuple):
    """What a MotorSaturationController did to one torque command.

    VALIDATED CORRECT

    saturation_active is a tuple of the names of the limits that clipped the command, in
    the order they were applied: "motor_torque_speed_curve", "max_brake_current",
    "gearbox_max_torque", "motor_peak_torque", and either "esc_peak_current" (the
    controller's peak current bound it) or "esc_continuous_current" (the controller was
    locked, so its continuous current bound it). limits holds the value (N*m) of each of
    those limits, in the same order. max_torque_possible is the largest torque (N*m)
    that the motor could apply at its speed, given the active limits. The two peak time
    fields are how much of the motor's and the controller's peak allowances have been
    used (s), the two remaining fields are how much is left (s), and the two locked
    fields say whether each is held to its continuous limit.
    """

    saturation_active: tuple[str, ...]
    limits: tuple[float, ...]
    max_torque_possible: float
    motor_peak_time_elapsed: float
    esc_peak_time_elapsed: float
    motor_peak_time_remaining: float
    esc_peak_time_remaining: float
    motor_locked: bool
    esc_locked: bool


class _LeakyBucket:
    """A time stamped record of how much of a peak allowance has been used.

    The level at any time t is level_reference + rate * (t - time_reference), never
    below 0.0. rate is 1.0 while filling and -1.0 while draining. lock_end is the time
    at which the current lock to the continuous limit ends.
    """

    __slots__ = ("level_reference", "time_reference", "rate", "lock_end")

    def __init__(self) -> None:
        """The initialization method.

        :return: None
        """
        self.level_reference = 0.0
        self.time_reference: float | None = None
        self.rate = -1.0
        self.lock_end = -np.inf


class MotorSaturationController:
    """A class used to limit the torque that a motor, its motor controller, and its
    gearbox can apply.

    VALIDATED CORRECT

    It is a port of the motor saturation controller of the blade element theory motor
    limit simulation. A torque command passes through these limits in order:

    **1. Torque-speed curve.** A torque that drives the motor along its direction of
    motion is limited by the motor's torque-speed curve, a polynomial in the absolute
    speed that is clamped at 0.0, so it falls to 0.0 at the no-load speed. A torque that
    opposes the motion is braking, which a drive can still supply above the no-load
    speed, so the curve doesn't apply to it.

    **2. Brake current.** A braking torque is limited by the torque at the largest
    current that the drive can push into braking, which models a brake resistor or the
    bus's ability to absorb regenerated energy.

    **3. Gearbox.** Every torque is limited by the gearbox's largest output torque
    divided by its gear ratio.

    **4. Motor peak.** The motor can exceed its continuous torque only for a peak
    duration. Once that allowance runs out, the motor is held to its continuous torque.

    **5. Controller peak.** The motor controller can supply its peak current only for a
    peak duration, and its continuous current otherwise. Currents are converted to
    torques with the motor's torque-current polynomial.

    **Time stamped peak allowances:**

    The motor and the controller each have a leaky bucket that holds how much of their
    peak allowance has been used (s). It fills at 1.0 s/s while the applied torque is
    above that limiter's continuous limit and drains at 1.0 s/s otherwise. Each bucket
    stores its level at a reference time and whether it is filling or draining, so its
    level is a function of the time it is asked about, not of how often it is asked. An
    integrator can therefore evaluate the limits several times per step, as adaptive
    integrators do. When a filling bucket passes the peak duration, the allowance runs
    out at an exact time, and the limiter is locked to its continuous limit for half the
    peak duration while the bucket drains.

    Time never runs backward inside the controller. A time earlier than one already
    seen, as from a retried integrator step, is treated as the latest time seen, so the
    error is at most one integrator step. A trial evaluation at a later time that the
    integrator then rejects can advance a lock slightly early.
    """

    __slots__ = (
        "_speed_torque_coefficients",
        "_torque_current_coefficients",
        "_motor_continuous_torque",
        "_motor_peak_duration",
        "_esc_peak_current",
        "_esc_continuous_current",
        "_esc_peak_duration",
        "_esc_max_brake_current",
        "_gearbox_max_torque",
        "_gear_ratio",
        "_motor_bucket",
        "_esc_bucket",
        "_last_info",
    )

    def __init__(
        self,
        speed_torque_coefficients: np.ndarray | list[float] | tuple[float, ...],
        torque_current_coefficients: np.ndarray | list[float] | tuple[float, ...],
        motor_continuous_torque: float | int,
        motor_peak_duration: float | int,
        esc_peak_current: float | int,
        esc_continuous_current: float | int,
        esc_peak_duration: float | int,
        gearbox_max_torque: float | int,
        gear_ratio: float | int,
        esc_max_brake_current: float | int = np.inf,
    ) -> None:
        """The initialization method.
        

        :param speed_torque_coefficients: An array-like object of numbers (int or float)
            holding the polynomial coefficients, highest power first, of the motor's
            largest torque (N*m) as a function of its absolute speed (rad/s), at the
            supply voltage in use. Can be a tuple, list, or ndarray. It must have at
            least one value.
        :param torque_current_coefficients: An array-like object of numbers (int or
            float) holding the polynomial coefficients, highest power first, of the
            motor's torque (N*m) as a function of its current (A). Can be a tuple, list,
            or ndarray. It must have at least one value.
        :param motor_continuous_torque: The torque (N*m) that the motor can apply
            indefinitely. It must be a positive number (int or float).
        :param motor_peak_duration: How long (s) the motor can exceed its continuous
            torque. It must be a positive number (int or float).
        :param esc_peak_current: The current (A) that the motor controller can supply
            for its peak duration. It must be a positive number (int or float).
        :param esc_continuous_current: The current (A) that the motor controller can
            supply indefinitely. It must be a positive number (int or float).
        :param esc_peak_duration: How long (s) the motor controller can supply its peak
            current. It must be a positive number (int or float).
        :param gearbox_max_torque: The largest output torque (N*m) of the gearbox. It
            must be a positive number (int or float).
        :param gear_ratio: The gearbox's reduction from the motor to its output. It must
            be a positive number (int or float).
        :param esc_max_brake_current: The largest current (A) that the drive can push
            into braking. It must be a positive number (int or float), and it can be
            np.inf for no brake specific limit. The default is np.inf.
        :return: None
        """
        self._speed_torque_coefficients = self._validate_coefficients(
            speed_torque_coefficients, "speed_torque_coefficients"
        )
        self._torque_current_coefficients = self._validate_coefficients(
            torque_current_coefficients, "torque_current_coefficients"
        )
        self._motor_continuous_torque = (
            _parameter_validation.number_in_range_return_float(
                motor_continuous_torque,
                "motor_continuous_torque",
                min_val=0.0,
                min_inclusive=False,
            )
        )
        self._motor_peak_duration = _parameter_validation.number_in_range_return_float(
            motor_peak_duration, "motor_peak_duration", min_val=0.0, min_inclusive=False
        )
        self._esc_peak_current = _parameter_validation.number_in_range_return_float(
            esc_peak_current, "esc_peak_current", min_val=0.0, min_inclusive=False
        )
        self._esc_continuous_current = (
            _parameter_validation.number_in_range_return_float(
                esc_continuous_current,
                "esc_continuous_current",
                min_val=0.0,
                min_inclusive=False,
            )
        )
        self._esc_peak_duration = _parameter_validation.number_in_range_return_float(
            esc_peak_duration, "esc_peak_duration", min_val=0.0, min_inclusive=False
        )
        self._gearbox_max_torque = _parameter_validation.number_in_range_return_float(
            gearbox_max_torque, "gearbox_max_torque", min_val=0.0, min_inclusive=False
        )
        self._gear_ratio = _parameter_validation.number_in_range_return_float(
            gear_ratio, "gear_ratio", min_val=0.0, min_inclusive=False
        )
        if isinstance(esc_max_brake_current, (int, float)) and np.isinf(
            esc_max_brake_current
        ):
            if esc_max_brake_current < 0.0:
                raise ValueError("esc_max_brake_current must be positive.")
            self._esc_max_brake_current = float(np.inf)
        else:
            self._esc_max_brake_current = (
                _parameter_validation.number_in_range_return_float(
                    esc_max_brake_current,
                    "esc_max_brake_current",
                    min_val=0.0,
                    min_inclusive=False,
                )
            )

        self._motor_bucket = _LeakyBucket()
        self._esc_bucket = _LeakyBucket()
        self._last_info: SaturationInfo | None = None

    @staticmethod
    def _validate_coefficients(
        value: np.ndarray | list[float] | tuple[float, ...], name: str
    ) -> np.ndarray:
        """Validates polynomial coefficients and returns them as a read only array.

        VALIDATED CORRECT

        :param value: The coefficients to validate.
        :param name: The name of the coefficients, for error messages.
        :return: A (N,) read only ndarray of floats.
        """
        coefficients = np.array(value, dtype=float)
        if coefficients.ndim != 1 or coefficients.size == 0:
            raise ValueError(f"{name} must be a non empty 1D array-like.")
        if not np.isfinite(coefficients).all():
            raise ValueError(f"{name} can't contain nan, inf, or -inf.")
        coefficients.flags.writeable = False
        return coefficients

    # --- Immutable: read only properties ---
    @property
    def speed_torque_coefficients(self) -> np.ndarray:
        return self._speed_torque_coefficients

    @property
    def torque_current_coefficients(self) -> np.ndarray:
        return self._torque_current_coefficients

    @property
    def motor_continuous_torque(self) -> float:
        return self._motor_continuous_torque

    @property
    def motor_peak_duration(self) -> float:
        return self._motor_peak_duration

    @property
    def esc_peak_current(self) -> float:
        return self._esc_peak_current

    @property
    def esc_continuous_current(self) -> float:
        return self._esc_continuous_current

    @property
    def esc_peak_duration(self) -> float:
        return self._esc_peak_duration

    @property
    def esc_max_brake_current(self) -> float:
        return self._esc_max_brake_current

    @property
    def gearbox_max_torque(self) -> float:
        return self._gearbox_max_torque

    @property
    def gear_ratio(self) -> float:
        return self._gear_ratio

    @property
    def last_info(self) -> SaturationInfo | None:
        """What the most recent call to clip_torque did, or None before the first call.

        VALIDATED CORRECT

        :return: The SaturationInfo of the most recent call, or None.
        """
        return self._last_info

    # --- Torque limits ---
    def max_torque_from_speed(self, motor_velocity: float | int) -> float:
        """Returns the torque-speed curve's limit at a motor speed.

        VALIDATED CORRECT

        :param motor_velocity: The motor's velocity (rad/s). Its sign is ignored.
        :return: The largest driving torque (N*m), which is never below 0.0.
        """
        return max(
            0.0,
            float(np.polyval(self._speed_torque_coefficients, abs(motor_velocity))),
        )

    def torque_at_current(self, current: float | int) -> float:
        """Returns the motor's torque at a current.

        VALIDATED CORRECT

        :param current: The current (A).
        :return: The torque (N*m).
        """
        return float(np.polyval(self._torque_current_coefficients, current))

    def max_brake_torque(self) -> float:
        """Returns the largest braking torque that the brake current allows.

        VALIDATED CORRECT

        :return: The torque (N*m), or np.inf when there is no brake current limit.
        """
        if np.isinf(self._esc_max_brake_current):
            return float(np.inf)
        return self.torque_at_current(self._esc_max_brake_current)

    def gearbox_torque_limit(self) -> float:
        """Returns the gearbox's torque limit at the motor.

        VALIDATED CORRECT

        :return: The gearbox's largest output torque divided by the gear ratio (N*m).
        """
        return self._gearbox_max_torque / self._gear_ratio

    def max_torque_possible(
        self, motor_velocity: float | int, motor_locked: bool, esc_locked: bool
    ) -> float:
        """Returns the largest torque that the motor could apply, given the limits that
        are active.

        VALIDATED CORRECT

        :param motor_velocity: The motor's velocity (rad/s).
        :param motor_locked: Whether the motor is held to its continuous torque.
        :param esc_locked: Whether the motor controller is held to its continuous
            current.
        :return: The smallest of the active limits (N*m).
        """
        motor_limit = self._motor_continuous_torque if motor_locked else float(np.inf)
        esc_current = (
            self._esc_continuous_current if esc_locked else self._esc_peak_current
        )
        return min(
            self.max_torque_from_speed(motor_velocity),
            motor_limit,
            self.torque_at_current(esc_current),
            self.gearbox_torque_limit(),
        )

    def max_torque_continuous(self, motor_velocity: float | int) -> float:
        """Returns the largest torque that the motor could apply indefinitely.

        VALIDATED CORRECT

        :param motor_velocity: The motor's velocity (rad/s).
        :return: The smallest of the continuous limits (N*m).
        """
        return min(
            self.max_torque_from_speed(motor_velocity),
            self._motor_continuous_torque,
            self.torque_at_current(self._esc_continuous_current),
            self.gearbox_torque_limit(),
        )

    # --- Peak allowance bookkeeping ---
    def reset_state(self) -> None:
        """Empties both peak allowance buckets and forgets the last call.

        VALIDATED CORRECT

        :return: None
        """
        self._motor_bucket = _LeakyBucket()
        self._esc_bucket = _LeakyBucket()
        self._last_info = None

    @staticmethod
    def _advance_bucket(
        bucket: _LeakyBucket, time: float, peak_duration: float
    ) -> tuple[float, bool, float]:
        """Evaluates a bucket at a time, recording an expiry of its allowance if one has
        happened by then.

        VALIDATED CORRECT

        :param bucket: The bucket, which is updated only if its allowance expired.
        :param time: The time (s).
        :param peak_duration: The limiter's peak duration (s).
        :return: A tuple of the allowance used at the time (s), whether the limiter is
            locked to its continuous limit, and the time actually used (s), which is
            never earlier than the bucket's reference time.
        """
        # This method doesn't integrate anything. The bucket stores a straight line (a
        # level at a reference time and a slope of +1.0 or -1.0), so the level at any
        # time is read off that line. This makes the result depend only on the time
        # asked for, not on how many calls came before it or how far apart they were,
        # which is what lets an adaptive ODE solver call it at any time.

        # On the first call ever, anchor the line at this time. The bucket starts empty
        # and draining, so the level is 0.0 here.
        if bucket.time_reference is None:
            bucket.time_reference = time

        # An ODE solver can ask for a time before the bucket's reference time (a
        # rejected step that is retried with a smaller step). The bucket can't be
        # rewound, because the history before its reference time is gone, so clamp to
        # the reference time. A repeated or earlier time then returns the state at the
        # reference time and never changes the bucket.
        effective_time = max(time, bucket.time_reference)

        # Read the level off the line. rate is +1.0 while the limiter is above its
        # continuous limit (one second of allowance used per second) and -1.0 while it
        # is at or below (one second recovered per second). The level can't go below
        # empty, so clamp at 0.0.
        level = max(
            0.0,
            bucket.level_reference
            + bucket.rate * (effective_time - bucket.time_reference),
        )

        # If the bucket is filling and has passed its capacity (the peak duration), the
        # allowance ran out somewhere between the reference time and now. Find that time
        # exactly and switch the line to draining from it, instead of declaring the
        # expiry at the current time, which would depend on the step size.
        if bucket.rate > 0.0 and level > peak_duration:
            # The allowance ran out at an exact time: when the excursion started plus
            # the allowance that was left. From then on the limiter is held to its
            # continuous limit, so the bucket drains.
            expire_time = bucket.time_reference + (
                peak_duration - bucket.level_reference
            )

            # The limiter stays locked to its continuous limit for half the peak
            # duration after the expiry, as in the blade element theory controller. This
            # is a cool down: a limiter that just ran out can't immediately return to
            # its peak limit as soon as the level drops below capacity.
            bucket.lock_end = expire_time + peak_duration / 2.0

            # Re-anchor the line at the expiry: full, and draining from there. This
            # write is the only way this method changes the bucket, and it records an
            # event that has already happened, so a later call at any time sees the same
            # line.
            bucket.level_reference = peak_duration
            bucket.time_reference = expire_time
            bucket.rate = -1.0

            # Re-read the level at the effective time from the new, draining line.
            level = max(0.0, peak_duration - (effective_time - expire_time))

        # The limiter is locked while the effective time is still inside the cool down
        # window. lock_end is -inf until the first expiry, so a bucket that has never
        # expired is never locked. The effective time is returned so the caller can
        # re-anchor the bucket at the same time when it sets the next regime.
        return level, effective_time < bucket.lock_end, effective_time

    @staticmethod
    def _set_regime(
        bucket: _LeakyBucket, above_continuous: bool, level: float, time: float
    ) -> None:
        """Sets a bucket to fill or drain, re-anchoring it only when that changes.

        VALIDATED CORRECT

        :param bucket: The bucket.
        :param above_continuous: Whether the applied torque is above the limiter's
            continuous limit.
        :param level: The bucket's level at the time (s).
        :param time: The time (s).
        :return: None
        """
        rate = 1.0 if above_continuous else -1.0
        if rate != bucket.rate:
            bucket.level_reference = level
            bucket.time_reference = time
            bucket.rate = rate

    def clip_torque(
        self,
        torque_desired: float | int,
        motor_velocity: float | int,
        time: float | int,
    ) -> tuple[float, SaturationInfo]:
        """Limits a torque command and advances the peak allowances to a time.

        VALIDATED CORRECT

        :param torque_desired: The commanded torque (N*m).
        :param motor_velocity: The motor's velocity (rad/s).
        :param time: The simulation time (s), not a step size. The peak allowances are
            evaluated at this time.
        :return: A tuple of the torque that the motor applies (N*m) and a SaturationInfo
            describing what limited it.
        """
        # Convert to plain floats so that numpy scalars and ints behave the same in the
        # comparisons and in the returned SaturationInfo.
        torque = float(torque_desired)
        velocity = float(motor_velocity)
        time = float(time)

        # Read both peak allowances at this time before clipping anything. This tells
        # each limiter whether it is locked to its continuous limit, and it is also
        # where an allowance that ran out since the last call is recorded. The levels
        # and effective times are kept so that step 5 can re-anchor each bucket at the
        # same point. The motor and the motor controller have separate allowances with
        # separate peak durations, so they are tracked by separate buckets.
        motor_level, motor_locked, motor_time = self._advance_bucket(
            self._motor_bucket, time, self._motor_peak_duration
        )
        esc_level, esc_locked, esc_time = self._advance_bucket(
            self._esc_bucket, time, self._esc_peak_duration
        )

        # Each limit below only reduces the torque's magnitude and keeps its sign. The
        # limits are applied in order, so a limit is reported in active (with its value
        # in limits) only if it cut the torque further than the ones before it. The
        # final torque is the same as clipping to the smallest limit, but the order
        # decides which limit is reported as the one that bound.
        active: list[str] = []
        limits: list[float] = []

        # 1. Only driving torque is limited by the torque-speed curve. The torque is
        # braking when it opposes the motion (opposite signs). A zero torque or a zero
        # velocity counts as driving, so at stall the curve's stall torque applies.
        # Braking above the no-load speed is still possible, which is why the curve,
        # which is 0.0 there, must not be applied to it.
        is_braking = torque * velocity < 0.0
        speed_limit = self.max_torque_from_speed(velocity)
        if not is_braking and abs(torque) > speed_limit:
            active.append("motor_torque_speed_curve")
            limits.append(speed_limit)
            torque = float(np.sign(torque)) * speed_limit

        # 1b. Braking torque is limited by the brake current instead, which models how
        # much regenerated current the drive can absorb. With no brake current limit,
        # max_brake_torque returns inf and this never binds.
        if is_braking:
            brake_limit = self.max_brake_torque()
            if abs(torque) > brake_limit:
                active.append("max_brake_current")
                limits.append(brake_limit)
                torque = float(np.sign(torque)) * brake_limit

        # 2. The gearbox limits every torque, driving or braking. Its limit is on the
        # output side, so it is divided by the gear ratio to give a motor torque.
        gearbox_limit = self.gearbox_torque_limit()
        if abs(torque) > gearbox_limit:
            active.append("gearbox_max_torque")
            limits.append(gearbox_limit)
            torque = float(np.sign(torque)) * gearbox_limit

        # 3. Once its peak allowance runs out, the motor is held to its continuous
        # torque. While it is not locked there is no motor torque limit here at all: the
        # motor's peak torque is bounded only by the curve, the gearbox, and the motor
        # controller's current.
        if motor_locked and abs(torque) > self._motor_continuous_torque:
            active.append("motor_peak_torque")
            limits.append(self._motor_continuous_torque)
            torque = float(np.sign(torque)) * self._motor_continuous_torque

        # 4. The motor controller is held to its continuous current once its peak
        # allowance runs out, and to its peak current otherwise. The current is
        # converted to a torque through the motor's torque-current polynomial. This
        # applies to driving and braking torque alike.
        esc_limit = self.torque_at_current(
            self._esc_continuous_current if esc_locked else self._esc_peak_current
        )
        if abs(torque) > esc_limit:
            # Name the current that actually bound, so a peak limit and a locked
            # controller's continuous limit can be told apart. (The blade element theory
            # controller reports "esc_peak_current" for both.)
            active.append(
                "esc_continuous_current" if esc_locked else "esc_peak_current"
            )
            limits.append(esc_limit)
            torque = float(np.sign(torque)) * esc_limit

        # 5. Fill or drain each bucket from the final applied torque, not the desired
        # one: the allowance is used only by torque that is actually applied. A bucket
        # fills while the applied torque is above that limiter's continuous limit and
        # drains otherwise. A bucket is only re-anchored when its regime changes, and
        # the line it stores is evaluated, never accumulated, so asking again at the
        # same time counts nothing twice.
        self._set_regime(
            self._motor_bucket,
            abs(torque) > self._motor_continuous_torque,
            motor_level,
            motor_time,
        )
        self._set_regime(
            self._esc_bucket,
            abs(torque) > self.torque_at_current(self._esc_continuous_current),
            esc_level,
            esc_time,
        )

        # Record what happened. max_torque_possible is the driving limit given the locks
        # at this time, and it ignores the brake current limit. The level never exceeds
        # the peak duration, because _advance_bucket turns an overfull bucket into an
        # expiry, so clamping the time remaining at 0.0 is only a guard.
        info = SaturationInfo(
            saturation_active=tuple(active),
            limits=tuple(limits),
            max_torque_possible=self.max_torque_possible(
                velocity, motor_locked, esc_locked
            ),
            motor_peak_time_elapsed=motor_level,
            esc_peak_time_elapsed=esc_level,
            motor_peak_time_remaining=max(0.0, self._motor_peak_duration - motor_level),
            esc_peak_time_remaining=max(0.0, self._esc_peak_duration - esc_level),
            motor_locked=motor_locked,
            esc_locked=esc_locked,
        )
        self._last_info = info
        return torque, info
