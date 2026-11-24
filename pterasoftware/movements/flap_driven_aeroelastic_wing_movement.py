"""Contains the FlapDrivenAeroelasticWingMovement class."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

import numpy as np

from .. import _step_parameters, geometry
from . import (
    aeroelastic_wing_cross_section_movement as aeroelastic_wing_cross_section_movement_mod,
)
from . import aeroelastic_wing_movement as aeroelastic_wing_movement_mod


def _yz_components(values: object, name: str) -> tuple:
    """Returns the y and z components of an angular motion parameter as a tuple.

    VALIDATED CORRECT

    The values themselves are validated by AeroelasticWingMovement once the x component
    is added, so this only checks that there are exactly two of them.

    :param values: The parameter's value, which must be an array-like object of length
        2.
    :param name: The parameter's name, for the error message.
    :return: A tuple of the two components, y first.
    """
    if isinstance(values, str):
        raise TypeError(f"{name} must be an array-like object of length 2.")
    try:
        components = tuple(cast(Sequence, values))
    except TypeError:
        raise TypeError(f"{name} must be an array-like object of length 2.") from None
    if len(components) != 2:
        raise ValueError(
            f"{name} must have exactly 2 components, the y and z components. There is "
            f"no x (flap) component, because the FlapDrive's FlapReference defines the "
            f"flap trajectory."
        )
    return components


class FlapDrivenAeroelasticWingMovement(
    aeroelastic_wing_movement_mod.AeroelasticWingMovement
):
    """A subclass of AeroelasticWingMovement used for Wings whose flap angle comes from
    a FlapDrive.

    Inherits all methods from AeroelasticWingMovement, but it can't prescribe a flap
    motion, because the FlapDrive's FlapReference is the only definition of the flap
    trajectory. To make that impossible to get wrong, its angular oscillation parameters
    take only the y and z components (ampAnglesYZ_Gs_to_Wn_ixyz,
    periodAnglesYZ_Gs_to_Wn_ixyz, spacingAnglesYZ_Gs_to_Wn_ixyz,
    phaseAnglesYZ_Gs_to_Wn_ixyz, and spacingAnglesSecondDerivativeYZ_Gs_to_Wn_ixyz)
    instead of AeroelasticWingMovement's three. There is no x (flap) component to set.
    The initialization method puts a static x component (an amplitude, period, and phase
    of 0.0, with "sine" spacing) in front of them before passing them to
    AeroelasticWingMovement, so the inherited three component attributes, such as
    ampAngles_Gs_to_Wn_ixyz, always have 0.0 as their x component. The other motions
    (the leading edge position, the WingCrossSections' motion, and the rotation point
    offset) work as they do in AeroelasticWingMovement.

    A FlapDrivenAeroelasticWingMovement holds the flap state of one simulation: the flap
    angle at each time step so far, the Wing's flap angular acceleration over the latest
    time step, and the period of the FlapDrive's reference. A
    FlapDrivenAeroelasticUnsteadyProblem writes this state. It records the first time
    step's flap angle before any Wing is generated, and each later one as it advances
    the simulation. A time step with no recorded flap angle uses the base Wing's flap
    angle. The period is added to this movement's periods, so that the cycle averaged
    results cover whole wingbeats. Because the state belongs to one simulation, build a
    new movement for each problem.
    """

    __slots__ = (
        "_listOverrideAnglesX_Gs_to_Wn_ixyz",
        "_angleSecondDerivativeXRad_function",
        "_flap_period",
    )

    def __init__(
        self,
        base_wing: geometry.wing.Wing,
        wing_cross_section_movements: list[
            aeroelastic_wing_cross_section_movement_mod.AeroelasticWingCrossSectionMovement
        ],
        ampLer_Gs_Cgs: np.ndarray | Sequence[float | int] = (0.0, 0.0, 0.0),
        periodLer_Gs_Cgs: np.ndarray | Sequence[float | int] = (0.0, 0.0, 0.0),
        spacingLer_Gs_Cgs: np.ndarray | Sequence[str | Callable[[float], float]] = (
            "sine",
            "sine",
            "sine",
        ),
        phaseLer_Gs_Cgs: np.ndarray | Sequence[float | int] = (0.0, 0.0, 0.0),
        ampAnglesYZ_Gs_to_Wn_ixyz: np.ndarray | Sequence[float | int] = (0.0, 0.0),
        periodAnglesYZ_Gs_to_Wn_ixyz: np.ndarray | Sequence[float | int] = (0.0, 0.0),
        spacingAnglesYZ_Gs_to_Wn_ixyz: (
            np.ndarray | Sequence[str | Callable[[float], float]]
        ) = (
            "sine",
            "sine",
        ),
        phaseAnglesYZ_Gs_to_Wn_ixyz: np.ndarray | Sequence[float | int] = (0.0, 0.0),
        rotationPointOffset_Gs_Ler: np.ndarray | Sequence[float | int] = (
            0.0,
            0.0,
            0.0,
        ),
        spacingAnglesSecondDerivativeYZ_Gs_to_Wn_ixyz: (
            Sequence[Callable[[float], float] | None] | None
        ) = None,
    ) -> None:
        """The initialization method.

        See AeroelasticWingMovement's initialization method for descriptions of the
        parameters that share its names. The angular oscillation parameters differ: each
        takes only the y and z components of its AeroelasticWingMovement counterpart,
        because this class can't prescribe a flap motion (the x component). The
        FlapDrive's FlapReference is the only definition of the flap trajectory.

        :param base_wing: The base Wing. See AeroelasticWingMovement.
        :param wing_cross_section_movements: The AeroelasticWingCrossSectionMovements.
            See AeroelasticWingMovement.
        :param ampLer_Gs_Cgs: See AeroelasticWingMovement.
        :param periodLer_Gs_Cgs: See AeroelasticWingMovement.
        :param spacingLer_Gs_Cgs: See AeroelasticWingMovement.
        :param phaseLer_Gs_Cgs: See AeroelasticWingMovement.
        :param ampAnglesYZ_Gs_to_Wn_ixyz: An array-like object of 2 non negative numbers
            (int or float), the y and z components of AeroelasticWingMovement's
            ampAngles_Gs_to_Wn_ixyz. The units are in degrees. The default is (0.0,
            0.0).
        :param periodAnglesYZ_Gs_to_Wn_ixyz: An array-like object of 2 non negative
            numbers (int or float), the y and z components of AeroelasticWingMovement's
            periodAngles_Gs_to_Wn_ixyz. The units are in seconds. The default is (0.0,
            0.0).
        :param spacingAnglesYZ_Gs_to_Wn_ixyz: An array-like object of 2 strs or
            callables, the y and z components of AeroelasticWingMovement's
            spacingAngles_Gs_to_Wn_ixyz. The default is ("sine", "sine").
        :param phaseAnglesYZ_Gs_to_Wn_ixyz: An array-like object of 2 numbers (int or
            float), the y and z components of AeroelasticWingMovement's
            phaseAngles_Gs_to_Wn_ixyz. The units are in degrees. The default is (0.0,
            0.0).
        :param rotationPointOffset_Gs_Ler: See AeroelasticWingMovement.
        :param spacingAnglesSecondDerivativeYZ_Gs_to_Wn_ixyz: A sequence of 2 callables
            or Nones, the y and z components of AeroelasticWingMovement's
            spacingAnglesSecondDerivative_Gs_to_Wn_ixyz, or None. The default is None.
        :return: None
        """
        # There is deliberately no parameter for the x (flap) component of the angular
        # motion. The FlapDrive's FlapReference is the only definition of the flap
        # trajectory, so a prescribed one here would be a second definition that could
        # disagree with it. Each parameter below takes only the y and z components.
        ampAnglesYZ_Gs_to_Wn_ixyz = _yz_components(
            ampAnglesYZ_Gs_to_Wn_ixyz, "ampAnglesYZ_Gs_to_Wn_ixyz"
        )
        periodAnglesYZ_Gs_to_Wn_ixyz = _yz_components(
            periodAnglesYZ_Gs_to_Wn_ixyz, "periodAnglesYZ_Gs_to_Wn_ixyz"
        )
        spacingAnglesYZ_Gs_to_Wn_ixyz = _yz_components(
            spacingAnglesYZ_Gs_to_Wn_ixyz, "spacingAnglesYZ_Gs_to_Wn_ixyz"
        )
        phaseAnglesYZ_Gs_to_Wn_ixyz = _yz_components(
            phaseAnglesYZ_Gs_to_Wn_ixyz, "phaseAnglesYZ_Gs_to_Wn_ixyz"
        )
        spacingAnglesSecondDerivative_Gs_to_Wn_ixyz = None
        if spacingAnglesSecondDerivativeYZ_Gs_to_Wn_ixyz is not None:
            spacingAnglesSecondDerivative_Gs_to_Wn_ixyz = (
                None,
                *_yz_components(
                    spacingAnglesSecondDerivativeYZ_Gs_to_Wn_ixyz,
                    "spacingAnglesSecondDerivativeYZ_Gs_to_Wn_ixyz",
                ),
            )

        # AeroelasticWingMovement takes three components, so put a static x component in
        # front of the y and z ones: an amplitude, period, and phase of 0.0 and "sine"
        # spacing. With an amplitude of 0.0 the base Wing's flap angle never oscillates,
        # and the FlapDrivenAeroelasticUnsteadyProblem replaces it at every time step
        # with the FlapDrive's angle through the override.
        super().__init__(
            base_wing=base_wing,
            wing_cross_section_movements=wing_cross_section_movements,
            ampLer_Gs_Cgs=ampLer_Gs_Cgs,
            periodLer_Gs_Cgs=periodLer_Gs_Cgs,
            spacingLer_Gs_Cgs=spacingLer_Gs_Cgs,
            phaseLer_Gs_Cgs=phaseLer_Gs_Cgs,
            ampAngles_Gs_to_Wn_ixyz=(0.0, *ampAnglesYZ_Gs_to_Wn_ixyz),
            periodAngles_Gs_to_Wn_ixyz=(0.0, *periodAnglesYZ_Gs_to_Wn_ixyz),
            spacingAngles_Gs_to_Wn_ixyz=("sine", *spacingAnglesYZ_Gs_to_Wn_ixyz),
            phaseAngles_Gs_to_Wn_ixyz=(0.0, *phaseAnglesYZ_Gs_to_Wn_ixyz),
            rotationPointOffset_Gs_Ler=rotationPointOffset_Gs_Ler,
            spacingAnglesSecondDerivative_Gs_to_Wn_ixyz=spacingAnglesSecondDerivative_Gs_to_Wn_ixyz,
        )

        # This is the flap angle (degrees) at each time step that a FlapDrive has
        # driven, indexed by time step. These are the x components of the Wing's
        # angles_Gs_to_Wn_ixyz (angles describing the orientation of the wing axes
        # relative to the geometry axes, after accounting for symmetry, using an
        # intrinsic xy'z" sequence). It is empty until the problem records step 0. The
        # problem fills it with record_flap_step, and generate_wing_at_time_step reads
        # it (see that method for the full path).
        self._listOverrideAnglesX_Gs_to_Wn_ixyz: list[float] = []

        # This is the Wing's flap angular acceleration (rad/s^2) as a function of time
        # over the latest recorded time step's interval, or None before any is recorded.
        # Unlike the angles, only the latest one is kept, because the problem's
        # _generate_inertial_moment_function reads it for the torsional springs right
        # after it is recorded, while it builds the next time step.
        self._angleSecondDerivativeXRad_function: Callable[[float], float] | None = None

        # This is the period (s) of the FlapDrive's reference, which the problem sets,
        # or None if it has set none or the reference isn't periodic.
        self._flap_period: float | None = None

    @property
    def flap_period(self) -> float | None:
        """The period of the flap trajectory that drives this movement's Wing.

        :return: The period (s) of the FlapDrive's reference, which the
            FlapDrivenAeroelasticUnsteadyProblem sets, or None if it hasn't set one or
            the reference isn't periodic.
        """
        return self._flap_period

    @property
    def all_periods(self) -> tuple[float, ...]:
        """All unique non zero periods of motion from this Wing and each
        WingCrossSection's movement class, including the flap period.

        Extends the parent property with the flap period, which is not cached, so that
        it reflects the latest one the problem set.

        :return: A tuple of all unique non zero periods in seconds. If all motion is
            static, this will be an empty tuple.
        """
        periods = super().all_periods
        if self._flap_period is not None:
            periods = periods + (self._flap_period,)
        return periods

    @property
    def max_period(self) -> float:
        """The longest period of motion across this Wing and each WingCrossSection's
        movement class, including the flap period.

        :return: The longest period in seconds. If all the motion is static, this will
            be 0.0.
        """
        if self._flap_period is None:
            return super().max_period
        return max(super().max_period, self._flap_period)

    def set_flap_period(self, flap_period: float | int | None) -> None:
        """Sets the period of the flap trajectory that drives this movement's Wing.

        The FlapDrivenAeroelasticUnsteadyProblem calls this with its FlapDrive's
        reference's period. The parent movements cache their periods, so the caller must
        clear those caches afterwards.

        :param flap_period: The period (s), or None if the reference isn't periodic. If
            not None, it must be a positive number (int or float).
        :return: None
        """
        if flap_period is not None:
            if not isinstance(flap_period, (int, float)) or not (
                np.isfinite(flap_period) and flap_period > 0.0
            ):
                raise ValueError("flap_period must be a positive number or None.")
            flap_period = float(flap_period)
        self._flap_period = flap_period

    @classmethod
    def from_aeroelastic_wing_movement(
        cls,
        wing_movement: aeroelastic_wing_movement_mod.AeroelasticWingMovement,
    ) -> FlapDrivenAeroelasticWingMovement:
        """Returns a FlapDrivenAeroelasticWingMovement with the same parameters as an
        existing AeroelasticWingMovement, except for its flap motion.

        VALIDATED CORRECT

        It lets an existing aeroelastic simulation setup gain a FlapDrive by swapping
        its AeroelasticWingMovements for these. The existing movement's flap motion (the
        x components of its angular oscillation parameters) is discarded, because the
        FlapDrive's FlapReference defines the flap trajectory. Every other parameter is
        copied. The new movement shares the existing movement's base Wing and its
        WingCrossSection movements, which are immutable.

        :param wing_movement: The AeroelasticWingMovement to copy the parameters from.
        :return: A FlapDrivenAeroelasticWingMovement with the same parameters, no flap
            motion, and an empty flap state.
        """
        if not isinstance(
            wing_movement, aeroelastic_wing_movement_mod.AeroelasticWingMovement
        ):
            raise TypeError("wing_movement must be an AeroelasticWingMovement.")

        # Keep only the y and z components (indices 1 and 2) of the angular motion. The
        # x (flap) component is discarded, because this class has no parameter for it.
        spacingAnglesSecondDerivativeYZ_Gs_to_Wn_ixyz = None
        if wing_movement.spacingAnglesSecondDerivative_Gs_to_Wn_ixyz is not None:
            spacingAnglesSecondDerivativeYZ_Gs_to_Wn_ixyz = list(
                wing_movement.spacingAnglesSecondDerivative_Gs_to_Wn_ixyz
            )[1:]

        return cls(
            base_wing=wing_movement.base_wing,
            wing_cross_section_movements=cast(
                list[
                    aeroelastic_wing_cross_section_movement_mod.AeroelasticWingCrossSectionMovement
                ],
                list(wing_movement.wing_cross_section_movements),
            ),
            ampLer_Gs_Cgs=wing_movement.ampLer_Gs_Cgs,
            periodLer_Gs_Cgs=wing_movement.periodLer_Gs_Cgs,
            spacingLer_Gs_Cgs=wing_movement.spacingLer_Gs_Cgs,
            phaseLer_Gs_Cgs=wing_movement.phaseLer_Gs_Cgs,
            ampAnglesYZ_Gs_to_Wn_ixyz=wing_movement.ampAngles_Gs_to_Wn_ixyz[1:],
            periodAnglesYZ_Gs_to_Wn_ixyz=wing_movement.periodAngles_Gs_to_Wn_ixyz[1:],
            spacingAnglesYZ_Gs_to_Wn_ixyz=wing_movement.spacingAngles_Gs_to_Wn_ixyz[1:],
            phaseAnglesYZ_Gs_to_Wn_ixyz=wing_movement.phaseAngles_Gs_to_Wn_ixyz[1:],
            rotationPointOffset_Gs_Ler=wing_movement.rotationPointOffset_Gs_Ler,
            spacingAnglesSecondDerivativeYZ_Gs_to_Wn_ixyz=(
                spacingAnglesSecondDerivativeYZ_Gs_to_Wn_ixyz
            ),
        )

    def __deepcopy__(self, memo: dict) -> FlapDrivenAeroelasticWingMovement:
        """Creates a deep copy of this FlapDrivenAeroelasticWingMovement.

        Extends the parent deep copy to also copy the recorded flap state and the flap
        period.

        :param memo: A dict used by the copy module to track already copied objects and
            avoid infinite recursion.
        :return: A new instance with copied attributes.
        """
        new_movement = cast(
            FlapDrivenAeroelasticWingMovement, super().__deepcopy__(memo)
        )
        new_movement._flap_period = self._flap_period
        new_movement._listOverrideAnglesX_Gs_to_Wn_ixyz = list(
            self._listOverrideAnglesX_Gs_to_Wn_ixyz
        )
        # The function is shared, not copied. It only reads arrays that are never
        # modified after they are recorded.
        new_movement._angleSecondDerivativeXRad_function = (
            self._angleSecondDerivativeXRad_function
        )
        return new_movement

    @property
    def listOverrideAnglesX_Gs_to_Wn_ixyz(self) -> tuple[float, ...]:
        """The recorded flap angles, indexed by time step.

        :return: A tuple of floats holding the flap angle (the x component of the Wing's
            angles_Gs_to_Wn_ixyz) at each time step recorded so far. The units are in
            degrees.
        """
        return tuple(self._listOverrideAnglesX_Gs_to_Wn_ixyz)

    @property
    def angleSecondDerivativeXRad_function(self) -> Callable[[float], float] | None:
        """The Wing's flap angular acceleration over the latest recorded time step.

        :return: None if no time step has been recorded with an acceleration. Otherwise,
            a callable that accepts a time (s) and returns the flap angle's second time
            derivative (rad/s^2).
        """
        return self._angleSecondDerivativeXRad_function

    def reset_flap_state(self) -> None:
        """Clears the recorded flap angles and acceleration, and the flap period.

        :return: None
        """
        self._listOverrideAnglesX_Gs_to_Wn_ixyz = []
        self._angleSecondDerivativeXRad_function = None
        self._flap_period = None

    def record_flap_step(
        self,
        step: int,
        angleX_Gs_to_Wn_ixyz: float | int,
        angleSecondDerivativeXRad_function: Callable[[float], float] | None,
    ) -> None:
        """Records the flap angle at a time step, and the acceleration leading to it.

        Time steps must be recorded in order, starting at step 0.

        :param step: The time step index. It must equal the number of time steps
            recorded so far.
        :param angleX_Gs_to_Wn_ixyz: The flap angle (the x component of the Wing's
            angles_Gs_to_Wn_ixyz) at this time step. It must be a finite number (int or
            float) in the range [-90.0, 90.0], which is the range the Wing allows for
            its angles. The units are in degrees.
        :param angleSecondDerivativeXRad_function: A callable that accepts a time (s)
            and returns the flap angle's second time derivative (rad/s^2) over the
            interval that ends at this time step, or None if there is no such interval,
            as at step 0.
        :return: None
        """
        if step != len(self._listOverrideAnglesX_Gs_to_Wn_ixyz):
            raise ValueError(
                f"Time steps must be recorded in order. Expected step "
                f"{len(self._listOverrideAnglesX_Gs_to_Wn_ixyz)}, got {step}."
            )
        if not isinstance(angleX_Gs_to_Wn_ixyz, (int, float)) or not (
            -90.0 <= angleX_Gs_to_Wn_ixyz <= 90.0
        ):
            raise ValueError(
                "angleX_Gs_to_Wn_ixyz must be a finite number in the range [-90.0, "
                "90.0]."
            )
        if angleSecondDerivativeXRad_function is not None and not callable(
            angleSecondDerivativeXRad_function
        ):
            raise TypeError("angleSecondDerivativeXRad_function must be a callable.")

        self._listOverrideAnglesX_Gs_to_Wn_ixyz.append(float(angleX_Gs_to_Wn_ixyz))
        self._angleSecondDerivativeXRad_function = angleSecondDerivativeXRad_function

    def _generate_wing_parameters_at_time_step(
        self,
        step: int,
        delta_time: float | int,
        deformationAngles_Wcsp_to_Wcs_ixyz: np.ndarray | None = None,
        overrideAngleX_Gs_to_Wn_ixyz: float | int | None = None,
    ) -> _step_parameters.WingStepParameters:
        """Evaluates the motion of the Wing and of each of its WingCrossSections at a
        single time step, using the recorded flap angle if there is one.

        It exists so the recorded flap angle is applied to the values, which both paths
        use, instead of only to a built Wing.

        This is the first of the two halves of generate_wing_at_time_step. It only picks
        the flap angle, then has the parent class's version of this method evaluate the
        motion with it as the override, without building a Wing. The second half, the
        inherited _wing_from_parameters, builds the Wing from the returned values.

        The flap angle is resolved exactly as generate_wing_at_time_step resolves it: an
        explicit override wins, then the flap angle recorded for this time step, and the
        base Wing's flap angle is used only if neither exists.

        :param step: The time step index. Must be a non negative int.
        :param delta_time: The time between each time step in seconds. Must be a
            positive number (int or float).
        :param deformationAngles_Wcsp_to_Wcs_ixyz: An ndarray of floats with one row per
            WingCrossSection in this Wing, in degrees, or None. The default is None.
        :param overrideAngleX_Gs_to_Wn_ixyz: A number (int or float) in degrees that
            replaces the flap angle, or None. The default is None.
        :return: The WingStepParameters at this time step.
        """
        if overrideAngleX_Gs_to_Wn_ixyz is None and step < len(
            self._listOverrideAnglesX_Gs_to_Wn_ixyz
        ):
            overrideAngleX_Gs_to_Wn_ixyz = self._listOverrideAnglesX_Gs_to_Wn_ixyz[step]

        return super()._generate_wing_parameters_at_time_step(
            step,
            delta_time,
            deformationAngles_Wcsp_to_Wcs_ixyz=deformationAngles_Wcsp_to_Wcs_ixyz,
            overrideAngleX_Gs_to_Wn_ixyz=overrideAngleX_Gs_to_Wn_ixyz,
        )

    def generate_wing_at_time_step(
        self,
        step: int,
        delta_time: float | int,
        deformationAngles_Wcsp_to_Wcs_ixyz: np.ndarray | None = None,
        overrideAngleX_Gs_to_Wn_ixyz: float | int | None = None,
    ) -> geometry.wing.Wing:
        """Creates the Wing at a single time step, using the recorded flap angle if
        there is one.

        See AeroelasticWingMovement's generate_wing_at_time_step method for descriptions
        of the parameters. The only difference is that when overrideAngleX_Gs_to_Wn_ixyz
        is None, the flap angle recorded for this time step is used, and the base Wing's
        flap angle is used only if none has been recorded.

        :param step: The time step index. Must be a non negative int.
        :param delta_time: The time between each time step in seconds. Must be a
            positive number (int or float).
        :param deformationAngles_Wcsp_to_Wcs_ixyz: An ndarray of floats with one row per
            WingCrossSection in this Wing, in degrees, or None. The default is None.
        :param overrideAngleX_Gs_to_Wn_ixyz: A number (int or float) in degrees that
            replaces the flap angle, or None. The default is None.
        :return: The Wing at this time step.
        """
        # In normal operation, the flap angle reaches this method through the record,
        # not through overrideAngleX_Gs_to_Wn_ixyz:
        #
        # 1. Before the parent problem builds the first time step, the problem's
        #    _prepare_flap_drives records step 0 with record_flap_step, at the flap
        #    angle of the FlapDrive's initial actuator position.
        # 2. After the solver solves time step s, the problem's _advance_flap_drives
        #    integrates the FlapDrive across the interval to the next time step and
        #    records that step with record_flap_step, along with the flap acceleration
        #    over the interval.
        # 3. The parent problem's initialize_next_problem then builds step s + 1's
        #    Airplane through the unchanged movement chain (AeroelasticMovement, then
        #    AeroelasticAirplaneMovement, then this method). That chain passes
        #    deformationAngles_Wcsp_to_Wcs_ixyz but has no flap angle argument, so
        #    overrideAngleX_Gs_to_Wn_ixyz is None here, and the recorded angle for this
        #    step is used.
        #
        # So every step is recorded before it is generated. The explicit
        # overrideAngleX_Gs_to_Wn_ixyz parameter exists only to keep the parent's
        # signature, and if given, it wins over the record. The base Wing's flap angle
        # is used only for a step with no record, which happens only when no FlapDrive
        # owns this Wing, or when the geometry is built outside or ahead of a flap
        # driven problem (for example, FreeFlightAeroelasticMovement builds every
        # Airplane when it is constructed). That fallback is silent.
        #
        # The recorded angle is in degrees, like every angle in the geometry, while the
        # recorded acceleration is in rad/s^2, because it is only read by the problem's
        # _generate_inertial_moment_function to compute a moment for the torsional
        # springs. The problem's _advance_flap_drives converts the angle from radians.
        if overrideAngleX_Gs_to_Wn_ixyz is None and step < len(
            self._listOverrideAnglesX_Gs_to_Wn_ixyz
        ):
            overrideAngleX_Gs_to_Wn_ixyz = self._listOverrideAnglesX_Gs_to_Wn_ixyz[step]

        # The parent replaces the x component of the Wing's angles_Gs_to_Wn_ixyz with
        # the override, if there is one, before it applies the rotation point offset, so
        # the Wing flaps about its shoulder.
        return super().generate_wing_at_time_step(
            step,
            delta_time,
            deformationAngles_Wcsp_to_Wcs_ixyz=deformationAngles_Wcsp_to_Wcs_ixyz,
            overrideAngleX_Gs_to_Wn_ixyz=overrideAngleX_Gs_to_Wn_ixyz,
        )
