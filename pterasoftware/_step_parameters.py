"""Contains the records of the geometry values that change at each time step.

The movement classes evaluate each motion at a time step and return these records, and
their object builders build the WingCrossSections, Wings, and Airplanes from them. Only
the values that can change between time steps are here. Everything else, such as each
WingCrossSection's Airfoil and chord, and each Wing's panel counts and symmetry
settings, is fixed for a whole simulation and is read from the movement classes' base
geometry objects instead.

The arrays in every record are read only.

validate_wing_step_parameters and validate_airplane_step_parameters check a record's
values the way the object builders' constructors check them, raising the same errors,
for the code that uses the records without building the objects.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from . import _parameter_validation


class WingCrossSectionStepParameters(NamedTuple):
    """The values of a WingCrossSection that can change at each time step."""

    # A (3,) ndarray of floats representing the position of the WingCrossSection's
    # leading point (in wing cross section parent axes, relative to the leading point
    # parent). Its units are in meters.
    Lp_Wcsp_Lpp: np.ndarray

    # A (3,) ndarray of floats representing the angles from the wing cross section
    # parent axes to the wing cross section axes, using an intrinsic xy'z" sequence,
    # including any structural deformation. Its units are in degrees.
    angles_Wcsp_to_Wcs_ixyz: np.ndarray


class WingStepParameters(NamedTuple):
    """The values of a Wing, and of each of its WingCrossSections, that can change at
    each time step.

    For a Wing with type 5 symmetry, the Airplane replaces it with two Wings, the
    original and its reflection, and both are built from these same values. The symmetry
    type can't change between time steps, so it is fixed for a whole simulation.
    """

    # A (3,) ndarray of floats representing the position of the Wing's leading edge root
    # point (in geometry axes after accounting for symmetry, relative to the CG after
    # accounting for symmetry), after the adjustment for a rotation point offset. Its
    # units are in meters.
    Ler_Gs_Cgs: np.ndarray

    # A (3,) ndarray of floats representing the angles from the geometry axes after
    # accounting for symmetry to the wing axes, using an intrinsic xy'z" sequence. For a
    # flap driven Wing, the x component is the flap angle the drive reached. Its units
    # are in degrees.
    angles_Gs_to_Wn_ixyz: np.ndarray

    # A (K,3) ndarray of floats, with one row per WingCrossSection, each row being that
    # WingCrossSection's Lp_Wcsp_Lpp. Its units are in meters.
    stackLp_Wcsp_Lpp: np.ndarray

    # A (K,3) ndarray of floats, with one row per WingCrossSection, each row being that
    # WingCrossSection's angles_Wcsp_to_Wcs_ixyz, including any structural deformation.
    # Its units are in degrees.
    stackAngles_Wcsp_to_Wcs_ixyz: np.ndarray


class AirplaneStepParameters(NamedTuple):
    """The values of an Airplane, and of each of its Wings, that can change at each time
    step."""

    # A (3,) ndarray of floats representing the position of the Airplane's CG (in the
    # first Airplane's geometry axes, relative to the first Airplane's CG). Its units
    # are in meters.
    Cg_GP1_CgP1: np.ndarray

    # A tuple of WingStepParameters, one per Wing movement, in order. They describe the
    # Wings before symmetry processing, so a Wing with type 5 symmetry has one entry,
    # which both of the Airplane's resulting Wings share.
    wings: tuple[WingStepParameters, ...]


def stack_wing_step_parameters(
    Ler_Gs_Cgs: np.ndarray,
    angles_Gs_to_Wn_ixyz: np.ndarray,
    wing_cross_section_parameters: Sequence[WingCrossSectionStepParameters],
) -> WingStepParameters:
    """Builds a Wing's WingStepParameters from its own values and its WingCrossSections'
    WingCrossSectionStepParameters.

    :param Ler_Gs_Cgs: A (3,) ndarray of floats representing the position of the Wing's
        leading edge root point (in geometry axes after accounting for symmetry,
        relative to the CG after accounting for symmetry), after the adjustment for a
        rotation point offset. Its units are in meters. It is marked read only.
    :param angles_Gs_to_Wn_ixyz: A (3,) ndarray of floats representing the angles from
        the geometry axes after accounting for symmetry to the wing axes, using an
        intrinsic xy'z" sequence. Its units are in degrees. It is marked read only.
    :param wing_cross_section_parameters: A sequence of the
        WingCrossSectionStepParameters of each of the Wing's WingCrossSections, in
        order.
    :return: The Wing's WingStepParameters.
    """
    return WingStepParameters(
        Ler_Gs_Cgs=make_read_only(Ler_Gs_Cgs),
        angles_Gs_to_Wn_ixyz=make_read_only(angles_Gs_to_Wn_ixyz),
        stackLp_Wcsp_Lpp=make_read_only(
            np.stack(
                [parameters.Lp_Wcsp_Lpp for parameters in wing_cross_section_parameters]
            ).astype(float, copy=False)
        ),
        stackAngles_Wcsp_to_Wcs_ixyz=make_read_only(
            np.stack(
                [
                    parameters.angles_Wcsp_to_Wcs_ixyz
                    for parameters in wing_cross_section_parameters
                ]
            ).astype(float, copy=False)
        ),
    )


def make_read_only(array: np.ndarray) -> np.ndarray:
    """Marks an ndarray as read only and returns it.

    :param array: The ndarray to mark as read only.
    :return: The same ndarray, now read only.
    """
    array.flags.writeable = False
    return array


def validate_wing_step_parameters(parameters: WingStepParameters) -> None:
    """Checks a Wing's values at one time step the way building its WingCrossSections
    and Wing on the object path checks them, and raises the same error, with the same
    type and message, that the object path raises first.

    The object path validates each time step's values in the constructors of the objects
    it builds, and the array path builds no objects, so it calls this instead. It checks
    that every value is finite, that each WingCrossSection's Lp_Wcsp_Lpp[1] is non
    negative and each component of its angles_Wcsp_to_Wcs_ixyz is in [-90.0, 90.0]
    degrees (the WingCrossSection's constructor), that the root WingCrossSection's
    Lp_Wcsp_Lpp and angles_Wcsp_to_Wcs_ixyz are zero vectors and every other
    WingCrossSection's Lp_Wcsp_Lpp[1] is positive, and that each component of the Wing's
    angles_Gs_to_Wn_ixyz is in [-90.0, 90.0] degrees (the Wing's constructor). The
    values fixed for a whole simulation (chords, Airfoils, panel counts, spacings, and
    control surfaces) are validated when the base objects are built.

    :param parameters: The Wing's WingStepParameters.
    :return: None
    """
    stackLp_Wcsp_Lpp = parameters.stackLp_Wcsp_Lpp
    stackAngles_Wcsp_to_Wcs_ixyz = parameters.stackAngles_Wcsp_to_Wcs_ixyz
    angles_Gs_to_Wn_ixyz = parameters.angles_Gs_to_Wn_ixyz

    # Check every value at once, and only when something fails, repeat the object path's
    # checks one at a time, in its order, to raise its first error. For finite values,
    # np.allclose(a, 0.0) (the root constraints) is np.abs(a) <= 1.0e-8, its default
    # absolute tolerance, which is much faster.
    is_valid = bool(
        np.isfinite(stackLp_Wcsp_Lpp).all()
        and np.isfinite(stackAngles_Wcsp_to_Wcs_ixyz).all()
        and np.isfinite(parameters.Ler_Gs_Cgs).all()
        and np.isfinite(angles_Gs_to_Wn_ixyz).all()
        and np.all(np.abs(stackAngles_Wcsp_to_Wcs_ixyz) <= 90.0)
        and np.all(np.abs(angles_Gs_to_Wn_ixyz) <= 90.0)
        and np.all(np.abs(stackLp_Wcsp_Lpp[0]) <= 1.0e-8)
        and np.all(np.abs(stackAngles_Wcsp_to_Wcs_ixyz[0]) <= 1.0e-8)
        and stackLp_Wcsp_Lpp[0, 1] >= 0.0
        and np.all(stackLp_Wcsp_Lpp[1:, 1] > 0.0)
    )
    if not is_valid:
        _raise_wing_step_parameters_error(parameters)


def validate_airplane_step_parameters(parameters: AirplaneStepParameters) -> None:
    """Checks an Airplane's values at one time step the way building its Wings and the
    Airplane on the object path checks them, and raises the same error the object path
    raises first.

    The object path builds each Wing in order, then the Airplane, whose constructor
    checks that Cg_GP1_CgP1 is finite.

    :param parameters: The Airplane's AirplaneStepParameters.
    :return: None
    """
    for wing_parameters in parameters.wings:
        validate_wing_step_parameters(wing_parameters)
    if not np.isfinite(parameters.Cg_GP1_CgP1).all():
        _parameter_validation.threeD_number_vectorLike_return_float(
            parameters.Cg_GP1_CgP1, "Cg_GP1_CgP1"
        )


def _raise_wing_step_parameters_error(
    parameters: WingStepParameters,
) -> None:
    """Repeats the object path's checks of a Wing's values at one geometry build, in its
    order, and raises the first error they find.

    The movements' _wing_from_parameters builds every WingCrossSection, then the Wing,
    so this checks each WingCrossSection's constructor values in order, then the Wing's
    WingCrossSection constraints, then the Wing's own values, with the same validation
    functions and error messages.

    :param parameters: The Wing's WingStepParameters, which
        validate_wing_step_parameters found invalid.
    :return: None
    """
    # Each WingCrossSection's constructor.
    for Lp_Wcsp_Lpp, angles_Wcsp_to_Wcs_ixyz in zip(
        parameters.stackLp_Wcsp_Lpp, parameters.stackAngles_Wcsp_to_Wcs_ixyz
    ):
        validLp_Wcsp_Lpp = _parameter_validation.threeD_number_vectorLike_return_float(
            Lp_Wcsp_Lpp, "Lp_Wcsp_Lpp"
        )
        _parameter_validation.number_in_range_return_float(
            validLp_Wcsp_Lpp[1], "Lp_Wcsp_Lpp[1]", min_val=0.0, min_inclusive=True
        )
        validAngles_Wcsp_to_Wcs_ixyz = (
            _parameter_validation.threeD_number_vectorLike_return_float(
                angles_Wcsp_to_Wcs_ixyz, "angles_Wcsp_to_Wcs_ixyz"
            )
        )
        for angle_id, angle in enumerate(validAngles_Wcsp_to_Wcs_ixyz):
            _parameter_validation.number_in_range_return_float(
                angle, f"angles_Wcsp_to_Wcs_ixyz[{angle_id}]", -90.0, True, 90.0, True
            )

    # The Wing's constructor's root, middle, and tip WingCrossSection constraints, with
    # the messages of WingCrossSection.validate_root_constraints,
    # validate_mid_constraints, and validate_tip_constraints.
    for index, Lp_Wcsp_Lpp in enumerate(parameters.stackLp_Wcsp_Lpp):
        if index == 0:
            if not np.allclose(Lp_Wcsp_Lpp, np.array([0.0, 0.0, 0.0])):
                raise ValueError(
                    "The root WingCrossSection's Lp_Wcsp_Lpp must be np.array([0.0, "
                    "0.0, 0.0])."
                )
            if not np.allclose(
                parameters.stackAngles_Wcsp_to_Wcs_ixyz[0], np.array([0.0, 0.0, 0.0])
            ):
                raise ValueError(
                    "The root WingCrossSection's angles_Wcsp_to_Wcs_ixyz must be "
                    "np.array([0.0, 0.0, 0.0])."
                )
        elif Lp_Wcsp_Lpp[1] <= 0.0:
            raise ValueError(
                "Non root WingCrossSections must have a strictly positive second "
                "component of Lp_Wcsp_Lpp. To create a vertical surface, define "
                "the WingCrossSections with spanwise (y) offsets and rotate the "
                "Wing using angles_Gs_to_Wn_ixyz (e.g., (90.0, 0.0, 0.0) for a "
                "vertical tail). To create a winglet, rotate the previous "
                "WingCrossSection's axes using angles_Wcsp_to_Wcs_ixyz."
            )

    # The Wing's constructor's own values.
    _parameter_validation.threeD_number_vectorLike_return_float(
        parameters.Ler_Gs_Cgs, "Ler_Gs_Cgs"
    )
    angles_Gs_to_Wn_ixyz = _parameter_validation.threeD_number_vectorLike_return_float(
        parameters.angles_Gs_to_Wn_ixyz, "angles_Gs_to_Wn_ixyz"
    )
    if not np.all((-90.0 <= angles_Gs_to_Wn_ixyz) & (angles_Gs_to_Wn_ixyz <= 90.0)):
        raise ValueError(
            "All elements of angles_Gs_to_Wn_ixyz must lie in the range [-90, 90] "
            "degrees."
        )

    # The fast check in validate_wing_step_parameters is equivalent to these, so one of
    # them always raises.
    raise AssertionError(
        "validate_wing_step_parameters found invalid values that the object path's "
        "checks accept."
    )
