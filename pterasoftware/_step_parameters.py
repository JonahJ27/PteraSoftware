"""Contains the records of the geometry values that change at each time step.

The movement classes evaluate each motion at a time step and return these records, and
their object builders build the WingCrossSections, Wings, and Airplanes from them. Only
the values that can change between time steps are here. Everything else, such as each
WingCrossSection's Airfoil and chord, and each Wing's panel counts and symmetry
settings, is fixed for a whole simulation and is read from the movement classes' base
geometry objects instead.

The arrays in every record are read only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np


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
