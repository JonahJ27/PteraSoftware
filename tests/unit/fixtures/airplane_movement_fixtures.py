"""This module contains functions to create AirplaneMovements for use in tests."""

import pterasoftware as ps

from . import geometry_fixtures, wing_movement_fixtures


def make_static_airplane_movement_fixture() -> (
    ps.movements.airplane_movement.AirplaneMovement
):
    """This method makes a fixture that is an AirplaneMovement with all parameters zero
    (no movement).

    :return static_airplane_movement_fixture: AirplaneMovement This is the
        AirplaneMovement with no movement.
    """
    # Initialize the constructing fixtures. Build the base Airplane first, then build
    # the WingMovement around its own Wing.
    base_airplane = geometry_fixtures.make_origin_airplane_fixture()
    wing_movements = [
        wing_movement_fixtures.make_static_wing_movement_fixture(base_airplane.wings[0])
    ]

    # Create the static AirplaneMovement.
    static_airplane_movement_fixture = ps.movements.airplane_movement.AirplaneMovement(
        base_airplane=base_airplane,
        wing_movements=wing_movements,
        ampCg_GP1_CgP1=(0.0, 0.0, 0.0),
        periodCg_GP1_CgP1=(0.0, 0.0, 0.0),
        spacingCg_GP1_CgP1=("sine", "sine", "sine"),
        phaseCg_GP1_CgP1=(0.0, 0.0, 0.0),
    )

    # Return the AirplaneMovement fixture.
    return static_airplane_movement_fixture


def make_basic_airplane_movement_fixture() -> (
    ps.movements.airplane_movement.AirplaneMovement
):
    """This method makes a fixture that is an AirplaneMovement with general-purpose
    moderate values.

    :return basic_airplane_movement_fixture: AirplaneMovement This is the
        AirplaneMovement with general-purpose values.
    """
    # Initialize the constructing fixtures. Build the base Airplane first, then build
    # the WingMovement around its own Wing.
    base_airplane = geometry_fixtures.make_origin_airplane_fixture()
    wing_movements = [
        wing_movement_fixtures.make_basic_wing_movement_fixture(base_airplane.wings[0])
    ]

    # Create the basic AirplaneMovement.
    basic_airplane_movement_fixture = ps.movements.airplane_movement.AirplaneMovement(
        base_airplane=base_airplane,
        wing_movements=wing_movements,
        ampCg_GP1_CgP1=(0.0, 0.0, 0.0),
        periodCg_GP1_CgP1=(0.0, 0.0, 0.0),
        spacingCg_GP1_CgP1=("sine", "sine", "sine"),
        phaseCg_GP1_CgP1=(0.0, 0.0, 0.0),
    )

    # Return the AirplaneMovement fixture.
    return basic_airplane_movement_fixture


def make_periodic_geometry_airplane_movement_fixture() -> (
    ps.movements.airplane_movement.AirplaneMovement
):
    """This method makes a fixture that is an AirplaneMovement with periodic geometry
    motion suitable for testing the variable geometry optimization.

    The fixture uses a 0.1s period which aligns well with common delta_time values like
    0.01s (10 steps per period) and 0.02s (5 steps per period).

    :return periodic_geometry_airplane_movement_fixture: AirplaneMovement This is the
        AirplaneMovement with periodic geometry motion.
    """
    # Initialize the constructing fixtures. Build the base Airplane first, then build
    # the WingMovement around its own Wing.
    base_airplane = geometry_fixtures.make_origin_airplane_fixture()
    wing_movements = [
        wing_movement_fixtures.make_periodic_geometry_wing_movement_fixture(
            base_airplane.wings[0]
        )
    ]

    # Create the periodic geometry AirplaneMovement.
    periodic_geometry_airplane_movement_fixture = (
        ps.movements.airplane_movement.AirplaneMovement(
            base_airplane=base_airplane,
            wing_movements=wing_movements,
            ampCg_GP1_CgP1=(0.0, 0.0, 0.0),
            periodCg_GP1_CgP1=(0.0, 0.0, 0.0),
            spacingCg_GP1_CgP1=("sine", "sine", "sine"),
            phaseCg_GP1_CgP1=(0.0, 0.0, 0.0),
        )
    )

    # Return the AirplaneMovement fixture.
    return periodic_geometry_airplane_movement_fixture


def make_out_of_range_airplane_movement_fixture(
    ampAngleY_Gs_to_Wn: float = 0.0,
    ampTipAngleY_Wcsp_to_Wcs: float = 0.0,
) -> ps.movements.airplane_movement.AirplaneMovement:
    """This method makes a fixture that is an AirplaneMovement whose only Wing's y
    angle, and whose tip WingCrossSection's y angle, oscillate with given amplitudes
    from zero, with a period of 0.4 s, so an amplitude above 90.0 degrees is valid at
    time step 0 but out of range a quarter period later.

    :param ampAngleY_Gs_to_Wn: The amplitude of the Wing's angles_Gs_to_Wn_ixyz y
        component (degrees). The default is 0.0.
    :param ampTipAngleY_Wcsp_to_Wcs: The amplitude of the tip WingCrossSection's
        angles_Wcsp_to_Wcs_ixyz y component (degrees). The default is 0.0.
    :return out_of_range_airplane_movement_fixture: AirplaneMovement This is the
        AirplaneMovement, for testing that motion that is out of range at a later time
        step is rejected.
    """
    base_airplane = ps.geometry.airplane.Airplane(
        wings=[geometry_fixtures.make_origin_wing_fixture()],
        name="Origin Airplane",
        Cg_GP1_CgP1=(0.0, 0.0, 0.0),
        weight=0.0,
    )
    root_wing_cross_section, tip_wing_cross_section = base_airplane.wings[
        0
    ].wing_cross_sections
    wing_movement = ps.movements.wing_movement.WingMovement(
        base_wing=base_airplane.wings[0],
        wing_cross_section_movements=[
            ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
                base_wing_cross_section=root_wing_cross_section,
            ),
            ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
                base_wing_cross_section=tip_wing_cross_section,
                ampAngles_Wcsp_to_Wcs_ixyz=(0.0, ampTipAngleY_Wcsp_to_Wcs, 0.0),
                periodAngles_Wcsp_to_Wcs_ixyz=(
                    0.0,
                    0.4 if ampTipAngleY_Wcsp_to_Wcs else 0.0,
                    0.0,
                ),
            ),
        ],
        ampAngles_Gs_to_Wn_ixyz=(0.0, ampAngleY_Gs_to_Wn, 0.0),
        periodAngles_Gs_to_Wn_ixyz=(0.0, 0.4 if ampAngleY_Gs_to_Wn else 0.0, 0.0),
    )
    out_of_range_airplane_movement_fixture = (
        ps.movements.airplane_movement.AirplaneMovement(
            base_airplane=base_airplane,
            wing_movements=[wing_movement],
        )
    )
    return out_of_range_airplane_movement_fixture
