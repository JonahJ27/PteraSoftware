"""This module creates movement objects to be used as fixtures."""

import pterasoftware as ps
from tests.integration.fixtures import airplane_fixtures, operating_point_fixtures

# These are the flapping free flight aeroelastic fixtures' time step (s) and their
# prescribed and free time step counts, which match the flapping free flight fixtures'.
FLAPPING_FREE_FLIGHT_DELTA_TIME = 0.01292
FLAPPING_FREE_FLIGHT_PRESCRIBED_NUM_STEPS = 5
FLAPPING_FREE_FLIGHT_FREE_NUM_STEPS = 10


def make_static_validation_movement() -> ps.movements.movement.Movement:
    """This function creates a Movement with static geometry to be used as a fixture.

    :return unsteady_validation_movement: Movement This is a Movement with static
        geometry to be used as a fixture.
    """
    unsteady_validation_airplane = (
        airplane_fixtures.make_symmetric_unsteady_validation_airplane()
    )
    unsteady_validation_operating_point = (
        operating_point_fixtures.make_validation_operating_point()
    )

    unsteady_validation_root_wing_cross_section_movement = (
        ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
            base_wing_cross_section=unsteady_validation_airplane.wings[
                0
            ].wing_cross_sections[0]
        )
    )

    unsteady_validation_tip_wing_cross_section_movement = (
        ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
            base_wing_cross_section=unsteady_validation_airplane.wings[
                0
            ].wing_cross_sections[1]
        )
    )

    unsteady_validation_wing_movement = ps.movements.wing_movement.WingMovement(
        base_wing=unsteady_validation_airplane.wings[0],
        wing_cross_section_movements=[
            unsteady_validation_root_wing_cross_section_movement,
            unsteady_validation_tip_wing_cross_section_movement,
        ],
    )

    unsteady_validation_airplane_movement = (
        ps.movements.airplane_movement.AirplaneMovement(
            base_airplane=unsteady_validation_airplane,
            wing_movements=[unsteady_validation_wing_movement],
        )
    )

    unsteady_validation_operating_point_movement = (
        ps.movements.operating_point_movement.OperatingPointMovement(
            base_operating_point=unsteady_validation_operating_point
        )
    )

    unsteady_validation_movement = ps.movements.movement.Movement(
        airplane_movements=[unsteady_validation_airplane_movement],
        operating_point_movement=unsteady_validation_operating_point_movement,
        num_chords=6,
    )

    return unsteady_validation_movement


def make_edge_defined_static_validation_movement() -> ps.movements.movement.Movement:
    """This function creates a static Movement over an edge-defined Airplane, to be used
    as a fixture for testing the convergence functions' edge-defined refinement.

    Every WingCrossSectionMovement is static, which edge-defined convergence requires,
    since resampling the Wing changes its number of WingCrossSections.

    :return edge_defined_validation_movement: Movement This is the Movement fixture.
    """
    edge_defined_validation_airplane = (
        airplane_fixtures.make_edge_defined_validation_airplane()
    )
    edge_defined_validation_operating_point = (
        operating_point_fixtures.make_validation_operating_point()
    )

    edge_defined_wing_cross_section_movements = [
        ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
            base_wing_cross_section=wing_cross_section
        )
        for wing_cross_section in edge_defined_validation_airplane.wings[
            0
        ].wing_cross_sections
    ]

    edge_defined_wing_movement = ps.movements.wing_movement.WingMovement(
        base_wing=edge_defined_validation_airplane.wings[0],
        wing_cross_section_movements=edge_defined_wing_cross_section_movements,
    )

    edge_defined_airplane_movement = ps.movements.airplane_movement.AirplaneMovement(
        base_airplane=edge_defined_validation_airplane,
        wing_movements=[edge_defined_wing_movement],
    )

    edge_defined_operating_point_movement = (
        ps.movements.operating_point_movement.OperatingPointMovement(
            base_operating_point=edge_defined_validation_operating_point
        )
    )

    edge_defined_validation_movement = ps.movements.movement.Movement(
        airplane_movements=[edge_defined_airplane_movement],
        operating_point_movement=edge_defined_operating_point_movement,
        num_chords=6,
    )

    return edge_defined_validation_movement


def make_edge_defined_non_static_validation_movement() -> (
    ps.movements.movement.Movement
):
    """This function creates a Movement over an edge-defined Airplane whose second
    WingCrossSectionMovement pitches, to be used as a fixture for testing that edge-
    defined convergence rejects non-static WingCrossSectionMovements.

    Resampling an edge-defined Wing changes its number of WingCrossSections, so per-
    WingCrossSection motion cannot be preserved. This Movement gives one interior
    WingCrossSection a pitching motion so the convergence function must reject it.

    :return edge_defined_non_static_validation_movement: Movement This is the Movement
        fixture.
    """
    edge_defined_validation_airplane = (
        airplane_fixtures.make_edge_defined_validation_airplane()
    )
    edge_defined_validation_operating_point = (
        operating_point_fixtures.make_validation_operating_point()
    )

    edge_defined_wing_cross_sections = edge_defined_validation_airplane.wings[
        0
    ].wing_cross_sections
    edge_defined_wing_cross_section_movements = [
        ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
            base_wing_cross_section=wing_cross_section
        )
        for wing_cross_section in edge_defined_wing_cross_sections
    ]

    # Replace an interior WingCrossSectionMovement with a pitching one. The root
    # WingCrossSection cannot pitch, so the second WingCrossSection is used.
    edge_defined_wing_cross_section_movements[1] = (
        ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
            base_wing_cross_section=edge_defined_wing_cross_sections[1],
            ampAngles_Wcsp_to_Wcs_ixyz=(0.0, 5.0, 0.0),
            periodAngles_Wcsp_to_Wcs_ixyz=(0.0, 1.0, 0.0),
        )
    )

    edge_defined_wing_movement = ps.movements.wing_movement.WingMovement(
        base_wing=edge_defined_validation_airplane.wings[0],
        wing_cross_section_movements=edge_defined_wing_cross_section_movements,
    )

    edge_defined_airplane_movement = ps.movements.airplane_movement.AirplaneMovement(
        base_airplane=edge_defined_validation_airplane,
        wing_movements=[edge_defined_wing_movement],
    )

    edge_defined_operating_point_movement = (
        ps.movements.operating_point_movement.OperatingPointMovement(
            base_operating_point=edge_defined_validation_operating_point
        )
    )

    edge_defined_non_static_validation_movement = ps.movements.movement.Movement(
        airplane_movements=[edge_defined_airplane_movement],
        operating_point_movement=edge_defined_operating_point_movement,
        num_cycles=1,
    )

    return edge_defined_non_static_validation_movement


def make_variable_validation_movement() -> ps.movements.movement.Movement:
    """This function creates a Movement with variable geometry to be used as a fixture.

    :return unsteady_validation_movement: Movement This is a Movement with variable
        geometry to be used as a fixture.
    """
    unsteady_validation_airplane = (
        airplane_fixtures.make_symmetric_unsteady_validation_airplane()
    )
    unsteady_validation_operating_point = (
        operating_point_fixtures.make_validation_operating_point()
    )

    unsteady_validation_root_wing_cross_section_movement = (
        ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
            base_wing_cross_section=unsteady_validation_airplane.wings[
                0
            ].wing_cross_sections[0]
        )
    )

    unsteady_validation_tip_wing_cross_section_movement = (
        ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
            base_wing_cross_section=unsteady_validation_airplane.wings[
                0
            ].wing_cross_sections[1],
            ampLp_Wcsp_Lpp=(0.0, 0.0, 0.0),
            periodLp_Wcsp_Lpp=(0.0, 0.0, 0.0),
            spacingLp_Wcsp_Lpp=("sine", "sine", "sine"),
            phaseLp_Wcsp_Lpp=(0.0, 0.0, 0.0),
            ampAngles_Wcsp_to_Wcs_ixyz=(30.0, 30.0, 30.0),
            periodAngles_Wcsp_to_Wcs_ixyz=(1.0, 0.5, 0.5),
            spacingAngles_Wcsp_to_Wcs_ixyz=("sine", "sine", "sine"),
            phaseAngles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
        )
    )

    unsteady_validation_wing_movement = ps.movements.wing_movement.WingMovement(
        base_wing=unsteady_validation_airplane.wings[0],
        wing_cross_section_movements=[
            unsteady_validation_root_wing_cross_section_movement,
            unsteady_validation_tip_wing_cross_section_movement,
        ],
    )

    unsteady_validation_airplane_movement = (
        ps.movements.airplane_movement.AirplaneMovement(
            base_airplane=unsteady_validation_airplane,
            wing_movements=[
                unsteady_validation_wing_movement,
            ],
        )
    )

    unsteady_validation_operating_point_movement = (
        ps.movements.operating_point_movement.OperatingPointMovement(
            base_operating_point=unsteady_validation_operating_point
        )
    )

    unsteady_validation_movement = ps.movements.movement.Movement(
        airplane_movements=[unsteady_validation_airplane_movement],
        operating_point_movement=unsteady_validation_operating_point_movement,
        num_cycles=1,
    )

    return unsteady_validation_movement


def make_multiple_wing_static_validation_movement() -> ps.movements.movement.Movement:
    """This function creates a Movement with static, multi-wing geometry to be used as a
    fixture.

    :return unsteady_validation_movement: Movement This is a Movement with variable
        geometry to be used as a fixture.
    """
    unsteady_validation_airplane = (
        airplane_fixtures.make_symmetric_multiple_wing_unsteady_validation_airplane()
    )
    unsteady_validation_operating_point = (
        operating_point_fixtures.make_validation_operating_point()
    )

    unsteady_validation_main_wing_root_cross_section_movement = (
        ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
            base_wing_cross_section=unsteady_validation_airplane.wings[
                0
            ].wing_cross_sections[0]
        )
    )

    unsteady_validation_main_wing_tip_cross_section_movement = (
        ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
            base_wing_cross_section=unsteady_validation_airplane.wings[
                0
            ].wing_cross_sections[1]
        )
    )

    unsteady_validation_hstab_root_cross_section_movement = (
        ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
            base_wing_cross_section=unsteady_validation_airplane.wings[
                1
            ].wing_cross_sections[0]
        )
    )

    unsteady_validation_hstab_tip_cross_section_movement = (
        ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
            base_wing_cross_section=unsteady_validation_airplane.wings[
                1
            ].wing_cross_sections[1]
        )
    )

    unsteady_validation_vstab_root_cross_section_movement = (
        ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
            base_wing_cross_section=unsteady_validation_airplane.wings[
                2
            ].wing_cross_sections[0]
        )
    )

    unsteady_validation_vstab_tip_cross_section_movement = (
        ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
            base_wing_cross_section=unsteady_validation_airplane.wings[
                2
            ].wing_cross_sections[1]
        )
    )

    unsteady_validation_main_wing_movement = ps.movements.wing_movement.WingMovement(
        base_wing=unsteady_validation_airplane.wings[0],
        wing_cross_section_movements=[
            unsteady_validation_main_wing_root_cross_section_movement,
            unsteady_validation_main_wing_tip_cross_section_movement,
        ],
    )

    unsteady_validation_hstab_movement = ps.movements.wing_movement.WingMovement(
        base_wing=unsteady_validation_airplane.wings[1],
        wing_cross_section_movements=[
            unsteady_validation_hstab_root_cross_section_movement,
            unsteady_validation_hstab_tip_cross_section_movement,
        ],
    )

    unsteady_validation_vstab_movement = ps.movements.wing_movement.WingMovement(
        base_wing=unsteady_validation_airplane.wings[2],
        wing_cross_section_movements=[
            unsteady_validation_vstab_root_cross_section_movement,
            unsteady_validation_vstab_tip_cross_section_movement,
        ],
    )

    unsteady_validation_airplane_movement = (
        ps.movements.airplane_movement.AirplaneMovement(
            base_airplane=unsteady_validation_airplane,
            wing_movements=[
                unsteady_validation_main_wing_movement,
                unsteady_validation_hstab_movement,
                unsteady_validation_vstab_movement,
            ],
        )
    )

    unsteady_validation_operating_point_movement = (
        ps.movements.operating_point_movement.OperatingPointMovement(
            base_operating_point=unsteady_validation_operating_point
        )
    )

    unsteady_validation_movement = ps.movements.movement.Movement(
        airplane_movements=[unsteady_validation_airplane_movement],
        operating_point_movement=unsteady_validation_operating_point_movement,
        num_steps=8,
        delta_time=1 / 8 / 10,
    )

    return unsteady_validation_movement


def make_surface_effect_static_movement() -> ps.movements.movement.Movement:
    """This function creates a Movement with static geometry and an image surface for
    surface effect testing.

    :return surface_effect_movement: Movement This is a Movement fixture for surface
        effect testing.
    """
    surface_effect_airplane = airplane_fixtures.make_surface_effect_airplane()
    surface_effect_operating_point = (
        operating_point_fixtures.make_surface_effect_operating_point()
    )

    root_wing_cross_section_movement = (
        ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
            base_wing_cross_section=surface_effect_airplane.wings[
                0
            ].wing_cross_sections[0]
        )
    )

    tip_wing_cross_section_movement = (
        ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
            base_wing_cross_section=surface_effect_airplane.wings[
                0
            ].wing_cross_sections[1]
        )
    )

    wing_movement = ps.movements.wing_movement.WingMovement(
        base_wing=surface_effect_airplane.wings[0],
        wing_cross_section_movements=[
            root_wing_cross_section_movement,
            tip_wing_cross_section_movement,
        ],
    )

    airplane_movement = ps.movements.airplane_movement.AirplaneMovement(
        base_airplane=surface_effect_airplane,
        wing_movements=[wing_movement],
    )

    operating_point_movement = (
        ps.movements.operating_point_movement.OperatingPointMovement(
            base_operating_point=surface_effect_operating_point
        )
    )

    surface_effect_movement = ps.movements.movement.Movement(
        airplane_movements=[airplane_movement],
        operating_point_movement=operating_point_movement,
        num_chords=6,
    )

    return surface_effect_movement


def make_surface_effect_free_air_static_movement() -> ps.movements.movement.Movement:
    """This function creates a Movement with static geometry and no image surface, for
    use as a free-air baseline in surface effect validation tests.

    :return free_air_movement: Movement This is a Movement fixture for the free-air
        baseline.
    """
    surface_effect_airplane = airplane_fixtures.make_surface_effect_airplane()
    free_air_operating_point = (
        operating_point_fixtures.make_surface_effect_free_air_operating_point()
    )

    root_wing_cross_section_movement = (
        ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
            base_wing_cross_section=surface_effect_airplane.wings[
                0
            ].wing_cross_sections[0]
        )
    )

    tip_wing_cross_section_movement = (
        ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
            base_wing_cross_section=surface_effect_airplane.wings[
                0
            ].wing_cross_sections[1]
        )
    )

    wing_movement = ps.movements.wing_movement.WingMovement(
        base_wing=surface_effect_airplane.wings[0],
        wing_cross_section_movements=[
            root_wing_cross_section_movement,
            tip_wing_cross_section_movement,
        ],
    )

    airplane_movement = ps.movements.airplane_movement.AirplaneMovement(
        base_airplane=surface_effect_airplane,
        wing_movements=[wing_movement],
    )

    operating_point_movement = (
        ps.movements.operating_point_movement.OperatingPointMovement(
            base_operating_point=free_air_operating_point
        )
    )

    free_air_movement = ps.movements.movement.Movement(
        airplane_movements=[airplane_movement],
        operating_point_movement=operating_point_movement,
        num_chords=6,
    )

    return free_air_movement


def make_multiple_wing_variable_validation_movement() -> ps.movements.movement.Movement:
    """This function creates a Movement with variable, multi-wing geometry to be used as
    a fixture.

    :return unsteady_validation_movement: Movement This is a Movement with variable
        geometry to be used as a fixture.
    """
    unsteady_validation_airplane = (
        airplane_fixtures.make_symmetric_multiple_wing_unsteady_validation_airplane()
    )
    unsteady_validation_operating_point = (
        operating_point_fixtures.make_validation_operating_point()
    )

    unsteady_validation_main_wing_root_cross_section_movement = (
        ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
            base_wing_cross_section=unsteady_validation_airplane.wings[
                0
            ].wing_cross_sections[0]
        )
    )

    unsteady_validation_main_wing_tip_cross_section_movement = (
        ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
            base_wing_cross_section=unsteady_validation_airplane.wings[
                0
            ].wing_cross_sections[1],
            ampLp_Wcsp_Lpp=(0.0, 0.0, 0.0),
            periodLp_Wcsp_Lpp=(0.0, 0.0, 0.0),
            spacingLp_Wcsp_Lpp=("sine", "sine", "sine"),
            phaseLp_Wcsp_Lpp=(0.0, 0.0, 0.0),
            ampAngles_Wcsp_to_Wcs_ixyz=(30.0, 15.0, 15.0),
            periodAngles_Wcsp_to_Wcs_ixyz=(1.0, 1.0, 0.5),
            spacingAngles_Wcsp_to_Wcs_ixyz=("sine", "sine", "sine"),
            phaseAngles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
        )
    )

    unsteady_validation_hstab_root_cross_section_movement = (
        ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
            base_wing_cross_section=unsteady_validation_airplane.wings[
                1
            ].wing_cross_sections[0]
        )
    )

    unsteady_validation_hstab_tip_cross_section_movement = (
        ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
            base_wing_cross_section=unsteady_validation_airplane.wings[
                1
            ].wing_cross_sections[1]
        )
    )

    unsteady_validation_vstab_root_cross_section_movement = (
        ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
            base_wing_cross_section=unsteady_validation_airplane.wings[
                2
            ].wing_cross_sections[0]
        )
    )

    unsteady_validation_vstab_tip_cross_section_movement = (
        ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
            base_wing_cross_section=unsteady_validation_airplane.wings[
                2
            ].wing_cross_sections[1]
        )
    )

    unsteady_validation_main_wing_movement = ps.movements.wing_movement.WingMovement(
        base_wing=unsteady_validation_airplane.wings[0],
        wing_cross_section_movements=[
            unsteady_validation_main_wing_root_cross_section_movement,
            unsteady_validation_main_wing_tip_cross_section_movement,
        ],
    )

    unsteady_validation_hstab_movement = ps.movements.wing_movement.WingMovement(
        base_wing=unsteady_validation_airplane.wings[1],
        wing_cross_section_movements=[
            unsteady_validation_hstab_root_cross_section_movement,
            unsteady_validation_hstab_tip_cross_section_movement,
        ],
    )

    unsteady_validation_vstab_movement = ps.movements.wing_movement.WingMovement(
        base_wing=unsteady_validation_airplane.wings[2],
        wing_cross_section_movements=[
            unsteady_validation_vstab_root_cross_section_movement,
            unsteady_validation_vstab_tip_cross_section_movement,
        ],
    )

    unsteady_validation_airplane_movement = (
        ps.movements.airplane_movement.AirplaneMovement(
            base_airplane=unsteady_validation_airplane,
            wing_movements=[
                unsteady_validation_main_wing_movement,
                unsteady_validation_hstab_movement,
                unsteady_validation_vstab_movement,
            ],
        )
    )

    unsteady_validation_operating_point_movement = (
        ps.movements.operating_point_movement.OperatingPointMovement(
            base_operating_point=unsteady_validation_operating_point
        )
    )

    unsteady_validation_movement = ps.movements.movement.Movement(
        airplane_movements=[unsteady_validation_airplane_movement],
        operating_point_movement=unsteady_validation_operating_point_movement,
        num_steps=20,
        delta_time=1 / 8 / 10,
    )

    return unsteady_validation_movement


def make_simple_glider_free_flight_movement(
    ground_height: float | None = None,
) -> ps.movements.free_flight_movement.FreeFlightMovement:
    """This function creates the simple glider's FreeFlightMovement to be used as a
    fixture.

    The motion is static (the prescribed geometry does not deform or flap), so the only
    motion in the simulation comes from the rigid body dynamics during the free flight
    phase. The glider first holds its trimmed condition for prescribed_num_steps time
    steps so the wake can develop, then the solver releases the rigid body dynamics for
    the remaining free_num_steps time steps.

    :param ground_height: None for free air, or the height (m) of the glider's CG above
        a flat ground (the Earth xy plane), for ground effect. The default is None.
    :return simple_glider_free_flight_movement: FreeFlightMovement This is the simple
        glider FreeFlightMovement fixture.
    """
    simple_glider_airplane = airplane_fixtures.make_simple_glider_airplane()
    simple_glider_operating_point = (
        operating_point_fixtures.make_simple_glider_operating_point(ground_height)
    )

    # Build a static WingMovement for each of the glider's three Wings.
    wing_movements = []
    for wing in simple_glider_airplane.wings:
        wing_cross_section_movements = [
            ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
                base_wing_cross_section=wing_cross_section
            )
            for wing_cross_section in wing.wing_cross_sections
        ]
        wing_movements.append(
            ps.movements.wing_movement.WingMovement(
                base_wing=wing,
                wing_cross_section_movements=wing_cross_section_movements,
            )
        )

    simple_glider_airplane_movement = ps.movements.airplane_movement.AirplaneMovement(
        base_airplane=simple_glider_airplane,
        wing_movements=wing_movements,
    )

    simple_glider_operating_point_movement = ps.movements.free_flight_operating_point_movement.FreeFlightOperatingPointMovement(
        base_operating_point=simple_glider_operating_point,
    )

    simple_glider_free_flight_movement = (
        ps.movements.free_flight_movement.FreeFlightMovement(
            airplane_movements=[simple_glider_airplane_movement],
            operating_point_movement=simple_glider_operating_point_movement,
            delta_time=0.01292,
            prescribed_num_steps=15,
            free_num_steps=10,
        )
    )

    return simple_glider_free_flight_movement


def make_flapping_free_flight_movement(
    explode_into_strips: bool = False,
) -> ps.movements.free_flight_movement.FreeFlightMovement:
    """This function creates the flapping-wing FreeFlightMovement to be used as a
    fixture.

    The main wing flaps symmetrically (a 15 degree amplitude at 1 Hz) while the V-tail
    stays static. The main wing has type 5 symmetry, so it is split into two mirrored
    Wings (the airplane's first two Wings); both halves flap with the same amplitude so
    the flapping stays laterally symmetric. The airplane first holds its initial
    condition for prescribed_num_steps time steps so the wake can develop, then the
    solver releases the rigid body dynamics for the remaining free_num_steps time steps,
    where the strongly coupled sub-iteration engages.

    The step counts are much smaller than the example's; they are enough to develop a
    wake and exercise the strongly coupled free flight phase under oscillatory flapping
    loads while keeping the integration test affordable.

    :param explode_into_strips: Whether the main Wing is exploded into one
        WingCrossSection per spanwise strip. The default is False.
    :return flapping_free_flight_movement: FreeFlightMovement This is the flapping-wing
        FreeFlightMovement fixture.
    """
    flapping_airplane = airplane_fixtures.make_flapping_free_flight_airplane(
        explode_into_strips
    )
    flapping_operating_point = (
        operating_point_fixtures.make_flapping_free_flight_operating_point()
    )

    # The main wing's two mirrored halves are the airplane's first two Wings; flap both
    # with the same 15 degree, 1 Hz roll amplitude so the flapping is symmetric. The
    # V-tail (the third Wing) stays static.
    flapping_wing_indices = (0, 1)

    wing_movements = []
    for wing_index, wing in enumerate(flapping_airplane.wings):
        wing_cross_section_movements = [
            ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
                base_wing_cross_section=wing_cross_section
            )
            for wing_cross_section in wing.wing_cross_sections
        ]
        if wing_index in flapping_wing_indices:
            ampAngles_Gs_to_Wn_ixyz = (15.0, 0.0, 0.0)
            periodAngles_Gs_to_Wn_ixyz = (1.0, 0.0, 0.0)
        else:
            ampAngles_Gs_to_Wn_ixyz = (0.0, 0.0, 0.0)
            periodAngles_Gs_to_Wn_ixyz = (0.0, 0.0, 0.0)
        wing_movements.append(
            ps.movements.wing_movement.WingMovement(
                base_wing=wing,
                wing_cross_section_movements=wing_cross_section_movements,
                ampAngles_Gs_to_Wn_ixyz=ampAngles_Gs_to_Wn_ixyz,
                periodAngles_Gs_to_Wn_ixyz=periodAngles_Gs_to_Wn_ixyz,
            )
        )

    flapping_airplane_movement = ps.movements.airplane_movement.AirplaneMovement(
        base_airplane=flapping_airplane,
        wing_movements=wing_movements,
    )

    flapping_operating_point_movement = ps.movements.free_flight_operating_point_movement.FreeFlightOperatingPointMovement(
        base_operating_point=flapping_operating_point,
    )

    flapping_free_flight_movement = (
        ps.movements.free_flight_movement.FreeFlightMovement(
            airplane_movements=[flapping_airplane_movement],
            operating_point_movement=flapping_operating_point_movement,
            delta_time=0.01292,
            prescribed_num_steps=5,
            free_num_steps=10,
        )
    )

    return flapping_free_flight_movement


def make_formation_validation_movement() -> ps.movements.movement.Movement:
    """This function creates a static Movement over two Airplanes to be used as a
    fixture.

    The second Airplane's name carries a path separator, so this Movement backs the
    tests that prove the output functions reject such a name before writing any of the
    first Airplane's files.

    :return formation_validation_movement: Movement This is the Movement fixture.
    """
    formation_airplanes = [
        airplane_fixtures.make_formation_lead_validation_airplane(),
        airplane_fixtures.make_separator_named_validation_airplane(),
    ]

    formation_airplane_movements = []
    for formation_airplane in formation_airplanes:
        formation_wing_cross_section_movements = [
            ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
                base_wing_cross_section=wing_cross_section
            )
            for wing_cross_section in formation_airplane.wings[0].wing_cross_sections
        ]

        formation_wing_movement = ps.movements.wing_movement.WingMovement(
            base_wing=formation_airplane.wings[0],
            wing_cross_section_movements=formation_wing_cross_section_movements,
        )

        formation_airplane_movements.append(
            ps.movements.airplane_movement.AirplaneMovement(
                base_airplane=formation_airplane,
                wing_movements=[formation_wing_movement],
            )
        )

    formation_operating_point_movement = ps.movements.operating_point_movement.OperatingPointMovement(
        base_operating_point=operating_point_fixtures.make_validation_operating_point()
    )

    formation_validation_movement = ps.movements.movement.Movement(
        airplane_movements=formation_airplane_movements,
        operating_point_movement=formation_operating_point_movement,
        num_chords=3,
    )

    return formation_validation_movement


def make_flapping_aeroelastic_airplane_movement() -> (
    ps.movements.aeroelastic_airplane_movement.AeroelasticAirplaneMovement
):
    """This function creates the aeroelastic version of the flapping free flight
    fixtures' AirplaneMovement.

    It flaps the same Airplane (with its main Wing exploded into strips, which the
    torsional springs need) through the same symmetric 15 degree, 1 Hz flap, but the two
    halves of the main Wing are moved by AeroelasticWingMovements, so they deform. The
    V-tail stays rigid and static.

    :return flapping_aeroelastic_airplane_movement: AeroelasticAirplaneMovement This is
        the AeroelasticAirplaneMovement fixture.
    """
    flapping_airplane = airplane_fixtures.make_flapping_free_flight_airplane(
        explode_into_strips=True
    )
    wing_movements: list[
        ps.movements.aeroelastic_wing_movement.AeroelasticWingMovement
        | ps.movements.wing_movement.WingMovement
    ] = []
    for wing_index, wing in enumerate(flapping_airplane.wings):
        if wing_index in (0, 1):
            wing_movements.append(
                ps.movements.aeroelastic_wing_movement.AeroelasticWingMovement(
                    base_wing=wing,
                    wing_cross_section_movements=[
                        ps.movements.aeroelastic_wing_cross_section_movement.AeroelasticWingCrossSectionMovement(
                            base_wing_cross_section=wing_cross_section
                        )
                        for wing_cross_section in wing.wing_cross_sections
                    ],
                    ampAngles_Gs_to_Wn_ixyz=(15.0, 0.0, 0.0),
                    periodAngles_Gs_to_Wn_ixyz=(1.0, 0.0, 0.0),
                )
            )
        else:
            wing_movements.append(
                ps.movements.wing_movement.WingMovement(
                    base_wing=wing,
                    wing_cross_section_movements=[
                        ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
                            base_wing_cross_section=wing_cross_section
                        )
                        for wing_cross_section in wing.wing_cross_sections
                    ],
                )
            )

    flapping_aeroelastic_airplane_movement = (
        ps.movements.aeroelastic_airplane_movement.AeroelasticAirplaneMovement(
            base_airplane=flapping_airplane, wing_movements=wing_movements
        )
    )

    return flapping_aeroelastic_airplane_movement


def make_flapping_free_flight_aeroelastic_movement() -> (
    ps.movements.free_flight_movement.FreeFlightAeroelasticMovement
):
    """This function creates the flapping free flight aeroelastic movement to be used as
    a fixture.

    It is the aeroelastic version of the flapping free flight movement fixture, with the
    same initial OperatingPoint, time step, and prescribed and free time step counts.

    :return flapping_free_flight_aeroelastic_movement: FreeFlightAeroelasticMovement
        This is the FreeFlightAeroelasticMovement fixture.
    """
    flapping_free_flight_aeroelastic_movement = ps.movements.free_flight_movement.FreeFlightAeroelasticMovement(
        airplane_movements=[make_flapping_aeroelastic_airplane_movement()],
        operating_point_movement=ps.movements.free_flight_operating_point_movement.FreeFlightOperatingPointMovement(
            base_operating_point=operating_point_fixtures.make_flapping_free_flight_operating_point()
        ),
        delta_time=FLAPPING_FREE_FLIGHT_DELTA_TIME,
        prescribed_num_steps=FLAPPING_FREE_FLIGHT_PRESCRIBED_NUM_STEPS,
        free_num_steps=FLAPPING_FREE_FLIGHT_FREE_NUM_STEPS,
    )

    return flapping_free_flight_aeroelastic_movement


def make_flapping_held_aeroelastic_movement(
    num_steps: int,
) -> ps.movements.aeroelastic_movement.AeroelasticMovement:
    """This function creates an AeroelasticMovement that flaps the flapping free flight
    aeroelastic fixtures' Airplane while holding it at their initial OperatingPoint, as
    the free flight fixture's prescribed time steps do.

    :param num_steps: The number of time steps.
    :return flapping_held_aeroelastic_movement: AeroelasticMovement This is the
        AeroelasticMovement fixture.
    """
    flapping_held_aeroelastic_movement = ps.movements.aeroelastic_movement.AeroelasticMovement(
        airplane_movements=[make_flapping_aeroelastic_airplane_movement()],
        operating_point_movement=ps.movements.operating_point_movement.OperatingPointMovement(
            base_operating_point=operating_point_fixtures.make_flapping_free_flight_operating_point()
        ),
        delta_time=FLAPPING_FREE_FLIGHT_DELTA_TIME,
        num_steps=num_steps,
    )

    return flapping_held_aeroelastic_movement


def make_moving_formation_validation_movement() -> ps.movements.movement.Movement:
    """This function creates the formation validation Movement with its second Airplane
    moving, to be used as a fixture.

    The second Airplane's CG oscillates in every direction (ampCg_GP1_CgP1), and its
    Wing pitches (its Wing's angles_Gs_to_Wn_ixyz y component), so each time step's
    Airplanes are in different relative positions, which tests that each Airplane's
    moments are found about its CG at that time step. (Its Wing is symmetric, so it
    pitches rather than flaps: flapping it would change its symmetry type.)

    :return moving_formation_validation_movement: Movement This is the Movement fixture.
    """
    formation_airplanes = [
        airplane_fixtures.make_formation_lead_validation_airplane(),
        airplane_fixtures.make_separator_named_validation_airplane(),
    ]

    formation_airplane_movements = []
    for airplane_id, formation_airplane in enumerate(formation_airplanes):
        moving = airplane_id == 1
        wing = formation_airplane.wings[0]
        formation_wing_movement = ps.movements.wing_movement.WingMovement(
            base_wing=wing,
            wing_cross_section_movements=[
                ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
                    base_wing_cross_section=wing_cross_section
                )
                for wing_cross_section in wing.wing_cross_sections
            ],
            ampAngles_Gs_to_Wn_ixyz=(0.0, 5.0, 0.0) if moving else (0.0, 0.0, 0.0),
            periodAngles_Gs_to_Wn_ixyz=(0.0, 0.5, 0.0) if moving else (0.0, 0.0, 0.0),
        )
        formation_airplane_movements.append(
            ps.movements.airplane_movement.AirplaneMovement(
                base_airplane=formation_airplane,
                wing_movements=[formation_wing_movement],
                ampCg_GP1_CgP1=(0.3, 0.2, 0.1) if moving else (0.0, 0.0, 0.0),
                periodCg_GP1_CgP1=(0.5, 0.5, 0.5) if moving else (0.0, 0.0, 0.0),
            )
        )

    moving_formation_validation_movement = ps.movements.movement.Movement(
        airplane_movements=formation_airplane_movements,
        operating_point_movement=ps.movements.operating_point_movement.OperatingPointMovement(
            base_operating_point=operating_point_fixtures.make_validation_operating_point()
        ),
        num_cycles=2,
    )

    return moving_formation_validation_movement
