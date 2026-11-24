"""This module contains functions to create fixtures for the springs array function
tests."""

import pterasoftware as ps

from . import geometry_fixtures


def _make_strip_wing(mirror_only: bool) -> ps.geometry.wing.Wing:
    """Makes a Wing whose strips each have one spanwise Panel, as an aeroelastic Wing's
    do, with chords and spanwise spacings that change from strip to strip, so every
    strip's width and inertias differ and a reversed strip order is detectable.

    :param mirror_only: Whether the Wing is mirror only (type 3 symmetry). If not, it
        has type 1 symmetry.
    :return: The Wing, not yet processed by an Airplane.
    """
    chords = (2.0, 1.7, 1.3, 0.8)
    Lps_Wcsp_Lpp = (
        (0.0, 0.0, 0.0),
        (0.1, 0.5, 0.05),
        (0.2, 0.7, 0.1),
        (0.3, 0.9, 0.2),
    )
    wing_cross_sections = [
        ps.geometry.wing_cross_section.WingCrossSection(
            airfoil=geometry_fixtures.make_test_airfoil_fixture(),
            num_spanwise_panels=None if index == len(chords) - 1 else 1,
            chord=chord,
            Lp_Wcsp_Lpp=Lp_Wcsp_Lpp,
            angles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
            spanwise_spacing=None if index == len(chords) - 1 else "uniform",
        )
        for index, (chord, Lp_Wcsp_Lpp) in enumerate(zip(chords, Lps_Wcsp_Lpp))
    ]
    if mirror_only:
        return ps.geometry.wing.Wing(
            wing_cross_sections=wing_cross_sections,
            name="Mirror Only Strip Wing",
            Ler_Gs_Cgs=(0.0, 0.0, 0.0),
            angles_Gs_to_Wn_ixyz=(0.0, 0.0, 0.0),
            symmetric=False,
            mirror_only=True,
            symmetryNormal_G=(0.0, 1.0, 0.0),
            symmetryPoint_G_Cg=(0.0, -0.5, 0.0),
            num_chordwise_panels=3,
            chordwise_spacing="cosine",
        )
    return ps.geometry.wing.Wing(
        wing_cross_sections=wing_cross_sections,
        name="Strip Wing",
        Ler_Gs_Cgs=(1.0, 0.0, 0.5),
        angles_Gs_to_Wn_ixyz=(0.0, 5.0, 0.0),
        symmetric=False,
        mirror_only=False,
        symmetryNormal_G=None,
        symmetryPoint_G_Cg=None,
        num_chordwise_panels=3,
        chordwise_spacing="cosine",
    )


def make_strip_wings_airplane_fixture() -> ps.geometry.airplane.Airplane:
    """This method makes a fixture that is an Airplane, at the first Airplane's CG, with
    two Wings whose strips each have one spanwise Panel: a type 1 Wing, then a mirror
    only (type 3) Wing, whose Panel grid runs tip to root.

    :return strip_wings_airplane_fixture: Airplane This is the Airplane, for testing the
        springs array functions on both a plain and a mirror only Wing, with the second
        Wing offset in the global Panel order.
    """
    strip_wings_airplane_fixture = ps.geometry.airplane.Airplane(
        wings=[_make_strip_wing(False), _make_strip_wing(True)],
        name="Strip Wings Airplane",
        Cg_GP1_CgP1=(0.0, 0.0, 0.0),
    )
    return strip_wings_airplane_fixture


def make_strip_wings_aeroelastic_problem_fixture() -> (
    ps.problems.AeroelasticUnsteadyProblem
):
    """This method makes a fixture that is an AeroelasticUnsteadyProblem whose Airplane
    is the strip Wings Airplane, both Wings aeroelastic and flapping, at an angle of
    attack, so the springs deform strips of different widths on a plain and a mirror
    only Wing. The flapping matters: a strip's width enters the springs only through its
    flapping axis inertia, which scales the flapping acceleration. Only the first time
    step is discarded, so the springs respond from the second.

    :return strip_wings_aeroelastic_problem_fixture: AeroelasticUnsteadyProblem This is
        the problem, not yet run.
    """
    airplane = make_strip_wings_airplane_fixture()
    airplane_movement = ps.movements.aeroelastic_airplane_movement.AeroelasticAirplaneMovement(
        base_airplane=airplane,
        wing_movements=[
            ps.movements.aeroelastic_wing_movement.AeroelasticWingMovement(
                base_wing=wing,
                wing_cross_section_movements=[
                    ps.movements.aeroelastic_wing_cross_section_movement.AeroelasticWingCrossSectionMovement(
                        base_wing_cross_section=wing_cross_section
                    )
                    for wing_cross_section in wing.wing_cross_sections
                ],
                ampAngles_Gs_to_Wn_ixyz=(5.0, 0.0, 0.0),
                periodAngles_Gs_to_Wn_ixyz=(0.32, 0.0, 0.0),
            )
            for wing in airplane.wings
        ],
    )
    movement = ps.movements.aeroelastic_movement.AeroelasticMovement(
        airplane_movements=[airplane_movement],
        operating_point_movement=ps.movements.operating_point_movement.OperatingPointMovement(
            base_operating_point=ps.operating_point.OperatingPoint(
                rho=1.225, vCg__E=10.0, alpha=5.0
            )
        ),
        delta_time=0.02,
        num_steps=8,
    )
    strip_wings_aeroelastic_problem_fixture = ps.problems.AeroelasticUnsteadyProblem(
        movement=movement,
        wing_density=0.5,
        spring_constant_rad=50.0,
        damping_constant_rad=1.0,
        step_discards=1,
    )
    return strip_wings_aeroelastic_problem_fixture
