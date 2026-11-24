"""This module contains functions to create Wings and their WingStepParameters for use
in the array meshing tests."""

import numpy as np

import pterasoftware as ps

# noinspection PyProtectedMember
from pterasoftware import _step_parameters
from tests.unit.fixtures import geometry_fixtures

# The Airfoils the random Wings choose from: one symmetric and two cambered.
_AIRFOIL_NAMES = ("naca0012", "naca2412", "naca4415")


def make_wing_step_parameters_fixture(
    wing: ps.geometry.wing.Wing,
) -> _step_parameters.WingStepParameters:
    """This method makes a fixture that is the WingStepParameters holding a Wing's own
    values, which a movement would produce for that Wing.

    :param wing: The Wing.
    :return: The WingStepParameters.
    """
    return _step_parameters.stack_wing_step_parameters(
        np.array(wing.Ler_Gs_Cgs, dtype=float),
        np.array(wing.angles_Gs_to_Wn_ixyz, dtype=float),
        [
            _step_parameters.WingCrossSectionStepParameters(
                Lp_Wcsp_Lpp=np.array(wing_cross_section.Lp_Wcsp_Lpp, dtype=float),
                angles_Wcsp_to_Wcs_ixyz=np.array(
                    wing_cross_section.angles_Wcsp_to_Wcs_ixyz, dtype=float
                ),
            )
            for wing_cross_section in wing.wing_cross_sections
        ],
    )


def make_type_4_asymmetric_control_surface_wing_fixture() -> ps.geometry.wing.Wing:
    """This method makes a fixture that is a Wing with type 4 symmetry whose tip
    WingCrossSection has an asymmetric control surface with a non zero deflection, so
    its mirrored half deflects the other way.

    :return: The Wing, not yet symmetry processed.
    """
    root_wing_cross_section = geometry_fixtures.make_root_wing_cross_section_fixture()
    root_wing_cross_section.control_surface_symmetry_type = "symmetric"
    tip_wing_cross_section = (
        geometry_fixtures.make_asymmetric_control_surface_wing_cross_section_fixture()
    )

    return ps.geometry.wing.Wing(
        wing_cross_sections=[root_wing_cross_section, tip_wing_cross_section],
        name="Type 4 Asymmetric Control Surface Test Wing",
        Ler_Gs_Cgs=[0.5, 0.0, 0.2],
        angles_Gs_to_Wn_ixyz=[0.0, 4.0, 0.0],
        symmetric=True,
        mirror_only=False,
        symmetryNormal_G=[0.0, 1.0, 0.0],
        symmetryPoint_G_Cg=[0.0, 0.0, 0.0],
        num_chordwise_panels=5,
        chordwise_spacing="uniform",
    )


def make_random_wing_fixture(rng: np.random.Generator) -> ps.geometry.wing.Wing:
    """This method makes a fixture that is a Wing with random geometry, for comparing
    the array meshing to mesh_wing on Wings no hand written fixture covers.

    The Wing has two to five WingCrossSections with random positions, angles, chords,
    Airfoils, spacings, and control surfaces. Its symmetry settings are random, and when
    it has a symmetry plane, half the time the plane is coincident with the wing axes'
    xz plane (giving type 2 or 4 symmetry) and otherwise it generally isn't (giving type
    3 or 5). Some are exploded into strips. Some random Wings are invalid (for example,
    an exploded Wing with cosine spanwise spacing, or a type 4 Wing with a deflected
    asymmetric control surface at its root), so the caller should skip any for which
    this method or its Airplane raises a ValueError.

    :param rng: The random number generator.
    :return: The Wing, not yet symmetry processed.
    """
    symmetric = bool(rng.random() < 0.4)
    mirror_only = (not symmetric) and bool(rng.random() < 0.4)
    num_wing_cross_sections = int(rng.integers(2, 6))

    wing_cross_sections = []
    for index in range(num_wing_cross_sections):
        is_root = index == 0
        is_tip = index == num_wing_cross_sections - 1
        wing_cross_sections.append(
            ps.geometry.wing_cross_section.WingCrossSection(
                airfoil=ps.geometry.airfoil.Airfoil(
                    name=_AIRFOIL_NAMES[int(rng.integers(len(_AIRFOIL_NAMES)))]
                ),
                num_spanwise_panels=None if is_tip else int(rng.integers(1, 6)),
                chord=float(rng.uniform(0.2, 2.0)),
                Lp_Wcsp_Lpp=(
                    np.zeros(3, dtype=float)
                    if is_root
                    else rng.uniform([-0.5, 0.1, -0.3], [0.5, 2.0, 0.3])
                ),
                angles_Wcsp_to_Wcs_ixyz=(
                    np.zeros(3, dtype=float) if is_root else rng.uniform(-30.0, 30.0, 3)
                ),
                control_surface_symmetry_type=(
                    str(rng.choice(["symmetric", "asymmetric"])) if symmetric else None
                ),
                control_surface_hinge_point=float(rng.uniform(0.5, 0.9)),
                control_surface_deflection=(
                    float(rng.uniform(-5.0, 5.0)) if rng.random() < 0.6 else 0.0
                ),
                spanwise_spacing=(
                    None if is_tip else str(rng.choice(["uniform", "cosine"]))
                ),
            )
        )

    Ler_Gs_Cgs = rng.uniform(-1.0, 1.0, 3)
    angles_Gs_to_Wn_ixyz = rng.uniform(-40.0, 40.0, 3)
    symmetryNormal_G = None
    symmetryPoint_G_Cg = None
    if symmetric or mirror_only:
        symmetryPoint_G_Cg = rng.uniform(-0.5, 0.5, 3)
        if rng.random() < 0.5:
            # A coincident symmetry plane: its normal is along the wing axes' y axis (so
            # the Wing may only rotate about it) and the leading edge root point is on
            # it.
            symmetryNormal_G = np.array([0.0, 1.0, 0.0], dtype=float)
            Ler_Gs_Cgs[1] = symmetryPoint_G_Cg[1]
            angles_Gs_to_Wn_ixyz[[0, 2]] = 0.0
        else:
            symmetryNormal_G = np.array([0.0, 1.0, 0.0], dtype=float) + rng.normal(
                0.0, 0.2, 3
            )

    return ps.geometry.wing.Wing(
        wing_cross_sections=wing_cross_sections,
        name="Random Test Wing",
        Ler_Gs_Cgs=Ler_Gs_Cgs,
        angles_Gs_to_Wn_ixyz=angles_Gs_to_Wn_ixyz,
        symmetric=symmetric,
        mirror_only=mirror_only,
        symmetryNormal_G=symmetryNormal_G,
        symmetryPoint_G_Cg=symmetryPoint_G_Cg,
        explode_into_strips=bool(rng.random() < 0.3),
        num_chordwise_panels=int(rng.integers(1, 9)),
        chordwise_spacing=str(rng.choice(["uniform", "cosine"])),
    )


def make_random_edge_defined_wing_fixture(
    rng: np.random.Generator,
) -> ps.geometry.wing.Wing:
    """This method makes a fixture that is a Wing built from swept, tapered leading edge
    and trailing edge curves, with a random number of WingCrossSections and chordwise
    panels, and either type 4 symmetry or none at a random position and orientation.

    :param rng: The random number generator.
    :return: The Wing, not yet symmetry processed.
    """
    spanwise_positions = np.linspace(0.0, 1.5, 30)
    leadingEdgePoints_Wn_Ler = np.column_stack(
        [
            0.3 * spanwise_positions**2,
            spanwise_positions,
            np.zeros_like(spanwise_positions),
        ]
    )
    trailingEdgePoints_Wn_Ler = np.column_stack(
        [
            1.0 - 0.2 * spanwise_positions,
            spanwise_positions,
            np.zeros_like(spanwise_positions),
        ]
    )
    symmetric = bool(rng.random() < 0.5)

    return ps.geometry.wing.Wing.from_edge_points(
        leadingEdgePoints_Wn_Ler=leadingEdgePoints_Wn_Ler,
        trailingEdgePoints_Wn_Ler=trailingEdgePoints_Wn_Ler,
        num_wing_cross_sections=int(rng.integers(3, 12)),
        airfoil=ps.geometry.airfoil.Airfoil(name="naca2412"),
        name="Random Edge Defined Test Wing",
        Ler_Gs_Cgs=(
            np.zeros(3, dtype=float) if symmetric else rng.uniform(-1.0, 1.0, 3)
        ),
        angles_Gs_to_Wn_ixyz=(
            np.zeros(3, dtype=float) if symmetric else rng.uniform(-20.0, 20.0, 3)
        ),
        symmetric=symmetric,
        symmetryNormal_G=(0.0, 1.0, 0.0) if symmetric else None,
        symmetryPoint_G_Cg=(0.0, 0.0, 0.0) if symmetric else None,
        num_chordwise_panels=int(rng.integers(1, 7)),
        tip_trim_fraction=0.05,
    )


def make_processed_wings_fixture(
    wing: ps.geometry.wing.Wing,
) -> list[ps.geometry.wing.Wing]:
    """This method makes a fixture that is a Wing's processed and meshed Wings, after an
    Airplane has processed its symmetry.

    :param wing: The Wing, not yet symmetry processed.
    :return: The Airplane's Wings (two for a Wing with type 5 symmetry, otherwise one).
    """
    return list(ps.geometry.airplane.Airplane(wings=[wing], weight=0.0).wings)


def make_random_processed_wings_fixture(
    seed: int, count: int
) -> list[ps.geometry.wing.Wing]:
    """This method makes a fixture that is the processed and meshed Wings of a number of
    valid random Wings (about one in seven edge defined), skipping invalid random
    combinations of settings.

    :param seed: The random number generator's seed.
    :param count: The number of valid random Wings to make. A type 5 Wing contributes
        both of its processed Wings.
    :return: The processed Wings.
    """
    rng = np.random.default_rng(seed)
    processed_wings: list[ps.geometry.wing.Wing] = []
    num_valid_wings = 0
    while num_valid_wings < count:
        try:
            if rng.random() < 0.15:
                wing = make_random_edge_defined_wing_fixture(rng)
            else:
                wing = make_random_wing_fixture(rng)
            processed_wings.extend(make_processed_wings_fixture(wing))
        except ValueError:
            # Some random combinations of settings are invalid.
            continue
        num_valid_wings += 1
    return processed_wings
