"""This module contains classes to test the array meshing's per run plans and its per
time step meshing."""

import unittest
from collections.abc import Callable
from typing import Any

import numpy as np

import pterasoftware as ps

# noinspection PyProtectedMember
from pterasoftware import _mesh_arrays, _step_parameters, _transformations

# noinspection PyProtectedMember
from pterasoftware.geometry import _meshing
from tests.unit.fixtures import (
    free_flight_movement_fixtures,
    geometry_fixtures,
    mesh_arrays_fixtures,
    movement_fixtures,
)

# The Panel corner attributes, in the order of the WingCorners fields.
_CORNER_NAMES = ("Flpp_G_Cg", "Frpp_G_Cg", "Blpp_G_Cg", "Brpp_G_Cg")

# The largest difference allowed between a corner from mesh_wing_arrays and the same
# corner from mesh_wing, in meters. The two compute the same operations in the same
# order and agree bit for bit on the development machine, but a batched NumPy call is
# not guaranteed to round exactly as the matching unbatched call on every platform.
_CORNER_TOLERANCE = 1.0e-12

# The same for the transformation and rotation matrices, which are unitless and of order
# one.
_MATRIX_TOLERANCE = 1.0e-14

# The Wing fixtures with every symmetry type (a type 5 Wing is split into a type 1 Wing
# and its type 3 reflection), multiple wing sections, sweep, dihedral, taper, twist, and
# control surfaces.
_WING_FIXTURE_MAKERS = (
    geometry_fixtures.make_type_1_wing_fixture,
    geometry_fixtures.make_type_2_wing_fixture,
    geometry_fixtures.make_type_3_wing_fixture,
    geometry_fixtures.make_type_4_wing_fixture,
    geometry_fixtures.make_type_5_wing_fixture,
    geometry_fixtures.make_three_section_wing_fixture,
    geometry_fixtures.make_four_section_wing_fixture,
    geometry_fixtures.make_simple_tapered_wing_fixture,
    geometry_fixtures.make_three_section_tapered_wing_fixture,
    geometry_fixtures.make_symmetric_continuous_rectangular_wing_fixture,
    geometry_fixtures.make_wing_with_rotated_cross_sections_fixture,
    geometry_fixtures.make_swept_wing_fixture,
    geometry_fixtures.make_dihedral_wing_fixture,
    geometry_fixtures.make_wing_with_2_chordwise_panels,
    mesh_arrays_fixtures.make_type_4_asymmetric_control_surface_wing_fixture,
)


def _helper_test_wings() -> list[ps.geometry.wing.Wing]:
    """Returns the processed Wings the helper function tests use: every Wing fixture's
    and 60 random Wings'.

    :return: The processed Wings.
    """
    wings = [
        wing
        for make in _WING_FIXTURE_MAKERS
        for wing in mesh_arrays_fixtures.make_processed_wings_fixture(make())
    ]
    return wings + mesh_arrays_fixtures.make_random_processed_wings_fixture(
        seed=1, count=60
    )


class TestBuildMeshPlan(unittest.TestCase):
    """This is a class with functions to test build_mesh_plan."""

    def test_plans_match_the_processed_wings(self) -> None:
        """Test that each processed Wing's MeshPlan has its symmetry type and panel
        counts, that its columns cover each wing section from 0.0 to 1.0 in order, that
        it holds one row of camber line points per WingCrossSection and the symmetry
        plane of every Wing that has one, and that its arrays are read only, for every
        symmetry type (a type 5 Wing is split into a type 1 Wing and its type 3
        reflection)."""
        makers = (
            geometry_fixtures.make_type_1_wing_fixture,
            geometry_fixtures.make_type_2_wing_fixture,
            geometry_fixtures.make_type_3_wing_fixture,
            geometry_fixtures.make_type_4_wing_fixture,
            geometry_fixtures.make_type_5_wing_fixture,
        )
        for make in makers:
            for wing in mesh_arrays_fixtures.make_processed_wings_fixture(make()):
                with self.subTest(wing=wing.name, symmetry_type=wing.symmetry_type):
                    plan = _mesh_arrays.build_mesh_plan(wing)
                    self.assertEqual(plan.symmetry_type, wing.symmetry_type)
                    self.assertEqual(
                        plan.num_chordwise_panels, wing.num_chordwise_panels
                    )
                    self.assertEqual(plan.num_spanwise_panels, wing.num_spanwise_panels)
                    assert wing.panels is not None
                    self.assertEqual(
                        wing.panels.shape,
                        (plan.num_chordwise_panels, plan.num_spanwise_panels),
                    )

                    # The columns describe the main half, which is half of a type 4
                    # Wing's Panel grid.
                    num_main_columns = (
                        plan.num_spanwise_panels // 2
                        if plan.symmetry_type == 4
                        else plan.num_spanwise_panels
                    )
                    self.assertEqual(plan.column_wing_section.size, num_main_columns)
                    self.assertEqual(
                        plan.chordwise_coordinates.size, plan.num_chordwise_panels + 1
                    )
                    self.assertEqual(plan.chordwise_coordinates[0], 0.0)
                    self.assertEqual(plan.chordwise_coordinates[-1], 1.0)

                    # Within each wing section, the columns run from 0.0 to 1.0 with no
                    # gaps.
                    for wing_section in np.unique(plan.column_wing_section):
                        columns = plan.column_wing_section == wing_section
                        inner = plan.column_inner_fraction[columns]
                        outer = plan.column_outer_fraction[columns]
                        self.assertEqual(inner[0], 0.0)
                        self.assertEqual(outer[-1], 1.0)
                        self.assertTrue(np.array_equal(inner[1:], outer[:-1]))
                        self.assertTrue(np.all(outer > inner))

                    # One row of camber line points per WingCrossSection, each starting
                    # at its leading point and in its xz plane.
                    self.assertEqual(
                        plan.gridMclPoints_Wcs_Lp.shape,
                        (
                            len(wing.wing_cross_sections),
                            plan.num_chordwise_panels + 1,
                            3,
                        ),
                    )
                    self.assertTrue(
                        np.array_equal(
                            plan.gridMclPoints_Wcs_Lp[:, 0],
                            np.zeros((len(wing.wing_cross_sections), 3)),
                        )
                    )
                    self.assertTrue(np.all(plan.gridMclPoints_Wcs_Lp[..., 1] == 0.0))

                    # Only Wings with a symmetry plane hold it.
                    if plan.symmetry_type == 1:
                        self.assertIsNone(plan.symmetryNormal_G)
                        self.assertIsNone(plan.symmetryPoint_G_Cg)
                    else:
                        assert plan.symmetryNormal_G is not None
                        assert plan.symmetryPoint_G_Cg is not None
                        assert wing.symmetryNormal_G is not None
                        assert wing.symmetryPoint_G_Cg is not None
                        self.assertTrue(
                            np.array_equal(plan.symmetryNormal_G, wing.symmetryNormal_G)
                        )
                        self.assertTrue(
                            np.array_equal(
                                plan.symmetryPoint_G_Cg, wing.symmetryPoint_G_Cg
                            )
                        )
                        self.assertFalse(plan.symmetryNormal_G.flags.writeable)
                        self.assertFalse(plan.symmetryPoint_G_Cg.flags.writeable)

                    for array in (
                        plan.chordwise_coordinates,
                        plan.column_wing_section,
                        plan.column_inner_fraction,
                        plan.column_outer_fraction,
                        plan.gridMclPoints_Wcs_Lp,
                    ):
                        self.assertFalse(array.flags.writeable)

    def test_mirrored_camber_lines_only_for_asymmetric_deflections(self) -> None:
        """Test that only a type 4 Wing with a deflected asymmetric control surface
        holds separate camber line points for its mirrored half, which differ from the
        main half's only at the asymmetric WingCrossSections."""
        for make in _WING_FIXTURE_MAKERS:
            for wing in mesh_arrays_fixtures.make_processed_wings_fixture(make()):
                with self.subTest(wing=wing.name):
                    plan = _mesh_arrays.build_mesh_plan(wing)
                    has_asymmetric_deflection = plan.symmetry_type == 4 and any(
                        wing_cross_section.control_surface_symmetry_type == "asymmetric"
                        and wing_cross_section.control_surface_deflection != 0.0
                        for wing_cross_section in wing.wing_cross_sections
                    )
                    if not has_asymmetric_deflection:
                        self.assertIsNone(plan.gridMirroredMclPoints_Wcs_Lp)
                        continue

                    assert plan.gridMirroredMclPoints_Wcs_Lp is not None
                    self.assertFalse(plan.gridMirroredMclPoints_Wcs_Lp.flags.writeable)
                    for index, wing_cross_section in enumerate(
                        wing.wing_cross_sections
                    ):
                        differs = (
                            wing_cross_section.control_surface_symmetry_type
                            == "asymmetric"
                            and wing_cross_section.control_surface_deflection != 0.0
                        )
                        self.assertEqual(
                            not np.array_equal(
                                plan.gridMirroredMclPoints_Wcs_Lp[index],
                                plan.gridMclPoints_Wcs_Lp[index],
                            ),
                            differs,
                        )

    def test_unprocessed_wing_is_rejected(self) -> None:
        """Test that a Wing whose symmetry hasn't been processed by an Airplane is
        rejected."""
        with self.assertRaises(ValueError):
            _mesh_arrays.build_mesh_plan(geometry_fixtures.make_type_1_wing_fixture())


class TestMeshWingArrays(unittest.TestCase):
    """This is a class with functions to test mesh_wing_arrays."""

    def _assert_matches_mesh_wing(
        self,
        wing: ps.geometry.wing.Wing,
        plan: _mesh_arrays.MeshPlan,
        corners: _mesh_arrays.WingCorners,
    ) -> None:
        """Asserts that WingCorners match the Panel corners mesh_wing gave a Wing, in
        the same Panel grid order.

        :param wing: The meshed Wing.
        :param plan: The MeshPlan the WingCorners were meshed with.
        :param corners: The WingCorners.
        :return: None
        """
        assert wing.panels is not None
        expected_shape = (plan.num_chordwise_panels, plan.num_spanwise_panels, 3)
        for name, grid in zip(_CORNER_NAMES, corners):
            expected = np.array(
                [[getattr(panel, name) for panel in row] for row in wing.panels],
                dtype=float,
            )
            self.assertEqual(grid.shape, expected_shape)
            self.assertEqual(expected.shape, expected_shape)
            np.testing.assert_allclose(
                grid, expected, rtol=0.0, atol=_CORNER_TOLERANCE, err_msg=name
            )

    def test_matches_mesh_wing_on_wing_fixtures(self) -> None:
        """Test that mesh_wing_arrays gives the Panel corners mesh_wing gives each Wing
        fixture, for every symmetry type, both spacings, and control surfaces (including
        a type 4 Wing whose mirrored half deflects the other way)."""
        for make in _WING_FIXTURE_MAKERS:
            for wing in mesh_arrays_fixtures.make_processed_wings_fixture(make()):
                with self.subTest(wing=wing.name, symmetry_type=wing.symmetry_type):
                    plan = _mesh_arrays.build_mesh_plan(wing)
                    corners = _mesh_arrays.mesh_wing_arrays(
                        plan,
                        mesh_arrays_fixtures.make_wing_step_parameters_fixture(wing),
                    )
                    self._assert_matches_mesh_wing(wing, plan, corners)

    def test_matches_mesh_wing_on_airplane_fixtures(self) -> None:
        """Test that mesh_wing_arrays gives the Panel corners mesh_wing gives each Wing
        of the multiple Wing Airplane fixtures."""
        for airplane in (
            geometry_fixtures.make_multi_wing_airplane_fixture(),
            geometry_fixtures.make_type_5_wing_airplane_fixture(),
        ):
            for wing in airplane.wings:
                with self.subTest(airplane=airplane.name, wing=wing.name):
                    plan = _mesh_arrays.build_mesh_plan(wing)
                    corners = _mesh_arrays.mesh_wing_arrays(
                        plan,
                        mesh_arrays_fixtures.make_wing_step_parameters_fixture(wing),
                    )
                    self._assert_matches_mesh_wing(wing, plan, corners)

    def test_matches_mesh_wing_on_random_wings(self) -> None:
        """Test that mesh_wing_arrays gives the Panel corners mesh_wing gives random
        Wings, and that the random Wings cover every symmetry type, exploded strips,
        edge defined Wings, and a type 4 Wing whose mirrored half deflects the other
        way."""
        covered: set[int | str | None] = set()
        for case, processed_wing in enumerate(
            mesh_arrays_fixtures.make_random_processed_wings_fixture(seed=0, count=150)
        ):
            with self.subTest(case=case, wing=processed_wing.name):
                plan = _mesh_arrays.build_mesh_plan(processed_wing)
                corners = _mesh_arrays.mesh_wing_arrays(
                    plan,
                    mesh_arrays_fixtures.make_wing_step_parameters_fixture(
                        processed_wing
                    ),
                )
                self._assert_matches_mesh_wing(processed_wing, plan, corners)
            covered.add(processed_wing.symmetry_type)
            covered.add(processed_wing.spanwise_mesh)
            if plan.gridMirroredMclPoints_Wcs_Lp is not None:
                covered.add("mirrored deflection")

        self.assertEqual(
            covered,
            {
                1,
                2,
                3,
                4,
                "trapezoidal",
                "exploded",
                "edge_defined",
                "mirrored deflection",
            },
        )

    def test_matches_movement_airplanes(self) -> None:
        """Test that, with MeshPlans built from a movement's first Airplane,
        mesh_wing_arrays gives each later time step's Wings, from the movement's
        WingStepParameters, the Panel corners of the Airplane the movement builds at
        that time step."""
        movements = (
            movement_fixtures.make_basic_movement_fixture(),
            movement_fixtures.make_movement_with_multiple_airplanes_fixture(),
            movement_fixtures.make_moving_aeroelastic_movement_fixture(),
            free_flight_movement_fixtures.make_moving_free_flight_aeroelastic_movement_fixture(),
        )
        for movement in movements:
            delta_time = movement.delta_time
            num_steps = movement.num_steps
            for airplane_movement in movement.airplane_movements:
                first_airplane = airplane_movement.generate_airplane_at_time_step(
                    0, delta_time
                )
                plans = [
                    _mesh_arrays.build_mesh_plan(wing) for wing in first_airplane.wings
                ]
                for step in sorted({1, num_steps // 2, num_steps - 1}):
                    # noinspection PyProtectedMember
                    parameters = (
                        airplane_movement._generate_airplane_parameters_at_time_step(
                            step, delta_time
                        )
                    )
                    airplane = airplane_movement.generate_airplane_at_time_step(
                        step, delta_time
                    )
                    for plan, wing_parameters, wing in zip(
                        plans, parameters.wings, airplane.wings
                    ):
                        with self.subTest(
                            movement=type(movement).__name__, step=step, wing=wing.name
                        ):
                            corners = _mesh_arrays.mesh_wing_arrays(
                                plan, wing_parameters
                            )
                            self._assert_matches_mesh_wing(wing, plan, corners)

    def test_corners_are_read_only_and_contiguous(self) -> None:
        """Test that the returned corner arrays are read only and C contiguous, for
        every symmetry type's column ordering."""
        for make in _WING_FIXTURE_MAKERS[:5]:
            for wing in mesh_arrays_fixtures.make_processed_wings_fixture(make()):
                with self.subTest(wing=wing.name):
                    corners = _mesh_arrays.mesh_wing_arrays(
                        _mesh_arrays.build_mesh_plan(wing),
                        mesh_arrays_fixtures.make_wing_step_parameters_fixture(wing),
                    )
                    for grid in corners:
                        self.assertFalse(grid.flags.writeable)
                        self.assertTrue(grid.flags.c_contiguous)


class TestMeshArraysHelpers(unittest.TestCase):
    """This is a class with functions to test each of the array meshing's helper
    functions against the object code it replaces, so that a difference points at the
    helper that caused it."""

    wings: list[ps.geometry.wing.Wing]
    plans: list[_mesh_arrays.MeshPlan]

    @classmethod
    def setUpClass(cls) -> None:
        """Builds the processed Wings and their MeshPlans once for every test.

        :return: None
        """
        cls.wings = _helper_test_wings()
        cls.plans = [_mesh_arrays.build_mesh_plan(wing) for wing in cls.wings]

    def test_stack_rot_R_act_ixyz_matches_generate_rot_T(self) -> None:
        """Test that _stack_rot_R_act_ixyz gives the rotation matrices
        generate_rot_T(..., passive=False, intrinsic=True, order="xyz") gives, for
        random angles over the whole circle and for the edge angles."""
        rng = np.random.default_rng(2)
        edge_angles = np.array(
            [
                [0.0, 0.0, 0.0],
                [90.0, -90.0, 180.0],
                [-180.0, 90.0, -90.0],
                [45.0, 90.0, 45.0],
            ],
            dtype=float,
        )
        stackAngles_ixyz = np.vstack(
            [edge_angles, rng.uniform(-180.0, 180.0, (2000, 3))]
        )

        stack_R_act = _mesh_arrays._stack_rot_R_act_ixyz(stackAngles_ixyz)

        expected = np.stack(
            [
                _transformations.generate_rot_T(
                    angles, passive=False, intrinsic=True, order="xyz"
                )[:3, :3]
                for angles in stackAngles_ixyz
            ]
        )
        np.testing.assert_allclose(
            stack_R_act, expected, rtol=0.0, atol=_MATRIX_TOLERANCE
        )

    def test_stack_T_pas_Wcs_Lp_to_Wn_Ler_matches_wing(self) -> None:
        """Test that _stack_T_pas_Wcs_Lp_to_Wn_Ler gives each Wing's
        children_T_pas_Wcs_Lp_to_Wn_Ler from its WingCrossSections' positions and
        angles."""
        for wing in self.wings:
            with self.subTest(wing=wing.name):
                parameters = mesh_arrays_fixtures.make_wing_step_parameters_fixture(
                    wing
                )
                stackT_pas_Wcs_Lp_to_Wn_Ler = (
                    _mesh_arrays._stack_T_pas_Wcs_Lp_to_Wn_Ler(
                        parameters.stackLp_Wcsp_Lpp,
                        parameters.stackAngles_Wcsp_to_Wcs_ixyz,
                    )
                )
                np.testing.assert_allclose(
                    stackT_pas_Wcs_Lp_to_Wn_Ler,
                    np.stack(wing.children_T_pas_Wcs_Lp_to_Wn_Ler),
                    rtol=0.0,
                    atol=_MATRIX_TOLERANCE,
                )

    def test_T_pas_G_Cg_to_Wn_Ler_matches_wing(self) -> None:
        """Test that _T_pas_G_Cg_to_Wn_Ler gives each Wing's T_pas_G_Cg_to_Wn_Ler, for
        every symmetry type (only types 2 and 3 include the reflection)."""
        for wing, plan in zip(self.wings, self.plans):
            with self.subTest(wing=wing.name, symmetry_type=wing.symmetry_type):
                T_pas_G_Cg_to_Wn_Ler = _mesh_arrays._T_pas_G_Cg_to_Wn_Ler(
                    plan, mesh_arrays_fixtures.make_wing_step_parameters_fixture(wing)
                )
                expected = wing.T_pas_G_Cg_to_Wn_Ler
                assert expected is not None
                np.testing.assert_allclose(
                    T_pas_G_Cg_to_Wn_Ler, expected, rtol=0.0, atol=_MATRIX_TOLERANCE
                )

    def test_mcl_points_match_get_mcs_points(self) -> None:
        """Test that each MeshPlan's camber line points (and a type 4 Wing's mirrored
        half's) are the points mesh_wing's _get_mcl_points and _get_mcs_points give each
        WingCrossSection's deflected Airfoil, which _mcl_points_Wcs_Lp computes."""
        identity_T = np.eye(4, dtype=float)
        for wing, plan in zip(self.wings, self.plans):
            mirrored = plan.gridMirroredMclPoints_Wcs_Lp
            halves = [("main", plan.gridMclPoints_Wcs_Lp, False)]
            if mirrored is not None:
                halves.append(("mirrored", mirrored, True))
            for half, gridMclPoints_Wcs_Lp, is_mirrored in halves:
                for index, wing_cross_section in enumerate(wing.wing_cross_sections):
                    with self.subTest(wing=wing.name, half=half, index=index):
                        # As in mesh_wing, the mirrored half deflects each asymmetric
                        # control surface the other way.
                        deflection = wing_cross_section.control_surface_deflection
                        if (
                            is_mirrored
                            and wing_cross_section.control_surface_symmetry_type
                            == "asymmetric"
                        ):
                            deflection = -deflection
                        airfoil = wing_cross_section.airfoil.add_control_surface(
                            deflection=deflection,
                            hinge_point=wing_cross_section.control_surface_hinge_point,
                        )

                        # With identity transformations and spanwise coordinates of 0.0
                        # and 1.0, the inner edge of _get_mcs_points' one column is the
                        # WingCrossSection's camber line points in its own axes.
                        (
                            inner_mclY,
                            inner_mclX,
                            outer_mclY,
                            outer_mclX,
                        ) = _meshing._get_mcl_points(
                            airfoil, airfoil, plan.chordwise_coordinates
                        )
                        Fipp, _, Bipp, _ = _meshing._get_mcs_points(
                            identity_T,
                            identity_T,
                            wing_cross_section,
                            wing_cross_section,
                            inner_mclY,
                            inner_mclX,
                            outer_mclY,
                            outer_mclX,
                            np.array([0.0, 1.0], dtype=float),
                        )
                        expected = np.vstack([Fipp[:, 0], Bipp[-1:, 0]])

                        np.testing.assert_allclose(
                            gridMclPoints_Wcs_Lp[index],
                            expected,
                            rtol=0.0,
                            atol=_CORNER_TOLERANCE,
                        )

    def test_column_edge_points_match_get_mcs_points(self) -> None:
        """Test that _column_edge_points_Wn_Ler gives, for each wing section, the corner
        points mesh_wing's _get_mcs_points gives it (in wing axes, relative to the
        leading edge root point), when both use the Wing's own transformations."""
        for wing, plan in zip(self.wings, self.plans):
            children_T_pas_Wcs_Lp_to_Wn_Ler = wing.children_T_pas_Wcs_Lp_to_Wn_Ler
            gridInnerPoints_Wn_Ler, gridOuterPoints_Wn_Ler = (
                _mesh_arrays._column_edge_points_Wn_Ler(
                    plan,
                    plan.gridMclPoints_Wcs_Lp,
                    np.stack(children_T_pas_Wcs_Lp_to_Wn_Ler),
                )
            )
            wing_cross_sections = wing.wing_cross_sections
            for wing_section in range(len(wing_cross_sections) - 1):
                with self.subTest(wing=wing.name, wing_section=wing_section):
                    inner_wing_cross_section = wing_cross_sections[wing_section]
                    outer_wing_cross_section = wing_cross_sections[wing_section + 1]
                    columns = plan.column_wing_section == wing_section
                    spanwise_coordinates = np.append(
                        plan.column_inner_fraction[columns],
                        plan.column_outer_fraction[columns][-1],
                    )
                    (
                        inner_mclY,
                        inner_mclX,
                        outer_mclY,
                        outer_mclX,
                    ) = _meshing._get_mcl_points(
                        inner_wing_cross_section.airfoil.add_control_surface(
                            deflection=inner_wing_cross_section.control_surface_deflection,
                            hinge_point=inner_wing_cross_section.control_surface_hinge_point,
                        ),
                        outer_wing_cross_section.airfoil.add_control_surface(
                            deflection=outer_wing_cross_section.control_surface_deflection,
                            hinge_point=outer_wing_cross_section.control_surface_hinge_point,
                        ),
                        plan.chordwise_coordinates,
                    )
                    expected = _meshing._get_mcs_points(
                        children_T_pas_Wcs_Lp_to_Wn_Ler[wing_section],
                        children_T_pas_Wcs_Lp_to_Wn_Ler[wing_section + 1],
                        inner_wing_cross_section,
                        outer_wing_cross_section,
                        inner_mclY,
                        inner_mclX,
                        outer_mclY,
                        outer_mclX,
                        spanwise_coordinates,
                    )

                    # Each Panel's front corners are on the first M points of its
                    # column's edges and its back corners on their last M.
                    actual = (
                        gridInnerPoints_Wn_Ler[:-1, columns],
                        gridOuterPoints_Wn_Ler[:-1, columns],
                        gridInnerPoints_Wn_Ler[1:, columns],
                        gridOuterPoints_Wn_Ler[1:, columns],
                    )
                    for name, actual_grid, expected_grid in zip(
                        ("Fipp", "Fopp", "Bipp", "Bopp"), actual, expected
                    ):
                        np.testing.assert_allclose(
                            actual_grid,
                            expected_grid,
                            rtol=0.0,
                            atol=_CORNER_TOLERANCE,
                            err_msg=name,
                        )


class TestCheckSymmetryType(unittest.TestCase):
    """This is a class with functions to test that _check_symmetry_type, which
    mesh_wing_arrays calls, rejects a change of symmetry type."""

    def test_symmetry_type_change_raises(self) -> None:
        """Test that a Wing whose values change its symmetry type raises a ValueError (a
        type 4 Wing rotated about its x axis becomes type 5, and a type 2 Wing moved off
        its symmetry plane becomes type 3), where the object path's Airplane would
        instead process the Wing as its new type."""
        cases: tuple[
            tuple[Callable[[], ps.geometry.wing.Wing], dict[str, Any], int], ...
        ] = (
            (
                geometry_fixtures.make_type_4_wing_fixture,
                dict(angles_Gs_to_Wn_ixyz=np.array([10.0, 0.0, 0.0])),
                5,
            ),
            (
                geometry_fixtures.make_type_2_wing_fixture,
                dict(Ler_Gs_Cgs=np.array([0.0, 0.5, 0.0])),
                3,
            ),
        )
        for make, change, new_symmetry_type in cases:
            wing = mesh_arrays_fixtures.make_processed_wings_fixture(make())[0]
            with self.subTest(wing=wing.name):
                plan = _mesh_arrays.build_mesh_plan(wing)
                parameters = mesh_arrays_fixtures.make_wing_step_parameters_fixture(
                    wing
                )._replace(**change)
                for function in (
                    _mesh_arrays._check_symmetry_type,
                    _mesh_arrays.mesh_wing_arrays,
                ):
                    with self.assertRaisesRegex(
                        ValueError,
                        f"type {wing.symmetry_type} symmetry to type "
                        f"{new_symmetry_type} symmetry",
                    ):
                        function(plan, parameters)


if __name__ == "__main__":
    unittest.main()
