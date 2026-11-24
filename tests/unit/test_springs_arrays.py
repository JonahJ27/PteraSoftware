"""This module contains classes to test the array path's springs functions against the
object code they replace."""

import unittest

import numpy as np

import pterasoftware as ps

# noinspection PyProtectedMember
from pterasoftware import _mesh_arrays, _springs_arrays, _step_geometry
from tests.unit.fixtures import problem_fixtures, springs_arrays_fixtures


class TestSpringsArrays(unittest.TestCase):
    """This is a class with functions to test the springs array functions against
    AeroelasticUnsteadyProblem's _calculate_mass_matrix and _calculate_strip_inertias,
    and the Panel reads they make, on a plain and a mirror only Wing."""

    def setUp(self) -> None:
        """Builds the Airplane's geometry as the array path does, and an aeroelastic
        problem whose mass and inertia methods are the reference."""
        self.airplane = springs_arrays_fixtures.make_strip_wings_airplane_fixture()
        self.wings = self.airplane.wings
        # The second Wing must be mirror only, so the reversed strip order is tested.
        self.assertFalse(self.wings[0].mirror_only)
        self.assertTrue(self.wings[1].mirror_only)
        self.layout_plan = _step_geometry.build_layout_plan([self.airplane])
        self.geometry = _step_geometry.build_step_geometry(
            self.layout_plan,
            [_mesh_arrays.wing_mesh_from_wing(wing) for wing in self.wings],
            [self.airplane.Cg_GP1_CgP1 for _ in self.wings],
            np.array([-10.0, 0.0, 0.0]),
            0.1,
            None,
        )
        self.problem = (
            problem_fixtures.make_aeroelastic_unsteady_problem_with_standard_wing_fixture()
        )

    def _areas_and_strip_widths(
        self, wing_id: int, wing: ps.geometry.wing.Wing
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns a Wing's Panel areas and strip widths from the array function.

        :param wing_id: The Wing's index in global Wing order.
        :param wing: The Wing.
        :return: The Panel areas and the strip widths, root to tip.
        """
        return _springs_arrays.wing_panel_areas_and_strip_widths(
            self.layout_plan, self.geometry, wing_id, wing.mirror_only
        )

    def test_panel_areas_and_strip_widths_match_the_panels(self) -> None:
        """Test that the Panel areas are Panel.area in the Wing's Panel grid order, and
        that the strip widths are exactly the norms of the first chordwise row's Panels'
        front legs, root to tip, which for the mirror only Wing is its Panel grid's
        columns reversed.

        The areas come from the StepGeometry, whose vectorized norms round about one
        unit in the last place differently from Panel.area (step 3 of the arrays
        rewrite), so they are compared to round off.
        """
        for wing_id, wing in enumerate(self.wings):
            with self.subTest(wing=wing.name):
                panel_areas, strip_widths = self._areas_and_strip_widths(wing_id, wing)
                panels = wing.panels
                assert panels is not None
                expected_areas = np.array(
                    [[panel.area for panel in row] for row in panels]
                )
                np.testing.assert_allclose(
                    panel_areas, expected_areas, rtol=1.0e-15, atol=0.0
                )

                num_spanwise_panels = wing.num_spanwise_panels
                assert num_spanwise_panels is not None
                columns = range(num_spanwise_panels)
                if wing.mirror_only:
                    columns = range(num_spanwise_panels - 1, -1, -1)
                expected_widths = np.array(
                    [
                        float(np.linalg.norm(panels[0][column].frontLeg_G))
                        for column in columns
                    ]
                )
                self.assertTrue(np.array_equal(strip_widths, expected_widths))
                # The strips' widths differ, so a wrong order would be caught.
                self.assertFalse(np.array_equal(strip_widths, strip_widths[::-1]))

    def test_panel_mass_matrix_matches_calculate_mass_matrix(self) -> None:
        """Test that panel_mass_matrix, given the Panels' areas, gives exactly
        _calculate_mass_matrix's masses."""
        for wing in self.wings:
            with self.subTest(wing=wing.name):
                panels = wing.panels
                assert panels is not None
                panel_areas = np.array(
                    [[panel.area for panel in row] for row in panels]
                )
                self.assertTrue(
                    np.array_equal(
                        _springs_arrays.panel_mass_matrix(
                            panel_areas, self.problem.wing_density
                        ),
                        self.problem._calculate_mass_matrix(wing),
                    )
                )

    def test_strip_inertias_match_calculate_strip_inertias(self) -> None:
        """Test that strip_inertias gives exactly _calculate_strip_inertias's torsional
        and flapping axis inertias, with the mass matrix in root to tip order, as
        _calculate_wing_deformation passes it."""
        for wing_id, wing in enumerate(self.wings):
            with self.subTest(wing=wing.name):
                _, strip_widths = self._areas_and_strip_widths(wing_id, wing)
                mass_matrix = self.problem._calculate_mass_matrix(wing)
                if wing.mirror_only:
                    mass_matrix = mass_matrix[:, ::-1, :]
                num_spanwise_panels = wing.num_spanwise_panels
                assert num_spanwise_panels is not None
                expected = self.problem._calculate_strip_inertias(
                    wing, num_spanwise_panels, mass_matrix
                )
                actual = _springs_arrays.strip_inertias(
                    mass_matrix,
                    [wcs.chord for wcs in wing.wing_cross_sections],
                    strip_widths,
                )
                for actual_inertias, expected_inertias in zip(actual, expected):
                    self.assertTrue(np.array_equal(actual_inertias, expected_inertias))


class TestSpringSolveOnTheArrayPath(unittest.TestCase):
    """This is a class with functions to test that the aeroelastic spring solve uses the
    springs array functions' values in the right order."""

    def test_deformations_match_the_object_path(self) -> None:
        """Test that the deformations an array path run finds, on a plain and a mirror
        only Wing whose strips have different widths, are the object path's to round
        off.

        The Katz force method with a prescribed wake keeps the two paths' round off
        differences from growing.
        """
        histories = []
        for use_arrays in (False, True):
            problem = (
                springs_arrays_fixtures.make_strip_wings_aeroelastic_problem_fixture()
            )
            solver = ps.aeroelastic_unsteady_ring_vortex_lattice_method.AeroelasticUnsteadyRingVortexLatticeMethodSolver(
                aeroelastic_unsteady_problem=problem
            )
            solver.run(
                prescribed_wake=True,
                show_progress=False,
                calculate_streamlines=False,
                force_method="katz",
                use_arrays=use_arrays,
            )
            histories.append(
                [
                    np.array(history, dtype=float)
                    for history in problem.listDeformationAnglesYRad_Wcsp_to_Wcs_ixyz
                ]
            )
        for wing_id, (object_history, array_history) in enumerate(zip(*histories)):
            with self.subTest(wing_id=wing_id):
                scale = float(np.max(np.abs(object_history)))
                self.assertGreater(scale, 0.0)
                np.testing.assert_allclose(
                    array_history, object_history, rtol=1.0e-10, atol=1.0e-10 * scale
                )


if __name__ == "__main__":
    unittest.main()
