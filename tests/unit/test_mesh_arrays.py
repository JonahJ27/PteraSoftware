"""This module contains a class to test the array meshing's per run plans."""

import unittest

import numpy as np

import pterasoftware as ps

# noinspection PyProtectedMember
from pterasoftware import _mesh_arrays
from tests.unit.fixtures import geometry_fixtures


class TestBuildMeshPlan(unittest.TestCase):
    """This is a class with functions to test build_mesh_plan."""

    def test_plans_match_the_processed_wings(self) -> None:
        """Test that each processed Wing's MeshPlan has its symmetry type and panel
        counts, that its columns cover each wing section from 0.0 to 1.0 in order, and
        that its arrays are read only, for every symmetry type (a type 5 Wing is split
        into a type 1 Wing and its type 3 reflection)."""
        makers = (
            geometry_fixtures.make_type_1_wing_fixture,
            geometry_fixtures.make_type_2_wing_fixture,
            geometry_fixtures.make_type_3_wing_fixture,
            geometry_fixtures.make_type_4_wing_fixture,
            geometry_fixtures.make_type_5_wing_fixture,
        )
        for make in makers:
            airplane = ps.geometry.airplane.Airplane(wings=[make()], weight=0.0)
            for wing in airplane.wings:
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

                    for array in (
                        plan.chordwise_coordinates,
                        plan.column_wing_section,
                        plan.column_inner_fraction,
                        plan.column_outer_fraction,
                    ):
                        self.assertFalse(array.flags.writeable)

    def test_unprocessed_wing_is_rejected(self) -> None:
        """Test that a Wing whose symmetry hasn't been processed by an Airplane is
        rejected."""
        with self.assertRaises(ValueError):
            _mesh_arrays.build_mesh_plan(geometry_fixtures.make_type_1_wing_fixture())


if __name__ == "__main__":
    unittest.main()
