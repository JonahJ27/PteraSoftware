"""This module contains a class to test the function that determines a Wing's symmetry
type."""

import unittest
from typing import Any

import numpy as np

import pterasoftware as ps

# noinspection PyProtectedMember
from pterasoftware.geometry import _symmetry
from tests.unit.fixtures import geometry_fixtures


class TestWingSymmetryType(unittest.TestCase):
    """This is a class with functions to test wing_symmetry_type."""

    def test_each_symmetry_type_fixture(self) -> None:
        """Test that each symmetry type fixture's Wing is given its symmetry type, and
        the same type the Airplane finds when it processes the Wing."""
        makers = {
            1: geometry_fixtures.make_type_1_wing_fixture,
            2: geometry_fixtures.make_type_2_wing_fixture,
            3: geometry_fixtures.make_type_3_wing_fixture,
            4: geometry_fixtures.make_type_4_wing_fixture,
            5: geometry_fixtures.make_type_5_wing_fixture,
        }
        for expected, make in makers.items():
            with self.subTest(symmetry_type=expected):
                wing = make()
                symmetry_type = _symmetry.wing_symmetry_type(
                    symmetric=wing.symmetric,
                    mirror_only=wing.mirror_only,
                    symmetryNormal_G=wing.symmetryNormal_G,
                    symmetryPoint_G_Cg=wing.symmetryPoint_G_Cg,
                    angles_Gs_to_Wn_ixyz=wing.angles_Gs_to_Wn_ixyz,
                    Ler_Gs_Cgs=wing.Ler_Gs_Cgs,
                )
                self.assertEqual(symmetry_type, expected)

                # A type 5 Wing is split into a type 1 Wing and its type 3 reflection,
                # so the processed Airplane's first Wing has type 1.
                airplane = ps.geometry.airplane.Airplane(wings=[make()], weight=0.0)
                processed_type = airplane.wings[0].symmetry_type
                self.assertEqual(processed_type, 1 if expected == 5 else expected)

    def test_moving_the_root_off_the_plane_changes_type_4_to_type_5(self) -> None:
        """Test that a symmetric Wing whose leading edge root point leaves its symmetry
        plane changes from type 4 to type 5, and that rotating it so its wing axes' y
        axis is no longer normal to the plane does too."""
        wing = geometry_fixtures.make_type_4_wing_fixture()
        arguments: dict[str, Any] = {
            "symmetric": wing.symmetric,
            "mirror_only": wing.mirror_only,
            "symmetryNormal_G": wing.symmetryNormal_G,
            "symmetryPoint_G_Cg": wing.symmetryPoint_G_Cg,
        }
        self.assertEqual(
            _symmetry.wing_symmetry_type(
                **arguments,
                angles_Gs_to_Wn_ixyz=wing.angles_Gs_to_Wn_ixyz,
                Ler_Gs_Cgs=wing.Ler_Gs_Cgs,
            ),
            4,
        )
        self.assertEqual(
            _symmetry.wing_symmetry_type(
                **arguments,
                angles_Gs_to_Wn_ixyz=wing.angles_Gs_to_Wn_ixyz,
                Ler_Gs_Cgs=wing.Ler_Gs_Cgs + np.array([0.0, 0.1, 0.0], dtype=float),
            ),
            5,
        )
        self.assertEqual(
            _symmetry.wing_symmetry_type(
                **arguments,
                angles_Gs_to_Wn_ixyz=wing.angles_Gs_to_Wn_ixyz
                + np.array([0.0, 0.0, 20.0], dtype=float),
                Ler_Gs_Cgs=wing.Ler_Gs_Cgs,
            ),
            5,
        )


if __name__ == "__main__":
    unittest.main()
