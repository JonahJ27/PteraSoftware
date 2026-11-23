"""This module contains a class to test the movement classes' geometry against a stored
reference."""

import unittest
from pathlib import Path

import numpy as np

from tests.unit.fixtures import movement_reference_fixtures

# The stored reference, and the absolute tolerance its values are compared with. The
# tolerance allows for last bit differences between platforms and NumPy versions, and is
# far below any real change in the geometry.
_REFERENCE_PATH = (
    Path(__file__).resolve().parents[1] / "references" / "movement_reference.npz"
)
_TOLERANCE = 1e-12


class TestMovementReference(unittest.TestCase):
    """This is a class with functions to test the movement classes' geometry against a
    stored reference."""

    def test_movement_values_match_the_reference(self) -> None:
        """Test that every movement fixture still produces the stored reference geometry
        (each WingCrossSection's, Wing's, and Airplane's position and angles, with and
        without deformation and overrides, at several time steps).

        A change that is meant to move these values must regenerate the reference with
        scripts/regenerate_movement_reference.py, and say so in its commit. Values from
        fixtures added after the reference was stored aren't checked until then.
        """
        with np.load(_REFERENCE_PATH) as packed:
            reference = movement_reference_fixtures.unpack_values(dict(packed))
        values = movement_reference_fixtures.collect_movement_reference_values()

        missing = sorted(set(reference) - set(values))
        self.assertEqual(
            missing,
            [],
            f"{len(missing)} reference values are no longer produced (a fixture was "
            f"renamed or removed?). Regenerate the reference if this is intended.",
        )

        mismatched = []
        for key in reference:
            expected, actual = reference[key], values[key]
            if expected.shape != actual.shape or not np.allclose(
                actual, expected, rtol=0.0, atol=_TOLERANCE
            ):
                mismatched.append(key)
        self.assertEqual(
            mismatched[:10],
            [],
            f"{len(mismatched)} of {len(reference)} reference values changed. "
            f"Regenerate the reference only if the change is intended.",
        )


if __name__ == "__main__":
    unittest.main()
