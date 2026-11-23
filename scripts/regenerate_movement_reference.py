"""Regenerates the stored movement reference that tests/unit/test_movement_reference.py
compares against.

Run it only when a change is meant to move the geometry the movement classes produce,
and say so in the change's commit. Run from the repository root:

python -u scripts/regenerate_movement_reference.py
"""

import sys
from pathlib import Path

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

# The import needs the repository root on the path, which the line above adds.
from tests.unit.fixtures import movement_reference_fixtures  # noqa: E402

REFERENCE_PATH = REPOSITORY / "tests" / "references" / "movement_reference.npz"

if __name__ == "__main__":
    values = movement_reference_fixtures.collect_movement_reference_values()
    packed = movement_reference_fixtures.pack_values(values)
    np.savez_compressed(
        REFERENCE_PATH,
        keys=packed["keys"],
        shapes=packed["shapes"],
        offsets=packed["offsets"],
        flat=packed["flat"],
    )
    num_numbers = sum(value.size for value in values.values())
    print(
        f"Wrote {len(values)} arrays ({num_numbers} numbers) to {REFERENCE_PATH} "
        f"({REFERENCE_PATH.stat().st_size / 1e3:.0f} kB)."
    )
