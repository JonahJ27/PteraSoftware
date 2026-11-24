"""Contains the functions that build the solvers' per time step geometry as arrays, for
the solvers' array path.

build_layout_plan runs once per run and holds the global Panel layout: where each Wing's
Panels sit in the solvers' per Panel arrays, and which Panels are on each edge of their
Wing. It never changes between time steps. See docs/ARRAY_LAYOUTS.md for the layout
conventions.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from . import _step_parameters, geometry


class LayoutPlan(NamedTuple):
    """The global Panel layout, fixed for a whole run."""

    # Each Wing's (num_chordwise_panels, num_spanwise_panels), in global order (every
    # Airplane's Wings in order, after symmetry processing).
    wing_shapes: tuple[tuple[int, int], ...]

    # Each Wing's first index in the solvers' per Panel arrays.
    wing_offsets: tuple[int, ...]

    # The total number of Panels.
    num_panels: int

    # Four (num_panels,) ndarrays of bools, one entry per Panel in global order: whether
    # the Panel is on its Wing's leading edge (its first chordwise row), trailing edge
    # (its last chordwise row), left edge (its first spanwise column), or right edge
    # (its last spanwise column).
    is_leading_edge: np.ndarray
    is_trailing_edge: np.ndarray
    is_left_edge: np.ndarray
    is_right_edge: np.ndarray


def build_layout_plan(
    airplanes: Sequence[geometry.airplane.Airplane],
) -> LayoutPlan:
    """Builds the LayoutPlan from one time step's Airplanes.

    Within each Wing, Panels are in row major order (chordwise row outer, spanwise
    column inner), as the solvers lay them out.

    :param airplanes: The Airplanes at any time step (their Panel counts don't change
        between time steps), with meshed Wings.
    :return: The LayoutPlan, with read only arrays.
    """
    wing_shapes: list[tuple[int, int]] = []
    wing_offsets: list[int] = []
    flags: dict[str, list[np.ndarray]] = {
        "leading": [],
        "trailing": [],
        "left": [],
        "right": [],
    }
    offset = 0
    for airplane in airplanes:
        for wing in airplane.wings:
            num_chordwise_panels = wing.num_chordwise_panels
            num_spanwise_panels = wing.num_spanwise_panels
            assert num_spanwise_panels is not None
            wing_shapes.append((num_chordwise_panels, num_spanwise_panels))
            wing_offsets.append(offset)
            offset += num_chordwise_panels * num_spanwise_panels

            rows, columns = np.meshgrid(
                np.arange(num_chordwise_panels),
                np.arange(num_spanwise_panels),
                indexing="ij",
            )
            flags["leading"].append((rows == 0).ravel())
            flags["trailing"].append((rows == num_chordwise_panels - 1).ravel())
            flags["left"].append((columns == 0).ravel())
            flags["right"].append((columns == num_spanwise_panels - 1).ravel())

    def stacked(name: str) -> np.ndarray:
        return _step_parameters.make_read_only(np.concatenate(flags[name]).astype(bool))

    return LayoutPlan(
        wing_shapes=tuple(wing_shapes),
        wing_offsets=tuple(wing_offsets),
        num_panels=offset,
        is_leading_edge=stacked("leading"),
        is_trailing_edge=stacked("trailing"),
        is_left_edge=stacked("left"),
        is_right_edge=stacked("right"),
    )
