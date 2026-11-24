"""Contains the functions that update the unsteady solvers' wakes as arrays, for the
solvers' array path.

Each Wing's wake is a grid of wake ring vortex points, (R + 1, N + 1, 3) for R wake rows
and N spanwise Panels, whose R x N cells are its wake ring vortices. At every time step,
the grid is convected, a new row is shed from the trailing edge in front of it, and its
oldest row is dropped when the wake is truncated. The next time step's wake ring
vortices are then the grid's cells. On the object path, each Wing's grid is stored on
the next time step's Wing (gridWrvp_GP1_CgP1), and two Python loops build the new row
and the ring vortices. These functions replace those loops with the same operations, in
the same order, so their results are bit for bit the same. The velocities that convect
the grid (the induced velocities from the kernels, the freestream, and the body rate)
stay in the solver. See docs/ARRAY_LAYOUTS.md for the layout conventions.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from . import _step_geometry


def new_wake_rows(
    layout_plan: _step_geometry.LayoutPlan,
    stackBlbrvp_GP1_CgP1: np.ndarray,
    stackBrbrvp_GP1_CgP1: np.ndarray,
) -> list[np.ndarray]:
    """Returns the row of wake ring vortex points each Wing sheds, as
    _populate_next_airplanes_wake_vortex_points builds it.

    A Wing's new row is the back left bound ring vortex corners of its trailing edge
    Panels, in spanwise order, followed by the back right corner of the last one.

    :param layout_plan: The run's LayoutPlan.
    :param stackBlbrvp_GP1_CgP1: A (num_panels, 3) ndarray of floats representing the
        next time step's bound ring vortices' back left corners (in the first Airplane's
        geometry axes, relative to the first Airplane's CG). The units are in meters.
    :param stackBrbrvp_GP1_CgP1: The same for the back right corners.
    :return: For each Wing, in global Wing order, a (1, N + 1, 3) ndarray of floats
        representing its new row's points (in the first Airplane's geometry axes,
        relative to the first Airplane's CG), where N is its number of spanwise Panels.
        The units are in meters.
    """
    rows = []
    for (num_chordwise_panels, num_spanwise_panels), offset in zip(
        layout_plan.wing_shapes, layout_plan.wing_offsets
    ):
        trailing_edge_base = offset + (num_chordwise_panels - 1) * num_spanwise_panels
        newRowWrvp_GP1_CgP1 = np.zeros((1, num_spanwise_panels + 1, 3), dtype=float)
        newRowWrvp_GP1_CgP1[0, :num_spanwise_panels] = stackBlbrvp_GP1_CgP1[
            trailing_edge_base : trailing_edge_base + num_spanwise_panels
        ]
        newRowWrvp_GP1_CgP1[0, num_spanwise_panels] = stackBrbrvp_GP1_CgP1[
            trailing_edge_base + num_spanwise_panels - 1
        ]
        rows.append(newRowWrvp_GP1_CgP1)
    return rows


def first_wake_grid(
    newRowWrvp_GP1_CgP1: np.ndarray,
    vRowWrvp_GP1__E: np.ndarray,
    delta_time: float,
) -> np.ndarray:
    """Returns a Wing's wake point grid after the first time step, as
    _populate_next_airplanes_wake_vortex_points builds it: the new row, and the new row
    convected over one time step behind it.

    :param newRowWrvp_GP1_CgP1: A (1, N + 1, 3) ndarray of floats representing the
        Wing's new row of wake ring vortex points (in the first Airplane's geometry
        axes, relative to the first Airplane's CG). The units are in meters.
    :param vRowWrvp_GP1__E: An ndarray of floats that broadcasts to the new row's shape,
        representing the velocity that convects each of its points (in the first
        Airplane's geometry axes, observed from the Earth frame). The units are in
        meters per second.
    :param delta_time: The time between time steps. Its units are in seconds.
    :return: A (2, N + 1, 3) ndarray of floats representing the Wing's wake point grid.
        The units are in meters.
    """
    gridWrvp_GP1_CgP1 = np.copy(newRowWrvp_GP1_CgP1)
    secondNewRowWrvp_GP1_CgP1 = gridWrvp_GP1_CgP1 + vRowWrvp_GP1__E * delta_time
    return np.vstack((gridWrvp_GP1_CgP1, secondNewRowWrvp_GP1_CgP1))


def next_wake_grid(
    gridWrvp_GP1_CgP1: np.ndarray,
    vGridWrvp_GP1__E: np.ndarray,
    newRowWrvp_GP1_CgP1: np.ndarray,
    delta_time: float,
    max_wake_rows: int | None,
) -> np.ndarray:
    """Returns a Wing's wake point grid at the next time step, after the first, as
    _populate_next_airplanes_wake_vortex_points builds it: the current grid convected
    over one time step, with the new row in front of it, and without the oldest row when
    the wake is truncated.

    :param gridWrvp_GP1_CgP1: A (R + 1, N + 1, 3) ndarray of floats representing the
        Wing's current wake point grid (in the first Airplane's geometry axes, relative
        to the first Airplane's CG). The units are in meters.
    :param vGridWrvp_GP1__E: An ndarray of floats that broadcasts to the grid's shape,
        representing the velocity that convects each of its points (in the first
        Airplane's geometry axes, observed from the Earth frame). The units are in
        meters per second.
    :param newRowWrvp_GP1_CgP1: A (1, N + 1, 3) ndarray of floats representing the
        Wing's new row of wake ring vortex points. The units are in meters.
    :param delta_time: The time between time steps. Its units are in seconds.
    :param max_wake_rows: The maximum number of wake rows each Wing keeps, or None if it
        keeps all of them.
    :return: The Wing's next wake point grid, a (R + 2, N + 1, 3) ndarray of floats, or
        (max_wake_rows + 1, N + 1, 3) when truncated. The units are in meters.
    """
    agedGridWrvp_GP1_CgP1 = np.copy(gridWrvp_GP1_CgP1) + vGridWrvp_GP1__E * delta_time
    nextGridWrvp_GP1_CgP1 = np.vstack((newRowWrvp_GP1_CgP1, agedGridWrvp_GP1_CgP1))

    # The point grid has one more row than the vortex grid, because points are vertices
    # and vortices are cells.
    if max_wake_rows is not None and nextGridWrvp_GP1_CgP1.shape[0] > max_wake_rows + 1:
        nextGridWrvp_GP1_CgP1 = nextGridWrvp_GP1_CgP1[: max_wake_rows + 1]
    return nextGridWrvp_GP1_CgP1


class WakeRingVortices(NamedTuple):
    """The next time step's wake ring vortices, from wake_ring_vortices, in the solvers'
    wake order: Wing by Wing in global Wing order, and within a Wing, row by row from
    the newest, spanwise inner."""

    # Four (num_wake_vortices, 3) ndarrays of floats: the wake ring vortices' front
    # left, front right, back left, and back right corners (in the first Airplane's
    # geometry axes, relative to the first Airplane's CG). Their units are in meters.
    stackFlwrvp_GP1_CgP1: np.ndarray
    stackFrwrvp_GP1_CgP1: np.ndarray
    stackBlwrvp_GP1_CgP1: np.ndarray
    stackBrwrvp_GP1_CgP1: np.ndarray

    # A (num_wake_vortices,) ndarray of floats: their strengths. The units are in meters
    # squared per second.
    strengths: np.ndarray

    # A (num_wake_vortices,) ndarray of floats: their initial core radii. The units are
    # in meters.
    r_c0s: np.ndarray


def wake_ring_vortices(
    layout_plan: _step_geometry.LayoutPlan,
    nextGridsWrvp_GP1_CgP1: Sequence[np.ndarray],
    step: int,
    max_wake_rows: int | None,
    bound_strengths: np.ndarray,
    this_wake_strengths: np.ndarray,
    this_wake_r_c0s: np.ndarray,
    wing_standard_mean_chords: np.ndarray,
) -> WakeRingVortices:
    """Returns the next time step's wake ring vortices, as
    _populate_next_airplanes_wake_vortices builds them.

    Each Wing's wake ring vortices are its next wake point grid's cells. Its newest row
    was just shed by its trailing edge Panels, so it takes their bound ring vortices'
    strengths and 3.0% of the Wing's standard mean chord as its initial core radius. Its
    other rows are this time step's rows, one time step older, so they keep this time
    step's strengths and initial core radii (without this time step's oldest row when
    the wake is truncated).

    :param layout_plan: The run's LayoutPlan.
    :param nextGridsWrvp_GP1_CgP1: Each Wing's next wake point grid, in global Wing
        order. The units are in meters.
    :param step: This time step.
    :param max_wake_rows: The maximum number of wake rows each Wing keeps, or None if it
        keeps all of them.
    :param bound_strengths: A (num_panels,) ndarray of floats representing this time
        step's bound ring vortex strengths. The units are in meters squared per second.
    :param this_wake_strengths: A (num_wake_vortices at this time step,) ndarray of
        floats representing this time step's wake ring vortex strengths. The units are
        in meters squared per second.
    :param this_wake_r_c0s: The same for their initial core radii. The units are in
        meters.
    :param wing_standard_mean_chords: A (num_wings,) ndarray of floats representing this
        time step's Wings' standard mean chords. The units are in meters.
    :return: The next time step's WakeRingVortices.
    """
    next_num_rows = step + 1
    this_num_rows = step
    if max_wake_rows is not None:
        next_num_rows = min(step + 1, max_wake_rows)
        this_num_rows = min(step, max_wake_rows)

    parts: dict[str, list[np.ndarray]] = {
        name: [] for name in ("Fl", "Fr", "Bl", "Br", "strengths", "r_c0s")
    }
    this_wing_base = 0
    for (
        nextGridWrvp_GP1_CgP1,
        (num_chordwise_panels, num_spanwise_panels),
        offset,
        standard_mean_chord,
    ) in zip(
        nextGridsWrvp_GP1_CgP1,
        layout_plan.wing_shapes,
        layout_plan.wing_offsets,
        wing_standard_mean_chords,
    ):
        assert nextGridWrvp_GP1_CgP1.shape[0] == next_num_rows + 1
        assert nextGridWrvp_GP1_CgP1.shape[1] == num_spanwise_panels + 1

        # A cell's four corners are four shifted views of the grid, so the whole Wing's
        # wake is four slices, raveled row by row.
        cell_shape = (next_num_rows * num_spanwise_panels, 3)
        parts["Fl"].append(nextGridWrvp_GP1_CgP1[:-1, :-1].reshape(cell_shape))
        parts["Fr"].append(nextGridWrvp_GP1_CgP1[:-1, 1:].reshape(cell_shape))
        parts["Bl"].append(nextGridWrvp_GP1_CgP1[1:, :-1].reshape(cell_shape))
        parts["Br"].append(nextGridWrvp_GP1_CgP1[1:, 1:].reshape(cell_shape))

        trailing_edge_base = offset + (num_chordwise_panels - 1) * num_spanwise_panels
        num_inherited = (next_num_rows - 1) * num_spanwise_panels
        parts["strengths"].append(
            np.concatenate(
                (
                    bound_strengths[
                        trailing_edge_base : trailing_edge_base + num_spanwise_panels
                    ],
                    this_wake_strengths[
                        this_wing_base : this_wing_base + num_inherited
                    ],
                )
            )
        )
        parts["r_c0s"].append(
            np.concatenate(
                (
                    np.full(
                        num_spanwise_panels, 0.03 * standard_mean_chord, dtype=float
                    ),
                    this_wake_r_c0s[this_wing_base : this_wing_base + num_inherited],
                )
            )
        )
        this_wing_base += this_num_rows * num_spanwise_panels

    return WakeRingVortices(
        *(
            np.concatenate(parts[name]).astype(float, copy=False)
            for name in ("Fl", "Fr", "Bl", "Br", "strengths", "r_c0s")
        )
    )
