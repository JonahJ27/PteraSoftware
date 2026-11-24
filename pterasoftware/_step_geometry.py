"""Contains the functions that build the solvers' per time step geometry as arrays, for
the solvers' array path.

build_layout_plan runs once per run and holds the global Panel layout: where each Wing's
Panels sit in the solvers' per Panel arrays, and which Panels are on each edge of their
Wing. It never changes between time steps. build_step_geometry runs at every geometry
build and computes every per Panel quantity from the Wings' meshes, and bound_r_c0s,
seed_points, and katz_geometry derive the rest of what the solvers' collapse writes. See
docs/ARRAY_LAYOUTS.md for the layout conventions.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from . import _mesh_arrays, _panel, _step_parameters, _transformations, geometry


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


class StepGeometry(NamedTuple):
    """One geometry build's per Panel quantities, in global Panel order, from
    build_step_geometry.

    Every array is read only. Positions are in the first Airplane's geometry axes,
    relative to the first Airplane's CG, and vectors in the first Airplane's geometry
    axes. Units are in meters (or square meters for areas).
    """

    # Four (num_panels, 3) ndarrays of floats: the Panels' front right, front left, back
    # left, and back right corners, as SteadyProblem gives each Panel's Frpp_GP1_CgP1,
    # Flpp_GP1_CgP1, Blpp_GP1_CgP1, and Brpp_GP1_CgP1.
    stackFrpp_GP1_CgP1: np.ndarray
    stackFlpp_GP1_CgP1: np.ndarray
    stackBlpp_GP1_CgP1: np.ndarray
    stackBrpp_GP1_CgP1: np.ndarray

    # A (num_panels, 3) ndarray of floats: the Panels' collocation points, as
    # Panel.Cpp_GP1_CgP1.
    stackCpp_GP1_CgP1: np.ndarray

    # A (num_panels, 3) ndarray of floats: the Panels' unit normals, as
    # Panel.unitNormal_GP1.
    stackUnitNormals_GP1: np.ndarray

    # A (num_panels,) ndarray of floats: the Panels' areas, as Panel.area.
    panel_areas: np.ndarray

    # A (num_wings,) ndarray of floats: each Wing's standard mean chord, as
    # Wing.standard_mean_chord.
    wing_standard_mean_chords: np.ndarray

    # Four (num_panels, 3) ndarrays of floats: the Panels' bound ring vortices' front
    # right, front left, back left, and back right corners, as
    # _initialize_panel_vortices_at writes them into the solver's per step lists.
    stackFrbrvp_GP1_CgP1: np.ndarray
    stackFlbrvp_GP1_CgP1: np.ndarray
    stackBlbrvp_GP1_CgP1: np.ndarray
    stackBrbrvp_GP1_CgP1: np.ndarray


def build_step_geometry(
    layout_plan: LayoutPlan,
    wing_meshes: Sequence[_mesh_arrays.WingMesh],
    wingsCg_GP1_CgP1: Sequence[np.ndarray],
    vInf_GP1__E: np.ndarray,
    delta_time: float,
    last_geometry: StepGeometry | None,
) -> StepGeometry:
    """Builds one geometry build's per Panel quantities and bound ring vortex corners.

    It computes, for every Panel at once, what SteadyProblem and set_derived_GP1_caches
    store on each Panel, what Wing.standard_mean_chord gives each Wing, and what
    _initialize_panel_vortices_at writes, in the same order of operations. The norms (in
    the unit normals, areas, and projected areas) are vectorized, so they, and the
    projected areas' sums, round differently from the object path's one row at a time
    np.linalg.norm and running sum. Everything else is bit for bit the same.

    :param layout_plan: The run's LayoutPlan.
    :param wing_meshes: Each Wing's WingMesh, in global Wing order.
    :param wingsCg_GP1_CgP1: For each Wing, in global Wing order, a (3,) ndarray of
        floats representing its Airplane's Cg_GP1_CgP1 (its CG's position, in the first
        Airplane's geometry axes, relative to the first Airplane's CG). The units are in
        meters.
    :param vInf_GP1__E: A (3,) ndarray of floats representing the freestream velocity
        (in the first Airplane's geometry axes, observed from the Earth frame) at this
        geometry build, which sets the trailing edge Panels' ring vortices' back
        corners. The units are in meters per second.
    :param delta_time: The time between time steps. Its units are in seconds.
    :param last_geometry: The previous time step's StepGeometry, or None at the first
        time step.
    :return: The StepGeometry, with read only arrays.
    """
    # Each corner of every Wing's Panels, in the solvers' order (chordwise row outer,
    # spanwise column inner), in its Airplane's geometry axes, relative to its CG.
    grid_names = ("gridFrpp_G_Cg", "gridFlpp_G_Cg", "gridBlpp_G_Cg", "gridBrpp_G_Cg")
    stacksPerWing_G_Cg = [
        [getattr(wing_mesh, name).reshape(-1, 3) for name in grid_names]
        for wing_mesh in wing_meshes
    ]

    # Each corner in the first Airplane's geometry axes, relative to its CG, with the
    # transformation SteadyProblem applies (Airplane.T_pas_G_Cg_to_GP1_CgP1, through
    # apply_T_to_vectors, which computes each vector's product independently).
    stacksPerWing_GP1_CgP1 = []
    for wing_stacks, Cg_GP1_CgP1 in zip(stacksPerWing_G_Cg, wingsCg_GP1_CgP1):
        T_pas_G_Cg_to_GP1_CgP1 = _transformations.generate_trans_T(
            translations=-Cg_GP1_CgP1, passive=True
        )
        stacksPerWing_GP1_CgP1.append(
            [
                _transformations.apply_T_to_vectors(
                    T_pas_G_Cg_to_GP1_CgP1, stack, is_position=True
                )
                for stack in wing_stacks
            ]
        )
    Fr, Fl, Bl, Br = (
        np.concatenate([wing_stacks[corner] for wing_stacks in stacksPerWing_GP1_CgP1])
        for corner in range(4)
    )
    Fr_G, Fl_G, Bl_G, Br_G = (
        np.concatenate([wing_stacks[corner] for wing_stacks in stacksPerWing_G_Cg])
        for corner in range(4)
    )

    # The legs, front bound vortex points, and collocation points, with
    # set_derived_GP1_caches' formulas.
    stackRightLeg_GP1 = Fr - Br
    stackLeftLeg_GP1 = Bl - Fl
    stackFrbvp_GP1_CgP1 = Br + 0.75 * stackRightLeg_GP1
    stackFlbvp_GP1_CgP1 = Fl + 0.25 * stackLeftLeg_GP1
    stackRightThreeQuarterChord_GP1_CgP1 = Br + 0.25 * stackRightLeg_GP1
    stackLeftThreeQuarterChord_GP1_CgP1 = Fl + 0.75 * stackLeftLeg_GP1
    stackCpp_GP1_CgP1 = stackRightThreeQuarterChord_GP1_CgP1 + 0.5 * (
        stackLeftThreeQuarterChord_GP1_CgP1 - stackRightThreeQuarterChord_GP1_CgP1
    )

    # The unit normals, from the diagonals' cross product (in the first Airplane's
    # geometry axes), and the areas, from the same cross product in each Airplane's own
    # geometry axes, as Panel.unitNormal_GP1 and Panel.area compute them.
    stackCross_GP1 = _panel.cross_rows(Fr - Bl, Fl - Br)
    stackUnitNormals_GP1 = (
        stackCross_GP1 / np.linalg.norm(stackCross_GP1, axis=1)[:, np.newaxis]
    )
    stackCross_G = _panel.cross_rows(Fr_G - Bl_G, Fl_G - Br_G)
    panel_areas = np.linalg.norm(stackCross_G, axis=1) / 2

    # Each Wing's standard mean chord: its projected area on its wing axes' xy plane,
    # with Wing.projected_area's formulas, over its span.
    wing_standard_mean_chords = np.empty(len(wing_meshes), dtype=float)
    for wing_id, (wing_mesh, wing_stacks) in enumerate(
        zip(wing_meshes, stacksPerWing_G_Cg)
    ):
        wingFr_G, wingFl_G, wingBl_G, wingBr_G = wing_stacks
        stackFirstDiagonal_G = wingFr_G - wingBl_G
        stackSecondDiagonal_G = wingFl_G - wingBr_G
        unitWnZ_G = wing_mesh.WnZ_G / np.linalg.norm(wing_mesh.WnZ_G)
        stackProjFirstDiagonalOnPlane_G = (
            stackFirstDiagonal_G
            - (stackFirstDiagonal_G @ unitWnZ_G)[:, np.newaxis] * unitWnZ_G
        )
        stackProjSecondDiagonalOnPlane_G = (
            stackSecondDiagonal_G
            - (stackSecondDiagonal_G @ unitWnZ_G)[:, np.newaxis] * unitWnZ_G
        )
        stackProjDiagonalsOnPlaneCross_G = _panel.cross_rows(
            stackProjFirstDiagonalOnPlane_G, stackProjSecondDiagonalOnPlane_G
        )
        projected_area = float(
            np.sum(np.linalg.norm(stackProjDiagonalsOnPlaneCross_G, axis=1) / 2)
        )
        wing_standard_mean_chords[wing_id] = projected_area / wing_mesh.span

    # The bound ring vortex corners, as _initialize_panel_vortices_at builds them. The
    # front corners are the Panel's front bound vortex points. A Panel's back corners
    # are the next Panel back's front bound vortex points, except on the trailing edge.
    stackFrbrvp_GP1_CgP1 = stackFrbvp_GP1_CgP1
    stackFlbrvp_GP1_CgP1 = stackFlbvp_GP1_CgP1
    stackBlbrvp_GP1_CgP1 = np.empty_like(Bl)
    stackBrbrvp_GP1_CgP1 = np.empty_like(Br)
    for (num_chordwise_panels, num_spanwise_panels), offset in zip(
        layout_plan.wing_shapes, layout_plan.wing_offsets
    ):
        body = slice(offset, offset + (num_chordwise_panels - 1) * num_spanwise_panels)
        next_row = slice(
            offset + num_spanwise_panels,
            offset + num_chordwise_panels * num_spanwise_panels,
        )
        stackBlbrvp_GP1_CgP1[body] = stackFlbvp_GP1_CgP1[next_row]
        stackBrbrvp_GP1_CgP1[body] = stackFrbvp_GP1_CgP1[next_row]

    # The trailing edge Panels' ring vortices' back corners are a quarter of the
    # trailing edge's travel during a time step behind its back corners (pages 37 to 39
    # of "Modeling of aerodynamic forces in flapping flight with the Unsteady Vortex
    # Lattice Method" by Thomas Lambert). At the first time step, the trailing edge's
    # travel is the freestream's. Afterwards, the trailing edge's own motion since the
    # previous time step is subtracted from it.
    is_trailing_edge = layout_plan.is_trailing_edge
    teBl = Bl[is_trailing_edge]
    teBr = Br[is_trailing_edge]
    if last_geometry is None:
        stackBlbrvp_GP1_CgP1[is_trailing_edge] = teBl + vInf_GP1__E * delta_time * 0.25
        stackBrbrvp_GP1_CgP1[is_trailing_edge] = teBr + vInf_GP1__E * delta_time * 0.25
    else:
        lastTeBl = last_geometry.stackBlpp_GP1_CgP1[is_trailing_edge]
        lastTeBr = last_geometry.stackBrpp_GP1_CgP1[is_trailing_edge]
        stackBlbrvp_GP1_CgP1[is_trailing_edge] = (
            teBl + (vInf_GP1__E - (teBl - lastTeBl) / delta_time) * delta_time * 0.25
        )
        stackBrbrvp_GP1_CgP1[is_trailing_edge] = (
            teBr + (vInf_GP1__E - (teBr - lastTeBr) / delta_time) * delta_time * 0.25
        )

    return StepGeometry(
        *(
            _step_parameters.make_read_only(array)
            for array in (
                Fr,
                Fl,
                Bl,
                Br,
                stackCpp_GP1_CgP1,
                stackUnitNormals_GP1,
                panel_areas,
                wing_standard_mean_chords,
                stackFrbrvp_GP1_CgP1,
                stackFlbrvp_GP1_CgP1,
                stackBlbrvp_GP1_CgP1,
                stackBrbrvp_GP1_CgP1,
            )
        )
    )


def bound_r_c0s(layout_plan: LayoutPlan, geometry: StepGeometry) -> np.ndarray:
    """Returns the bound ring vortices' legs' initial core radii, as _collapse_geometry
    writes them.

    They are zero, except the trailing edge Panels' back legs, which coincide with the
    wake row the time step will shed and take the wake's initial core radius, 3.0% of
    their Wing's standard mean chord (Ramasamy and Leishman, 2007).

    :param layout_plan: The run's LayoutPlan.
    :param geometry: The time step's StepGeometry.
    :return: A (num_panels, 4) ndarray of floats, each row being a Panel's right, front,
        left, and back legs' initial core radii. The units are in meters.
    """
    stackWingR_c0s = np.repeat(
        0.03 * geometry.wing_standard_mean_chords,
        [
            num_chordwise_panels * num_spanwise_panels
            for num_chordwise_panels, num_spanwise_panels in layout_plan.wing_shapes
        ],
    )
    r_c0s = np.zeros((layout_plan.num_panels, 4), dtype=float)
    r_c0s[layout_plan.is_trailing_edge, 3] = stackWingR_c0s[
        layout_plan.is_trailing_edge
    ]
    return r_c0s


def seed_points(layout_plan: LayoutPlan, geometry: StepGeometry) -> np.ndarray:
    """Returns the streamline seed points, as _collapse_geometry writes them: the middle
    of each trailing edge Panel's back leg, in global Panel order.

    :param layout_plan: The run's LayoutPlan.
    :param geometry: The time step's StepGeometry.
    :return: A (num_trailing_edge_panels, 3) ndarray of floats representing the seed
        points (in the first Airplane's geometry axes, relative to the first Airplane's
        CG). The units are in meters.
    """
    is_trailing_edge = layout_plan.is_trailing_edge
    teBl = geometry.stackBlpp_GP1_CgP1[is_trailing_edge]
    teBr = geometry.stackBrpp_GP1_CgP1[is_trailing_edge]
    stackSeedPoints_GP1_CgP1: np.ndarray = teBl + 0.5 * (teBr - teBl)
    return stackSeedPoints_GP1_CgP1


def wake_vortex_ages(
    layout_plan: LayoutPlan, step: int, max_wake_rows: int | None, delta_time: float
) -> np.ndarray:
    """Returns the wake ring vortices' ages at a time step, as _collapse_geometry writes
    them.

    Each Wing has min(step, max_wake_rows) chordwise wake rows, newest first, each one
    time step older than the row shed after it, so row c (0 based) has age (c + 1) *
    delta_time. Each row's age is repeated across the Wing's spanwise wake ring
    vortices, and the Wings' blocks follow the global Wing order.

    :param layout_plan: The run's LayoutPlan.
    :param step: The time step.
    :param max_wake_rows: The maximum number of chordwise wake rows each Wing keeps, or
        None if it keeps all of them.
    :param delta_time: The time between time steps. Its units are in seconds.
    :return: A (num_wake_vortices,) ndarray of floats representing the wake ring
        vortices' ages. The units are in seconds.
    """
    num_chordwise_wake_rows = step
    if max_wake_rows is not None:
        num_chordwise_wake_rows = min(step, max_wake_rows)
    row_ages = np.arange(1, num_chordwise_wake_rows + 1, dtype=float) * delta_time
    return np.concatenate(
        [np.zeros(0, dtype=float)]
        + [
            np.repeat(row_ages, num_spanwise_panels)
            for _, num_spanwise_panels in layout_plan.wing_shapes
        ]
    )


class KatzGeometry(NamedTuple):
    """The Katz force method's per Panel geometry, from katz_geometry."""

    # Two (num_panels,) ndarrays of floats: each Panel's chordwise and spanwise lengths.
    # Their units are in meters.
    panel_chord_lengths: np.ndarray
    panel_span_lengths: np.ndarray

    # Two (num_panels, 3) ndarrays of floats: each Panel's chordwise and spanwise unit
    # tangents (in the first Airplane's geometry axes), zero where the length is zero.
    stackChordwiseTangents_GP1: np.ndarray
    stackSpanwiseTangents_GP1: np.ndarray


def katz_geometry(geometry: StepGeometry) -> KatzGeometry:
    """Returns the Katz force method's per Panel geometry, as
    _collapse_geometry_katz_data computes it from each Panel's legs.

    The chordwise tangent is the average of the right leg and the reversed left leg, and
    the spanwise tangent the average of the front leg and the reversed back leg, each
    normalized. The norms are vectorized, so they round differently from the object
    path's one row at a time np.linalg.norm.

    :param geometry: The time step's StepGeometry.
    :return: The KatzGeometry.
    """
    Fr = geometry.stackFrpp_GP1_CgP1
    Fl = geometry.stackFlpp_GP1_CgP1
    Bl = geometry.stackBlpp_GP1_CgP1
    Br = geometry.stackBrpp_GP1_CgP1

    # The legs, as Panel.rightLeg_GP1, frontLeg_GP1, leftLeg_GP1, and backLeg_GP1.
    stackRightLeg_GP1 = Fr - Br
    stackFrontLeg_GP1 = Fl - Fr
    stackLeftLeg_GP1 = Bl - Fl
    stackBackLeg_GP1 = Br - Bl

    stackChordwise_GP1 = (stackRightLeg_GP1 - stackLeftLeg_GP1) / 2
    stackSpanwise_GP1 = (stackFrontLeg_GP1 - stackBackLeg_GP1) / 2
    panel_chord_lengths = np.linalg.norm(stackChordwise_GP1, axis=1)
    panel_span_lengths = np.linalg.norm(stackSpanwise_GP1, axis=1)

    # Divide only where the length is positive, leaving zero tangents elsewhere, as the
    # object path does.
    stackChordwiseTangents_GP1 = np.zeros_like(stackChordwise_GP1)
    has_chord = panel_chord_lengths > 0
    stackChordwiseTangents_GP1[has_chord] = (
        stackChordwise_GP1[has_chord] / panel_chord_lengths[has_chord, np.newaxis]
    )
    stackSpanwiseTangents_GP1 = np.zeros_like(stackSpanwise_GP1)
    has_span = panel_span_lengths > 0
    stackSpanwiseTangents_GP1[has_span] = (
        stackSpanwise_GP1[has_span] / panel_span_lengths[has_span, np.newaxis]
    )
    return KatzGeometry(
        panel_chord_lengths=panel_chord_lengths,
        panel_span_lengths=panel_span_lengths,
        stackChordwiseTangents_GP1=stackChordwiseTangents_GP1,
        stackSpanwiseTangents_GP1=stackSpanwiseTangents_GP1,
    )
