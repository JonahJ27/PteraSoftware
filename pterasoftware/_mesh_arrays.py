"""Contains the functions that mesh Wings as arrays, for the solvers' array path.

Meshing is split in two. build_mesh_plan runs once per run and holds everything that
depends only on a Wing's structure (its symmetry type, panel counts, and spacings),
which never changes between time steps. The per time step meshing, which takes a Wing's
WingStepParameters and returns its Panel corners, is built on top of these plans. See
docs/ARRAY_LAYOUTS.md for the layout conventions.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from . import _functions, _step_parameters, geometry


class MeshPlan(NamedTuple):
    """The values of one Wing's mesh that are fixed for a whole run."""

    # The Wing's symmetry type, an int from 1 to 4 (a type 5 Wing has already been split
    # into a type 1 Wing and its type 3 reflection by its Airplane).
    symmetry_type: int

    # The number of chordwise panels.
    num_chordwise_panels: int

    # The Wing's number of spanwise panels: the sum over its wing sections, doubled for
    # a type 4 Wing, whose Panel grid holds both halves.
    num_spanwise_panels: int

    # A (num_chordwise_panels + 1,) ndarray of floats, the normalized chordwise
    # coordinates of the Panels' front and back edges, from 0.0 at the leading edge to
    # 1.0 at the trailing edge.
    chordwise_coordinates: np.ndarray

    # Three (num_spanwise_panels,) ndarrays, one entry per Panel column of the Wing's
    # main half, in the order its wing sections are meshed (root to tip): the index of
    # the column's wing section (its inner WingCrossSection's index), and the normalized
    # spanwise coordinates of the column's inner and outer edges within that wing
    # section. A type 4 Wing's mirrored half reuses them.
    column_wing_section: np.ndarray
    column_inner_fraction: np.ndarray
    column_outer_fraction: np.ndarray


def build_mesh_plan(wing: geometry.wing.Wing) -> MeshPlan:
    """Builds a Wing's MeshPlan from its structure.

    The chordwise and spanwise coordinates are computed exactly as mesh_wing computes
    them, so the array meshing matches it.

    :param wing: The Wing, after its Airplane has processed its symmetry (so its
        symmetry type is set and is 1, 2, 3, or 4).
    :return: The Wing's MeshPlan, with read only arrays.
    """
    symmetry_type = wing.symmetry_type
    if symmetry_type not in (1, 2, 3, 4):
        raise ValueError(
            f"A MeshPlan needs a Wing whose symmetry has been processed (type 1, 2, "
            f"3, or 4), got symmetry type {symmetry_type}."
        )

    num_chordwise_panels = wing.num_chordwise_panels
    if wing.chordwise_spacing == "uniform":
        chordwise_coordinates = np.linspace(0, 1, num_chordwise_panels + 1)
    else:
        chordwise_coordinates = _functions.cosspace(0, 1, num_chordwise_panels + 1)

    wing_sections, inner_fractions, outer_fractions = [], [], []
    wing_cross_sections = wing.wing_cross_sections
    for wing_section in range(len(wing_cross_sections) - 1):
        # As in mesh_wing, each wing section's spanwise panel count and spacing come
        # from its inner WingCrossSection.
        inner_wing_cross_section = wing_cross_sections[wing_section]
        num_spanwise_panels = inner_wing_cross_section.num_spanwise_panels
        assert num_spanwise_panels is not None
        if inner_wing_cross_section.spanwise_spacing == "uniform":
            spanwise_coordinates = np.linspace(0, 1, num_spanwise_panels + 1)
        else:
            spanwise_coordinates = _functions.cosspace(0, 1, num_spanwise_panels + 1)
        wing_sections.append(np.full(num_spanwise_panels, wing_section, dtype=int))
        inner_fractions.append(spanwise_coordinates[:-1])
        outer_fractions.append(spanwise_coordinates[1:])

    column_wing_section = np.concatenate(wing_sections)
    num_main_spanwise_panels = int(column_wing_section.size)
    return MeshPlan(
        symmetry_type=symmetry_type,
        num_chordwise_panels=num_chordwise_panels,
        num_spanwise_panels=(
            2 * num_main_spanwise_panels
            if symmetry_type == 4
            else num_main_spanwise_panels
        ),
        chordwise_coordinates=_step_parameters.make_read_only(
            np.asarray(chordwise_coordinates, dtype=float)
        ),
        column_wing_section=_step_parameters.make_read_only(column_wing_section),
        column_inner_fraction=_step_parameters.make_read_only(
            np.concatenate(inner_fractions).astype(float)
        ),
        column_outer_fraction=_step_parameters.make_read_only(
            np.concatenate(outer_fractions).astype(float)
        ),
    )
