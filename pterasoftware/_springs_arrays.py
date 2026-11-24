"""Contains the functions that compute the aeroelastic problems' strip structural
properties as arrays, for the solvers' array path.

The torsional spring solve already works on arrays, except for what it reads from the
current time step's Wings: each Panel's area (for its mass) and each strip's span width
(the length of its first chordwise row Panel's front leg). These functions compute the
same values from the time step's geometry, with the same formulas and operation order,
and the strips' inertias from them. See docs/ARRAY_LAYOUTS.md for the layout
conventions.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from . import _step_geometry


def panel_mass_matrix(panel_areas: np.ndarray, wing_density: float) -> np.ndarray:
    """Returns a Wing's Panels' masses, as AeroelasticUnsteadyProblem's
    _calculate_mass_matrix builds them: each Panel's area times the wing density,
    repeated for the x, y, and z components.

    It replaces AeroelasticUnsteadyProblem._calculate_mass_matrix.

    It exists because _calculate_mass_matrix takes a built Wing, and this takes the
    Panel areas the array path already has.

    :param panel_areas: A (M, N) ndarray of floats representing the Wing's Panels'
        areas, in its Panel grid order. The units are in square meters.
    :param wing_density: The wing density. The units are in kilograms per square meter.
    :return: A (M, N, 3) ndarray of floats representing the Panels' masses. The units
        are in kilograms.
    """
    mass_matrix: np.ndarray = (
        np.repeat(panel_areas[:, :, None], 3, axis=2) * wing_density
    )
    return mass_matrix


def wing_panel_areas_and_strip_widths(
    layout_plan: _step_geometry.LayoutPlan,
    geometry: _step_geometry.StepGeometry,
    wing_id: int,
    is_mirror_only: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns a Wing's Panels' areas, in its Panel grid order, and its strips' span
    widths, root to tip, from a time step's geometry.

    It replaces the Panel reads in AeroelasticUnsteadyProblem._calculate_mass_matrix
    (Panel.area) and _calculate_strip_inertias (Panel.frontLeg_G).

    It exists because the spring solve read these values from the current time step's
    Panels, which the array path doesn't use. This reads them from the time step's
    StepGeometry.

    Each strip's span width is the length of its first chordwise row Panel's front leg,
    as AeroelasticUnsteadyProblem's _calculate_strip_inertias reads it. A mirror only
    Wing's Panel grid runs tip to root, so its strips' widths are read from the last
    column to the first.

    The front legs are taken from the corners in the first Airplane's geometry axes,
    which are the Wing's own geometry axes moved by its Airplane's CG. That is zero for
    the only Airplane the array path supports, so the legs are bit for bit the object
    path's.

    :param layout_plan: The run's LayoutPlan.
    :param geometry: The time step's StepGeometry.
    :param wing_id: The Wing's index in global Wing order.
    :param is_mirror_only: Whether the Wing is mirror only (symmetry type 2 or 3).
    :return: A (M, N) ndarray of floats representing the Panels' areas (square meters),
        and a (N,) ndarray of floats representing the strips' span widths (meters), root
        to tip.
    """
    num_chordwise_panels, num_spanwise_panels = layout_plan.wing_shapes[wing_id]
    offset = layout_plan.wing_offsets[wing_id]
    panel_areas = geometry.panel_areas[
        offset : offset + num_chordwise_panels * num_spanwise_panels
    ].reshape(num_chordwise_panels, num_spanwise_panels)

    first_row = slice(offset, offset + num_spanwise_panels)
    stackFrontLegs_GP1 = (
        geometry.stackFlpp_GP1_CgP1[first_row] - geometry.stackFrpp_GP1_CgP1[first_row]
    )
    if is_mirror_only:
        stackFrontLegs_GP1 = stackFrontLegs_GP1[::-1]
    # The norms are taken one leg at a time with np.linalg.norm, as the object path
    # takes them, so they are bit for bit the same.
    strip_widths = np.array(
        [float(np.linalg.norm(frontLeg_GP1)) for frontLeg_GP1 in stackFrontLegs_GP1],
        dtype=float,
    )
    return panel_areas, strip_widths


def strip_inertias(
    mass_matrix: np.ndarray,
    wing_cross_section_chords: Sequence[float],
    strip_widths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns each of a Wing's strips' torsional and flapping axis inertias, as
    AeroelasticUnsteadyProblem's _calculate_strip_inertias computes them.

    It replaces AeroelasticUnsteadyProblem._calculate_strip_inertias.

    It exists because _calculate_strip_inertias reads the strip widths from the Wing's
    Panels. This takes them as an array, from wing_panel_areas_and_strip_widths.

    The torsional inertia is (1 / 2) * M * L^2 and the flapping axis inertia is (1 / 12)
    * M * (L^2 + W^2) + M * d^2, where M is the strip's mass, L its mean chord, W its
    span width, and d the distance from the flapping axis to its centroid, accumulated
    in half span width steps.

    :param mass_matrix: A (M, N, 3) ndarray of floats representing the Wing's Panels'
        masses, with the spanwise axis root to tip. The units are in kilograms.
    :param wing_cross_section_chords: The Wing's WingCrossSections' chords, root to tip.
        The units are in meters.
    :param strip_widths: A (N,) ndarray of floats representing the strips' span widths,
        root to tip. The units are in meters.
    :return: Two (N,) ndarrays of floats representing the strips' torsional and flapping
        axis inertias, root to tip. The units are in kilogram square meters.
    """
    num_spanwise_panels = strip_widths.shape[0]
    torsional_inertias = np.zeros(num_spanwise_panels, dtype=float)
    flapping_axis_inertias = np.zeros(num_spanwise_panels, dtype=float)
    d = 0.0
    for span_panel in range(num_spanwise_panels):
        mass = mass_matrix[:, span_panel, :].sum()
        L = (
            wing_cross_section_chords[span_panel]
            + wing_cross_section_chords[span_panel + 1]
        ) / 2
        W = float(strip_widths[span_panel])
        d += W / 2
        torsional_inertias[span_panel] = 1 / 2 * mass * (L**2)
        flapping_axis_inertias[span_panel] = 1 / 12 * mass * (
            L**2 + W**2
        ) + mass * (d**2)
        d += W / 2
    return torsional_inertias, flapping_axis_inertias
