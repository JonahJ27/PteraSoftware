"""Contains the functions that compute the unsteady solvers' loads as arrays, for the
solvers' array path.

Most of both force methods already work on the solvers' per Panel arrays. These
functions replace the parts that walked the Panels: the effective line vortex strengths
of the Kutta-Joukowski method, the vorticity densities and differences of the Katz
method (all of which read each Panel's edge flags to find its neighbors' strengths),
process_solver_loads' per Panel and per Airplane bookkeeping, and the aeroelastic
solver's strip leading edge points. Each one computes the same operations, in the same
order, as the code it replaces, so its results are bit for bit the same. See
docs/ARRAY_LAYOUTS.md for the layout conventions.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from . import (
    _functions,
    _step_geometry,
    _step_parameters,
    _transformations,
    geometry,
)


def effective_line_vortex_strengths(
    layout_plan: _step_geometry.LayoutPlan,
    strengths: np.ndarray,
    last_strengths: np.ndarray,
    step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns the effective strengths of every Panel's bound ring vortex's right,
    front, left, and back line vortices, as _calculate_loads_joukowski finds them.

    A line vortex shared by two Panels' ring vortices has half the difference of their
    strengths, and one on a Wing's edge has its Panel's full strength. A trailing edge
    Panel's back line vortex is partly canceled by the newest wake row, which carries
    the Panel's strength at the last time step, so its effective strength is the
    difference (or the full strength at the first time step, before any wake exists).

    :param layout_plan: The run's LayoutPlan.
    :param strengths: A (num_panels,) ndarray of floats representing the bound ring
        vortices' strengths at this time step. The units are in meters squared per
        second.
    :param last_strengths: A (num_panels,) ndarray of floats representing their
        strengths at the last time step. The units are in meters squared per second.
    :param step: The time step.
    :return: Four (num_panels,) ndarrays of floats representing the right, front, left,
        and back line vortices' effective strengths. The units are in meters squared per
        second.
    """
    right = np.where(
        layout_plan.is_right_edge,
        strengths,
        (strengths - strengths[layout_plan.right_neighbors]) / 2,
    )
    front = np.where(
        layout_plan.is_leading_edge,
        strengths,
        (strengths - strengths[layout_plan.front_neighbors]) / 2,
    )
    left = np.where(
        layout_plan.is_left_edge,
        strengths,
        (strengths - strengths[layout_plan.left_neighbors]) / 2,
    )
    if step == 0:
        trailing_edge_back = strengths
    else:
        trailing_edge_back = strengths - last_strengths
    back = np.where(
        layout_plan.is_trailing_edge,
        trailing_edge_back,
        (strengths - strengths[layout_plan.back_neighbors]) / 2,
    )
    return right, front, left, back


def chordwise_vorticity_differences(
    layout_plan: _step_geometry.LayoutPlan, strengths: np.ndarray
) -> np.ndarray:
    """Returns each Panel's bound ring vortex strength minus that of the Panel in front
    of it (zero in front of a leading edge Panel), as
    _calculate_chordwise_vorticity_differences finds them.

    :param layout_plan: The run's LayoutPlan.
    :param strengths: A (num_panels,) ndarray of floats representing the bound ring
        vortices' strengths. The units are in meters squared per second.
    :return: A (num_panels,) ndarray of floats representing the differences. The units
        are in meters squared per second.
    """
    differences: np.ndarray = np.where(
        layout_plan.is_leading_edge,
        strengths,
        strengths - strengths[layout_plan.front_neighbors],
    )
    return differences


def chordwise_vorticity_densities(
    layout_plan: _step_geometry.LayoutPlan,
    strengths: np.ndarray,
    panel_chord_lengths: np.ndarray,
) -> np.ndarray:
    """Returns each Panel's average chordwise bound vorticity density, as
    _calculate_chordwise_vorticity_densities finds them: its chordwise circulation
    difference over its chord length, or zero for a Panel with no chord length.

    :param layout_plan: The run's LayoutPlan.
    :param strengths: A (num_panels,) ndarray of floats representing the bound ring
        vortices' strengths. The units are in meters squared per second.
    :param panel_chord_lengths: A (num_panels,) ndarray of floats representing the
        Panels' chord lengths. The units are in meters.
    :return: A (num_panels,) ndarray of floats representing the densities. The units are
        in meters per second.
    """
    circulation_jumps = chordwise_vorticity_differences(layout_plan, strengths)
    densities = np.zeros(layout_plan.num_panels, dtype=float)
    has_chord = panel_chord_lengths > 0
    densities[has_chord] = circulation_jumps[has_chord] / panel_chord_lengths[has_chord]
    return densities


def spanwise_vorticity_densities(
    layout_plan: _step_geometry.LayoutPlan,
    strengths: np.ndarray,
    panel_span_lengths: np.ndarray,
) -> np.ndarray:
    """Returns each Panel's average spanwise bound vorticity density, as
    _calculate_spanwise_vorticity_densities finds them.

    Each Panel is attributed half of each shared left or right line vortex's net
    circulation and all of an edge one's, signed so the density is positive when the
    strengths increase from left to right, and the attributed circulation is divided by
    the Panel's span length (or the density is zero for a Panel with no span length).

    :param layout_plan: The run's LayoutPlan.
    :param strengths: A (num_panels,) ndarray of floats representing the bound ring
        vortices' strengths. The units are in meters squared per second.
    :param panel_span_lengths: A (num_panels,) ndarray of floats representing the
        Panels' span lengths. The units are in meters.
    :return: A (num_panels,) ndarray of floats representing the densities. The units are
        in meters per second.
    """
    left_leg_circulations = np.where(
        layout_plan.is_left_edge,
        strengths,
        strengths - strengths[layout_plan.left_neighbors],
    )
    left_shares = np.where(layout_plan.is_left_edge, 1.0, 0.5)
    right_leg_circulations = np.where(
        layout_plan.is_right_edge,
        -strengths,
        strengths[layout_plan.right_neighbors] - strengths,
    )
    right_shares = np.where(layout_plan.is_right_edge, 1.0, 0.5)
    attributed_circulations = (
        left_shares * left_leg_circulations + right_shares * right_leg_circulations
    )
    densities = np.zeros(layout_plan.num_panels, dtype=float)
    has_span = panel_span_lengths > 0
    densities[has_span] = (
        attributed_circulations[has_span] / panel_span_lengths[has_span]
    )
    return densities


class StepLoads(NamedTuple):
    """One aerodynamic evaluation's loads, from step_loads.

    Every array is read only. The Panel arrays are in global Panel order, the Airplane
    arrays in Airplane order, and the Wing arrays in global Wing order. Forces are in
    Newtons, and moments in Newton meters.
    """

    # Four (num_panels, 3) ndarrays of floats: each Panel's force (in the first
    # Airplane's geometry axes and in wind axes) and moment (in the first Airplane's
    # geometry axes and in wind axes, relative to the first Airplane's CG), as
    # process_solver_loads gives each Panel's forces_GP1, forces_W, moments_GP1_CgP1,
    # and moments_W_CgP1.
    stackPanelForces_GP1: np.ndarray
    stackPanelForces_W: np.ndarray
    stackPanelMoments_GP1_CgP1: np.ndarray
    stackPanelMoments_W_CgP1: np.ndarray

    # Five (num_airplanes, 3) ndarrays of floats: each Airplane's total force (in wind
    # axes and in its geometry axes) and total moment (in wind axes, relative to the
    # first Airplane's CG; in its geometry axes, relative to its CG; and in wind axes,
    # relative to its CG), as process_solver_loads gives each Airplane's forces_W,
    # forces_G, moments_W_CgP1, moments_G_Cg, and moments_W_Cg.
    stackAirplaneForces_W: np.ndarray
    stackAirplaneForces_G: np.ndarray
    stackAirplaneMoments_W_CgP1: np.ndarray
    stackAirplaneMoments_G_Cg: np.ndarray
    stackAirplaneMoments_W_Cg: np.ndarray

    # Five (num_airplanes, 3) ndarrays of floats: the matching coefficients, as
    # process_solver_loads gives each Airplane's forceCoefficients_W,
    # forceCoefficients_G, momentCoefficients_W_CgP1, momentCoefficients_G_Cg, and
    # momentCoefficients_W_Cg. They are unitless.
    stackAirplaneForceCoefficients_W: np.ndarray
    stackAirplaneForceCoefficients_G: np.ndarray
    stackAirplaneMomentCoefficients_W_CgP1: np.ndarray
    stackAirplaneMomentCoefficients_G_Cg: np.ndarray
    stackAirplaneMomentCoefficients_W_Cg: np.ndarray

    # Two (num_wings, 3) ndarrays of floats: each Wing's total force (in the first
    # Airplane's geometry axes) and moment (in the first Airplane's geometry axes,
    # relative to the first Airplane's CG), summed over its Panels in order, as the flap
    # driven problem sums them for its flap drives.
    stackWingForces_GP1: np.ndarray
    stackWingMoments_GP1_CgP1: np.ndarray


def step_loads(
    layout_plan: _step_geometry.LayoutPlan,
    stackPanelForces_GP1: np.ndarray,
    stackPanelMoments_GP1_CgP1: np.ndarray,
    T_pas_GP1_CgP1_to_W_CgP1: np.ndarray,
    qInf__E: float,
    airplanesCg_GP1_CgP1: Sequence[np.ndarray],
    airplane_reference_dimensions: Sequence[tuple[float, float, float]],
) -> StepLoads:
    """Finds the per Panel loads in wind axes, each Airplane's total loads and their
    coefficients, and each Wing's total loads, as process_solver_loads does.

    process_solver_loads sums each Airplane's loads Panel by Panel, from zero. A sum
    over the rows of a stack adds the rows in the same order, so the totals are bit for
    bit the same.

    :param layout_plan: The run's LayoutPlan.
    :param stackPanelForces_GP1: A (num_panels, 3) ndarray of floats representing each
        Panel's force (in the first Airplane's geometry axes). The units are in Newtons.
    :param stackPanelMoments_GP1_CgP1: A (num_panels, 3) ndarray of floats representing
        each Panel's moment (in the first Airplane's geometry axes, relative to the
        first Airplane's CG). The units are in Newton meters.
    :param T_pas_GP1_CgP1_to_W_CgP1: The OperatingPoint's (4,4) passive transformation
        matrix from the first Airplane's geometry axes, relative to its CG, to wind
        axes, relative to the first Airplane's CG.
    :param qInf__E: The OperatingPoint's freestream dynamic pressure. The units are in
        Pascals.
    :param airplanesCg_GP1_CgP1: For each Airplane, a (3,) ndarray of floats
        representing its CG's position (in the first Airplane's geometry axes, relative
        to the first Airplane's CG). The units are in meters.
    :param airplane_reference_dimensions: For each Airplane, its (s_ref, c_ref, b_ref).
        The units are in square meters and meters.
    :return: The StepLoads, with read only arrays.
    """
    num_panels = layout_plan.num_panels
    if stackPanelForces_GP1.shape[0] != num_panels:
        raise ValueError(
            f"The first dimension of stackPanelForces_GP1 must equal "
            f"solver.num_panels ({num_panels}), "
            f"got {stackPanelForces_GP1.shape[0]}."
        )
    if stackPanelMoments_GP1_CgP1.shape[0] != num_panels:
        raise ValueError(
            f"The first dimension of stackPanelMoments_GP1_CgP1 must equal "
            f"solver.num_panels ({num_panels}), "
            f"got {stackPanelMoments_GP1_CgP1.shape[0]}."
        )

    stackPanelForces_W = _transformations.apply_T_to_vectors(
        T_pas_GP1_CgP1_to_W_CgP1, stackPanelForces_GP1, is_position=False
    )
    stackPanelMoments_W_CgP1 = _transformations.apply_T_to_vectors(
        T_pas_GP1_CgP1_to_W_CgP1, stackPanelMoments_GP1_CgP1, is_position=False
    )

    num_airplanes = len(layout_plan.airplane_panel_ranges)
    airplane_arrays = {
        name: np.zeros((num_airplanes, 3), dtype=float)
        for name in (
            "forces_W",
            "forces_G",
            "moments_W_CgP1",
            "moments_G_Cg",
            "moments_W_Cg",
            "forceCoefficients_W",
            "forceCoefficients_G",
            "momentCoefficients_W_CgP1",
            "momentCoefficients_G_Cg",
            "momentCoefficients_W_Cg",
        )
    }
    for airplane_id, ((start, stop), Cg_GP1_CgP1, (s_ref, c_ref, b_ref)) in enumerate(
        zip(
            layout_plan.airplane_panel_ranges,
            airplanesCg_GP1_CgP1,
            airplane_reference_dimensions,
        )
    ):
        theseAirplaneForces_GP1 = stackPanelForces_GP1[start:stop].sum(axis=0)
        theseAirplaneMoments_GP1_CgP1 = stackPanelMoments_GP1_CgP1[start:stop].sum(
            axis=0
        )

        forces_W = _transformations.apply_T_to_vectors(
            T_pas_GP1_CgP1_to_W_CgP1, theseAirplaneForces_GP1, is_position=False
        )
        moments_W_CgP1 = _transformations.apply_T_to_vectors(
            T_pas_GP1_CgP1_to_W_CgP1, theseAirplaneMoments_GP1_CgP1, is_position=False
        )
        # The geometry axes are parallel, so the force is relabeled, and the moment's
        # reference point moves to this Airplane's CG, as in process_solver_loads.
        forces_G = theseAirplaneForces_GP1
        moments_G_Cg = theseAirplaneMoments_GP1_CgP1 - np.cross(
            Cg_GP1_CgP1, theseAirplaneForces_GP1
        )
        moments_W_Cg = _transformations.apply_T_to_vectors(
            T_pas_GP1_CgP1_to_W_CgP1, moments_G_Cg, is_position=False
        )

        # The coefficients, with process_solver_loads' formulas: the force over the
        # dynamic pressure and reference area, and the moment also over the reference
        # span (x and z) or chord (y).
        moment_lengths = np.array([b_ref, c_ref, b_ref], dtype=float)
        airplane_arrays["forces_W"][airplane_id] = forces_W
        airplane_arrays["forces_G"][airplane_id] = forces_G
        airplane_arrays["moments_W_CgP1"][airplane_id] = moments_W_CgP1
        airplane_arrays["moments_G_Cg"][airplane_id] = moments_G_Cg
        airplane_arrays["moments_W_Cg"][airplane_id] = moments_W_Cg
        airplane_arrays["forceCoefficients_W"][airplane_id] = forces_W / qInf__E / s_ref
        airplane_arrays["forceCoefficients_G"][airplane_id] = forces_G / qInf__E / s_ref
        airplane_arrays["momentCoefficients_W_CgP1"][airplane_id] = (
            moments_W_CgP1 / qInf__E / s_ref / moment_lengths
        )
        airplane_arrays["momentCoefficients_G_Cg"][airplane_id] = (
            moments_G_Cg / qInf__E / s_ref / moment_lengths
        )
        airplane_arrays["momentCoefficients_W_Cg"][airplane_id] = (
            moments_W_Cg / qInf__E / s_ref / moment_lengths
        )

    # Each Wing's totals, summed over its Panels in order.
    stackWingForces_GP1 = np.zeros((len(layout_plan.wing_shapes), 3), dtype=float)
    stackWingMoments_GP1_CgP1 = np.zeros((len(layout_plan.wing_shapes), 3), dtype=float)
    for wing_id, ((num_chordwise_panels, num_spanwise_panels), offset) in enumerate(
        zip(layout_plan.wing_shapes, layout_plan.wing_offsets)
    ):
        wing_panels = slice(offset, offset + num_chordwise_panels * num_spanwise_panels)
        stackWingForces_GP1[wing_id] = stackPanelForces_GP1[wing_panels].sum(axis=0)
        stackWingMoments_GP1_CgP1[wing_id] = stackPanelMoments_GP1_CgP1[
            wing_panels
        ].sum(axis=0)

    return StepLoads(
        *(
            _step_parameters.make_read_only(np.array(array, dtype=float))
            for array in (
                stackPanelForces_GP1,
                stackPanelForces_W,
                stackPanelMoments_GP1_CgP1,
                stackPanelMoments_W_CgP1,
                airplane_arrays["forces_W"],
                airplane_arrays["forces_G"],
                airplane_arrays["moments_W_CgP1"],
                airplane_arrays["moments_G_Cg"],
                airplane_arrays["moments_W_Cg"],
                airplane_arrays["forceCoefficients_W"],
                airplane_arrays["forceCoefficients_G"],
                airplane_arrays["momentCoefficients_W_CgP1"],
                airplane_arrays["momentCoefficients_G_Cg"],
                airplane_arrays["momentCoefficients_W_Cg"],
                stackWingForces_GP1,
                stackWingMoments_GP1_CgP1,
            )
        )
    )


def strip_leading_edge_points(
    geometry: _step_geometry.StepGeometry,
    slep_outboard_is_left: np.ndarray,
    slep_point_indices: np.ndarray,
) -> np.ndarray:
    """Returns each Panel's strip leading edge point, as the aeroelastic solver's
    _populate_slep_positions finds them.

    Each Panel's strip leading edge point is the outboard front corner (its front left
    corner on a mirror meshed grid, and its front right corner otherwise) of the first
    chordwise row's Panel its strip starts at.

    :param geometry: The time step's StepGeometry.
    :param slep_outboard_is_left: A (num_panels,) ndarray of bools, whether each Panel's
        outboard front corner is its front left corner.
    :param slep_point_indices: A (num_panels,) ndarray of ints, the global index of the
        Panel whose outboard front corner is each Panel's strip leading edge point.
    :return: A (num_panels, 3) ndarray of floats representing the points (in the first
        Airplane's geometry axes, relative to the first Airplane's CG). The units are in
        meters.
    """
    stackOutboardFrontPoints_GP1_CgP1 = np.where(
        slep_outboard_is_left[:, np.newaxis],
        geometry.stackFlpp_GP1_CgP1,
        geometry.stackFrpp_GP1_CgP1,
    )
    stackSlep_GP1_CgP1: np.ndarray = stackOutboardFrontPoints_GP1_CgP1[
        slep_point_indices
    ]
    return stackSlep_GP1_CgP1


def strip_leading_edge_moments(
    stackPanelForces_GP1: np.ndarray,
    stackPanelMoments_GP1_CgP1: np.ndarray,
    stackSlep_GP1_CgP1: np.ndarray,
) -> np.ndarray:
    """Returns each Panel's moment about its strip leading edge point, as the
    aeroelastic solver's _process_panel_loads_hook finds them: its moment about the
    first Airplane's CG minus the strip leading edge point's position crossed with its
    force.

    :param stackPanelForces_GP1: A (num_panels, 3) ndarray of floats representing each
        Panel's force (in the first Airplane's geometry axes). The units are in Newtons.
    :param stackPanelMoments_GP1_CgP1: A (num_panels, 3) ndarray of floats representing
        each Panel's moment (in the first Airplane's geometry axes, relative to the
        first Airplane's CG). The units are in Newton meters.
    :param stackSlep_GP1_CgP1: A (num_panels, 3) ndarray of floats representing each
        Panel's strip leading edge point (in the first Airplane's geometry axes,
        relative to the first Airplane's CG). The units are in meters.
    :return: A (num_panels, 3) ndarray of floats representing the moments (in the first
        Airplane's geometry axes, relative to each Panel's strip leading edge point).
        The units are in Newton meters.
    """
    moments_GP1_Slep: np.ndarray = (
        stackPanelMoments_GP1_CgP1
        - _functions.numba_1d_explicit_cross(stackSlep_GP1_CgP1, stackPanelForces_GP1)
    )
    return moments_GP1_Slep


def apply_step_loads_to_objects(
    loads: StepLoads,
    panels: np.ndarray,
    airplanes: Sequence[geometry.airplane.Airplane],
) -> None:
    """Writes a StepLoads' values onto the Panels and Airplanes, as process_solver_loads
    writes them.

    This is temporary. Output, plotting, and the coupled problems still read the loads
    from the objects, so the array path writes them there until step 7 of the arrays
    rewrite builds the objects on demand from the stored loads. As in
    process_solver_loads, each Panel gets rows of writeable stacks, and each Airplane
    gets its own writeable arrays.

    :param loads: The StepLoads.
    :param panels: A (num_panels,) ndarray of the Panels, in global Panel order.
    :param airplanes: The Airplanes, in order.
    :return: None
    """
    stackPanelForces_GP1 = loads.stackPanelForces_GP1.copy()
    stackPanelMoments_GP1_CgP1 = loads.stackPanelMoments_GP1_CgP1.copy()
    stackPanelForces_W = loads.stackPanelForces_W.copy()
    stackPanelMoments_W_CgP1 = loads.stackPanelMoments_W_CgP1.copy()
    for panel_num, panel in enumerate(panels):
        panel.forces_GP1 = stackPanelForces_GP1[panel_num]
        panel.moments_GP1_CgP1 = stackPanelMoments_GP1_CgP1[panel_num]
        panel.forces_W = stackPanelForces_W[panel_num]
        panel.moments_W_CgP1 = stackPanelMoments_W_CgP1[panel_num]

    for airplane_id, airplane in enumerate(airplanes):
        airplane.forces_W = loads.stackAirplaneForces_W[airplane_id].copy()
        airplane.moments_W_CgP1 = loads.stackAirplaneMoments_W_CgP1[airplane_id].copy()
        airplane.forces_G = loads.stackAirplaneForces_G[airplane_id].copy()
        airplane.moments_G_Cg = loads.stackAirplaneMoments_G_Cg[airplane_id].copy()
        airplane.moments_W_Cg = loads.stackAirplaneMoments_W_Cg[airplane_id].copy()
        airplane.forceCoefficients_W = loads.stackAirplaneForceCoefficients_W[
            airplane_id
        ].copy()
        airplane.momentCoefficients_W_CgP1 = (
            loads.stackAirplaneMomentCoefficients_W_CgP1[airplane_id].copy()
        )
        airplane.forceCoefficients_G = loads.stackAirplaneForceCoefficients_G[
            airplane_id
        ].copy()
        airplane.momentCoefficients_G_Cg = loads.stackAirplaneMomentCoefficients_G_Cg[
            airplane_id
        ].copy()
        airplane.momentCoefficients_W_Cg = loads.stackAirplaneMomentCoefficients_W_Cg[
            airplane_id
        ].copy()
