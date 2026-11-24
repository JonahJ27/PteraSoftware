"""Contains the functions that mesh Wings as arrays, for the solvers' array path.

Meshing is split in two. build_mesh_plan runs once per run and holds everything that
depends only on a Wing's structure and on values that are fixed for a whole run (its
symmetry type and plane, panel counts, spacings, and each WingCrossSection's chord,
Airfoil, and control surface). mesh_wing_arrays runs at every geometry build: it takes a
Wing's WingStepParameters and returns its Panel corners. See docs/ARRAY_LAYOUTS.md for
the layout conventions.

mesh_wing_arrays computes the same quantities as geometry._meshing.mesh_wing, in the
same order of operations, but for every WingCrossSection and every Panel column at once
instead of one wing section at a time.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from . import _functions, _step_parameters, _transformations, geometry
from .geometry import _symmetry


class MeshPlan(NamedTuple):
    """The values of one Wing's mesh that are fixed for a whole run.

    It exists because much of meshing a Wing (its chordwise and spanwise coordinates,
    its WingCrossSections' camber lines, and its symmetry type) doesn't change during a
    run, so the array path computes it once per run instead of at every time step, as
    mesh_wing does.
    """

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

    # A (K, num_chordwise_panels + 1, 3) ndarray of floats, with one entry per
    # WingCrossSection: the points of its mean camber line at the chordwise coordinates
    # (in its wing cross section axes, relative to its leading point), with its control
    # surface deflected and scaled by its chord. Each WingCrossSection's Airfoil, chord,
    # and control surface are fixed for a whole run, so these are too. Its units are in
    # meters.
    gridMclPoints_Wcs_Lp: np.ndarray

    # For a type 4 Wing with at least one asymmetric control surface with a non zero
    # deflection, the same points for its mirrored half, whose asymmetric control
    # surfaces deflect the other way. Otherwise None, and a type 4 Wing's mirrored half
    # uses gridMclPoints_Wcs_Lp. Its units are in meters.
    gridMirroredMclPoints_Wcs_Lp: np.ndarray | None

    # The Wing's symmetry plane, as a (3,) ndarray of floats for its unit normal (in
    # geometry axes) and one for a point on it (in geometry axes, relative to the CG, in
    # meters), both None for a type 1 Wing.
    symmetryNormal_G: np.ndarray | None
    symmetryPoint_G_Cg: np.ndarray | None


class WingMesh(NamedTuple):
    """One Wing's mesh at one geometry build, from mesh_wing_arrays (or, for a Wing
    already built, wing_mesh_from_wing, which the tests use): its Panel corners, and the
    two values of the Wing that the step geometry needs besides them.

    It exists so the array path can hand a meshed Wing's geometry to build_step_geometry
    without building the Wing or its Panels.
    """

    # The front left, front right, back left, and back right Panel corners. Each is a
    # (num_chordwise_panels, num_spanwise_panels, 3) ndarray of floats, read only, in
    # the Wing's Panel grid order (see docs/ARRAY_LAYOUTS.md), holding the position of
    # each Panel's corner (in geometry axes, relative to the CG). Their units are in
    # meters.
    gridFlpp_G_Cg: np.ndarray
    gridFrpp_G_Cg: np.ndarray
    gridBlpp_G_Cg: np.ndarray
    gridBrpp_G_Cg: np.ndarray

    # The Wing's span, as Wing.span computes it (doubled for a type 4 Wing). Its units
    # are in meters.
    span: float

    # A (3,) ndarray of floats, read only, representing the wing axes' third basis
    # vector (in geometry axes), as Wing.WnZ_G computes it.
    WnZ_G: np.ndarray


def build_mesh_plan(wing: geometry.wing.Wing) -> MeshPlan:
    """Builds a Wing's MeshPlan from its structure and its run fixed values.

    It replaces the parts of mesh_wing (in geometry/_meshing.py) that are fixed for a
    whole run: its chordwise and spanwise coordinates, and each WingCrossSection's
    deflected camber line.

    It exists because the parts of meshing that are fixed for a run need computing only
    once, so the array path computes them here, at the start of the run, and
    mesh_wing_arrays does only each time step's work.

    The chordwise and spanwise coordinates and the camber line points are computed
    exactly as mesh_wing computes them, so the array meshing matches it.

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

    # Each WingCrossSection's camber line points, with its control surface deflected as
    # mesh_wing deflects it. For a type 4 Wing's mirrored half, mesh_wing deflects each
    # asymmetric control surface the other way. A zero deflection leaves the Airfoil
    # unchanged either way, so the mirrored half only needs its own points when some
    # asymmetric control surface has a non zero deflection.
    gridMclPoints_Wcs_Lp = np.stack(
        [
            _mcl_points_Wcs_Lp(
                wing_cross_section,
                wing_cross_section.control_surface_deflection,
                chordwise_coordinates,
            )
            for wing_cross_section in wing_cross_sections
        ]
    )
    gridMirroredMclPoints_Wcs_Lp = None
    if symmetry_type == 4 and any(
        wing_cross_section.control_surface_symmetry_type == "asymmetric"
        and wing_cross_section.control_surface_deflection != 0.0
        for wing_cross_section in wing_cross_sections
    ):
        gridMirroredMclPoints_Wcs_Lp = _step_parameters.make_read_only(
            np.stack(
                [
                    _mcl_points_Wcs_Lp(
                        wing_cross_section,
                        (
                            -wing_cross_section.control_surface_deflection
                            if wing_cross_section.control_surface_symmetry_type
                            == "asymmetric"
                            else wing_cross_section.control_surface_deflection
                        ),
                        chordwise_coordinates,
                    )
                    for wing_cross_section in wing_cross_sections
                ]
            )
        )

    symmetryNormal_G = None
    symmetryPoint_G_Cg = None
    if symmetry_type != 1:
        assert wing.symmetryNormal_G is not None
        assert wing.symmetryPoint_G_Cg is not None
        symmetryNormal_G = _step_parameters.make_read_only(
            np.array(wing.symmetryNormal_G, dtype=float)
        )
        symmetryPoint_G_Cg = _step_parameters.make_read_only(
            np.array(wing.symmetryPoint_G_Cg, dtype=float)
        )

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
        gridMclPoints_Wcs_Lp=_step_parameters.make_read_only(gridMclPoints_Wcs_Lp),
        gridMirroredMclPoints_Wcs_Lp=gridMirroredMclPoints_Wcs_Lp,
        symmetryNormal_G=symmetryNormal_G,
        symmetryPoint_G_Cg=symmetryPoint_G_Cg,
    )


def _mcl_points_Wcs_Lp(
    wing_cross_section: geometry.wing_cross_section.WingCrossSection,
    deflection: float,
    chordwise_coordinates: np.ndarray,
) -> np.ndarray:
    """Returns a WingCrossSection's mean camber line points at the chordwise
    coordinates, as mesh_wing's _get_mcl_points and _get_mcs_points compute them.

    It replaces _get_mcl_points and the camber line scaling at the start of
    _get_mcs_points (in geometry/_meshing.py), with the Airfoil.add_control_surface
    calls mesh_wing makes.

    It exists so the MeshPlan can hold each WingCrossSection's camber line points, found
    once per run, instead of finding them from the Airfoil at every time step.

    :param wing_cross_section: The WingCrossSection.
    :param deflection: The control surface deflection to use, in degrees (the
        WingCrossSection's own, or its negative for a type 4 Wing's mirrored half).
    :param chordwise_coordinates: A (M,) ndarray of floats for the normalized chordwise
        coordinates at which to sample the mean camber line. The values are unitless.
    :return: A (M,3) ndarray of floats for the points (in the WingCrossSection's axes,
        relative to its leading point). The units are in meters.
    """
    airfoil = wing_cross_section.airfoil.add_control_surface(
        deflection=deflection,
        hinge_point=wing_cross_section.control_surface_hinge_point,
    )
    mcl_points_A_Lp = airfoil.get_resampled_mcl(chordwise_coordinates)

    mcl_pointsX_Wcs_Lp = wing_cross_section.chord * np.expand_dims(
        mcl_points_A_Lp[:, 0], 1
    )
    mcl_pointsZ_Wcs_Lp = wing_cross_section.chord * np.expand_dims(
        mcl_points_A_Lp[:, 1], 1
    )
    return np.hstack(
        [mcl_pointsX_Wcs_Lp, np.zeros_like(mcl_pointsX_Wcs_Lp), mcl_pointsZ_Wcs_Lp]
    )


def mesh_wing_arrays(
    plan: MeshPlan, parameters: _step_parameters.WingStepParameters
) -> WingMesh:
    """Meshes one Wing from its values at one geometry build, without building any Wing,
    WingCrossSection, Airfoil, or Panel.

    It replaces mesh_wing (in geometry/_meshing.py).

    It exists because building and meshing a Wing on the object path creates a
    WingCrossSection, Airfoil, and Panel object for every part of it, at every time
    step. This computes the same Panel corners with array operations from the Wing's
    step parameters and its MeshPlan.

    The Panel corners match those mesh_wing gives the Wing built from the same values,
    in the same Panel grid order.

    It first validates the values as the object path's constructors would (see
    _step_parameters.validate_wing_step_parameters), raising the same error for invalid
    values, then checks that the Wing's symmetry type is still its MeshPlan's.

    :param plan: The Wing's MeshPlan.
    :param parameters: The Wing's WingStepParameters at this geometry build.
    :return: The Wing's Panel corners.
    """
    _step_parameters.validate_wing_step_parameters(parameters)
    _check_symmetry_type(plan, parameters)

    symmetry_type = plan.symmetry_type

    # Each WingCrossSection's transformation from its axes, relative to its leading
    # point, to wing axes, relative to the leading edge root point, as
    # Wing.children_T_pas_Wcs_Lp_to_Wn_Ler.
    stackT_pas_Wcs_Lp_to_Wn_Ler = _stack_T_pas_Wcs_Lp_to_Wn_Ler(
        parameters.stackLp_Wcsp_Lpp, parameters.stackAngles_Wcsp_to_Wcs_ixyz
    )

    # The Wing's transformations between geometry axes, relative to the CG, and wing
    # axes, relative to the leading edge root point.
    T_pas_G_Cg_to_Wn_Ler = _T_pas_G_Cg_to_Wn_Ler(plan, parameters)
    T_pas_Wn_Ler_to_G_Cg = _transformations.invert_T_pas(T_pas_G_Cg_to_Wn_Ler)

    # The Wing's span and wing axes' third basis vector (in geometry axes), as Wing.span
    # and Wing.WnZ_G compute them.
    span = _span(plan, parameters, stackT_pas_Wcs_Lp_to_Wn_Ler)
    WnZ_G = _step_parameters.make_read_only(
        _transformations.apply_T_to_vectors(
            T_pas_Wn_Ler_to_G_Cg, np.array([0.0, 0.0, 1.0]), is_position=False
        )
    )

    # The main half's Panel column edges (in wing axes, relative to the leading edge
    # root point).
    gridInnerPoints_Wn_Ler, gridOuterPoints_Wn_Ler = _column_edge_points_Wn_Ler(
        plan, plan.gridMclPoints_Wcs_Lp, stackT_pas_Wcs_Lp_to_Wn_Ler
    )

    # The main half's column edges (in geometry axes, relative to the CG). Every corner
    # is one of these points, so they are transformed once rather than once per corner.
    gridInnerPoints_G_Cg = _transformations.apply_T_to_vectors(
        T_pas_Wn_Ler_to_G_Cg, gridInnerPoints_Wn_Ler, is_position=True
    )
    gridOuterPoints_G_Cg = _transformations.apply_T_to_vectors(
        T_pas_Wn_Ler_to_G_Cg, gridOuterPoints_Wn_Ler, is_position=True
    )

    # Each Panel's front corners are on its column's chordwise edges' first M points and
    # its back corners on their last M points.
    gridFipp_G_Cg = gridInnerPoints_G_Cg[:-1]
    gridFopp_G_Cg = gridOuterPoints_G_Cg[:-1]
    gridBipp_G_Cg = gridInnerPoints_G_Cg[1:]
    gridBopp_G_Cg = gridOuterPoints_G_Cg[1:]

    if symmetry_type == 1:
        return _wing_mesh(
            span, WnZ_G, gridFipp_G_Cg, gridFopp_G_Cg, gridBipp_G_Cg, gridBopp_G_Cg
        )

    if symmetry_type in (2, 3):
        # mesh_wing puts each mirror only wing section's Panels, flipped, to the left of
        # those before it, so the columns end up in reverse order. Each Panel's outer
        # corners are on its left.
        return _wing_mesh(
            span,
            WnZ_G,
            gridFopp_G_Cg[:, ::-1],
            gridFipp_G_Cg[:, ::-1],
            gridBopp_G_Cg[:, ::-1],
            gridBipp_G_Cg[:, ::-1],
        )

    # A type 4 Wing's mirrored half. Its column edges are the main half's, unless an
    # asymmetric control surface deflects the other way on this half.
    if plan.gridMirroredMclPoints_Wcs_Lp is None:
        gridMirroredInnerPoints_Wn_Ler = gridInnerPoints_Wn_Ler
        gridMirroredOuterPoints_Wn_Ler = gridOuterPoints_Wn_Ler
    else:
        gridMirroredInnerPoints_Wn_Ler, gridMirroredOuterPoints_Wn_Ler = (
            _column_edge_points_Wn_Ler(
                plan, plan.gridMirroredMclPoints_Wcs_Lp, stackT_pas_Wcs_Lp_to_Wn_Ler
            )
        )

    # As in mesh_wing, the mirrored half is reflected across the symmetry plane
    # (expressed in wing axes, relative to the leading edge root point) with an active
    # transformation, then expressed in geometry axes, relative to the CG.
    assert plan.symmetryNormal_G is not None
    assert plan.symmetryPoint_G_Cg is not None
    symmetryNormal_Wn = _transformations.apply_T_to_vectors(
        T_pas_G_Cg_to_Wn_Ler, plan.symmetryNormal_G, is_position=False
    )
    symmetryPoint_Wn_Ler = _transformations.apply_T_to_vectors(
        T_pas_G_Cg_to_Wn_Ler, plan.symmetryPoint_G_Cg, is_position=True
    )
    reflect_T_act = _transformations.generate_reflect_T(
        plane_point_A_a=symmetryPoint_Wn_Ler,
        plane_normal_A=symmetryNormal_Wn,
        passive=False,
    )
    gridMirroredInnerPoints_G_Cg = _transformations.apply_T_to_vectors(
        T_pas_Wn_Ler_to_G_Cg,
        _transformations.apply_T_to_vectors(
            reflect_T_act, gridMirroredInnerPoints_Wn_Ler, is_position=True
        ),
        is_position=True,
    )
    gridMirroredOuterPoints_G_Cg = _transformations.apply_T_to_vectors(
        T_pas_Wn_Ler_to_G_Cg,
        _transformations.apply_T_to_vectors(
            reflect_T_act, gridMirroredOuterPoints_Wn_Ler, is_position=True
        ),
        is_position=True,
    )

    # mesh_wing puts each mirrored wing section's Panels, flipped, to the left of
    # everything before it, so the mirrored columns end up in reverse order to the left
    # of the main half. Each mirrored Panel's outer corners are on its left.
    gridMirroredInnerPoints_G_Cg = gridMirroredInnerPoints_G_Cg[:, ::-1]
    gridMirroredOuterPoints_G_Cg = gridMirroredOuterPoints_G_Cg[:, ::-1]
    return _wing_mesh(
        span,
        WnZ_G,
        np.concatenate([gridMirroredOuterPoints_G_Cg[:-1], gridFipp_G_Cg], axis=1),
        np.concatenate([gridMirroredInnerPoints_G_Cg[:-1], gridFopp_G_Cg], axis=1),
        np.concatenate([gridMirroredOuterPoints_G_Cg[1:], gridBipp_G_Cg], axis=1),
        np.concatenate([gridMirroredInnerPoints_G_Cg[1:], gridBopp_G_Cg], axis=1),
    )


def _check_symmetry_type(
    plan: MeshPlan, parameters: _step_parameters.WingStepParameters
) -> None:
    """Raises a ValueError if a Wing's values at one geometry build give it a different
    symmetry type than its MeshPlan's.

    The object path has no such check (its Airplane would process the Wing as its new
    type, changing its Panel grid), but the array path's plans assume the symmetry type
    is fixed for the run, so a change raises an error like the movements' symmetry type
    check. It uses _symmetry.wing_symmetry_type, the decision the Airplane makes.

    :param plan: The Wing's MeshPlan.
    :param parameters: The Wing's WingStepParameters at this geometry build.
    :return: None
    """
    # The symmetry type only depends on the symmetry plane, which a type 1 Wing doesn't
    # have.
    if plan.symmetry_type != 1:
        symmetry_type = _symmetry.wing_symmetry_type(
            symmetric=plan.symmetry_type == 4,
            mirror_only=plan.symmetry_type in (2, 3),
            symmetryNormal_G=plan.symmetryNormal_G,
            symmetryPoint_G_Cg=plan.symmetryPoint_G_Cg,
            angles_Gs_to_Wn_ixyz=parameters.angles_Gs_to_Wn_ixyz,
            Ler_Gs_Cgs=parameters.Ler_Gs_Cgs,
        )
        if symmetry_type != plan.symmetry_type:
            raise ValueError(
                f"A Wing changed from type {plan.symmetry_type} symmetry to type "
                f"{symmetry_type} symmetry. Wings cannot undergo motion that changes "
                f"their symmetry type. This happens when a symmetric Wing moves such "
                f"that its symmetry plane is no longer coincident with the wing axes' "
                f"xz plane or vice versa."
            )


def _T_pas_G_Cg_to_Wn_Ler(
    plan: MeshPlan, parameters: _step_parameters.WingStepParameters
) -> np.ndarray:
    """Returns a Wing's passive transformation matrix from geometry axes, relative to
    the CG, to wing axes, relative to the leading edge root point.

    It replaces Wing.T_pas_G_Cg_to_Wn_Ler.

    It exists because Wing.T_pas_G_Cg_to_Wn_Ler needs a built Wing. This builds the same
    matrix from the Wing's step parameters.

    It computes what Wing.T_pas_G_Cg_to_Wn_Ler computes, with the same scalar functions
    in the same order, from the MeshPlan's symmetry plane and the WingStepParameters'
    position and angles.

    :param plan: The Wing's MeshPlan.
    :param parameters: The Wing's WingStepParameters at this geometry build.
    :return: A (4,4) ndarray of floats representing the transformation matrix.
    """
    # Only mirror only Wings (types 2 and 3) are reflected, not type 4 Wings.
    if plan.symmetry_type in (2, 3):
        assert plan.symmetryPoint_G_Cg is not None
        assert plan.symmetryNormal_G is not None
        T_reflect_pas_G_Cg_to_Gs_Cgs = _transformations.generate_reflect_T(
            plane_point_A_a=plan.symmetryPoint_G_Cg,
            plane_normal_A=plan.symmetryNormal_G,
            passive=True,
        )
    else:
        T_reflect_pas_G_Cg_to_Gs_Cgs = np.eye(4, dtype=float)
    return _transformations.compose_T_pas(
        T_reflect_pas_G_Cg_to_Gs_Cgs,
        _transformations.generate_trans_T(parameters.Ler_Gs_Cgs, passive=True),
        _transformations.generate_rot_T(
            parameters.angles_Gs_to_Wn_ixyz, passive=True, intrinsic=True, order="xyz"
        ),
    )


def _stack_T_pas_Wcs_Lp_to_Wn_Ler(
    stackLp_Wcsp_Lpp: np.ndarray, stackAngles_Wcsp_to_Wcs_ixyz: np.ndarray
) -> np.ndarray:
    """Returns each WingCrossSection's passive transformation matrix from its axes,
    relative to its leading point, to wing axes, relative to the leading edge root
    point.

    It replaces WingCrossSection.T_pas_Wcsp_Lpp_to_Wcs_Lp,
    Wing.children_T_pas_Wn_Ler_to_Wcs_Lp, and Wing.children_T_pas_Wcs_Lp_to_Wn_Ler.

    It exists because the Wing's properties build these matrices one WingCrossSection at
    a time from built objects. This builds all of them at once from the step parameters.

    It computes what WingCrossSection.T_pas_Wcsp_Lpp_to_Wcs_Lp and
    Wing.children_T_pas_Wcs_Lp_to_Wn_Ler compute, in the same order of operations, with
    the per WingCrossSection matrices built for every WingCrossSection at once.

    :param stackLp_Wcsp_Lpp: A (K,3) ndarray of floats, each row being a
        WingCrossSection's Lp_Wcsp_Lpp. The units are in meters.
    :param stackAngles_Wcsp_to_Wcs_ixyz: A (K,3) ndarray of floats, each row being a
        WingCrossSection's angles_Wcsp_to_Wcs_ixyz. The units are in degrees.
    :return: A (K,4,4) ndarray of floats, the transformation matrices in order.
    """
    num_wing_cross_sections = stackLp_Wcsp_Lpp.shape[0]

    # Each WingCrossSection's passive translation matrix, as generate_trans_T(...,
    # passive=True) builds it.
    stackT_trans_pas_Wcsp_Lpp_to_Wcsp_Lp = np.tile(
        np.eye(4, dtype=float), (num_wing_cross_sections, 1, 1)
    )
    stackT_trans_pas_Wcsp_Lpp_to_Wcsp_Lp[:, :3, 3] = -stackLp_Wcsp_Lpp

    # Each WingCrossSection's passive rotation matrix, as generate_rot_T(...,
    # passive=True, intrinsic=True, order="xyz") builds it.
    stackT_rot_pas_Wcsp_to_Wcs = np.tile(
        np.eye(4, dtype=float), (num_wing_cross_sections, 1, 1)
    )
    stackT_rot_pas_Wcsp_to_Wcs[:, :3, :3] = np.swapaxes(
        _stack_rot_R_act_ixyz(stackAngles_Wcsp_to_Wcs_ixyz), 1, 2
    )

    # Each WingCrossSection's transformation from its parent's axes, relative to its
    # parent's leading point, as compose_T_pas composes it (starting from the identity
    # matrix).
    stackT_pas_Wcsp_Lpp_to_Wcs_Lp = np.matmul(
        stackT_rot_pas_Wcsp_to_Wcs,
        np.matmul(stackT_trans_pas_Wcsp_Lpp_to_Wcsp_Lp, np.eye(4, dtype=float)),
    )

    # Chain them from the root, then invert each chain, as
    # Wing.children_T_pas_Wn_Ler_to_Wcs_Lp and children_T_pas_Wcs_Lp_to_Wn_Ler do.
    # Composing the first i + 1 matrices from the identity matrix multiplies each onto
    # the chain before it, so building each chain from the previous one gives the same
    # products.
    stackT_pas_Wcs_Lp_to_Wn_Ler = np.empty((num_wing_cross_sections, 4, 4), dtype=float)
    T_pas_Wn_Ler_to_Wcs_Lp = np.eye(4, dtype=float)
    for index in range(num_wing_cross_sections):
        T_pas_Wn_Ler_to_Wcs_Lp = (
            stackT_pas_Wcsp_Lpp_to_Wcs_Lp[index] @ T_pas_Wn_Ler_to_Wcs_Lp
        )
        stackT_pas_Wcs_Lp_to_Wn_Ler[index] = _transformations.invert_T_pas(
            T_pas_Wn_Ler_to_Wcs_Lp
        )
    return stackT_pas_Wcs_Lp_to_Wn_Ler


def _stack_rot_R_act_ixyz(stackAngles_ixyz: np.ndarray) -> np.ndarray:
    """Returns the active intrinsic xyz rotation matrices for a stack of angle vectors,
    as generate_rot_T builds them before it transposes them for a passive matrix.

    It replaces the rotation matrix part of _transformations.generate_rot_T.

    It exists because generate_rot_T builds one matrix per call. This builds a whole
    stack at once for _stack_T_pas_Wcs_Lp_to_Wn_Ler, with the same operations, so the
    results match.

    :param stackAngles_ixyz: A (K,3) ndarray of floats, each row being an angle vector
        using an intrinsic xy'z" sequence. The units are in degrees.
    :return: A (K,3,3) ndarray of floats, the rotation matrices in order.
    """
    stackAnglesRad_ixyz = np.deg2rad(stackAngles_ixyz)
    cosines = np.cos(stackAnglesRad_ixyz)
    sines = np.sin(stackAnglesRad_ixyz)
    num_angles = stackAngles_ixyz.shape[0]

    stack_x_R_act = np.tile(np.eye(3, dtype=float), (num_angles, 1, 1))
    stack_x_R_act[:, 1, 1] = cosines[:, 0]
    stack_x_R_act[:, 1, 2] = -sines[:, 0]
    stack_x_R_act[:, 2, 1] = sines[:, 0]
    stack_x_R_act[:, 2, 2] = cosines[:, 0]

    stack_y_R_act = np.tile(np.eye(3, dtype=float), (num_angles, 1, 1))
    stack_y_R_act[:, 0, 0] = cosines[:, 1]
    stack_y_R_act[:, 0, 2] = sines[:, 1]
    stack_y_R_act[:, 2, 0] = -sines[:, 1]
    stack_y_R_act[:, 2, 2] = cosines[:, 1]

    stack_z_R_act = np.tile(np.eye(3, dtype=float), (num_angles, 1, 1))
    stack_z_R_act[:, 0, 0] = cosines[:, 2]
    stack_z_R_act[:, 0, 1] = -sines[:, 2]
    stack_z_R_act[:, 1, 0] = sines[:, 2]
    stack_z_R_act[:, 1, 1] = cosines[:, 2]

    # An intrinsic xyz sequence multiplies the z, then the y, then the x matrix on the
    # left of the identity matrix, as generate_rot_T does.
    stack_R_act: np.ndarray = np.matmul(stack_z_R_act, np.eye(3, dtype=float))
    stack_R_act = np.matmul(stack_y_R_act, stack_R_act)
    stack_R_act = np.matmul(stack_x_R_act, stack_R_act)
    return stack_R_act


def _column_edge_points_Wn_Ler(
    plan: MeshPlan,
    gridMclPoints_Wcs_Lp: np.ndarray,
    stackT_pas_Wcs_Lp_to_Wn_Ler: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the points on the inner and outer edges of each of the main half's Panel
    columns (in wing axes, relative to the leading edge root point).

    It replaces the transformation and interpolation in _get_mcs_points (in
    geometry/_meshing.py), including _functions.interp_between_points.

    It exists because _get_mcs_points transforms and interpolates the points one pair of
    WingCrossSections at a time. This does every Panel column at once.

    :param plan: The Wing's MeshPlan.
    :param gridMclPoints_Wcs_Lp: A (K,M,3) ndarray of floats, each WingCrossSection's
        mean camber line points (in its axes, relative to its leading point), where M is
        the number of chordwise points. The units are in meters.
    :param stackT_pas_Wcs_Lp_to_Wn_Ler: A (K,4,4) ndarray of floats, each
        WingCrossSection's transformation matrix to wing axes, relative to the leading
        edge root point.
    :return: Two (M,N,3) ndarrays of floats, where N is the number of the main half's
        Panel columns, for the points on the columns' inner and outer edges, in the
        order mesh_wing meshes the columns (root to tip). The units are in meters.
    """
    # Each WingCrossSection's camber line points in wing axes, as apply_T_to_vectors
    # computes them, for every WingCrossSection at once.
    num_wing_cross_sections, num_chordwise_points = gridMclPoints_Wcs_Lp.shape[:2]
    gridMclPointsHomog_Wcs_Lp = np.ones(
        (num_wing_cross_sections, num_chordwise_points, 4), dtype=float
    )
    gridMclPointsHomog_Wcs_Lp[..., :3] = gridMclPoints_Wcs_Lp
    gridMclPoints_Wn_Ler = np.einsum(
        "kij,kpj->kpi", stackT_pas_Wcs_Lp_to_Wn_Ler, gridMclPointsHomog_Wcs_Lp
    )[..., :3]

    # Each column interpolates between its wing section's inner and outer
    # WingCrossSections at its inner and outer spanwise coordinates, as
    # interp_between_points does (the start point plus the coordinate times the vector
    # from the start to the end point). The columns come first here, so move the
    # chordwise index first.
    gridStartPoints_Wn_Ler = gridMclPoints_Wn_Ler[plan.column_wing_section]
    gridVectors_Wn = (
        gridMclPoints_Wn_Ler[plan.column_wing_section + 1] - gridStartPoints_Wn_Ler
    )
    gridInnerPoints_Wn_Ler = (
        gridStartPoints_Wn_Ler
        + plan.column_inner_fraction[:, None, None] * gridVectors_Wn
    )
    gridOuterPoints_Wn_Ler = (
        gridStartPoints_Wn_Ler
        + plan.column_outer_fraction[:, None, None] * gridVectors_Wn
    )
    return (
        np.swapaxes(gridInnerPoints_Wn_Ler, 0, 1),
        np.swapaxes(gridOuterPoints_Wn_Ler, 0, 1),
    )


def _span(
    plan: MeshPlan,
    parameters: _step_parameters.WingStepParameters,
    stackT_pas_Wcs_Lp_to_Wn_Ler: np.ndarray,
) -> float:
    """Returns a Wing's span, as Wing.span computes it: the length of its tip
    WingCrossSection's leading point's projection on the wing axes' y axis (in wing
    axes, relative to the leading edge root point), doubled for a type 4 Wing.

    It replaces Wing.span.

    It exists because Wing.span needs a built Wing, and the array path needs the span
    without one.

    :param plan: The Wing's MeshPlan.
    :param parameters: The Wing's WingStepParameters at this geometry build.
    :param stackT_pas_Wcs_Lp_to_Wn_Ler: A (K,4,4) ndarray of floats, each
        WingCrossSection's transformation matrix to wing axes, relative to the leading
        edge root point.
    :return: The span. Its units are in meters.
    """
    # The tip's leading point is given in its parent WingCrossSection's axes, so it uses
    # the parent's transformation.
    tipLp_Wn_Ler = _transformations.apply_T_to_vectors(
        stackT_pas_Wcs_Lp_to_Wn_Ler[-2],
        parameters.stackLp_Wcsp_Lpp[-1],
        is_position=True,
    )
    projectedTipLp_Wn_Ler = np.dot(tipLp_Wn_Ler, np.array([0.0, 1.0, 0.0])) * np.array(
        [0.0, 1.0, 0.0]
    )
    span = float(np.linalg.norm(projectedTipLp_Wn_Ler))
    if plan.symmetry_type == 4:
        span *= 2
    return span


def _wing_mesh(
    span: float,
    WnZ_G: np.ndarray,
    gridFlpp_G_Cg: np.ndarray,
    gridFrpp_G_Cg: np.ndarray,
    gridBlpp_G_Cg: np.ndarray,
    gridBrpp_G_Cg: np.ndarray,
) -> WingMesh:
    """Returns a WingMesh holding contiguous, read only copies of four corner grids.

    It exists so every WingMesh holds its own contiguous, read only arrays, which keeps
    the array operations that read them fast and stops one part of the run from changing
    a mesh another part reads.

    :param span: The Wing's span. Its units are in meters.
    :param WnZ_G: A read only (3,) ndarray of floats representing the wing axes' third
        basis vector (in geometry axes).
    :param gridFlpp_G_Cg: A (M,N,3) ndarray of floats for the front left corners (in
        geometry axes, relative to the CG). The units are in meters.
    :param gridFrpp_G_Cg: The same for the front right corners.
    :param gridBlpp_G_Cg: The same for the back left corners.
    :param gridBrpp_G_Cg: The same for the back right corners.
    :return: The WingMesh.
    """

    def read_only(grid: np.ndarray) -> np.ndarray:
        return _step_parameters.make_read_only(np.ascontiguousarray(grid, dtype=float))

    return WingMesh(
        gridFlpp_G_Cg=read_only(gridFlpp_G_Cg),
        gridFrpp_G_Cg=read_only(gridFrpp_G_Cg),
        gridBlpp_G_Cg=read_only(gridBlpp_G_Cg),
        gridBrpp_G_Cg=read_only(gridBrpp_G_Cg),
        span=span,
        WnZ_G=WnZ_G,
    )


def wing_mesh_from_wing(wing: geometry.wing.Wing) -> WingMesh:
    """Returns the WingMesh of a Wing that is already built and meshed, read from its
    Panels and properties.

    It exists so the tests can give the array functions exactly the geometry of Wings
    the object path built, and compare the two paths on the same inputs. No solver calls
    it since the coupled solvers started meshing their recorded step parameters (step 6
    of the arrays rewrite).

    :param wing: The meshed Wing.
    :return: Its WingMesh, with the same values as the Wing's Panels' corners and its
        span and WnZ_G properties.
    """
    panels = wing.panels
    assert panels is not None
    span = wing.span
    assert span is not None
    WnZ_G = wing.WnZ_G
    assert WnZ_G is not None
    return _wing_mesh(
        span,
        WnZ_G,
        *(
            np.array(
                [[getattr(panel, name) for panel in row] for row in panels],
                dtype=float,
            )
            for name in ("Flpp_G_Cg", "Frpp_G_Cg", "Blpp_G_Cg", "Brpp_G_Cg")
        ),
    )
