"""Contains the function that determines a Wing's symmetry type."""

from __future__ import annotations

import numpy as np

from .. import _transformations


def wing_symmetry_type(
    symmetric: bool,
    mirror_only: bool,
    symmetryNormal_G: np.ndarray | None,
    symmetryPoint_G_Cg: np.ndarray | None,
    angles_Gs_to_Wn_ixyz: np.ndarray,
    Ler_Gs_Cgs: np.ndarray,
) -> int:
    """Determines a Wing's symmetry type from its symmetry settings, angles, and leading
    edge root point.

    It is the decision Airplane.process_wing_symmetry makes for each Wing, separated so
    the movement classes can check a Wing's symmetry type at every time step from its
    parameters, without building the Wing. See the Wing class documentation for the
    interpretation of the symmetry types.

    :param symmetric: Whether the Wing is symmetric.
    :param mirror_only: Whether the Wing is mirror only.
    :param symmetryNormal_G: A (3,) ndarray of floats representing the symmetry plane's
        unit normal (in geometry axes), or None if the Wing has no symmetry plane.
    :param symmetryPoint_G_Cg: A (3,) ndarray of floats representing a point on the
        symmetry plane (in geometry axes, relative to the CG), or None if the Wing has
        no symmetry plane. Its units are in meters.
    :param angles_Gs_to_Wn_ixyz: A (3,) ndarray of floats representing the angles from
        the geometry axes after accounting for symmetry to the wing axes, using an
        intrinsic xy'z" sequence. Its units are in degrees.
    :param Ler_Gs_Cgs: A (3,) ndarray of floats representing the position of the Wing's
        leading edge root point (in geometry axes after accounting for symmetry,
        relative to the CG after accounting for symmetry). Its units are in meters.
    :return: The symmetry type, an int from 1 to 5.
    """
    # Determine if the symmetry plane is coincident with the wing axes' xz plane. If
    # symmetryNormal_G or symmetryPoint_G_Cg is None, then there is no symmetry and the
    # symmetry plane doesn't exist. Otherwise, the symmetry plane is coincident to the
    # wing axes' xz plane if Ler_Gs_Cgs lies on the symmetry plane, and if
    # symmetryNormal_G is parallel with WnY_G. We don't need to check types, values, or
    # normalize because this is done in Wing's init method.
    coincident_symmetry_plane = True
    if symmetryPoint_G_Cg is None or symmetryNormal_G is None:
        coincident_symmetry_plane = False
    else:
        # If the symmetry plane exists, we first need to check if its normal vector is
        # parallel with the wing axes' y axis vector.

        # Actively transform geometry axes' second basis vector (in geometry axes) to
        # this Wing's axes' second basis vector (in geometry axes). We can skip the
        # translation step (step 2) as we are only transforming a direction vector, not
        # a position vector.
        GY_G = np.array([0.0, 1.0, 0.0], dtype=float)
        GsY_G = _transformations.apply_T_to_vectors(
            _transformations.generate_reflect_T(
                plane_point_A_a=symmetryPoint_G_Cg,
                plane_normal_A=symmetryNormal_G,
                passive=False,
            ),
            GY_G,
            is_position=False,
        )
        WnY_G = _transformations.apply_T_to_vectors(
            _transformations.generate_rot_T(
                angles_Gs_to_Wn_ixyz,
                passive=False,
                intrinsic=True,
                order="xyz",
            ),
            GsY_G,
            is_position=False,
        )

        # If symmetryNormal_G is parallel with WnY_G, their cross product will be the
        # zero vector.
        is_parallel = np.allclose(
            np.cross(symmetryNormal_G, WnY_G),
            np.array([0.0, 0.0, 0.0], dtype=float),
        )

        if not is_parallel:
            coincident_symmetry_plane = False
        else:
            # If the symmetry plane's normal vector and the wing axes' y axis vector are
            # parallel, then the last check for a coincident symmetry plane is to check
            # if the Ler is on the symmetry plane.

            # To do this, we first find the symmetry plane's normal vector (in geometry
            # axes after accounting for symmetry) and the symmetry plane's point (in
            # geometry axes after accounting for symmetry, relative to the CG after
            # accounting for symmetry). As the symmetry plane is defined using these
            # quantities, they don't change after reflection.
            symmetryPoint_Gs_Cgs = symmetryPoint_G_Cg
            symmetryNormal_Gs_Cgs = symmetryNormal_G

            # The leading edge root point is on the symmetry plane if the distance
            # between it and the symmetry plane is zero.
            Ler_on_plane = np.allclose(
                np.dot(symmetryNormal_Gs_Cgs, (Ler_Gs_Cgs - symmetryPoint_Gs_Cgs)),
                0.0,
            )

            if not Ler_on_plane:
                coincident_symmetry_plane = False

    # See the Wing class docstring for the interpretation of the different symmetry
    # types.
    if not symmetric:
        if not mirror_only:
            # Type 1 Symmetry:
            # symmetric = False, mirror_only = False
            return 1
        if coincident_symmetry_plane:
            # Type 2 Symmetry:
            # symmetric = False, mirror_only = True, coincident_symmetry_plane = True
            return 2
        # Type 3 Symmetry:
        # symmetric = False, mirror_only = True, coincident_symmetry_plane = False
        return 3
    if coincident_symmetry_plane:
        # Type 4 Symmetry:
        # symmetric = True, coincident_symmetry_plane = True
        return 4
    # Type 5 Symmetry:
    # symmetric = True, coincident_symmetry_plane = False
    return 5
