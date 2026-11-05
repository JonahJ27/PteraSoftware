"""Contains the crank linkage kinematics that FlapTransmission.crank_linkage uses.

ENTIRE SCRIPT VALIDATED CORRECT

The linkage connects a gearbox crank to a wing crank through a rigid link. Its geometry
is described in the plane of the flap (the body y and z axes), and every derivative here
is analytical.
"""

from __future__ import annotations

import numpy as np

from . import _parameter_validation

# These are the ways a wing trajectory can be converted into a motor trajectory.
# "analytical" applies the chain rule with the inverse kinematics' analytical Jacobian
# and Hessian, like the blade element theory code's convertWingTrajToMotorAnalytical.
# "numerical" differentiates the motor position in time with finite differences, like
# its convertWingTrajToMotorNumerical.
DERIVATIVE_METHODS = ("analytical", "numerical")

# This is the number of crank angles at which the linkage is checked for assembling at
# construction.
_NUM_ASSEMBLY_CHECK_ANGLES = 3600

# This is the fraction of a trajectory's largest wing velocity below which a sample is
# taken to sit exactly on a turnaround, where the motor's velocity and acceleration are
# indeterminate and are filled from the neighboring samples. prevents trajectory spikes
# at singulairty
_INDETERMINATE_RATE_FRACTION = 1e-9

# This is the number of samples on each side of a turnaround whose motor velocity and
# acceleration are also filled from the neighboring samples. Next to a turnaround, the
# motor acceleration is a near cancellation of two large terms, and with a 1e-4 s
# reference step at 2 Hz, the error is about 22000 rad/s^2 one sample away, 2800 two
# samples away, and still about 100 seven samples away. Ten samples leaves a margin.
_TURNAROUND_FILL_HALF_WIDTH = 10


def _law_of_cosines_terms(
    fixed_length: float, opposite_length: float, squared_distance: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Finds the angle of a triangle at the end of a fixed side, and the terms that its
    derivatives are built from.

    VALIDATED CORRECT

    The triangle has a side of known length fixed_length, a side whose squared length is
    squared_distance, and a third side of known length opposite_length that faces the
    angle. With D the squared distance, the angle is arccos((L^2 + D - Lo^2) / (2 * L *
    sqrt(D))). Its derivative with respect to any variable x is -h(D) * D'(x) / 2, where
    h(D) = (D - L^2 + Lo^2) / (D * sqrt(delta)) and delta = 4 * L^2 * D - (L^2 + D -
    Lo^2)^2, and its second derivative follows from h and its derivative with respect to
    D.

    :param fixed_length: The length of the side that is fixed (m).
    :param opposite_length: The length of the side opposite the angle (m).
    :param squared_distance: A ndarray or float with the squared length of the third
        side (m^2).
    :return: A tuple of the angle (rad), h, and the derivative of h with respect to the
        squared distance. Each has the shape of squared_distance.
    """
    # Why this function returns h and dh/dD rather than only the angle: the blade
    # element theory code gets the crank Jacobian and Hessian from MATLAB's symbolic
    # diff of calcCrankshaftFK, and there's no symbolic engine here. The crank angle
    # moves D, but L and Lo stay fixed, so the triangle's only moving part is D. The
    # chain rule therefore splits every derivative of the arccos into a factor that
    # depends only on the triangle (h and dh/dD) and a factor that depends only on how D
    # moves (D' and D''). Computing the triangle factor once here lets one helper serve
    # both triangles in the forward kinematics and both in the inverse kinematics. Each
    # caller supplies only its own D' and D'', instead of differentiating the arccos by
    # hand four times. The result matches the blade element theory code's symbolic
    # Jacobian and Hessian to about 1e-16. It also exposes the singularity directly: h
    # becomes infinite exactly where the linkage locks up.
    #
    # The triangle's sides are the fixed side (length L), the side to the moving point
    # (length sqrt(D)), and the opposite side (length Lo). The angle sits between the
    # first two, and the law of cosines gives Lo^2 = L^2 + D - 2 * L * sqrt(D) *
    # cos(angle). Working with D instead of sqrt(D) keeps every derivative below free of
    # square roots of the moving quantity, because D is a simple trigonometric function
    # of the crank or wing angle.
    D = squared_distance
    L2 = fixed_length**2
    Lo2 = opposite_length**2

    # Solve the law of cosines for the angle. The argument of arccos is u(D), which is
    # numerator / (2 * L * sqrt(D)) with numerator = L^2 + D - Lo^2.
    numerator = L2 + D - Lo2
    angle = np.arccos(numerator / (2.0 * fixed_length * np.sqrt(D)))

    # By the chain rule, d(angle)/dD = -1 / sqrt(1 - u^2) * du/dD. Expanding both
    # factors gives:
    #
    # du/dD = (D - L^2 + Lo^2) / (4 * L * D^1.5)
    #
    # sqrt(1 - u^2) = sqrt(delta) / (2 * L * sqrt(D))
    #
    # where delta = 4 * L^2 * D - numerator^2. Their ratio simplifies to d(angle)/dD =
    # -h / 2, with h = N / (D * sqrt(delta)) and N = D - L^2 + Lo^2. So h is the angle's
    # sensitivity to D, scaled by -2. Because it depends only on D, the angle's
    # derivative with respect to any variable x is -h * D'(x) / 2, and the callers
    # supply D'(x) for their own x.
    #
    # Geometrically, delta is 16 times the square of the triangle's area (Heron's
    # formula), so delta is 0.0 when the triangle flattens into a line. There the
    # linkage locks up, h is infinite, and the angle can't change smoothly.  VALIDATED
    # CORRECT
    delta = 4.0 * L2 * D - numerator**2
    N = D - L2 + Lo2
    root_delta = np.sqrt(delta)
    h = N / (D * root_delta)

    # The second derivative of the angle with respect to x follows from the product rule
    # on -h(D) * D'(x) / 2:
    #
    # d^2(angle)/dx^2 = -(dh/dD * D'(x)^2 + h * D''(x)) / 2
    #
    # so the callers also need dh/dD. Writing h = N * D^-1 * delta^-0.5 and
    # differentiating each factor with respect to D (dN/dD = 1.0, and d(delta)/dD =
    # delta_derivative) gives the three terms below, one per factor.
    delta_derivative = 4.0 * L2 - 2.0 * numerator
    # VALIDATED CORRECT
    h_derivative = (
        # This is the term from differentiating N.
        1.0 / (D * root_delta)
        # This is the term from differentiating D^-1.
        - N / (D**2 * root_delta)
        # This is the term from differentiating delta^-0.5.
        - N * delta_derivative / (2.0 * D * delta**1.5)
    )
    return angle, h, h_derivative


class CrankLinkage:
    """A class used to evaluate the kinematics of a crank linkage.

    VALIDATED CORRECT

    A gearbox crank of length gearbox_crank_length turns about the gearbox origin. A
    rigid link of length linkage_length joins its tip to the tip of a wing crank of
    length wing_crank_length, which turns about the shoulder origin with the wing. The
    shoulder origin sits at (gearbox_to_shoulder_y, gearbox_to_shoulder_z) from the
    gearbox origin in the body y and z axes. The wing crank is rotated from the wing by
    wing_crank_angle_offset_rad.

    The crank angle q is the gearbox crank's angle from the body y axis. The wing angle
    follows from it by the linkage's forward kinematics, and every derivative of the
    wing angle with respect to q is analytical. The inverse kinematics has two solutions
    for each wing angle, and the derivatives of both are analytical as well.
    """

    __slots__ = (
        "_gearbox_to_shoulder_y",
        "_gearbox_to_shoulder_z",
        "_wing_crank_length",
        "_wing_crank_angle_offset_rad",
        "_gearbox_crank_length",
        "_linkage_length",
        "_GoSoDist_squared",
    )

    def __init__(
        self,
        gearbox_to_shoulder_y: float | int,
        gearbox_to_shoulder_z: float | int,
        wing_crank_length: float | int,
        wing_crank_angle_offset_rad: float | int,
        gearbox_crank_length: float | int,
        linkage_length: float | int,
    ) -> None:
        """The initialization method.

        :param gearbox_to_shoulder_y: The y component of the shoulder origin's position
            (in body axes, relative to the gearbox origin). It must be a number (int or
            float) and will be converted internally to a float. The units are in meters.
        :param gearbox_to_shoulder_z: The z component of the shoulder origin's position
            (in body axes, relative to the gearbox origin). It must be a number (int or
            float) and will be converted internally to a float. The units are in meters.
        :param wing_crank_length: The length of the wing crank. It must be a positive
            number (int or float) and will be converted internally to a float. The units
            are in meters.
        :param wing_crank_angle_offset_rad: The angle from the wing to the wing crank.
            It must be a number (int or float) and will be converted internally to a
            float. The units are in radians.
        :param gearbox_crank_length: The length of the gearbox crank. It must be a
            positive number (int or float) and will be converted internally to a float.
            The units are in meters.
        :param linkage_length: The length of the link between the two cranks. It must be
            a positive number (int or float) and will be converted internally to a
            float. The units are in meters.
        :return: None
        """
        self._gearbox_to_shoulder_y = (
            _parameter_validation.number_in_range_return_float(
                gearbox_to_shoulder_y, "gearbox_to_shoulder_y"
            )
        )
        self._gearbox_to_shoulder_z = (
            _parameter_validation.number_in_range_return_float(
                gearbox_to_shoulder_z, "gearbox_to_shoulder_z"
            )
        )
        self._wing_crank_length = _parameter_validation.number_in_range_return_float(
            wing_crank_length, "wing_crank_length", 0.0, False
        )
        self._wing_crank_angle_offset_rad = (
            _parameter_validation.number_in_range_return_float(
                wing_crank_angle_offset_rad, "wing_crank_angle_offset_rad"
            )
        )
        self._gearbox_crank_length = _parameter_validation.number_in_range_return_float(
            gearbox_crank_length, "gearbox_crank_length", 0.0, False
        )
        self._linkage_length = _parameter_validation.number_in_range_return_float(
            linkage_length, "linkage_length", 0.0, False
        )
        # Store the squared magnitude of p_GoSo_B (GoSoDist^2), which appears in the
        # derivatives of both kinematics directions.
        self._GoSoDist_squared = (
            self._gearbox_to_shoulder_y**2 + self._gearbox_to_shoulder_z**2
        )

        # The linkage must stay assembled through a full turn of the gearbox crank, or
        # the wing angle is undefined at some crank angles.
        crank_angles = np.linspace(0.0, 2.0 * np.pi, _NUM_ASSEMBLY_CHECK_ANGLES)
        if not np.isfinite(self.wing_angle(crank_angles)).all():
            raise ValueError(
                "The linkage can't assemble at every crank angle, so its lengths and "
                "positions don't describe a crank that turns through a full revolution."
            )

    def _forward_terms(
        self, reduction_angle: np.ndarray | float
    ) -> tuple[np.ndarray, ...]:
        """Finds the quantities that the forward kinematics and its derivatives share.

        VALIDATED CORRECT
        :param reduction_angle: The crank angle, in radians.
        :return: A tuple of GcSoDist^2, p_GoSo_along_crank, p_GoSo_across_crank,
            ang_GcSo_B_y_axis, and ang_GcSoVec_wing_crank, followed by h and its
            derivative with respect to GcSoDist^2.
        """
        # These follow the steps of the blade element theory code's calcCrankshaftFK,
        # with the same variable names. The shoulder origin's position relative to the
        # gearbox origin is p_GoSo_B, the crank angle is reduction_angle, and the
        # gearbox crank's length is gearbox_crank_length.
        p_GoSo_B_y = self._gearbox_to_shoulder_y
        p_GoSo_B_z = self._gearbox_to_shoulder_z
        gearbox_crank_length = self._gearbox_crank_length
        reduction_angle_rad = np.asarray(reduction_angle, dtype=float)
        cos_reduction_angle = np.cos(reduction_angle_rad)
        sin_reduction_angle = np.sin(reduction_angle_rad)

        # Get the position of the gearbox crank tip with respect to the gearbox origin.
        p_GoGc_B_y = gearbox_crank_length * cos_reduction_angle
        p_GoGc_B_z = gearbox_crank_length * sin_reduction_angle

        # Get the position from the gearbox crank tip to the shoulder origin.
        p_GcSo_B_y = -p_GoGc_B_y + p_GoSo_B_y
        p_GcSo_B_z = -p_GoGc_B_z + p_GoSo_B_z

        # Get the angle of this vector with respect to the body y axis. VALIDATED
        # CORRECT
        ang_GcSo_B_y_axis_rad = np.arctan2(p_GcSo_B_z, p_GcSo_B_y)

        # Get the squared magnitude of p_GcSo_B (GcSoDist^2). The squared form is kept
        # because every derivative below is simpler in it. VALIDATED CORRECT
        GcSoDist_squared = p_GcSo_B_y**2 + p_GcSo_B_z**2

        # These two projections aren't in the blade element theory code. They are
        # p_GoSo_B projected onto the gearbox crank's direction and onto the direction
        # perpendicular to it, and they shorten the derivatives, because the derivative
        # of the first with respect to reduction_angle is the second, and the derivative
        # of the second is minus the first. With them, GcSoDist^2 = GoSoDist^2 +
        # gearbox_crank_length^2 - 2 * gearbox_crank_length * p_GoSo_along_crank.
        p_GoSo_along_crank = p_GoSo_B_y * cos_reduction_angle + p_GoSo_B_z * (
            sin_reduction_angle
        )
        p_GoSo_across_crank = -p_GoSo_B_y * sin_reduction_angle + p_GoSo_B_z * (
            cos_reduction_angle
        )

        # Get the angle between p_GcSo_B and the wing crank from the law of cosines on
        # the triangle formed by p_GcSo_B, the wing crank, and the linkage. The wing
        # crank's side is fixed, and the linkage's side is opposite the angle. Also get
        # the terms that this angle's derivatives are built from.
        ang_GcSoVec_wing_crank_rad, h, h_derivative = _law_of_cosines_terms(
            self._wing_crank_length, self._linkage_length, GcSoDist_squared
        )
        return (
            GcSoDist_squared,
            p_GoSo_along_crank,
            p_GoSo_across_crank,
            ang_GcSo_B_y_axis_rad,
            ang_GcSoVec_wing_crank_rad,
            h,
            h_derivative,
        )

    def wing_angle(self, crank_angle: np.ndarray | float) -> np.ndarray:
        """Returns the wing angle at a crank angle, by the forward kinematics.

        VALIDATED CORRECT
        :param crank_angle: The crank angle, in radians. Can be a float or a ndarray.
        :return: The wing angle, in radians, with the shape of crank_angle. It is nan
            where the linkage can't assemble.
        """
        with np.errstate(invalid="ignore", divide="ignore"):
            _, _, _, ang_GcSo_B_y_axis_rad, ang_GcSoVec_wing_crank_rad, _, _ = (
                self._forward_terms(crank_angle)
            )

        # The angle of the wing crank with respect to the y axis is pi -
        # (ang_GcSoVec_wing_crank - ang_GcSo_B_y_axis).
        wing_crank_angle_wrt_y_axis_rad = np.pi - (
            ang_GcSoVec_wing_crank_rad - ang_GcSo_B_y_axis_rad
        )

        # The wing angle is wing_crank_angle_wrt_y_axis -
        # wing_crank_conn_angle_wrt_wing.
        return np.asarray(
            wing_crank_angle_wrt_y_axis_rad - self._wing_crank_angle_offset_rad,
            dtype=float,
        )

    def wing_angle_jacobian(self, crank_angle: np.ndarray | float) -> np.ndarray:
        """Returns the derivative of the wing angle with respect to the crank angle.

        VALIDATED CORRECT - DIFFERENT FORMULATION

        :param crank_angle: The crank angle, in radians. Can be a float or a ndarray.
        :return: The derivative, with the shape of crank_angle. It is unitless.
        """
        gearbox_crank_length = self._gearbox_crank_length
        with np.errstate(invalid="ignore", divide="ignore"):
            (
                GcSoDist_squared,
                p_GoSo_along_crank,
                p_GoSo_across_crank,
                _,
                _,
                h,
                _,
            ) = self._forward_terms(crank_angle)

        # The wing angle is pi - ang_GcSoVec_wing_crank + ang_GcSo_B_y_axis minus a
        # constant, so its derivative is the difference of the two angles' derivatives.
        # This matches the blade element theory code's calcCrankshaftJacFK, which
        # differentiates calcCrankshaftFK symbolically. Here, a d_ prefix marks a
        # derivative with respect to reduction_angle.

        # Differentiating the arctangent gives the rate at which p_GcSo_B turns.
        d_ang_GcSo_B_y_axis = (
            gearbox_crank_length
            * (gearbox_crank_length - p_GoSo_along_crank)
            / GcSoDist_squared
        )

        # The law of cosines angle's derivative is -h * d(GcSoDist^2) / 2.
        d_GcSoDist_squared = -2.0 * gearbox_crank_length * p_GoSo_across_crank
        d_ang_GcSoVec_wing_crank = -0.5 * h * d_GcSoDist_squared
        return np.asarray(d_ang_GcSo_B_y_axis - d_ang_GcSoVec_wing_crank, dtype=float)

    def wing_angle_hessian(self, crank_angle: np.ndarray | float) -> np.ndarray:
        """Returns the second derivative of the wing angle with respect to the crank
        angle.

        VALIDATED CORRECT - DIFFERENT FORMULATION

        :param crank_angle: The crank angle, in radians. Can be a float or a ndarray.
        :return: The second derivative, with the shape of crank_angle. It is unitless.
        """
        gearbox_crank_length = self._gearbox_crank_length
        with np.errstate(invalid="ignore", divide="ignore"):
            (
                GcSoDist_squared,
                p_GoSo_along_crank,
                p_GoSo_across_crank,
                _,
                _,
                h,
                h_derivative,
            ) = self._forward_terms(crank_angle)

        # The second derivative is the difference of the two angles' second derivatives.
        # This matches the blade element theory code's calcCrankshaftHessFK, which
        # differentiates the Jacobian symbolically. In a gearbox it would be 0.0, but
        # the linkage's mechanical advantage changes with the crank angle, so it isn't.
        # Here, a d_ prefix marks a first derivative with respect to reduction_angle,
        # and a dd_ prefix marks a second derivative.

        # Differentiate d_ang_GcSo_B_y_axis again, using the projections' derivatives.
        dd_ang_GcSo_B_y_axis = (
            gearbox_crank_length
            * p_GoSo_across_crank
            * (gearbox_crank_length**2 - self._GoSoDist_squared)
            / GcSoDist_squared**2
        )

        # Get the first and second derivatives of GcSoDist^2, and combine them with h
        # and dh/d(GcSoDist^2) by the chain rule.
        d_GcSoDist_squared = -2.0 * gearbox_crank_length * p_GoSo_across_crank
        dd_GcSoDist_squared = 2.0 * gearbox_crank_length * p_GoSo_along_crank
        dd_ang_GcSoVec_wing_crank = -0.5 * (
            h_derivative * d_GcSoDist_squared**2 + h * dd_GcSoDist_squared
        )
        return np.asarray(dd_ang_GcSo_B_y_axis - dd_ang_GcSoVec_wing_crank, dtype=float)

    def _inverse_terms(self, wing_angle: np.ndarray | float) -> tuple[np.ndarray, ...]:
        """Finds the quantities that the inverse kinematics and its derivatives share.

        VALIDATED CORRECT

        :param wing_angle: The wing angle, in radians.
        :return: A tuple of ang_GoWc_B_y_axis and its first and second derivatives with
            respect to the wing angle, followed by ang_gearbox_crank_GoWcVec and its
            first and second derivatives with respect to the wing angle.
        """
        # These follow the steps of the blade element theory code's
        # calcCrankshaftIKNoMod, with the same variable names, and the derivatives match
        # its calcCrankshaftJacIK and calcCrankshaftHessIK, which differentiate it
        # symbolically. Here, a d_ prefix marks a first derivative with respect to the
        # wing angle, and a dd_ prefix marks a second derivative.
        p_GoSo_B_y = self._gearbox_to_shoulder_y
        p_GoSo_B_z = self._gearbox_to_shoulder_z
        wing_crank_connection_length = self._wing_crank_length
        wing_angle_rad = np.asarray(wing_angle, dtype=float)

        # First get the wing crank angle with respect to the body y axis.
        wing_crank_ang_rad = self._wing_crank_angle_offset_rad + wing_angle_rad
        cos_wing_crank_ang = np.cos(wing_crank_ang_rad)
        sin_wing_crank_ang = np.sin(wing_crank_ang_rad)

        # Get the wing crank tip's position with respect to the shoulder origin.
        p_SoWc_B_y = wing_crank_connection_length * cos_wing_crank_ang
        p_SoWc_B_z = wing_crank_connection_length * sin_wing_crank_ang

        # Get the vector from the gearbox origin to the wing crank tip.
        p_GoWc_B_y = p_GoSo_B_y + p_SoWc_B_y
        p_GoWc_B_z = p_GoSo_B_z + p_SoWc_B_z

        # Get its squared magnitude (GoWcDist^2). The squared form is kept because every
        # derivative below is simpler in it.
        GoWcDist_squared = p_GoWc_B_y**2 + p_GoWc_B_z**2

        # These two projections aren't in the blade element theory code. They are
        # p_GoSo_B projected onto the wing crank's direction and onto the direction
        # perpendicular to it, and they shorten the derivatives, because the derivative
        # of the first with respect to the wing angle is the second, and the derivative
        # of the second is minus the first.
        p_GoSo_along_wing_crank = p_GoSo_B_y * cos_wing_crank_ang + p_GoSo_B_z * (
            sin_wing_crank_ang
        )
        p_GoSo_across_wing_crank = -p_GoSo_B_y * sin_wing_crank_ang + p_GoSo_B_z * (
            cos_wing_crank_ang
        )

        # Get the angle of p_GoWc_B with respect to the body y axis, and its first and
        # second derivatives from differentiating the arctangent.
        ang_GoWc_B_y_axis_rad = np.arctan2(p_GoWc_B_z, p_GoWc_B_y)
        d_ang_GoWc_B_y_axis = (
            wing_crank_connection_length
            * (p_GoSo_along_wing_crank + wing_crank_connection_length)
            / GoWcDist_squared
        )
        dd_ang_GoWc_B_y_axis = (
            wing_crank_connection_length
            * p_GoSo_across_wing_crank
            * (self._GoSoDist_squared - wing_crank_connection_length**2)
            / GoWcDist_squared**2
        )

        # Get the angle between p_GoWc_B and the gearbox crank from the law of cosines
        # on the triangle formed by p_GoWc_B, the gearbox crank, and the linkage. The
        # gearbox crank's side is fixed, and the linkage's side is opposite the angle.
        ang_gearbox_crank_GoWcVec_rad, h, h_derivative = _law_of_cosines_terms(
            self._gearbox_crank_length, self._linkage_length, GoWcDist_squared
        )

        # Get the first and second derivatives of GoWcDist^2, and combine them with h
        # and dh/d(GoWcDist^2) by the chain rule to get the angle's derivatives.
        d_GoWcDist_squared = (
            2.0 * wing_crank_connection_length * (p_GoSo_across_wing_crank)
        )
        dd_GoWcDist_squared = (
            -2.0 * wing_crank_connection_length * (p_GoSo_along_wing_crank)
        )
        d_ang_gearbox_crank_GoWcVec = -0.5 * h * d_GoWcDist_squared
        dd_ang_gearbox_crank_GoWcVec = -0.5 * (
            h_derivative * d_GoWcDist_squared**2 + h * dd_GoWcDist_squared
        )
        return (
            ang_GoWc_B_y_axis_rad,
            d_ang_GoWc_B_y_axis,
            dd_ang_GoWc_B_y_axis,
            ang_gearbox_crank_GoWcVec_rad,
            d_ang_gearbox_crank_GoWcVec,
            dd_ang_gearbox_crank_GoWcVec,
        )

    def crank_angles(
        self, wing_angle: np.ndarray | float, upstroke: np.ndarray | bool
    ) -> np.ndarray:
        """Returns the crank angle for a wing angle, by the inverse kinematics.

        VALIDATED CORRECT

        :param wing_angle: The wing angle, in radians. Can be a float or a ndarray.
        :param upstroke: Whether to take the solution for an upstroke (True) or for a
            downstroke (False). It broadcasts against wing_angle.
        :return: The crank angle, in radians, with the shape of wing_angle. The upstroke
            solution is not wrapped, and the downstroke solution is wrapped to [0, 2 *
            pi).
        """
        with np.errstate(invalid="ignore", divide="ignore"):
            ang_GoWc_B_y_axis_rad, _, _, ang_gearbox_crank_GoWcVec_rad, _, _ = (
                self._inverse_terms(wing_angle)
            )

        # Reduction angle 1 adds the angle between the gearbox crank and p_GoWc_B to the
        # angle of p_GoWc_B.
        reduction_angle_1_rad = ang_GoWc_B_y_axis_rad + ang_gearbox_crank_GoWcVec_rad

        # Reduction angle 2 is mirrored about p_GoWc_B, so it subtracts the angle. Like
        # calcCrankshaftIK, it is wrapped to [0, 2 * pi).
        reduction_angle_2_rad = np.mod(
            ang_GoWc_B_y_axis_rad - ang_gearbox_crank_GoWcVec_rad, 2.0 * np.pi
        )

        # Pick the solution by the direction of the stroke (upstroke_Q in
        # convertWingTrajToMotorAnalytical).
        return np.where(upstroke, reduction_angle_1_rad, reduction_angle_2_rad)

    def crank_angle_derivatives(
        self, wing_angle: np.ndarray | float, upstroke: np.ndarray | bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns the first and second derivatives of the crank angle with respect to
        the wing angle.

        VALIDATED CORRECT

        Both diverge where the wing angle reaches an extreme of its stroke, because the
        wing barely moves there while the crank still turns.

        :param wing_angle: The wing angle, in radians. Can be a float or a ndarray.
        :param upstroke: Whether to take the solution for an upstroke (True) or for a
            downstroke (False). It broadcasts against wing_angle.
        :return: A tuple of the first and second derivatives, each with the shape of
            wing_angle. They are unitless.
        """
        with np.errstate(invalid="ignore", divide="ignore"):
            (
                _,
                d_ang_GoWc_B_y_axis,
                dd_ang_GoWc_B_y_axis,
                _,
                d_ang_gearbox_crank_GoWcVec,
                dd_ang_gearbox_crank_GoWcVec,
            ) = self._inverse_terms(wing_angle)

        # Each reduction angle is ang_GoWc_B_y_axis plus or minus
        # ang_gearbox_crank_GoWcVec, so its derivatives are the sums or differences of
        # theirs (dRedOne_dWing and dRedTwo_dWing in calcCrankshaftJacIK, and
        # ddRedOne_ddWing and ddRedTwo_ddWing in calcCrankshaftHessIK).
        d_reduction_angle = np.where(
            upstroke,
            d_ang_GoWc_B_y_axis + d_ang_gearbox_crank_GoWcVec,
            d_ang_GoWc_B_y_axis - d_ang_gearbox_crank_GoWcVec,
        )
        dd_reduction_angle = np.where(
            upstroke,
            dd_ang_GoWc_B_y_axis + dd_ang_gearbox_crank_GoWcVec,
            dd_ang_GoWc_B_y_axis - dd_ang_gearbox_crank_GoWcVec,
        )
        return d_reduction_angle, dd_reduction_angle

    def convert_trajectory(
        self,
        gear_ratio: float,
        wing_angles: np.ndarray,
        wing_angle_derivatives: np.ndarray,
        wing_angle_second_derivatives: np.ndarray,
        derivative_method: str = "analytical",
        times: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Converts a wing angle trajectory into the motor trajectory that produces it,
        using either the analytical derivatives of the inverse kinematics or numerical
        differentiation in time.

        VALIDATED CORRECT

        The inverse kinematics solution is chosen by the direction of the wing's motion,
        an upstroke where the wing angle is increasing and a downstroke otherwise. The
        crank angle is then unwrapped so that it is continuous, and every motor quantity
        is the crank quantity scaled by the gear ratio.

        The derivatives of the inverse kinematics diverge at the extremes of the stroke,
        and the product of a diverging derivative with a vanishing wing velocity loses a
        little precision there. A trajectory that is consistent with a turning crank,
        meaning one that reaches the full stroke at each turnaround, recovers the motor
        trajectory to well below the size of its accelerations. A trajectory that turns
        around short of the full stroke is not consistent with a turning crank, and the
        solution switch at each turnaround makes the motor position jump.

        With the "analytical" method, the turnaround samples get special handling that
        the blade element theory code's convertWingTrajToMotorAnalytical doesn't have.
        At a turnaround the wing is at the end of its stroke, where the linkage sits at
        a dead center. There, dphi/dq is 0.0, so the inverse kinematics' Jacobian dq/dphi
        = 1 / (dphi/dq) is infinite, while the wing's velocity phi' is 0.0. The motor
        velocity q' = (dq/dphi) * phi' is then 0.0 * infinity. Its true value is finite,
        because the crank keeps turning through a dead center, but evaluating the
        product directly gives nan, infinity, or a huge number. The acceleration has the
        same problem, worse, because d^2q/dphi^2 diverges faster than dq/dphi. This is
        what makes the motor velocity and acceleration spike at every turnaround in that
        code. A wing angle that round off pushes just past the end of the stroke also
        makes the arccos argument exceed 1.0, which gives a complex number in MATLAB and
        nan here.

        The samples near a turnaround have a related problem. There, the motor
        acceleration is the sum of two large terms, (dq/dphi) * phi'' and (d^2q/dphi^2)
        * phi'^2, that nearly cancel, so it loses precision and spikes even a few
        samples away from the turnaround.

        To avoid those spikes, this method marks the samples at each turnaround: those
        where the wing's speed is a negligible fraction of its largest speed in the
        trajectory, and the two on either side of each change in the stroke direction.
        It widens each marked turnaround by _TURNAROUND_FILL_HALF_WIDTH samples on each
        side, and also marks any sample whose motor velocity or acceleration isn't
        finite. It then replaces the motor velocity and acceleration at each marked
        sample by linear interpolation between the nearest unmarked samples on either
        side. The motor position is never replaced, because the inverse kinematics gives
        it accurately even at a dead center. The replaced values approximate the
        trajectory rather than computing it exactly, which is accurate when the filled
        width is small compared with the flapping period. With a 1e-4 s step at 2 Hz,
        that width is about 0.4% of a period.

        The "numerical" method never uses the diverging derivatives, so it needs no
        such handling. However, it still turns a jump in the motor position into a
        spike in velocity and acceleration.

        :param gear_ratio: The ratio from the motor to the gearbox crank. It is
            unitless.
        :param wing_angles: A (N,) ndarray of the wing angles, in radians.
        :param wing_angle_derivatives: A (N,) ndarray of the wing angles' time
            derivatives, in radians per second.
        :param wing_angle_second_derivatives: A (N,) ndarray of the wing angles' second
            time derivatives, in radians per second squared.
        :param derivative_method: How to find the motor's velocity and acceleration.
            Valid options are "analytical", which applies the chain rule with the
            inverse kinematics' analytical Jacobian and Hessian (the blade element
            theory code's convertWingTrajToMotorAnalytical), and "numerical", which
            differentiates the motor position in time with second order central
            differences (its convertWingTrajToMotorNumerical). "numerical" ignores the
            wing's velocity and acceleration, except for picking the stroke direction,
            and its accuracy depends on the time step. The default is "analytical".
        :param times: A (N,) ndarray of the sample times, in seconds. It is required for
            the "numerical" method and ignored by the "analytical" method. The default
            is None.
        :return: A tuple of three (N,) ndarrays holding the motor position (rad),
            velocity (rad/s), and acceleration (rad/s^2).
        """
        if derivative_method not in DERIVATIVE_METHODS:
            raise ValueError(
                f"derivative_method must be one of {DERIVATIVE_METHODS}, but it is "
                f"{derivative_method}."
            )
        if derivative_method == "numerical" and times is None:
            raise ValueError('The "numerical" derivative_method needs times.')

        # Both methods start the same way. First pick the inverse kinematics solution
        # for each sample by the direction of the stroke (upstroke_Q there), and get the
        # crank angle from it.
        upstroke_Q = wing_angle_derivatives > 0.0

        # Unwrap the reduction angle so that it is continuous from sample to sample,
        # like that code's unwrap_custom. It unwraps by 2 * pi * gearbox_reduction after
        # scaling to the motor, which is the same thing.
        red_pos_rad = np.unwrap(self.crank_angles(wing_angles, upstroke_Q))

        if times is not None and derivative_method == "numerical":
            # This follows convertWingTrajToMotorNumerical: differentiate the motor
            # position in time twice with gradient, which uses second order central
            # differences inside the trajectory and one sided differences at its ends.
            # Unlike that code, which uses the mean time step, this uses the actual
            # sample times, so uneven spacing is handled too.
            sample_times = np.asarray(times, dtype=float)
            motor_pos_vec_des = np.asarray(gear_ratio * red_pos_rad, dtype=float)
            motor_velo_vec_des = np.asarray(
                np.gradient(motor_pos_vec_des, sample_times), dtype=float
            )
            motor_acc_vec_des = np.asarray(
                np.gradient(motor_velo_vec_des, sample_times), dtype=float
            )
            return motor_pos_vec_des, motor_velo_vec_des, motor_acc_vec_des

        # Otherwise, follow convertWingTrajToMotorAnalytical.

        # Convert the wing's velocity and acceleration with the chain rule, using the
        # inverse kinematics Jacobian and Hessian (ik_jac_red1 or ik_jac_red2, and
        # ik_hess_red1 or ik_hess_red2, there) to get red_velo_curr and red_accel_curr:
        #
        # q' = (dq/dphi) * phi'
        #
        # q'' = (dq/dphi) * phi'' + (d^2q/dphi^2) * phi'^2
        with np.errstate(invalid="ignore", over="ignore"):
            ik_jac_red, ik_hess_red = self.crank_angle_derivatives(
                wing_angles, upstroke_Q
            )
            red_velo_rad = ik_jac_red * wing_angle_derivatives
            red_accel_rad = (
                ik_jac_red * wing_angle_second_derivatives
                + ik_hess_red * wing_angle_derivatives**2
            )

        # At a sample exactly on a turnaround, the wing's velocity is 0.0 while the
        # inverse kinematics' derivatives are infinite, so the products above are an
        # indeterminate 0.0 * infinity that evaluates to 0.0 or to a non finite value
        # instead of their finite limits. Fill those samples from their neighbors, as
        # the docstring describes.
        #
        # First, find the indeterminate samples. A sample is on a turnaround if the
        # wing's speed there is at most _INDETERMINATE_RATE_FRACTION (1e-9) of its
        # largest speed. The threshold is relative, so it doesn't depend on the flapping
        # frequency or the stroke amplitude. Any sample whose motor velocity or
        # acceleration came out nan or infinite is marked too, which also catches a wing
        # angle just past the end of the stroke.
        largest_rate = np.max(np.abs(wing_angle_derivatives))
        turnaround = (
            np.abs(wing_angle_derivatives)
            <= _INDETERMINATE_RATE_FRACTION * largest_rate
        )

        # A turnaround usually falls between two samples rather than on one, so also
        # mark both samples on either side of every change in the stroke direction.
        direction_changes = np.flatnonzero(upstroke_Q[1:] != upstroke_Q[:-1])
        turnaround[direction_changes] = True
        turnaround[direction_changes + 1] = True

        # Widen each turnaround by _TURNAROUND_FILL_HALF_WIDTH samples on each side, to
        # also cover the samples whose acceleration has lost precision, by marking every
        # sample within that many samples of a marked one.
        widened = turnaround.copy()
        for offset in range(1, _TURNAROUND_FILL_HALF_WIDTH + 1):
            widened[offset:] |= turnaround[:-offset]
            widened[:-offset] |= turnaround[offset:]
        indeterminate = (
            widened | ~np.isfinite(red_velo_rad) | ~np.isfinite(red_accel_rad)
        )
        # Then replace the marked samples' motor velocity and acceleration by linear
        # interpolation, by sample index, between the nearest unmarked samples on either
        # side. A marked sample at either end of the trajectory takes the value of the
        # nearest unmarked sample, because interp holds its end values. If no sample is
        # marked there is nothing to fill, and if every sample is marked there is
        # nothing to fill from, so both cases are left unchanged.
        if indeterminate.any() and not indeterminate.all():
            sample_indices = np.arange(len(wing_angles))
            valid = ~indeterminate
            red_velo_rad = np.interp(
                sample_indices, sample_indices[valid], red_velo_rad[valid]
            )
            red_accel_rad = np.interp(
                sample_indices, sample_indices[valid], red_accel_rad[valid]
            )

        # Adjust for the gearbox reduction to get the motor's position, velocity, and
        # acceleration (motor_pos_curr, motor_velo_curr, and motor_accel_curr there).
        return (
            gear_ratio * red_pos_rad,
            gear_ratio * red_velo_rad,
            gear_ratio * red_accel_rad,
        )
