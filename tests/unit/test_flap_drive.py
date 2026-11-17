"""This module contains classes to test FlapTransmissions, FlapReferences, and
FlapDrives."""

import unittest
from typing import Any

import numpy as np
import numpy.testing as npt

from pterasoftware import crank_linkage, flap_drive
from pterasoftware import operating_point as operating_point_mod
from tests.unit.fixtures import flap_drive_fixtures

WING_INERTIA = 0.4

# These are cases computed with the blade element theory code's MATLAB
# singleWingDynamics (MATLAB R2025b), for a cross check of
# FlapDrive.single_wing_dynamics. The setup is the pterosaur crank linkage (gearbox to
# shoulder offsets 0.11 m and -0.05 m, a 0.065 m wing crank at pi / 2 to the wing, a
# 0.032 m gearbox crank, and a 0.11 m link) with a 15:1 gearbox, a 0.0004158 kg*m^2
# rotor with 0.001 N*m*s/rad of damping, a motor spring of 0.3 N*m/rad about 0.1 rad, a
# wing spring of 5.0 N*m/rad about -0.2 rad, and the model 5 wing, whose inertia about
# the flap axis that code computes as _MATLAB_WING_INERTIA. Each case is (motor position
# (rad), motor velocity (rad/s), applied motor torque (N*m), the aerodynamic reaction
# torque on the wing that its blade element theory model computed (N*m), and its
# outputs: the motor acceleration (rad/s^2), the flap acceleration (rad/s^2), and
# wing_torques.total, the torque the linkage supplies to the wing (N*m)). That code's
# torques are reaction torques, so the fluid torque on the wing is the negative of the
# aerodynamic reaction torque.
_MATLAB_WING_INERTIA = 1.46104774864228
_MATLAB_SINGLE_WING_DYNAMICS_CASES = (
    (
        0.3,
        373.25131775386097,
        -0.98009024549762802,
        -23.091272511606999,
        -1228.1299227500999,
        472.25359547361899,
        665.33527599096396,
    ),
    (
        32.153094636450199,
        -109.351603019656,
        -0.87062078862560799,
        -66.689097230038897,
        -5349.4718137604495,
        -160.072308327071,
        -298.13498499815103,
    ),
    (
        64.0061892729004,
        501.45025341865102,
        2.4815680635814501,
        -238.42918963465999,
        -28892.674426150599,
        319.51098026136202,
        231.07006938728401,
    ),
    (
        95.859283909350594,
        352.98659660359903,
        1.36428401567644,
        5.0722052374009499,
        -68726.299552950994,
        195.860905526636,
        289.68124184093699,
    ),
    (
        127.712378545801,
        250.86163937199601,
        6.2544520405727804,
        104.45758069530299,
        -23098.841106787699,
        -676.29424039542596,
        -881.03904473400496,
    ),
    (
        143.63892586402599,
        -6.99568241924867,
        -5.7820089147411302,
        -13.9235079708256,
        -116261.086664348,
        202.05996959501701,
        284.88842016224498,
    ),
    (
        159.565473182251,
        275.44147921764801,
        5.4514760957385997,
        -138.93225679528501,
        -39056.970278010798,
        843.20504011574997,
        1095.5564722531201,
    ),
    (
        191.41856781870101,
        265.34576067823502,
        -4.85447599310067,
        19.716261191079798,
        -132747.10844267701,
        -723.79906628244999,
        -1039.3088586086001,
    ),
)


class TestFlapTransmission(unittest.TestCase):
    """This is a class with functions to test FlapTransmissions."""

    def setUp(self) -> None:
        """Set up test fixtures for FlapTransmission tests."""
        self.identity_transmission = flap_drive.FlapTransmission.identity()
        self.nonlinear_transmission = (
            flap_drive_fixtures.make_nonlinear_transmission_fixture()
        )

    def test_initialization_stores_functions(self) -> None:
        """Test that the nonlinear transmission evaluates its supplied functions."""
        for position in (-2.0, 0.0, 0.3, 1.7):
            self.assertAlmostEqual(
                self.nonlinear_transmission.evaluate_angle(position),
                flap_drive_fixtures.nonlinear_angle(position),
            )
            self.assertAlmostEqual(
                self.nonlinear_transmission.evaluate_jacobian(position),
                flap_drive_fixtures.nonlinear_jacobian(position),
            )
            self.assertAlmostEqual(
                self.nonlinear_transmission.evaluate_hessian(position),
                flap_drive_fixtures.nonlinear_hessian(position),
            )

    def test_initialization_rejects_non_callables(self) -> None:
        """Test that each of the functions must be a callable."""
        not_callable: Any = 1.0
        good = flap_drive_fixtures.nonlinear_angle
        trajectory = flap_drive_fixtures.unused_trajectory
        with self.assertRaises(TypeError):
            flap_drive.FlapTransmission(not_callable, good, good, trajectory)
        with self.assertRaises(TypeError):
            flap_drive.FlapTransmission(good, not_callable, good, trajectory)
        with self.assertRaises(TypeError):
            flap_drive.FlapTransmission(good, good, not_callable, trajectory)
        with self.assertRaises(TypeError):
            flap_drive.FlapTransmission(good, good, good, not_callable)

    def test_trajectory_function_is_required(self) -> None:
        """Test that a FlapTransmission can't be made without inverse kinematics, so
        that a wing trajectory can always be converted."""
        good = flap_drive_fixtures.nonlinear_angle
        missing: Any = None
        with self.assertRaises(TypeError):
            flap_drive.FlapTransmission(good, good, good, missing)
        with self.assertRaises(TypeError):
            flap_drive.FlapTransmission.from_forward_kinematics(good, missing)

    def test_identity(self) -> None:
        """Test that the identity transmission is a direct drive."""
        for position in (-1.0, 0.0, 2.5):
            self.assertEqual(
                self.identity_transmission.evaluate_angle(position), position
            )
            self.assertEqual(
                self.identity_transmission.evaluate_jacobian(position), 1.0
            )
            self.assertEqual(self.identity_transmission.evaluate_hessian(position), 0.0)

    def test_geared(self) -> None:
        """Test that a geared transmission divides by the gear ratio."""
        geared = flap_drive.FlapTransmission.geared(5)
        self.assertAlmostEqual(geared.evaluate_angle(10.0), 2.0)
        self.assertAlmostEqual(geared.evaluate_jacobian(3.0), 0.2)
        self.assertEqual(geared.evaluate_hessian(3.0), 0.0)

    def test_geared_negative_ratio_reverses_direction(self) -> None:
        """Test that a negative gear ratio reverses the flap direction."""
        reversed_geared = flap_drive.FlapTransmission.geared(-4.0)
        self.assertAlmostEqual(reversed_geared.evaluate_angle(8.0), -2.0)
        self.assertAlmostEqual(reversed_geared.evaluate_jacobian(0.0), -0.25)

    def test_geared_validation(self) -> None:
        """Test that invalid gear ratios are rejected."""
        with self.assertRaises(ValueError):
            flap_drive.FlapTransmission.geared(0.0)
        invalid_ratios: list[Any] = ["5", None, [5.0]]
        for ratio in invalid_ratios:
            with self.assertRaises(TypeError):
                flap_drive.FlapTransmission.geared(ratio)
        for ratio in (np.nan, np.inf):
            with self.assertRaises(ValueError):
                flap_drive.FlapTransmission.geared(ratio)

    def test_from_forward_kinematics_matches_analytic_derivatives(self) -> None:
        """Test that finite difference derivatives agree with the analytic ones."""
        numerical = flap_drive.FlapTransmission.from_forward_kinematics(
            flap_drive_fixtures.nonlinear_angle, flap_drive_fixtures.unused_trajectory
        )
        for position in (-1.2, 0.0, 0.9, 2.4):
            self.assertAlmostEqual(
                numerical.evaluate_jacobian(position),
                flap_drive_fixtures.nonlinear_jacobian(position),
                places=7,
            )
            self.assertAlmostEqual(
                numerical.evaluate_hessian(position),
                flap_drive_fixtures.nonlinear_hessian(position),
                places=5,
            )

    def test_numerical_trajectory_function(self) -> None:
        """Test that the time differences of an inverse position function recover a
        gearbox's exact velocity and acceleration on a sine."""
        ratio = 15.0
        trajectory = flap_drive.numerical_trajectory_function(
            lambda angles, rates: ratio * angles
        )
        times = np.linspace(0.0, 1.0, 10001)
        omega = 2.0 * np.pi * 2.0
        angles = 0.5 * np.sin(omega * times)
        rates = 0.5 * omega * np.cos(omega * times)
        accelerations = -0.5 * omega**2 * np.sin(omega * times)
        position, velocity, acceleration = trajectory(
            times, angles, rates, accelerations
        )
        npt.assert_allclose(position, ratio * angles)
        npt.assert_allclose(velocity[2:-2], ratio * rates[2:-2], atol=1e-3)
        npt.assert_allclose(acceleration[2:-2], ratio * accelerations[2:-2], atol=1e-1)

    def test_numerical_trajectory_function_validation(self) -> None:
        """Test that a non callable, or a function returning the wrong number of
        positions, is rejected."""
        not_callable: Any = 1.0
        with self.assertRaises(TypeError):
            flap_drive.numerical_trajectory_function(not_callable)
        wrong_length = flap_drive.numerical_trajectory_function(
            lambda angles, rates: angles[:-1]
        )
        times = np.array([0.0, 1.0, 2.0])
        with self.assertRaises(ValueError):
            wrong_length(times, np.zeros(3), np.zeros(3), np.zeros(3))

    def test_from_positions(self) -> None:
        """Test that a transmission built from position maps alone has numerical
        derivatives in both directions."""
        ratio = 4.0
        transmission = flap_drive.FlapTransmission.from_positions(
            lambda position: position / ratio, lambda angles, rates: ratio * angles
        )
        self.assertAlmostEqual(transmission.evaluate_jacobian(2.0), 1.0 / ratio)
        self.assertAlmostEqual(transmission.evaluate_hessian(2.0), 0.0, places=6)
        times = np.linspace(0.0, 1.0, 11)
        angles = 0.3 * times
        position, velocity, acceleration = transmission.convert_trajectory(
            times, angles, np.full(11, 0.3), np.zeros(11)
        )
        npt.assert_allclose(position, ratio * angles)
        npt.assert_allclose(velocity, ratio * 0.3)
        npt.assert_allclose(acceleration, 0.0, atol=1e-12)

    def test_from_forward_kinematics_validation(self) -> None:
        """Test that from_forward_kinematics rejects bad arguments."""
        not_callable: Any = "angle"
        trajectory = flap_drive_fixtures.unused_trajectory
        with self.assertRaises(TypeError):
            flap_drive.FlapTransmission.from_forward_kinematics(
                not_callable, trajectory
            )
        good = flap_drive_fixtures.nonlinear_angle
        for bad_step in (0.0, -1e-4):
            with self.assertRaises(ValueError):
                flap_drive.FlapTransmission.from_forward_kinematics(
                    good, trajectory, bad_step
                )
        bad_type_step: Any = "small"
        with self.assertRaises(TypeError):
            flap_drive.FlapTransmission.from_forward_kinematics(
                good, trajectory, bad_type_step
            )

    def test_non_finite_function_output_raises(self) -> None:
        """Test that a function returning nan or inf is caught when it is evaluated."""
        broken = flap_drive.FlapTransmission(
            lambda position: np.nan,
            lambda position: np.inf,
            lambda position: -np.inf,
            flap_drive_fixtures.unused_trajectory,
        )
        with self.assertRaises(ValueError):
            broken.evaluate_angle(0.0)
        with self.assertRaises(ValueError):
            broken.evaluate_jacobian(0.0)
        with self.assertRaises(ValueError):
            broken.evaluate_hessian(0.0)

    def test_slots(self) -> None:
        """Test that the class has no instance dict and rejects new attributes."""
        self.assertFalse(hasattr(self.identity_transmission, "__dict__"))
        with self.assertRaises(AttributeError):
            setattr(self.identity_transmission, "new_attribute", 1.0)

    # --- Trajectory conversion ---
    def test_convert_trajectory_identity(self) -> None:
        """Test that a direct drive returns copies of the flap trajectory."""
        times = np.array([0.0, 0.1, 0.2])
        angles = np.array([0.0, 0.1, 0.2])
        rates = np.array([1.0, 2.0, 3.0])
        accelerations = np.array([4.0, 5.0, 6.0])
        position, velocity, acceleration = (
            self.identity_transmission.convert_trajectory(
                times, angles, rates, accelerations
            )
        )
        npt.assert_array_equal(position, angles)
        npt.assert_array_equal(velocity, rates)
        npt.assert_array_equal(acceleration, accelerations)
        self.assertIsNot(position, angles)

    def test_convert_trajectory_geared(self) -> None:
        """Test that a gearbox multiplies the whole trajectory by the ratio."""
        geared = flap_drive.FlapTransmission.geared(25.0)
        times = np.linspace(0.0, 1.0, 5)
        angles = np.linspace(0.0, 1.0, 5)
        position, velocity, acceleration = geared.convert_trajectory(
            times, angles, 2.0 * angles, 3.0 * angles
        )
        npt.assert_allclose(position, 25.0 * angles)
        npt.assert_allclose(velocity, 50.0 * angles)
        npt.assert_allclose(acceleration, 75.0 * angles)
        # Converting back through the forward map recovers the flap angle.
        npt.assert_allclose([geared.evaluate_angle(q) for q in position], angles)

    def test_convert_trajectory_validates_inputs(self) -> None:
        """Test that mismatched, 2D, or badly ordered inputs are rejected."""
        times = np.array([0.0, 1.0, 2.0])
        with self.assertRaises(ValueError):
            self.identity_transmission.convert_trajectory(
                times, np.zeros(3), np.zeros(4), np.zeros(3)
            )
        with self.assertRaises(ValueError):
            self.identity_transmission.convert_trajectory(
                np.zeros((3, 1)), np.zeros((3, 1)), np.zeros((3, 1)), np.zeros((3, 1))
            )
        for bad_times in (
            np.array([0.0, 2.0, 1.0]),
            np.array([0.0, 1.0, 1.0]),
            np.array([0.0, 1.0, np.nan]),
        ):
            with self.assertRaises(ValueError):
                self.identity_transmission.convert_trajectory(
                    bad_times, np.zeros(3), np.zeros(3), np.zeros(3)
                )

    def test_convert_trajectory_validates_outputs(self) -> None:
        """Test that a user function returning bad data is caught."""
        good = flap_drive_fixtures.nonlinear_angle

        def wrong_count(
            t: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
        ) -> Any:
            return (a, b)

        def wrong_shape(
            t: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
        ) -> Any:
            return (a[:-1], b, c)

        def not_finite(
            t: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
        ) -> Any:
            return (a * np.nan, b, c)

        zeros = np.zeros(3)
        times = np.array([0.0, 1.0, 2.0])
        for bad in (wrong_count, wrong_shape, not_finite):
            transmission = flap_drive.FlapTransmission(good, good, good, bad)
            with self.assertRaises(ValueError):
                transmission.convert_trajectory(times, zeros, zeros, zeros)

    def test_custom_trajectory_function_is_used(self) -> None:
        """Test that a user trajectory function (like a crank's inverse kinematics) is
        what convert_trajectory calls, and that it receives the sample times."""
        received: list[np.ndarray] = []

        def doubled(t: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Any:
            received.append(t)
            return (2.0 * a, 3.0 * b, 4.0 * c)

        good = flap_drive_fixtures.nonlinear_angle
        transmission = flap_drive.FlapTransmission(good, good, good, doubled)
        times = np.array([0.0, 0.5, 1.0])
        ones = np.ones(3)
        position, velocity, acceleration = transmission.convert_trajectory(
            times, ones, ones, ones
        )
        npt.assert_array_equal(position, 2.0 * ones)
        npt.assert_array_equal(velocity, 3.0 * ones)
        npt.assert_array_equal(acceleration, 4.0 * ones)
        npt.assert_array_equal(received[0], times)


class TestFlapReference(unittest.TestCase):
    """This is a class with functions to test FlapReferences."""

    def setUp(self) -> None:
        """Set up a monotonic reference, like a continuously rotating crank."""
        self.times = np.linspace(0.0, 1.0, 11)
        self.positions = 2.0 * np.pi * self.times
        self.velocities = np.full(11, 2.0 * np.pi)
        self.accelerations = 10.0 * self.times
        self.reference = flap_drive.FlapReference(
            self.times, self.positions, self.velocities, self.accelerations, period=None
        )

    def test_initialization_stores_arrays_read_only(self) -> None:
        """Test that the table is stored as read only float arrays."""
        npt.assert_array_equal(self.reference.times, self.times)
        npt.assert_array_equal(self.reference.positions, self.positions)
        npt.assert_array_equal(self.reference.velocities, self.velocities)
        npt.assert_array_equal(self.reference.accelerations, self.accelerations)
        for array in (
            self.reference.times,
            self.reference.positions,
            self.reference.velocities,
            self.reference.accelerations,
        ):
            self.assertFalse(array.flags.writeable)

    def test_inputs_are_copied(self) -> None:
        """Test that later changes to the caller's arrays don't change the table."""
        positions = self.positions.copy()
        reference = flap_drive.FlapReference(
            self.times, positions, self.velocities, self.accelerations, period=None
        )
        positions[0] = 99.0
        self.assertEqual(reference.positions[0], 0.0)

    def test_positions_are_monotonic(self) -> None:
        """Test that the reference reports whether its positions are monotonic."""
        self.assertTrue(self.reference.positions_are_monotonic)
        back_and_forth = flap_drive.FlapReference(
            self.times,
            np.sin(2.0 * np.pi * self.times),
            self.velocities,
            self.accelerations,
            period=None,
        )
        self.assertFalse(back_and_forth.positions_are_monotonic)

    def test_position_lookup_on_non_monotonic_positions_raises(self) -> None:
        """Test that an ambiguous position lookup is refused by the reference."""
        back_and_forth = flap_drive.FlapReference(
            self.times,
            np.sin(2.0 * np.pi * self.times),
            self.velocities,
            self.accelerations,
            period=None,
        )
        with self.assertRaises(ValueError):
            back_and_forth.evaluate_at_position(0.5)

    def test_evaluate_interpolates_in_time(self) -> None:
        """Test linear interpolation, including between samples and past the ends."""
        position, velocity, acceleration = self.reference.evaluate(0.25)
        self.assertAlmostEqual(position, 2.0 * np.pi * 0.25)
        self.assertAlmostEqual(velocity, 2.0 * np.pi)
        self.assertAlmostEqual(acceleration, 2.5)
        self.assertEqual(self.reference.evaluate(-5.0)[0], 0.0)
        self.assertAlmostEqual(self.reference.evaluate(5.0)[0], 2.0 * np.pi)

    def test_evaluate_at_position(self) -> None:
        """Test interpolation of velocity and acceleration against position."""
        velocity, acceleration = self.reference.evaluate_at_position(np.pi)
        self.assertAlmostEqual(velocity, 2.0 * np.pi)
        self.assertAlmostEqual(acceleration, 5.0)

    def test_evaluate_at_position_decreasing_table(self) -> None:
        """Test that a table that runs backward is looked up correctly."""
        reference = flap_drive.FlapReference(
            self.times,
            -self.positions,
            self.velocities,
            self.accelerations,
            period=None,
        )
        self.assertTrue(reference.positions_are_monotonic)
        velocity, acceleration = reference.evaluate_at_position(-np.pi)
        self.assertAlmostEqual(velocity, 2.0 * np.pi)
        self.assertAlmostEqual(acceleration, 5.0)

    def test_validation(self) -> None:
        """Test that malformed tables are rejected."""
        good = (self.times, self.positions, self.velocities, self.accelerations)
        with self.assertRaises(ValueError):
            flap_drive.FlapReference(
                self.times[:1],
                self.positions[:1],
                self.velocities[:1],
                self.accelerations[:1],
                period=None,
            )
        with self.assertRaises(ValueError):
            flap_drive.FlapReference(
                self.times[::-1],
                self.positions,
                self.velocities,
                self.accelerations,
                period=None,
            )
        with self.assertRaises(ValueError):
            flap_drive.FlapReference(
                self.times,
                self.positions[:-1],
                self.velocities,
                self.accelerations,
                period=None,
            )
        with self.assertRaises(ValueError):
            flap_drive.FlapReference(
                self.times,
                self.positions * np.nan,
                self.velocities,
                self.accelerations,
                period=None,
            )
        with self.assertRaises(ValueError):
            flap_drive.FlapReference(
                self.times,
                self.positions.reshape(-1, 1),
                self.velocities,
                self.accelerations,
                period=None,
            )

    def test_from_flap_trajectory(self) -> None:
        """Test building a reference through a transmission's trajectory function."""
        geared = flap_drive.FlapTransmission.geared(10.0)
        angles = np.linspace(0.0, 0.5, 11)
        reference = flap_drive.FlapReference.from_flap_trajectory(
            geared, self.times, angles, 2.0 * angles, 3.0 * angles, period=None
        )
        npt.assert_allclose(reference.positions, 10.0 * angles)
        npt.assert_allclose(reference.velocities, 20.0 * angles)
        npt.assert_allclose(reference.accelerations, 30.0 * angles)

    def test_period(self) -> None:
        """Test that the period is stored as a float, can be None, is required, and must
        be positive."""
        table = (self.times, self.positions, self.velocities, self.accelerations)
        self.assertEqual(flap_drive.FlapReference(*table, period=2).period, 2.0)
        self.assertIsInstance(flap_drive.FlapReference(*table, period=2).period, float)
        self.assertIsNone(self.reference.period)
        missing: Any = flap_drive.FlapReference
        with self.assertRaises(TypeError):
            missing(*table)
        with self.assertRaises(TypeError):
            missing(*table, 1.0)
        for bad in (0.0, -1.0, np.nan, np.inf):
            with self.assertRaises(ValueError):
                flap_drive.FlapReference(*table, period=bad)
        not_a_number: Any = "1.0"
        with self.assertRaises(TypeError):
            flap_drive.FlapReference(*table, period=not_a_number)

    def test_constructors_pass_the_period(self) -> None:
        """Test that from_flap_trajectory and constant_velocity store the period."""
        angles = np.linspace(0.0, 0.5, 11)
        from_trajectory = flap_drive.FlapReference.from_flap_trajectory(
            flap_drive.FlapTransmission.identity(),
            self.times,
            angles,
            angles,
            angles,
            period=0.25,
        )
        self.assertEqual(from_trajectory.period, 0.25)
        constant = flap_drive.FlapReference.constant_velocity(0.0, 3.0, 1.0, period=0.4)
        self.assertEqual(constant.period, 0.4)
        self.assertIsNone(
            flap_drive.FlapReference.constant_velocity(
                0.0, 3.0, 1.0, period=None
            ).period
        )

    def test_from_flap_trajectory_rejects_a_non_transmission(self) -> None:
        """Test that an object that doesn't conform to the Transmission protocol is
        rejected."""
        zeros = np.zeros(11)
        not_a_transmission: Any = object()
        with self.assertRaises(TypeError):
            flap_drive.FlapReference.from_flap_trajectory(
                not_a_transmission, self.times, zeros, zeros, zeros, period=None
            )

    def test_slots(self) -> None:
        """Test that the class has no instance dict and rejects new attributes."""
        self.assertFalse(hasattr(self.reference, "__dict__"))
        with self.assertRaises(AttributeError):
            setattr(self.reference, "new_attribute", 1.0)


class TestFlapDriveRecords(unittest.TestCase):
    """This is a class with functions to test the FlapDrive record types."""

    def test_inputs_fields(self) -> None:
        """Test that FlapDriveInputs has the documented fields in order."""
        self.assertEqual(
            flap_drive.FlapDriveInputs._fields,
            (
                "time",
                "actuator_position",
                "actuator_velocity",
                "angleXRad_Gs_to_Wn_ixyz",
                "angleDerivativeXRad_Gs_to_Wn_ixyz",
                "jacobian",
                "hessian",
                "wing_flap_inertia",
                "desired_actuator_position",
                "desired_actuator_velocity",
                "desired_actuator_acceleration",
                "feedforward_force",
                "feedforward_inertial_force",
                "feedforward_damping_force",
                "feedforward_passive_force",
                "feedforward_aerodynamic_force",
                "feedforward_fluid_torque",
            ),
        )

    def test_accelerations_fields(self) -> None:
        """Test that FlapDriveAccelerations has the documented fields in order."""
        self.assertEqual(
            flap_drive.FlapDriveAccelerations._fields,
            (
                "actuator_acceleration",
                "angleSecondDerivativeXRad_Gs_to_Wn_ixyz",
                "actuator_force",
                "transmission_torque",
            ),
        )


class TestFlapDrive(unittest.TestCase):
    """This is a class with functions to test FlapDrives."""

    def setUp(self) -> None:
        """Set up test fixtures for FlapDrive tests."""
        self.basic_drive = flap_drive_fixtures.make_basic_flap_drive_fixture()
        self.nonlinear_drive = flap_drive_fixtures.make_nonlinear_flap_drive_fixture()

    def _make_drive(self, **overrides: Any) -> flap_drive.FlapDrive:
        """Builds a FlapDrive from valid defaults with some arguments replaced.

        :param overrides: The constructor arguments to replace.
        :return: The FlapDrive.
        """
        arguments: dict[str, Any] = {
            "wing_indices": (0,),
            "use_position_lookup": False,
            "transmission": flap_drive.FlapTransmission.identity(),
            "actuator": flap_drive_fixtures.constant_actuator,
            "actuator_inertia": 0.02,
            "reference": flap_drive_fixtures.make_hold_reference_fixture(),
        }
        arguments.update(overrides)
        return flap_drive.FlapDrive(**arguments)

    # --- Initialization and properties ---
    def test_initialization_stores_parameters(self) -> None:
        """Test that the constructor stores every parameter and applies defaults."""
        self.assertEqual(self.basic_drive.wing_indices, (0, 1))
        self.assertEqual(self.basic_drive.actuator_inertia, 0.02)
        self.assertEqual(self.basic_drive.actuator_damping, 0.0)
        self.assertIsNone(self.basic_drive.initial_actuator_position)
        self.assertIsNone(self.basic_drive.initial_actuator_velocity)
        self.assertIsInstance(self.basic_drive.reference, flap_drive.FlapReference)
        self.assertIsNone(self.basic_drive.passive_wing_torque_function)
        self.assertIsNone(self.basic_drive.passive_actuator_force_function)
        self.assertIsNone(self.basic_drive.wing_flap_inertia)
        self.assertIs(self.basic_drive.actuator, flap_drive_fixtures.constant_actuator)

        self.assertEqual(self.nonlinear_drive.actuator_damping, 0.05)
        self.assertEqual(self.nonlinear_drive.initial_actuator_position, 0.3)
        self.assertEqual(self.nonlinear_drive.initial_actuator_velocity, 0.4)
        self.assertIsNotNone(self.nonlinear_drive.passive_wing_torque_function)
        self.assertIsNotNone(self.nonlinear_drive.passive_actuator_force_function)

    def test_values_are_converted_to_floats(self) -> None:
        """Test that integer inputs are stored as floats."""
        drive = self._make_drive(
            actuator_inertia=1,
            actuator_damping=2,
            initial_actuator_position=3,
            initial_actuator_velocity=4,
            wing_flap_inertia=5,
        )
        for value in (
            drive.actuator_inertia,
            drive.actuator_damping,
            drive.initial_actuator_position,
            drive.initial_actuator_velocity,
            drive.wing_flap_inertia,
        ):
            self.assertIsInstance(value, float)

    def test_wing_indices_accepts_sequences_and_arrays(self) -> None:
        """Test that lists, tuples, and ndarrays all become a tuple of ints."""
        for indices in ([0, 2], (0, 2), np.array([0, 2])):
            drive = self._make_drive(wing_indices=indices)
            self.assertEqual(drive.wing_indices, (0, 2))
            for index in drive.wing_indices:
                self.assertIs(type(index), int)

    def test_wing_indices_validation(self) -> None:
        """Test that invalid wing_indices are rejected."""
        invalid_types: list[Any] = ["01", b"01", 3, None, {0, 1}]
        for value in invalid_types:
            with self.assertRaises(TypeError):
                self._make_drive(wing_indices=value)

        with self.assertRaises(ValueError):
            self._make_drive(wing_indices=())
        with self.assertRaises(ValueError):
            self._make_drive(wing_indices=(0, 1, 0))
        with self.assertRaises(ValueError):
            self._make_drive(wing_indices=(0, -1))
        with self.assertRaises(TypeError):
            self._make_drive(wing_indices=(0, 1.0))

    def test_transmission_and_actuator_validation(self) -> None:
        """Test that a wrong typed transmission or actuator is rejected."""
        not_a_transmission: Any = lambda position: position
        with self.assertRaises(TypeError):
            self._make_drive(transmission=not_a_transmission)
        not_callable: Any = 1.0
        with self.assertRaises(TypeError):
            self._make_drive(actuator=not_callable)

    def test_numeric_parameter_validation(self) -> None:
        """Test the ranges of the numeric parameters."""
        self._make_drive(actuator_inertia=0.0)
        self._make_drive(actuator_damping=0.0)
        with self.assertRaises(ValueError):
            self._make_drive(actuator_inertia=-0.1)
        with self.assertRaises(ValueError):
            self._make_drive(actuator_damping=-0.1)
        with self.assertRaises(ValueError):
            self._make_drive(wing_flap_inertia=0.0)
        with self.assertRaises(ValueError):
            self._make_drive(wing_flap_inertia=-1.0)
        for name in (
            "actuator_inertia",
            "actuator_damping",
            "initial_actuator_position",
            "initial_actuator_velocity",
            "wing_flap_inertia",
        ):
            with self.assertRaises(TypeError):
                self._make_drive(**{name: "1.0"})
            with self.assertRaises(ValueError):
                self._make_drive(**{name: np.nan})

    def test_passive_callable_validation(self) -> None:
        """Test that the passive elements must be callables or None."""
        with self.assertRaises(TypeError):
            self._make_drive(passive_wing_torque_function=1.0)
        with self.assertRaises(TypeError):
            self._make_drive(passive_actuator_force_function=1.0)

    def test_properties_are_read_only(self) -> None:
        """Test that no property can be reassigned."""
        for name in (
            "wing_indices",
            "transmission",
            "actuator",
            "actuator_inertia",
            "actuator_damping",
            "initial_actuator_position",
            "initial_actuator_velocity",
            "passive_wing_torque_function",
            "passive_actuator_force_function",
            "wing_flap_inertia",
            "reference",
        ):
            with self.assertRaises(AttributeError):
                setattr(self.basic_drive, name, None)

    def test_slots(self) -> None:
        """Test that the class has no instance dict and rejects new attributes."""
        self.assertFalse(hasattr(self.basic_drive, "__dict__"))
        with self.assertRaises(AttributeError):
            setattr(self.basic_drive, "new_attribute", 1.0)

    # --- Wing inertia ---
    def test_inverse_dynamics_controller_picks_the_wing_flap_inertia(self) -> None:
        """Test that inverse_dynamics_controller uses a drive's measured inertia over
        the caller's estimate, and the estimate when there is no measured inertia."""
        estimated = self.basic_drive.inverse_dynamics_controller(
            0.0, 0.0, 0.0, 0.0, 0.4
        )
        self.assertEqual(estimated.wing_flap_inertia, 0.4)
        measured = self._make_drive(wing_flap_inertia=0.08)
        self.assertEqual(
            measured.inverse_dynamics_controller(
                0.0, 0.0, 0.0, 0.0, 0.4
            ).wing_flap_inertia,
            0.08,
        )
        self.assertEqual(
            measured.inverse_dynamics_controller(0.0, 0.0, 0.0, 0.0).wing_flap_inertia,
            0.08,
        )

    def test_missing_or_invalid_wing_flap_inertia_raises(self) -> None:
        """Test that inverse_dynamics_controller requires some positive source of Wing
        inertia."""
        with self.assertRaises(ValueError):
            self.basic_drive.inverse_dynamics_controller(0.0, 0.0, 0.0, 0.0)
        for bad_inertia in (0.0, -1.0):
            with self.assertRaises(ValueError):
                self.basic_drive.inverse_dynamics_controller(
                    0.0, 0.0, 0.0, 0.0, bad_inertia
                )

    # --- inverse_dynamics_controller ---
    def test_inverse_dynamics_controller(self) -> None:
        """Test that inverse_dynamics_controller evaluates the transmission and copies
        the state."""
        position, velocity = 1.3, 2.1
        inputs = self.nonlinear_drive.inverse_dynamics_controller(
            time=0.5,
            actuator_position=position,
            actuator_velocity=velocity,
            measured_fluid_torque=0.25,
            computed_wing_flap_inertia=WING_INERTIA,
        )
        jacobian = flap_drive_fixtures.nonlinear_jacobian(position)
        self.assertEqual(inputs.time, 0.5)
        self.assertEqual(inputs.actuator_position, position)
        self.assertEqual(inputs.actuator_velocity, velocity)
        self.assertAlmostEqual(
            inputs.angleXRad_Gs_to_Wn_ixyz,
            flap_drive_fixtures.nonlinear_angle(position),
        )
        self.assertAlmostEqual(
            inputs.angleDerivativeXRad_Gs_to_Wn_ixyz, jacobian * velocity
        )
        self.assertAlmostEqual(inputs.jacobian, jacobian)
        self.assertAlmostEqual(
            inputs.hessian, flap_drive_fixtures.nonlinear_hessian(position)
        )
        self.assertEqual(inputs.feedforward_fluid_torque, 0.25)
        self.assertEqual(inputs.wing_flap_inertia, WING_INERTIA)

    def test_reference_is_required(self) -> None:
        """Test that a FlapDrive can't be built without a FlapReference, and that the
        reference can't be passed by position."""
        arguments: dict[str, Any] = {
            "wing_indices": (0,),
            "use_position_lookup": False,
            "transmission": flap_drive.FlapTransmission.identity(),
            "actuator": flap_drive_fixtures.constant_actuator,
            "actuator_inertia": 0.02,
        }
        with self.assertRaises(TypeError):
            flap_drive.FlapDrive(**arguments)
        not_a_reference: Any = None
        with self.assertRaises(TypeError):
            flap_drive.FlapDrive(**arguments, reference=not_a_reference)
        positional: Any = flap_drive.FlapDrive
        with self.assertRaises(TypeError):
            positional(
                (0,),
                flap_drive.FlapTransmission.identity(),
                flap_drive_fixtures.constant_actuator,
                0.02,
                0.0,
                None,
                None,
                None,
                None,
                None,
                flap_drive_fixtures.make_hold_reference_fixture(),
                use_position_lookup=False,
            )

    def test_inverse_dynamics_controller_returns_floats(self) -> None:
        """Test that integer arguments are converted to floats."""
        inputs = self.basic_drive.inverse_dynamics_controller(1, 2, 3, 4, 5)
        for value in inputs:
            self.assertIsInstance(value, float)

    def test_inverse_dynamics_controller_fills_desired_state_from_reference(
        self,
    ) -> None:
        """Test that the desired state is read from the reference at the time."""
        times = np.linspace(0.0, 1.0, 11)
        reference = flap_drive.FlapReference(
            times, 2.0 * times, np.full(11, 2.0), np.zeros(11), period=None
        )
        drive = self._make_drive(reference=reference)
        inputs = drive.inverse_dynamics_controller(0.5, 0.0, 0.0, 0.0, WING_INERTIA)
        self.assertAlmostEqual(inputs.desired_actuator_position, 1.0)
        self.assertAlmostEqual(inputs.desired_actuator_velocity, 2.0)
        self.assertAlmostEqual(inputs.desired_actuator_acceleration, 0.0)

    # --- single_wing_dynamics ---
    def _nonlinear_state(
        self,
    ) -> tuple[float, float, float, flap_drive.FlapDriveInputs]:
        """Builds a representative state for the nonlinear drive.

        :return: The actuator position, actuator velocity, fluid torque, and the
            FlapDriveInputs for that state.
        """
        position, velocity, fluid_torque = 1.3, 2.1, 0.25
        inputs = self.nonlinear_drive.inverse_dynamics_controller(
            0.0, position, velocity, fluid_torque, WING_INERTIA
        )
        return position, velocity, fluid_torque, inputs

    def test_actuator_side_balance(self) -> None:
        """Test virtual work: the actuator force balances inertia, damping, the passive
        force, and the torque the transmission carries to the Wings."""
        position, velocity, fluid_torque, inputs = self._nonlinear_state()
        result = self.nonlinear_drive.single_wing_dynamics(inputs, fluid_torque)

        # The actuator force, the damping, and the passive actuator force of 0.2, which
        # is applied to the actuator in the direction of increasing position.
        applied = 0.7 - 0.05 * velocity + 0.2
        absorbed = (
            0.02 * result.actuator_acceleration
            + flap_drive_fixtures.nonlinear_jacobian(position)
            * result.transmission_torque
        )
        self.assertAlmostEqual(applied, absorbed, places=12)

    def test_wing_side_balance(self) -> None:
        """Test that the Wings obey I * phi'' = transmission + fluid + passive."""
        _, _, fluid_torque, inputs = self._nonlinear_state()
        result = self.nonlinear_drive.single_wing_dynamics(inputs, fluid_torque)

        self.assertAlmostEqual(
            WING_INERTIA * result.angleSecondDerivativeXRad_Gs_to_Wn_ixyz,
            result.transmission_torque + fluid_torque - 0.03,
            places=12,
        )

    def test_flap_acceleration_matches_finite_difference_of_trajectory(self) -> None:
        """Test phi'' against a finite difference of the flap angle along the trajectory
        that the actuator acceleration produces."""
        position, velocity, fluid_torque, inputs = self._nonlinear_state()
        result = self.nonlinear_drive.single_wing_dynamics(inputs, fluid_torque)

        def position_at(time: float) -> float:
            return (
                position
                + velocity * time
                + 0.5 * result.actuator_acceleration * time**2
            )

        step = 1e-4
        numerical = (
            flap_drive_fixtures.nonlinear_angle(position_at(step))
            - 2.0 * flap_drive_fixtures.nonlinear_angle(position_at(0.0))
            + flap_drive_fixtures.nonlinear_angle(position_at(-step))
        ) / step**2
        self.assertAlmostEqual(
            result.angleSecondDerivativeXRad_Gs_to_Wn_ixyz, numerical, places=5
        )

    def test_direct_drive_reduces_to_rigid_body(self) -> None:
        """Test that an identity transmission gives (I_a + I_w) * q'' = F + T."""
        actuator_inertia = 0.02
        fluid_torque = 0.15
        drive = self._make_drive(
            actuator=lambda inputs: 0.7, actuator_inertia=actuator_inertia
        )
        inputs = drive.inverse_dynamics_controller(
            0.0, 0.2, 0.6, fluid_torque, WING_INERTIA
        )
        result = drive.single_wing_dynamics(inputs, fluid_torque)

        expected = (0.7 + fluid_torque) / (actuator_inertia + WING_INERTIA)
        self.assertAlmostEqual(result.actuator_acceleration, expected, places=12)
        self.assertAlmostEqual(
            result.angleSecondDerivativeXRad_Gs_to_Wn_ixyz, expected, places=12
        )

    def test_geared_drive_reflects_inertia(self) -> None:
        """Test that a gearbox reflects the Wings' inertia by the square of its
        Jacobian."""
        ratio = 5.0
        actuator_inertia = 0.01
        drive = self._make_drive(
            transmission=flap_drive.FlapTransmission.geared(ratio),
            actuator=lambda inputs: 0.3,
            actuator_inertia=actuator_inertia,
        )
        inputs = drive.inverse_dynamics_controller(0.0, 1.0, 0.5, 0.0, WING_INERTIA)
        result = drive.single_wing_dynamics(inputs, 0.0)

        expected = 0.3 / (actuator_inertia + WING_INERTIA / ratio**2)
        self.assertAlmostEqual(result.actuator_acceleration, expected, places=12)
        self.assertAlmostEqual(
            result.angleSecondDerivativeXRad_Gs_to_Wn_ixyz, expected / ratio, places=12
        )

    def test_damping_opposes_velocity(self) -> None:
        """Test that damping reduces the acceleration of a moving actuator."""
        undamped = self._make_drive(actuator=lambda inputs: 0.0)
        damped = self._make_drive(actuator=lambda inputs: 0.0, actuator_damping=0.5)
        undamped_inputs = undamped.inverse_dynamics_controller(
            0.0, 0.0, 2.0, 0.0, WING_INERTIA
        )
        damped_inputs = damped.inverse_dynamics_controller(
            0.0, 0.0, 2.0, 0.0, WING_INERTIA
        )
        self.assertEqual(
            undamped.single_wing_dynamics(undamped_inputs, 0.0).actuator_acceleration,
            0.0,
        )
        self.assertLess(
            damped.single_wing_dynamics(damped_inputs, 0.0).actuator_acceleration, 0.0
        )

    def test_fluid_torque_sign(self) -> None:
        """Test that a positive fluid torque accelerates the flap angle upward."""
        drive = self._make_drive(actuator=lambda inputs: 0.0)
        inputs = drive.inverse_dynamics_controller(0.0, 0.0, 0.0, 1.0, WING_INERTIA)
        result = drive.single_wing_dynamics(inputs, 1.0)
        self.assertGreater(result.angleSecondDerivativeXRad_Gs_to_Wn_ixyz, 0.0)
        # The actuator's own inertia absorbs part of the fluid torque, so the
        # transmission must carry the opposing share: -I_a / (I_a + I_w) * T_fluid.
        self.assertAlmostEqual(
            result.transmission_torque, -0.02 / (0.02 + WING_INERTIA), places=12
        )

    def test_wing_flap_inertia_override_takes_precedence(self) -> None:
        """Test that the drive's own wing_flap_inertia replaces the computed one."""
        drive = self._make_drive(
            actuator=lambda inputs: 1.0, wing_flap_inertia=0.08, actuator_inertia=0.02
        )
        inputs = drive.inverse_dynamics_controller(0.0, 0.0, 0.0, 0.0, 999.0)
        self.assertEqual(inputs.wing_flap_inertia, 0.08)
        result = drive.single_wing_dynamics(inputs, 0.0)
        self.assertAlmostEqual(result.actuator_acceleration, 1.0 / 0.10, places=12)

    def test_singular_jacobian_without_inertia_raises(self) -> None:
        """Test that zero effective inertia is reported instead of dividing by zero."""
        drive = flap_drive_fixtures.make_singular_flap_drive_fixture(0.0)
        inputs = drive.inverse_dynamics_controller(0.0, 0.0, 0.0, 0.0, WING_INERTIA)
        with self.assertRaises(ValueError):
            drive.single_wing_dynamics(inputs, 0.0)

    def test_singular_jacobian_with_inertia_stays_finite(self) -> None:
        """Test that the Wing side torque stays finite where the Jacobian is zero."""
        drive = flap_drive_fixtures.make_singular_flap_drive_fixture(0.01)
        inputs = drive.inverse_dynamics_controller(0.0, 0.0, 0.0, 0.3, WING_INERTIA)
        result = drive.single_wing_dynamics(inputs, 0.3)
        npt.assert_allclose(
            [
                result.actuator_acceleration,
                result.angleSecondDerivativeXRad_Gs_to_Wn_ixyz,
                result.transmission_torque,
            ],
            [100.0, 0.0, -0.3],
        )

    def test_hessian_term_changes_acceleration(self) -> None:
        """Test that the centripetal term J * I_w * H * q'^2 enters the equation."""
        hessian_free = flap_drive.FlapTransmission(
            flap_drive_fixtures.nonlinear_angle,
            flap_drive_fixtures.nonlinear_jacobian,
            lambda position: 0.0,
            flap_drive_fixtures.unused_trajectory,
        )
        with_hessian = self._make_drive(
            transmission=flap_drive_fixtures.make_nonlinear_transmission_fixture(),
            actuator=lambda inputs: 0.0,
        )
        without_hessian = self._make_drive(
            transmission=hessian_free, actuator=lambda inputs: 0.0
        )
        position, velocity = 1.3, 3.0
        a = with_hessian.single_wing_dynamics(
            with_hessian.inverse_dynamics_controller(
                0.0, position, velocity, 0.0, WING_INERTIA
            ),
            0.0,
        )
        b = without_hessian.single_wing_dynamics(
            without_hessian.inverse_dynamics_controller(
                0.0, position, velocity, 0.0, WING_INERTIA
            ),
            0.0,
        )
        jacobian = flap_drive_fixtures.nonlinear_jacobian(position)
        hessian = flap_drive_fixtures.nonlinear_hessian(position)
        expected = (
            -jacobian
            * WING_INERTIA
            * hessian
            * velocity**2
            / (0.02 + jacobian**2 * WING_INERTIA)
        )
        self.assertAlmostEqual(a.actuator_acceleration, expected, places=12)
        self.assertEqual(b.actuator_acceleration, 0.0)

    def test_actuator_receives_the_inputs_record(self) -> None:
        """Test that the actuator function is called with the same FlapDriveInputs."""
        received: list[flap_drive.FlapDriveInputs] = []

        def spy(inputs: flap_drive.FlapDriveInputs) -> float:
            received.append(inputs)
            return 0.0

        drive = self._make_drive(actuator=spy)
        inputs = drive.inverse_dynamics_controller(0.25, 0.1, 0.2, 0.3, WING_INERTIA)
        drive.single_wing_dynamics(inputs, 0.3)
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0], inputs)

    def test_passive_callables_receive_the_right_arguments(self) -> None:
        """Test that the passive callables get the angle and actuator state in the
        equation of motion.

        The passive feedforward is off, so that its calls at the desired velocity don't
        mix with these.
        """
        wing_calls: list[tuple[float, float]] = []
        actuator_calls: list[tuple[float, float]] = []

        def passive_wing(angle_rad: float, angle_derivative_rad: float) -> float:
            wing_calls.append((angle_rad, angle_derivative_rad))
            return 0.0

        def passive_actuator(position: float, velocity: float) -> float:
            actuator_calls.append((position, velocity))
            return 0.0

        drive = self._make_drive(
            transmission=flap_drive_fixtures.make_nonlinear_transmission_fixture(),
            passive_wing_torque_function=passive_wing,
            passive_actuator_force_function=passive_actuator,
            feedforward_passive=False,
        )
        position, velocity = 0.8, 1.5
        inputs = drive.inverse_dynamics_controller(
            0.0, position, velocity, 0.0, WING_INERTIA
        )
        drive.single_wing_dynamics(inputs, 0.0)

        self.assertEqual(actuator_calls, [(position, velocity)])
        self.assertEqual(len(wing_calls), 1)
        self.assertAlmostEqual(
            wing_calls[0][0], flap_drive_fixtures.nonlinear_angle(position)
        )
        self.assertAlmostEqual(
            wing_calls[0][1],
            flap_drive_fixtures.nonlinear_jacobian(position) * velocity,
        )

    def test_passive_springs_restore(self) -> None:
        """Test that a spring written with the applied convention, -k * (x - x_0), pulls
        the actuator back toward its offset from either side, for both the actuator
        spring and the Wing spring."""
        stiffness = 4.0
        for transmission, passive in (
            (
                flap_drive.FlapTransmission.identity(),
                {"passive_actuator_force_function": lambda q, q_dot: -stiffness * q},
            ),
            (
                flap_drive.FlapTransmission.geared(3.0),
                {"passive_wing_torque_function": lambda phi, phi_dot: -stiffness * phi},
            ),
        ):
            drive = self._make_drive(
                transmission=transmission, actuator=lambda inputs: 0.0, **passive
            )
            for position in (0.5, -0.5):
                inputs = drive.inverse_dynamics_controller(
                    0.0, position, 0.0, 0.0, WING_INERTIA
                )
                acceleration = drive.single_wing_dynamics(
                    inputs, 0.0
                ).actuator_acceleration
                self.assertLess(acceleration * position, 0.0)

    def test_non_finite_user_outputs_raise(self) -> None:
        """Test that nan or inf from any user callable is reported."""
        inputs = self.basic_drive.inverse_dynamics_controller(
            0.0, 0.0, 0.0, 0.0, WING_INERTIA
        )
        for bad_value in (np.nan, np.inf):
            drives = (
                self._make_drive(actuator=lambda inputs, v=bad_value: v),
                self._make_drive(
                    passive_wing_torque_function=lambda a, d, v=bad_value: v
                ),
                self._make_drive(
                    passive_actuator_force_function=lambda q, qd, v=bad_value: v
                ),
            )
            for drive in drives:
                with self.assertRaises(ValueError):
                    drive.single_wing_dynamics(inputs, 0.0)

    # --- Feedforward (inverse dynamics) ---
    def _feedforward_drive_and_reference(
        self,
    ) -> tuple[flap_drive.FlapDrive, flap_drive.FlapReference]:
        """Builds a geared drive with every optional term, and a rotating reference.

        :return: The FlapDrive and the FlapReference it follows.
        """
        times = np.linspace(0.0, 2.0, 201)
        positions = 5.0 * times + 0.5 * times**2
        reference = flap_drive.FlapReference(
            times, positions, 5.0 + times, np.ones_like(times), period=None
        )
        drive = self._make_drive(
            transmission=flap_drive.FlapTransmission.geared(8.0),
            actuator=lambda inputs: 0.0,
            actuator_inertia=0.01,
            actuator_damping=0.03,
            passive_wing_torque_function=lambda angle, rate: -0.5 * angle - 0.01 * rate,
            passive_actuator_force_function=lambda q, qd: 0.02 * q,
            reference=reference,
            use_position_lookup=True,
        )
        return drive, reference

    def test_feedforward_reproduces_the_reference_acceleration(self) -> None:
        """Test inverse dynamics against forward dynamics: applying the feedforward
        force on the reference trajectory gives exactly the desired acceleration."""
        drive, reference = self._feedforward_drive_and_reference()
        for time in (0.2, 0.9, 1.7):
            position, velocity, acceleration = reference.evaluate(time)
            fluid_torque = 0.37
            inputs = drive.inverse_dynamics_controller(
                time, position, velocity, fluid_torque, WING_INERTIA
            )
            self.assertNotEqual(inputs.feedforward_force, 0.0)
            applied = inputs._replace(feedforward_force=inputs.feedforward_force)
            driven = flap_drive.FlapDrive(
                wing_indices=drive.wing_indices,
                transmission=drive.transmission,
                actuator=lambda i: i.feedforward_force,
                actuator_inertia=drive.actuator_inertia,
                actuator_damping=drive.actuator_damping,
                passive_wing_torque_function=drive.passive_wing_torque_function,
                passive_actuator_force_function=drive.passive_actuator_force_function,
                reference=reference,
                use_position_lookup=False,
            )
            result = driven.single_wing_dynamics(applied, fluid_torque)
            self.assertAlmostEqual(result.actuator_acceleration, acceleration, places=9)

    def test_feedforward_uses_the_position_lookup_when_behind(self) -> None:
        """Test that a lagging actuator gets the feedforward for where it actually is in
        its stroke, not for where the schedule says it should be."""
        drive, reference = self._feedforward_drive_and_reference()
        self.assertTrue(drive.use_position_lookup)
        time, lagging_position = 1.0, 3.0
        lagging = drive.inverse_dynamics_controller(
            time, lagging_position, 6.0, 0.0, WING_INERTIA
        )
        by_time = self._make_drive(
            transmission=drive.transmission,
            actuator=drive.actuator,
            actuator_inertia=drive.actuator_inertia,
            actuator_damping=drive.actuator_damping,
            passive_wing_torque_function=drive.passive_wing_torque_function,
            passive_actuator_force_function=drive.passive_actuator_force_function,
            reference=reference,
            use_position_lookup=False,
        )
        self.assertFalse(by_time.use_position_lookup)
        scheduled = by_time.inverse_dynamics_controller(
            time, lagging_position, 6.0, 0.0, WING_INERTIA
        )
        self.assertNotAlmostEqual(
            lagging.feedforward_force, scheduled.feedforward_force
        )
        # The desired state itself is always read by time.
        self.assertEqual(
            lagging.desired_actuator_position, scheduled.desired_actuator_position
        )

    def test_position_lookup_must_be_given(self) -> None:
        """Test that use_position_lookup has no default and can't be passed by position,
        so every FlapDrive states its choice."""
        with self.assertRaises(TypeError):
            flap_drive.FlapDrive(  # type: ignore[call-arg]
                wing_indices=(0,),
                transmission=flap_drive.FlapTransmission.identity(),
                actuator=flap_drive_fixtures.constant_actuator,
                actuator_inertia=0.02,
                reference=flap_drive_fixtures.make_hold_reference_fixture(),
            )
        positional: Any = flap_drive.FlapDrive
        with self.assertRaises(TypeError):
            positional(
                (0,),
                flap_drive.FlapTransmission.identity(),
                flap_drive_fixtures.constant_actuator,
                0.02,
                0.0,
                None,
                None,
                None,
                None,
                None,
                None,
                False,
            )

    def test_position_lookup_choice_is_validated(self) -> None:
        """Test that the position lookup is off unless asked for, even for a monotonic
        reference, that it can't be turned on for a reference that moves back and forth,
        and that a bad flag is rejected."""
        times = np.linspace(0.0, 1.0, 11)
        back_and_forth = flap_drive.FlapReference(
            times, np.sin(2.0 * np.pi * times), np.ones(11), np.zeros(11), period=None
        )
        with self.assertRaises(ValueError):
            self._make_drive(reference=back_and_forth, use_position_lookup=True)
        default = self._make_drive(reference=back_and_forth)
        self.assertFalse(default.use_position_lookup)
        monotonic = flap_drive.FlapReference(
            times, times, np.ones(11), np.zeros(11), period=None
        )
        self.assertFalse(self._make_drive(reference=monotonic).use_position_lookup)
        bad_flag: Any = "yes"
        with self.assertRaises(TypeError):
            self._make_drive(use_position_lookup=bad_flag)

    def test_feedforward_stays_finite_at_a_singular_jacobian(self) -> None:
        """Test that the inverse dynamics need no division by the Jacobian."""
        times = np.linspace(0.0, 1.0, 11)
        reference = flap_drive.FlapReference(
            times, times, np.ones(11), np.zeros(11), period=None
        )
        drive = flap_drive_fixtures.make_singular_flap_drive_fixture(0.01, reference)
        inputs = drive.inverse_dynamics_controller(0.0, 0.0, 0.0, 0.3, WING_INERTIA)
        self.assertEqual(inputs.jacobian, 0.0)
        self.assertTrue(np.isfinite(inputs.feedforward_force))


class TestEquationOfMotionTerms(unittest.TestCase):
    """This is a class with functions to test FlapDrive._equation_of_motion_terms term by term.

    Each test turns on one term of the equation of motion at a time and compares the
    effective inertia and the bias force with the term's hand formula,

    (I_a + J^2 * I_w) * q'' = F_a - c * q' + F_p + J * (T_fluid + T_p) - J * I_w * H *
    q'^2

    where the bias is everything on the right side except F_a.
    """

    # The state and parameters every test uses, chosen so that no two terms are equal.
    POSITION = 0.7
    VELOCITY = 2.5
    JACOBIAN = 0.3
    HESSIAN = -0.8
    ANGLE_RAD = 0.2
    ACTUATOR_INERTIA = 0.05
    WING_INERTIA = 1.2

    def _terms(self, **overrides: Any) -> tuple[float, float, float]:
        """Builds a FlapDrive with every optional term off, except those given, and
        evaluates its terms at the shared state.

        :param overrides: The FlapDrive arguments to set, plus "fluid_torque",
            "jacobian", "hessian", and "velocity" to replace the shared state.
        :return: The effective inertia, the bias force, and the passive Wing torque.
        """
        fluid_torque = overrides.pop("fluid_torque", 0.0)
        jacobian = overrides.pop("jacobian", self.JACOBIAN)
        hessian = overrides.pop("hessian", self.HESSIAN)
        velocity = overrides.pop("velocity", self.VELOCITY)
        arguments: dict[str, Any] = {
            "wing_indices": (0,),
            "use_position_lookup": False,
            "transmission": flap_drive.FlapTransmission.identity(),
            "actuator": lambda inputs: 0.0,
            "actuator_inertia": self.ACTUATOR_INERTIA,
            "reference": flap_drive_fixtures.make_hold_reference_fixture(),
        }
        arguments.update(overrides)
        drive = flap_drive.FlapDrive(**arguments)
        return drive._equation_of_motion_terms(
            self.POSITION,
            velocity,
            jacobian,
            hessian,
            self.ANGLE_RAD,
            fluid_torque,
            self.WING_INERTIA,
        )

    def test_effective_inertia(self) -> None:
        """Test that the effective inertia is the actuator's plus the reflected Wing
        inertia, J^2 * I_w, and doesn't depend on the velocity or any force."""
        expected = self.ACTUATOR_INERTIA + self.JACOBIAN**2 * self.WING_INERTIA
        for velocity in (0.0, 2.5, -7.0):
            effective_inertia, _, _ = self._terms(velocity=velocity, fluid_torque=3.0)
            self.assertAlmostEqual(effective_inertia, expected, places=14)

    def test_centripetal_term_only(self) -> None:
        """Test the bias with no forces: only -J * I_w * H * q'^2 remains."""
        _, bias, _ = self._terms()
        expected = -self.JACOBIAN * self.WING_INERTIA * self.HESSIAN * self.VELOCITY**2
        self.assertAlmostEqual(bias, expected, places=14)

    def test_centripetal_term_vanishes_without_hessian_or_velocity(self) -> None:
        """Test that a linear transmission (H = 0) or a stopped actuator has no
        centripetal term, and that the term is even in the velocity."""
        self.assertEqual(self._terms(hessian=0.0)[1], 0.0)
        self.assertEqual(self._terms(velocity=0.0)[1], 0.0)
        self.assertAlmostEqual(
            self._terms(velocity=2.5)[1], self._terms(velocity=-2.5)[1], places=14
        )

    def test_damping_term(self) -> None:
        """Test that damping adds -c * q' and opposes the velocity in either
        direction."""
        damping = 0.4
        for velocity in (2.5, -2.5):
            centripetal = self._terms(velocity=velocity)[1]
            _, bias, _ = self._terms(actuator_damping=damping, velocity=velocity)
            self.assertAlmostEqual(bias - centripetal, -damping * velocity, places=14)
            self.assertLess((bias - centripetal) * velocity, 0.0)

    def test_passive_actuator_force_term(self) -> None:
        """Test that the passive actuator force is added as applied, +F_p, with the
        actuator position and velocity as its arguments."""
        received: list[tuple[float, float]] = []

        def spring(position: float, velocity: float) -> float:
            received.append((position, velocity))
            return -3.0 * position

        centripetal = self._terms()[1]
        _, bias, _ = self._terms(passive_actuator_force_function=spring)
        self.assertAlmostEqual(bias - centripetal, -3.0 * self.POSITION, places=14)
        self.assertEqual(received, [(self.POSITION, self.VELOCITY)])

    def test_fluid_torque_term(self) -> None:
        """Test that the fluid torque is carried to the actuator as +J * T_fluid, so it
        helps the actuator where J is positive and resists it where J is negative."""
        centripetal = self._terms()[1]
        _, bias, _ = self._terms(fluid_torque=2.0)
        self.assertAlmostEqual(bias - centripetal, self.JACOBIAN * 2.0, places=14)
        flipped_centripetal = self._terms(jacobian=-self.JACOBIAN)[1]
        _, flipped, _ = self._terms(jacobian=-self.JACOBIAN, fluid_torque=2.0)
        self.assertAlmostEqual(
            flipped - flipped_centripetal, -self.JACOBIAN * 2.0, places=14
        )

    def test_passive_wing_torque_term(self) -> None:
        """Test that the passive Wing torque is carried to the actuator as +J * T_p, is
        evaluated at the flap angle and the flap angle's rate J * q', and is returned as
        the third value."""
        received: list[tuple[float, float]] = []

        def wing_spring(angle: float, rate: float) -> float:
            received.append((angle, rate))
            return -5.0 * angle - 0.1 * rate

        centripetal = self._terms()[1]
        _, bias, passive = self._terms(passive_wing_torque_function=wing_spring)
        expected_torque = -5.0 * self.ANGLE_RAD - 0.1 * self.JACOBIAN * self.VELOCITY
        self.assertAlmostEqual(passive, expected_torque, places=14)
        self.assertAlmostEqual(
            bias - centripetal, self.JACOBIAN * expected_torque, places=14
        )
        self.assertEqual(len(received), 1)
        self.assertAlmostEqual(received[0][0], self.ANGLE_RAD)
        self.assertAlmostEqual(received[0][1], self.JACOBIAN * self.VELOCITY)

    def test_passive_wing_torque_is_zero_without_a_function(self) -> None:
        """Test that the returned passive Wing torque is 0.0 with no function."""
        self.assertEqual(self._terms()[2], 0.0)

    def test_all_terms_superpose(self) -> None:
        """Test that with every term on, the bias is the sum of the separate terms."""
        damping = 0.4
        fluid_torque = 2.0

        def actuator_spring(position: float, velocity: float) -> float:
            return -3.0 * position

        def wing_spring(angle: float, rate: float) -> float:
            return -5.0 * angle

        _, bias, _ = self._terms(
            actuator_damping=damping,
            passive_actuator_force_function=actuator_spring,
            passive_wing_torque_function=wing_spring,
            fluid_torque=fluid_torque,
        )
        expected = (
            -damping * self.VELOCITY
            + (-3.0 * self.POSITION)
            + self.JACOBIAN * (fluid_torque + (-5.0 * self.ANGLE_RAD))
            - self.JACOBIAN * self.WING_INERTIA * self.HESSIAN * self.VELOCITY**2
        )
        self.assertAlmostEqual(bias, expected, places=13)

    def test_singular_jacobian_drops_every_wing_term(self) -> None:
        """Test that at J = 0 the Wing's fluid torque, passive torque, reflected
        inertia, and centripetal term all vanish, leaving the actuator alone."""
        effective_inertia, bias, _ = self._terms(
            jacobian=0.0,
            fluid_torque=2.0,
            passive_wing_torque_function=lambda angle, rate: -5.0 * angle,
            actuator_damping=0.4,
        )
        self.assertEqual(effective_inertia, self.ACTUATOR_INERTIA)
        self.assertAlmostEqual(bias, -0.4 * self.VELOCITY, places=14)

    def test_non_positive_effective_inertia_raises(self) -> None:
        """Test that a drive with no inertia at a singular Jacobian is rejected."""
        with self.assertRaises(ValueError):
            self._terms(actuator_inertia=0.0, jacobian=0.0)


class TestTransmissionTorque(unittest.TestCase):
    """This is a class with functions to test FlapDriveAccelerations'
    transmission_torque, the torque that the transmission applies to the Wings.

    The Wings obey I_w * phi'' = T_transmission + T_fluid + T_p, so the transmission
    supplies T_transmission = I_w * phi'' - T_fluid - T_p. Each test isolates one
    contribution, then the combination, the blade element theory code's
    wing_torques.total, and the power balance at the actuator.
    """

    WING_INERTIA = 1.5
    GEAR_RATIO = 4.0

    def _drive(self, **overrides: Any) -> flap_drive.FlapDrive:
        """Builds a geared FlapDrive with every optional term off, except those given.

        :param overrides: The FlapDrive arguments to replace.
        :return: The FlapDrive.
        """
        arguments: dict[str, Any] = {
            "wing_indices": (0,),
            "use_position_lookup": False,
            "transmission": flap_drive.FlapTransmission.geared(self.GEAR_RATIO),
            "actuator": lambda inputs: 0.0,
            "actuator_inertia": 0.02,
            "reference": flap_drive_fixtures.make_hold_reference_fixture(),
        }
        arguments.update(overrides)
        return flap_drive.FlapDrive(**arguments)

    def _holding_force(
        self,
        drive: flap_drive.FlapDrive,
        position: float,
        fluid_torque: float,
    ) -> float:
        """Returns the actuator force that keeps a stopped actuator from accelerating,
        which cancels every other force on it.

        :param drive: The FlapDrive.
        :param position: The actuator position.
        :param fluid_torque: The fluid torque on the Wings (N*m).
        :return: The actuator force.
        """
        inputs = drive.inverse_dynamics_controller(
            0.0, position, 0.0, fluid_torque, self.WING_INERTIA
        )
        _, bias, _ = drive._equation_of_motion_terms(
            position,
            0.0,
            inputs.jacobian,
            inputs.hessian,
            inputs.angleXRad_Gs_to_Wn_ixyz,
            fluid_torque,
            self.WING_INERTIA,
        )
        return -bias

    def test_inertia_only(self) -> None:
        """Test that with no fluid or passive torque the transmission supplies exactly
        I_w * phi'', positive when the Wings accelerate toward increasing flap angle."""
        for force in (3.0, -3.0):
            drive = self._drive(actuator=lambda inputs, f=force: f)
            inputs = drive.inverse_dynamics_controller(
                0.0, 0.4, 0.0, 0.0, self.WING_INERTIA
            )
            result = drive.single_wing_dynamics(inputs, 0.0)
            self.assertAlmostEqual(
                result.transmission_torque,
                self.WING_INERTIA * result.angleSecondDerivativeXRad_Gs_to_Wn_ixyz,
                places=14,
            )
            self.assertGreater(result.transmission_torque * force, 0.0)

    def test_holding_against_the_fluid(self) -> None:
        """Test that when the actuator holds the Wings still against the fluid, the
        transmission's torque exactly cancels the fluid torque."""
        drive = self._drive()
        fluid_torque = -2.5
        force = self._holding_force(drive, 0.4, fluid_torque)
        held = self._drive(actuator=lambda inputs: force)
        inputs = held.inverse_dynamics_controller(
            0.0, 0.4, 0.0, fluid_torque, self.WING_INERTIA
        )
        result = held.single_wing_dynamics(inputs, fluid_torque)
        self.assertAlmostEqual(result.actuator_acceleration, 0.0, places=12)
        self.assertAlmostEqual(result.transmission_torque, -fluid_torque, places=12)

    def test_holding_against_a_spring(self) -> None:
        """Test that when the actuator holds the Wings still against a restoring wing
        spring, displaced toward increasing flap angle, the transmission pushes toward
        increasing flap angle with exactly the spring's torque."""
        stiffness = 6.0

        def spring(angle: float, rate: float) -> float:
            return -stiffness * angle

        position = 1.2
        drive = self._drive(passive_wing_torque_function=spring)
        force = self._holding_force(drive, position, 0.0)
        held = self._drive(
            passive_wing_torque_function=spring, actuator=lambda inputs: force
        )
        inputs = held.inverse_dynamics_controller(
            0.0, position, 0.0, 0.0, self.WING_INERTIA
        )
        result = held.single_wing_dynamics(inputs, 0.0)
        angle = position / self.GEAR_RATIO
        self.assertAlmostEqual(result.actuator_acceleration, 0.0, places=12)
        self.assertAlmostEqual(result.transmission_torque, stiffness * angle, places=12)
        self.assertGreater(result.transmission_torque, 0.0)

    def test_all_contributions(self) -> None:
        """Test the full expression with inertia, fluid, and a spring and damper at the
        shoulder, while the Wings move and accelerate."""
        drive = self._drive(
            actuator=lambda inputs: 1.7,
            actuator_damping=0.05,
            passive_wing_torque_function=lambda angle, rate: -6.0 * angle - 0.3 * rate,
        )
        fluid_torque = -0.8
        inputs = drive.inverse_dynamics_controller(
            0.0, 0.9, 3.0, fluid_torque, self.WING_INERTIA
        )
        result = drive.single_wing_dynamics(inputs, fluid_torque)
        passive = -6.0 * inputs.angleXRad_Gs_to_Wn_ixyz - 0.3 * (
            inputs.angleDerivativeXRad_Gs_to_Wn_ixyz
        )
        expected = (
            self.WING_INERTIA * result.angleSecondDerivativeXRad_Gs_to_Wn_ixyz
            - fluid_torque
            - passive
        )
        self.assertAlmostEqual(result.transmission_torque, expected, places=13)

    def test_matches_the_blade_element_theory_total(self) -> None:
        """Test that transmission_torque equals the blade element theory code's
        wing_torques.total = tau_aero_wing + tau_inertial_wing + wing_spring_torque,
        built with that code's reaction torque signs."""
        stiffness, offset = 6.0, 0.05
        drive = self._drive(
            actuator=lambda inputs: 1.7,
            passive_wing_torque_function=lambda angle, rate: -stiffness
            * (angle - offset),
        )
        fluid_torque = -0.8
        inputs = drive.inverse_dynamics_controller(
            0.0, 0.9, 3.0, fluid_torque, self.WING_INERTIA
        )
        result = drive.single_wing_dynamics(inputs, fluid_torque)
        tau_aero_wing = -fluid_torque
        tau_inertial_wing = (
            self.WING_INERTIA * result.angleSecondDerivativeXRad_Gs_to_Wn_ixyz
        )
        wing_spring_torque = stiffness * (inputs.angleXRad_Gs_to_Wn_ixyz - offset)
        total = tau_aero_wing + tau_inertial_wing + wing_spring_torque
        self.assertAlmostEqual(result.transmission_torque, total, places=13)

    def test_power_balance_over_a_crank_revolution(self) -> None:
        """Test virtual work on a real crank linkage over a whole revolution,

        including its dead centers: the actuator force left after the actuator's own
        inertia, damping, and spring equals J * T_transmission, and T_transmission stays
        finite where J is zero.
        """
        from pterasoftware import crank_linkage

        linkage = crank_linkage.CrankLinkage(
            0.11, -0.05, 0.065, np.pi / 2, 0.032, 0.11, gear_ratio=15.0
        )
        actuator_inertia, damping = 0.0004158, 1e-3
        drive = flap_drive.FlapDrive(
            wing_indices=(0,),
            transmission=linkage,
            actuator=lambda inputs: 2.0 + 0.5 * np.sin(inputs.actuator_position),
            actuator_inertia=actuator_inertia,
            actuator_damping=damping,
            passive_actuator_force_function=lambda q, q_dot: -0.01 * q,
            passive_wing_torque_function=lambda angle, rate: -3.0 * angle - 0.1 * rate,
            reference=flap_drive_fixtures.make_hold_reference_fixture(),
            use_position_lookup=False,
        )
        velocity = 188.0
        for position in np.linspace(0.0, 2.0 * np.pi * 15.0, 361):
            fluid_torque = -2.0 * np.tanh(
                linkage.evaluate_jacobian(position) * velocity
            )
            inputs = drive.inverse_dynamics_controller(
                0.0, position, velocity, fluid_torque, self.WING_INERTIA
            )
            result = drive.single_wing_dynamics(inputs, fluid_torque)
            self.assertTrue(np.isfinite(result.transmission_torque))
            actuator_side = (
                result.actuator_force
                - damping * velocity
                - 0.01 * position
                - actuator_inertia * result.actuator_acceleration
            )
            self.assertAlmostEqual(
                actuator_side, inputs.jacobian * result.transmission_torque, places=9
            )


class TestFeedforwardTerms(unittest.TestCase):
    """This is a class with functions to test the switches for each term of the
    feedforward force, and the user aerodynamic model."""

    FLUID_TORQUE = -1.3
    WING_INERTIA = 1.1
    POSITION, VELOCITY = 0.9, 2.0

    def _drive_and_reference(
        self, **overrides: Any
    ) -> tuple[flap_drive.FlapDrive, flap_drive.FlapReference]:
        """Builds a geared drive with every passive element and damping, and a rotating
        reference.

        :param overrides: The FlapDrive arguments to replace.
        :return: The FlapDrive and the FlapReference.
        """
        arguments: dict[str, Any] = {
            "wing_indices": (0,),
            "use_position_lookup": False,
            "transmission": flap_drive.FlapTransmission.geared(5.0),
            "actuator": lambda inputs: 0.0,
            "actuator_inertia": 0.03,
            "actuator_damping": 0.04,
            "passive_actuator_force_function": lambda q, q_dot: -0.2 * q,
            "passive_wing_torque_function": lambda angle, rate: -3.0 * angle
            - 0.2 * rate,
        }
        times = np.linspace(0.0, 2.0, 201)
        reference = flap_drive.FlapReference(
            times,
            4.0 * times + 0.3 * times**2,
            4.0 + 0.6 * times,
            np.full(201, 0.6),
            period=None,
        )
        arguments["reference"] = reference
        arguments.update(overrides)
        return flap_drive.FlapDrive(**arguments), reference

    def _inputs(self, **overrides: Any) -> flap_drive.FlapDriveInputs:
        """Evaluates the drive's inputs at a shared state.

        :param overrides: The FlapDrive arguments to replace.
        :return: The FlapDriveInputs.
        """
        drive, reference = self._drive_and_reference(**overrides)
        return drive.inverse_dynamics_controller(
            0.5,
            self.POSITION,
            self.VELOCITY,
            self.FLUID_TORQUE,
            self.WING_INERTIA,
        )

    def test_defaults_are_all_on(self) -> None:
        """Test that every term is on by default."""
        drive, _ = self._drive_and_reference()
        self.assertTrue(drive.feedforward_inertial)
        self.assertTrue(drive.feedforward_damping)
        self.assertTrue(drive.feedforward_passive)
        self.assertIs(drive.feedforward_aerodynamic, True)

    def test_terms_sum_to_the_full_inverse_dynamics(self) -> None:
        """Test that with every term on, the components sum to the feedforward force,
        and it equals effective_inertia * q''_desired - bias from the equation of
        motion."""
        drive, reference = self._drive_and_reference()
        inputs = drive.inverse_dynamics_controller(
            0.5,
            self.POSITION,
            self.VELOCITY,
            self.FLUID_TORQUE,
            self.WING_INERTIA,
        )
        _, desired_velocity, desired_acceleration = reference.evaluate(0.5)
        effective_inertia, bias, _ = drive._equation_of_motion_terms(
            self.POSITION,
            desired_velocity,
            inputs.jacobian,
            inputs.hessian,
            inputs.angleXRad_Gs_to_Wn_ixyz,
            self.FLUID_TORQUE,
            self.WING_INERTIA,
        )
        self.assertAlmostEqual(
            inputs.feedforward_force,
            effective_inertia * desired_acceleration - bias,
            places=12,
        )
        self.assertAlmostEqual(
            inputs.feedforward_force,
            inputs.feedforward_inertial_force
            + inputs.feedforward_damping_force
            + inputs.feedforward_passive_force
            + inputs.feedforward_aerodynamic_force,
            places=12,
        )

    def test_each_switch_removes_only_its_term(self) -> None:
        """Test that turning one term off sets it to 0.0, removes exactly it from the
        feedforward force, and leaves the other terms unchanged."""
        full = self._inputs()
        for switch, field in (
            ("feedforward_inertial", "feedforward_inertial_force"),
            ("feedforward_damping", "feedforward_damping_force"),
            ("feedforward_passive", "feedforward_passive_force"),
            ("feedforward_aerodynamic", "feedforward_aerodynamic_force"),
        ):
            reduced = self._inputs(**{switch: False})
            self.assertEqual(getattr(reduced, field), 0.0)
            self.assertNotEqual(getattr(full, field), 0.0)
            self.assertAlmostEqual(
                reduced.feedforward_force,
                full.feedforward_force - getattr(full, field),
                places=12,
            )

    def test_individual_term_formulas(self) -> None:
        """Test each term against its hand formula."""
        drive, reference = self._drive_and_reference()
        inputs = self._inputs()
        _, v, a = reference.evaluate(0.5)
        jacobian, hessian = inputs.jacobian, inputs.hessian
        angle = inputs.angleXRad_Gs_to_Wn_ixyz
        self.assertAlmostEqual(
            inputs.feedforward_inertial_force,
            (0.03 + jacobian**2 * self.WING_INERTIA) * a
            + jacobian * self.WING_INERTIA * hessian * v**2,
            places=12,
        )
        self.assertAlmostEqual(inputs.feedforward_damping_force, 0.04 * v, places=12)
        self.assertAlmostEqual(
            inputs.feedforward_passive_force,
            -(-0.2 * self.POSITION) - jacobian * (-3.0 * angle - 0.2 * jacobian * v),
            places=12,
        )
        self.assertAlmostEqual(
            inputs.feedforward_aerodynamic_force,
            -jacobian * self.FLUID_TORQUE,
            places=12,
        )

    def test_user_aerodynamic_model(self) -> None:
        """Test that a user aerodynamic model replaces the given fluid torque in the
        feedforward, and receives an AerodynamicModelInputs holding the actual and
        desired flap motion, the actuator state, the measured torque, and the
        OperatingPoint."""
        received: list[flap_drive.AerodynamicModelInputs] = []

        def model(aerodynamic_inputs: flap_drive.AerodynamicModelInputs) -> float:
            received.append(aerodynamic_inputs)
            return -0.5 * aerodynamic_inputs.desiredAngleDerivativeXRad_Gs_to_Wn_ixyz

        operating_point = operating_point_mod.OperatingPoint()
        drive, reference = self._drive_and_reference(feedforward_aerodynamic=model)
        self.assertIs(drive.feedforward_aerodynamic, model)
        inputs = drive.inverse_dynamics_controller(
            0.5,
            self.POSITION,
            self.VELOCITY,
            self.FLUID_TORQUE,
            self.WING_INERTIA,
            operating_point=operating_point,
        )
        _, v, a = reference.evaluate(0.5)
        desired_rate = inputs.jacobian * v
        self.assertEqual(len(received), 1)
        aerodynamic_inputs = received[0]
        self.assertIsInstance(aerodynamic_inputs, flap_drive.AerodynamicModelInputs)
        self.assertEqual(aerodynamic_inputs.time, 0.5)
        self.assertEqual(aerodynamic_inputs.wing_indices, drive.wing_indices)
        self.assertAlmostEqual(
            aerodynamic_inputs.angleXRad_Gs_to_Wn_ixyz, inputs.angleXRad_Gs_to_Wn_ixyz
        )
        self.assertAlmostEqual(
            aerodynamic_inputs.angleDerivativeXRad_Gs_to_Wn_ixyz,
            inputs.angleDerivativeXRad_Gs_to_Wn_ixyz,
        )
        self.assertAlmostEqual(
            aerodynamic_inputs.desiredAngleDerivativeXRad_Gs_to_Wn_ixyz, desired_rate
        )
        self.assertAlmostEqual(
            aerodynamic_inputs.desiredAngleSecondDerivativeXRad_Gs_to_Wn_ixyz,
            inputs.hessian * v**2 + inputs.jacobian * a,
        )
        self.assertEqual(aerodynamic_inputs.actuator_position, self.POSITION)
        self.assertEqual(aerodynamic_inputs.actuator_velocity, self.VELOCITY)
        self.assertEqual(aerodynamic_inputs.desired_actuator_velocity, v)
        self.assertEqual(aerodynamic_inputs.desired_actuator_acceleration, a)
        self.assertEqual(aerodynamic_inputs.measured_fluid_torque, self.FLUID_TORQUE)
        self.assertIs(aerodynamic_inputs.operating_point, operating_point)

        # Outside a simulation, there may be no OperatingPoint.
        drive.inverse_dynamics_controller(
            0.5, self.POSITION, self.VELOCITY, self.FLUID_TORQUE, self.WING_INERTIA
        )
        self.assertIsNone(received[1].operating_point)
        self.assertAlmostEqual(
            inputs.feedforward_aerodynamic_force,
            -inputs.jacobian * (-0.5 * desired_rate),
            places=12,
        )

    def test_switches_do_not_change_the_dynamics(self) -> None:
        """Test that the switches only change the feedforward, not the equation of
        motion that single_wing_dynamics integrates."""
        for switch in (
            "feedforward_inertial",
            "feedforward_damping",
            "feedforward_passive",
            "feedforward_aerodynamic",
        ):
            full_drive, _ = self._drive_and_reference(actuator=lambda inputs: 0.7)
            reduced_drive, _ = self._drive_and_reference(
                actuator=lambda inputs: 0.7, **{switch: False}
            )
            results = [
                drive.single_wing_dynamics(
                    drive.inverse_dynamics_controller(
                        0.5,
                        self.POSITION,
                        self.VELOCITY,
                        self.FLUID_TORQUE,
                        self.WING_INERTIA,
                    ),
                    self.FLUID_TORQUE,
                ).actuator_acceleration
                for drive in (full_drive, reduced_drive)
            ]
            self.assertEqual(results[0], results[1])

    def test_controller_estimate_is_separate_from_the_true_torque(self) -> None:
        """Test that the controller's estimate of the fluid torque only reaches the
        feedforward: the record carries the estimate, and a model passed as
        feedforward_aerodynamic leaves the dynamics, which take the true torque
        separately, unchanged."""
        measured_drive, _ = self._drive_and_reference(actuator=lambda inputs: 0.7)
        model_drive, _ = self._drive_and_reference(
            actuator=lambda inputs: 0.7,
            feedforward_aerodynamic=lambda aerodynamic_inputs: 123.0,
        )
        off_drive, _ = self._drive_and_reference(
            actuator=lambda inputs: 0.7, feedforward_aerodynamic=False
        )
        estimates = []
        accelerations = []
        for drive in (measured_drive, model_drive, off_drive):
            inputs = drive.inverse_dynamics_controller(
                0.5,
                self.POSITION,
                self.VELOCITY,
                self.FLUID_TORQUE,
                self.WING_INERTIA,
            )
            estimates.append(inputs.feedforward_fluid_torque)
            accelerations.append(
                drive.single_wing_dynamics(
                    inputs, self.FLUID_TORQUE
                ).actuator_acceleration
            )
        self.assertEqual(estimates, [self.FLUID_TORQUE, 123.0, 0.0])
        self.assertEqual(accelerations[0], accelerations[1])
        self.assertEqual(accelerations[0], accelerations[2])
        self.assertFalse(hasattr(inputs, "fluid_torque"))

        # The dynamics follow the true torque they are given, whatever the estimate.
        changed = measured_drive.single_wing_dynamics(
            inputs, self.FLUID_TORQUE + 1.0
        ).actuator_acceleration
        self.assertNotEqual(changed, accelerations[0])

    def test_validation(self) -> None:
        """Test that the switches reject values that aren't bools or, for the
        aerodynamic term, callables, and that a model returning nan is caught."""
        bad: Any = "yes"
        for switch in (
            "feedforward_inertial",
            "feedforward_damping",
            "feedforward_passive",
            "feedforward_aerodynamic",
        ):
            with self.assertRaises(TypeError):
                self._drive_and_reference(**{switch: bad})
        with self.assertRaises(ValueError):
            self._inputs(feedforward_aerodynamic=lambda aerodynamic_inputs: np.nan)


if __name__ == "__main__":
    unittest.main()


class TestSingleWingDynamicsMatchesMatlab(unittest.TestCase):
    """This is a class with functions to test that FlapDrive.single_wing_dynamics
    reproduces the blade element theory code's MATLAB singleWingDynamics."""

    def test_matches_matlab_single_wing_dynamics(self) -> None:
        """Test that, given the same motor state, applied torque, aerodynamic torque,
        linkage, wing inertia, damping, and springs, single_wing_dynamics gives the same
        motor acceleration, flap acceleration, and wing side torque as the MATLAB code,
        to near machine precision."""
        transmission = crank_linkage.CrankLinkage(
            gearbox_to_shoulder_y=0.11,
            gearbox_to_shoulder_z=-0.05,
            wing_crank_length=0.065,
            wing_crank_angle_offset_rad=np.pi / 2.0,
            gearbox_crank_length=0.032,
            linkage_length=0.11,
            gear_ratio=15.0,
        )
        for case in _MATLAB_SINGLE_WING_DYNAMICS_CASES:
            (
                motor_position,
                motor_velocity,
                motor_torque,
                aerodynamic_reaction_torque,
                expected_motor_acceleration,
                expected_flap_acceleration,
                expected_wing_side_torque,
            ) = case
            with self.subTest(motor_position=motor_position):

                def actuator(
                    inputs: flap_drive.FlapDriveInputs, torque: float = motor_torque
                ) -> float:
                    return torque

                drive = flap_drive.FlapDrive(
                    wing_indices=(0,),
                    transmission=transmission,
                    actuator=actuator,
                    actuator_inertia=0.0004158,
                    actuator_damping=0.001,
                    passive_wing_torque_function=lambda angle, rate: -5.0
                    * (angle + 0.2),
                    passive_actuator_force_function=lambda position, rate: -0.3
                    * (position - 0.1),
                    wing_flap_inertia=_MATLAB_WING_INERTIA,
                    reference=flap_drive_fixtures.make_hold_reference_fixture(),
                    use_position_lookup=False,
                )
                inputs = drive.inverse_dynamics_controller(
                    0.0,
                    motor_position,
                    motor_velocity,
                    measured_fluid_torque=-aerodynamic_reaction_torque,
                    computed_wing_flap_inertia=_MATLAB_WING_INERTIA,
                )
                accelerations = drive.single_wing_dynamics(
                    inputs, fluid_torque=-aerodynamic_reaction_torque
                )
                for actual, expected in (
                    (accelerations.actuator_acceleration, expected_motor_acceleration),
                    (
                        accelerations.angleSecondDerivativeXRad_Gs_to_Wn_ixyz,
                        expected_flap_acceleration,
                    ),
                    (accelerations.transmission_torque, expected_wing_side_torque),
                ):
                    self.assertAlmostEqual(
                        actual, expected, delta=1e-10 * abs(expected)
                    )
