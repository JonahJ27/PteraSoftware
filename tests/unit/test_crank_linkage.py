"""This module contains classes to test the crank linkage transmission."""

import unittest
from typing import Any

import numpy as np
import numpy.testing as npt

import pterasoftware as ps
from pterasoftware import crank_linkage, flap_drive
from tests.unit.fixtures import flap_drive_fixtures

GEOMETRY = (0.11, -0.05, 0.065, np.pi / 2, 0.032, 0.11)


def _reference_wing_angle(crank_angle: np.ndarray) -> np.ndarray:
    """Evaluates the linkage's forward kinematics by an independent, step by step route.

    :param crank_angle: The crank angle (rad).
    :return: The wing angle (rad).
    """
    sy, sz, lw, offset, lg, ll = GEOMETRY
    py = -lg * np.cos(crank_angle) + sy
    pz = -lg * np.sin(crank_angle) + sz
    distance = np.sqrt(py**2 + pz**2)
    angle = np.arccos((ll**2 - lw**2 - distance**2) / (-2.0 * lw * distance))
    return np.pi - (angle - np.arctan2(pz, py)) - offset


class TestCrankLinkage(unittest.TestCase):
    """This is a class with functions to test CrankLinkage."""

    def setUp(self) -> None:
        """Set up a linkage and a set of crank angles."""
        self.linkage = crank_linkage.CrankLinkage(*GEOMETRY)
        self.crank = np.linspace(0.0, 2.0 * np.pi, 361)

    def test_forward_kinematics(self) -> None:
        """Test the wing angle against an independent evaluation."""
        npt.assert_allclose(
            self.linkage.wing_angle(self.crank),
            _reference_wing_angle(self.crank),
            atol=1e-12,
        )

    def test_jacobian_and_hessian_match_differences(self) -> None:
        """Test the analytical derivatives against differences of the forward
        kinematics, which are used here only as an independent check."""
        step = 1e-5
        reference = _reference_wing_angle
        jacobian = (reference(self.crank + step) - reference(self.crank - step)) / (
            2.0 * step
        )
        npt.assert_allclose(
            self.linkage.wing_angle_jacobian(self.crank), jacobian, atol=1e-8
        )
        npt.assert_allclose(
            self.linkage.wing_angle_hessian(self.crank),
            (
                self.linkage.wing_angle_jacobian(self.crank + step)
                - self.linkage.wing_angle_jacobian(self.crank - step)
            )
            / (2.0 * step),
            atol=1e-7,
        )

    def test_inverse_kinematics_round_trip(self) -> None:
        """Test that both inverse solutions recover the wing angle."""
        angles = self.linkage.wing_angle(self.crank)
        wing = np.linspace(angles.min() + 0.02, angles.max() - 0.02, 100)
        for upstroke in (True, False):
            crank = self.linkage.crank_angles(wing, upstroke)
            npt.assert_allclose(self.linkage.wing_angle(crank), wing, atol=1e-10)

    def test_inverse_derivatives_invert_the_forward_jacobian(self) -> None:
        """Test that dq/dphi times dphi/dq is one on each branch."""
        angles = self.linkage.wing_angle(self.crank)
        wing = np.linspace(angles.min() + 0.02, angles.max() - 0.02, 100)
        for upstroke in (True, False):
            crank = self.linkage.crank_angles(wing, upstroke)
            first, _ = self.linkage.crank_angle_derivatives(wing, upstroke)
            npt.assert_allclose(
                first * self.linkage.wing_angle_jacobian(crank), 1.0, atol=1e-9
            )

    def test_unassemblable_linkage_is_rejected(self) -> None:
        """Test that lengths that can't close the linkage are rejected."""
        with self.assertRaises(ValueError):
            crank_linkage.CrankLinkage(0.5, 0.0, 0.065, 0.0, 0.032, 0.11)

    def test_validation(self) -> None:
        """Test that bad lengths are rejected."""
        for index in (2, 4, 5):
            values: list[Any] = list(GEOMETRY)
            values[index] = -0.1
            with self.assertRaises(ValueError):
                crank_linkage.CrankLinkage(*values)
        bad: Any = "0.1"
        with self.assertRaises(TypeError):
            crank_linkage.CrankLinkage(bad, *GEOMETRY[1:])

    def test_uniform_crank_trajectory_is_recovered(self) -> None:
        """Test that a trajectory built from a uniformly turning motor converts back to
        that motor's constant speed and zero acceleration."""
        ratio = 15.0
        speed = 2.0 * np.pi * ratio
        time = np.arange(0.0, 2.0, 0.001)
        crank = speed * time / ratio
        phi = self.linkage.wing_angle(crank)
        phi_dot = self.linkage.wing_angle_jacobian(crank) * speed / ratio
        phi_ddot = self.linkage.wing_angle_hessian(crank) * (speed / ratio) ** 2
        position, velocity, acceleration = crank_linkage.CrankLinkage(
            *GEOMETRY, gear_ratio=ratio
        ).convert_trajectory(time, phi, phi_dot, phi_ddot)
        trimmed = slice(5, -5)
        npt.assert_allclose(velocity[trimmed], speed, atol=1e-4)
        npt.assert_allclose(acceleration[trimmed], 0.0, atol=0.05)
        self.assertTrue((np.diff(position) > 0.0).all())

    def test_numerical_trajectory_conversion(self) -> None:
        """Test that the numerical method recovers a uniformly turning motor and gives
        the same positions as the analytical method."""
        ratio = 15.0
        speed = 2.0 * np.pi * ratio
        time = np.arange(0.0, 2.0, 0.001)
        crank = speed * time / ratio
        phi = self.linkage.wing_angle(crank)
        phi_dot = self.linkage.wing_angle_jacobian(crank) * speed / ratio
        phi_ddot = self.linkage.wing_angle_hessian(crank) * (speed / ratio) ** 2
        analytical = crank_linkage.CrankLinkage(
            *GEOMETRY, gear_ratio=ratio
        ).convert_trajectory(time, phi, phi_dot, phi_ddot)
        position, velocity, acceleration = crank_linkage.CrankLinkage(
            *GEOMETRY, gear_ratio=ratio, derivative_method="numerical"
        ).convert_trajectory(time, phi, phi_dot, phi_ddot)
        npt.assert_allclose(position, analytical[0], atol=1e-10)
        trimmed = slice(5, -5)
        npt.assert_allclose(velocity[trimmed], speed, rtol=1e-4)
        npt.assert_allclose(acceleration[trimmed], 0.0, atol=1.0)

    def test_derivative_method_validation(self) -> None:
        """Test that an unknown derivative method is rejected, and that the private
        conversion rejects the numerical method without times."""
        with self.assertRaises(ValueError):
            crank_linkage.CrankLinkage(*GEOMETRY, derivative_method="symbolic")
        zeros = np.zeros(3)
        with self.assertRaises(ValueError):
            self.linkage._convert_wing_trajectory(1.0, zeros, zeros, zeros, "numerical")

    def test_properties(self) -> None:
        """Test that the linkage reports the parameters it was built with."""
        linkage = crank_linkage.CrankLinkage(
            *GEOMETRY, gear_ratio=15.0, derivative_method="numerical"
        )
        self.assertEqual(
            (
                linkage.gearbox_to_shoulder_y,
                linkage.gearbox_to_shoulder_z,
                linkage.wing_crank_length,
                linkage.wing_crank_angle_offset_rad,
                linkage.gearbox_crank_length,
                linkage.linkage_length,
            ),
            GEOMETRY,
        )
        self.assertEqual(linkage.gear_ratio, 15.0)
        self.assertEqual(linkage.derivative_method, "numerical")


class TestCrankLinkageTransmission(unittest.TestCase):
    """This is a class with functions to test CrankLinkage as a FlapDrive's
    Transmission, in the motor's angle."""

    def setUp(self) -> None:
        """Set up a geared crank linkage and the same linkage without a gearbox."""
        self.ratio = 15.0
        self.transmission = crank_linkage.CrankLinkage(*GEOMETRY, gear_ratio=self.ratio)
        self.linkage = crank_linkage.CrankLinkage(*GEOMETRY)

    def test_satisfies_the_transmission_interface(self) -> None:
        """Test that a CrankLinkage is a Transmission and that a FlapDrive accepts it
        directly, without a wrapper."""
        self.assertIsInstance(self.transmission, flap_drive.Transmission)
        drive = flap_drive.FlapDrive(
            wing_indices=(0,),
            transmission=self.transmission,
            actuator=lambda inputs: 0.0,
            actuator_inertia=1e-3,
            reference=flap_drive_fixtures.make_hold_reference_fixture(),
            use_position_lookup=False,
        )
        self.assertIs(drive.transmission, self.transmission)

    def test_angle_range(self) -> None:
        """Test the sampled stroke against a finer sampling of the forward
        kinematics."""
        low, high = self.transmission.angle_range()
        angles = self.linkage.wing_angle(np.linspace(0.0, 2.0 * np.pi, 200001))
        self.assertAlmostEqual(low, float(angles.min()), places=6)
        self.assertAlmostEqual(high, float(angles.max()), places=6)

    def test_save_and_load(self) -> None:
        """Test that a CrankLinkage survives a save and load round trip."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "linkage.psz"
            ps.save(path, self.transmission)
            loaded = ps.load(path)
        assert isinstance(loaded, crank_linkage.CrankLinkage)
        self.assertEqual(loaded.gear_ratio, self.ratio)
        self.assertAlmostEqual(
            loaded.evaluate_jacobian(7.3), self.transmission.evaluate_jacobian(7.3)
        )

    def test_scaling_by_the_gear_ratio(self) -> None:
        """Test that the motor side derivatives are the crank side ones scaled."""
        motor = 7.3
        crank = motor / self.ratio
        self.assertAlmostEqual(
            self.transmission.evaluate_angle(motor),
            float(self.linkage.wing_angle(crank)),
        )
        self.assertAlmostEqual(
            self.transmission.evaluate_jacobian(motor),
            float(self.linkage.wing_angle_jacobian(crank)) / self.ratio,
        )
        self.assertAlmostEqual(
            self.transmission.evaluate_hessian(motor),
            float(self.linkage.wing_angle_hessian(crank)) / self.ratio**2,
        )

    def test_trajectory_conversion_scales_by_the_gear_ratio(self) -> None:
        """Test the motor trajectory of a uniformly turning crank."""
        omega = 2.0 * np.pi
        time = np.arange(0.0, 1.0, 0.001)
        crank = omega * time
        phi = self.linkage.wing_angle(crank)
        phi_dot = self.linkage.wing_angle_jacobian(crank) * omega
        phi_ddot = self.linkage.wing_angle_hessian(crank) * omega**2
        _, velocity, _ = self.transmission.convert_trajectory(
            time, phi, phi_dot, phi_ddot
        )
        npt.assert_allclose(velocity[5:-5], self.ratio * omega, atol=1e-3)

    def test_gear_ratio_validation(self) -> None:
        """Test that a zero gear ratio is rejected."""
        with self.assertRaises(ValueError):
            crank_linkage.CrankLinkage(*GEOMETRY, gear_ratio=0.0)

    def test_derivative_method_validation(self) -> None:
        """Test that an unknown derivative method is rejected."""
        with self.assertRaises(ValueError):
            crank_linkage.CrankLinkage(*GEOMETRY, derivative_method="symbolic")

    def test_numerical_method_through_the_transmission(self) -> None:
        """Test that the transmission passes the sample times to the numerical
        method."""
        transmission = crank_linkage.CrankLinkage(
            *GEOMETRY, gear_ratio=self.ratio, derivative_method="numerical"
        )
        omega = 2.0 * np.pi
        time = np.arange(0.0, 1.0, 0.001)
        crank = omega * time
        phi = self.linkage.wing_angle(crank)
        phi_dot = self.linkage.wing_angle_jacobian(crank) * omega
        phi_ddot = self.linkage.wing_angle_hessian(crank) * omega**2
        _, velocity, _ = transmission.convert_trajectory(time, phi, phi_dot, phi_ddot)
        npt.assert_allclose(velocity[5:-5], self.ratio * omega, rtol=1e-4)

    def test_jacobian_reaches_near_zero(self) -> None:
        """Test that the linkage has the dead centers that make it singular."""
        motor = np.linspace(0.0, 2.0 * np.pi * self.ratio, 20001)
        jacobian = np.array([self.transmission.evaluate_jacobian(m) for m in motor])
        self.assertLess(float(np.min(np.abs(jacobian))), 1e-3)

    def test_drive_with_a_crank_stays_finite_at_the_dead_center(self) -> None:
        """Test the drive's dynamics across a full revolution of the crank."""
        drive = flap_drive.FlapDrive(
            wing_indices=(0,),
            transmission=self.transmission,
            actuator=lambda inputs: 0.5,
            actuator_inertia=4e-4,
            reference=flap_drive_fixtures.make_hold_reference_fixture(),
            use_position_lookup=False,
        )
        for motor in np.linspace(0.0, 2.0 * np.pi * self.ratio, 500):
            inputs = drive.inverse_dynamics_controller(0.0, motor, 90.0, 0.2, 1.4)
            result = drive.single_wing_dynamics(inputs, 0.2)
            self.assertTrue(np.isfinite(result.actuator_acceleration))
            self.assertTrue(np.isfinite(result.transmission_torque))


if __name__ == "__main__":
    unittest.main()
