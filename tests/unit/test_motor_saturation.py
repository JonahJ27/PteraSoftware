"""This module contains classes to test MotorSaturationControllers."""

import unittest
from typing import Any

import numpy as np

import pterasoftware as ps

# These are the MAD M13 EEE KV90 (motor 9) lines at 48 V, built the same way as in the
# blade element theory motor limit simulation.
_MEASURED_TORQUES = np.array([0.0, 8.85]) / 48.0
_SPEED_TORQUE = np.polyfit(
    np.array([4300.0, 3424.0]) / 48.0 / 60.0 * 2.0 * np.pi, _MEASURED_TORQUES, 1
)
_SPEED_TORQUE[1] *= 48.0
_TORQUE_CURRENT = np.polyfit(np.array([4.8, 88.0]) / 48.0, _MEASURED_TORQUES, 1)


def _make_controller(**overrides: Any) -> ps.motor_saturation.MotorSaturationController:
    """Builds the motor 9 controller with an ODrive Pro and a 15:1, 225 N*m gearbox.

    :param overrides: The constructor arguments to replace.
    :return: The MotorSaturationController.
    """
    arguments: dict[str, Any] = {
        "speed_torque_coefficients": _SPEED_TORQUE,
        "torque_current_coefficients": _TORQUE_CURRENT,
        "motor_continuous_torque": 8.8,
        "motor_peak_duration": 0.5,
        "esc_peak_current": 150.0,
        "esc_continuous_current": 100.0,
        "esc_peak_duration": 0.1,
        "gearbox_max_torque": 225.0,
        "gear_ratio": 15.0,
        "esc_max_brake_current": 1000.0,
    }
    arguments.update(overrides)
    return ps.motor_saturation.MotorSaturationController(**arguments)


class TestMotorSaturationController(unittest.TestCase):
    """This is a class with functions to test MotorSaturationControllers."""

    def setUp(self) -> None:
        """Set up a fresh motor 9 controller."""
        self.controller = _make_controller()

    def test_static_limits(self) -> None:
        """Test the limits against the values in the motor limit simulation's notes."""
        no_load_speed = -_SPEED_TORQUE[1] / _SPEED_TORQUE[0]
        self.assertAlmostEqual(no_load_speed, 450.29, places=1)
        self.assertAlmostEqual(self.controller.max_torque_from_speed(0.0), 43.44, 1)
        self.assertEqual(self.controller.max_torque_from_speed(500.0), 0.0)
        self.assertAlmostEqual(self.controller.torque_at_current(30.0), 3.18, places=2)
        self.assertAlmostEqual(self.controller.max_brake_torque(), 106.36, places=1)
        self.assertEqual(self.controller.gearbox_torque_limit(), 15.0)

    def test_no_brake_limit_by_default(self) -> None:
        """Test that the brake current is unlimited unless one is given."""
        controller = _make_controller(esc_max_brake_current=np.inf)
        self.assertEqual(controller.max_brake_torque(), np.inf)

    def test_torque_speed_curve_limits_only_driving(self) -> None:
        """Test that driving above the no-load speed is cut to 0.0 and braking is
        not."""
        driving, info = self.controller.clip_torque(5.0, 500.0, 0.0)
        self.assertEqual(driving, 0.0)
        self.assertIn("motor_torque_speed_curve", info.saturation_active)
        braking, info = self.controller.clip_torque(-5.0, 500.0, 0.0)
        self.assertEqual(braking, -5.0)
        self.assertEqual(info.saturation_active, ())

    def test_brake_current_limits_braking(self) -> None:
        """Test that a braking torque is capped by the brake current."""
        controller = _make_controller(esc_max_brake_current=30.0)
        torque, info = controller.clip_torque(-14.0, 100.0, 0.0)
        self.assertAlmostEqual(torque, -controller.torque_at_current(30.0))
        self.assertIn("max_brake_current", info.saturation_active)

    def test_gearbox_limit(self) -> None:
        """Test that a command above the gearbox's limit at the motor is clipped."""
        torque, info = self.controller.clip_torque(40.0, 0.0, 0.0)
        self.assertEqual(torque, 15.0)
        self.assertEqual(info.saturation_active, ("gearbox_max_torque",))

    def test_controller_lock_windows(self) -> None:
        """Test the controller's lock windows under a constant 14 N*m demand, which the
        motor limit simulation's notes give as 0.1001-0.1501 s, 0.2003-0.2502 s,
        0.3003-0.3503 s, and 0.4004-0.4504 s."""
        times = np.arange(0.0, 0.46, 1e-4)
        locked = np.array(
            [self.controller.clip_torque(14.0, 10.0, t)[1].esc_locked for t in times]
        )
        edges = times[np.flatnonzero(np.diff(locked.astype(int))) + 1]
        expected = [0.1001, 0.1501, 0.2003, 0.2502, 0.3003, 0.3503, 0.4004, 0.4504]
        np.testing.assert_allclose(edges, expected, atol=2e-4)

    def test_locked_controller_holds_continuous_current(self) -> None:
        """Test that a locked controller applies only its continuous current's
        torque."""
        torque = 0.0
        for time in np.arange(0.0, 0.12, 1e-4):
            torque, info = self.controller.clip_torque(14.0, 10.0, time)
        self.assertTrue(info.esc_locked)
        self.assertAlmostEqual(torque, self.controller.torque_at_current(100.0))

    def test_controller_limit_names(self) -> None:
        """Test that the controller's limit is reported as its peak current while it is
        unlocked and as its continuous current while it is locked."""
        controller = _make_controller(gearbox_max_torque=1000.0)
        _, info = controller.clip_torque(20.0, 10.0, 0.0)
        self.assertEqual(info.saturation_active, ("esc_peak_current",))
        self.assertAlmostEqual(info.limits[0], controller.torque_at_current(150.0))
        for time in np.arange(0.0, 0.12, 1e-4):
            _, info = controller.clip_torque(20.0, 10.0, time)
        self.assertTrue(info.esc_locked)
        self.assertEqual(info.saturation_active, ("esc_continuous_current",))
        self.assertAlmostEqual(info.limits[0], controller.torque_at_current(100.0))

    def test_motor_peak_runs_out(self) -> None:
        """Test that the motor is held to its continuous torque after its peak
        allowance, with a demand that the controller can sustain indefinitely."""
        torque = 0.0
        for time in np.arange(0.0, 0.52, 1e-3):
            torque, info = self.controller.clip_torque(10.0, 10.0, time)
        self.assertTrue(info.motor_locked)
        self.assertAlmostEqual(torque, 8.8)

    def test_below_continuous_never_locks(self) -> None:
        """Test that a demand below every continuous limit never uses the allowance."""
        for time in np.arange(0.0, 1.0, 1e-3):
            torque, info = self.controller.clip_torque(5.0, 10.0, time)
            self.assertEqual(torque, 5.0)
        self.assertFalse(info.motor_locked or info.esc_locked)
        self.assertEqual(info.esc_peak_time_elapsed, 0.0)

    def test_repeated_and_earlier_calls_change_nothing(self) -> None:
        """Test that the allowances depend on time stamps, not on how often or in what
        order they are evaluated."""
        once = []
        for time in np.arange(0.0, 0.3, 1e-3):
            once.append(self.controller.clip_torque(14.0, 10.0, time)[0])
        repeated_controller = _make_controller()
        repeated = []
        for time in np.arange(0.0, 0.3, 1e-3):
            for _ in range(7):
                torque = repeated_controller.clip_torque(14.0, 10.0, time)[0]
            repeated_controller.clip_torque(14.0, 10.0, max(0.0, time - 5e-4))
            repeated.append(torque)
        np.testing.assert_allclose(once, repeated)

    def test_reset_state(self) -> None:
        """Test that resetting empties the allowances and forgets the last call."""
        for time in np.arange(0.0, 0.12, 1e-4):
            self.controller.clip_torque(14.0, 10.0, time)
        self.controller.reset_state()
        self.assertIsNone(self.controller.last_info)
        _, info = self.controller.clip_torque(14.0, 10.0, 0.0)
        self.assertFalse(info.esc_locked)
        self.assertEqual(info.esc_peak_time_elapsed, 0.0)

    def test_max_torque_possible_and_continuous(self) -> None:
        """Test the combined limits."""
        self.assertAlmostEqual(
            self.controller.max_torque_possible(0.0, False, False), 15.0
        )
        self.assertAlmostEqual(
            self.controller.max_torque_possible(0.0, True, False), 8.8
        )
        self.assertAlmostEqual(self.controller.max_torque_continuous(0.0), 8.8)
        self.assertEqual(self.controller.max_torque_possible(460.0, False, False), 0.0)

    def test_validation(self) -> None:
        """Test that invalid parameters are rejected."""
        for name in (
            "motor_continuous_torque",
            "motor_peak_duration",
            "esc_peak_current",
            "esc_continuous_current",
            "esc_peak_duration",
            "gearbox_max_torque",
            "gear_ratio",
            "esc_max_brake_current",
        ):
            with self.assertRaises(ValueError):
                _make_controller(**{name: 0.0})
        for bad in ([], [[1.0]], [np.nan]):
            with self.assertRaises(ValueError):
                _make_controller(speed_torque_coefficients=bad)
        with self.assertRaises(ValueError):
            _make_controller(esc_max_brake_current=-np.inf)

    def test_slots(self) -> None:
        """Test that the class has no instance dict and rejects new attributes."""
        self.assertFalse(hasattr(self.controller, "__dict__"))
        with self.assertRaises(AttributeError):
            setattr(self.controller, "new_attribute", 1.0)


if __name__ == "__main__":
    unittest.main()
