"""This module contains classes to test the flap trajectory generator, the constant
velocity reference, and the crank linkage's trajectory conversion."""

import unittest
from typing import Any

import numpy as np
import numpy.testing as npt

import pterasoftware as ps
from pterasoftware import crank_linkage, flap_drive

GEOMETRY = (0.11, -0.05, 0.065, np.pi / 2, 0.032, 0.11)


class TestGenerateFlapTrajectory(unittest.TestCase):
    """This is a class with functions to test generate_flap_trajectory."""

    def _generate(self, style: str, **overrides: Any) -> tuple[np.ndarray, ...]:
        """Generates a 2 Hz trajectory with wide stroke limits.

        :param style: The style.
        :param overrides: The arguments to replace.
        :return: The times, angles, rates, and accelerations.
        """
        arguments: dict[str, Any] = {
            "style": style,
            "flapping_frequency": 2.0,
            "wing_beat_amplitude_rad": 1.0,
            "dihedral_rad": 0.1,
            "min_angle_rad": -5.0,
            "max_angle_rad": 5.0,
        }
        arguments.update(overrides)
        return flap_drive.generate_flap_trajectory(**arguments)

    def test_sine(self) -> None:
        """Test the sine against its closed form, starting on an upstroke."""
        times, angles, rates, accelerations = self._generate("sine")
        omega = 2.0 * np.pi * 2.0
        npt.assert_allclose(angles, 0.1 + 0.5 * np.sin(omega * times), atol=1e-12)
        npt.assert_allclose(rates, 0.5 * omega * np.cos(omega * times), atol=1e-12)
        npt.assert_allclose(
            accelerations, -0.5 * omega**2 * np.sin(omega * times), atol=1e-9
        )
        self.assertGreater(rates[0], 0.0)
        self.assertAlmostEqual(times[1] - times[0], 1e-4)
        self.assertAlmostEqual(times[-1], 0.5)

    def test_square(self) -> None:
        """Test the square's constant velocities and that it covers the full wing beat
        amplitude, centered on the dihedral."""
        times, angles, rates, accelerations = self._generate("square")
        self.assertTrue(np.all(rates[times <= 0.25] == -4.0))
        self.assertTrue(np.all(rates[times > 0.25] == 4.0))
        self.assertTrue(np.all(accelerations == 0.0))
        self.assertAlmostEqual(float(np.ptp(angles)), 1.0, places=6)
        self.assertAlmostEqual(float(np.mean(angles)), 0.1, places=12)

    def test_clipping_zeroes_the_derivatives(self) -> None:
        """Test that clipped samples hold the limit with zero velocity."""
        _, angles, rates, accelerations = self._generate(
            "sine", min_angle_rad=-0.2, max_angle_rad=0.4
        )
        clipped = (angles <= -0.2) | (angles >= 0.4)
        self.assertTrue(clipped.any())
        self.assertTrue(np.all(rates[clipped] == 0.0))
        self.assertTrue(np.all(accelerations[clipped] == 0.0))
        self.assertLessEqual(float(np.max(angles)), 0.4)

    def test_cycles_repeat_without_duplicates(self) -> None:
        """Test that the cycles join without repeating the shared sample."""
        times, angles, _, _ = self._generate("sine", num_cycles=3)
        self.assertTrue(np.all(np.diff(times) > 0.0))
        self.assertAlmostEqual(times[-1], 1.5)
        one_cycle = len(self._generate("sine")[0])
        self.assertEqual(len(times), 3 * (one_cycle - 1) + 1)
        npt.assert_allclose(
            angles[: one_cycle - 1],
            angles[one_cycle - 1 : 2 * (one_cycle - 1)],
            atol=1e-12,
        )

    def test_validation(self) -> None:
        """Test that invalid arguments are rejected."""
        with self.assertRaises(ValueError):
            self._generate("triangle")
        with self.assertRaises(ValueError):
            self._generate("sine", min_angle_rad=1.0, max_angle_rad=0.0)
        with self.assertRaises(ValueError):
            self._generate("quasi_square")
        with self.assertRaises(ValueError):
            self._generate("sine", num_cycles=0)
        with self.assertRaises(ValueError):
            self._generate("sine", flapping_frequency=0.0)


class TestConstantVelocityReference(unittest.TestCase):
    """This is a class with functions to test FlapReference.constant_velocity."""

    def test_constant_velocity(self) -> None:
        """Test the table, its spacing, and its position lookup."""
        reference = flap_drive.FlapReference.constant_velocity(
            1.0, 50.0, 0.5, period=None
        )
        self.assertAlmostEqual(reference.times[1], 1e-4)
        self.assertAlmostEqual(reference.times[-1], 0.5)
        npt.assert_allclose(reference.positions, 1.0 + 50.0 * reference.times)
        self.assertTrue(np.all(reference.velocities == 50.0))
        self.assertTrue(np.all(reference.accelerations == 0.0))
        self.assertTrue(reference.positions_are_monotonic)

    def test_position_lookup_takes_the_nearest_sample(self) -> None:
        """Test that the lookup returns the nearest entry, not an interpolation."""
        times = np.array([0.0, 1.0, 2.0])
        reference = flap_drive.FlapReference(
            times, [0.0, 1.0, 2.0], [10.0, 20.0, 30.0], [1.0, 2.0, 3.0], period=None
        )
        self.assertEqual(reference.evaluate_at_position(1.4), (20.0, 2.0))
        self.assertEqual(reference.evaluate_at_position(1.6), (30.0, 3.0))

    def test_validation(self) -> None:
        """Test that a too short duration is rejected."""
        with self.assertRaises(ValueError):
            flap_drive.FlapReference.constant_velocity(0.0, 1.0, 1e-5, period=None)


class TestCrankTrajectories(unittest.TestCase):
    """This is a class with functions to test the crank linkage with the generated
    trajectories."""

    def setUp(self) -> None:
        """Set up the crank and its stroke."""
        self.ratio = 15.0
        self.transmission = crank_linkage.CrankLinkage(*GEOMETRY, gear_ratio=self.ratio)
        self.low, self.high = self.transmission.angle_range()

    def test_angle_range(self) -> None:
        """Test the sampled stroke against the linkage's exact extremes."""
        linkage = crank_linkage.CrankLinkage(*GEOMETRY)
        angles = linkage.wing_angle(np.linspace(0.0, 2.0 * np.pi, 200001))
        self.assertAlmostEqual(self.low, float(angles.min()), places=6)
        self.assertAlmostEqual(self.high, float(angles.max()), places=6)

    def _trajectory(
        self, style: str, stroke_fraction: float = 1.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generates a 3 Hz trajectory over a fraction of the crank's stroke.

        :param style: The style.
        :param stroke_fraction: The fraction of the crank's full stroke to cover.
        :return: The times, angles, rates, and accelerations.
        """
        return flap_drive.generate_flap_trajectory(
            style,
            3.0,
            stroke_fraction * (self.high - self.low),
            (self.high + self.low) / 2.0,
            self.low,
            self.high,
            num_cycles=2,
        )

    def test_sine_converts_without_jumps_or_glitches(self) -> None:
        """Test that a full stroke sine converts smoothly, including the samples exactly
        on the turnarounds, where the analytical product is indeterminate."""
        times, angles, rates, accelerations = self._trajectory("sine")
        reference = flap_drive.FlapReference.from_flap_trajectory(
            self.transmission, times, angles, rates, accelerations, period=1.0 / 3.0
        )
        self.assertTrue(np.all(np.diff(reference.positions) > 0.0))
        self.assertGreater(float(np.min(reference.velocities)), 100.0)
        self.assertTrue(np.isfinite(reference.accelerations).all())
        # Converting back through the forward kinematics recovers the flap angle.
        recovered = [self.transmission.evaluate_angle(q) for q in reference.positions]
        npt.assert_allclose(recovered, angles, atol=1e-9)

    def test_short_turnarounds_are_rejected(self) -> None:
        """Test that trajectories that turn around short of the stroke are rejected,
        because the crank would have to jump between inverse kinematics solutions."""
        for style in ("sine", "square"):
            times, angles, rates, accelerations = self._trajectory(style, 0.8)
            with self.assertRaises(ValueError):
                flap_drive.FlapReference.from_flap_trajectory(
                    self.transmission,
                    times,
                    angles,
                    rates,
                    accelerations,
                    period=1.0 / 3.0,
                )


class TestDriveWithItsOwnReference(unittest.TestCase):
    """This is a class with functions to test a FlapDrive that carries a reference."""

    def test_drive_starts_where_its_reference_starts(self) -> None:
        """Test the default initial state, the first time step's flap angle, and that
        the drive keeps its reference."""
        from tests.unit.fixtures import flap_drive_fixtures

        transmission = flap_drive.FlapTransmission.identity()
        reference = flap_drive.FlapReference.constant_velocity(
            0.2, 0.5, 1.0, period=None
        )
        drive = flap_drive.FlapDrive(
            wing_indices=(0, 1),
            transmission=transmission,
            actuator=lambda inputs: inputs.feedforward_force,
            actuator_inertia=1e-3,
            reference=reference,
            use_position_lookup=False,
        )
        problem = ps.flap_driven_aeroelastic_unsteady_problem.FlapDrivenAeroelasticUnsteadyProblem(
            movement=flap_drive_fixtures.make_flap_driven_aeroelastic_movement_fixture(),
            wing_density=0.5,
            spring_constant_rad=20.0,
            damping_constant_rad=1.0,
            flap_drives=[drive],
        )
        self.assertIs(problem.flap_drives[0].reference, reference)
        self.assertEqual(problem.flap_actuator_positions[0][0], 0.2)
        self.assertEqual(problem.flap_actuator_velocities[0][0], 0.5)
        wing = problem.steady_problems[0].airplanes[0].wings[0]
        self.assertAlmostEqual(wing.angles_Gs_to_Wn_ixyz[0], np.rad2deg(0.2))

    def test_reference_must_cover_the_simulation(self) -> None:
        """Test that a reference shorter than the simulation is rejected."""
        from tests.unit.fixtures import flap_drive_fixtures

        drive = flap_drive.FlapDrive(
            wing_indices=(0, 1),
            transmission=flap_drive.FlapTransmission.identity(),
            actuator=lambda inputs: 0.0,
            actuator_inertia=1e-3,
            reference=flap_drive.FlapReference.constant_velocity(
                0.0, 0.0, 0.1, period=None
            ),
            use_position_lookup=False,
        )
        with self.assertRaises(ValueError):
            ps.flap_driven_aeroelastic_unsteady_problem.FlapDrivenAeroelasticUnsteadyProblem(
                movement=flap_drive_fixtures.make_flap_driven_aeroelastic_movement_fixture(),
                wing_density=0.5,
                spring_constant_rad=20.0,
                damping_constant_rad=1.0,
                flap_drives=[drive],
            )

    def test_reference_type_is_validated(self) -> None:
        """Test that the reference must be a FlapReference."""
        bad: Any = "reference"
        with self.assertRaises(TypeError):
            flap_drive.FlapDrive(
                wing_indices=(0,),
                transmission=flap_drive.FlapTransmission.identity(),
                actuator=lambda inputs: 0.0,
                actuator_inertia=1e-3,
                reference=bad,
                use_position_lookup=False,
            )


if __name__ == "__main__":
    unittest.main()
