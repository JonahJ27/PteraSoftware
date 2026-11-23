"""This module contains a class to test oscillation functions."""

import unittest
from collections.abc import Callable

import numpy as np
import numpy.testing as npt

# noinspection PyProtectedMember
from pterasoftware import _oscillation
from tests.unit.fixtures import oscillation_fixtures


class TestOscillation(unittest.TestCase):
    """This is a class with functions to test oscillation functions."""

    static_amp: float
    static_period: float
    static_phase: float
    static_base: float
    phase_offset_amp: float
    phase_offset_period: float
    phase_offset_phase: float
    phase_offset_base: float
    max_phase_amp: float
    max_phase_period: float
    max_phase_phase: float
    max_phase_base: float
    time: float
    valid_custom_sine: Callable[[float], float]

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test fixtures once for all oscillation function tests."""
        # Parameter fixtures.
        (
            cls.static_amp,
            cls.static_period,
            cls.static_phase,
            cls.static_base,
        ) = oscillation_fixtures.make_static_parameters_fixture()

        (
            cls.phase_offset_amp,
            cls.phase_offset_period,
            cls.phase_offset_phase,
            cls.phase_offset_base,
        ) = oscillation_fixtures.make_phase_offset_parameters_fixture()

        (
            cls.max_phase_amp,
            cls.max_phase_period,
            cls.max_phase_phase,
            cls.max_phase_base,
        ) = oscillation_fixtures.make_max_phase_parameters_fixture()

        # Time fixture.
        cls.time = oscillation_fixtures.make_time_fixture()

        # Custom function fixtures.
        cls.valid_custom_sine = staticmethod(
            oscillation_fixtures.make_valid_custom_sine_function_fixture()
        )

    def test_oscillating_sin_at_time_static_parameters(self) -> None:
        """Test oscillating_sin_at_time with static parameters."""
        result = _oscillation.oscillating_sin_at_time(
            amp=self.static_amp,
            period=self.static_period,
            phase=self.static_phase,
            base=self.static_base,
            time=self.time,
        )

        # Verify output is equal to base.
        npt.assert_allclose(result, self.static_base, rtol=1e-10, atol=1e-14)

    def test_oscillating_sin_at_time_phase_offset(self) -> None:
        """Test oscillating_sin_at_time with phase offset."""
        result = _oscillation.oscillating_sin_at_time(
            amp=self.phase_offset_amp,
            period=self.phase_offset_period,
            phase=self.phase_offset_phase,
            base=self.phase_offset_base,
            time=0.0,
        )

        # Verify that phase offset shifts the waveform. At t = 0.0, sine with 90.0
        # degree phase should equal the amplitude.
        npt.assert_allclose(result, self.phase_offset_amp, rtol=1e-10, atol=1e-14)

    def test_oscillating_sin_at_time_max_phase(self) -> None:
        """Test oscillating_sin_at_time with maximum phase."""
        result = _oscillation.oscillating_sin_at_time(
            amp=self.max_phase_amp,
            period=self.max_phase_period,
            phase=self.max_phase_phase,
            base=self.max_phase_base,
            time=0.0,
        )

        # Verify that max phase (180.0 degrees) inverts the waveform. At t = 0.0, sine
        # with 180.0 degree phase should be approximately 0.0.
        npt.assert_allclose(result, 0.0, rtol=1e-10, atol=1e-14)

    def test_oscillating_lin_at_time_static_parameters(self) -> None:
        """Test oscillating_lin_at_time with static parameters."""
        result = _oscillation.oscillating_lin_at_time(
            amp=self.static_amp,
            period=self.static_period,
            phase=self.static_phase,
            base=self.static_base,
            time=self.time,
        )

        # Verify output is equal to base.
        npt.assert_allclose(result, self.static_base, rtol=1e-10, atol=1e-14)

    def test_oscillating_lin_at_time_phase_offset(self) -> None:
        """Test oscillating_lin_at_time with phase offset."""
        result = _oscillation.oscillating_lin_at_time(
            amp=self.phase_offset_amp,
            period=self.phase_offset_period,
            phase=self.phase_offset_phase,
            base=self.phase_offset_base,
            time=0.0,
        )

        # Verify that phase offset shifts the waveform. At t = 0.0, triangular wave with
        # 90.0 degree phase should be near maximum.
        self.assertGreater(result, 0.5 * self.phase_offset_amp)

    def test_oscillating_custom_at_time_static_parameters(self) -> None:
        """Test oscillating_custom_at_time with static parameters."""
        result = _oscillation.oscillating_custom_at_time(
            amp=self.static_amp,
            period=self.static_period,
            base=self.static_base,
            phase=self.static_phase,
            time=self.time,
            custom_function=self.valid_custom_sine,
        )

        # Verify output is equal to base.
        npt.assert_allclose(result, self.static_base, rtol=1e-10, atol=1e-14)

    def test_oscillating_vector_at_time_matches_per_component_calls(self) -> None:
        """Test that oscillating_vector_at_time returns exactly the per component
        functions' values, for every kind of spacing and for mixed spacings."""
        amps = np.array([1.5, 2.0, 0.0], dtype=float)
        periods = np.array([0.1, 0.25, 0.0], dtype=float)
        phases = np.array([30.0, -90.0, 0.0], dtype=float)
        bases = np.array([0.2, -1.0, 3.0], dtype=float)
        time = 0.0375

        for spacings in (
            ("sine", "sine", "sine"),
            ("uniform", "uniform", "uniform"),
            (self.valid_custom_sine, self.valid_custom_sine, self.valid_custom_sine),
            ("sine", "uniform", self.valid_custom_sine),
        ):
            with self.subTest(spacings=spacings):
                expected = np.zeros(3, dtype=float)
                for dim, spacing in enumerate(spacings):
                    if spacing == "sine":
                        expected[dim] = _oscillation.oscillating_sin_at_time(
                            amp=amps[dim],
                            period=periods[dim],
                            phase=phases[dim],
                            base=bases[dim],
                            time=time,
                        )
                    elif spacing == "uniform":
                        expected[dim] = _oscillation.oscillating_lin_at_time(
                            amp=amps[dim],
                            period=periods[dim],
                            phase=phases[dim],
                            base=bases[dim],
                            time=time,
                        )
                    else:
                        assert callable(spacing)
                        expected[dim] = _oscillation.oscillating_custom_at_time(
                            amp=amps[dim],
                            period=periods[dim],
                            phase=phases[dim],
                            base=bases[dim],
                            time=time,
                            custom_function=spacing,
                        )

                result = _oscillation.oscillating_vector_at_time(
                    spacings=spacings,
                    amps=amps,
                    periods=periods,
                    phases=phases,
                    bases=bases,
                    time=time,
                )

                # The results must be bit for bit identical, not merely close.
                self.assertEqual(result.dtype, np.dtype(float))
                self.assertEqual(result.shape, (3,))
                self.assertTrue(np.array_equal(result, expected))

    def test_oscillating_vector_at_time_invalid_spacing(self) -> None:
        """Test that oscillating_vector_at_time raises a ValueError for an invalid
        spacing."""
        zeros = np.zeros(3, dtype=float)
        with self.assertRaises(ValueError):
            _oscillation.oscillating_vector_at_time(
                spacings=("sine", "invalid", "sine"),
                amps=zeros,
                periods=zeros,
                phases=zeros,
                bases=zeros,
                time=0.0,
            )
