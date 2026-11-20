"""This module contains classes to test Approximations and the free flight problems' use
of it."""

import logging
import unittest
from typing import Any

import numpy as np

import pterasoftware as ps

# noinspection PyProtectedMember
from pterasoftware import _serialization, approximations, problems
from tests.unit.fixtures import problem_fixtures


class TestApproximations(unittest.TestCase):
    """This is a class with functions to test Approximations."""

    def test_default_is_exact(self) -> None:
        """Test that the default Approximations turns every option off."""
        default = approximations.Approximations()
        self.assertFalse(default.adaptive_relaxation)
        self.assertEqual(default.coupling_relative_tolerance, 1e-6)
        self.assertFalse(default.any_enabled)
        self.assertEqual(default.describe(), "none")

    def test_exact_tolerance_matches_the_problems_module(self) -> None:
        """Test that the default coupling_relative_tolerance is exactly the free flight
        problems' own convergence tolerance, so the default changes nothing."""
        # noinspection PyProtectedMember
        self.assertEqual(
            approximations.Approximations().coupling_relative_tolerance,
            problems._SUBITERATION_RELATIVE_TOLERANCE,
        )

    def test_fast(self) -> None:
        """Test that fast() turns on adaptive_relaxation and sets the tolerance to
        1e-5."""
        fast = approximations.Approximations.fast()
        self.assertTrue(fast.adaptive_relaxation)
        self.assertEqual(fast.coupling_relative_tolerance, 1e-5)
        self.assertTrue(fast.any_enabled)
        self.assertEqual(
            fast.describe(), "adaptive_relaxation, coupling_relative_tolerance=1e-05"
        )

    def test_fast_coarse(self) -> None:
        """Test that fast_coarse() turns on adaptive_relaxation and sets the tolerance
        to 1e-4, the coarsest allowed."""
        coarse = approximations.Approximations.fast_coarse()
        self.assertTrue(coarse.adaptive_relaxation)
        self.assertEqual(coarse.coupling_relative_tolerance, 1e-4)
        # noinspection PyProtectedMember
        self.assertEqual(
            coarse.coupling_relative_tolerance,
            approximations._MAX_COUPLING_RELATIVE_TOLERANCE,
        )
        self.assertGreater(
            coarse.coupling_relative_tolerance,
            approximations.Approximations.fast().coupling_relative_tolerance,
        )

    def test_each_option_alone_is_enabled(self) -> None:
        """Test that any_enabled is True when either option alone is on."""
        self.assertTrue(
            approximations.Approximations(adaptive_relaxation=True).any_enabled
        )
        self.assertTrue(
            approximations.Approximations(coupling_relative_tolerance=1e-4).any_enabled
        )

    def test_adaptive_relaxation_validation(self) -> None:
        """Test that adaptive_relaxation must be a bool-like, and that a numpy bool is
        converted to a bool."""
        bad_value: Any = "yes"
        with self.assertRaises(TypeError):
            approximations.Approximations(adaptive_relaxation=bad_value)
        converted = approximations.Approximations(adaptive_relaxation=np.bool_(True))
        self.assertIs(converted.adaptive_relaxation, True)

    def test_coupling_relative_tolerance_validation(self) -> None:
        """Test that coupling_relative_tolerance must be a number in [1e-6, 1e-4], and
        that an int is converted to a float."""
        bad_value: Any = "1e-5"
        with self.assertRaises(TypeError):
            approximations.Approximations(coupling_relative_tolerance=bad_value)
        for value in (9.9e-7, 1.01e-4, 0.0, -1e-5, 1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    approximations.Approximations(coupling_relative_tolerance=value)
        for value in (1e-6, 1e-5, 1e-4):
            with self.subTest(value=value):
                tolerance = approximations.Approximations(
                    coupling_relative_tolerance=value
                ).coupling_relative_tolerance
                self.assertIsInstance(tolerance, float)
                self.assertEqual(tolerance, value)

    def test_attributes_are_immutable(self) -> None:
        """Test that the options can't be reassigned."""
        fast: Any = approximations.Approximations.fast()
        with self.assertRaises(AttributeError):
            fast.adaptive_relaxation = False
        with self.assertRaises(AttributeError):
            fast.coupling_relative_tolerance = 1e-6

    def test_slots(self) -> None:
        """Test that Approximations has no __dict__ and rejects new attributes."""
        default: Any = approximations.Approximations()
        self.assertFalse(hasattr(default, "__dict__"))
        with self.assertRaises(AttributeError):
            default.new_attribute = 1.0

    def test_available_through_the_package(self) -> None:
        """Test that Approximations is reachable as ps.approximations.Approximations."""
        self.assertIs(ps.approximations.Approximations, approximations.Approximations)


class TestRelaxationFactorHelpers(unittest.TestCase):
    """This is a class with functions to test the helpers that choose and carry the free
    flight sub-iteration's first relaxation factor."""

    def setUp(self) -> None:
        """Set up an exact and an adaptive Approximations."""
        self.exact = approximations.Approximations()
        self.adaptive = approximations.Approximations(adaptive_relaxation=True)

    def test_first_factor_without_adaptive_relaxation(self) -> None:
        """Test that without adaptive_relaxation the first factor is always the initial
        factor, even if a factor is passed as carried."""
        # noinspection PyProtectedMember
        for carried in (None, 0.9):
            with self.subTest(carried=carried):
                self.assertEqual(
                    problems._first_relaxation_factor(self.exact, carried),
                    problems._SUBITERATION_INITIAL_RELAXATION_FACTOR,
                )

    def test_first_factor_with_adaptive_relaxation(self) -> None:
        """Test that with adaptive_relaxation the first factor is the carried one, or
        the initial factor if none is carried."""
        # noinspection PyProtectedMember
        self.assertEqual(problems._first_relaxation_factor(self.adaptive, 0.93), 0.93)
        # noinspection PyProtectedMember
        self.assertEqual(
            problems._first_relaxation_factor(self.adaptive, None),
            problems._SUBITERATION_INITIAL_RELAXATION_FACTOR,
        )

    def test_carry_without_adaptive_relaxation(self) -> None:
        """Test that without adaptive_relaxation nothing is ever carried."""
        # noinspection PyProtectedMember
        self.assertIsNone(problems._relaxation_factor_to_carry(self.exact, 0.9, 0.8))

    def test_carry_in_bounds(self) -> None:
        """Test that a last Aitken factor within [0.1, 1.0] is carried, bounds
        included."""
        for factor in (0.1, 0.5, 0.93, 1.0):
            with self.subTest(factor=factor):
                # noinspection PyProtectedMember
                self.assertEqual(
                    problems._relaxation_factor_to_carry(self.adaptive, factor, None),
                    factor,
                )

    def test_out_of_bounds_restarts_at_the_initial_factor(self) -> None:
        """Test that a last Aitken factor outside [0.1, 1.0] isn't carried, so the next
        time step starts at the initial factor again."""
        for factor in (1.0000001, 1.5, 0.0999, 0.0, -0.4):
            with self.subTest(factor=factor):
                # noinspection PyProtectedMember
                carried = problems._relaxation_factor_to_carry(
                    self.adaptive, factor, 0.9
                )
                self.assertIsNone(carried)
                # noinspection PyProtectedMember
                self.assertEqual(
                    problems._first_relaxation_factor(self.adaptive, carried),
                    problems._SUBITERATION_INITIAL_RELAXATION_FACTOR,
                )

    def test_no_aitken_factor_keeps_the_carried_one(self) -> None:
        """Test that a time step that computed no Aitken factor (it converged on its
        first trial) keeps the factor it was carrying."""
        # noinspection PyProtectedMember
        self.assertEqual(
            problems._relaxation_factor_to_carry(self.adaptive, None, 0.87), 0.87
        )
        # noinspection PyProtectedMember
        self.assertIsNone(
            problems._relaxation_factor_to_carry(self.adaptive, None, None)
        )


class TestFreeFlightProblemApproximationsParameter(unittest.TestCase):
    """This is a class with functions to test the free flight problems' approximations
    parameter."""

    @staticmethod
    def _make_problem(
        value: Any,
    ) -> ps.problems.FreeFlightUnsteadyProblem:
        """Makes a basic FreeFlightUnsteadyProblem with the given approximations.

        :param value: The approximations argument.
        :return: The FreeFlightUnsteadyProblem.
        """
        fixture = problem_fixtures.make_basic_free_flight_unsteady_problem_fixture()
        # noinspection PyProtectedMember
        return ps.problems.FreeFlightUnsteadyProblem(
            movement=fixture._free_flight_movement,
            mass=fixture.mass,
            I_BP1_CgP1=fixture.I_BP1_CgP1,
            approximations=value,
        )

    def test_default_is_exact(self) -> None:
        """Test that the default is an exact Approximations and nothing is carried."""
        problem = problem_fixtures.make_basic_free_flight_unsteady_problem_fixture()
        self.assertIsInstance(problem.approximations, approximations.Approximations)
        self.assertFalse(problem.approximations.any_enabled)
        # noinspection PyProtectedMember
        self.assertIsNone(problem._carried_relaxation_factor)

    def test_stored(self) -> None:
        """Test that a given Approximations is stored as is."""
        fast = approximations.Approximations.fast()
        self.assertIs(self._make_problem(fast).approximations, fast)

    def test_type_validation(self) -> None:
        """Test that approximations must be an Approximations or None."""
        for value in (True, 1e-5, "fast", {"adaptive_relaxation": True}):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    self._make_problem(value)

    def test_logs_when_on(self) -> None:
        """Test that the problem logs which approximations are on, and logs nothing
        about them when none is."""
        with self.assertLogs("pterasoftware.problems", level=logging.INFO) as logs:
            self._make_problem(approximations.Approximations.fast())
        self.assertTrue(
            any("approximations are on" in message for message in logs.output)
        )
        with self.assertNoLogs("pterasoftware.problems", level=logging.INFO):
            self._make_problem(None)

    def test_hash(self) -> None:
        """Test that the approximations survive hashing: an exact and a fast() problem
        have different digests, and two exact problems the same."""
        exact_digest = _serialization.hash_object(self._make_problem(None))
        self.assertEqual(
            exact_digest, _serialization.hash_object(self._make_problem(None))
        )
        self.assertNotEqual(
            exact_digest,
            _serialization.hash_object(
                self._make_problem(approximations.Approximations.fast())
            ),
        )


if __name__ == "__main__":
    unittest.main()
