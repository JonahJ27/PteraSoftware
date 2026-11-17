"""This module contains classes to test the AeroelasticUnsteadyProblem class."""

import time
import unittest
from collections.abc import Callable, Sequence
from typing import Any, Literal
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
from scipy.integrate import solve_ivp

import pterasoftware as ps
from pterasoftware import flap_driven_aeroelastic_unsteady_problem
from pterasoftware import problems as problems_module
from tests.unit.fixtures import (
    aeroelastic_wing_movement_fixtures,
    movement_fixtures,
    problem_fixtures,
)


def _closed_form_spring_solution(
    spring_constant_rad: float,
    damping_constant_rad: float,
    torsional_inertia: float,
    theta0_rad: float,
    theta_derivative0_rad: float,
    moment: float,
    duration: float,
) -> tuple[float, float]:
    """Returns the exact solution of I * theta'' + c * theta' + k * theta = M for a
    constant M, after a duration.

    :param spring_constant_rad: The spring constant k (N*m/rad).
    :param damping_constant_rad: The damping constant c (N*m*s/rad).
    :param torsional_inertia: The torsional inertia I (kg*m^2).
    :param theta0_rad: The initial angle (radians).
    :param theta_derivative0_rad: The initial angle's time derivative (rad/s).
    :param moment: The constant moment M (N*m).
    :param duration: The time (s) after which to evaluate the solution.
    :return: A tuple of the angle (radians) and its time derivative (rad/s).
    """
    k = spring_constant_rad
    c = damping_constant_rad
    inertia = torsional_inertia
    # The solution is the equilibrium angle plus a homogeneous solution that starts at
    # the initial offset from it.
    equilibrium_rad = moment / k
    offset_rad = theta0_rad - equilibrium_rad
    rate_rad = theta_derivative0_rad
    discriminant = c**2 - 4.0 * k * inertia
    if discriminant > 0.0:
        # Overdamped: two real roots, computed in the cancellation free form.
        q = -0.5 * (c + np.sqrt(discriminant))
        fast_root = q / inertia
        slow_root = k / q
        fast_part = (rate_rad - slow_root * offset_rad) / (fast_root - slow_root)
        slow_part = offset_rad - fast_part
        fast_decay = np.exp(fast_root * duration)
        slow_decay = np.exp(slow_root * duration)
        theta_rad = equilibrium_rad + fast_part * fast_decay + slow_part * slow_decay
        theta_derivative_rad = (
            fast_part * fast_root * fast_decay + slow_part * slow_root * slow_decay
        )
    elif discriminant < 0.0:
        # Underdamped: a decaying oscillation.
        decay_rate = -c / (2.0 * inertia)
        frequency_rad = np.sqrt(-discriminant) / (2.0 * inertia)
        sine_part = (rate_rad - decay_rate * offset_rad) / frequency_rad
        envelope = np.exp(decay_rate * duration)
        cosine = np.cos(frequency_rad * duration)
        sine = np.sin(frequency_rad * duration)
        theta_rad = equilibrium_rad + envelope * (
            offset_rad * cosine + sine_part * sine
        )
        theta_derivative_rad = envelope * (
            decay_rate * (offset_rad * cosine + sine_part * sine)
            + frequency_rad * (sine_part * cosine - offset_rad * sine)
        )
    else:
        # Critically damped: a repeated real root.
        root = -c / (2.0 * inertia)
        linear_part = rate_rad - root * offset_rad
        decay = np.exp(root * duration)
        theta_rad = equilibrium_rad + (offset_rad + linear_part * duration) * decay
        theta_derivative_rad = (
            linear_part + root * (offset_rad + linear_part * duration)
        ) * decay
    return float(theta_rad), float(theta_derivative_rad)


def _closed_form_ramp_solution(
    spring_constant_rad: float,
    damping_constant_rad: float,
    torsional_inertia: float,
    theta0_rad: float,
    theta_derivative0_rad: float,
    start_moment: float,
    moment_slope: float,
    duration: float,
) -> tuple[float, float]:
    """Returns the exact solution of I * theta'' + c * theta' + k * theta = M(s) for a
    moment that changes linearly, M(s) = start_moment + moment_slope * s, after a
    duration.

    The solution is a particular solution that follows the ramp, theta_p(s) = M(s) / k
    - c * moment_slope / k^2, plus the unforced solution of the offset from it, which
    _closed_form_spring_solution gives.

    :param spring_constant_rad: The spring constant k (N*m/rad).
    :param damping_constant_rad: The damping constant c (N*m*s/rad).
    :param torsional_inertia: The torsional inertia I (kg*m^2).
    :param theta0_rad: The initial angle (radians).
    :param theta_derivative0_rad: The initial angle's time derivative (rad/s).
    :param start_moment: The moment at the start (N*m).
    :param moment_slope: The moment's rate of change (N*m/s).
    :param duration: The time (s) after which to evaluate the solution.
    :return: A tuple of the angle (radians) and its time derivative (rad/s).
    """
    k = spring_constant_rad
    c = damping_constant_rad
    lag_rad = c * moment_slope / k**2
    offset_rad, offset_rate_rad = _closed_form_spring_solution(
        k,
        c,
        torsional_inertia,
        theta0_rad - start_moment / k + lag_rad,
        theta_derivative0_rad - moment_slope / k,
        0.0,
        duration,
    )
    end_moment = start_moment + moment_slope * duration
    return (
        offset_rad + end_moment / k - lag_rad,
        offset_rate_rad + moment_slope / k,
    )


def _spring_method_errors(
    spring_constant_rad: float,
    damping_constant_rad: float,
    torsional_inertia: float,
    duration: float = 0.005,
) -> dict[str, tuple[float, float, float]]:
    """Solves one strip's time step with several methods, under a moment that ramps
    linearly so that the exact answer is known in closed form, and returns each method's
    error.

    The methods are the analytical solve (_spring_analytical_ode), the existing
    numerical solve (_spring_numerical_ode, which uses solve_ivp's default RK45), and
    solve_ivp's Radau and LSODA methods with the same tolerances as the existing one and
    the exact Jacobian.

    Each error is measured against the closed form solution and divided by a scale for
    the solution, instead of by the exact value itself, which can be near zero. The
    angle scale is the largest of the initial angle, the angle that the largest moment
    would hold, and the exact final angle. The rate scale is the largest of the initial
    rate, the exact final rate, and the angle scale times the strip's natural frequency
    sqrt(k / I), which is how fast the angle can change on its own.

    :param spring_constant_rad: The spring constant k (N*m/rad).
    :param damping_constant_rad: The damping constant c (N*m*s/rad).
    :param torsional_inertia: The torsional inertia I (kg*m^2).
    :param duration: The time step (s). The default is 0.005.
    :return: A dict mapping each method's name to a tuple of its angle error, its rate
        error (both as fractions of their scales), and its run time (s).
    """
    k = spring_constant_rad
    c = damping_constant_rad
    inertia = torsional_inertia
    start_time = 0.1
    theta0_rad, theta_derivative0_rad = 2e-6, -1e-4
    start_moment, moment_slope = 1e-3, 0.4
    forcing = lambda s: start_moment + moment_slope * (s - start_time)
    t = np.array([start_time, start_time + duration])
    expected = _closed_form_ramp_solution(
        k,
        c,
        inertia,
        theta0_rad,
        theta_derivative0_rad,
        start_moment,
        moment_slope,
        duration,
    )
    largest_moment = abs(start_moment) + abs(moment_slope) * duration
    angle_scale_rad = max(abs(theta0_rad), largest_moment / k, abs(expected[0]))
    rate_scale_rad = max(
        abs(theta_derivative0_rad),
        abs(expected[1]),
        angle_scale_rad * float(np.sqrt(k / inertia)),
    )

    def derivative(time_s: float, state: np.ndarray) -> list[float]:
        """Returns the time derivative of the state (theta, d(theta)/dt)."""
        return [state[1], (forcing(time_s) - c * state[1] - k * state[0]) / inertia]

    def solve_ivp_method(method: Literal["Radau", "LSODA"]) -> tuple[float, float]:
        """Integrates the time step with one of solve_ivp's implicit methods."""
        solution = solve_ivp(
            derivative,
            (float(t[0]), float(t[-1])),
            np.array([theta0_rad, theta_derivative0_rad], dtype=float),
            method=method,
            jac=[[0.0, 1.0], [-k / inertia, -c / inertia]],
            rtol=problems_module._SPRING_ODE_RELATIVE_TOLERANCE,
            atol=problems_module._SPRING_ODE_ABSOLUTE_TOLERANCE_RAD,
        )
        return float(solution.y[0, -1]), float(solution.y[1, -1])

    problem_class = ps.problems.AeroelasticUnsteadyProblem
    methods: dict[str, Callable[[], tuple[float, float]]] = {
        "analytical": lambda: problem_class._spring_analytical_ode(
            t, k, c, inertia, theta0_rad, theta_derivative0_rad, 0.0, forcing
        ),
        "RK45 (existing)": lambda: problem_class._spring_numerical_ode(
            t, k, c, inertia, theta0_rad, theta_derivative0_rad, 0.0, forcing
        ),
        "Radau": lambda: solve_ivp_method("Radau"),
        "LSODA": lambda: solve_ivp_method("LSODA"),
    }
    errors = {}
    for name, method in methods.items():
        start = time.perf_counter()
        actual = method()
        elapsed = time.perf_counter() - start
        errors[name] = (
            abs(actual[0] - expected[0]) / angle_scale_rad,
            abs(actual[1] - expected[1]) / rate_scale_rad,
            elapsed,
        )
    return errors


def _reference_spring_solution(
    spring_constant_rad: float,
    damping_constant_rad: float,
    torsional_inertia: float,
    theta0_rad: float,
    theta_derivative0_rad: float,
    forcing: Callable[[float], float],
    start_time: float,
    end_time: float,
    breakpoints: Sequence[float] = (),
) -> tuple[float, float]:
    """Integrates I * theta'' + c * theta' + k * theta = M(t) with an implicit solver
    at very tight tolerances, restarting it at each breakpoint.

    :param spring_constant_rad: The spring constant k (N*m/rad).
    :param damping_constant_rad: The damping constant c (N*m*s/rad).
    :param torsional_inertia: The torsional inertia I (kg*m^2).
    :param theta0_rad: The initial angle (radians).
    :param theta_derivative0_rad: The initial angle's time derivative (rad/s).
    :param forcing: The moment M as a function of time (N*m).
    :param start_time: The start time (s).
    :param end_time: The end time (s).
    :param breakpoints: The times (s) strictly between the start and end times where
        the forcing isn't smooth. The default is no breakpoints.
    :return: A tuple of the angle (radians) and its time derivative (rad/s) at the end
        time.
    """
    k = spring_constant_rad
    c = damping_constant_rad
    inertia = torsional_inertia
    jacobian = [[0.0, 1.0], [-k / inertia, -c / inertia]]

    def derivative(time_s: float, state: np.ndarray) -> list[float]:
        return [state[1], (forcing(time_s) - c * state[1] - k * state[0]) / inertia]

    times = [start_time, *breakpoints, end_time]
    state = np.array([theta0_rad, theta_derivative0_rad], dtype=float)
    for interval_start, interval_end in zip(times[:-1], times[1:]):
        solution = solve_ivp(
            derivative,
            (interval_start, interval_end),
            state,
            method="Radau",
            jac=jacobian,
            rtol=1e-12,
            atol=1e-20,
        )
        assert solution.success, solution.message
        state = solution.y[:, -1]
    return float(state[0]), float(state[1])


class TestAeroelasticUnsteadyProblem(unittest.TestCase):
    """This class contains unit tests for the AeroelasticUnsteadyProblem class."""

    def setUp(self) -> None:
        """Set up test fixtures for AeroelasticUnsteadyProblem tests."""
        self.problem = (
            problem_fixtures.make_basic_aeroelastic_unsteady_problem_fixture()
        )
        # The structural methods take the AeroelasticWingMovement whose prescribed
        # flapping parameters they read, so grab the fixture problem's first Wing's
        # movement.
        wing_movement = self.problem._aeroelastic_movement.airplane_movements[
            0
        ].wing_movements[0]
        assert isinstance(
            wing_movement,
            ps.movements.aeroelastic_wing_movement.AeroelasticWingMovement,
        )
        self.wing_movement = wing_movement

    def test_initialization_accepts_aeroelastic_movement(self) -> None:
        """Test that AeroelasticUnsteadyProblem accepts an AeroelasticMovement."""
        self.assertIsInstance(self.problem, ps.problems.AeroelasticUnsteadyProblem)

    def test_initialization_rejects_non_aeroelastic_movement(self) -> None:
        """Test that a non-AeroelasticMovement raises TypeError."""
        basic_movement: Any = movement_fixtures.make_basic_movement_fixture()
        with self.assertRaises(TypeError):
            ps.problems.AeroelasticUnsteadyProblem(
                movement=basic_movement,
                wing_density=0.01,
                spring_constant_rad=10.0,
                damping_constant_rad=0.5,
            )

    def test_num_steps(self) -> None:
        """Test that num_steps matches the movement's num_steps."""
        self.assertEqual(self.problem.num_steps, 3)

    def test_delta_time(self) -> None:
        """Test that delta_time matches the movement's delta_time."""
        self.assertAlmostEqual(self.problem.delta_time, 0.1)

    def test_initial_steady_problems_count(self) -> None:
        """Test that exactly one SteadyProblem exists at initialization (step 0)."""
        self.assertEqual(len(self.problem.steady_problems), 1)

    def test_wing_density_stored(self) -> None:
        """Test that wing_density is stored correctly."""
        self.assertAlmostEqual(self.problem.wing_density, 0.01)

    def test_spring_constant_rad_stored(self) -> None:
        """Test that spring_constant_rad is stored correctly."""
        self.assertAlmostEqual(self.problem.spring_constant_rad, 10.0)

    def test_damping_constant_rad_stored(self) -> None:
        """Test that damping_constant_rad is stored correctly."""
        self.assertAlmostEqual(self.problem.damping_constant_rad, 0.5)

    def test_calculate_mass_matrix_shape(self) -> None:
        """Test that _calculate_mass_matrix returns an array with shape
        (num_chordwise_panels, num_spanwise_panels, 3)."""
        wing = self.problem.steady_problems[0].airplanes[0].wings[0]
        mass_matrix = self.problem._calculate_mass_matrix(wing)
        self.assertEqual(len(mass_matrix.shape), 3)
        self.assertEqual(mass_matrix.shape[0], wing.num_chordwise_panels)
        self.assertEqual(mass_matrix.shape[1], wing.num_spanwise_panels)
        self.assertEqual(mass_matrix.shape[2], 3)

    def test_calculate_mass_matrix_non_negative(self) -> None:
        """Test that all entries in the mass matrix are non-negative."""
        wing = self.problem.steady_problems[0].airplanes[0].wings[0]
        mass_matrix = self.problem._calculate_mass_matrix(wing)
        self.assertTrue(np.all(mass_matrix >= 0.0))

    def test_calculate_mass_matrix_components_equal(self) -> None:
        """Test that all three spatial components of the mass matrix are equal."""
        wing = self.problem.steady_problems[0].airplanes[0].wings[0]
        mass_matrix = self.problem._calculate_mass_matrix(wing)
        np.testing.assert_array_equal(mass_matrix[:, :, 0], mass_matrix[:, :, 1])
        np.testing.assert_array_equal(mass_matrix[:, :, 1], mass_matrix[:, :, 2])

    def test_calculate_spring_moments_accumulates_state_across_steps(self) -> None:
        """Test that a strip's spring-damper ODE is re-seeded from its own state at the
        end of the previous time step, so a constant aerodynamic moment drives the
        strip's deformation angle y component to accumulate across steps toward the
        static equilibrium M / k.

        Uses a single-strip Wing in a slow, overdamped regime where the one-step
        response from rest is a small fraction of M / k. If a strip's state were
        discarded between steps (a restart from rest every step), the deformation angle
        y component would freeze at that one-step response instead of converging.
        """
        problem = ps.problems.AeroelasticUnsteadyProblem(
            movement=movement_fixtures.make_basic_aeroelastic_movement_fixture(),
            wing_density=0.01,
            spring_constant_rad=10.0,
            damping_constant_rad=20.0,
        )
        wing = problem.steady_problems[0].airplanes[0].wings[0]
        num_spanwise_panels = wing.num_spanwise_panels
        num_chordwise_panels = wing.num_chordwise_panels
        assert num_spanwise_panels == 1
        static_wing_movement = (
            aeroelastic_wing_movement_fixtures.make_static_aeroelastic_wing_movement_fixture()
        )

        # Choose the per-entry mass so the strip's torsional inertia, I = (1 / 2) * mass
        # * L^2 with mass summed over the strip's mass-matrix entries, is exactly 1.0.
        # With spring_constant_rad = 10.0 and damping_constant_rad = 20.0, the ODE is
        # then overdamped (c^2 > 4 * k * I) with a slowest decay rate of about 0.5 per
        # second, so it is far from settled within one 0.1 s time step.
        L = (wing.wing_cross_sections[0].chord + wing.wing_cross_sections[1].chord) / 2
        num_mass_entries = num_chordwise_panels * num_spanwise_panels * 3
        mass_matrix = np.full(
            (num_chordwise_panels, num_spanwise_panels, 3),
            2.0 / (num_mass_entries * L**2),
        )

        # Apply a constant aerodynamic moment y component (in the first Airplane's
        # geometry axes, relative to the strip's leading edge point), summing to 2.0 N*m
        # over the strip's chordwise panels, so the static deformation angle y component
        # is M / k = 0.2 rad.
        aeroMoments_GP1_Slep = np.zeros(
            (num_chordwise_panels, num_spanwise_panels, 3), dtype=float
        )
        aeroMoments_GP1_Slep[:, 0, 1] = 2.0 / num_chordwise_panels
        static_theta_rad = 2.0 / problem.spring_constant_rad

        # March the structural solve, replicating _record_structural_state's
        # post-discard recording after each time step.
        theta_history_rad = []
        for step in range(problem.step_discards + 1, problem.step_discards + 61):
            (
                newDeformationAnglesYRad_Wcsp_to_Wcs_ixyz,
                newDeformationAnglesDerivativeYRad_Wcsp_to_Wcs_ixyz,
            ) = problem._calculate_spring_moments(
                num_spanwise_panels=num_spanwise_panels,
                wing=wing,
                mass_matrix=mass_matrix,
                aeroMoments_GP1_Slep=aeroMoments_GP1_Slep,
                step=step,
                wing_idx=0,
                wing_movement=static_wing_movement,
            )
            problem.listDeformationAnglesYRad_Wcsp_to_Wcs_ixyz[0].append(
                np.array(newDeformationAnglesYRad_Wcsp_to_Wcs_ixyz, dtype=float)
            )
            problem._listDeformationAnglesDerivativeYRad_Wcsp_to_Wcs_ixyz[0].append(
                np.array(
                    newDeformationAnglesDerivativeYRad_Wcsp_to_Wcs_ixyz, dtype=float
                )
            )
            theta_history_rad.append(
                float(newDeformationAnglesYRad_Wcsp_to_Wcs_ixyz[1])
            )

        # The regime check: the one-step response must be well below the static
        # equilibrium, or freezing and converging would be indistinguishable.
        self.assertLess(theta_history_rad[0], 0.5 * static_theta_rad)
        # The deformation angle y component must accumulate monotonically, as an
        # overdamped rise from rest.
        for earlier, later in zip(theta_history_rad[:10], theta_history_rad[1:11]):
            self.assertGreater(later, earlier)
        # The deformation angle y component must approach the static equilibrium.
        self.assertAlmostEqual(
            theta_history_rad[-1], static_theta_rad, delta=0.1 * static_theta_rad
        )

    def test_record_structural_state_records_new_state_after_discard_window(
        self,
    ) -> None:
        """Test that _record_structural_state appends the new state to both time series
        as fresh arrays when the time step is past the discard window."""
        problem = self.problem
        wing_idx = 0
        num_wing_cross_sections = problem.listDeformationAnglesYRad_Wcsp_to_Wcs_ixyz[
            wing_idx
        ][-1].shape[0]
        newDeformationAnglesYRad_Wcsp_to_Wcs_ixyz = np.full(
            num_wing_cross_sections, 0.25, dtype=float
        )
        newDeformationAnglesDerivativeYRad_Wcsp_to_Wcs_ixyz = np.full(
            num_wing_cross_sections, -0.5, dtype=float
        )

        problem._record_structural_state(
            step=problem.step_discards + 1,
            newDeformationAnglesYRad_Wcsp_to_Wcs_ixyz=(
                newDeformationAnglesYRad_Wcsp_to_Wcs_ixyz
            ),
            newDeformationAnglesDerivativeYRad_Wcsp_to_Wcs_ixyz=(
                newDeformationAnglesDerivativeYRad_Wcsp_to_Wcs_ixyz
            ),
            wing_idx=wing_idx,
        )

        deformationAnglesYRad_Wcsp_to_Wcs_ixyz = (
            problem.listDeformationAnglesYRad_Wcsp_to_Wcs_ixyz[wing_idx]
        )
        deformationAnglesDerivativeYRad_Wcsp_to_Wcs_ixyz = (
            problem._listDeformationAnglesDerivativeYRad_Wcsp_to_Wcs_ixyz[wing_idx]
        )
        self.assertEqual(len(deformationAnglesYRad_Wcsp_to_Wcs_ixyz), 2)
        self.assertEqual(len(deformationAnglesDerivativeYRad_Wcsp_to_Wcs_ixyz), 2)
        np.testing.assert_array_equal(
            deformationAnglesYRad_Wcsp_to_Wcs_ixyz[-1],
            newDeformationAnglesYRad_Wcsp_to_Wcs_ixyz,
        )
        np.testing.assert_array_equal(
            deformationAnglesDerivativeYRad_Wcsp_to_Wcs_ixyz[-1],
            newDeformationAnglesDerivativeYRad_Wcsp_to_Wcs_ixyz,
        )
        # The recorded entries are fresh arrays, not aliases of the caller's arrays.
        self.assertIsNot(
            deformationAnglesYRad_Wcsp_to_Wcs_ixyz[-1],
            newDeformationAnglesYRad_Wcsp_to_Wcs_ixyz,
        )
        self.assertIsNot(
            deformationAnglesDerivativeYRad_Wcsp_to_Wcs_ixyz[-1],
            newDeformationAnglesDerivativeYRad_Wcsp_to_Wcs_ixyz,
        )

    def test_record_structural_state_holds_previous_state_during_discard_window(
        self,
    ) -> None:
        """Test that _record_structural_state ignores the new state during the discard
        window, appending a copy of the previous entry to both time series."""
        problem = self.problem
        wing_idx = 0
        num_wing_cross_sections = problem.listDeformationAnglesYRad_Wcsp_to_Wcs_ixyz[
            wing_idx
        ][-1].shape[0]
        newDeformationAnglesYRad_Wcsp_to_Wcs_ixyz = np.full(
            num_wing_cross_sections, 0.25, dtype=float
        )
        newDeformationAnglesDerivativeYRad_Wcsp_to_Wcs_ixyz = np.full(
            num_wing_cross_sections, -0.5, dtype=float
        )

        problem._record_structural_state(
            step=problem.step_discards,
            newDeformationAnglesYRad_Wcsp_to_Wcs_ixyz=(
                newDeformationAnglesYRad_Wcsp_to_Wcs_ixyz
            ),
            newDeformationAnglesDerivativeYRad_Wcsp_to_Wcs_ixyz=(
                newDeformationAnglesDerivativeYRad_Wcsp_to_Wcs_ixyz
            ),
            wing_idx=wing_idx,
        )

        deformationAnglesYRad_Wcsp_to_Wcs_ixyz = (
            problem.listDeformationAnglesYRad_Wcsp_to_Wcs_ixyz[wing_idx]
        )
        deformationAnglesDerivativeYRad_Wcsp_to_Wcs_ixyz = (
            problem._listDeformationAnglesDerivativeYRad_Wcsp_to_Wcs_ixyz[wing_idx]
        )
        self.assertEqual(len(deformationAnglesYRad_Wcsp_to_Wcs_ixyz), 2)
        self.assertEqual(len(deformationAnglesDerivativeYRad_Wcsp_to_Wcs_ixyz), 2)
        # Both series hold the seed (zero) state, not the passed values.
        np.testing.assert_array_equal(
            deformationAnglesYRad_Wcsp_to_Wcs_ixyz[-1],
            np.zeros(num_wing_cross_sections, dtype=float),
        )
        np.testing.assert_array_equal(
            deformationAnglesDerivativeYRad_Wcsp_to_Wcs_ixyz[-1],
            np.zeros(num_wing_cross_sections, dtype=float),
        )
        # The held entries are copies, not aliases of the previous entries.
        self.assertIsNot(
            deformationAnglesYRad_Wcsp_to_Wcs_ixyz[-1],
            deformationAnglesYRad_Wcsp_to_Wcs_ixyz[-2],
        )
        self.assertIsNot(
            deformationAnglesDerivativeYRad_Wcsp_to_Wcs_ixyz[-1],
            deformationAnglesDerivativeYRad_Wcsp_to_Wcs_ixyz[-2],
        )

    def test_spring_numerical_ode_zero_initial_zero_forces_theta_stays_zero(
        self,
    ) -> None:
        """Test that with zero initial conditions and zero external forces, theta
        remains zero."""
        t = np.array([0.0, 0.05])
        zero_moment_func = lambda time: 0.0
        theta_rad, theta_derivative_rad = self.problem._spring_numerical_ode(
            t,
            spring_constant_rad=10.0,
            damping_constant_rad=0.5,
            torsional_inertia=1.0,
            theta0_rad=0.0,
            theta_derivative0_rad=0.0,
            aero_moment=0.0,
            inertial_moment_func=zero_moment_func,
        )
        self.assertAlmostEqual(theta_rad, 0.0, places=6)
        self.assertAlmostEqual(theta_derivative_rad, 0.0, places=6)

    def test_spring_numerical_ode_returns_floats(self) -> None:
        """Test that _spring_numerical_ode returns Python floats."""
        t = np.array([0.0, 0.1])
        zero_moment_func = lambda time: 0.0
        theta_rad, theta_derivative_rad = self.problem._spring_numerical_ode(
            t,
            spring_constant_rad=10.0,
            damping_constant_rad=0.5,
            torsional_inertia=1.0,
            theta0_rad=0.0,
            theta_derivative0_rad=0.0,
            aero_moment=0.0,
            inertial_moment_func=zero_moment_func,
        )
        self.assertIsInstance(theta_rad, float)
        self.assertIsInstance(theta_derivative_rad, float)

    def test_spring_numerical_ode_spring_restores_toward_zero(self) -> None:
        """Test that a positive initial displacement decays toward zero with positive
        spring constant and damping."""
        t = np.array([0.0, 0.5, 1.0])
        zero_moment_func = lambda time: 0.0
        theta0_rad = 1.0
        theta_rad, theta_derivative_rad = self.problem._spring_numerical_ode(
            t,
            spring_constant_rad=10.0,
            damping_constant_rad=2.0,
            torsional_inertia=1.0,
            theta0_rad=theta0_rad,
            theta_derivative0_rad=0.0,
            aero_moment=0.0,
            inertial_moment_func=zero_moment_func,
        )
        # With damping and spring, displacement should decrease in magnitude.
        self.assertLess(abs(theta_rad), abs(theta0_rad))

    def test_generate_inertial_moment_function_returns_callable(self) -> None:
        """Test that _generate_inertial_moment_function returns a callable."""
        moment_func = self.problem._generate_inertial_moment_function(
            flapping_axis_inertia=1.0, wing_movement=self.wing_movement
        )
        self.assertTrue(callable(moment_func))

    def test_generate_inertial_moment_function_returns_float_at_zero(self) -> None:
        """Test that the moment function returned by _generate_inertial_moment_function
        returns a numeric value when evaluated at time zero."""
        moment_func = self.problem._generate_inertial_moment_function(
            flapping_axis_inertia=1.0, wing_movement=self.wing_movement
        )
        result = moment_func(0.0)
        self.assertIsInstance(result, (float, np.floating))

    def test_generate_inertial_moment_function_scales_with_flapping_axis_inertia(
        self,
    ) -> None:
        """Test that the moment function scales linearly with flapping_axis_inertia."""
        moment_func_1 = self.problem._generate_inertial_moment_function(
            flapping_axis_inertia=1.0, wing_movement=self.wing_movement
        )
        moment_func_2 = self.problem._generate_inertial_moment_function(
            flapping_axis_inertia=2.0, wing_movement=self.wing_movement
        )
        t_eval = 0.25
        self.assertAlmostEqual(
            moment_func_2(t_eval), 2.0 * moment_func_1(t_eval), places=10
        )

    def test_generate_inertial_moment_function_sine_matches_analytic_value(
        self,
    ) -> None:
        """Test that the sinusoidal-spacing moment equals -I * b^2 * sin(b * t + h) * A
        with the amplitude converted from degrees to radians so the moment is in SI
        N*m."""
        wing_movement = self.wing_movement
        amp_deg = wing_movement.ampAngles_Gs_to_Wn_ixyz[0]
        period = wing_movement.periodAngles_Gs_to_Wn_ixyz[0]
        phase_deg = wing_movement.phaseAngles_Gs_to_Wn_ixyz[0]

        flapping_axis_inertia = 3.0
        t_eval = 0.3
        moment_func = self.problem._generate_inertial_moment_function(
            flapping_axis_inertia=flapping_axis_inertia, wing_movement=wing_movement
        )

        b_rad = 2.0 * np.pi / period
        h_rad = np.deg2rad(phase_deg)
        expected = (
            -1.0
            * b_rad**2
            * np.sin(b_rad * t_eval + h_rad)
            * np.deg2rad(amp_deg)
            * flapping_axis_inertia
        )
        self.assertAlmostEqual(moment_func(t_eval), expected, places=10)

    def test_generate_inertial_moment_function_static_motion_returns_zero(self) -> None:
        """Test that a motion-off wing movement (zero flapping amplitude and period)
        produces an identically zero, finite inertial moment at every time.

        A wing with no prescribed flapping applies no inertial moment, so the returned
        function must evaluate to exactly 0.0 rather than to NaN from the 2 * pi /
        period frequency computation with a zero period.
        """
        static_wing_movement = (
            aeroelastic_wing_movement_fixtures.make_static_aeroelastic_wing_movement_fixture()
        )
        moment_func = self.problem._generate_inertial_moment_function(
            flapping_axis_inertia=1.0, wing_movement=static_wing_movement
        )
        for t_eval in (0.0, 0.1, 0.5):
            result = moment_func(t_eval)
            self.assertTrue(
                np.isfinite(result),
                msg=f"The inertial moment at time {t_eval} is {result}, which is "
                f"not finite.",
            )
            self.assertAlmostEqual(float(result), 0.0, places=12)

    def test_generate_inertial_moment_function_uniform_spacing_raises(self) -> None:
        """Test that _generate_inertial_moment_function raises ValueError when the wing
        motion spacing is "uniform" (sawtooth), which is not differentiable."""
        wing_movement = self.wing_movement
        with patch.object(
            type(wing_movement),
            "spacingAngles_Gs_to_Wn_ixyz",
            new_callable=PropertyMock,
            return_value=("uniform", "sine", "sine"),
        ):
            with self.assertRaises(ValueError):
                self.problem._generate_inertial_moment_function(
                    flapping_axis_inertia=1.0, wing_movement=wing_movement
                )

    def test_generate_inertial_moment_function_callable_spacing_with_derivative(
        self,
    ) -> None:
        """Test that _generate_inertial_moment_function uses the
        AeroelasticWingMovement's second derivative when the spacing is a custom
        callable."""
        wing_movement = self.wing_movement
        with (
            patch.object(
                type(wing_movement),
                "spacingAngles_Gs_to_Wn_ixyz",
                new_callable=PropertyMock,
                return_value=(lambda t: np.sin(t), "sine", "sine"),
            ),
            patch.object(
                type(wing_movement),
                "spacingAnglesSecondDerivative_Gs_to_Wn_ixyz",
                new_callable=PropertyMock,
                return_value=(lambda t: -np.sin(t), None, None),
            ),
        ):
            moment_func = self.problem._generate_inertial_moment_function(
                flapping_axis_inertia=2.0, wing_movement=wing_movement
            )
            self.assertTrue(callable(moment_func))
            result = moment_func(0.5)
            # The amplitude is stored in degrees but the moment (N*m) must be in SI
            # units, so the second derivative is scaled by np.deg2rad(amp).
            amp = wing_movement.ampAngles_Gs_to_Wn_ixyz[0]
            expected = np.deg2rad(amp) * -np.sin(0.5) * 2.0
            self.assertAlmostEqual(result, expected, places=10)


class TestRecordNullStepForWing(unittest.TestCase):
    """This class contains unit tests for the _record_null_step_for_wing method and the
    standard WingMovement code path in _calculate_wing_deformation."""

    def setUp(self) -> None:
        """Set up a fresh AeroelasticUnsteadyProblem for each test."""
        self.problem_aero = (
            problem_fixtures.make_basic_aeroelastic_unsteady_problem_fixture()
        )
        self.problem_std = (
            problem_fixtures.make_aeroelastic_unsteady_problem_with_standard_wing_fixture()
        )

    def test_record_null_step_for_wing_appends_one_entry_to_each_time_series(
        self,
    ) -> None:
        """Test that _record_null_step_for_wing appends exactly one entry to each per-
        wing time series list."""
        wing_idx = 0
        problem = self.problem_aero
        num_previous_states = len(
            problem.listDeformationAnglesYRad_Wcsp_to_Wcs_ixyz[wing_idx]
        )
        # The two time series are seeded together and recorded together, so their
        # lengths are always equal.
        self.assertEqual(
            len(
                problem._listDeformationAnglesDerivativeYRad_Wcsp_to_Wcs_ixyz[wing_idx]
            ),
            num_previous_states,
        )

        problem._record_null_step_for_wing(wing_idx)

        self.assertEqual(
            len(problem.listDeformationAnglesYRad_Wcsp_to_Wcs_ixyz[wing_idx]),
            num_previous_states + 1,
        )
        self.assertEqual(
            len(
                problem._listDeformationAnglesDerivativeYRad_Wcsp_to_Wcs_ixyz[wing_idx]
            ),
            num_previous_states + 1,
        )

    def test_record_null_step_for_wing_entry_shape_and_values(self) -> None:
        """Test that _record_null_step_for_wing appends zero-valued entries, with one
        element per WingCrossSection, to listDeformationAnglesYRad_Wcsp_to_Wcs_ixyz and
        _listDeformationAnglesDerivativeYRad_Wcsp_to_Wcs_ixyz."""
        wing_idx = 0
        num_wing_cross_sections = (
            self.problem_aero.listDeformationAnglesYRad_Wcsp_to_Wcs_ixyz[wing_idx][
                -1
            ].shape[0]
        )

        self.problem_aero._record_null_step_for_wing(wing_idx)

        lastDeformationAnglesYRad_Wcsp_to_Wcs_ixyz = (
            self.problem_aero.listDeformationAnglesYRad_Wcsp_to_Wcs_ixyz[wing_idx][-1]
        )
        lastDeformationAnglesDerivativeYRad_Wcsp_to_Wcs_ixyz = (
            self.problem_aero._listDeformationAnglesDerivativeYRad_Wcsp_to_Wcs_ixyz[
                wing_idx
            ][-1]
        )
        expected_shape = (num_wing_cross_sections,)

        self.assertEqual(
            lastDeformationAnglesYRad_Wcsp_to_Wcs_ixyz.shape, expected_shape
        )
        self.assertEqual(
            lastDeformationAnglesDerivativeYRad_Wcsp_to_Wcs_ixyz.shape, expected_shape
        )
        np.testing.assert_array_equal(
            lastDeformationAnglesYRad_Wcsp_to_Wcs_ixyz, np.zeros(expected_shape)
        )
        np.testing.assert_array_equal(
            lastDeformationAnglesDerivativeYRad_Wcsp_to_Wcs_ixyz,
            np.zeros(expected_shape),
        )

    def test_calculate_wing_deformation_returns_none_for_standard_wing_movement(
        self,
    ) -> None:
        """Test that _calculate_wing_deformation returns None for a Wing backed by a
        standard WingMovement (the else branch)."""
        mock_solver = MagicMock()

        results = self.problem_std._calculate_wing_deformation(
            solver=mock_solver, step=0
        )

        self.assertEqual(len(results), 1)
        self.assertIsNone(results[0])

    def test_calculate_wing_deformation_appends_entries_for_standard_wing_movement(
        self,
    ) -> None:
        """Test that _calculate_wing_deformation appends to the time series lists when a
        Wing is backed by a standard WingMovement."""
        mock_solver = MagicMock()
        wing_idx = 0

        self.problem_std._calculate_wing_deformation(solver=mock_solver, step=0)

        problem = self.problem_std
        # The two structural state time series are seeded with an initial-state entry at
        # construction, so one recorded step brings them to two entries.
        self.assertEqual(
            len(problem.listDeformationAnglesYRad_Wcsp_to_Wcs_ixyz[wing_idx]), 2
        )
        self.assertEqual(
            len(
                problem._listDeformationAnglesDerivativeYRad_Wcsp_to_Wcs_ixyz[wing_idx]
            ),
            2,
        )


class TestUseAnalyticalSpringSolver(unittest.TestCase):
    """This class contains unit tests for AeroelasticUnsteadyProblem's
    use_analytical_spring_solver flag."""

    def setUp(self) -> None:
        """Set up test fixtures for the flag tests."""
        self.problem = (
            problem_fixtures.make_basic_aeroelastic_unsteady_problem_fixture()
        )
        wing_movement = self.problem._aeroelastic_movement.airplane_movements[
            0
        ].wing_movements[0]
        assert isinstance(
            wing_movement,
            ps.movements.aeroelastic_wing_movement.AeroelasticWingMovement,
        )
        self.wing_movement = wing_movement

    def _make_problem(self, flag: Any) -> ps.problems.AeroelasticUnsteadyProblem:
        """Makes the basic problem with a given value for the flag.

        :param flag: The value to pass as use_analytical_spring_solver.
        :return: The problem.
        """
        return ps.problems.AeroelasticUnsteadyProblem(
            movement=movement_fixtures.make_basic_aeroelastic_movement_fixture(),
            wing_density=0.01,
            spring_constant_rad=10.0,
            damping_constant_rad=0.5,
            use_analytical_spring_solver=flag,
        )

    def _spring_step(self, problem: ps.problems.AeroelasticUnsteadyProblem) -> Any:
        """Runs one strip's spring step through _calculate_torsional_spring_moment.

        :param problem: The problem whose solver choice is used.
        :return: The new angle and its time derivative.
        """
        return problem._calculate_torsional_spring_moment(
            dt=0.1,
            torsional_inertia=1.0,
            lastDeformationAngleYRad_Wcsp_to_Wcs_ixyz=0.01,
            lastDeformationAngleDerivativeYRad_Wcsp_to_Wcs_ixyz=0.2,
            aeroMomentY_GP1_Slep=0.3,
            step=1,
            flapping_axis_inertia=0.5,
            wing_movement=self.wing_movement,
        )

    def test_default_is_false(self) -> None:
        """Test that the flag defaults to False, which keeps solve_ivp."""
        self.assertFalse(self.problem.use_analytical_spring_solver)

    def test_true_is_stored(self) -> None:
        """Test that a True flag is stored."""
        self.assertTrue(self._make_problem(True).use_analytical_spring_solver)

    def test_numpy_bool_is_converted_to_bool(self) -> None:
        """Test that a numpy bool is accepted and stored as a Python bool."""
        problem = self._make_problem(np.True_)
        self.assertIs(problem.use_analytical_spring_solver, True)

    def test_non_bool_raises_type_error(self) -> None:
        """Test that a value that is not a bool raises a TypeError."""
        for bad_value in (1, "True", None, 1.0):
            with self.subTest(bad_value=bad_value):
                with self.assertRaises(TypeError):
                    self._make_problem(bad_value)

    def test_flag_is_read_only(self) -> None:
        """Test that the flag can't be reassigned."""
        with self.assertRaises(AttributeError):
            setattr(self.problem, "use_analytical_spring_solver", True)

    def test_false_uses_numerical_solver(self) -> None:
        """Test that a False flag routes the spring step to _spring_numerical_ode."""
        problem_class = ps.problems.AeroelasticUnsteadyProblem
        with (
            patch.object(
                problem_class,
                "_spring_numerical_ode",
                wraps=problem_class._spring_numerical_ode,
            ) as numerical,
            patch.object(
                problem_class,
                "_spring_analytical_ode",
                wraps=problem_class._spring_analytical_ode,
            ) as analytical,
        ):
            self._spring_step(self._make_problem(False))
        numerical.assert_called_once()
        analytical.assert_not_called()

    def test_true_uses_analytical_solver(self) -> None:
        """Test that a True flag routes the spring step to _spring_analytical_ode."""
        problem_class = ps.problems.AeroelasticUnsteadyProblem
        with (
            patch.object(
                problem_class,
                "_spring_numerical_ode",
                wraps=problem_class._spring_numerical_ode,
            ) as numerical,
            patch.object(
                problem_class,
                "_spring_analytical_ode",
                wraps=problem_class._spring_analytical_ode,
            ) as analytical,
        ):
            self._spring_step(self._make_problem(True))
        analytical.assert_called_once()
        numerical.assert_not_called()

    def test_both_solvers_agree_through_the_spring_step(self) -> None:
        """Test that the two solvers give the same spring step for the fixture's
        sinusoidally flapping Wing, within the numerical solver's tolerance."""
        numerical = self._spring_step(self._make_problem(False))
        analytical = self._spring_step(self._make_problem(True))
        np.testing.assert_allclose(analytical, numerical, rtol=1e-5, atol=1e-9)

    def test_sample_count_matches_the_flap_acceleration_sample_count(self) -> None:
        """Test that every FlapDrive acceleration sample time is also an analytical
        solver sample time, which is what makes the solve exact for FlapDrives."""
        spring_samples = problems_module._SPRING_FORCING_SAMPLES
        flap_samples = (
            flap_driven_aeroelastic_unsteady_problem._FLAP_ACCELERATION_SAMPLES
        )
        self.assertEqual((spring_samples - 1) % (flap_samples - 1), 0)


class TestSpringAnalyticalOde(unittest.TestCase):
    """This class contains unit tests for AeroelasticUnsteadyProblem's
    _spring_analytical_ode, comparing its accuracy against closed form solutions and
    tightly converged numerical references."""

    # These are the strip inertias (kg*m^2) the tests sweep, from a heavy strip to the
    # light, very stiff strips of a small membrane wing.
    _INERTIAS = (10.0, 1e-1, 1e-3, 1e-5, 1e-7, 1e-9)

    # This is the time step (s) most tests use, about that of a 5 Hz flap with 40 time
    # steps per flap.
    _DT = 0.005

    @staticmethod
    def _solve(
        spring_constant_rad: float,
        damping_constant_rad: float,
        torsional_inertia: float,
        theta0_rad: float,
        theta_derivative0_rad: float,
        aero_moment: float,
        inertial_moment_func: Callable[[float], float],
        start_time: float = 0.1,
        duration: float = 0.005,
    ) -> tuple[float, float]:
        """Calls _spring_analytical_ode over one time step.

        :return: The new angle (radians) and its time derivative (rad/s).
        """
        return ps.problems.AeroelasticUnsteadyProblem._spring_analytical_ode(
            np.array([start_time, start_time + duration]),
            spring_constant_rad,
            damping_constant_rad,
            torsional_inertia,
            theta0_rad,
            theta_derivative0_rad,
            aero_moment,
            inertial_moment_func,
        )

    def _assert_state_close(
        self,
        actual: tuple[float, float],
        expected: tuple[float, float],
        angle_scale_rad: float,
        rate_scale_rad: float,
        rtol: float,
    ) -> None:
        """Asserts that two states agree, relative to the scales of the solution.

        :param actual: The angle (radians) and its time derivative (rad/s) to check.
        :param expected: The reference angle and time derivative.
        :param angle_scale_rad: The angle's typical size (radians).
        :param rate_scale_rad: The time derivative's typical size (rad/s).
        :param rtol: The allowed error as a fraction of each scale.
        """
        self.assertLessEqual(
            abs(actual[0] - expected[0]),
            rtol * angle_scale_rad,
            msg=(actual, expected),
        )
        self.assertLessEqual(
            abs(actual[1] - expected[1]),
            rtol * rate_scale_rad,
            msg=(actual, expected),
        )

    @staticmethod
    def _precision(damping_constant_rad: float, inertia: float, dt: float) -> float:
        """Returns the relative precision to expect from the analytical solve.

        The matrix exponential's rounding error grows roughly in proportion to c / I *
        dt, so light, strongly overdamped strips are solved less precisely (about 8e-8
        for I = 1e-9 kg*m^2 with c = 1000.0 and dt = 0.005 s). Everywhere else, the
        precision is near rounding level, so the floor is 1e-9.

        :param damping_constant_rad: The damping constant c (N*m*s/rad).
        :param inertia: The torsional inertia I (kg*m^2).
        :param dt: The time step (s).
        :return: The allowed error as a fraction of the solution's scale.
        """
        return max(1e-9, 2e-16 * damping_constant_rad / inertia * dt)

    @staticmethod
    def _damping_cases(inertia: float) -> dict[str, tuple[float, float]]:
        """Returns overdamped, critically damped, and lightly damped (k, c) pairs.

        :param inertia: The strip's torsional inertia (kg*m^2).
        :return: A dict mapping each case's name to its spring and damping constants.
        """
        k = 1000.0
        return {
            "overdamped": (k, 1000.0),
            "critically damped": (k, float(2.0 * np.sqrt(k * inertia))),
            "lightly damped": (k, float(0.01 * np.sqrt(k * inertia))),
        }

    @staticmethod
    def _scales(
        k: float,
        inertia: float,
        angle_rad: float,
        rate_rad: float,
        expected: tuple[float, float],
    ) -> tuple[float, float]:
        """Returns the angle and rate scales that errors are measured against.

        :param k: The spring constant (N*m/rad).
        :param inertia: The torsional inertia (kg*m^2).
        :param angle_rad: The largest angle (radians) that the forcing or the initial
            state set.
        :param rate_rad: The largest initial rate (rad/s).
        :param expected: The reference state.
        :return: The angle scale (radians) and the rate scale (rad/s).
        """
        angle_scale_rad = max(angle_rad, abs(expected[0]))
        rate_scale_rad = max(
            rate_rad, abs(expected[1]), angle_scale_rad * float(np.sqrt(k / inertia))
        )
        return angle_scale_rad, rate_scale_rad

    def test_zero_state_and_zero_forcing_stays_zero(self) -> None:
        """Test that a strip at rest with no forcing stays exactly at rest."""
        for inertia in self._INERTIAS:
            with self.subTest(inertia=inertia):
                result = self._solve(
                    1000.0, 1000.0, inertia, 0.0, 0.0, 0.0, lambda s: 0.0
                )
                self.assertEqual(result, (0.0, 0.0))

    def test_returns_floats(self) -> None:
        """Test that _spring_analytical_ode returns Python floats."""
        theta_rad, theta_derivative_rad = self._solve(
            10.0, 0.5, 1.0, 0.1, 0.0, 0.2, lambda s: np.float64(0.1)
        )
        self.assertIsInstance(theta_rad, float)
        self.assertIsInstance(theta_derivative_rad, float)

    def test_equilibrium_is_kept(self) -> None:
        """Test that a strip resting at its equilibrium angle under a constant moment
        stays there."""
        for inertia in self._INERTIAS:
            with self.subTest(inertia=inertia):
                moment = 2e-3
                equilibrium_rad = moment / 1000.0
                theta_rad, theta_derivative_rad = self._solve(
                    1000.0,
                    1000.0,
                    inertia,
                    equilibrium_rad,
                    0.0,
                    moment,
                    lambda s: 0.0,
                )
                self.assertAlmostEqual(
                    theta_rad,
                    equilibrium_rad,
                    delta=self._precision(1000.0, inertia, self._DT) * equilibrium_rad,
                )
                self.assertAlmostEqual(theta_derivative_rad, 0.0, delta=1e-9)

    def test_spring_restores_toward_zero(self) -> None:
        """Test that a displaced, unforced strip moves back toward zero."""
        theta_rad, _ = self._solve(
            10.0, 2.0, 1.0, 1.0, 0.0, 0.0, lambda s: 0.0, duration=1.0
        )
        self.assertLess(abs(theta_rad), 1.0)

    def test_constant_forcing_matches_closed_form(self) -> None:
        """Test that, under a constant moment, the solve matches the closed form
        solution for every inertia and damping regime, including free decay."""
        theta0_rad, theta_derivative0_rad = 1e-4, 3e-3
        for inertia in self._INERTIAS:
            for name, (k, c) in self._damping_cases(inertia).items():
                for moment in (0.0, 2e-3, -5e-2):
                    with self.subTest(inertia=inertia, damping=name, moment=moment):
                        expected = _closed_form_spring_solution(
                            k,
                            c,
                            inertia,
                            theta0_rad,
                            theta_derivative0_rad,
                            moment,
                            self._DT,
                        )
                        actual = self._solve(
                            k,
                            c,
                            inertia,
                            theta0_rad,
                            theta_derivative0_rad,
                            moment,
                            lambda s: 0.0,
                        )
                        angle_scale_rad, rate_scale_rad = self._scales(
                            k,
                            inertia,
                            max(theta0_rad, abs(moment) / k),
                            theta_derivative0_rad,
                            expected,
                        )
                        self._assert_state_close(
                            actual,
                            expected,
                            angle_scale_rad,
                            rate_scale_rad,
                            self._precision(c, inertia, self._DT),
                        )

    def test_linear_forcing_is_exact(self) -> None:
        """Test that a moment changing linearly in time, which the linear pieces
        represent exactly, gives the closed form solution."""
        start_moment, moment_slope = 1e-3, 0.4
        start_time = 0.1
        forcing = lambda s: start_moment + moment_slope * (s - start_time)
        for inertia in self._INERTIAS:
            for name, (k, c) in self._damping_cases(inertia).items():
                with self.subTest(inertia=inertia, damping=name):
                    expected = _closed_form_ramp_solution(
                        k, c, inertia, 2e-6, -1e-4, start_moment, moment_slope, self._DT
                    )
                    actual = self._solve(
                        k, c, inertia, 2e-6, -1e-4, 0.0, forcing, start_time
                    )
                    angle_scale_rad, rate_scale_rad = self._scales(
                        k, inertia, max(2e-6, 3e-3 / k), 1e-4, expected
                    )
                    self._assert_state_close(
                        actual,
                        expected,
                        angle_scale_rad,
                        rate_scale_rad,
                        self._precision(c, inertia, self._DT),
                    )

    def test_flap_drive_style_forcing_is_exact(self) -> None:
        """Test that a forcing interpolated linearly between samples at the solver's own
        sample times, as a FlapDrive's flap acceleration is, gives the closed form
        solution chained across the samples, even when the samples jump."""
        rng = np.random.default_rng(seed=0)
        start_time = 0.1
        sample_times = np.linspace(
            start_time, start_time + self._DT, problems_module._SPRING_FORCING_SAMPLES
        )

        # The samples are a smooth trend plus noise, with a 30 percent drop partway
        # through, as when a saturating motor's peak allowance runs out.
        sample_moments = 1e-3 * np.sin(np.linspace(0.0, 3.0, sample_times.size))
        sample_moments += 1e-4 * rng.standard_normal(sample_times.size)
        sample_moments[sample_times.size // 2 :] *= 0.7
        forcing = lambda s: float(np.interp(s, sample_times, sample_moments))
        piece_duration = float(sample_times[1] - sample_times[0])
        for inertia in self._INERTIAS:
            for name, (k, c) in self._damping_cases(inertia).items():
                with self.subTest(inertia=inertia, damping=name):
                    # Chain the closed form ramp solution across the pieces.
                    expected = (1e-6, 0.0)
                    for piece in range(sample_times.size - 1):
                        expected = _closed_form_ramp_solution(
                            k,
                            c,
                            inertia,
                            expected[0],
                            expected[1],
                            float(sample_moments[piece]),
                            float(sample_moments[piece + 1] - sample_moments[piece])
                            / piece_duration,
                            piece_duration,
                        )
                    actual = self._solve(
                        k, c, inertia, 1e-6, 0.0, 0.0, forcing, start_time
                    )
                    angle_scale_rad, rate_scale_rad = self._scales(
                        k, inertia, max(1e-6, 1e-3 / k), 0.0, expected
                    )
                    self._assert_state_close(
                        actual,
                        expected,
                        angle_scale_rad,
                        rate_scale_rad,
                        self._precision(c, inertia, self._DT),
                    )

    @staticmethod
    def _sinusoidal_error(
        k: float,
        c: float,
        inertia: float,
        flap_frequency: float,
        dt: float,
    ) -> tuple[float, float]:
        """Returns the analytical solve's error on a sinusoidal inertial moment, as a
        fraction of the angle's scale, and the predicted size of that error.

        :param k: The spring constant (N*m/rad).
        :param c: The damping constant (N*m*s/rad).
        :param inertia: The torsional inertia (kg*m^2).
        :param flap_frequency: The sinusoid's frequency (Hz).
        :param dt: The time step (s).
        :return: The error and (omega * h)^2, where omega is the sinusoid's angular
            frequency and h is the duration of one linear piece. Straight lines through
            a sinusoid miss it by about (omega * h)^2 / 8 of its amplitude.
        """
        omega_rad = 2.0 * np.pi * flap_frequency
        inertial = lambda s: 5e-3 * float(np.sin(omega_rad * s + 0.4))
        expected = _reference_spring_solution(
            k, c, inertia, 1e-6, 0.0, lambda s: 1e-3 + inertial(s), 0.1, 0.1 + dt
        )
        actual = TestSpringAnalyticalOde._solve(
            k, c, inertia, 1e-6, 0.0, 1e-3, inertial, duration=dt
        )
        angle_scale_rad = max(1e-6, 6e-3 / k, abs(expected[0]))
        piece_duration = dt / (problems_module._SPRING_FORCING_SAMPLES - 1)
        return (
            abs(actual[0] - expected[0]) / angle_scale_rad,
            (omega_rad * piece_duration) ** 2,
        )

    def test_sinusoidal_forcing_error_stays_within_its_predicted_size(self) -> None:
        """Test that a prescribed sinusoidal flap's inertial moment, which the linear.

        pieces only approximate, gives the converged reference solution to within the
        predicted size of the straight line error, (omega * h)^2 / 8, for slow and fast
        flaps. (For a 10 Hz flap at a 0.01 s time step, that is about 5e-6.)
        """
        for flap_frequency, dt in ((1.0, 0.03), (5.0, 0.005), (10.0, 0.01)):
            for inertia in self._INERTIAS[:4]:
                for name, (k, c) in self._damping_cases(inertia).items():
                    with self.subTest(
                        frequency=flap_frequency, inertia=inertia, damping=name
                    ):
                        error, predicted = self._sinusoidal_error(
                            k, c, inertia, flap_frequency, dt
                        )
                        self.assertLess(error, max(1e-9, predicted / 8.0))

    def test_sinusoidal_forcing_error_shrinks_with_more_samples(self) -> None:
        """Test that halving the linear pieces' duration cuts the error on a smooth
        sinusoidal moment by about four, as straight line interpolation should."""
        k, c, inertia = 1000.0, 2.0 * float(np.sqrt(1000.0 * 1e-3)), 1e-3
        coarse_error, _ = self._sinusoidal_error(k, c, inertia, 10.0, 0.01)
        with patch.object(
            problems_module,
            "_SPRING_FORCING_SAMPLES",
            2 * problems_module._SPRING_FORCING_SAMPLES - 1,
        ):
            fine_error, _ = self._sinusoidal_error(k, c, inertia, 10.0, 0.01)
        self.assertGreater(coarse_error / fine_error, 3.5)
        self.assertLess(coarse_error / fine_error, 4.5)

    def test_matches_numerical_solver_for_non_stiff_strips(self) -> None:
        """Test that the analytical and numerical solvers agree within the numerical
        solver's tolerance on strips that the numerical solver handles easily."""
        forcing = lambda s: 2e-3 * float(np.cos(2.0 * np.pi * 2.0 * s))
        problem_class = ps.problems.AeroelasticUnsteadyProblem
        for inertia, k, c in (
            (1.0, 10.0, 0.5),
            (4.4, 1000.0, 1000.0),
            (0.1, 50.0, 0.2),
        ):
            with self.subTest(inertia=inertia, k=k, c=c):
                t = np.array([0.2, 0.23])
                numerical = problem_class._spring_numerical_ode(
                    t, k, c, inertia, 3e-4, -2e-3, 1e-3, forcing
                )
                analytical = problem_class._spring_analytical_ode(
                    t, k, c, inertia, 3e-4, -2e-3, 1e-3, forcing
                )
                np.testing.assert_allclose(analytical, numerical, rtol=1e-5, atol=1e-10)

    def test_more_accurate_than_numerical_methods(self) -> None:
        """Test that, against the exact solution, the analytical solve is at least as
        accurate as the existing numerical solve (RK45) and as solve_ivp's Radau and
        LSODA methods at the same tolerances, for heavy to light strips in every damping
        regime that RK45 can finish quickly."""
        for inertia in (10.0, 1e-1, 1e-3):
            for name, (k, c) in self._damping_cases(inertia).items():
                with self.subTest(inertia=inertia, damping=name):
                    errors = _spring_method_errors(k, c, inertia)
                    analytical_angle_error, analytical_rate_error, _ = errors.pop(
                        "analytical"
                    )
                    self.assertLess(analytical_angle_error, 1e-9)
                    self.assertLess(analytical_rate_error, 1e-9)
                    for method, (angle_error, rate_error, _) in errors.items():
                        with self.subTest(method=method):
                            self.assertLessEqual(
                                analytical_angle_error, max(1e-12, angle_error)
                            )
                            self.assertLessEqual(
                                analytical_rate_error, max(1e-12, rate_error)
                            )

    def test_consecutive_steps_compose(self) -> None:
        """Test that two consecutive half steps give the same state as one full step
        under a linear moment, which both represent exactly."""
        forcing = lambda s: 1e-3 - 0.2 * s
        half_dt = self._DT / 2.0
        for inertia in self._INERTIAS:
            for name, (k, c) in self._damping_cases(inertia).items():
                with self.subTest(inertia=inertia, damping=name):
                    full = self._solve(
                        k, c, inertia, 1e-5, 2e-4, 0.0, forcing, 0.1, self._DT
                    )
                    half = self._solve(
                        k, c, inertia, 1e-5, 2e-4, 0.0, forcing, 0.1, half_dt
                    )
                    two_halves = self._solve(
                        k,
                        c,
                        inertia,
                        half[0],
                        half[1],
                        0.0,
                        forcing,
                        0.1 + half_dt,
                        half_dt,
                    )
                    angle_scale_rad, rate_scale_rad = self._scales(
                        k, inertia, max(1e-5, 1e-3 / k), 2e-4, full
                    )
                    self._assert_state_close(
                        two_halves,
                        full,
                        angle_scale_rad,
                        rate_scale_rad,
                        self._precision(c, inertia, self._DT),
                    )

    def test_multi_step_run_tracks_reference(self) -> None:
        """Test that a 100 time step run of a prescribed sinusoidal flap, restarted from
        its own state every time step as the problem does, stays within 1e-6 of a
        converged reference run (the aeroelastic example's settings)."""
        k, c, inertia, dt = 1000.0, 1000.0, 4.4, 0.03
        omega_rad = 2.0 * np.pi
        amplitude_rad = float(np.deg2rad(15.0))
        inertial = (
            lambda s: 5.6 * omega_rad**2 * amplitude_rad * float(np.sin(omega_rad * s))
        )
        aero_moments = 50.0 * np.sin(2.0 * np.pi * np.arange(100) * dt + 0.3)
        analytical_state = (0.0, 0.0)
        reference_state = (0.0, 0.0)
        analytical_history = []
        reference_history = []
        for step in range(1, 100):
            start_time = dt * (step - 1)
            aero_moment = float(aero_moments[step])
            analytical_state = self._solve(
                k,
                c,
                inertia,
                analytical_state[0],
                analytical_state[1],
                aero_moment,
                inertial,
                start_time,
                dt,
            )

            def total_moment(time_s: float, moment: float = aero_moment) -> float:
                return moment + inertial(time_s)

            reference_state = _reference_spring_solution(
                k,
                c,
                inertia,
                reference_state[0],
                reference_state[1],
                total_moment,
                start_time,
                start_time + dt,
            )
            analytical_history.append(analytical_state[0])
            reference_history.append(reference_state[0])
        error = np.max(
            np.abs(np.array(analytical_history) - np.array(reference_history))
        )
        self.assertLess(error, 1e-6 * np.max(np.abs(reference_history)))

    def test_forcing_is_sampled_at_evenly_spaced_times_within_the_step(self) -> None:
        """Test that the forcing is evaluated only at the evenly spaced sample times
        between the first and last time points, whatever points lie between them."""
        sampled_times: list[float] = []

        def recording_forcing(time_s: float) -> float:
            sampled_times.append(time_s)
            return 0.0

        ps.problems.AeroelasticUnsteadyProblem._spring_analytical_ode(
            np.array([0.2, 0.21, 0.25]),
            10.0,
            0.5,
            1.0,
            0.0,
            0.0,
            0.0,
            recording_forcing,
        )
        np.testing.assert_allclose(
            sampled_times,
            np.linspace(0.2, 0.25, problems_module._SPRING_FORCING_SAMPLES),
            rtol=0.0,
            atol=1e-15,
        )

    def test_stiff_strip_is_fast_and_accurate(self) -> None:
        """Test that a very light, stiff strip (c / I = 1e12 per second), which the
        numerical solver can't integrate in practical time, is solved quickly and
        matches the closed form solution."""
        k, c, inertia, moment = 1000.0, 1000.0, 1e-9, 2e-3
        start = time.perf_counter()
        for _ in range(20):
            actual = self._solve(k, c, inertia, 1e-6, 0.0, moment, lambda s: 0.0)
        elapsed_per_call = (time.perf_counter() - start) / 20.0
        expected = _closed_form_spring_solution(
            k, c, inertia, 1e-6, 0.0, moment, self._DT
        )
        self.assertLess(elapsed_per_call, 0.05)
        self.assertAlmostEqual(
            actual[0],
            expected[0],
            delta=self._precision(c, inertia, self._DT) * abs(expected[0]),
        )
