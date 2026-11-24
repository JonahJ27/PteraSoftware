"""This module tests the FreeFlightUnsteadyRingVortexLatticeMethodSolver by running two
coupled free flight cases and checking that the resulting motion is physically sensible.

The first case is the simple glider in an unpowered glide. The glider is statically
stable in pitch and yaw (its geometry, center of gravity, and inertia were tuned for
this and verified in XFLR5), so a successful coupled run produces a bounded, damped
glide: the angle of attack oscillates gently about its trimmed value rather than
diverging, lift balances weight, the glider descends and moves forward, the flight stays
laterally symmetric, and total mechanical energy decreases as drag dissipates it.

The second case is a flapping-wing airframe whose main wing flaps symmetrically.
Flapping produces large, oscillatory aerodynamic loads with significant added mass, the
regime that an explicit, loosely coupled handoff renders unstable and diverges. The
strongly coupled, Aitken-relaxed sub-iteration drives the loads and the body state to
within-step consistency each time step instead, so the run stays finite and bounded.
This case validates that the coupling completes a flapping free flight without
diverging, that the flapping genuinely excites large oscillatory loads, that the sub-
iteration converges within its cap every step, and that symmetric flapping keeps the
flight laterally symmetric.

The third case reruns the simple glider with the Katz force method. The rigid body
dynamics consume only the total loads that process_solver_loads writes, which both force
methods produce, so the coupled run must again yield a finite, full-length load history
whose mean lift approximately balances the glider's weight.
"""

import logging
import re
import unittest
from typing import Any
from unittest.mock import patch

import numpy as np

import pterasoftware as ps
from tests.integration.fixtures import (
    array_path_fixtures,
    problem_fixtures,
    solver_fixtures,
)


class TestFreeFlightUnsteadyRingVortexLatticeMethod(unittest.TestCase):
    """This is a class for testing the FreeFlightUnsteadyRingVortexLatticeMethodSolver
    on a statically stable glider."""

    solver: (
        ps.free_flight_unsteady_ring_vortex_lattice_method.FreeFlightUnsteadyRingVortexLatticeMethodSolver
    )
    speeds: np.ndarray
    alphas: np.ndarray
    betas: np.ndarray
    positions_E_Eo: np.ndarray
    omegas_BP1__E: np.ndarray
    lifts: np.ndarray
    crosswind_forces: np.ndarray
    weight: float
    g_E: np.ndarray
    mass: float
    inertia_matrix: np.ndarray

    @classmethod
    def setUpClass(cls) -> None:
        """This method sets up the test by running the simple glider's free flight
        simulation once and extracting its time histories.

        :return: None
        """
        cls.solver = solver_fixtures.make_simple_glider_free_flight_solver()
        cls.solver.run(prescribed_wake=True, show_progress=False)

        problem = cls.solver.unsteady_problem
        assert isinstance(problem, ps.problems.FreeFlightUnsteadyProblem)
        movement = problem.movement
        assert isinstance(
            movement, ps.movements.free_flight_movement.FreeFlightMovement
        )
        operating_points = movement.operating_point_movement.operating_points

        # Extract per-time-step time histories from the dynamically populated
        # OperatingPoints and each time step's solved Airplane.
        cls.speeds = np.array(
            [operating_point.vCg__E for operating_point in operating_points]
        )
        cls.alphas = np.array(
            [operating_point.alpha for operating_point in operating_points]
        )
        cls.betas = np.array(
            [operating_point.beta for operating_point in operating_points]
        )
        cls.positions_E_Eo = np.array(
            [operating_point.CgP1_E_Eo for operating_point in operating_points]
        )
        cls.omegas_BP1__E = np.array(
            [operating_point.omegas_BP1__E for operating_point in operating_points]
        )

        # In Ptera Software's wind axes, lift is the negative z component of the wind
        # axes force.
        forces_W = np.array(
            [
                steady_problem.airplanes[0].forces_W
                for steady_problem in problem.steady_problems
            ]
        )
        cls.lifts = -forces_W[:, 2]
        cls.crosswind_forces = -forces_W[:, 1]

        cls.weight = cls.solver.current_airplanes[0].weight
        cls.g_E = operating_points[0].g_E
        cls.mass = problem.mass
        cls.inertia_matrix = problem.I_BP1_CgP1

    def test_run_produces_finite_full_length_history(self) -> None:
        """This method tests that the coupled run completed for every time step and
        produced finite state and load histories (no divergence or NaNs).

        :return: None
        """
        num_steps = self.solver.unsteady_problem.num_steps

        self.assertEqual(len(self.speeds), num_steps)
        self.assertEqual(len(self.lifts), num_steps)

        self.assertTrue(np.all(np.isfinite(self.speeds)))
        self.assertTrue(np.all(np.isfinite(self.alphas)))
        self.assertTrue(np.all(np.isfinite(self.positions_E_Eo)))
        self.assertTrue(np.all(np.isfinite(self.lifts)))

    def test_lift_approximately_balances_weight(self) -> None:
        """This method tests that the mean lift over the glide is close to the glider's
        weight, as it should be near the trimmed condition.

        :return: None
        """
        mean_lift = float(np.mean(self.lifts))
        lift_to_weight_ratio = mean_lift / self.weight

        self.assertGreater(lift_to_weight_ratio, 0.85)
        self.assertLess(lift_to_weight_ratio, 1.15)

    def test_glider_descends(self) -> None:
        """This method tests that the glider loses altitude over the glide.

        In Earth axes, the +z direction points down (gravity acts along +z), so a
        descending glider's z coordinate increases.

        :return: None
        """
        initial_z_E = self.positions_E_Eo[0, 2]
        final_z_E = self.positions_E_Eo[-1, 2]

        self.assertGreater(final_z_E, initial_z_E)

    def test_speed_remains_bounded(self) -> None:
        """This method tests that the glide speed stays within a reasonable band of its
        initial value rather than running away.

        :return: None
        """
        initial_speed = self.speeds[0]

        self.assertLess(float(np.max(self.speeds)), 1.2 * initial_speed)
        self.assertGreater(float(np.min(self.speeds)), 0.8 * initial_speed)

    def test_flight_stays_laterally_symmetric(self) -> None:
        """This method tests that the glide stays in the longitudinal plane.

        The glider is laterally symmetric and starts with zero sideslip, so the sideslip
        angle, the crosswind force, and the lateral position should all stay negligible.

        :return: None
        """
        self.assertLess(float(np.max(np.abs(self.betas))), 1.0)
        self.assertLess(float(np.max(np.abs(self.crosswind_forces))), 1.0)
        self.assertLess(float(np.max(np.abs(self.positions_E_Eo[:, 1]))), 1.0e-3)

    def test_angle_of_attack_stays_bounded(self) -> None:
        """This method tests that the angle of attack stays in a narrow band about its
        trimmed value rather than diverging. This is the signature of the glider's
        static pitch stability: a stable airframe makes a small restoring oscillation,
        whereas an unstable one would run away toward the angle limits.

        :return: None
        """
        trimmed_alpha = self.alphas[0]

        self.assertLess(float(np.max(self.alphas)), trimmed_alpha + 3.0)
        self.assertGreater(float(np.min(self.alphas)), trimmed_alpha - 3.0)

    def test_total_mechanical_energy_decreases(self) -> None:
        """This method tests that the glider's total mechanical energy at the end of the
        glide is less than at the start. With no thrust, drag is the only non-
        conservative load, so it must dissipate mechanical energy over the glide.

        :return: None
        """
        # Translational kinetic energy.
        kinetic_energies = 0.5 * self.mass * self.speeds**2

        # Gravitational potential energy. With g_E along +z (down), a descending glider
        # (increasing z) loses potential energy.
        potential_energies = -self.mass * (self.positions_E_Eo @ self.g_E)

        # Rotational kinetic energy, using the body-axes angular velocity (converted
        # from degrees per second to radians per second) and inertia matrix.
        omegasRad_BP1__E = np.deg2rad(self.omegas_BP1__E)
        rotational_kinetic_energies = np.array(
            [
                0.5 * omegaRad @ self.inertia_matrix @ omegaRad
                for omegaRad in omegasRad_BP1__E
            ]
        )

        total_energies = (
            kinetic_energies + potential_energies + rotational_kinetic_energies
        )

        self.assertLess(total_energies[-1], total_energies[0])


class TestFreeFlightUnsteadyRingVortexLatticeMethodFlapping(unittest.TestCase):
    """This is a class for testing the FreeFlightUnsteadyRingVortexLatticeMethodSolver
    on a flapping-wing airframe whose strongly coupled sub-iteration must remain stable
    under large oscillatory loads."""

    solver: (
        ps.free_flight_unsteady_ring_vortex_lattice_method.FreeFlightUnsteadyRingVortexLatticeMethodSolver
    )
    betas: np.ndarray
    positions_E_Eo: np.ndarray
    forces_W: np.ndarray
    lifts: np.ndarray
    crosswind_forces: np.ndarray
    weight: float

    _warning_records: list[logging.LogRecord]

    @classmethod
    def setUpClass(cls) -> None:
        """This method sets up the test by running the flapping-wing free flight
        simulation once and extracting its time histories.

        It attaches a handler to the solver's logger for the duration of the run so the
        test can later assert that the strongly coupled sub-iteration never hit its
        iteration cap (the solver logs a warning on a capped, non-converged step).

        :return: None
        """
        # Collect the WARNING-level records the solver emits during the run so the test
        # can check for the sub-iteration cap warning.
        cls._warning_records = []

        class _WarningCollector(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                cls._warning_records.append(record)

        solver_logger = logging.getLogger("pterasoftware.problems")
        collector = _WarningCollector(level=logging.WARNING)
        solver_logger.addHandler(collector)

        try:
            cls.solver = solver_fixtures.make_flapping_free_flight_solver()
            cls.solver.run(prescribed_wake=True, show_progress=False)
        finally:
            solver_logger.removeHandler(collector)

        problem = cls.solver.unsteady_problem
        movement = problem.movement
        assert isinstance(
            movement, ps.movements.free_flight_movement.FreeFlightMovement
        )
        operating_points = movement.operating_point_movement.operating_points

        # Extract per-time-step time histories from the dynamically populated
        # OperatingPoints and each time step's solved Airplane.
        cls.betas = np.array(
            [operating_point.beta for operating_point in operating_points]
        )
        cls.positions_E_Eo = np.array(
            [operating_point.CgP1_E_Eo for operating_point in operating_points]
        )

        # In Ptera Software's wind axes, lift is the negative z component of the wind
        # axes force and the crosswind force is the negative y component.
        forces_W = np.array(
            [
                steady_problem.airplanes[0].forces_W
                for steady_problem in problem.steady_problems
            ]
        )
        cls.forces_W = forces_W
        cls.lifts = -forces_W[:, 2]
        cls.crosswind_forces = -forces_W[:, 1]

        cls.weight = cls.solver.current_airplanes[0].weight

    def test_run_produces_finite_full_length_history(self) -> None:
        """This method tests that the coupled run completed for every time step and
        produced finite state and load histories. Under loose coupling the flapping case
        diverges; a finite, full-length history is the signature that the strongly
        coupled sub-iteration kept the integration stable.

        :return: None
        """
        num_steps = self.solver.unsteady_problem.num_steps

        self.assertEqual(len(self.forces_W), num_steps)

        self.assertTrue(np.all(np.isfinite(self.forces_W)))
        self.assertTrue(np.all(np.isfinite(self.betas)))
        self.assertTrue(np.all(np.isfinite(self.positions_E_Eo)))

    def test_sub_iteration_converges_within_cap(self) -> None:
        """This method tests that the strongly coupled sub-iteration converged within
        its iteration cap on every free flight time step. The solver logs a warning when
        a step reaches the cap without converging and accepts the capped iterate anyway,
        so the absence of that warning confirms every step reached the convergence
        tolerance.

        :return: None
        """
        cap_warnings = [
            record for record in self._warning_records if "cap" in record.getMessage()
        ]

        self.assertEqual(cap_warnings, [])

    def test_flapping_excites_oscillatory_loads(self) -> None:
        """This method tests that the flapping genuinely drives large, oscillatory
        aerodynamic loads rather than settling to a near-constant value. The peak-to-
        peak lift over the run should exceed the airframe's weight, confirming that the
        case exercises the coupling in the large-load regime it is meant to stress.

        :return: None
        """
        peak_to_peak_lift = float(np.max(self.lifts) - np.min(self.lifts))

        self.assertGreater(peak_to_peak_lift, self.weight)

    def test_flight_stays_laterally_symmetric(self) -> None:
        """This method tests that the flight stays in the longitudinal plane.

        The main wing flaps symmetrically and the airframe starts with zero sideslip, so
        the sideslip angle, the crosswind force, and the lateral position should all
        stay negligible despite the large flapping loads.

        :return: None
        """
        self.assertLess(float(np.max(np.abs(self.betas))), 1.0)
        self.assertLess(float(np.max(np.abs(self.crosswind_forces))), 1.0)
        self.assertLess(float(np.max(np.abs(self.positions_E_Eo[:, 1]))), 1.0e-3)


class TestFreeFlightFastApproximations(unittest.TestCase):
    """This is a class for testing the flapping-wing free flight solver with the opt in
    approximations of Approximations.fast() against the same run without them."""

    # The largest allowed force difference at any time step between the exact run and
    # the fast() run, relative to the exact run's peak force. The measured difference
    # was at most 1.1e-5 (1.4e-5 before fast() included reuse_final_trial).
    _FORCE_TOLERANCE = 1e-4

    exact_factors: dict[int, list[float]]
    exact_forces_W: np.ndarray
    exact_caps: int
    fast_factors: dict[int, list[float]]
    fast_forces_W: np.ndarray
    fast_caps: int

    _pattern = re.compile(
        r"step (\d+), sub-iteration (\d+): weighted residual norm (\S+), relaxation "
        r"factor (\S+)"
    )

    @classmethod
    def _run(
        cls, approximations: ps.approximations.Approximations | None
    ) -> tuple[dict[int, list[float]], np.ndarray, int]:
        """Runs the flapping free flight fixture and records each free flight time
        step's relaxation factors from the sub-iteration's debug log.

        :param approximations: The problem's Approximations, or None for none.
        :return: A tuple of a dict mapping each free flight time step to the relaxation
            factors its sub-iterations logged, in order, the (N,3) ndarray of floats of
            each time step's forces in wind axes, and the number of cap warnings.
        """
        messages: list[str] = []

        class _Collector(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                messages.append(record.getMessage())

        problems_logger = logging.getLogger("pterasoftware.problems")
        original_level = problems_logger.level
        problems_logger.setLevel(logging.DEBUG)
        collector = _Collector(level=logging.DEBUG)
        problems_logger.addHandler(collector)
        try:
            solver = solver_fixtures.make_flapping_free_flight_solver(approximations)
            solver.run(prescribed_wake=True, show_progress=False)
        finally:
            problems_logger.removeHandler(collector)
            problems_logger.setLevel(original_level)

        factors: dict[int, list[float]] = {}
        for message in messages:
            match = cls._pattern.search(message)
            if match:
                factors.setdefault(int(match.group(1)), []).append(
                    float(match.group(4))
                )
        forces_W = np.array(
            [
                steady_problem.airplanes[0].forces_W
                for steady_problem in solver.unsteady_problem.steady_problems
            ]
        )
        cap_warnings = sum("cap" in message for message in messages)
        return factors, forces_W, cap_warnings

    @classmethod
    def setUpClass(cls) -> None:
        """This method sets up the test by running the flapping free flight fixture once
        without approximations and once with Approximations.fast().

        :return: None
        """
        cls.exact_factors, cls.exact_forces_W, cls.exact_caps = cls._run(None)
        cls.fast_factors, cls.fast_forces_W, cls.fast_caps = cls._run(
            ps.approximations.Approximations.fast()
        )

    def test_exact_run_starts_every_step_at_the_initial_factor(self) -> None:
        """Test that without approximations every free flight time step's first update
        uses the initial relaxation factor, 0.5."""
        self.assertGreater(len(self.exact_factors), 1)
        for step, factors in self.exact_factors.items():
            with self.subTest(step=step):
                self.assertEqual(factors[0], 0.5)

    def test_fast_run_carries_the_last_aitken_factor(self) -> None:
        """Test that with adaptive_relaxation the first free flight time step starts at
        0.5, and each later one at the previous time step's last Aitken factor when it
        is within [0.1, 1.0], and at 0.5 otherwise."""
        steps = sorted(self.fast_factors)
        self.assertEqual(self.fast_factors[steps[0]][0], 0.5)
        for previous, current in zip(steps, steps[1:]):
            with self.subTest(step=current):
                previous_factors = self.fast_factors[previous]
                if len(previous_factors) == 1:
                    # The previous time step converged on its first trial, so it
                    # computed no Aitken factor, and the one it carried is kept.
                    expected = previous_factors[0]
                else:
                    last = previous_factors[-1]
                    expected = last if 0.1 <= last <= 1.0 else 0.5
                # The log prints 3 significant digits.
                self.assertAlmostEqual(
                    self.fast_factors[current][0], expected, delta=5e-3 * expected
                )

    def test_fast_run_takes_fewer_trials_without_cap_hits(self) -> None:
        """Test that the fast() run converges every free flight time step within the
        cap, as the exact run does, using fewer sub-iterations in total."""
        self.assertEqual(self.exact_caps, 0)
        self.assertEqual(self.fast_caps, 0)
        exact_trials = sum(len(factors) for factors in self.exact_factors.values())
        fast_trials = sum(len(factors) for factors in self.fast_factors.values())
        self.assertLess(fast_trials, exact_trials)

    def test_fast_run_forces_match_the_exact_run(self) -> None:
        """Test that the fast() run's forces match the exact run's at every time step,
        within _FORCE_TOLERANCE of the peak force, and exactly in the prescribed phase,
        where the sub-iteration doesn't run."""
        self.assertEqual(self.fast_forces_W.shape, self.exact_forces_W.shape)
        first_free_step = min(self.exact_factors)
        np.testing.assert_array_equal(
            self.fast_forces_W[:first_free_step], self.exact_forces_W[:first_free_step]
        )
        peak = float(np.max(np.abs(self.exact_forces_W)))
        np.testing.assert_allclose(
            self.fast_forces_W,
            self.exact_forces_W,
            rtol=0.0,
            atol=self._FORCE_TOLERANCE * peak,
        )


class TestFreeFlightReuseFinalTrial(unittest.TestCase):
    """This is a class for testing the flapping-wing free flight solver with the
    reuse_final_trial approximation alone against the same run without it."""

    # The number of free flight time steps, from the first, over which the forces are
    # compared. Free flight without attitude control is chaotic, so any difference grows
    # over a run, and only the first free flight time steps measure the approximation
    # itself.
    _NUM_COMPARED_FREE_STEPS = 5

    # The largest allowed force difference over the compared time steps between the
    # exact run and the reuse_final_trial run, relative to the exact run's peak force.
    # The measured difference was 5.8e-7.
    _FORCE_TOLERANCE = 1e-5

    exact_factors: dict[int, list[float]]
    exact_forces_W: np.ndarray
    exact_full_solves: int
    reuse_factors: dict[int, list[float]]
    reuse_forces_W: np.ndarray
    reuse_full_solves: int
    reuse_caps: int
    num_steps: int

    @classmethod
    def _run(
        cls, approximations: ps.approximations.Approximations | None
    ) -> tuple[dict[int, list[float]], np.ndarray, int, int]:
        """Runs the flapping free flight fixture, counting its full solves.

        :param approximations: The problem's Approximations, or None for none.
        :return: A tuple of a dict mapping each free flight time step to the relaxation
            factors its sub-iterations logged, the (N,3) ndarray of floats of each time
            step's forces in wind axes, the number of cap warnings, and the number of
            full solves (each one solves for the bound ring vortex strengths).
        """
        solver_class = (
            ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver
        )
        original = solver_class._calculate_vortex_strengths
        with patch.object(
            solver_class,
            "_calculate_vortex_strengths",
            autospec=True,
            side_effect=original,
        ) as counted:
            factors, forces_W, caps = TestFreeFlightFastApproximations._run(
                approximations
            )
        return factors, forces_W, caps, counted.call_count

    @classmethod
    def setUpClass(cls) -> None:
        """This method sets up the test by running the flapping free flight fixture once
        without approximations and once with only reuse_final_trial.

        :return: None
        """
        (
            cls.exact_factors,
            cls.exact_forces_W,
            _,
            cls.exact_full_solves,
        ) = cls._run(None)
        (
            cls.reuse_factors,
            cls.reuse_forces_W,
            cls.reuse_caps,
            cls.reuse_full_solves,
        ) = cls._run(ps.approximations.Approximations(reuse_final_trial=True))
        cls.num_steps = cls.exact_forces_W.shape[0]

    def test_each_coupled_step_skips_one_full_solve(self) -> None:
        """Test that the official solve after every converged coupling reuses the final
        trial: the run solves in full once per time step and once per trial, less one
        solve per coupled time step."""
        self.assertEqual(self.reuse_caps, 0)
        num_trials = sum(len(factors) for factors in self.reuse_factors.values())
        num_coupled_steps = len(self.reuse_factors)
        self.assertGreater(num_coupled_steps, 1)
        self.assertEqual(
            self.reuse_full_solves,
            self.num_steps + num_trials - num_coupled_steps,
        )
        exact_trials = sum(len(factors) for factors in self.exact_factors.values())
        self.assertEqual(self.exact_full_solves, self.num_steps + exact_trials)

    def test_forces_match_the_exact_run(self) -> None:
        """Test that the held time steps are identical, and the first free flight time
        steps within _FORCE_TOLERANCE of the peak force, to the exact run."""
        first_free_step = min(self.exact_factors)
        np.testing.assert_array_equal(
            self.reuse_forces_W[: first_free_step + 1],
            self.exact_forces_W[: first_free_step + 1],
        )
        last_compared_step = first_free_step + 1 + self._NUM_COMPARED_FREE_STEPS
        peak = float(np.max(np.abs(self.exact_forces_W)))
        np.testing.assert_allclose(
            self.reuse_forces_W[:last_compared_step],
            self.exact_forces_W[:last_compared_step],
            rtol=0.0,
            atol=self._FORCE_TOLERANCE * peak,
        )


class TestFreeFlightUnsteadyRingVortexLatticeMethodKatzForceMethod(unittest.TestCase):
    """This is a class for testing the FreeFlightUnsteadyRingVortexLatticeMethodSolver
    on the statically stable glider with the Katz force method."""

    solver: (
        ps.free_flight_unsteady_ring_vortex_lattice_method.FreeFlightUnsteadyRingVortexLatticeMethodSolver
    )
    forces_W: np.ndarray
    lifts: np.ndarray
    weight: float

    @classmethod
    def setUpClass(cls) -> None:
        """This method sets up the test by running the simple glider's free flight
        simulation once with the Katz force method and extracting its load history.

        :return: None
        """
        cls.solver = solver_fixtures.make_simple_glider_free_flight_solver()
        cls.solver.run(prescribed_wake=True, show_progress=False, force_method="katz")

        problem = cls.solver.unsteady_problem

        # In Ptera Software's wind axes, lift is the negative z component of the wind
        # axes force.
        forces_W = np.array(
            [
                steady_problem.airplanes[0].forces_W
                for steady_problem in problem.steady_problems
            ]
        )
        cls.forces_W = forces_W
        cls.lifts = -forces_W[:, 2]

        cls.weight = cls.solver.current_airplanes[0].weight

    def test_run_produces_finite_full_length_history(self) -> None:
        """This method tests that the coupled run completed for every time step and
        produced a finite load history with the Katz force method.

        :return: None
        """
        num_steps = self.solver.unsteady_problem.num_steps

        self.assertEqual(len(self.forces_W), num_steps)

        self.assertTrue(np.all(np.isfinite(self.forces_W)))

    def test_lift_approximately_balances_weight(self) -> None:
        """This method tests that the mean lift over the glide is close to the glider's
        weight, as it should be near the trimmed condition. This confirms that the Katz
        force method feeds physically sensible loads into the rigid body dynamics.

        :return: None
        """
        mean_lift = float(np.mean(self.lifts))
        lift_to_weight_ratio = mean_lift / self.weight

        self.assertGreater(lift_to_weight_ratio, 0.85)
        self.assertLess(lift_to_weight_ratio, 1.15)


def _assert_restore_matches_resolve(
    test_case: unittest.TestCase,
    solver: (
        ps.free_flight_unsteady_ring_vortex_lattice_method.FreeFlightUnsteadyRingVortexLatticeMethodSolver
        | ps.free_flight_aeroelastic_unsteady_ring_vortex_lattice_method.FreeFlightAeroelasticUnsteadyRingVortexLatticeMethodSolver
    ),
) -> None:
    """Runs a free flight solver with a free wake and asserts, at every free step, that
    the state restore_substep puts back equals the state a fresh solve of that step
    produces.

    restore_substep puts back the solver state freeze_substep saved before the trials,
    instead of solving the current step again. This check wraps it: after the real
    restore, it records every solver attribute, solves the current step again (what
    restore_substep used to do), and asserts that every attribute is unchanged. A missed
    attribute, or one the trials changed in place, would show up as a difference.

    :param test_case: The TestCase whose assertions to use.
    :param solver: The free flight solver to run. It must not have run yet.
    :return: None
    """
    solver_class = type(solver)
    original_restore_substep = solver_class.restore_substep
    slot_names = [
        name
        for cls in solver_class.__mro__
        for name in getattr(cls, "__slots__", ())
        if not name.startswith("_substep")
    ]
    checked_steps: list[int] = []

    def checking_restore_substep(self: Any, step: int) -> None:
        original_restore_substep(self, step)

        # Record the restored state, then solve the current step again, as
        # restore_substep used to.
        restored = {
            name: getattr(self, name) for name in slot_names if hasattr(self, name)
        }
        self._evaluate_step_aerodynamics()

        for name, expected in restored.items():
            actual = getattr(self, name)
            if actual is expected:
                continue
            with test_case.subTest(step=step, attribute=name):
                if isinstance(expected, np.ndarray):
                    test_case.assertTrue(
                        np.array_equal(
                            expected, actual, equal_nan=expected.dtype.kind == "f"
                        )
                    )
                else:
                    test_case.assertEqual(expected, actual)
        checked_steps.append(step)

    # The solver has slots, so patch the method on its class.
    with patch.object(solver_class, "restore_substep", checking_restore_substep):
        solver.run(prescribed_wake=False, show_progress=False)

    # The check must have run on the free steps.
    test_case.assertGreater(len(checked_steps), 0)


class TestFreeFlightRestoreSubstepMatchesResolve(unittest.TestCase):
    """This is a class for testing that restore_substep, which puts back the state
    freeze_substep saved instead of solving the current step again, leaves both free
    flight solvers in exactly the state a fresh solve would."""

    def test_restore_substep_matches_resolve(self) -> None:
        """Test, at every free step of the simple glider with a free wake, that the
        restored solver state equals the state a fresh solve of the step produces."""
        _assert_restore_matches_resolve(
            self, solver_fixtures.make_simple_glider_free_flight_solver()
        )

    def test_aeroelastic_restore_substep_matches_resolve(self) -> None:
        """Test the same for the aeroelastic free flight solver, on the flapping free
        flight aeroelastic problem with a free wake."""
        problem = problem_fixtures.make_flapping_free_flight_aeroelastic_problem()
        _assert_restore_matches_resolve(
            self,
            ps.free_flight_aeroelastic_unsteady_ring_vortex_lattice_method.FreeFlightAeroelasticUnsteadyRingVortexLatticeMethodSolver(
                problem
            ),
        )


class TestFreeFlightArrayPath(unittest.TestCase):
    """Verifies that the array path gives the object path's loads on the gliding and the
    flapping free flight problems."""

    def test_matches_the_object_path(self) -> None:
        """Running the solver with use_arrays=True gives the object path's loads at
        every time step, to within each problem's sensitivity (see
        FREE_FLIGHT_GLIDER_RELATIVE_TOLERANCE and
        FREE_FLIGHT_FLAPPING_RELATIVE_TOLERANCE).

        :return: None
        """
        for make_solver, relative_tolerance in (
            (
                solver_fixtures.make_simple_glider_free_flight_solver,
                array_path_fixtures.FREE_FLIGHT_GLIDER_RELATIVE_TOLERANCE,
            ),
            (
                solver_fixtures.make_flapping_free_flight_solver,
                array_path_fixtures.FREE_FLIGHT_FLAPPING_RELATIVE_TOLERANCE,
            ),
        ):
            with self.subTest(problem=make_solver.__name__):
                object_solver, array_solver = (
                    array_path_fixtures.make_solvers_run_on_both_paths(
                        make_solver, prescribed_wake=True
                    )
                )
                array_path_fixtures.assert_loads_match(
                    self, object_solver, array_solver, relative_tolerance
                )


class TestFreeFlightArrayPathWithApproximations(unittest.TestCase):
    """Verifies that the array path gives the object path's loads with the opt in
    approximations, including reusing the final coupling trial."""

    def test_matches_the_object_path_with_fast_approximations(self) -> None:
        """Running the flapping free flight problem with Approximations.fast() on both
        paths gives the same loads at every time step, to within the problem's
        sensitivity (see FREE_FLIGHT_FLAPPING_RELATIVE_TOLERANCE), and the array path
        reuses final coupling trials.

        :return: None
        """
        approximations = ps.approximations.Approximations.fast()
        self.assertTrue(approximations.reuse_final_trial)
        solver_class = (
            ps.free_flight_unsteady_ring_vortex_lattice_method.FreeFlightUnsteadyRingVortexLatticeMethodSolver
        )
        original = solver_class.keep_final_trial
        kept = {"array": 0}

        def counting_keep_final_trial(self_: Any) -> None:
            if self_.use_arrays:
                kept["array"] += 1
            original(self_)

        with patch.object(solver_class, "keep_final_trial", counting_keep_final_trial):
            object_solver, array_solver = (
                array_path_fixtures.make_solvers_run_on_both_paths(
                    lambda: solver_fixtures.make_flapping_free_flight_solver(
                        approximations
                    ),
                    prescribed_wake=True,
                )
            )
        self.assertGreater(kept["array"], 0)
        array_path_fixtures.assert_loads_match(
            self,
            object_solver,
            array_solver,
            array_path_fixtures.FREE_FLIGHT_FLAPPING_RELATIVE_TOLERANCE,
        )
