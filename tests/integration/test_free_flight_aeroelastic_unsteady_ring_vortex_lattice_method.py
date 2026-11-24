"""Integration tests for the FreeFlightAeroelasticUnsteadyRingVortexLatticeMethodSolver.

The fixture is the aeroelastic version of the flapping free flight fixture: the same
Airplane, mass, inertia, initial OperatingPoint, and 15 degree, 1 Hz symmetric flap,
with the main Wing's two halves deforming through the torsional springs (by up to about
0.6 rad). The vehicle is held at its initial state for 5 time steps and then flies
freely for 10.

The completion test checks that the run fills every history with finite values and that
the Wings deform. The held phase test checks that the body holds its initial state
through the prescribed time steps and moves afterward.

The prescribed phase test checks that, while the body is held, the run matches an
AeroelasticUnsteadyProblem that flaps the same Airplane at the same OperatingPoint, in
the Wings' deformations and the lift. They match to about 1e-9.

The symmetry test checks that the symmetric flap keeps the lateral states (the sideslip,
the roll and yaw rates, and the sideways position) near zero, and that the two halves of
the main Wing deform the same way.

The rigid limit test checks that with very stiff, heavily damped springs, which leave
the Wings nearly rigid, the vehicle flies as the non aeroelastic
FreeFlightUnsteadyProblem fixture does with the same Airplane. They match to about 1e-4
of how much each body state changes over the run.

The regression tests check that a run reproduces stored results: the speed, angle of
attack, pitch rate, tip deformation, and lift at every time step. The stored values were
computed with the problem's default numerical torsional spring solver, so a run with it
must reproduce them to near machine precision. A run with the analytical spring solver
must agree with them to 1e-5 of each history's largest value. They agree to about 1e-6.
"""

import unittest

import numpy as np

import pterasoftware as ps
from tests.integration.fixtures import (
    array_path_fixtures,
    movement_fixtures,
    problem_fixtures,
)

# These are the stored results of the fixture run, computed with the default numerical
# spring solver, one entry per time step: the speed (m/s), the angle of attack
# (degrees), the pitch rate (degrees per second), the y component (radians) of the main
# Wing's tip WingCrossSection's deformation angle, and the lift (N).
_FLAPPING_SPEEDS = np.array(
    [
        12.9,
        12.9,
        12.9,
        12.9,
        12.9,
        12.9,
        13.0658636001,
        13.25038706,
        13.4551718859,
        13.6620377525,
        13.8518798626,
        13.995710342,
        14.0583216514,
        14.002068403,
        13.7939211315,
    ]
)
_FLAPPING_ALPHAS = np.array(
    [
        3.3,
        3.3,
        3.3,
        3.3,
        3.3,
        3.3,
        5.38547142986,
        7.93813552559,
        11.1031481162,
        14.5299139611,
        18.2189999161,
        22.2062707578,
        26.4854138726,
        31.0361524043,
        35.8216980602,
    ]
)
_FLAPPING_PITCH_RATES = np.array(
    [
        0,
        0,
        0,
        0,
        0,
        0,
        -0.380533399693,
        -0.468392983583,
        -0.121404134663,
        0.388874588815,
        1.27876940095,
        2.92973219129,
        5.73988733892,
        10.1288541255,
        16.4439243152,
    ]
)
_FLAPPING_TIP_DEFORMATION_RAD = np.array(
    [
        0,
        0,
        0,
        -0.00221309014615,
        -0.0109849482662,
        -0.0283636391124,
        -0.0551148693785,
        -0.0933349990303,
        -0.143937443511,
        -0.204816618311,
        -0.274845852156,
        -0.354107459703,
        -0.442358693841,
        -0.538966400844,
        -0.642860611805,
    ]
)
_FLAPPING_LIFTS = np.array(
    [
        3049.32180936,
        -11333.494146,
        -1651.8156171,
        -1612.1800063,
        -1856.87606528,
        -2175.8238938,
        -1138.22920402,
        -1514.51975581,
        -2011.90000793,
        -2252.70313865,
        -2499.58502088,
        -2773.69770922,
        -3030.17032022,
        -3241.95122167,
        -3378.1014251,
    ]
)

# This is the allowed difference between a history and its stored values, as a fraction
# of the stored history's largest magnitude, when the run uses a different spring solver
# from the one that produced the stored values.
_SOLVER_TOLERANCE = 1e-5

# This is the allowed difference when the run uses the same spring solver that produced
# the stored values, as a fraction of the stored history's largest magnitude.
_REPRODUCTION_TOLERANCE = 1e-9


def _solve(problem: ps.problems.FreeFlightAeroelasticUnsteadyProblem) -> None:
    """Solves a FreeFlightAeroelasticUnsteadyProblem with a free wake.

    :param problem: The problem to solve.
    :return: None
    """
    solver = ps.free_flight_aeroelastic_unsteady_ring_vortex_lattice_method.FreeFlightAeroelasticUnsteadyRingVortexLatticeMethodSolver(
        problem
    )
    solver.run(prescribed_wake=False, show_progress=False)


def _lifts(problem: ps.problems.AeroelasticUnsteadyProblem) -> np.ndarray:
    """Returns the first Airplane's lift (N) at every time step.

    :param problem: The solved problem.
    :return: A (num_steps,) ndarray of floats.
    """
    lifts = []
    for steady_problem in problem.steady_problems:
        forces_W = steady_problem.airplanes[0].forces_W
        assert forces_W is not None
        lifts.append(-float(forces_W[2]))
    return np.array(lifts, dtype=float)


def _body_states(
    problem: (
        ps.problems.FreeFlightUnsteadyProblem
        | ps.problems.FreeFlightAeroelasticUnsteadyProblem
    ),
) -> dict[str, np.ndarray]:
    """Returns the body's state at every time step.

    :param problem: The solved problem.
    :return: A dict holding the speeds (m/s), the angles of attack and sideslip
        (degrees), the body rates (in the first Airplane's body axes, in degrees per
        second, one row per time step), and the CG's positions (in Earth axes, relative
        to the Earth origin, in meters, one row per time step).
    """
    movement = problem.movement
    assert isinstance(
        movement,
        (
            ps.movements.free_flight_movement.FreeFlightMovement,
            ps.movements.free_flight_movement.FreeFlightAeroelasticMovement,
        ),
    )
    operating_points = movement.operating_point_movement.operating_points
    return {
        "speeds": np.array([point.vCg__E for point in operating_points], dtype=float),
        "alphas": np.array([point.alpha for point in operating_points], dtype=float),
        "betas": np.array([point.beta for point in operating_points], dtype=float),
        "omegas_BP1__E": np.array(
            [point.omegas_BP1__E for point in operating_points], dtype=float
        ),
        "positions_E_Eo": np.array(
            [point.CgP1_E_Eo for point in operating_points], dtype=float
        ),
    }


class TestFreeFlightAeroelasticUnsteadyRingVortexLatticeMethod(unittest.TestCase):
    """This is a class with functions to test the
    FreeFlightAeroelasticUnsteadyRingVortexLatticeMethodSolver."""

    problem: ps.problems.FreeFlightAeroelasticUnsteadyProblem

    @classmethod
    def setUpClass(cls) -> None:
        """Solves the flapping free flight aeroelastic problem fixture once for the
        tests that only read it."""
        cls.problem = problem_fixtures.make_flapping_free_flight_aeroelastic_problem()
        _solve(cls.problem)

    def _deformations(
        self, problem: ps.problems.AeroelasticUnsteadyProblem, wing_index: int
    ) -> np.ndarray:
        """Returns a Wing's deformation angle y components at every time step.

        :param problem: The solved problem.
        :param wing_index: The Wing's index.
        :return: A (num_steps, num_wing_cross_sections) ndarray of floats (radians).
        """
        return np.array(
            problem.listDeformationAnglesYRad_Wcsp_to_Wcs_ixyz[wing_index], dtype=float
        )

    def _assert_matches_stored(
        self,
        problem: ps.problems.FreeFlightAeroelasticUnsteadyProblem,
        tolerance: float,
    ) -> None:
        """Asserts that a solved problem's histories match the stored values.

        :param problem: The solved problem.
        :param tolerance: The allowed difference as a fraction of each stored history's
            largest magnitude.
        :return: None
        """
        states = _body_states(problem)
        for name, actual, expected in (
            ("speeds", states["speeds"], _FLAPPING_SPEEDS),
            ("angles of attack", states["alphas"], _FLAPPING_ALPHAS),
            ("pitch rates", states["omegas_BP1__E"][:, 1], _FLAPPING_PITCH_RATES),
            (
                "tip deformations",
                self._deformations(problem, 0)[:, -1],
                _FLAPPING_TIP_DEFORMATION_RAD,
            ),
            ("lifts", _lifts(problem), _FLAPPING_LIFTS),
        ):
            with self.subTest(history=name):
                self.assertEqual(actual.shape, expected.shape)
                self.assertLessEqual(
                    float(np.max(np.abs(actual - expected))),
                    tolerance * float(np.max(np.abs(expected))),
                )

    def test_runs_to_completion(self) -> None:
        """Test that the run fills every history with one finite entry per time step and
        that the Wings deform."""
        num_steps = self.problem.num_steps
        self.assertEqual(len(self.problem.steady_problems), num_steps)
        for name, history in _body_states(self.problem).items():
            with self.subTest(history=name):
                self.assertEqual(history.shape[0], num_steps)
                self.assertTrue(np.all(np.isfinite(history)))
        deformations = self._deformations(self.problem, 0)
        self.assertEqual(deformations.shape[0], num_steps)
        self.assertTrue(np.all(np.isfinite(deformations)))
        self.assertGreater(float(np.max(np.abs(deformations))), 0.1)

    def test_body_holds_then_moves(self) -> None:
        """Test that the body holds its initial state through the prescribed time steps
        and then moves in free flight."""
        prescribed = movement_fixtures.FLAPPING_FREE_FLIGHT_PRESCRIBED_NUM_STEPS
        states = _body_states(self.problem)
        for name in ("speeds", "alphas", "omegas_BP1__E"):
            with self.subTest(state=name):
                history = states[name]
                held = history[: prescribed + 1]
                np.testing.assert_allclose(
                    held, np.broadcast_to(history[0], held.shape)
                )
                self.assertGreater(float(np.max(np.ptp(history, axis=0))), 0.0)

    def test_prescribed_phase_matches_held_aeroelastic_problem(self) -> None:
        """Test that, while the body is held, the run matches an
        AeroelasticUnsteadyProblem that flaps the same Airplane at the same
        OperatingPoint."""
        num_held_steps = movement_fixtures.FLAPPING_FREE_FLIGHT_PRESCRIBED_NUM_STEPS + 1
        held = problem_fixtures.make_flapping_held_aeroelastic_problem(num_held_steps)
        ps.aeroelastic_unsteady_ring_vortex_lattice_method.AeroelasticUnsteadyRingVortexLatticeMethodSolver(
            aeroelastic_unsteady_problem=held
        ).run(
            prescribed_wake=False, show_progress=False
        )

        for wing_index in (0, 1):
            with self.subTest(wing=wing_index):
                expected = self._deformations(held, wing_index)
                actual = self._deformations(self.problem, wing_index)[:num_held_steps]
                self.assertGreater(float(np.max(np.abs(expected))), 0.0)
                np.testing.assert_allclose(
                    actual,
                    expected,
                    rtol=0.0,
                    atol=1e-7 * float(np.max(np.abs(expected))),
                )
        np.testing.assert_allclose(
            _lifts(self.problem)[:num_held_steps], _lifts(held), rtol=1e-8
        )

    def test_symmetric_flap_stays_symmetric(self) -> None:
        """Test that the symmetric flap keeps the lateral states near zero and deforms
        the two halves of the main Wing the same way."""
        states = _body_states(self.problem)
        omegas_BP1__E = states["omegas_BP1__E"]
        largest_pitch_rate = float(np.max(np.abs(omegas_BP1__E[:, 1])))
        self.assertGreater(largest_pitch_rate, 1.0)
        self.assertLess(float(np.max(np.abs(states["betas"]))), 1e-4)
        self.assertLess(
            float(np.max(np.abs(omegas_BP1__E[:, [0, 2]]))), 1e-4 * largest_pitch_rate
        )
        self.assertLess(float(np.max(np.abs(states["positions_E_Eo"][:, 1]))), 1e-6)

        main_deformations = self._deformations(self.problem, 0)
        reflected_deformations = self._deformations(self.problem, 1)
        np.testing.assert_allclose(
            reflected_deformations,
            main_deformations,
            rtol=0.0,
            atol=1e-5 * float(np.max(np.abs(main_deformations))),
        )

    def test_stiff_springs_reproduce_rigid_free_flight(self) -> None:
        """Test that, with very stiff and heavily damped springs, the vehicle barely
        deforms and flies as the non aeroelastic FreeFlightUnsteadyProblem does with the
        same Airplane."""
        stiff = problem_fixtures.make_flapping_free_flight_aeroelastic_problem(
            spring_constant_rad=1e8,
            damping_constant_rad=1e4,
            use_analytical_spring_solver=True,
        )
        _solve(stiff)
        self.assertLess(float(np.max(np.abs(self._deformations(stiff, 0)))), 1e-4)

        rigid = problem_fixtures.make_flapping_free_flight_problem(
            explode_into_strips=True
        )
        ps.free_flight_unsteady_ring_vortex_lattice_method.FreeFlightUnsteadyRingVortexLatticeMethodSolver(
            rigid
        ).run(
            prescribed_wake=False, show_progress=False
        )

        stiff_states = _body_states(stiff)
        rigid_states = _body_states(rigid)
        for name in ("speeds", "alphas", "omegas_BP1__E", "positions_E_Eo"):
            with self.subTest(state=name):
                change = float(np.max(np.ptp(rigid_states[name], axis=0)))
                self.assertGreater(change, 0.0)
                self.assertLess(
                    float(np.max(np.abs(stiff_states[name] - rigid_states[name]))),
                    1e-3 * change,
                )

    def test_numerical_spring_solver_reproduces_stored_solution(self) -> None:
        """Test that the default (numerical) spring solver, which produced the stored
        results, still reproduces them."""
        self.assertFalse(self.problem.use_analytical_spring_solver)
        self._assert_matches_stored(self.problem, _REPRODUCTION_TOLERANCE)

    def test_analytical_spring_solver_matches_stored_solution(self) -> None:
        """Test that the analytical spring solver agrees with the results stored from
        the numerical spring solver."""
        problem = problem_fixtures.make_flapping_free_flight_aeroelastic_problem(
            use_analytical_spring_solver=True
        )
        _solve(problem)
        self._assert_matches_stored(problem, _SOLVER_TOLERANCE)


class TestFreeFlightAeroelasticArrayPath(unittest.TestCase):
    """Verifies that the array path gives the object path's loads on a flapping free
    flight aeroelastic problem."""

    def test_matches_the_object_path(self) -> None:
        """Running the solver with use_arrays=True gives the object path's loads at
        every time step, to within the problem's sensitivity (see
        FREE_FLIGHT_AEROELASTIC_RELATIVE_TOLERANCE).

        :return: None
        """
        object_solver, array_solver = (
            array_path_fixtures.make_solvers_run_on_both_paths(
                lambda: ps.free_flight_aeroelastic_unsteady_ring_vortex_lattice_method.FreeFlightAeroelasticUnsteadyRingVortexLatticeMethodSolver(
                    problem_fixtures.make_flapping_free_flight_aeroelastic_problem()
                ),
                prescribed_wake=False,
            )
        )
        array_path_fixtures.assert_loads_match(
            self,
            object_solver,
            array_solver,
            array_path_fixtures.FREE_FLIGHT_AEROELASTIC_RELATIVE_TOLERANCE,
        )
