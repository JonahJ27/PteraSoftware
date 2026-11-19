"""Integration tests for flap driven runs of the
FreeFlightAeroelasticUnsteadyRingVortexLatticeMethodSolver.

The regression tests check that a flap driven free flight run reproduces stored results:
its flap angles, actuator forces, Wing deformations, lift, and the body's speed, angle
of attack, and pitch rate at every time step. The stored values were computed with the
default analytical torsional spring solver, so a run with it must reproduce them to near
machine precision. A run with the numerical torsional spring solver (which integrates
each strip with solve_ivp's RK45 to a relative tolerance of 1e-6) must agree with them,
through the coupling with the rigid body, to 1e-5 of each history's largest value. They
agree to about 3e-6.

Another test checks that the body holds its initial state through the prescribed time
steps and then moves in free flight.
"""

import unittest

import numpy as np

import pterasoftware as ps
from tests.integration.fixtures import flap_drive_fixtures

# These are the stored results of the fixture run, computed with the default analytical
# spring solver, one entry per time step. The deformations are the y components
# (radians) of the deformation angles of the main Wing's middle and tip
# WingCrossSections (its root WingCrossSection is clamped).
_FREE_FLIGHT_FLAP_ANGLES_RAD = np.array(
    [
        0.0499536781762,
        -0.0319053058959,
        -0.110641186371,
        -0.178546753844,
        -0.228974925068,
        -0.256989430884,
        -0.259848016064,
        -0.237270857692,
        -0.191467994401,
        -0.126922907396,
        -0.0499537163012,
        0.031905294891,
        0.110641190725,
        0.178546760286,
        0.22897492814,
        0.256989430983,
    ]
)
_FREE_FLIGHT_ACTUATOR_FORCES = np.array(
    [
        -1.69728880539,
        1.08405508073,
        3.75929314962,
        -6.84315815841,
        -3.66725233062,
        0.485348277833,
        14.5063334931,
        15.4715714295,
        11.2132820182,
        7.15831072673,
        5.04349100776,
        3.86591193986,
        2.56270147129,
        1.04100866515,
        -0.290118680344,
        -1.2227434068,
    ]
)
_FREE_FLIGHT_MIDDLE_DEFORMATION_RAD = np.array(
    [
        0,
        0,
        0,
        -0.0047448031792,
        -0.0165868296374,
        -0.0313751379775,
        -0.038966041247,
        -0.0292075544339,
        -0.0035956741448,
        0.0278500837049,
        0.0551322986251,
        0.0718157487927,
        0.0745405719067,
        0.0621990834779,
        0.0359855192135,
        -0.000419857828998,
    ]
)
_FREE_FLIGHT_TIP_DEFORMATION_RAD = np.array(
    [
        0,
        0,
        0,
        0.00381797524548,
        0.0180424465189,
        0.0419939594123,
        0.0734327108221,
        0.112644186679,
        0.151221664286,
        0.168534035689,
        0.149420176947,
        0.0946142588594,
        0.0183030646423,
        -0.0610485913572,
        -0.128503826777,
        -0.175437782751,
    ]
)
_FREE_FLIGHT_LIFTS = np.array(
    [
        0.117936263824,
        27.605189874,
        23.7076395928,
        20.7452750499,
        14.6301553474,
        -15.576999996,
        -17.1626856581,
        -11.0143176364,
        -7.88507479024,
        -9.20357423208,
        -11.5558547702,
        -12.4855000865,
        -11.9739399068,
        -10.8877325755,
        -9.57428407992,
        -7.93448585933,
    ]
)
_FREE_FLIGHT_SPEEDS = np.array(
    [
        10,
        10,
        10,
        10,
        10,
        9.75682719752,
        9.68026200773,
        9.66813147085,
        9.66808164301,
        9.66901028124,
        9.67687345782,
        9.69774467397,
        9.72472367818,
        9.74649704008,
        9.7551736185,
        9.7483880563,
    ]
)
_FREE_FLIGHT_ALPHAS = np.array(
    [
        0,
        0,
        0,
        0,
        0,
        1.42945196036,
        0.896045349261,
        0.0901673549887,
        -0.408065831858,
        -0.55773999308,
        -0.391572952016,
        0.108157287578,
        0.949725975316,
        2.05387259901,
        3.24931137415,
        4.34023269439,
    ]
)
_FREE_FLIGHT_PITCH_RATES = np.array(
    [
        0,
        0,
        0,
        0,
        0,
        -121.501963126,
        -101.665969202,
        -61.0235033313,
        -52.3896322643,
        -62.7247085371,
        -67.6638566408,
        -60.3013009956,
        -47.6149028175,
        -36.5822389821,
        -28.21215786,
        -21.3810212663,
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


def _histories(
    problem: ps.free_flight_flap_driven_aeroelastic_unsteady_problem.FreeFlightFlapDrivenAeroelasticUnsteadyProblem,
) -> dict[str, np.ndarray]:
    """Returns the histories that the tests compare.

    :param problem: The solved problem.
    :return: A dict mapping each history's name to a (num_steps,) ndarray of floats.
    """
    deformations = np.array(
        problem.listDeformationAnglesYRad_Wcsp_to_Wcs_ixyz[0], dtype=float
    )
    lifts = []
    for steady_problem in problem.steady_problems:
        forces_W = steady_problem.airplanes[0].forces_W
        assert forces_W is not None
        lifts.append(-float(forces_W[2]))
    movement = problem.movement
    assert isinstance(
        movement, ps.movements.free_flight_movement.FreeFlightAeroelasticMovement
    )
    operating_points = movement.operating_point_movement.operating_points
    return {
        "flap angles": np.array(problem.flap_angles_rad[0], dtype=float),
        "actuator forces": np.array(problem.flap_actuator_forces[0], dtype=float),
        "middle deformations": deformations[:, 1],
        "tip deformations": deformations[:, 2],
        "lifts": np.array(lifts, dtype=float),
        "speeds": np.array(
            [operating_point.vCg__E for operating_point in operating_points],
            dtype=float,
        ),
        "angles of attack": np.array(
            [operating_point.alpha for operating_point in operating_points],
            dtype=float,
        ),
        "pitch rates": np.array(
            [operating_point.omegas_BP1__E[1] for operating_point in operating_points],
            dtype=float,
        ),
    }


class TestFreeFlightFlapDrivenAeroelasticUnsteadyRingVortexLatticeMethod(
    unittest.TestCase
):
    """This is a class with functions to test flap driven runs of the
    FreeFlightAeroelasticUnsteadyRingVortexLatticeMethodSolver."""

    problem: (
        ps.free_flight_flap_driven_aeroelastic_unsteady_problem.FreeFlightFlapDrivenAeroelasticUnsteadyProblem
    )

    @classmethod
    def setUpClass(cls) -> None:
        """Solves the default free flight flap driven problem fixture once for the tests
        that only read it."""
        cls.problem = flap_drive_fixtures.make_free_flight_flap_driven_problem()
        _solve(cls.problem)

    def _assert_matches_stored(
        self,
        problem: ps.free_flight_flap_driven_aeroelastic_unsteady_problem.FreeFlightFlapDrivenAeroelasticUnsteadyProblem,
        tolerance: float,
    ) -> None:
        """Asserts that every stored history of a solved problem matches its stored
        values.

        :param problem: The solved problem.
        :param tolerance: The allowed difference as a fraction of each stored history's
            largest magnitude.
        :return: None
        """
        expected_histories = {
            "flap angles": _FREE_FLIGHT_FLAP_ANGLES_RAD,
            "actuator forces": _FREE_FLIGHT_ACTUATOR_FORCES,
            "middle deformations": _FREE_FLIGHT_MIDDLE_DEFORMATION_RAD,
            "tip deformations": _FREE_FLIGHT_TIP_DEFORMATION_RAD,
            "lifts": _FREE_FLIGHT_LIFTS,
            "speeds": _FREE_FLIGHT_SPEEDS,
            "angles of attack": _FREE_FLIGHT_ALPHAS,
            "pitch rates": _FREE_FLIGHT_PITCH_RATES,
        }
        for name, actual in _histories(problem).items():
            expected = expected_histories[name]
            with self.subTest(history=name):
                self.assertEqual(actual.shape, expected.shape)
                self.assertLessEqual(
                    float(np.max(np.abs(actual - expected))),
                    tolerance * float(np.max(np.abs(expected))),
                )

    def test_runs_to_completion(self) -> None:
        """Test that the run fills every history with one finite entry per time step and
        that the Wings deform."""
        for name, history in _histories(self.problem).items():
            with self.subTest(history=name):
                self.assertEqual(history.size, self.problem.num_steps)
                self.assertTrue(np.all(np.isfinite(history)))
        self.assertGreater(
            float(np.max(np.abs(_histories(self.problem)["tip deformations"]))), 0.0
        )

    def test_body_holds_then_moves(self) -> None:
        """Test that the body holds its initial state through the prescribed time steps
        and then moves in free flight."""
        movement = self.problem.movement
        assert isinstance(
            movement, ps.movements.free_flight_movement.FreeFlightAeroelasticMovement
        )
        prescribed = movement.prescribed_num_steps
        histories = _histories(self.problem)
        for name in ("speeds", "angles of attack", "pitch rates"):
            with self.subTest(history=name):
                history = histories[name]
                np.testing.assert_allclose(history[: prescribed + 1], history[0])
                self.assertGreater(float(np.ptp(history[prescribed:])), 0.0)

    def test_analytical_spring_solver_reproduces_stored_solution(self) -> None:
        """Test that the default (analytical) spring solver, which produced the stored
        results, still reproduces them."""
        self.assertTrue(self.problem.use_analytical_spring_solver)
        self._assert_matches_stored(self.problem, _REPRODUCTION_TOLERANCE)

    def test_numerical_spring_solver_matches_stored_solution(self) -> None:
        """Test that the numerical spring solver agrees with the results stored from the
        analytical spring solver."""
        problem = flap_drive_fixtures.make_free_flight_flap_driven_problem(
            use_analytical_spring_solver=False
        )
        _solve(problem)
        self._assert_matches_stored(problem, _SOLVER_TOLERANCE)
