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
        -6.84315815843,
        -3.66725233081,
        0.485348277931,
        14.506340281,
        15.4715644706,
        11.2132413068,
        7.15831290628,
        5.0434503885,
        3.8659231804,
        2.56271460294,
        1.04114486998,
        -0.290135684254,
        -1.22280738723,
    ]
)
_FREE_FLIGHT_MIDDLE_DEFORMATION_RAD = np.array(
    [
        0,
        0,
        0,
        -0.00474480317918,
        -0.0165868296376,
        -0.0313751379778,
        -0.0389660431709,
        -0.0292075718036,
        -0.00359573902954,
        0.0278499604259,
        0.055132131146,
        0.0718155711771,
        0.0745404238483,
        0.0621990115361,
        0.0359855484128,
        -0.000419756322193,
    ]
)
_FREE_FLIGHT_TIP_DEFORMATION_RAD = np.array(
    [
        0,
        0,
        0,
        0.00381797524562,
        0.0180424465141,
        0.0419939594003,
        0.0734327114795,
        0.112644195501,
        0.151221663589,
        0.16853391487,
        0.149419948818,
        0.0946140676554,
        0.0183030447153,
        -0.0610484557566,
        -0.128503592523,
        -0.175437557432,
    ]
)
_FREE_FLIGHT_LIFTS = np.array(
    [
        0.117936263823,
        27.605189874,
        23.7076395928,
        20.7452750507,
        14.6301553472,
        -15.5770143851,
        -17.1626692523,
        -11.0142289235,
        -7.88508473873,
        -9.20348352309,
        -11.5558833942,
        -12.4855264618,
        -11.9742336235,
        -10.8876772273,
        -9.57415064724,
        -7.93443393662,
    ]
)
_FREE_FLIGHT_SPEEDS = np.array(
    [
        10,
        10,
        10,
        10,
        10,
        9.75682726073,
        9.68026198187,
        9.66813151351,
        9.66808160316,
        9.66901014573,
        9.67687315318,
        9.69774420171,
        9.72472317942,
        9.7464968917,
        9.75517347114,
        9.74838789238,
    ]
)
_FREE_FLIGHT_ALPHAS = np.array(
    [
        0,
        -0,
        -0,
        -0,
        -0,
        1.42944745289,
        0.89604520238,
        0.0901844680642,
        -0.408056870295,
        -0.55771648581,
        -0.391562033086,
        0.108160392191,
        0.949683781792,
        2.05385437761,
        3.24930731474,
        4.34023355455,
    ]
)
_FREE_FLIGHT_PITCH_RATES = np.array(
    [
        0,
        0,
        0,
        0,
        0,
        -121.501937927,
        -101.66591652,
        -61.0234566887,
        -52.3897550246,
        -62.7247414652,
        -67.6640276064,
        -60.3013744276,
        -47.6151640144,
        -36.5818563028,
        -28.2116651231,
        -21.3809702789,
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
