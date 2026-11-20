"""Integration tests for flap driven runs of the
AeroelasticUnsteadyRingVortexLatticeMethodSolver.

The regression tests check that a flap driven run reproduces stored results: its flap
angles, actuator forces, Wing deformations, and lift at every time step. The stored
values were computed with the default analytical torsional spring solver, so a run with
it must reproduce them to near machine precision. A run with the numerical torsional
spring solver (which integrates each strip with solve_ivp's RK45 to a relative tolerance
of 1e-6) must agree with them to 1e-5 of each history's largest value. They agree to
about 3e-7.

The drive grouping test checks that one FlapDrive that flaps both Wings gives the same
run as two FlapDrives, one per Wing, each with half of its actuator inertia and feedback
gains. The shared FlapDrive carries both Wings' fluid torques, which the two per-Wing
FlapDrives split.

The saturation test checks that an actuator whose torque is limited to about 70 percent
of what the flap needs falls behind its commanded flap and flaps through a smaller
angle, while never exceeding its limit.

The prescribed flap test checks that a stiff, unlimited FlapDrive reproduces a run whose
Wings flap through the same sine flap prescribed directly (an
AeroelasticUnsteadyProblem), in flap angle, Wing deformation, and lift.
"""

import unittest

import numpy as np

import pterasoftware as ps
from tests.integration.fixtures import flap_drive_fixtures

# These are the stored results of the fixture run, computed with the default analytical
# spring solver, one entry per time step. The deformations are the y components
# (radians) of the deformation angles of the main Wing's middle and tip
# WingCrossSections (its root WingCrossSection is clamped).
_PRESCRIBED_FLAP_FLAP_ANGLES_RAD = np.array(
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
        0.259848015171,
        0.237270857046,
        0.191467994236,
        0.126922907485,
    ]
)
_PRESCRIBED_FLAP_ACTUATOR_FORCES = np.array(
    [
        -1.69728880539,
        1.08405508073,
        3.75929314962,
        -6.84315815841,
        -3.66725233062,
        0.485348277836,
        4.20368703403,
        6.77255523676,
        8.01646708259,
        8.18725452684,
        7.72667028211,
        7.05103088423,
        6.42785283353,
        5.95127785609,
        5.58067412757,
        5.19326169402,
        4.63127307719,
        3.75047222744,
        2.48157232957,
        0.886712753934,
    ]
)
_PRESCRIBED_FLAP_MIDDLE_DEFORMATION_RAD = np.array(
    [
        0,
        0,
        0,
        -0.0047448031792,
        -0.0165868296374,
        -0.0313751379775,
        -0.0442591237382,
        -0.0515530209373,
        -0.0518001411343,
        -0.0459231275999,
        -0.0366443609327,
        -0.0274839337945,
        -0.0216841222042,
        -0.021360758812,
        -0.0270525116829,
        -0.0377029833774,
        -0.0510260859065,
        -0.0641493965141,
        -0.0743662446452,
        -0.0797828418724,
    ]
)
_PRESCRIBED_FLAP_TIP_DEFORMATION_RAD = np.array(
    [
        0,
        0,
        0,
        0.00381797524548,
        0.0180424465189,
        0.0419939594123,
        0.0719547340422,
        0.100513657365,
        0.118448774611,
        0.11788659331,
        0.095187451131,
        0.0523033835106,
        -0.00386540523903,
        -0.0637004124742,
        -0.117594822466,
        -0.158012600665,
        -0.180335966145,
        -0.182811923313,
        -0.166251547716,
        -0.133812789408,
    ]
)
_PRESCRIBED_FLAP_LIFTS = np.array(
    [
        0.117936263824,
        27.605189874,
        23.7076395928,
        20.7452750499,
        14.6301553474,
        7.92647509058,
        1.85002941178,
        -3.16691101335,
        -7.28633928858,
        -10.8915558412,
        -14.2455150074,
        -17.2935125811,
        -19.7036939517,
        -21.099120337,
        -21.2515595731,
        -20.0684761556,
        -17.4558497594,
        -13.2762018289,
        -7.51056757148,
        -0.53444738317,
    ]
)

# This is the allowed difference between a history and its stored values, as a fraction
# of the stored history's largest magnitude, when the run uses a different spring solver
# from the one that produced the stored values.
_SOLVER_TOLERANCE = 1e-5

# This is the allowed difference when the run uses the same spring solver that produced
# the stored values, as a fraction of the stored history's largest magnitude.
_REPRODUCTION_TOLERANCE = 1e-9

# This is the torque limit (N*m) of the saturated FlapDrive, about 70 percent of the
# largest torque that the unlimited FlapDrive applies.
_TORQUE_LIMIT = 5.7


def _solve(problem: ps.problems.AeroelasticUnsteadyProblem) -> None:
    """Solves an AeroelasticUnsteadyProblem with a free wake.

    :param problem: The problem to solve.
    :return: None
    """
    solver = ps.aeroelastic_unsteady_ring_vortex_lattice_method.AeroelasticUnsteadyRingVortexLatticeMethodSolver(
        aeroelastic_unsteady_problem=problem
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


def _deformations(
    problem: ps.problems.AeroelasticUnsteadyProblem,
) -> np.ndarray:
    """Returns the main Wing's deformation angle y components at every time step.

    :param problem: The solved problem.
    :return: A (num_steps, num_wing_cross_sections) ndarray of floats (radians).
    """
    return np.array(problem.listDeformationAnglesYRad_Wcsp_to_Wcs_ixyz[0], dtype=float)


def _desired_flap_angles_rad(num_steps: int) -> np.ndarray:
    """Returns the fixtures' sine flap angle at each time step.

    :param num_steps: The number of time steps.
    :return: A (num_steps,) ndarray of floats (radians).
    """
    times = np.arange(num_steps, dtype=float) * flap_drive_fixtures.DELTA_TIME
    angles_rad: np.ndarray = np.deg2rad(flap_drive_fixtures.FLAP_AMPLITUDE) * np.sin(
        2.0 * np.pi / flap_drive_fixtures.FLAP_PERIOD * times
        + np.deg2rad(flap_drive_fixtures.FLAP_PHASE)
    )
    return angles_rad


class TestFlapDrivenAeroelasticUnsteadyRingVortexLatticeMethod(unittest.TestCase):
    """This is a class with functions to test flap driven runs of the
    AeroelasticUnsteadyRingVortexLatticeMethodSolver."""

    problem: (
        ps.flap_driven_aeroelastic_unsteady_problem.FlapDrivenAeroelasticUnsteadyProblem
    )

    @classmethod
    def setUpClass(cls) -> None:
        """Solves the default flap driven problem fixture once for the tests that only
        read it."""
        cls.problem = flap_drive_fixtures.make_flap_driven_problem()
        _solve(cls.problem)

    def _assert_history_matches(
        self, actual: np.ndarray, expected: np.ndarray, tolerance: float
    ) -> None:
        """Asserts that a history matches its expected values.

        :param actual: The history.
        :param expected: The expected values.
        :param tolerance: The allowed difference as a fraction of the expected values'
            largest magnitude.
        :return: None
        """
        self.assertEqual(actual.shape, expected.shape)
        scale = float(np.max(np.abs(expected)))
        self.assertLessEqual(
            float(np.max(np.abs(actual - expected))),
            tolerance * scale,
        )

    def _assert_matches_stored(
        self,
        problem: ps.flap_driven_aeroelastic_unsteady_problem.FlapDrivenAeroelasticUnsteadyProblem,
        tolerance: float,
    ) -> None:
        """Asserts that every stored history of a solved problem matches its stored
        values.

        :param problem: The solved problem.
        :param tolerance: The allowed difference as a fraction of each stored history's
            largest magnitude.
        :return: None
        """
        deformations = _deformations(problem)
        for name, actual, expected in (
            (
                "flap angles",
                np.array(problem.flap_angles_rad[0]),
                _PRESCRIBED_FLAP_FLAP_ANGLES_RAD,
            ),
            (
                "actuator forces",
                np.array(problem.flap_actuator_forces[0]),
                _PRESCRIBED_FLAP_ACTUATOR_FORCES,
            ),
            (
                "middle deformations",
                deformations[:, 1],
                _PRESCRIBED_FLAP_MIDDLE_DEFORMATION_RAD,
            ),
            (
                "tip deformations",
                deformations[:, 2],
                _PRESCRIBED_FLAP_TIP_DEFORMATION_RAD,
            ),
            ("lifts", _lifts(problem), _PRESCRIBED_FLAP_LIFTS),
        ):
            with self.subTest(history=name):
                self._assert_history_matches(actual, expected, tolerance)

    def test_runs_to_completion(self) -> None:
        """Test that the run fills every history with one finite entry per time step and
        that the Wings deform."""
        num_steps = self.problem.num_steps
        self.assertEqual(len(self.problem.flap_angles_rad[0]), num_steps)
        self.assertEqual(len(self.problem.steady_problems), num_steps)
        deformations = _deformations(self.problem)
        self.assertEqual(deformations.shape[0], num_steps)
        self.assertTrue(np.all(np.isfinite(deformations)))
        self.assertTrue(np.all(np.isfinite(_lifts(self.problem))))
        self.assertGreater(float(np.max(np.abs(deformations))), 0.0)

    def test_analytical_spring_solver_reproduces_stored_solution(self) -> None:
        """Test that the default (analytical) spring solver, which produced the stored
        results, still reproduces them."""
        self.assertTrue(self.problem.use_analytical_spring_solver)
        self._assert_matches_stored(self.problem, _REPRODUCTION_TOLERANCE)

    def test_numerical_spring_solver_matches_stored_solution(self) -> None:
        """Test that the numerical spring solver agrees with the results stored from the
        analytical spring solver."""
        problem = flap_drive_fixtures.make_flap_driven_problem(
            use_analytical_spring_solver=False
        )
        _solve(problem)
        self._assert_matches_stored(problem, _SOLVER_TOLERANCE)

    def test_shared_drive_matches_per_wing_drives(self) -> None:
        """Test that one FlapDrive for both Wings matches two per-Wing FlapDrives that
        each have half of its actuator inertia and feedback gains."""
        per_wing = flap_drive_fixtures.make_flap_driven_problem(
            flap_drives=[
                flap_drive_fixtures.make_tracking_drive((0,), share=0.5),
                flap_drive_fixtures.make_tracking_drive((1,), share=0.5),
            ]
        )
        _solve(per_wing)

        shared_angles = np.array(self.problem.flap_angles_rad[0])
        for drive_index in (0, 1):
            with self.subTest(per_wing_drive=drive_index):
                np.testing.assert_allclose(
                    per_wing.flap_angles_rad[drive_index],
                    shared_angles,
                    rtol=0.0,
                    atol=1e-12,
                )
        np.testing.assert_allclose(
            _deformations(per_wing),
            _deformations(self.problem),
            rtol=0.0,
            atol=1e-9 * float(np.max(np.abs(_deformations(self.problem)))),
        )

        # The shared FlapDrive carries both Wings' fluid torques, which the per-Wing
        # FlapDrives split between them.
        shared_torques = np.array(self.problem.flap_fluid_torques[0])
        summed_torques = np.array(per_wing.flap_fluid_torques[0]) + np.array(
            per_wing.flap_fluid_torques[1]
        )
        np.testing.assert_allclose(
            summed_torques,
            shared_torques,
            rtol=0.0,
            atol=1e-8 * float(np.max(np.abs(shared_torques))),
        )

    def test_saturated_drive_lags_and_flaps_less(self) -> None:
        """Test that a FlapDrive whose torque is limited below what the flap needs stays
        within its limit, reaches it, falls behind its commanded flap, and flaps through
        a smaller angle than commanded."""
        limited = flap_drive_fixtures.make_flap_driven_problem(
            flap_drives=[
                flap_drive_fixtures.make_tracking_drive(torque_limit=_TORQUE_LIMIT)
            ]
        )
        _solve(limited)

        # The unlimited FlapDrive needs more torque than the limit.
        self.assertGreater(
            float(np.max(np.abs(self.problem.flap_actuator_forces[0]))), _TORQUE_LIMIT
        )

        forces = np.abs(np.array(limited.flap_actuator_forces[0]))
        self.assertLessEqual(float(np.max(forces)), _TORQUE_LIMIT * (1.0 + 1e-12))
        self.assertGreaterEqual(int(np.sum(forces >= _TORQUE_LIMIT * (1.0 - 1e-9))), 5)

        angles_rad = np.array(limited.flap_angles_rad[0])
        desired_rad = _desired_flap_angles_rad(angles_rad.size)
        unlimited_error_rad = np.max(
            np.abs(np.array(self.problem.flap_angles_rad[0]) - desired_rad)
        )
        limited_error_rad = np.max(np.abs(angles_rad - desired_rad))
        self.assertLess(unlimited_error_rad, np.deg2rad(1e-3))
        self.assertGreater(limited_error_rad, np.deg2rad(5.0))
        self.assertLess(float(np.ptp(angles_rad)), 0.8 * float(np.ptp(desired_rad)))

    def test_stiff_drive_reproduces_prescribed_flap(self) -> None:
        """Test that a stiff, unlimited FlapDrive gives the same flap angles, Wing
        deformations, and lift as prescribing the same sine flap directly."""
        prescribed = flap_drive_fixtures.make_prescribed_problem()
        _solve(prescribed)

        prescribed_angles_rad = np.deg2rad(
            [
                steady_problem.airplanes[0].wings[0].angles_Gs_to_Wn_ixyz[0]
                for steady_problem in prescribed.steady_problems
            ]
        )
        np.testing.assert_allclose(
            self.problem.flap_angles_rad[0],
            prescribed_angles_rad,
            rtol=0.0,
            atol=np.deg2rad(1e-4),
        )
        self._assert_history_matches(
            _deformations(self.problem), _deformations(prescribed), 1e-5
        )
        self._assert_history_matches(_lifts(self.problem), _lifts(prescribed), 1e-5)
