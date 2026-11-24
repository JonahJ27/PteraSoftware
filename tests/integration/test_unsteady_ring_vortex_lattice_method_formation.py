"""This module contains a class to test the UnsteadyRingVortexLatticeMethodSolver on
formations of several Airplanes, on both the object path and the array path."""

import unittest

import numpy as np

from tests.integration.fixtures import array_path_fixtures, solver_fixtures

# The Airplane load attributes taken about each Airplane's own CG, which a formation's
# moving Airplanes change. assert_loads_match compares the others.
_OWN_CG_NAMES = (
    "moments_W_Cg",
    "moments_G_Cg",
    "momentCoefficients_W_Cg",
    "momentCoefficients_G_Cg",
)


class TestUnsteadyFormationArrayPath(unittest.TestCase):
    """Verifies that the array path gives the object path's results for formations of
    two Airplanes, with fixed and with moving relative positions."""

    def _assert_own_cg_loads_match(
        self, object_solver: object, array_solver: object, relative_tolerance: float
    ) -> None:
        """Asserts that every Airplane's moments and moment coefficients about its own
        CG match on both paths at every time step.

        :param object_solver: The solver run with use_arrays=False.
        :param array_solver: The solver run with use_arrays=True.
        :param relative_tolerance: The largest relative difference allowed.
        :return: None
        """
        object_problems = object_solver.steady_problems  # type: ignore[attr-defined]
        array_problems = array_solver.steady_problems  # type: ignore[attr-defined]
        for step, (object_problem, array_problem) in enumerate(
            zip(object_problems, array_problems)
        ):
            for airplane_id, (object_airplane, array_airplane) in enumerate(
                zip(object_problem.airplanes, array_problem.airplanes)
            ):
                for name in _OWN_CG_NAMES:
                    object_value = getattr(object_airplane, name)
                    array_value = getattr(array_airplane, name)
                    with self.subTest(step=step, airplane=airplane_id, name=name):
                        if object_value is None:
                            self.assertIsNone(array_value)
                            continue
                        np.testing.assert_allclose(
                            array_value,
                            object_value,
                            rtol=relative_tolerance,
                            atol=relative_tolerance
                            * float(np.max(np.abs(object_value))),
                        )

    def test_the_moving_fixtures_second_airplane_moves(self) -> None:
        """The moving formation's second Airplane's CG takes different positions over
        the run (the first Airplane's is always zeros), so the comparison below tests
        moments about a moving CG."""
        solver = (
            solver_fixtures.make_unsteady_ring_vortex_lattice_method_moving_formation_solver()
        )
        cgs = np.array(
            [problem.airplanes[1].Cg_GP1_CgP1 for problem in solver.steady_problems]
        )
        self.assertGreater(float(np.ptp(cgs, axis=0).min()), 0.1)
        for problem in solver.steady_problems:
            self.assertTrue(
                np.array_equal(problem.airplanes[0].Cg_GP1_CgP1, np.zeros(3))
            )

    def test_static_formation_matches_the_object_path(self) -> None:
        """A formation of two Airplanes in fixed relative positions gives the object
        path's loads at every time step, about every CG, to round off, with a prescribed
        and a free wake."""
        for prescribed_wake in (True, False):
            with self.subTest(prescribed_wake=prescribed_wake):
                object_solver, array_solver = (
                    array_path_fixtures.make_solvers_run_on_both_paths(
                        solver_fixtures.make_unsteady_ring_vortex_lattice_method_formation_solver,
                        prescribed_wake=prescribed_wake,
                    )
                )
                array_path_fixtures.assert_loads_match(
                    self, object_solver, array_solver
                )
                self._assert_own_cg_loads_match(
                    object_solver,
                    array_solver,
                    array_path_fixtures.DEFAULT_RELATIVE_TOLERANCE,
                )

    def test_moving_formation_matches_the_object_path(self) -> None:
        """A formation whose second Airplane moves and pitches gives the object path's
        loads at every time step, about every CG, to round off (see
        MOVING_FORMATION_RELATIVE_TOLERANCE), with a prescribed and a free wake."""
        for prescribed_wake in (True, False):
            with self.subTest(prescribed_wake=prescribed_wake):
                object_solver, array_solver = (
                    array_path_fixtures.make_solvers_run_on_both_paths(
                        solver_fixtures.make_unsteady_ring_vortex_lattice_method_moving_formation_solver,
                        prescribed_wake=prescribed_wake,
                    )
                )
                array_path_fixtures.assert_loads_match(
                    self,
                    object_solver,
                    array_solver,
                    array_path_fixtures.MOVING_FORMATION_RELATIVE_TOLERANCE,
                )
                self._assert_own_cg_loads_match(
                    object_solver,
                    array_solver,
                    array_path_fixtures.MOVING_FORMATION_RELATIVE_TOLERANCE,
                )


if __name__ == "__main__":
    unittest.main()
