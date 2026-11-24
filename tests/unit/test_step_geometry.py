"""This module contains a class to test the array path's per run Panel layout."""

import unittest

import numpy as np

import pterasoftware as ps

# noinspection PyProtectedMember
from pterasoftware import _step_geometry
from tests.unit.fixtures import problem_fixtures, solver_fixtures


class TestBuildLayoutPlan(unittest.TestCase):
    """This is a class with functions to test build_layout_plan."""

    def test_plan_matches_the_solvers_panel_layout(self) -> None:
        """Test that the LayoutPlan's Panel count and edge flags equal those the solver
        computes from the Panels when it collapses the first time step's geometry, and
        that its Wing shapes and offsets tile the Panels in order, for one and several
        Airplanes."""
        solvers = (
            solver_fixtures.make_unsteady_ring_solver_fixture(),
            ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(
                problem_fixtures.make_multi_airplane_unsteady_problem_fixture()
            ),
            solver_fixtures.make_aeroelastic_unsteady_ring_solver_fixture(),
        )
        for solver_index, solver in enumerate(solvers):
            with self.subTest(solver_index=solver_index):
                # Build and collapse the first time step's geometry, which sets the
                # solver's edge flags from the Panels, without solving it.
                first_steady_problem = solver.steady_problems[0]
                solver._current_step = 0
                solver._initialize_step_vortices(0)
                solver.current_airplanes = first_steady_problem.airplanes
                solver.current_operating_point = first_steady_problem.operating_point
                solver._prepare_step_geometry()
                airplanes = first_steady_problem.airplanes
                plan = _step_geometry.build_layout_plan(airplanes)

                self.assertEqual(plan.num_panels, solver.num_panels)
                for name in (
                    "is_leading_edge",
                    "is_trailing_edge",
                    "is_left_edge",
                    "is_right_edge",
                ):
                    self.assertTrue(
                        np.array_equal(
                            getattr(plan, name), getattr(solver, f"panel_{name}")
                        ),
                        name,
                    )
                    self.assertFalse(getattr(plan, name).flags.writeable)

                wings = [wing for airplane in airplanes for wing in airplane.wings]
                self.assertEqual(len(plan.wing_shapes), len(wings))
                offset = 0
                for wing, shape, wing_offset in zip(
                    wings, plan.wing_shapes, plan.wing_offsets
                ):
                    assert wing.panels is not None
                    self.assertEqual(shape, wing.panels.shape)
                    self.assertEqual(wing_offset, offset)
                    offset += shape[0] * shape[1]
                self.assertEqual(offset, plan.num_panels)


if __name__ == "__main__":
    unittest.main()
