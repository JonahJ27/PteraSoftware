"""This module contains classes to test the array path's load functions against the
object code they replace."""

import unittest
from typing import cast

import numpy as np

import pterasoftware as ps

# noinspection PyProtectedMember
from pterasoftware import _functions, _loads_arrays, _mesh_arrays, _step_geometry
from tests.unit.fixtures import (
    geometry_fixtures,
    operating_point_fixtures,
    problem_fixtures,
    solver_fixtures,
)

# The random strengths and loads the tests use.
_SEED = 4


def _prepared_solvers() -> (
    list[ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver]
):
    """Returns object path solvers of plain problems with one Wing, a pitching Wing, and
    several Airplanes, ready to have their geometry collapsed at any time step.

    :return: The solvers, with every time step's bound ring vortices initialized and the
        Katz force method selected, so collapsing a time step also fills the Katz chord
        and span lengths.
    """
    solvers = [
        ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(
            make()
        )
        for make in (
            problem_fixtures.make_basic_unsteady_problem_fixture,
            problem_fixtures.make_pitching_tip_unsteady_problem_fixture,
            problem_fixtures.make_multi_airplane_unsteady_problem_fixture,
        )
    ]
    for solver in solvers:
        solver._force_method = "katz"
        solver._initialize_panel_vortices()
    return solvers


def _collapse_at(
    solver: ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver,
    step: int,
    rng: np.random.Generator,
) -> _step_geometry.LayoutPlan:
    """Collapses a solver's geometry at a time step on the object path, gives it random
    current and last bound ring vortex strengths, and returns its LayoutPlan.

    :param solver: The solver, prepared by _prepared_solvers.
    :param step: The time step.
    :param rng: The random number generator for the strengths.
    :return: The solver's LayoutPlan.
    """
    steady_problem = solver.steady_problems[step]
    solver._current_step = step
    solver.current_airplanes = steady_problem.airplanes
    solver.current_operating_point = steady_problem.operating_point
    solver._prepare_step_geometry()
    solver._current_bound_vortex_strengths = rng.normal(size=solver.num_panels)
    solver._last_bound_vortex_strengths = rng.normal(size=solver.num_panels)
    return _step_geometry.build_layout_plan(steady_problem.airplanes)


class TestStrengthFunctions(unittest.TestCase):
    """This is a class with functions to test the strength based load functions against
    the solver's Panel loops they replace."""

    def test_effective_line_vortex_strengths(self) -> None:
        """Test that effective_line_vortex_strengths gives exactly the strengths the
        Kutta-Joukowski method's Panel loop finds, at the first time step (with no wake
        to cancel the trailing edge back line vortices) and afterwards."""
        rng = np.random.default_rng(_SEED)
        for solver_id, solver in enumerate(_prepared_solvers()):
            for step in sorted({0, 1, solver.num_steps - 1}):
                with self.subTest(solver=solver_id, step=step):
                    layout_plan = _collapse_at(solver, step, rng)
                    expected = solver._effective_line_vortex_strengths_from_panels()
                    actual = _loads_arrays.effective_line_vortex_strengths(
                        layout_plan,
                        solver._current_bound_vortex_strengths,
                        solver._last_bound_vortex_strengths,
                        step,
                    )
                    for name, actual_array, expected_array in zip(
                        ("right", "front", "left", "back"), actual, expected
                    ):
                        self.assertTrue(
                            np.array_equal(actual_array, expected_array), name
                        )

    def test_katz_vorticity_functions(self) -> None:
        """Test that the three Katz functions give exactly the chordwise and spanwise
        vorticity densities and chordwise vorticity differences the Katz method's Panel
        loops find."""
        rng = np.random.default_rng(_SEED)
        for solver_id, solver in enumerate(_prepared_solvers()):
            for step in sorted({0, solver.num_steps - 1}):
                with self.subTest(solver=solver_id, step=step):
                    layout_plan = _collapse_at(solver, step, rng)
                    strengths = solver._current_bound_vortex_strengths
                    self.assertTrue(
                        np.array_equal(
                            _loads_arrays.chordwise_vorticity_densities(
                                layout_plan, strengths, solver._panel_chord_lengths
                            ),
                            solver._calculate_chordwise_vorticity_densities(),
                        )
                    )
                    self.assertTrue(
                        np.array_equal(
                            _loads_arrays.spanwise_vorticity_densities(
                                layout_plan, strengths, solver._panel_span_lengths
                            ),
                            solver._calculate_spanwise_vorticity_densities(),
                        )
                    )
                    self.assertTrue(
                        np.array_equal(
                            _loads_arrays.chordwise_vorticity_differences(
                                layout_plan, strengths
                            ),
                            solver._calculate_chordwise_vorticity_differences(),
                        )
                    )

    def test_densities_are_zero_without_length(self) -> None:
        """Test that a Panel with no chord or span length gets a zero density, as on the
        object path."""
        solver = _prepared_solvers()[0]
        layout_plan = _collapse_at(solver, 0, np.random.default_rng(_SEED))
        no_lengths = np.zeros(layout_plan.num_panels, dtype=float)
        strengths = solver._current_bound_vortex_strengths
        self.assertTrue(
            np.all(
                _loads_arrays.chordwise_vorticity_densities(
                    layout_plan, strengths, no_lengths
                )
                == 0.0
            )
        )
        self.assertTrue(
            np.all(
                _loads_arrays.spanwise_vorticity_densities(
                    layout_plan, strengths, no_lengths
                )
                == 0.0
            )
        )


class TestStepLoads(unittest.TestCase):
    """This is a class with functions to test step_loads and apply_step_loads_to_objects
    against process_solver_loads."""

    def _assert_matches_process_solver_loads(
        self,
        solver: (
            ps.steady_ring_vortex_lattice_method.SteadyRingVortexLatticeMethodSolver
            | ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver
        ),
        airplanes: tuple[ps.geometry.airplane.Airplane, ...],
        operating_point: ps.operating_point.OperatingPoint,
        rng: np.random.Generator,
    ) -> None:
        """Asserts that step_loads gives exactly the loads process_solver_loads sets on
        a solver's Panels and Airplanes for random Panel loads, that each Wing's totals
        are its Panels' loads summed in order, and that apply_step_loads_to_objects
        writes the same values onto the objects.

        :param solver: The solver, whose panels attribute holds its Panels in global
            order.
        :param airplanes: The solver's Airplanes.
        :param operating_point: The solver's OperatingPoint.
        :param rng: The random number generator for the loads.
        :return: None
        """
        layout_plan = _step_geometry.build_layout_plan(airplanes)
        forces_GP1 = rng.normal(size=(layout_plan.num_panels, 3))
        moments_GP1_CgP1 = rng.normal(size=(layout_plan.num_panels, 3))
        loads = _loads_arrays.step_loads(
            layout_plan,
            forces_GP1,
            moments_GP1_CgP1,
            operating_point.T_pas_GP1_CgP1_to_W_CgP1,
            operating_point.qInf__E,
            [airplane.Cg_GP1_CgP1 for airplane in airplanes],
            [
                (airplane.s_ref, airplane.c_ref, airplane.b_ref)
                for airplane in airplanes
            ],
        )
        _functions.process_solver_loads(solver, forces_GP1, moments_GP1_CgP1)

        panel_names = (
            ("stackPanelForces_GP1", "forces_GP1"),
            ("stackPanelForces_W", "forces_W"),
            ("stackPanelMoments_GP1_CgP1", "moments_GP1_CgP1"),
            ("stackPanelMoments_W_CgP1", "moments_W_CgP1"),
        )
        airplane_names = (
            ("stackAirplaneForces_W", "forces_W"),
            ("stackAirplaneForces_G", "forces_G"),
            ("stackAirplaneMoments_W_CgP1", "moments_W_CgP1"),
            ("stackAirplaneMoments_G_Cg", "moments_G_Cg"),
            ("stackAirplaneMoments_W_Cg", "moments_W_Cg"),
            ("stackAirplaneForceCoefficients_W", "forceCoefficients_W"),
            ("stackAirplaneForceCoefficients_G", "forceCoefficients_G"),
            ("stackAirplaneMomentCoefficients_W_CgP1", "momentCoefficients_W_CgP1"),
            ("stackAirplaneMomentCoefficients_G_Cg", "momentCoefficients_G_Cg"),
            ("stackAirplaneMomentCoefficients_W_Cg", "momentCoefficients_W_Cg"),
        )
        assert solver.panels is not None
        expected_panels = {
            field: np.array([getattr(panel, name) for panel in solver.panels])
            for field, name in panel_names
        }
        expected_airplanes = {
            field: np.array([getattr(airplane, name) for airplane in airplanes])
            for field, name in airplane_names
        }
        for field, _ in panel_names:
            self.assertTrue(
                np.array_equal(getattr(loads, field), expected_panels[field]), field
            )
        for field, _ in airplane_names:
            self.assertTrue(
                np.array_equal(getattr(loads, field), expected_airplanes[field]), field
            )

        # Each Wing's totals, summed over its Panels in order, as the flap driven
        # problem sums them.
        wing_id = 0
        for airplane in airplanes:
            for wing in airplane.wings:
                assert wing.panels is not None
                wingForces_GP1 = np.zeros(3, dtype=float)
                wingMoments_GP1_CgP1 = np.zeros(3, dtype=float)
                for panel in np.ravel(wing.panels):
                    wingForces_GP1 += panel.forces_GP1
                    wingMoments_GP1_CgP1 += panel.moments_GP1_CgP1
                self.assertTrue(
                    np.array_equal(loads.stackWingForces_GP1[wing_id], wingForces_GP1)
                )
                self.assertTrue(
                    np.array_equal(
                        loads.stackWingMoments_GP1_CgP1[wing_id], wingMoments_GP1_CgP1
                    )
                )
                wing_id += 1

        for array in loads:
            self.assertFalse(array.flags.writeable)

        # Writing the loads onto the objects gives them the same values, in writeable
        # arrays, as process_solver_loads does. Clear every load first, so a load the
        # function doesn't write is caught.
        for panel in solver.panels:
            for _, name in panel_names:
                setattr(panel, name, None)
        for airplane in airplanes:
            for _, name in airplane_names:
                setattr(airplane, name, None)
        _loads_arrays.apply_step_loads_to_objects(loads, solver.panels, airplanes)
        for field, name in panel_names:
            written = np.array([getattr(panel, name) for panel in solver.panels])
            self.assertTrue(np.array_equal(written, expected_panels[field]), field)
        for field, name in airplane_names:
            written = np.array([getattr(airplane, name) for airplane in airplanes])
            self.assertTrue(np.array_equal(written, expected_airplanes[field]), field)
        first_panel_forces_GP1 = solver.panels[0].forces_GP1
        first_airplane_forces_W = airplanes[0].forces_W
        assert first_panel_forces_GP1 is not None
        assert first_airplane_forces_W is not None
        self.assertTrue(first_panel_forces_GP1.flags.writeable)
        self.assertTrue(first_airplane_forces_W.flags.writeable)

    def test_matches_process_solver_loads_on_unsteady_solvers(self) -> None:
        """Test step_loads against process_solver_loads on plain solvers with one Wing,
        a pitching Wing, and several Airplanes."""
        rng = np.random.default_rng(_SEED)
        for solver_id, solver in enumerate(_prepared_solvers()):
            with self.subTest(solver=solver_id):
                _collapse_at(solver, solver.num_steps - 1, rng)
                self._assert_matches_process_solver_loads(
                    solver,
                    solver.current_airplanes,
                    solver.current_operating_point,
                    rng,
                )

    def test_matches_process_solver_loads_with_an_offset_airplane(self) -> None:
        """Test step_loads against process_solver_loads on a problem whose second
        Airplane's CG is away from the first Airplane's, so its moments about its own CG
        differ from those about the first Airplane's CG."""
        steady_problem = ps.problems.SteadyProblem(
            airplanes=[
                geometry_fixtures.make_first_airplane_fixture(),
                geometry_fixtures.make_follower_airplane_fixture(),
            ],
            operating_point=operating_point_fixtures.make_basic_operating_point_fixture(),
        )
        solver = (
            ps.steady_ring_vortex_lattice_method.SteadyRingVortexLatticeMethodSolver(
                steady_problem
            )
        )
        solver.panels = np.array(
            [
                panel
                for airplane in steady_problem.airplanes
                for wing in airplane.wings
                for panel in np.ravel(cast(np.ndarray, wing.panels))
            ],
            dtype=object,
        )
        self._assert_matches_process_solver_loads(
            solver,
            steady_problem.airplanes,
            steady_problem.operating_point,
            np.random.default_rng(_SEED),
        )

    def test_raises_the_object_paths_shape_errors(self) -> None:
        """Test that step_loads raises process_solver_loads' ValueErrors, with the same
        messages, for Panel forces or moments with the wrong number of rows."""
        solver = _prepared_solvers()[0]
        layout_plan = _collapse_at(solver, 0, np.random.default_rng(_SEED))
        operating_point = solver.current_operating_point
        airplanes = solver.current_airplanes
        good = np.zeros((layout_plan.num_panels, 3), dtype=float)
        bad = np.zeros((layout_plan.num_panels + 1, 3), dtype=float)
        for forces, moments in ((bad, good), (good, bad)):
            with self.subTest(forces_rows=forces.shape[0]):
                with self.assertRaises(ValueError) as expected:
                    _functions.process_solver_loads(solver, forces, moments)
                with self.assertRaises(ValueError) as actual:
                    _loads_arrays.step_loads(
                        layout_plan,
                        forces,
                        moments,
                        operating_point.T_pas_GP1_CgP1_to_W_CgP1,
                        operating_point.qInf__E,
                        [airplane.Cg_GP1_CgP1 for airplane in airplanes],
                        [
                            (airplane.s_ref, airplane.c_ref, airplane.b_ref)
                            for airplane in airplanes
                        ],
                    )
                self.assertEqual(str(actual.exception), str(expected.exception))


class TestStripLeadingEdgeFunctions(unittest.TestCase):
    """This is a class with functions to test the strip leading edge functions against
    the aeroelastic solver's hook."""

    def test_match_the_aeroelastic_hook(self) -> None:
        """Test that strip_leading_edge_points gives exactly the points the aeroelastic
        solver's _populate_slep_positions gathers from the Panels, and that
        strip_leading_edge_moments gives exactly the moments its hook finds, for random
        Panel loads."""
        solver = solver_fixtures.make_aeroelastic_unsteady_ring_solver_fixture()
        solver.run(show_progress=False)
        rng = np.random.default_rng(_SEED)
        step = solver.num_steps - 1
        steady_problem = solver._get_steady_problem_at(step)

        # The object path, on the last time step's Panels.
        solver._populate_slep_positions()
        forces_GP1 = rng.normal(size=(solver.num_panels, 3))
        moments_GP1_CgP1 = rng.normal(size=(solver.num_panels, 3))
        solver._process_panel_loads_hook(forces_GP1, moments_GP1_CgP1)

        # The array path, on the same time step's geometry.
        layout_plan = _step_geometry.build_layout_plan(steady_problem.airplanes)
        wing_meshes = []
        wingsCg_GP1_CgP1 = []
        for airplane in steady_problem.airplanes:
            for wing in airplane.wings:
                wing_meshes.append(_mesh_arrays.wing_mesh_from_wing(wing))
                wingsCg_GP1_CgP1.append(airplane.Cg_GP1_CgP1)
        geometry = _step_geometry.build_step_geometry(
            layout_plan,
            wing_meshes,
            wingsCg_GP1_CgP1,
            steady_problem.operating_point.vInf_GP1__E,
            solver.delta_time,
            None,
        )
        stackSlep_GP1_CgP1 = _loads_arrays.strip_leading_edge_points(
            geometry, solver._slep_outboard_is_left, solver._slep_point_indices
        )
        self.assertTrue(np.array_equal(stackSlep_GP1_CgP1, solver._stackSlep_GP1_CgP1))
        self.assertTrue(
            np.array_equal(
                _loads_arrays.strip_leading_edge_moments(
                    forces_GP1, moments_GP1_CgP1, stackSlep_GP1_CgP1
                ),
                solver.moments_GP1_Slep,
            )
        )


if __name__ == "__main__":
    unittest.main()
