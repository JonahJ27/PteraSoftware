"""This module contains a class to test that everything a user can read from a run is
the same on the array path and on the object path."""

import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pterasoftware as ps
from tests.integration.fixtures import (
    array_path_fixtures,
    flap_drive_fixtures,
    problem_fixtures,
    solver_fixtures,
)
from tests.integration.test_aeroelastic_unsteady_ring_vortex_lattice_method import (
    _make_aeroelastic_solver,
)
from tests.integration.test_unsteady_ring_vortex_lattice_method_wake_truncation import (
    _make_static_movement_components,
)


def _run(make_solver: Callable[[], Any], use_arrays: bool, **run_arguments: Any) -> Any:
    """Makes a solver and runs it.

    :param make_solver: A function that makes a new solver.
    :param use_arrays: Whether to run on the array path.
    :param run_arguments: The other run arguments.
    :return: The solver, run.
    """
    solver = make_solver()
    solver.run(show_progress=False, use_arrays=use_arrays, **run_arguments)
    return solver


class TestArrayPathPublicResults(unittest.TestCase):
    """Verifies that every public result of a run, for one problem of each solver type,
    is the same on both paths, including every time step's objects, which an array path
    run only builds when they are read."""

    def _assert_match(
        self,
        make_solver: Callable[[], Any],
        relative_tolerance: float,
        **run_arguments: Any,
    ) -> None:
        """Runs a problem on both paths and asserts that every public result matches.

        :param make_solver: A function that makes a new solver of the problem.
        :param relative_tolerance: The largest relative difference allowed.
        :param run_arguments: The run arguments both runs use.
        :return: None
        """
        object_solver = _run(make_solver, False, **run_arguments)
        array_solver = _run(make_solver, True, **run_arguments)
        num_values = array_path_fixtures.assert_public_results_match(
            self, object_solver, array_solver, relative_tolerance
        )
        # Every time step's objects were compared, so there are many values.
        self.assertGreater(num_values, 100 * object_solver.num_steps)

    def test_plain_variable_geometry(self) -> None:
        """A plain solver with variable geometry and a prescribed wake."""
        self._assert_match(
            solver_fixtures.make_unsteady_ring_vortex_lattice_method_validation_solver_with_variable_geometry,
            array_path_fixtures.DEFAULT_RELATIVE_TOLERANCE,
            prescribed_wake=True,
        )

    def test_plain_multiple_wings(self) -> None:
        """A plain solver with two Wings, variable geometry, and a free wake."""
        self._assert_match(
            solver_fixtures.make_unsteady_ring_vortex_lattice_method_validation_solver_with_multiple_wing_variable_geometry,
            array_path_fixtures.DEFAULT_RELATIVE_TOLERANCE,
            prescribed_wake=False,
        )

    def test_plain_wake_truncation(self) -> None:
        """A plain solver whose wake is truncated to 3 chord lengths, with a free wake,
        so the wake point grids written onto the Wings after the run are truncated."""

        def make_solver() -> Any:
            airplane_movement, operating_point_movement = (
                _make_static_movement_components()
            )
            reference = ps.movements.movement.Movement(
                airplane_movements=[airplane_movement],
                operating_point_movement=operating_point_movement,
                num_chords=6,
            )
            movement = ps.movements.movement.Movement(
                airplane_movements=[airplane_movement],
                operating_point_movement=operating_point_movement,
                delta_time=reference.delta_time,
                num_steps=reference.num_steps,
                max_wake_chords=3,
            )
            assert movement.max_wake_rows is not None
            assert movement.max_wake_rows < movement.num_steps - 1
            return ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(
                ps.problems.UnsteadyProblem(movement=movement)
            )

        self._assert_match(
            make_solver,
            array_path_fixtures.DEFAULT_RELATIVE_TOLERANCE,
            prescribed_wake=False,
        )

    def test_moving_formation(self) -> None:
        """A plain solver with two Airplanes, the second moving and pitching, and a free
        wake, so each Airplane's loads about its own CG change every time step (see
        MOVING_FORMATION_RELATIVE_TOLERANCE)."""
        self._assert_match(
            solver_fixtures.make_unsteady_ring_vortex_lattice_method_moving_formation_solver,
            array_path_fixtures.MOVING_FORMATION_RELATIVE_TOLERANCE,
            prescribed_wake=False,
        )

    def test_aeroelastic(self) -> None:
        """An aeroelastic solver with the Katz force method and a free wake."""
        self._assert_match(
            lambda: _make_aeroelastic_solver(wing_density=0.01),
            array_path_fixtures.DEFAULT_RELATIVE_TOLERANCE,
            prescribed_wake=False,
            force_method="katz",
        )

    def test_flap_driven(self) -> None:
        """A flap driven aeroelastic solver with a free wake."""
        self._assert_match(
            lambda: ps.aeroelastic_unsteady_ring_vortex_lattice_method.AeroelasticUnsteadyRingVortexLatticeMethodSolver(
                aeroelastic_unsteady_problem=flap_drive_fixtures.make_flap_driven_problem()
            ),
            array_path_fixtures.FLAP_DRIVEN_RELATIVE_TOLERANCE,
            prescribed_wake=False,
        )

    def test_free_flight_glider(self) -> None:
        """A free flight solver with a prescribed wake."""
        self._assert_match(
            solver_fixtures.make_simple_glider_free_flight_solver,
            array_path_fixtures.FREE_FLIGHT_GLIDER_RELATIVE_TOLERANCE,
            prescribed_wake=True,
        )

    def test_free_flight_flap_driven(self) -> None:
        """A free flight flap driven aeroelastic solver with a free wake, whose free
        flight is chaotic, so its tolerance is loose (see
        FREE_FLIGHT_FLAP_DRIVEN_RELATIVE_TOLERANCE)."""
        self._assert_match(
            lambda: ps.free_flight_aeroelastic_unsteady_ring_vortex_lattice_method.FreeFlightAeroelasticUnsteadyRingVortexLatticeMethodSolver(
                flap_drive_fixtures.make_free_flight_flap_driven_problem()
            ),
            array_path_fixtures.FREE_FLIGHT_FLAP_DRIVEN_RELATIVE_TOLERANCE,
            prescribed_wake=False,
        )


class TestArrayPathOutputAndSerialization(unittest.TestCase):
    """Verifies that output, and saving and loading, work on an array path run whose
    time steps haven't been built, and give what they give for an object path run."""

    def test_output_builds_the_time_steps_it_reads(self) -> None:
        """Test that draw, animate, plot_results_versus_time, and log_results accept a
        free flight flap driven array path run whose time steps haven't been built, and
        that reading them builds them."""
        solver = _run(
            lambda: ps.free_flight_aeroelastic_unsteady_ring_vortex_lattice_method.FreeFlightAeroelasticUnsteadyRingVortexLatticeMethodSolver(
                flap_drive_fixtures.make_free_flight_flap_driven_problem()
            ),
            True,
            prescribed_wake=False,
        )
        problem: Any = solver.unsteady_problem
        self.assertTrue(
            any(
                not isinstance(entry, ps.problems.SteadyProblem)
                for entry in problem._steady_problems
            )
        )
        ps.output.draw(
            solver=solver,
            scalar_type="lift",
            show_wake_vortices=True,
            show_streamlines=False,
            testing=True,
        )
        ps.output.animate(
            unsteady_solver=solver,
            scalar_type="lift",
            show_wake_vortices=True,
            save=False,
            testing=True,
        )
        ps.output.plot_results_versus_time(unsteady_solver=solver, show=False)
        ps.output.log_results(solver=solver)
        self.assertTrue(
            all(
                isinstance(entry, ps.problems.SteadyProblem)
                for entry in problem._steady_problems
            )
        )

    def test_save_and_load_match_the_object_path(self) -> None:
        """Test that saving an aeroelastic array path run whose time steps haven't been
        built, and loading it, gives everything a user can read that saving and loading
        the same object path run gives."""
        loaded: dict[bool, Any] = {}
        with tempfile.TemporaryDirectory() as directory:
            for use_arrays in (False, True):
                solver = _run(
                    lambda: _make_aeroelastic_solver(wing_density=0.01),
                    use_arrays,
                    prescribed_wake=False,
                    force_method="katz",
                )
                path = Path(directory) / f"solver_{use_arrays}.psz"
                ps.save(path, solver)
                loaded[use_arrays] = ps.load(path)
        num_values = array_path_fixtures.assert_public_results_match(
            self,
            loaded[False],
            loaded[True],
            array_path_fixtures.DEFAULT_RELATIVE_TOLERANCE,
        )
        self.assertGreater(num_values, 100 * loaded[False].num_steps)


if __name__ == "__main__":
    unittest.main()
