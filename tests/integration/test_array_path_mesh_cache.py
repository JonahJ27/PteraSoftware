"""This module contains a class to test that the array path's mesh cache changes no
result: runs with it give everything a user can read bit for bit the same as runs that
mesh every Wing at every geometry build."""

import unittest
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import pterasoftware as ps
from pterasoftware import _mesh_arrays
from tests.integration.fixtures import (
    array_path_fixtures,
    flap_drive_fixtures,
    solver_fixtures,
)
from tests.integration.test_aeroelastic_unsteady_ring_vortex_lattice_method import (
    _make_aeroelastic_solver,
)


def _run_counting_meshings(
    make_solver: Callable[[], Any], cache: bool, **run_arguments: Any
) -> tuple[Any, int]:
    """Makes a solver, runs it on the array path, and counts its meshings.

    :param make_solver: A function that makes a new solver.
    :param cache: Whether the mesh cache is used. Without it, every geometry build
        meshes every Wing, because each build's key matches no other.
    :param run_arguments: The other run arguments.
    :return: The solver, run, and the number of times it meshed a Wing.
    """
    original = _mesh_arrays.mesh_wing_arrays
    count = {"meshings": 0}

    def counting_mesh_wing_arrays(*args: Any) -> Any:
        count["meshings"] += 1
        return original(*args)

    solver = make_solver()
    patches: list[Any] = [
        patch.object(_mesh_arrays, "mesh_wing_arrays", counting_mesh_wing_arrays)
    ]
    if not cache:
        patches.append(
            patch(
                "pterasoftware._step_parameters.wing_step_parameters_key",
                lambda parameters: object(),
            )
        )
    for active in patches:
        active.start()
    try:
        solver.run(show_progress=False, use_arrays=True, **run_arguments)
    finally:
        for active in patches:
            active.stop()
    return solver, count["meshings"]


class TestArrayPathMeshCache(unittest.TestCase):
    """Verifies that the mesh cache gives every public result bit for bit, on problems
    whose Wings deform, flap, fly freely, or belong to several Airplanes, and that it
    saves meshings where the coupling trials mesh unchanged Wings again."""

    def _assert_unchanged(
        self,
        make_solver: Callable[[], Any],
        expect_fewer_meshings: bool,
        **run_arguments: Any,
    ) -> None:
        """Runs a problem with and without the mesh cache, and asserts that every public
        result is bit for bit the same and that the cache meshes no more often.

        :param make_solver: A function that makes a new solver of the problem.
        :param expect_fewer_meshings: Whether the cache must save meshings, as it does
            for a free flight problem, whose coupling trials mesh unchanged Wings again.
        :param run_arguments: The run arguments both runs use.
        :return: None
        """
        uncached, uncached_meshings = _run_counting_meshings(
            make_solver, False, **run_arguments
        )
        cached, cached_meshings = _run_counting_meshings(
            make_solver, True, **run_arguments
        )
        num_values = array_path_fixtures.assert_public_results_match(
            self, uncached, cached, 0.0
        )
        self.assertGreater(num_values, 100 * uncached.num_steps)
        self.assertLessEqual(cached_meshings, uncached_meshings)
        if expect_fewer_meshings:
            self.assertLess(cached_meshings, uncached_meshings / 2)
        self.assertIsNone(cached._wing_mesh_cache)

    def test_free_flight_flap_driven(self) -> None:
        """A free flight flap driven aeroelastic problem with a free wake: its Wings
        flap and deform every time step, and its coupling trials mesh them again."""
        self._assert_unchanged(
            lambda: ps.free_flight_aeroelastic_unsteady_ring_vortex_lattice_method.FreeFlightAeroelasticUnsteadyRingVortexLatticeMethodSolver(
                flap_drive_fixtures.make_free_flight_flap_driven_problem()
            ),
            True,
            prescribed_wake=False,
        )

    def test_free_flight_glider(self) -> None:
        """A free flight glider with a prescribed wake, whose Wings never change."""
        self._assert_unchanged(
            solver_fixtures.make_simple_glider_free_flight_solver,
            True,
            prescribed_wake=True,
        )

    def test_aeroelastic(self) -> None:
        """An aeroelastic problem whose Wings deform every time step, with the Katz
        force method and a free wake."""
        self._assert_unchanged(
            lambda: _make_aeroelastic_solver(wing_density=0.01),
            False,
            prescribed_wake=False,
            force_method="katz",
        )

    def test_moving_formation(self) -> None:
        """Two Airplanes, the second moving and pitching, with a free wake, so each Wing
        has its own cache entry."""
        self._assert_unchanged(
            solver_fixtures.make_unsteady_ring_vortex_lattice_method_moving_formation_solver,
            False,
            prescribed_wake=False,
        )


if __name__ == "__main__":
    unittest.main()
