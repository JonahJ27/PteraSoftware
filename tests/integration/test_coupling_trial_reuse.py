"""This module contains a class to test that reusing work across a free flight time
step's coupling trials changes no result: the Wing Wing influence columns of unchanged
bound ring vortices, and the step geometry built from unchanged WingMeshes.

Runs with the reuse give everything a user can read bit for bit the same as runs without
it.
"""

import unittest
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import pterasoftware as ps
from pterasoftware import _step_geometry
from tests.integration.fixtures import (
    array_path_fixtures,
    flap_drive_fixtures,
    solver_fixtures,
)
from tests.integration.test_aeroelastic_unsteady_ring_vortex_lattice_method import (
    _make_aeroelastic_solver,
)

_SOLVER = ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver


def _run(
    make_solver: Callable[[], Any], reuse: bool, use_arrays: bool, **run_arguments: Any
) -> tuple[Any, dict[str, int]]:
    """Makes a solver and runs it, with or without the coupling trial reuse, counting
    how often each reuse was taken.

    :param make_solver: A function that makes a new solver.
    :param reuse: Whether the reuse is allowed. Without it, the caches are cleared
        before every use, so every calculation is made in full.
    :param use_arrays: Whether to run on the array path.
    :param run_arguments: The other run arguments.
    :return: The solver, run, and the counts of partial influence calculations (those
        that didn't make the full calculation) and of geometry rebuilds.
    """
    calls = {"influences": 0, "full influences": 0, "geometry rebuilds": 0}
    original_influences = _SOLVER._calculate_wing_wing_influences
    original_full = _SOLVER._full_wing_wing_influences
    original_geometry = _SOLVER._build_step_geometry_at
    original_rebuild = _step_geometry.rebuild_step_geometry_for_freestream

    def influences(self: Any) -> None:
        calls["influences"] += 1
        if not reuse:
            self._wing_wing_influence_cache = None
        original_influences(self)

    def full(self: Any, *args: Any) -> Any:
        calls["full influences"] += 1
        return original_full(self, *args)

    def geometry(self: Any, step: int) -> None:
        if not reuse:
            self._last_step_geometry_build = None
        original_geometry(self, step)

    def rebuild(*args: Any, **kwargs: Any) -> Any:
        calls["geometry rebuilds"] += 1
        return original_rebuild(*args, **kwargs)

    solver = make_solver()
    patches: list[Any] = [
        patch.object(_SOLVER, "_calculate_wing_wing_influences", influences),
        patch.object(_SOLVER, "_full_wing_wing_influences", full),
        patch.object(_SOLVER, "_build_step_geometry_at", geometry),
        patch.object(_step_geometry, "rebuild_step_geometry_for_freestream", rebuild),
    ]
    for active in patches:
        active.start()
    try:
        solver.run(show_progress=False, use_arrays=use_arrays, **run_arguments)
    finally:
        for active in patches:
            active.stop()
    return solver, {
        "partial influences": calls["influences"] - calls["full influences"],
        "geometry rebuilds": calls["geometry rebuilds"],
    }


class TestCouplingTrialReuse(unittest.TestCase):
    """Verifies that the coupling trial reuse gives every public result bit for bit, on
    problems whose Wings deform, flap, fly freely, fly near the ground, or belong to
    several Airplanes, and that it is taken where the trials share their inputs."""

    def _assert_unchanged(
        self,
        make_solver: Callable[[], Any],
        use_arrays: bool,
        expect_reuse: bool,
        **run_arguments: Any,
    ) -> None:
        """Runs a problem with and without the reuse, and asserts that every public
        result is bit for bit the same, and whether the reuse was taken.

        :param make_solver: A function that makes a new solver of the problem.
        :param use_arrays: Whether to run on the array path.
        :param expect_reuse: Whether both reuses must be taken, as in free flight
            without ground effect.
        :param run_arguments: The run arguments both runs use.
        :return: None
        """
        without, without_counts = _run(make_solver, False, use_arrays, **run_arguments)
        with_reuse, counts = _run(make_solver, True, use_arrays, **run_arguments)
        num_values = array_path_fixtures.assert_public_results_match(
            self, without, with_reuse, 0.0
        )
        self.assertGreater(num_values, 100 * without.num_steps)
        self.assertEqual(without_counts["partial influences"], 0)
        self.assertEqual(without_counts["geometry rebuilds"], 0)
        # Free flight's coupling trials must take both reuses. Other problems may take
        # them too (the aeroelastic problem builds some time steps' geometry twice),
        # which the results above show is harmless.
        if expect_reuse:
            self.assertGreater(counts["partial influences"], 0)
            if use_arrays:
                self.assertGreater(counts["geometry rebuilds"], 0)
        self.assertIsNone(with_reuse._wing_wing_influence_cache)
        self.assertIsNone(with_reuse._last_step_geometry_build)

    def test_free_flight_flap_driven(self) -> None:
        """A free flight flap driven aeroelastic problem with a free wake, on both
        paths."""
        for use_arrays in (True, False):
            with self.subTest(use_arrays=use_arrays):
                self._assert_unchanged(
                    lambda: ps.free_flight_aeroelastic_unsteady_ring_vortex_lattice_method.FreeFlightAeroelasticUnsteadyRingVortexLatticeMethodSolver(
                        flap_drive_fixtures.make_free_flight_flap_driven_problem()
                    ),
                    use_arrays,
                    True,
                    prescribed_wake=False,
                )

    def test_free_flight_glider(self) -> None:
        """A free flight glider with a prescribed wake."""
        self._assert_unchanged(
            solver_fixtures.make_simple_glider_free_flight_solver,
            True,
            True,
            prescribed_wake=True,
        )

    def test_free_flight_glider_near_the_ground(self) -> None:
        """The glider 1 m above the ground: the influences are always calculated in full
        (their image follows the trial's body state), and the geometry is still rebuilt
        from the last build."""
        without, _ = _run(
            lambda: solver_fixtures.make_simple_glider_free_flight_solver(
                ground_height=1.0
            ),
            False,
            True,
            prescribed_wake=True,
        )
        with_reuse, counts = _run(
            lambda: solver_fixtures.make_simple_glider_free_flight_solver(
                ground_height=1.0
            ),
            True,
            True,
            prescribed_wake=True,
        )
        array_path_fixtures.assert_public_results_match(self, without, with_reuse, 0.0)
        self.assertEqual(counts["partial influences"], 0)
        self.assertGreater(counts["geometry rebuilds"], 0)

    def test_geometry_is_rebuilt_only_from_the_same_inputs(self) -> None:
        """Test that the step geometry is rebuilt from the last build only when the time
        step, the CGs, and the previous geometry are all the last build's: in a free
        flight run, each build that would reuse the last one instead finds it tampered
        with in one of those (in turn), and must then build in full, with every public
        result still unchanged."""
        tampers = ("time step", "CGs", "previous geometry")
        forced = {name: 0 for name in tampers}
        calls = {"full": 0, "rebuild": 0}
        original_geometry = _SOLVER._build_step_geometry_at
        original_build = _step_geometry.build_step_geometry
        original_rebuild = _step_geometry.rebuild_step_geometry_for_freestream

        def build(*args: Any, **kwargs: Any) -> Any:
            calls["full"] += 1
            return original_build(*args, **kwargs)

        def rebuild(*args: Any, **kwargs: Any) -> Any:
            calls["rebuild"] += 1
            return original_rebuild(*args, **kwargs)

        def tampering_geometry(solver_self: Any, step: int) -> None:
            last_build = solver_self._last_step_geometry_build
            if last_build is None or last_build[0] != step:
                original_geometry(solver_self, step)
                return
            name = tampers[sum(forced.values()) % len(tampers)]
            if name == "time step":
                tampered = (step + 1, *last_build[1:])
            elif name == "CGs":
                tampered = (*last_build[:2], b"other", *last_build[3:])
            else:
                tampered = (*last_build[:3], object(), last_build[4])
            solver_self._last_step_geometry_build = tampered
            before = dict(calls)
            original_geometry(solver_self, step)
            self.assertEqual(calls["rebuild"], before["rebuild"], msg=name)
            self.assertEqual(calls["full"], before["full"] + 1, msg=name)
            forced[name] += 1

        without, _ = _run(
            solver_fixtures.make_simple_glider_free_flight_solver,
            False,
            True,
            prescribed_wake=True,
        )
        solver = solver_fixtures.make_simple_glider_free_flight_solver()
        patches: list[Any] = [
            patch.object(_SOLVER, "_build_step_geometry_at", tampering_geometry),
            patch.object(_step_geometry, "build_step_geometry", build),
            patch.object(
                _step_geometry, "rebuild_step_geometry_for_freestream", rebuild
            ),
        ]
        for active in patches:
            active.start()
        try:
            solver.run(show_progress=False, use_arrays=True, prescribed_wake=True)
        finally:
            for active in patches:
                active.stop()
        array_path_fixtures.assert_public_results_match(self, without, solver, 0.0)
        for name in tampers:
            self.assertGreater(forced[name], 0, msg=name)

    def test_aeroelastic(self) -> None:
        """An aeroelastic problem with the Katz force method and a free wake, which has
        no coupling trials."""
        self._assert_unchanged(
            lambda: _make_aeroelastic_solver(wing_density=0.01),
            True,
            False,
            prescribed_wake=False,
            force_method="katz",
        )

    def test_moving_formation(self) -> None:
        """Two Airplanes, the second moving and pitching, with a free wake, which has no
        coupling trials."""
        self._assert_unchanged(
            solver_fixtures.make_unsteady_ring_vortex_lattice_method_moving_formation_solver,
            True,
            False,
            prescribed_wake=False,
        )


if __name__ == "__main__":
    unittest.main()
