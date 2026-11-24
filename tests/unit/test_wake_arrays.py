"""This module contains classes to test the array path's wake functions against the
object code they replace."""

import unittest
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import numpy as np

import pterasoftware as ps

# noinspection PyProtectedMember
from pterasoftware import _step_geometry, _wake_arrays
from tests.unit.fixtures import (
    airplane_movement_fixtures,
    geometry_fixtures,
    operating_point_fixtures,
    problem_fixtures,
    wing_movement_fixtures,
)

_Solver = ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver


def _make_truncated_solver() -> _Solver:
    """Returns a plain solver whose wake is truncated to two rows.

    :return: The solver, not yet run.
    """
    movement = ps.movements.movement.Movement(
        airplane_movements=[
            airplane_movement_fixtures.make_basic_airplane_movement_fixture()
        ],
        operating_point_movement=ps.movements.operating_point_movement.OperatingPointMovement(
            base_operating_point=operating_point_fixtures.make_basic_operating_point_fixture()
        ),
        num_cycles=1,
        max_wake_rows=2,
    )
    return _Solver(ps.problems.UnsteadyProblem(movement=movement))


def _make_two_wing_solver(max_wake_rows: int | None = None) -> _Solver:
    """Returns a plain solver of a static Airplane with two Wings with different numbers
    of spanwise Panels (a type 1 Wing and a type 4 Wing), so each Wing's block in the
    wake stacks starts at a different offset.

    :param max_wake_rows: The maximum number of wake rows each Wing keeps, or None to
        keep all of them. The default is None.
    :return: The solver, not yet run.
    """
    airplane = ps.geometry.airplane.Airplane(
        wings=[
            geometry_fixtures.make_type_1_wing_fixture(),
            geometry_fixtures.make_type_4_wing_fixture(),
        ],
        weight=0.0,
    )
    movement = ps.movements.movement.Movement(
        airplane_movements=[
            ps.movements.airplane_movement.AirplaneMovement(
                base_airplane=airplane,
                wing_movements=[
                    wing_movement_fixtures.make_static_wing_movement_fixture(wing)
                    for wing in airplane.wings
                ],
            )
        ],
        operating_point_movement=ps.movements.operating_point_movement.OperatingPointMovement(
            base_operating_point=operating_point_fixtures.make_basic_operating_point_fixture()
        ),
        num_steps=6,
        delta_time=0.02,
        max_wake_rows=max_wake_rows,
    )
    return _Solver(ps.problems.UnsteadyProblem(movement=movement))


# Each case: a name, a function that makes a solver, and whether the wake is prescribed.
_CASES: tuple[tuple[str, Callable[[], _Solver], bool], ...] = (
    (
        "basic prescribed",
        lambda: _Solver(problem_fixtures.make_basic_unsteady_problem_fixture()),
        True,
    ),
    (
        "basic free",
        lambda: _Solver(problem_fixtures.make_basic_unsteady_problem_fixture()),
        False,
    ),
    ("truncated prescribed", _make_truncated_solver, True),
    ("truncated free", _make_truncated_solver, False),
    (
        "pitching tip free",
        lambda: _Solver(problem_fixtures.make_pitching_tip_unsteady_problem_fixture()),
        False,
    ),
    ("two Wings free", _make_two_wing_solver, False),
    ("two Wings truncated free", lambda: _make_two_wing_solver(2), False),
)


def _record_object_path_wake(
    make_solver: Callable[[], _Solver], prescribed_wake: bool
) -> tuple[_Solver, list[dict[str, Any]], list[dict[str, Any]]]:
    """Runs a solver on the object path, recording the inputs and outputs of its wake
    grid and wake ring vortex updates at every time step.

    :param make_solver: A function that makes the solver.
    :param prescribed_wake: Whether the wake is prescribed.
    :return: The run solver, the wake grid update records, and the wake ring vortex
        update records.
    """
    grid_records: list[dict[str, Any]] = []
    vortex_records: list[dict[str, Any]] = []
    populate_grids = _Solver._populate_next_wake_grids_on_wings
    populate_vortices = _Solver._populate_next_airplanes_wake_vortices

    def recording_populate_grids(
        self: Any,
        next_airplanes: Any,
        stackVIndGridWrvp_GP1__E: Any,
        convectionVInf_GP1__E: np.ndarray,
        convectionOmegasRad_GP1__E: np.ndarray,
        bound_singularity_counts: np.ndarray,
        wake_singularity_counts: np.ndarray,
    ) -> None:
        step = self._current_step
        record: dict[str, Any] = {
            "step": step,
            "this_grids": [
                (
                    None
                    if wing.gridWrvp_GP1_CgP1 is None
                    else wing.gridWrvp_GP1_CgP1.copy()
                )
                for airplane in self.current_airplanes
                for wing in airplane.wings
            ],
            "next_Bl": self._listStackBlbrvp_GP1_CgP1[step + 1].copy(),
            "next_Br": self._listStackBrbrvp_GP1_CgP1[step + 1].copy(),
            "induced": (
                None
                if stackVIndGridWrvp_GP1__E is None
                else [
                    velocity.copy()
                    for airplane_velocities in stackVIndGridWrvp_GP1__E
                    for velocity in airplane_velocities
                ]
            ),
            "vInf": convectionVInf_GP1__E.copy(),
        }
        populate_grids(
            self,
            next_airplanes,
            stackVIndGridWrvp_GP1__E,
            convectionVInf_GP1__E,
            convectionOmegasRad_GP1__E,
            bound_singularity_counts,
            wake_singularity_counts,
        )
        record["next_grids"] = [
            wing.gridWrvp_GP1_CgP1.copy()
            for airplane in next_airplanes
            for wing in airplane.wings
        ]
        # At the first time step of a free wake, the new row convects with its induced
        # velocity, which reads only state this update doesn't change.
        if step == 0 and not prescribed_wake:
            record["row_induced"] = [
                self._calculate_induced_velocity(
                    grid[:1].reshape(-1, 3),
                    bound_singularity_counts=np.zeros(3, dtype=np.int64),
                    wake_singularity_counts=np.zeros(3, dtype=np.int64),
                )
                for grid in record["next_grids"]
            ]
        grid_records.append(record)

    def recording_populate_vortices(self: Any) -> None:
        step = self._current_step
        record: dict[str, Any] = {"step": step}
        if step < self.num_steps - 1:
            next_problem = self._get_steady_problem_at(step + 1)
            record.update(
                next_grids=[
                    wing.gridWrvp_GP1_CgP1.copy()
                    for airplane in next_problem.airplanes
                    for wing in airplane.wings
                ],
                bound_strengths=self._current_bound_vortex_strengths.copy(),
                this_strengths=self._list_wake_vortex_strengths[step].copy(),
                this_r_c0s=self._list_wake_r_c0s[step].copy(),
                standard_mean_chords=np.array(
                    [
                        wing.standard_mean_chord
                        for airplane in self.current_airplanes
                        for wing in airplane.wings
                    ],
                    dtype=float,
                ),
            )
        populate_vortices(self)
        if step < self.num_steps - 1:
            record.update(
                Fl=self.listStackFlwrvp_GP1_CgP1[step + 1].copy(),
                Fr=self.listStackFrwrvp_GP1_CgP1[step + 1].copy(),
                Bl=self.listStackBlwrvp_GP1_CgP1[step + 1].copy(),
                Br=self.listStackBrwrvp_GP1_CgP1[step + 1].copy(),
                strengths=self._list_wake_vortex_strengths[step + 1].copy(),
                r_c0s=self._list_wake_r_c0s[step + 1].copy(),
            )
            vortex_records.append(record)

    solver = make_solver()
    with (
        patch.object(
            _Solver, "_populate_next_wake_grids_on_wings", recording_populate_grids
        ),
        patch.object(
            _Solver,
            "_populate_next_airplanes_wake_vortices",
            recording_populate_vortices,
        ),
    ):
        solver.run(
            prescribed_wake=prescribed_wake,
            show_progress=False,
            calculate_streamlines=False,
        )
    return solver, grid_records, vortex_records


class TestWakeGridFunctions(unittest.TestCase):
    """This is a class with functions to test new_wake_rows, first_wake_grid, and
    next_wake_grid against the object path's wake grid update."""

    def test_match_the_object_path(self) -> None:
        """Test that, at every time step, the functions build exactly the new rows and
        the next wake grids the object path builds on the next time step's Wings, for
        prescribed, free, and truncated wakes."""
        for name, make_solver, prescribed_wake in _CASES:
            solver, grid_records, _ = _record_object_path_wake(
                make_solver, prescribed_wake
            )
            layout_plan = _step_geometry.build_layout_plan(
                solver._get_steady_problem_at(0).airplanes
            )
            self.assertEqual(len(grid_records), solver.num_steps - 1)
            for record in grid_records:
                step = record["step"]
                new_rows = _wake_arrays.new_wake_rows(
                    layout_plan, record["next_Bl"], record["next_Br"]
                )
                for wing_id, (new_row, expected) in enumerate(
                    zip(new_rows, record["next_grids"])
                ):
                    with self.subTest(case=name, step=step, wing=wing_id):
                        self.assertTrue(np.array_equal(new_row[0], expected[0]))
                        if step == 0:
                            if prescribed_wake:
                                velocity = record["vInf"]
                            else:
                                velocity = (
                                    record["row_induced"][wing_id] + record["vInf"]
                                ).reshape(new_row.shape)
                            grid = _wake_arrays.first_wake_grid(
                                new_row, velocity, solver.delta_time
                            )
                        else:
                            this_grid = record["this_grids"][wing_id]
                            if prescribed_wake:
                                velocity = record["vInf"]
                            else:
                                velocity = (
                                    record["induced"][wing_id] + record["vInf"]
                                ).reshape(this_grid.shape)
                            grid = _wake_arrays.next_wake_grid(
                                this_grid,
                                velocity,
                                new_row,
                                solver.delta_time,
                                solver._max_wake_rows,
                            )
                        self.assertEqual(grid.shape, expected.shape)
                        self.assertTrue(np.array_equal(grid, expected))

    def test_truncation_drops_the_oldest_row(self) -> None:
        """Test that next_wake_grid keeps max_wake_rows + 1 rows of points, dropping the
        oldest, and keeps every row without truncation."""
        grid = np.arange(4 * 3 * 3, dtype=float).reshape(4, 3, 3)
        new_row = -np.ones((1, 3, 3), dtype=float)
        truncated = _wake_arrays.next_wake_grid(grid, 0.0 * grid, new_row, 0.1, 2)
        self.assertEqual(truncated.shape, (3, 3, 3))
        self.assertTrue(np.array_equal(truncated[0], new_row[0]))
        self.assertTrue(np.array_equal(truncated[1:], grid[:2]))
        untruncated = _wake_arrays.next_wake_grid(grid, 0.0 * grid, new_row, 0.1, None)
        self.assertEqual(untruncated.shape, (5, 3, 3))


class TestWakeRingVortices(unittest.TestCase):
    """This is a class with functions to test wake_ring_vortices against the object
    path's wake ring vortex update."""

    def test_matches_the_object_path(self) -> None:
        """Test that, at every time step, wake_ring_vortices gives exactly the next time
        step's wake ring vortex corners, strengths, and initial core radii the object
        path writes, for prescribed, free, and truncated wakes, and for a Wing whose
        standard mean chord changes every time step."""
        for name, make_solver, prescribed_wake in _CASES:
            solver, _, vortex_records = _record_object_path_wake(
                make_solver, prescribed_wake
            )
            layout_plan = _step_geometry.build_layout_plan(
                solver._get_steady_problem_at(0).airplanes
            )
            self.assertEqual(len(vortex_records), solver.num_steps - 1)
            for record in vortex_records:
                with self.subTest(case=name, step=record["step"]):
                    wake = _wake_arrays.wake_ring_vortices(
                        layout_plan,
                        record["next_grids"],
                        record["step"],
                        solver._max_wake_rows,
                        record["bound_strengths"],
                        record["this_strengths"],
                        record["this_r_c0s"],
                        record["standard_mean_chords"],
                    )
                    for field, key in (
                        ("stackFlwrvp_GP1_CgP1", "Fl"),
                        ("stackFrwrvp_GP1_CgP1", "Fr"),
                        ("stackBlwrvp_GP1_CgP1", "Bl"),
                        ("stackBrwrvp_GP1_CgP1", "Br"),
                        ("strengths", "strengths"),
                        ("r_c0s", "r_c0s"),
                    ):
                        self.assertTrue(
                            np.array_equal(getattr(wake, field), record[key]), field
                        )


if __name__ == "__main__":
    unittest.main()
