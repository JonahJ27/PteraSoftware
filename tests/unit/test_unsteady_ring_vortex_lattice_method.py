"""This module contains classes to test UnsteadyRingVortexLatticeMethodSolvers."""

import unittest
from typing import Any, cast
from unittest.mock import patch

import numpy as np

import pterasoftware as ps

# noinspection PyProtectedMember
from pterasoftware import _mesh_arrays
from tests.unit.fixtures import (
    airplane_movement_fixtures,
    flap_drive_fixtures,
    operating_point_fixtures,
    problem_fixtures,
    solver_fixtures,
)


class TestUnsteadyRingVortexLatticeMethodSolver(unittest.TestCase):
    """This is a class with functions to test UnsteadyRingVortexLatticeMethodSolvers."""

    def test_initialization_accepts_unsteady_problem(self) -> None:
        """Test that initialization accepts an UnsteadyProblem."""
        solver = solver_fixtures.make_unsteady_ring_solver_fixture()
        self.assertIsInstance(
            solver,
            ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver,
        )

    def test_initialization_rejects_coupled_unsteady_problem(self) -> None:
        """Test that initialization on the base solver raises TypeError for a
        _CoupledUnsteadyProblem, while still allowing the coupled subclass to pass one
        through super()."""
        coupled_problem = problem_fixtures.make_basic_coupled_unsteady_problem_fixture()
        with self.assertRaises(TypeError):
            ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(
                coupled_problem
            )

    def test_initialization_rejects_non_problem_types(self) -> None:
        """Test that initialization raises TypeError for non-problem inputs."""
        invalid_inputs: list[Any] = [
            None,
            "not_a_problem",
            123,
            [1, 2, 3],
            {"key": "value"},
        ]
        for invalid in invalid_inputs:
            with self.subTest(invalid=invalid):
                with self.assertRaises(TypeError):
                    ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(
                        invalid
                    )

    def test_initialization_rejects_non_zero_body_rates(self) -> None:
        """Test that initialization raises when any per-step operating point carries a
        non-zero body angular velocity, which the base solver does not model."""
        rotating_problem = (
            problem_fixtures.make_with_body_rates_unsteady_problem_fixture()
        )
        with self.assertRaises(ValueError):
            ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(
                rotating_problem
            )

    def test_steady_problems_is_read_only(self) -> None:
        """Test that the steady_problems property cannot be reassigned, since it is a
        read-only view of the UnsteadyProblem's SteadyProblems.

        The property is defined on this base solver and inherited by every coupled
        subclass.
        """
        solver = solver_fixtures.make_unsteady_ring_solver_fixture()
        with self.assertRaises(AttributeError):
            setattr(solver, "steady_problems", ())

    def test_prescribed_wake_raises_before_run(self) -> None:
        """Test that reading prescribed_wake before run raises a RuntimeError."""
        solver = solver_fixtures.make_unsteady_ring_solver_fixture()
        with self.assertRaises(RuntimeError):
            _ = solver.prescribed_wake

    def test_prescribed_wake_is_read_only(self) -> None:
        """Test that the prescribed_wake property cannot be reassigned."""
        solver = solver_fixtures.make_unsteady_ring_solver_fixture()
        with self.assertRaises(AttributeError):
            setattr(solver, "prescribed_wake", True)

    def test_force_method_raises_before_run(self) -> None:
        """Test that reading force_method before run raises a RuntimeError."""
        solver = solver_fixtures.make_unsteady_ring_solver_fixture()
        with self.assertRaises(RuntimeError):
            _ = solver.force_method

    def test_force_method_is_read_only(self) -> None:
        """Test that the force_method property cannot be reassigned."""
        solver = solver_fixtures.make_unsteady_ring_solver_fixture()
        with self.assertRaises(AttributeError):
            setattr(solver, "force_method", "joukowski")

    def test_ran_is_false_before_run(self) -> None:
        """Test that ran is False before run has been called."""
        solver = solver_fixtures.make_unsteady_ring_solver_fixture()
        self.assertFalse(solver.ran)

    def test_ran_is_read_only(self) -> None:
        """Test that the ran property cannot be reassigned."""
        solver = solver_fixtures.make_unsteady_ring_solver_fixture()
        with self.assertRaises(AttributeError):
            setattr(solver, "ran", True)

    def test_run_rejects_invalid_force_method_strings(self) -> None:
        """Test that run raises ValueError for invalid force_method strings."""
        solver = solver_fixtures.make_unsteady_ring_solver_fixture()
        invalid_values = ["invalid", "JOUKOWSKI", "Katz", "both", ""]
        for invalid in invalid_values:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    solver.run(force_method=invalid, show_progress=False)

    def test_run_rejects_non_string_force_methods(self) -> None:
        """Test that run raises TypeError for non-string force_method values."""
        solver = solver_fixtures.make_unsteady_ring_solver_fixture()
        invalid_types: list[Any] = [
            123,
            1.0,
            None,
            True,
            ["joukowski"],
            {"method": "joukowski"},
        ]
        for invalid in invalid_types:
            with self.subTest(invalid=invalid):
                with self.assertRaises(TypeError):
                    solver.run(force_method=invalid, show_progress=False)


class TestUnsteadyRingVortexLatticeMethodSolverUseArrays(unittest.TestCase):
    """This is a class with functions to test the use_arrays run argument."""

    def test_use_arrays_raises_before_run(self) -> None:
        """Test that reading use_arrays before run raises a RuntimeError."""
        solver = solver_fixtures.make_unsteady_ring_solver_fixture()
        with self.assertRaises(RuntimeError):
            _ = solver.use_arrays

    def test_use_arrays_is_read_only(self) -> None:
        """Test that the use_arrays property cannot be reassigned."""
        solver = solver_fixtures.make_unsteady_ring_solver_fixture()
        with self.assertRaises(AttributeError):
            setattr(solver, "use_arrays", True)

    def test_use_arrays_defaults_to_false(self) -> None:
        """Test that a run without use_arrays uses the geometry objects."""
        solver = solver_fixtures.make_unsteady_ring_solver_fixture()
        solver.run(prescribed_wake=True, show_progress=False)
        self.assertFalse(solver.use_arrays)

    def test_run_rejects_non_bool_use_arrays(self) -> None:
        """Test that run raises a TypeError for a use_arrays that isn't bool like."""
        invalid_values: list[Any] = ["True", 1, 0.0, None, [True]]
        for invalid in invalid_values:
            with self.subTest(invalid=invalid):
                solver = solver_fixtures.make_unsteady_ring_solver_fixture()
                with self.assertRaises(TypeError):
                    solver.run(use_arrays=invalid, show_progress=False)

    def test_array_path_rejects_unsupported_features(self) -> None:
        """Test that use_arrays=True raises a ValueError naming the feature for ground
        effect, before solving, and accepts the "katz" force method and multiple
        Airplanes (supported since M2 of the arrays rewrite; the multiple Airplane
        fixture is numerically singular, so only its support check runs here, and the
        formation integration module runs formations)."""
        solver_fixtures.make_unsteady_ring_solver_fixture().run(
            use_arrays=True, force_method="katz", show_progress=False
        )

        multi_airplane_solver = ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(
            problem_fixtures.make_multi_airplane_unsteady_problem_fixture()
        )
        self.assertGreater(multi_airplane_solver.num_airplanes, 1)
        multi_airplane_solver._check_array_path_support()

        ground_movement = ps.movements.movement.Movement(
            airplane_movements=[
                airplane_movement_fixtures.make_basic_airplane_movement_fixture()
            ],
            operating_point_movement=ps.movements.operating_point_movement.OperatingPointMovement(
                base_operating_point=operating_point_fixtures.make_with_ground_surface_operating_point_fixture()
            ),
            num_cycles=1,
        )
        ground_solver = ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(
            ps.problems.UnsteadyProblem(movement=ground_movement)
        )
        with self.assertRaisesRegex(ValueError, "ground effect"):
            ground_solver.run(use_arrays=True, show_progress=False)

    def test_array_path_run_matches_the_object_path(self) -> None:
        """Test that a run with use_arrays=True records the choice, clears its per run
        plans and step geometries, and gives the object path's loads at every time step
        to round off (the array path's vectorized norms round about one unit in the last
        place differently)."""
        object_solver = solver_fixtures.make_unsteady_ring_solver_fixture()
        object_solver.run(prescribed_wake=True, show_progress=False)
        array_solver = solver_fixtures.make_unsteady_ring_solver_fixture()
        array_solver.run(prescribed_wake=True, show_progress=False, use_arrays=True)

        self.assertTrue(array_solver.use_arrays)
        self.assertIsNone(array_solver._mesh_plans)
        self.assertIsNone(array_solver._layout_plan)
        self.assertIsNone(array_solver._step_geometries)
        self.assertEqual(
            len(array_solver.steady_problems), len(object_solver.steady_problems)
        )
        for step, (array_problem, object_problem) in enumerate(
            zip(array_solver.steady_problems, object_solver.steady_problems)
        ):
            for array_airplane, object_airplane in zip(
                array_problem.airplanes, object_problem.airplanes
            ):
                for name in ("forces_W", "moments_W_CgP1"):
                    with self.subTest(step=step, name=name):
                        array_value = getattr(array_airplane, name)
                        object_value = getattr(object_airplane, name)
                        if object_value is None:
                            self.assertIsNone(array_value)
                        else:
                            np.testing.assert_allclose(
                                array_value, object_value, rtol=1.0e-12, atol=1.0e-12
                            )

    def test_array_path_loads_match_the_object_path(self) -> None:
        """Test that, at every time step and for both force methods, the loads the array
        path keeps match the loads the object path writes onto the Airplanes and Panels
        at that time step, to round off, that after the run every time step's Airplanes
        and Panels carry the same loads on both paths, and that the kept loads are
        cleared when the run ends."""
        makers = (
            solver_fixtures.make_unsteady_ring_solver_fixture,
            lambda: ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(
                problem_fixtures.make_pitching_tip_unsteady_problem_fixture()
            ),
            solver_fixtures.make_aeroelastic_unsteady_ring_solver_fixture,
        )
        calculate_loads = (
            ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver._calculate_loads
        )
        for maker_index, make in enumerate(makers):
            for force_method in ("joukowski", "katz"):
                records: dict[bool, list[dict[str, Any]]] = {}
                solvers: dict[bool, Any] = {}
                for use_arrays in (False, True):
                    solver = make()
                    steps: list[dict[str, Any]] = []

                    def recording_calculate_loads(self: Any) -> None:
                        calculate_loads(self)
                        # The array path keeps the loads and writes no objects during
                        # the run, and the object path writes them onto the objects.
                        if self._use_arrays:
                            loads = self._current_step_loads
                            record: dict[str, Any] = {
                                "forces_W": np.copy(loads.stackAirplaneForces_W[0]),
                                "moments_W_CgP1": np.copy(
                                    loads.stackAirplaneMoments_W_CgP1[0]
                                ),
                                "momentCoefficients_W_Cg": np.copy(
                                    loads.stackAirplaneMomentCoefficients_W_Cg[0]
                                ),
                                "panel_forces_GP1": np.copy(loads.stackPanelForces_GP1),
                                "panel_moments_W_CgP1": np.copy(
                                    loads.stackPanelMoments_W_CgP1
                                ),
                            }
                        else:
                            airplane = self.current_airplanes[0]
                            record = {
                                "forces_W": np.copy(airplane.forces_W),
                                "moments_W_CgP1": np.copy(airplane.moments_W_CgP1),
                                "momentCoefficients_W_Cg": np.copy(
                                    airplane.momentCoefficients_W_Cg
                                ),
                                "panel_forces_GP1": np.array(
                                    [panel.forces_GP1 for panel in self.panels]
                                ),
                                "panel_moments_W_CgP1": np.array(
                                    [panel.moments_W_CgP1 for panel in self.panels]
                                ),
                            }
                        steps.append(record)

                    with patch.object(
                        type(solver), "_calculate_loads", recording_calculate_loads
                    ):
                        solver.run(
                            show_progress=False,
                            use_arrays=use_arrays,
                            force_method=force_method,
                        )
                    records[use_arrays] = steps
                    solvers[use_arrays] = solver
                    self.assertIsNone(solver._current_step_loads)

                # After the run, every time step's Airplanes and Panels carry the same
                # loads on both paths.
                for step in range(solvers[False].num_steps):
                    object_problem = solvers[False].steady_problems[step]
                    array_problem = solvers[True].steady_problems[step]
                    object_airplane = object_problem.airplanes[0]
                    array_airplane = array_problem.airplanes[0]
                    for name in (
                        "forces_W",
                        "moments_W_CgP1",
                        "forces_G",
                        "moments_G_Cg",
                        "moments_W_Cg",
                        "forceCoefficients_W",
                        "forceCoefficients_G",
                        "momentCoefficients_W_CgP1",
                        "momentCoefficients_G_Cg",
                        "momentCoefficients_W_Cg",
                    ):
                        object_value = getattr(object_airplane, name)
                        array_value = getattr(array_airplane, name)
                        with self.subTest(
                            maker=maker_index,
                            force_method=force_method,
                            step=step,
                            name=name,
                            after_run=True,
                        ):
                            if object_value is None:
                                self.assertIsNone(array_value)
                                continue
                            np.testing.assert_allclose(
                                array_value,
                                object_value,
                                rtol=1.0e-10,
                                atol=1.0e-10 * float(np.max(np.abs(object_value))),
                            )
                    object_panels = [
                        panel
                        for wing in object_airplane.wings
                        for panel in np.ravel(wing.panels)
                    ]
                    array_panels = [
                        panel
                        for wing in array_airplane.wings
                        for panel in np.ravel(wing.panels)
                    ]
                    for name in (
                        "forces_GP1",
                        "moments_GP1_CgP1",
                        "forces_W",
                        "moments_W_CgP1",
                    ):
                        object_values = [getattr(pn, name) for pn in object_panels]
                        array_values = [getattr(pn, name) for pn in array_panels]
                        with self.subTest(
                            maker=maker_index,
                            force_method=force_method,
                            step=step,
                            name="panel " + name,
                            after_run=True,
                        ):
                            if object_values[0] is None:
                                self.assertTrue(all(v is None for v in array_values))
                                continue
                            object_stack = np.array(object_values)
                            np.testing.assert_allclose(
                                np.array(array_values),
                                object_stack,
                                rtol=1.0e-10,
                                atol=1.0e-10 * float(np.max(np.abs(object_stack))),
                            )

                self.assertEqual(len(records[True]), len(records[False]))
                for step, (array_record, object_record) in enumerate(
                    zip(records[True], records[False])
                ):
                    for name, object_value in object_record.items():
                        with self.subTest(
                            maker=maker_index,
                            force_method=force_method,
                            step=step,
                            name=name,
                        ):
                            np.testing.assert_allclose(
                                array_record[name],
                                object_value,
                                rtol=1.0e-10,
                                atol=1.0e-10 * float(np.max(np.abs(object_value))),
                            )

    def test_array_path_wake_matches_the_object_path(self) -> None:
        """Test that, at every time step, for prescribed and free wakes, the array
        path's next wake ring vortices and the wake point grids it keeps match the
        object path's (which it sets on the next time step's Wings) to round off, that
        it keeps at most two time steps' grids and clears them when the run ends, and
        that after the run every time step's Wings carry the same wake point grids on
        both paths."""
        populate = (
            ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver._populate_next_airplanes_wake
        )
        for maker_index, make in enumerate(
            (
                solver_fixtures.make_unsteady_ring_solver_fixture,
                lambda: ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(
                    problem_fixtures.make_pitching_tip_unsteady_problem_fixture()
                ),
                solver_fixtures.make_aeroelastic_unsteady_ring_solver_fixture,
            )
        ):
            for prescribed_wake in (True, False):
                records: dict[bool, list[dict[str, Any]]] = {}
                solvers: dict[bool, Any] = {}
                for use_arrays in (False, True):
                    solver = make()
                    steps: list[dict[str, Any]] = []

                    def recording_populate(self: Any) -> None:
                        populate(self)
                        next_step = self._current_step + 1
                        if next_step >= self.num_steps:
                            return
                        if self._use_arrays:
                            next_grids = self._wake_grids[next_step]
                        else:
                            next_problem = self._get_steady_problem_at(next_step)
                            next_grids = [
                                wing.gridWrvp_GP1_CgP1
                                for airplane in next_problem.airplanes
                                for wing in airplane.wings
                            ]
                        steps.append(
                            {
                                "Fl": self.listStackFlwrvp_GP1_CgP1[next_step].copy(),
                                "Br": self.listStackBrwrvp_GP1_CgP1[next_step].copy(),
                                "strengths": self._list_wake_vortex_strengths[
                                    next_step
                                ].copy(),
                                "r_c0s": self._list_wake_r_c0s[next_step].copy(),
                                "grids": np.concatenate(
                                    [grid.ravel() for grid in next_grids]
                                ),
                                "num_kept_grids": (
                                    len(self._wake_grids)
                                    if self._wake_grids is not None
                                    else 0
                                ),
                            }
                        )

                    with patch.object(
                        type(solver),
                        "_populate_next_airplanes_wake",
                        recording_populate,
                    ):
                        solver.run(
                            show_progress=False,
                            use_arrays=use_arrays,
                            prescribed_wake=prescribed_wake,
                        )
                    records[use_arrays] = steps
                    solvers[use_arrays] = solver
                    self.assertIsNone(solver._wake_grids)

                # After the run, every time step's Wings carry the same wake point grids
                # on both paths (the first time step's carry none).
                for step in range(1, solvers[False].num_steps):
                    object_wings = [
                        wing
                        for airplane in solvers[False].steady_problems[step].airplanes
                        for wing in airplane.wings
                    ]
                    array_wings = [
                        wing
                        for airplane in solvers[True].steady_problems[step].airplanes
                        for wing in airplane.wings
                    ]
                    for wing_id, (object_wing, array_wing) in enumerate(
                        zip(object_wings, array_wings)
                    ):
                        with self.subTest(
                            maker=maker_index,
                            prescribed_wake=prescribed_wake,
                            step=step,
                            wing_id=wing_id,
                            after_run=True,
                        ):
                            np.testing.assert_allclose(
                                array_wing.gridWrvp_GP1_CgP1,
                                object_wing.gridWrvp_GP1_CgP1,
                                rtol=1.0e-10,
                                atol=1.0e-10
                                * max(
                                    float(
                                        np.max(np.abs(object_wing.gridWrvp_GP1_CgP1))
                                    ),
                                    1.0,
                                ),
                            )

                self.assertEqual(len(records[True]), len(records[False]))
                for step, (array_record, object_record) in enumerate(
                    zip(records[True], records[False])
                ):
                    self.assertLessEqual(array_record["num_kept_grids"], 2)
                    for name in ("Fl", "Br", "strengths", "r_c0s", "grids"):
                        with self.subTest(
                            maker=maker_index,
                            prescribed_wake=prescribed_wake,
                            step=step,
                            name=name,
                        ):
                            object_value = object_record[name]
                            np.testing.assert_allclose(
                                array_record[name],
                                object_value,
                                rtol=1.0e-10,
                                atol=1.0e-10
                                * max(
                                    float(np.max(np.abs(object_value), initial=0.0)),
                                    1.0,
                                ),
                            )

    def test_array_path_collapse_matches_the_object_path(self) -> None:
        """Test that, at every time step, the array path writes the object path's
        collapsed geometry into the solver's arrays (to round off), for plain solvers
        (whose meshes come from the movement's WingStepParameters) and an aeroelastic
        solver (whose meshes come from its recorded step parameters), that the array
        path keeps at most three time steps' geometry, and that after the run both paths
        leave the solver's Panels as the last time step's."""
        makers = (
            solver_fixtures.make_unsteady_ring_solver_fixture,
            lambda: ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(
                problem_fixtures.make_pitching_tip_unsteady_problem_fixture()
            ),
            solver_fixtures.make_aeroelastic_unsteady_ring_solver_fixture,
        )
        names = (
            "stackUnitNormals_GP1",
            "panel_areas",
            "stackCpp_GP1_CgP1",
            "_stackLastCpp_GP1_CgP1",
            "stackFrbrvp_GP1_CgP1",
            "stackFlbrvp_GP1_CgP1",
            "stackBlbrvp_GP1_CgP1",
            "stackBrbrvp_GP1_CgP1",
            "_lastStackFrbrvp_GP1_CgP1",
            "_lastStackFlbrvp_GP1_CgP1",
            "_lastStackBlbrvp_GP1_CgP1",
            "_lastStackBrbrvp_GP1_CgP1",
            "stackCblvpr_GP1_CgP1",
            "stackCblvpf_GP1_CgP1",
            "stackCblvpl_GP1_CgP1",
            "stackCblvpb_GP1_CgP1",
            "_lastStackCblvpr_GP1_CgP1",
            "_lastStackCblvpb_GP1_CgP1",
            "stackRbrv_GP1",
            "stackFbrv_GP1",
            "stackLbrv_GP1",
            "stackBbrv_GP1",
            "panel_is_trailing_edge",
            "panel_is_leading_edge",
            "panel_is_left_edge",
            "panel_is_right_edge",
            "stackSeedPoints_GP1_CgP1",
            "_current_bound_r_c0s",
            "_current_wake_vortex_ages",
        )
        prepare = (
            ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver._prepare_step_geometry
        )
        for maker_index, make in enumerate(makers):
            records: dict[bool, list[dict[str, Any]]] = {}
            for use_arrays in (False, True):
                solver = make()
                steps: list[dict[str, Any]] = []

                def recording_prepare(self: Any) -> None:
                    prepare(self)
                    record: dict[str, Any] = {
                        name: np.copy(getattr(self, name)) for name in names
                    }
                    record["num_step_geometries"] = (
                        len(self._step_geometries)
                        if self._step_geometries is not None
                        else 0
                    )
                    steps.append(record)

                with patch.object(
                    type(solver), "_prepare_step_geometry", recording_prepare
                ):
                    solver.run(show_progress=False, use_arrays=use_arrays)
                records[use_arrays] = steps
                last_panels = [
                    panel
                    for airplane in solver.steady_problems[-1].airplanes
                    for wing in airplane.wings
                    for panel in np.ravel(cast(np.ndarray, wing.panels))
                ]
                self.assertEqual(len(solver.panels), len(last_panels))
                for panel, last_panel in zip(solver.panels, last_panels):
                    self.assertIs(panel, last_panel)

            self.assertEqual(len(records[True]), len(records[False]))
            for step, (array_record, object_record) in enumerate(
                zip(records[True], records[False])
            ):
                self.assertLessEqual(array_record["num_step_geometries"], 3)
                for name in names:
                    with self.subTest(maker=maker_index, step=step, name=name):
                        np.testing.assert_allclose(
                            array_record[name],
                            object_record[name],
                            rtol=0.0,
                            atol=1.0e-12,
                        )


class TestUnsteadyRingVortexLatticeMethodSolverArrayPathBuildsNoObjects(
    unittest.TestCase
):
    """This is a class with functions to test that the array path builds and copies no
    geometry objects during the run."""

    def test_no_objects_are_built_during_the_run(self) -> None:
        """Test that, for a plain, an aeroelastic, a free flight, and a free flight flap
        driven solver, an array path run builds no Airplane, SteadyProblem, Wing,
        WingCrossSection, or Panel, and deep copies none, before it writes its results
        when the run ends, and that an object path run does (so the count works)."""
        classes = (
            ps.geometry.airplane.Airplane,
            ps.geometry.wing.Wing,
            ps.geometry.wing_cross_section.WingCrossSection,
            ps.problems.SteadyProblem,
            ps._panel.Panel,
        )
        makers = {
            "plain": solver_fixtures.make_unsteady_ring_solver_fixture,
            "aeroelastic": solver_fixtures.make_aeroelastic_unsteady_ring_solver_fixture,
            "free flight": lambda: ps.free_flight_unsteady_ring_vortex_lattice_method.FreeFlightUnsteadyRingVortexLatticeMethodSolver(
                problem_fixtures.make_basic_free_flight_unsteady_problem_fixture()
            ),
            "free flight flap driven": lambda: ps.free_flight_aeroelastic_unsteady_ring_vortex_lattice_method.FreeFlightAeroelasticUnsteadyRingVortexLatticeMethodSolver(
                flap_drive_fixtures.make_free_flight_flap_driven_aeroelastic_problem_fixture()
            ),
        }
        solver_class = (
            ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver
        )
        write_results = solver_class._write_array_results_to_objects
        for name, make in makers.items():
            for use_arrays in (False, True):
                with self.subTest(solver=name, use_arrays=use_arrays):
                    solver = make()
                    counts = {"built": 0}
                    in_loop = {"value": True}
                    patches = []
                    for cls in classes:
                        for method in ("__init__", "__deepcopy__"):
                            if method not in vars(cls):
                                continue
                            original = getattr(cls, method)

                            def counting(
                                self_: Any,
                                *args: Any,
                                _original: Any = original,
                                **kwargs: Any,
                            ) -> Any:
                                if in_loop["value"]:
                                    counts["built"] += 1
                                return _original(self_, *args, **kwargs)

                            patches.append(patch.object(cls, method, counting))

                    def stop_counting(self_: Any) -> None:
                        in_loop["value"] = False
                        write_results(self_)

                    stop_patch: Any = patch.object(
                        solver_class,
                        "_write_array_results_to_objects",
                        stop_counting,
                    )
                    patches.append(stop_patch)
                    for active in patches:
                        active.start()
                    try:
                        solver.run(show_progress=False, use_arrays=use_arrays)
                    finally:
                        for active in patches:
                            active.stop()
                    if use_arrays:
                        self.assertEqual(counts["built"], 0)
                    elif name != "plain":
                        # A plain problem's objects are all built by its movement,
                        # before the run, on both paths.
                        self.assertGreater(counts["built"], 0)


class TestUnsteadyRingVortexLatticeMethodSolverWingMeshesFromParameters(
    unittest.TestCase
):
    """This is a class with functions to test that the array path checks the step
    parameters it meshes, as building the objects would."""

    def test_invalid_step_parameters_raise_the_object_paths_error(self) -> None:
        """Test that _wing_meshes_from_parameters, given step parameters with a Wing
        angle out of range or a CG that isn't finite, raises the error building the
        Airplane from them raises, and meshes the same step parameters when they are
        valid.

        A run can't reach this check with invalid values yet (the movements check their
        motion when they are created, and the coupled problems build each Airplane
        before it is meshed), so it is tested directly.
        """
        solver = solver_fixtures.make_unsteady_ring_solver_fixture()
        # The run builds the mesh plans from the first time step's Wings and clears them
        # when it ends, so they are built here the same way.
        solver._mesh_plans = tuple(
            _mesh_arrays.build_mesh_plan(wing)
            for airplane in solver.steady_problems[0].airplanes
            for wing in airplane.wings
        )
        airplane_movement: Any = solver.unsteady_problem.movement.airplane_movements[0]
        parameters = airplane_movement._generate_airplane_parameters_at_time_step(
            0, solver.delta_time
        )
        wing_meshes, _ = solver._wing_meshes_from_parameters([parameters])
        self.assertEqual(len(wing_meshes), len(parameters.wings))

        # A Wing angle out of range (which mesh_wing_arrays also checks), and a CG that
        # isn't finite (which only the Airplane's check catches).
        bad_wing = parameters.wings[0]._replace(
            angles_Gs_to_Wn_ixyz=np.array([0.0, 100.0, 0.0])
        )
        for name, bad_parameters in (
            (
                "wing angle",
                parameters._replace(wings=(bad_wing, *parameters.wings[1:])),
            ),
            ("CG", parameters._replace(Cg_GP1_CgP1=np.array([np.nan, 0.0, 0.0]))),
        ):
            with self.subTest(name=name):
                with self.assertRaises(ValueError) as expected:
                    airplane_movement._airplane_from_parameters(bad_parameters)
                with self.assertRaises(ValueError) as context:
                    solver._wing_meshes_from_parameters([bad_parameters])
                self.assertEqual(str(context.exception), str(expected.exception))


class TestUnsteadyRingVortexLatticeMethodSolverWakeCoreRadii(unittest.TestCase):
    """Tests that each wake ring vortex keeps the initial core radius it was shed with,
    rather than taking the current step's Wing's value."""

    def setUp(self) -> None:
        """Solve an unsteady problem whose Wing's standard mean chord changes every time
        step."""
        self.problem = problem_fixtures.make_pitching_tip_unsteady_problem_fixture()
        self.solver = ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(
            self.problem
        )
        self.solver.run(show_progress=False)

        # The per step initial core radii that _collapse_geometry assigns to each Wing's
        # trailing edge back legs. The fixture has one Wing per step.
        self.step_r_c0s = []
        for steady_problem in self.problem.steady_problems:
            wing = steady_problem.airplanes[0].wings[0]
            assert wing.standard_mean_chord is not None
            self.step_r_c0s.append(0.03 * wing.standard_mean_chord)

        wing = self.problem.steady_problems[0].airplanes[0].wings[0]
        assert wing.num_spanwise_panels is not None
        self.num_spanwise_panels = wing.num_spanwise_panels

    def test_fixture_chord_changes_every_step(self) -> None:
        """Test that the fixture gives every step a distinct initial core radius, so the
        other tests in this class can tell a frozen value from a recomputed one."""
        self.assertEqual(len(set(self.step_r_c0s)), self.solver.num_steps)

    def test_wake_rows_keep_shedding_steps_core_radii(self) -> None:
        """Test that row c of step s's wake (0-based, newest first) holds the initial
        core radius of the Wing at step s - 1 - c, which shed it."""
        for step in range(1, self.solver.num_steps):
            step_r_c0s = self.solver._list_wake_r_c0s[step]
            self.assertEqual(step_r_c0s.shape, (step * self.num_spanwise_panels,))
            for row in range(step):
                start = row * self.num_spanwise_panels
                end = start + self.num_spanwise_panels
                with self.subTest(step=step, row=row):
                    np.testing.assert_allclose(
                        step_r_c0s[start:end],
                        self.step_r_c0s[step - 1 - row],
                    )

    def test_current_wake_core_radii_match_final_step_list(self) -> None:
        """Test that the current wake core radius array after run holds the final step's
        per step list values on all four legs of every wake ring vortex."""
        final_step = self.solver.num_steps - 1
        expected = np.repeat(
            self.solver._list_wake_r_c0s[final_step][:, np.newaxis], 4, axis=1
        )
        np.testing.assert_array_equal(self.solver._current_wake_r_c0s, expected)


class TestUnsteadyRingVortexLatticeMethodSolverJoukowskiLegCenters(unittest.TestCase):
    """Tests that the Joukowski loads evaluate each distinct leg center point once and
    still give every leg exactly the velocity that evaluating its own points would."""

    solver: ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver

    @classmethod
    def setUpClass(cls) -> None:
        """Solve an unsteady problem with moving geometry, so the leg centers' movement
        velocities are not zero, and leave the solver at its final step."""
        cls.solver = ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(
            problem_fixtures.make_pitching_tip_unsteady_problem_fixture()
        )
        cls.solver.run(show_progress=False)

    def test_evaluates_each_distinct_leg_center_once(self) -> None:
        """Test that the loads call calculate_solution_velocity once, on distinct points
        only, and on fewer points than the four leg center stacks hold."""
        solver_class = type(self.solver)
        with patch.object(
            solver_class,
            "calculate_solution_velocity",
            autospec=True,
            side_effect=solver_class.calculate_solution_velocity,
        ) as mock_velocity:
            self.solver._calculate_loads_joukowski()

        mock_velocity.assert_called_once()
        points = mock_velocity.call_args.kwargs["stackP_GP1_CgP1"]
        self.assertLess(points.shape[0], 4 * self.solver.num_panels)
        self.assertEqual(np.unique(points, axis=0).shape[0], points.shape[0])

    def test_leg_velocities_match_evaluating_each_leg_separately(self) -> None:
        """Test that each leg's velocity, as passed to _apply_body_rate, is bit
        identical to evaluating that leg's own center points in a separate call, as the
        loads did before the distinct points were shared."""
        solver = self.solver
        last_stacks = {
            id(solver.stackCblvpr_GP1_CgP1): solver._lastStackCblvpr_GP1_CgP1,
            id(solver.stackCblvpf_GP1_CgP1): solver._lastStackCblvpf_GP1_CgP1,
            id(solver.stackCblvpl_GP1_CgP1): solver._lastStackCblvpl_GP1_CgP1,
            id(solver.stackCblvpb_GP1_CgP1): solver._lastStackCblvpb_GP1_CgP1,
        }
        solver_class = type(solver)
        with patch.object(
            solver_class,
            "_apply_body_rate",
            autospec=True,
            side_effect=solver_class._apply_body_rate,
        ) as mock_body_rate:
            solver._calculate_loads_joukowski()

        self.assertEqual(mock_body_rate.call_count, 4)
        # The calls come in the order right, front, left, back.
        for leg, call in zip(
            ("right", "front", "left", "back"), mock_body_rate.call_args_list
        ):
            points, velocities = call.args[1], call.args[2]
            with self.subTest(leg=leg):
                expected = solver.calculate_solution_velocity(
                    stackP_GP1_CgP1=points
                ) + solver._calculate_current_movement_velocities(
                    points, last_stacks[id(points)]
                )
                self.assertTrue(np.array_equal(velocities, expected))


class TestUnsteadyRingVortexLatticeMethodSolverCollapsedCornerArrays(unittest.TestCase):
    """Tests that _collapse_geometry's vectorized corner, leg center, and leg vector
    arrays equal what the per Panel arithmetic in
    update_ring_vortex_solvers_panel_attributes writes."""

    solver: ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver

    @classmethod
    def setUpClass(cls) -> None:
        """Solve an unsteady problem with moving geometry and leave the solver at its
        final step."""
        cls.solver = ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(
            problem_fixtures.make_pitching_tip_unsteady_problem_fixture()
        )
        cls.solver.run(show_progress=False)

    def test_corner_arrays_match_per_panel_arithmetic(self) -> None:
        """Test that every collapsed corner derived array equals, bit for bit, the
        arrays that update_ring_vortex_solvers_panel_attributes writes one Panel at a
        time with write_corner_arrays left at its default."""
        solver = self.solver
        step = solver._current_step
        corner_stacks = (
            solver._listStackFrbrvp_GP1_CgP1[step],
            solver._listStackFlbrvp_GP1_CgP1[step],
            solver._listStackBlbrvp_GP1_CgP1[step],
            solver._listStackBrbrvp_GP1_CgP1[step],
        )
        names = (
            "stackFrbrvp_GP1_CgP1",
            "stackFlbrvp_GP1_CgP1",
            "stackBlbrvp_GP1_CgP1",
            "stackBrbrvp_GP1_CgP1",
            "stackCblvpr_GP1_CgP1",
            "stackCblvpf_GP1_CgP1",
            "stackCblvpl_GP1_CgP1",
            "stackCblvpb_GP1_CgP1",
            "stackRbrv_GP1",
            "stackFbrv_GP1",
            "stackLbrv_GP1",
            "stackBbrv_GP1",
        )
        vectorized = {name: getattr(solver, name).copy() for name in names}

        # Zero the arrays, then rewrite them one Panel at a time the original way.
        for name in names:
            getattr(solver, name)[:] = 0.0
        for position, panel in enumerate(solver.panels):
            ps._functions.update_ring_vortex_solvers_panel_attributes(
                ring_vortex_solver=solver,
                global_panel_position=position,
                panel=panel,
                Frrvp_GP1_CgP1=corner_stacks[0][position],
                Flrvp_GP1_CgP1=corner_stacks[1][position],
                Blrvp_GP1_CgP1=corner_stacks[2][position],
                Brrvp_GP1_CgP1=corner_stacks[3][position],
            )

        for name in names:
            with self.subTest(array=name):
                per_panel = getattr(solver, name)
                self.assertEqual(per_panel.tobytes(), vectorized[name].tobytes())


class TestUnsteadyRingVortexLatticeMethodSolverHookDefaults(unittest.TestCase):
    """Tests for the default implementations of the three solver extension hooks added
    to support coupled subclasses: _initialize_step_vortices,
    _reinitialize_step_arrays_hook, and _update_next_step_hook."""

    def setUp(self) -> None:
        """Set up a fresh solver for each hook-default test."""
        self.solver = solver_fixtures.make_unsteady_ring_solver_fixture()

    def test_initialize_step_vortices_initializes_all_on_step_zero(self) -> None:
        """Test that the default _initialize_step_vortices initializes bound vortices
        for all steps when called with step == 0."""
        with patch.object(
            type(self.solver), "_initialize_panel_vortices", autospec=True
        ) as mock_init:
            self.solver._initialize_step_vortices(0)
        mock_init.assert_called_once_with(self.solver)

    def test_initialize_step_vortices_is_noop_for_later_steps(self) -> None:
        """Test that the default _initialize_step_vortices is a no-op for any step
        greater than 0."""
        later_steps = [1, 2, self.solver.num_steps - 1]
        with patch.object(
            type(self.solver), "_initialize_panel_vortices", autospec=True
        ) as mock_init:
            for step in later_steps:
                self.solver._initialize_step_vortices(step)
        mock_init.assert_not_called()

    def test_reinitialize_step_arrays_hook_default_is_noop(self) -> None:
        """Test that the default _reinitialize_step_arrays_hook is a no-op."""
        self.solver._reinitialize_step_arrays_hook()

    def test_update_next_step_hook_default_is_noop(self) -> None:
        """Test that the default _update_next_step_hook is a no-op for all steps."""
        for step in [0, 1, self.solver.num_steps - 1]:
            with self.subTest(step=step):
                self.solver._update_next_step_hook(step)

    def test_models_body_rates_default_is_false(self) -> None:
        """Test that the base solver does not model body rates by default."""
        self.assertFalse(self.solver._models_body_rates)

    def test_current_omegas_default_is_zero(self) -> None:
        """Test that the default _currentOmegasRad_GP1__E returns a zero vector, so the
        base solver contributes no body-rotation velocity."""
        omegasRad_GP1__E = self.solver._currentOmegasRad_GP1__E()
        np.testing.assert_array_equal(omegasRad_GP1__E, np.zeros(3))

    def test_apply_body_rate_without_rotation_is_noop(self) -> None:
        """Test that _apply_body_rate returns the base velocities unchanged when the
        solver models no body rotation."""
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        base_velocity = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
        self.assertIs(
            self.solver._apply_body_rate(points, base_velocity), base_velocity
        )
