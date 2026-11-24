"""This module contains classes to test the array path's per run Panel layout and its
per time step geometry."""

import unittest

import numpy as np

import pterasoftware as ps

# noinspection PyProtectedMember
from pterasoftware import _mesh_arrays, _step_geometry
from tests.unit.fixtures import (
    geometry_fixtures,
    operating_point_fixtures,
    problem_fixtures,
    solver_fixtures,
)


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


# The largest differences allowed between the array path's step geometry and the object
# path's, in meters for positions and lengths, and unitless for unit vectors. Positions
# match bit for bit on the development machine, and the vectorized norms (in the unit
# normals, areas, standard mean chords, and Katz lengths) round about one unit in the
# last place differently.
_POSITION_TOLERANCE = 1.0e-12
_NORM_TOLERANCE = 1.0e-13


def _wing_panels(wing: ps.geometry.wing.Wing) -> np.ndarray:
    """Returns a meshed Wing's Panels in the solvers' order (chordwise row outer,
    spanwise column inner).

    :param wing: The meshed Wing.
    :return: A 1D ndarray of its Panels.
    """
    panels = wing.panels
    assert panels is not None
    return np.ravel(panels)


def _plain_solvers() -> (
    list[ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver]
):
    """Returns plain solvers whose geometry is static, pitching (so each Wing's standard
    mean chord changes every time step), and spread over several Airplanes.

    :return: The solvers, not yet run.
    """
    return [
        ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(
            make()
        )
        for make in (
            problem_fixtures.make_basic_unsteady_problem_fixture,
            problem_fixtures.make_pitching_tip_unsteady_problem_fixture,
            problem_fixtures.make_multi_airplane_unsteady_problem_fixture,
        )
    ]


def _step_geometries_from_wings(
    solver: ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver,
) -> list[_step_geometry.StepGeometry]:
    """Builds every time step's StepGeometry from the solver's problem's built Wings
    (through wing_mesh_from_wing), in order.

    :param solver: The solver.
    :return: The StepGeometries, one per time step.
    """
    layout_plan = _step_geometry.build_layout_plan(solver.steady_problems[0].airplanes)
    geometries: list[_step_geometry.StepGeometry] = []
    for steady_problem in solver.steady_problems:
        wing_meshes = []
        wingsCg_GP1_CgP1 = []
        for airplane in steady_problem.airplanes:
            for wing in airplane.wings:
                wing_meshes.append(_mesh_arrays.wing_mesh_from_wing(wing))
                wingsCg_GP1_CgP1.append(airplane.Cg_GP1_CgP1)
        geometries.append(
            _step_geometry.build_step_geometry(
                layout_plan,
                wing_meshes,
                wingsCg_GP1_CgP1,
                steady_problem.operating_point.vInf_GP1__E,
                solver.delta_time,
                geometries[-1] if geometries else None,
            )
        )
    return geometries


class TestBuildStepGeometry(unittest.TestCase):
    """This is a class with functions to test build_step_geometry against the Panels'
    and Wings' properties and _initialize_panel_vortices_at."""

    def test_matches_the_panels_and_wings(self) -> None:
        """Test that every time step's StepGeometry holds its Panels' corners and
        collocation points (in the first Airplane's geometry axes), unit normals, and
        areas, and its Wings' standard mean chords, in global Panel order, with read
        only arrays."""
        for solver in _plain_solvers():
            geometries = _step_geometries_from_wings(solver)
            for step, (steady_problem, geometry) in enumerate(
                zip(solver.steady_problems, geometries)
            ):
                with self.subTest(problem=solver.unsteady_problem, step=step):
                    wings = [
                        wing
                        for airplane in steady_problem.airplanes
                        for wing in airplane.wings
                    ]
                    panels = [panel for wing in wings for panel in _wing_panels(wing)]
                    for field, attribute, tolerance in (
                        ("stackFrpp_GP1_CgP1", "Frpp_GP1_CgP1", _POSITION_TOLERANCE),
                        ("stackFlpp_GP1_CgP1", "Flpp_GP1_CgP1", _POSITION_TOLERANCE),
                        ("stackBlpp_GP1_CgP1", "Blpp_GP1_CgP1", _POSITION_TOLERANCE),
                        ("stackBrpp_GP1_CgP1", "Brpp_GP1_CgP1", _POSITION_TOLERANCE),
                        ("stackCpp_GP1_CgP1", "Cpp_GP1_CgP1", _POSITION_TOLERANCE),
                        ("stackUnitNormals_GP1", "unitNormal_GP1", _NORM_TOLERANCE),
                        ("panel_areas", "area", _NORM_TOLERANCE),
                    ):
                        np.testing.assert_allclose(
                            getattr(geometry, field),
                            np.array(
                                [getattr(panel, attribute) for panel in panels],
                                dtype=float,
                            ),
                            rtol=0.0,
                            atol=tolerance,
                            err_msg=field,
                        )
                    np.testing.assert_allclose(
                        geometry.wing_standard_mean_chords,
                        np.array(
                            [wing.standard_mean_chord for wing in wings], dtype=float
                        ),
                        rtol=0.0,
                        atol=_NORM_TOLERANCE,
                    )
                    for array in geometry:
                        self.assertFalse(array.flags.writeable)

    def test_matches_the_panels_with_an_offset_airplane(self) -> None:
        """Test that the StepGeometry of a SteadyProblem whose second Airplane's CG is
        away from the first Airplane's holds the Panels' corners and collocation points
        in the first Airplane's geometry axes, relative to its CG, as SteadyProblem
        gives them."""
        steady_problem = ps.problems.SteadyProblem(
            airplanes=[
                geometry_fixtures.make_first_airplane_fixture(),
                geometry_fixtures.make_follower_airplane_fixture(),
            ],
            operating_point=operating_point_fixtures.make_basic_operating_point_fixture(),
        )
        self.assertFalse(np.all(steady_problem.airplanes[1].Cg_GP1_CgP1 == 0.0))
        wing_meshes = []
        wingsCg_GP1_CgP1 = []
        for airplane in steady_problem.airplanes:
            for wing in airplane.wings:
                wing_meshes.append(_mesh_arrays.wing_mesh_from_wing(wing))
                wingsCg_GP1_CgP1.append(airplane.Cg_GP1_CgP1)
        geometry = _step_geometry.build_step_geometry(
            _step_geometry.build_layout_plan(steady_problem.airplanes),
            wing_meshes,
            wingsCg_GP1_CgP1,
            steady_problem.operating_point.vInf_GP1__E,
            0.01,
            None,
        )
        panels = [
            panel
            for airplane in steady_problem.airplanes
            for wing in airplane.wings
            for panel in _wing_panels(wing)
        ]
        for field, attribute in (
            ("stackFrpp_GP1_CgP1", "Frpp_GP1_CgP1"),
            ("stackBlpp_GP1_CgP1", "Blpp_GP1_CgP1"),
            ("stackCpp_GP1_CgP1", "Cpp_GP1_CgP1"),
        ):
            np.testing.assert_allclose(
                getattr(geometry, field),
                np.array([getattr(panel, attribute) for panel in panels], dtype=float),
                rtol=0.0,
                atol=_POSITION_TOLERANCE,
                err_msg=field,
            )

    def test_ring_corners_match_initialize_panel_vortices_at(self) -> None:
        """Test that every time step's StepGeometry's bound ring vortex corners are the
        ones _initialize_panel_vortices_at writes on the object path, including the
        trailing edge Panels' back corners at the first time step and afterwards."""
        for solver in _plain_solvers():
            geometries = _step_geometries_from_wings(solver)
            solver._initialize_panel_vortices()
            for step, geometry in enumerate(geometries):
                with self.subTest(problem=solver.unsteady_problem, step=step):
                    for field, expected in (
                        (
                            "stackFrbrvp_GP1_CgP1",
                            solver._listStackFrbrvp_GP1_CgP1[step],
                        ),
                        (
                            "stackFlbrvp_GP1_CgP1",
                            solver._listStackFlbrvp_GP1_CgP1[step],
                        ),
                        (
                            "stackBlbrvp_GP1_CgP1",
                            solver._listStackBlbrvp_GP1_CgP1[step],
                        ),
                        (
                            "stackBrbrvp_GP1_CgP1",
                            solver._listStackBrbrvp_GP1_CgP1[step],
                        ),
                    ):
                        np.testing.assert_allclose(
                            getattr(geometry, field),
                            expected,
                            rtol=0.0,
                            atol=_POSITION_TOLERANCE,
                            err_msg=field,
                        )


class TestCollapseFunctions(unittest.TestCase):
    """This is a class with functions to test bound_r_c0s, seed_points,
    wake_vortex_ages, and katz_geometry against what the object path's collapse writes
    at every time step."""

    def test_match_the_object_paths_collapse(self) -> None:
        """Test that, at every time step, the functions give the object path's bound
        core radii, streamline seed points, wake ring vortex ages, and Katz chord and
        span lengths and tangents."""
        for solver in _plain_solvers():
            geometries = _step_geometries_from_wings(solver)
            layout_plan = _step_geometry.build_layout_plan(
                solver.steady_problems[0].airplanes
            )
            solver._force_method = "katz"
            solver._initialize_panel_vortices()
            for step, geometry in enumerate(geometries):
                with self.subTest(problem=solver.unsteady_problem, step=step):
                    steady_problem = solver.steady_problems[step]
                    solver._current_step = step
                    solver.current_airplanes = steady_problem.airplanes
                    solver.current_operating_point = steady_problem.operating_point
                    solver._prepare_step_geometry()

                    np.testing.assert_allclose(
                        _step_geometry.bound_r_c0s(layout_plan, geometry),
                        solver._current_bound_r_c0s,
                        rtol=0.0,
                        atol=_NORM_TOLERANCE,
                    )
                    np.testing.assert_allclose(
                        _step_geometry.seed_points(layout_plan, geometry),
                        solver.stackSeedPoints_GP1_CgP1,
                        rtol=0.0,
                        atol=_POSITION_TOLERANCE,
                    )
                    self.assertTrue(
                        np.array_equal(
                            _step_geometry.wake_vortex_ages(
                                layout_plan,
                                step,
                                solver._max_wake_rows,
                                solver.delta_time,
                            ),
                            solver._current_wake_vortex_ages,
                        )
                    )

                    katz_geometry = _step_geometry.katz_geometry(geometry)
                    for field, expected in (
                        ("panel_chord_lengths", solver._panel_chord_lengths),
                        ("panel_span_lengths", solver._panel_span_lengths),
                        (
                            "stackChordwiseTangents_GP1",
                            solver._stackChordwiseTangents_GP1,
                        ),
                        (
                            "stackSpanwiseTangents_GP1",
                            solver._stackSpanwiseTangents_GP1,
                        ),
                    ):
                        np.testing.assert_allclose(
                            getattr(katz_geometry, field),
                            expected,
                            rtol=0.0,
                            atol=_NORM_TOLERANCE,
                            err_msg=field,
                        )

    def test_wake_vortex_ages_with_truncation(self) -> None:
        """Test that wake_vortex_ages keeps at most max_wake_rows rows per Wing, each
        Wing's block holding its rows newest first, repeated across its spanwise
        Panels."""
        solver = _plain_solvers()[2]
        layout_plan = _step_geometry.build_layout_plan(
            solver.steady_problems[0].airplanes
        )
        ages = _step_geometry.wake_vortex_ages(layout_plan, 5, 2, 0.1)
        expected = np.concatenate(
            [
                np.repeat(np.array([0.1, 0.2]), num_spanwise_panels)
                for _, num_spanwise_panels in layout_plan.wing_shapes
            ]
        )
        np.testing.assert_allclose(ages, expected, rtol=0.0, atol=1.0e-15)
        self.assertEqual(
            _step_geometry.wake_vortex_ages(layout_plan, 0, None, 0.1).size, 0
        )


if __name__ == "__main__":
    unittest.main()
