"""This module contains classes to test the
FreeFlightUnsteadyRingVortexLatticeMethodSolver class."""

import unittest
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np

import pterasoftware as ps
from pterasoftware.free_flight_unsteady_ring_vortex_lattice_method import (
    FreeFlightUnsteadyRingVortexLatticeMethodSolver,
)
from tests.unit.fixtures import (
    operating_point_fixtures,
    problem_fixtures,
    solver_fixtures,
)


class TestFreeFlightUnsteadyRingVortexLatticeMethodSolver(unittest.TestCase):
    """This is a class with functions to test
    FreeFlightUnsteadyRingVortexLatticeMethodSolvers."""

    def setUp(self) -> None:
        """Set up a fresh solver for each test."""
        self.solver = solver_fixtures.make_free_flight_unsteady_ring_solver_fixture()

    def test_initialization_accepts_free_flight_unsteady_problem(self) -> None:
        """Test that initialization accepts a FreeFlightUnsteadyProblem."""
        self.assertIsInstance(
            self.solver, FreeFlightUnsteadyRingVortexLatticeMethodSolver
        )
        self.assertIsInstance(
            self.solver,
            ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver,
        )
        self.assertIsInstance(
            self.solver.unsteady_problem, ps.problems.FreeFlightUnsteadyProblem
        )

    def test_initialization_rejects_non_free_flight_problem(self) -> None:
        """Test that initialization raises TypeError for a coupled problem that is not a
        FreeFlightUnsteadyProblem."""
        coupled_problem: Any = (
            problem_fixtures.make_basic_coupled_unsteady_problem_fixture()
        )
        with self.assertRaises(TypeError):
            FreeFlightUnsteadyRingVortexLatticeMethodSolver(coupled_problem)

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
                    FreeFlightUnsteadyRingVortexLatticeMethodSolver(invalid)

    def test_free_flight_unsteady_problem_property_narrows_unsteady_problem(
        self,
    ) -> None:
        """Test that the _free_flight_unsteady_problem property returns the same object
        as unsteady_problem, narrowed to FreeFlightUnsteadyProblem."""
        self.assertIs(
            self.solver._free_flight_unsteady_problem, self.solver.unsteady_problem
        )
        self.assertIsInstance(
            self.solver._free_flight_unsteady_problem,
            ps.problems.FreeFlightUnsteadyProblem,
        )

    def test_models_body_rates_is_true(self) -> None:
        """Test that the free-flight solver declares that it models body rates, so the
        inherited constructor permits a non zero omegas_BP1__E."""
        self.assertTrue(self.solver._models_body_rates)

    def test_permits_non_zero_body_rates(self) -> None:
        """Test that the free-flight solver constructs from a problem whose initial
        OperatingPoint carries a non zero body angular velocity, which the base solver
        would reject."""
        rotating_problem = (
            problem_fixtures.make_with_body_rates_free_flight_unsteady_problem_fixture()
        )
        solver = FreeFlightUnsteadyRingVortexLatticeMethodSolver(rotating_problem)
        self.assertIsInstance(solver, FreeFlightUnsteadyRingVortexLatticeMethodSolver)

    def test_current_omegas_without_rotation_is_zero(self) -> None:
        """Test that _currentOmegasRad_GP1__E returns a zero vector when the current
        OperatingPoint carries no body rotation."""
        omegasRad_GP1__E = self.solver._currentOmegasRad_GP1__E()
        np.testing.assert_array_equal(omegasRad_GP1__E, np.zeros(3))

    def test_current_omegas_transforms_body_rate(self) -> None:
        """Test that _currentOmegasRad_GP1__E transforms the current OperatingPoint's
        body rate from the body axes in degrees per second to the geometry axes in
        radians per second.

        The body-to-geometry transformation negates the x and z components, and the
        result is converted from degrees per second to radians per second.
        """
        self.solver.current_operating_point = (
            operating_point_fixtures.make_with_body_rates_operating_point_fixture()
        )

        omegasRad_GP1__E = self.solver._currentOmegasRad_GP1__E()

        # The fixture's body rate is (0, 0, 1) degrees per second in body axes, which
        # becomes (0, 0, -1) degrees per second in geometry axes, then is converted to
        # radians per second.
        expected_omegasRad_GP1__E = np.deg2rad(np.array([0.0, 0.0, -1.0]))
        np.testing.assert_allclose(omegasRad_GP1__E, expected_omegasRad_GP1__E)

    def test_current_omegas_negates_x_and_z_preserves_y(self) -> None:
        """Test that _currentOmegasRad_GP1__E negates the x and z components and
        preserves the y component when transforming the body rate from the body axes to
        the geometry axes.

        Using a body rate with three distinct non zero components pins down the full
        body to geometry flip, which a body rate aligned with a single axis cannot.
        """
        self.solver.current_operating_point = (
            operating_point_fixtures.make_with_full_body_rates_operating_point_fixture()
        )

        omegasRad_GP1__E = self.solver._currentOmegasRad_GP1__E()

        # The fixture's body rate is (1, 2, 3) degrees per second in body axes, which
        # becomes (-1, 2, -3) degrees per second in geometry axes, then is converted to
        # radians per second.
        expected_omegasRad_GP1__E = np.deg2rad(np.array([-1.0, 2.0, -3.0]))
        np.testing.assert_allclose(omegasRad_GP1__E, expected_omegasRad_GP1__E)

    def test_declares_substep_slots(self) -> None:
        """Test that the subclass declares exactly the strongly coupled sub-iteration's
        transient slots and so does not gain an instance __dict__ that would defeat the
        parent's __slots__."""
        self.assertEqual(
            FreeFlightUnsteadyRingVortexLatticeMethodSolver.__slots__,
            (
                "_substep_next_step",
                "_substep_next_steady_problem",
                "_substep_next_operating_point",
                "_substepStackVIndGridWrvp_GP1__E",
                "_substep_gamma_n",
                "_substep_gamma_n_minus_1",
                "_substep_saved_state",
                "_substep_final_trial",
                "_stepStackPanelForces_GP1",
                "_stepStackPanelMoments_GP1_CgP1",
            ),
        )
        with self.assertRaises(AttributeError):
            setattr(self.solver, "not_a_real_attribute", 42)


class TestFreeFlightSolverSubstepDispatch(unittest.TestCase):
    """Tests for the substep-aware geometry and operating point dispatch overrides."""

    def setUp(self) -> None:
        """Set up a fresh solver for each dispatch test."""
        self.solver = solver_fixtures.make_free_flight_unsteady_ring_solver_fixture()

    def test_get_steady_problem_at_dispatches_on_substep_state(self) -> None:
        """Test that _get_steady_problem_at returns the transient next-step
        SteadyProblem only for the next step during a sub-iteration, and otherwise
        defers to the committed accessor."""
        committed = self.solver._get_steady_problem_at(0)
        transient = MagicMock()

        self.solver._substep_next_step = 1
        self.solver._substep_next_steady_problem = transient

        # The transient problem is returned for the matching next step only.
        self.assertIs(self.solver._get_steady_problem_at(1), transient)
        # A non-matching step falls through to the committed accessor.
        self.assertIs(self.solver._get_steady_problem_at(0), committed)

        # With no transient problem set, the committed accessor is used.
        self.solver._substep_next_steady_problem = None
        self.assertIs(self.solver._get_steady_problem_at(0), committed)

    def test_operating_point_at_dispatches_on_substep_state(self) -> None:
        """Test that _operating_point_at returns the trial OperatingPoint only for the
        next step during a sub-iteration, and otherwise defers to the committed
        accessor."""
        committed_operating_point = self.solver._operating_point_at(0)
        trial_operating_point = MagicMock()

        self.solver._substep_next_step = 1
        self.solver._substep_next_operating_point = trial_operating_point

        # The trial operating point is returned for the matching next step only.
        self.assertIs(self.solver._operating_point_at(1), trial_operating_point)
        # A non-matching step falls through to the committed accessor.
        self.assertIs(self.solver._operating_point_at(0), committed_operating_point)

        # With no trial operating point set, the committed accessor is used.
        self.solver._substep_next_operating_point = None
        self.assertIs(self.solver._operating_point_at(0), committed_operating_point)


class TestFreeFlightSolverSubstepLifecycle(unittest.TestCase):
    """Tests for the transient working state freeze_substep and restore_substep
    manage."""

    def setUp(self) -> None:
        """Set up a fresh solver for each lifecycle test."""
        self.solver = solver_fixtures.make_free_flight_unsteady_ring_solver_fixture()

    def test_freeze_substep_snapshots_strengths_as_copies(self) -> None:
        """Test that freeze_substep records the next step, stores the transient
        SteadyProblem, and snapshots the current and previous bound vortex strengths as
        independent copies.

        The snapshots must be copies, because the trials overwrite the solver's strength
        arrays while evaluating the next step, and a view would carry that mutation into
        the snapshot.
        """
        self.solver._current_step = 0
        self.solver._current_bound_vortex_strengths = np.array(
            [1.0, 2.0, 3.0], dtype=float
        )
        self.solver._last_bound_vortex_strengths = np.array(
            [4.0, 5.0, 6.0], dtype=float
        )
        next_steady_problem = MagicMock()

        self.solver.freeze_substep(next_steady_problem)

        self.assertEqual(self.solver._substep_next_step, 1)
        self.assertIs(self.solver._substep_next_steady_problem, next_steady_problem)
        self.assertIsNone(self.solver._substep_next_operating_point)
        np.testing.assert_array_equal(
            self.solver._substep_gamma_n, np.array([1.0, 2.0, 3.0])
        )
        np.testing.assert_array_equal(
            self.solver._substep_gamma_n_minus_1, np.array([4.0, 5.0, 6.0])
        )

        # Mutating the source strengths must not change the snapshots.
        self.solver._current_bound_vortex_strengths[0] = 999.0
        self.solver._last_bound_vortex_strengths[0] = 999.0
        assert self.solver._substep_gamma_n is not None
        assert self.solver._substep_gamma_n_minus_1 is not None
        self.assertEqual(self.solver._substep_gamma_n[0], 1.0)
        self.assertEqual(self.solver._substep_gamma_n_minus_1[0], 4.0)

    def test_freeze_substep_skips_induced_precompute_for_prescribed_wake(self) -> None:
        """Test that freeze_substep leaves the frozen wake induced velocities unset for
        a prescribed wake, which has no induced transport to precompute."""
        self.solver._prescribed_wake = True
        self.solver._current_step = 0
        self.solver._current_bound_vortex_strengths = np.ones(3, dtype=float)
        self.solver._last_bound_vortex_strengths = np.zeros(3, dtype=float)

        self.solver.freeze_substep(MagicMock())

        self.assertIsNone(self.solver._substepStackVIndGridWrvp_GP1__E)

    def test_freeze_substep_saves_solver_state(self) -> None:
        """Test that freeze_substep saves a reference to every solver attribute, so
        restore_substep can put the step's state back without solving it again, and
        skips the transient substep slots, which restore_substep manages itself."""
        self.solver._prescribed_wake = True
        self.solver._current_step = 0
        current_strengths = np.ones(3, dtype=float)
        self.solver._current_bound_vortex_strengths = current_strengths
        self.solver._last_bound_vortex_strengths = np.zeros(3, dtype=float)

        self.solver.freeze_substep(MagicMock())

        saved_state = self.solver._substep_saved_state
        assert saved_state is not None
        # References, not copies: the trials replace these attributes rather than
        # writing into them, so the saved objects stay intact.
        self.assertIs(saved_state["_current_bound_vortex_strengths"], current_strengths)
        self.assertEqual(saved_state["_current_step"], 0)
        self.assertEqual(saved_state["num_steps"], self.solver.num_steps)
        self.assertFalse(any(name.startswith("_substep") for name in saved_state))

    def test_restore_substep_clears_transient_state(self) -> None:
        """Test that restore_substep puts back the state freeze_substep saved, without
        solving the current step again, and clears every transient sub-iteration slot
        except the frozen wake grid induced velocity.

        A leftover non-None slot would redirect the next step's geometry and wake reads
        to a stale scratch copy, so the clearing is a correctness contract. The frozen
        wake grid induced velocity is the exception: it is kept for the official wake
        build, which reuses and clears it.
        """
        frozen_velocities = [[np.zeros(3, dtype=float)]]
        saved_strengths = np.array([1.0, 2.0, 3.0], dtype=float)
        self.solver._substep_next_step = 1
        self.solver._substep_next_steady_problem = MagicMock()
        self.solver._substep_next_operating_point = MagicMock()
        self.solver._substepStackVIndGridWrvp_GP1__E = frozen_velocities
        self.solver._substep_gamma_n = np.ones(3, dtype=float)
        self.solver._substep_gamma_n_minus_1 = np.zeros(3, dtype=float)
        self.solver._substep_saved_state = {
            "_current_step": 0,
            "_current_bound_vortex_strengths": saved_strengths,
        }
        # Stand in for the trials, which point the solver at the next step.
        self.solver._current_step = 1
        self.solver._current_bound_vortex_strengths = np.zeros(3, dtype=float)

        with patch.object(
            ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver,
            "_evaluate_step_aerodynamics",
        ) as mock_evaluate_step_aerodynamics:
            self.solver.restore_substep(step=0)

        mock_evaluate_step_aerodynamics.assert_not_called()
        self.assertEqual(self.solver._current_step, 0)
        self.assertIs(self.solver._current_bound_vortex_strengths, saved_strengths)
        self.assertIsNone(self.solver._substep_saved_state)
        self.assertIsNone(self.solver._substep_next_step)
        self.assertIsNone(self.solver._substep_next_steady_problem)
        self.assertIsNone(self.solver._substep_next_operating_point)
        self.assertIs(self.solver._substepStackVIndGridWrvp_GP1__E, frozen_velocities)
        self.assertIsNone(self.solver._substep_gamma_n)
        self.assertIsNone(self.solver._substep_gamma_n_minus_1)

    def test_populate_next_airplanes_wake_reuses_frozen_velocity(self) -> None:
        """Test that the official wake build passes the frozen wake grid induced
        velocity to the wake point update and then clears it, and passes None when there
        is no frozen velocity (a prescribed step)."""
        frozen_velocities = [[np.zeros(3, dtype=float)]]
        self.solver._substepStackVIndGridWrvp_GP1__E = frozen_velocities

        # The solver has slots, so patch the methods on its class.
        solver_class = type(self.solver)
        with (
            patch.object(
                solver_class, "_populate_next_airplanes_wake_vortex_points"
            ) as mock_points,
            patch.object(
                solver_class, "_populate_next_airplanes_wake_vortices"
            ) as mock_vortices,
        ):
            self.solver._populate_next_airplanes_wake()

            mock_points.assert_called_once_with(frozen_velocities)
            mock_vortices.assert_called_once_with()
            self.assertIsNone(self.solver._substepStackVIndGridWrvp_GP1__E)

            # A second build with no frozen velocity passes None, so the inherited wake
            # build computes the velocity itself.
            self.solver._populate_next_airplanes_wake()
            self.assertIsNone(mock_points.call_args.args[0])


class TestFreeFlightSolverFinalTrialReuse(unittest.TestCase):
    """Tests for keeping the final coupling trial and reusing it as the next time step's
    official solve (the reuse_final_trial approximation)."""

    def setUp(self) -> None:
        """Set up a fresh solver for each test."""
        self.solver = solver_fixtures.make_free_flight_unsteady_ring_solver_fixture()

    def _make_final_trial(self, step: int) -> Any:
        """Makes a kept final trial for a given time step with distinct arrays.

        :param step: The time step the trial belongs to.
        :return: The kept final trial.
        """
        # noinspection PyProtectedMember
        from pterasoftware.free_flight_unsteady_ring_vortex_lattice_method import (
            _FinalTrial,
        )

        return _FinalTrial(
            step=step,
            strengths=np.array([1.0, 2.0, 3.0]),
            grid_wing_wing_influences=np.eye(3),
            stack_freestream_wing_influences=np.full(3, 4.0),
            stack_wake_wing_influences=np.full(3, 5.0),
            stackPanelForces_GP1=np.full((3, 3), 6.0),
            stackPanelMoments_GP1_CgP1=np.full((3, 3), 7.0),
        )

    def test_process_panel_loads_hook_records_the_loads(self) -> None:
        """Test that the load hook records references to the per Panel loads and still
        passes them to the parent's hook."""
        forces_GP1 = np.ones((3, 3))
        moments_GP1_CgP1 = np.zeros((3, 3))
        with patch.object(
            ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver,
            "_process_panel_loads_hook",
        ) as mock_parent_hook:
            self.solver._process_panel_loads_hook(forces_GP1, moments_GP1_CgP1)

        mock_parent_hook.assert_called_once_with(forces_GP1, moments_GP1_CgP1)
        self.assertIs(self.solver._stepStackPanelForces_GP1, forces_GP1)
        self.assertIs(self.solver._stepStackPanelMoments_GP1_CgP1, moments_GP1_CgP1)

    def test_keep_final_trial_keeps_references(self) -> None:
        """Test that keep_final_trial keeps references to the final trial's strengths,
        influences, and recorded loads, with the time step the trial solved."""
        self.solver._substep_next_step = 1
        self.solver._current_step = 1
        self.solver.first_results_step = 0
        strengths = np.ones(3)
        forces_GP1 = np.ones((3, 3))
        moments_GP1_CgP1 = np.zeros((3, 3))
        self.solver._current_bound_vortex_strengths = strengths
        self.solver._currentGridWingWingInfluences__E = np.eye(3)
        self.solver._currentStackFreestreamWingInfluences__E = np.zeros(3)
        self.solver._currentStackWakeWingInfluences__E = np.zeros(3)
        self.solver._stepStackPanelForces_GP1 = forces_GP1
        self.solver._stepStackPanelMoments_GP1_CgP1 = moments_GP1_CgP1

        self.solver.keep_final_trial()

        final_trial = self.solver._substep_final_trial
        assert final_trial is not None
        self.assertEqual(final_trial.step, 1)
        self.assertIs(final_trial.strengths, strengths)
        self.assertIs(final_trial.stackPanelForces_GP1, forces_GP1)
        self.assertIs(final_trial.stackPanelMoments_GP1_CgP1, moments_GP1_CgP1)

    def test_keep_final_trial_before_first_results_step_keeps_nothing(self) -> None:
        """Test that a trial before the first results step, which calculated no loads,
        is not kept."""
        self.solver._substep_next_step = 1
        self.solver._current_step = 1
        self.solver.first_results_step = 2

        self.solver.keep_final_trial()

        self.assertIsNone(self.solver._substep_final_trial)

    def test_kept_trial_survives_restore_substep(self) -> None:
        """Test that restore_substep, which puts back every other slot, leaves the kept
        trial in place for the next time step's official solve."""
        final_trial = self._make_final_trial(step=1)
        self.solver._substep_final_trial = final_trial
        self.solver._substep_next_step = 1
        self.solver._substep_saved_state = {"_current_step": 0}

        self.solver.restore_substep(step=0)

        self.assertIs(self.solver._substep_final_trial, final_trial)

    def test_evaluate_uses_the_kept_trial_once(self) -> None:
        """Test that the official solve of the kept trial's time step builds the
        geometry but takes the strengths, influences, and loads from the trial instead
        of solving, and that the trial is used only once."""
        final_trial = self._make_final_trial(step=1)
        self.solver._substep_final_trial = final_trial
        self.solver._substep_next_step = None
        self.solver._current_step = 1

        # The solver has slots, so patch the methods on the classes.
        solver_class = type(self.solver)
        with (
            patch.object(
                ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver,
                "_evaluate_step_aerodynamics",
            ) as mock_full_solve,
            patch.object(solver_class, "_prepare_step_geometry") as mock_geometry,
            patch.object(solver_class, "_process_panel_loads_hook") as mock_loads,
        ):
            self.solver._evaluate_step_aerodynamics()

            mock_full_solve.assert_not_called()
            mock_geometry.assert_called_once_with()
            mock_loads.assert_called_once_with(
                final_trial.stackPanelForces_GP1, final_trial.stackPanelMoments_GP1_CgP1
            )
            self.assertIs(
                self.solver._current_bound_vortex_strengths, final_trial.strengths
            )
            self.assertIs(
                self.solver._currentGridWingWingInfluences__E,
                final_trial.grid_wing_wing_influences,
            )
            self.assertIsNone(self.solver._substep_final_trial)

            # The trial is used once: the next solve is a full one.
            self.solver._evaluate_step_aerodynamics()
            mock_full_solve.assert_called_once_with()

    def test_evaluate_solves_for_another_step_or_inside_a_substep(self) -> None:
        """Test that a kept trial is discarded, and the time step solved in full, when
        it belongs to a different time step or when a coupling trial is running."""
        for current_step, substep_next_step in ((2, None), (1, 1)):
            with self.subTest(
                current_step=current_step, substep_next_step=substep_next_step
            ):
                self.solver._substep_final_trial = self._make_final_trial(step=1)
                self.solver._current_step = current_step
                self.solver._substep_next_step = substep_next_step
                with patch.object(
                    ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver,
                    "_evaluate_step_aerodynamics",
                ) as mock_full_solve:
                    self.solver._evaluate_step_aerodynamics()

                mock_full_solve.assert_called_once_with()
                self.assertIsNone(self.solver._substep_final_trial)
