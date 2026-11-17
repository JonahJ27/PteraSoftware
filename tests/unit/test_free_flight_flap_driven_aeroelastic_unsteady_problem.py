"""This module contains a class to test
FreeFlightFlapDrivenAeroelasticUnsteadyProblems."""

import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import numpy.testing as npt

import pterasoftware as ps
from pterasoftware import flap_drive
from tests.unit.fixtures import flap_drive_fixtures


def _solve(
    problem: ps.free_flight_flap_driven_aeroelastic_unsteady_problem.FreeFlightFlapDrivenAeroelasticUnsteadyProblem,
) -> None:
    """Solves a FreeFlightFlapDrivenAeroelasticUnsteadyProblem.

    :param problem: The problem to solve.
    :return: None
    """
    solver = ps.free_flight_aeroelastic_unsteady_ring_vortex_lattice_method.FreeFlightAeroelasticUnsteadyRingVortexLatticeMethodSolver(
        problem
    )
    solver.run(prescribed_wake=False, show_progress=False)


class TestFreeFlightFlapDrivenAeroelasticUnsteadyProblem(unittest.TestCase):
    """This is a class with functions to test
    FreeFlightFlapDrivenAeroelasticUnsteadyProblems."""

    problem_class = (
        ps.free_flight_flap_driven_aeroelastic_unsteady_problem.FreeFlightFlapDrivenAeroelasticUnsteadyProblem
    )
    solved: (
        ps.free_flight_flap_driven_aeroelastic_unsteady_problem.FreeFlightFlapDrivenAeroelasticUnsteadyProblem
    )

    @classmethod
    def setUpClass(cls) -> None:
        """Solve one problem whose fluid torque reaches its FlapDrive, shared by the
        tests that only read a solved problem."""
        cls.solved = flap_drive_fixtures.make_free_flight_flap_driven_aeroelastic_problem_fixture(
            step_discards=1
        )
        _solve(cls.solved)

    def test_is_a_free_flight_aeroelastic_problem_with_flap_drives(self) -> None:
        """Test that the class is a FreeFlightAeroelasticUnsteadyProblem that shares the
        flap drive logic with FlapDrivenAeroelasticUnsteadyProblem."""
        problem = (
            flap_drive_fixtures.make_free_flight_flap_driven_aeroelastic_problem_fixture()
        )
        self.assertIsInstance(problem, ps.problems.FreeFlightAeroelasticUnsteadyProblem)
        self.assertIsInstance(
            problem,
            ps.flap_driven_aeroelastic_unsteady_problem._FlapDriveCoupling,
        )
        for method in (
            "initialize_next_problem",
            "_advance_flap_drives",
            "_generate_inertial_moment_function",
        ):
            self.assertIs(
                getattr(self.problem_class, method),
                getattr(
                    ps.flap_driven_aeroelastic_unsteady_problem.FlapDrivenAeroelasticUnsteadyProblem,
                    method,
                ),
            )

    def test_uses_analytical_spring_solver_by_default(self) -> None:
        """Test that a FreeFlightFlapDrivenAeroelasticUnsteadyProblem solves its
        torsional springs analytically unless told otherwise."""
        problem = (
            flap_drive_fixtures.make_free_flight_flap_driven_aeroelastic_problem_fixture()
        )
        self.assertTrue(problem.use_analytical_spring_solver)

    def test_analytical_spring_solver_can_be_turned_off(self) -> None:
        """Test that use_analytical_spring_solver=False passes through
        FreeFlightAeroelasticUnsteadyProblem to AeroelasticUnsteadyProblem."""
        problem = ps.free_flight_flap_driven_aeroelastic_unsteady_problem.FreeFlightFlapDrivenAeroelasticUnsteadyProblem(
            movement=flap_drive_fixtures.make_free_flight_flap_driven_aeroelastic_movement_fixture(
                2, 4
            ),
            wing_density=0.5,
            spring_constant_rad=20.0,
            damping_constant_rad=1.0,
            mass=1.0,
            I_BP1_CgP1=((0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1)),
            flap_drives=[flap_drive_fixtures.make_tracking_flap_drive_fixture()],
            use_analytical_spring_solver=False,
        )
        self.assertFalse(problem.use_analytical_spring_solver)

    def test_slots(self) -> None:
        """Test that instances have no dict and reject new attributes."""
        problem = (
            flap_drive_fixtures.make_free_flight_flap_driven_aeroelastic_problem_fixture()
        )
        self.assertFalse(hasattr(problem, "__dict__"))
        with self.assertRaises(AttributeError):
            setattr(problem, "new_attribute", 1.0)

    def test_movement_must_be_free_flight(self) -> None:
        """Test that an AeroelasticMovement is rejected."""
        movement: Any = (
            flap_drive_fixtures.make_flap_driven_aeroelastic_movement_fixture()
        )
        with self.assertRaises(TypeError):
            self.problem_class(
                movement=movement,
                wing_density=0.5,
                spring_constant_rad=20.0,
                damping_constant_rad=1.0,
                mass=1.0,
                I_BP1_CgP1=np.eye(3),
                flap_drives=[flap_drive_fixtures.make_tracking_flap_drive_fixture()],
            )

    def test_flap_drives_are_validated(self) -> None:
        """Test that the shared flap drive validation runs."""
        bad_drive_cases: tuple[tuple[Any, type[Exception]], ...] = (
            ([], ValueError),
            (["drive"], TypeError),
        )
        for bad_drives, error in bad_drive_cases:
            with self.subTest(bad_drives=bad_drives), self.assertRaises(error):
                self.problem_class(
                    movement=flap_drive_fixtures.make_free_flight_flap_driven_aeroelastic_movement_fixture(),
                    wing_density=0.5,
                    spring_constant_rad=20.0,
                    damping_constant_rad=1.0,
                    mass=1.0,
                    I_BP1_CgP1=np.eye(3),
                    flap_drives=bad_drives,
                )

    def test_first_step_starts_at_the_drive_angle(self) -> None:
        """Test that the first time step's Wings start at the drive's initial angle."""
        problem = (
            flap_drive_fixtures.make_free_flight_flap_driven_aeroelastic_problem_fixture()
        )
        expected = flap_drive_fixtures.sine_flap_angle_deg(0.0)
        for wing in problem.steady_problems[0].airplanes[0].wings:
            self.assertAlmostEqual(wing.angles_Gs_to_Wn_ixyz[0], expected, places=9)
        self.assertAlmostEqual(
            float(np.rad2deg(problem.flap_angles_rad[0][0])), expected, places=9
        )

    def test_reference_period_sets_the_averaging_window(self) -> None:
        """Test that the reference's period reaches the free flight movement, so the
        cycle averaging covers whole wingbeats."""
        period = 2.0 * flap_drive_fixtures.FIXTURE_DELTA_TIME
        problem = flap_drive_fixtures.make_free_flight_flap_driven_aeroelastic_problem_fixture(
            reference=flap_drive_fixtures.make_sine_flap_reference_fixture(
                period=period
            )
        )
        self.assertEqual(problem.movement.lcm_period, period)
        self.assertFalse(problem.movement.static)
        self.assertEqual(problem.first_averaging_step, problem.num_steps - 2)

    def test_step_discards_reaches_the_parent(self) -> None:
        """Test that step_discards is passed through to the aeroelastic parent."""
        self.assertEqual(self.solved.step_discards, 1)
        default = (
            flap_drive_fixtures.make_free_flight_flap_driven_aeroelastic_problem_fixture()
        )
        self.assertEqual(default.step_discards, 5)

    def test_histories_cover_every_step(self) -> None:
        """Test that every flap history has one entry per time step."""
        for history in (
            self.solved.flap_actuator_positions,
            self.solved.flap_actuator_velocities,
            self.solved.flap_actuator_accelerations,
            self.solved.flap_actuator_forces,
            self.solved.flap_fluid_torques,
            self.solved.flap_transmission_torques,
            self.solved.flap_angles_rad,
            self.solved.flap_angle_derivatives_rad,
            self.solved.flap_angle_second_derivatives_rad,
        ):
            self.assertEqual(len(history), 1)
            self.assertEqual(len(history[0]), self.solved.num_steps)

    def test_geometry_follows_the_drive(self) -> None:
        """Test that every time step's Wings, including those built during the free
        flight phase, are at the drive's flap angle."""
        expected = np.rad2deg(self.solved.flap_angles_rad[0])
        for step, steady_problem in enumerate(self.solved.steady_problems):
            for wing in steady_problem.airplanes[0].wings:
                self.assertAlmostEqual(
                    wing.angles_Gs_to_Wn_ixyz[0], expected[step], places=9
                )
        self.assertGreater(float(np.ptp(expected)), 1.0)

    def test_fluid_torque_acts_after_the_discarded_steps(self) -> None:
        """Test that the drive feels no fluid torque during the discarded steps, and the
        UVLM's torque after them."""
        torques = self.solved.flap_fluid_torques[0]
        self.assertEqual(torques[0], 0.0)
        self.assertEqual(torques[1], 0.0)
        self.assertTrue(all(torque != 0.0 for torque in torques[2:]))

    def test_body_moves_in_free_flight(self) -> None:
        """Test that the rigid body state stays at its initial value during the
        prescribed phase and changes during the free flight phase."""
        movement = self.solved.movement
        assert isinstance(
            movement, ps.movements.free_flight_movement.FreeFlightAeroelasticMovement
        )
        speeds = [
            operating_point.vCg__E
            for operating_point in movement.operating_point_movement.operating_points
        ]
        self.assertEqual(len(speeds), self.solved.num_steps)
        prescribed = movement.prescribed_num_steps
        npt.assert_allclose(speeds[: prescribed + 1], speeds[0])
        self.assertNotAlmostEqual(speeds[-1], speeds[0], places=6)

    def test_aerodynamic_model_receives_the_body_state(self) -> None:
        """Test that a FlapDrive's aerodynamic model receives the OperatingPoint of the
        time step just solved, which in free flight is the body's state, never a trial
        state from the sub-iteration."""
        received: list[Any] = []

        def model(aerodynamic_inputs: flap_drive.AerodynamicModelInputs) -> float:
            received.append(aerodynamic_inputs)
            return 0.0

        tracking = flap_drive_fixtures.make_tracking_flap_drive_fixture()
        drive = flap_drive.FlapDrive(
            wing_indices=tracking.wing_indices,
            transmission=tracking.transmission,
            actuator=tracking.actuator,
            actuator_inertia=tracking.actuator_inertia,
            reference=tracking.reference,
            use_position_lookup=False,
            feedforward_aerodynamic=model,
        )
        problem = flap_drive_fixtures.make_free_flight_flap_driven_aeroelastic_problem_fixture(
            drive=drive
        )
        _solve(problem)

        # Each call must see the OperatingPoint of a committed time step whose interval
        # to the next time step contains the call's time: the state of the time step
        # just solved when the drive advanced.
        committed = [
            steady_problem.operating_point for steady_problem in problem.steady_problems
        ]
        delta_time = problem.delta_time
        self.assertGreater(len(received), problem.num_steps)
        steps_seen = set()
        for aerodynamic_inputs in received:
            matches = [
                step
                for step, operating_point in enumerate(committed)
                if aerodynamic_inputs.operating_point is operating_point
            ]
            self.assertEqual(len(matches), 1)
            step = matches[0]
            self.assertGreaterEqual(aerodynamic_inputs.time, step * delta_time - 1e-12)
            self.assertLessEqual(
                aerodynamic_inputs.time, (step + 1) * delta_time + 1e-12
            )
            steps_seen.add(step)

        # Every time step but the last advances the drive, including the free flight
        # time steps, whose OperatingPoints come from the rigid body dynamics.
        self.assertEqual(steps_seen, set(range(problem.num_steps - 1)))

    def test_save_and_load(self) -> None:
        """Test that a solved problem saves and loads with its flap histories."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "problem.psz"
            ps.save(path, self.solved)
            loaded = ps.load(path)
        self.assertIsInstance(loaded, self.problem_class)
        assert isinstance(loaded, self.problem_class)
        npt.assert_array_equal(loaded.flap_angles_rad, self.solved.flap_angles_rad)
        npt.assert_array_equal(
            loaded.flap_fluid_torques, self.solved.flap_fluid_torques
        )
        self.assertEqual(loaded.step_discards, self.solved.step_discards)


if __name__ == "__main__":
    unittest.main()
