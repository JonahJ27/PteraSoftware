"""This module contains classes to test FlapDrivenAeroelasticWingMovements and
FlapDrivenAeroelasticUnsteadyProblems."""

import copy
import unittest
from typing import Any
from unittest import mock

import numpy as np
import numpy.testing as npt

import pterasoftware as ps
from pterasoftware import flap_drive
from tests.unit.fixtures import flap_drive_fixtures


def _cross_section_movements(wing_movement: Any) -> Any:
    """Returns a Wing movement's WingCrossSection movements as a list.

    The movement classes want a list of the aeroelastic type, but a Wing movement
    exposes a tuple of the core type, so the hand-off is typed as Any.

    :param wing_movement: The Wing movement to read.
    :return: The list of WingCrossSection movements.
    """
    return list(wing_movement.wing_cross_section_movements)


def _make_wing_movement() -> (
    ps.movements.flap_driven_aeroelastic_wing_movement.FlapDrivenAeroelasticWingMovement
):
    """Builds a FlapDrivenAeroelasticWingMovement from the problem fixtures' movement.

    :return: The first Wing's movement.
    """
    movement = flap_drive_fixtures.make_flap_driven_aeroelastic_movement_fixture()
    wing_movement = movement.airplane_movements[0].wing_movements[0]
    assert isinstance(
        wing_movement,
        ps.movements.flap_driven_aeroelastic_wing_movement.FlapDrivenAeroelasticWingMovement,
    )
    return wing_movement


class TestAeroelasticWingMovementOverride(unittest.TestCase):
    """This is a class with functions to test the flap angle override that
    AeroelasticWingMovement's generate_wing_at_time_step accepts."""

    def setUp(self) -> None:
        """Set up an ordinary AeroelasticWingMovement."""
        movement = flap_drive_fixtures.make_flap_driven_aeroelastic_movement_fixture()
        driven = movement.airplane_movements[0].wing_movements[0]
        self.wing_movement = (
            ps.movements.aeroelastic_wing_movement.AeroelasticWingMovement(
                base_wing=driven.base_wing,
                wing_cross_section_movements=_cross_section_movements(driven),
                ampAngles_Gs_to_Wn_ixyz=(
                    flap_drive_fixtures.SINE_FLAP_AMPLITUDE,
                    0.0,
                    0.0,
                ),
                periodAngles_Gs_to_Wn_ixyz=(
                    flap_drive_fixtures.SINE_FLAP_PERIOD,
                    0.0,
                    0.0,
                ),
                phaseAngles_Gs_to_Wn_ixyz=(
                    flap_drive_fixtures.SINE_FLAP_PHASE,
                    0.0,
                    0.0,
                ),
            )
        )

    def test_default_is_unchanged(self) -> None:
        """Test that the prescribed flap angle is used when no override is given."""
        wing = self.wing_movement.generate_wing_at_time_step(2, 0.05)
        self.assertAlmostEqual(
            wing.angles_Gs_to_Wn_ixyz[0],
            flap_drive_fixtures.sine_flap_angle_deg(0.1),
        )

    def test_override_replaces_only_the_flap_angle(self) -> None:
        """Test that the x component is replaced and the other components are not."""
        wing = self.wing_movement.generate_wing_at_time_step(
            2, 0.05, overrideAngleX_Gs_to_Wn_ixyz=-7.5
        )
        npt.assert_allclose(wing.angles_Gs_to_Wn_ixyz, [-7.5, 0.0, 0.0])

    def test_override_moves_the_rotation_point_adjustment(self) -> None:
        """Test that the rotation point offset adjustment sees the overridden angle."""
        driven = _make_wing_movement()
        offset_movement = (
            ps.movements.aeroelastic_wing_movement.AeroelasticWingMovement(
                base_wing=driven.base_wing,
                wing_cross_section_movements=_cross_section_movements(driven),
                ampAngles_Gs_to_Wn_ixyz=(15.0, 0.0, 0.0),
                periodAngles_Gs_to_Wn_ixyz=(1.0, 0.0, 0.0),
                rotationPointOffset_Gs_Ler=(0.0, 0.3, 0.0),
            )
        )
        at_zero = offset_movement.generate_wing_at_time_step(
            1, 0.05, overrideAngleX_Gs_to_Wn_ixyz=0.0
        )
        at_ten = offset_movement.generate_wing_at_time_step(
            1, 0.05, overrideAngleX_Gs_to_Wn_ixyz=10.0
        )
        self.assertFalse(np.allclose(at_zero.Ler_Gs_Cgs, at_ten.Ler_Gs_Cgs))

    def test_invalid_override_is_rejected_by_wing(self) -> None:
        """Test that an override outside a Wing's angle range is rejected."""
        with self.assertRaises(ValueError):
            self.wing_movement.generate_wing_at_time_step(
                1, 0.05, overrideAngleX_Gs_to_Wn_ixyz=120.0
            )


class TestFlapDrivenAeroelasticWingMovement(unittest.TestCase):
    """This is a class with functions to test FlapDrivenAeroelasticWingMovements."""

    def setUp(self) -> None:
        """Set up a fresh FlapDrivenAeroelasticWingMovement."""
        self.wing_movement = _make_wing_movement()

    def test_is_an_aeroelastic_wing_movement(self) -> None:
        """Test that the subclass relationship holds."""
        self.assertIsInstance(
            self.wing_movement,
            ps.movements.aeroelastic_wing_movement.AeroelasticWingMovement,
        )

    def test_slots(self) -> None:
        """Test that instances have no dict and reject new attributes."""
        self.assertFalse(hasattr(self.wing_movement, "__dict__"))
        with self.assertRaises(AttributeError):
            setattr(self.wing_movement, "new_attribute", 1.0)

    def test_starts_with_no_flap_state(self) -> None:
        """Test that a new movement has recorded nothing."""
        self.assertEqual(self.wing_movement.listOverrideAnglesX_Gs_to_Wn_ixyz, ())
        self.assertIsNone(self.wing_movement.angleSecondDerivativeXRad_function)

    def test_from_aeroelastic_wing_movement_drops_only_the_flap_motion(self) -> None:
        """Test that the helper discards the flap motion and copies every other
        parameter."""
        original = ps.movements.aeroelastic_wing_movement.AeroelasticWingMovement(
            base_wing=self.wing_movement.base_wing,
            wing_cross_section_movements=_cross_section_movements(self.wing_movement),
            ampAngles_Gs_to_Wn_ixyz=(15.0, 5.0, 0.0),
            periodAngles_Gs_to_Wn_ixyz=(1.0, 0.5, 0.0),
            phaseAngles_Gs_to_Wn_ixyz=(1.0, 2.0, 0.0),
            rotationPointOffset_Gs_Ler=(0.0, 0.3, 0.0),
        )
        copied = ps.movements.flap_driven_aeroelastic_wing_movement.FlapDrivenAeroelasticWingMovement.from_aeroelastic_wing_movement(
            original
        )
        self.assertIs(copied.base_wing, original.base_wing)
        npt.assert_array_equal(copied.ampAngles_Gs_to_Wn_ixyz, [0.0, 5.0, 0.0])
        npt.assert_array_equal(copied.periodAngles_Gs_to_Wn_ixyz, [0.0, 0.5, 0.0])
        npt.assert_array_equal(copied.phaseAngles_Gs_to_Wn_ixyz, [0.0, 2.0, 0.0])
        npt.assert_array_equal(
            copied.rotationPointOffset_Gs_Ler, original.rotationPointOffset_Gs_Ler
        )
        self.assertEqual(copied.all_periods, (0.5,))

    def test_angle_parameters_take_only_y_and_z(self) -> None:
        """Test that the angular motion parameters take only the y and z components,
        that a static x (flap) component is put in front of them, and that anything
        other than two components is rejected."""
        driven_class = (
            ps.movements.flap_driven_aeroelastic_wing_movement.FlapDrivenAeroelasticWingMovement
        )
        arguments: dict[str, Any] = {
            "base_wing": self.wing_movement.base_wing,
            "wing_cross_section_movements": _cross_section_movements(
                self.wing_movement
            ),
        }
        second_derivative_z = lambda time: -np.sin(time)
        pitching = driven_class(
            **arguments,
            ampAnglesYZ_Gs_to_Wn_ixyz=(5.0, 0.0),
            periodAnglesYZ_Gs_to_Wn_ixyz=(0.5, 0.0),
            spacingAnglesYZ_Gs_to_Wn_ixyz=("sine", np.sin),
            phaseAnglesYZ_Gs_to_Wn_ixyz=(2.0, 0.0),
            spacingAnglesSecondDerivativeYZ_Gs_to_Wn_ixyz=(None, second_derivative_z),
        )
        npt.assert_array_equal(pitching.ampAngles_Gs_to_Wn_ixyz, [0.0, 5.0, 0.0])
        npt.assert_array_equal(pitching.periodAngles_Gs_to_Wn_ixyz, [0.0, 0.5, 0.0])
        npt.assert_array_equal(pitching.phaseAngles_Gs_to_Wn_ixyz, [0.0, 2.0, 0.0])
        self.assertEqual(
            tuple(pitching.spacingAngles_Gs_to_Wn_ixyz), ("sine", "sine", np.sin)
        )
        second_derivatives = pitching.spacingAnglesSecondDerivative_Gs_to_Wn_ixyz
        assert second_derivatives is not None
        self.assertEqual(tuple(second_derivatives), (None, None, second_derivative_z))
        self.assertEqual(pitching.all_periods, (0.5,))

        # Three components, as an AeroelasticWingMovement takes, are rejected, so a flap
        # motion can't be given. The untyped handle lets the calls below pass arguments
        # that type checking would reject.
        untyped_class: Any = driven_class
        for name in (
            "ampAnglesYZ_Gs_to_Wn_ixyz",
            "periodAnglesYZ_Gs_to_Wn_ixyz",
            "phaseAnglesYZ_Gs_to_Wn_ixyz",
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                untyped_class(**arguments, **{name: (15.0, 0.0, 0.0)})
        with self.assertRaises(ValueError):
            driven_class(
                **arguments,
                spacingAnglesYZ_Gs_to_Wn_ixyz=(np.sin, "sine", "sine"),
            )
        with self.assertRaises(ValueError):
            driven_class(
                **arguments,
                spacingAnglesSecondDerivativeYZ_Gs_to_Wn_ixyz=(None, None, None),
            )
        for bad in (5.0, "sine"):
            bad_value: Any = bad
            with self.assertRaises(TypeError):
                driven_class(**arguments, ampAnglesYZ_Gs_to_Wn_ixyz=bad_value)

        # The old three component keywords don't exist.
        with self.assertRaises(TypeError):
            untyped_class(**arguments, ampAngles_Gs_to_Wn_ixyz=(15.0, 0.0, 0.0))

    def test_flap_period(self) -> None:
        """Test that the flap period starts unset, joins the movement's periods once
        set, is validated, is cleared by a reset, and is kept by a deep copy."""
        self.assertIsNone(self.wing_movement.flap_period)
        self.assertEqual(self.wing_movement.all_periods, ())
        self.assertEqual(self.wing_movement.max_period, 0.0)
        self.wing_movement.set_flap_period(0.4)
        self.assertEqual(self.wing_movement.flap_period, 0.4)
        self.assertEqual(self.wing_movement.all_periods, (0.4,))
        self.assertEqual(self.wing_movement.max_period, 0.4)
        self.assertEqual(copy.deepcopy(self.wing_movement).flap_period, 0.4)
        for bad in (0.0, -1.0, float("nan"), "1.0"):
            bad_period: Any = bad
            with self.assertRaises(ValueError):
                self.wing_movement.set_flap_period(bad_period)
        self.wing_movement.set_flap_period(None)
        self.assertEqual(self.wing_movement.all_periods, ())
        self.wing_movement.set_flap_period(0.4)
        self.wing_movement.reset_flap_state()
        self.assertIsNone(self.wing_movement.flap_period)

    def test_from_aeroelastic_wing_movement_rejects_other_types(self) -> None:
        """Test that the helper only accepts AeroelasticWingMovements."""
        not_a_movement: Any = "wing"
        with self.assertRaises(TypeError):
            ps.movements.flap_driven_aeroelastic_wing_movement.FlapDrivenAeroelasticWingMovement.from_aeroelastic_wing_movement(
                not_a_movement
            )

    def test_base_flap_angle_is_used_until_a_step_is_recorded(self) -> None:
        """Test that an unrecorded time step uses the base Wing's flap angle."""
        wing = self.wing_movement.generate_wing_at_time_step(3, 0.05)
        self.assertAlmostEqual(
            wing.angles_Gs_to_Wn_ixyz[0],
            self.wing_movement.base_wing.angles_Gs_to_Wn_ixyz[0],
        )

    def test_recorded_flap_is_used(self) -> None:
        """Test that a recorded flap angle replaces the base one."""
        self.wing_movement.record_flap_step(0, 1.0, None)
        self.wing_movement.record_flap_step(1, -4.0, lambda time: 2.0)
        wing = self.wing_movement.generate_wing_at_time_step(1, 0.05)
        self.assertAlmostEqual(wing.angles_Gs_to_Wn_ixyz[0], -4.0)
        self.assertEqual(
            self.wing_movement.listOverrideAnglesX_Gs_to_Wn_ixyz, (1.0, -4.0)
        )
        function = self.wing_movement.angleSecondDerivativeXRad_function
        assert function is not None
        self.assertEqual(function(0.0), 2.0)

    def test_explicit_override_wins_over_the_record(self) -> None:
        """Test that an override passed to the method beats a recorded angle."""
        self.wing_movement.record_flap_step(0, 1.0, None)
        wing = self.wing_movement.generate_wing_at_time_step(
            0, 0.05, overrideAngleX_Gs_to_Wn_ixyz=5.0
        )
        self.assertAlmostEqual(wing.angles_Gs_to_Wn_ixyz[0], 5.0)

    def test_record_flap_step_validation(self) -> None:
        """Test that records must be in order, in range, and well typed."""
        with self.assertRaises(ValueError):
            self.wing_movement.record_flap_step(1, 0.0, None)
        with self.assertRaises(ValueError):
            self.wing_movement.record_flap_step(0, 91.0, None)
        with self.assertRaises(ValueError):
            self.wing_movement.record_flap_step(0, float("nan"), None)
        bad_angle: Any = "ten"
        with self.assertRaises(ValueError):
            self.wing_movement.record_flap_step(0, bad_angle, None)
        bad_function: Any = 3.0
        with self.assertRaises(TypeError):
            self.wing_movement.record_flap_step(0, 0.0, bad_function)
        self.assertEqual(self.wing_movement.listOverrideAnglesX_Gs_to_Wn_ixyz, ())

    def test_reset_flap_state(self) -> None:
        """Test that reset clears the angles and the acceleration."""
        self.wing_movement.record_flap_step(0, 1.0, lambda time: 0.0)
        self.wing_movement.reset_flap_state()
        self.assertEqual(self.wing_movement.listOverrideAnglesX_Gs_to_Wn_ixyz, ())
        self.assertIsNone(self.wing_movement.angleSecondDerivativeXRad_function)

    def test_deepcopy_copies_the_record(self) -> None:
        """Test that a deep copy has its own record."""
        self.wing_movement.record_flap_step(0, 1.0, None)
        copied = copy.deepcopy(self.wing_movement)
        copied.record_flap_step(1, 2.0, None)
        self.assertEqual(self.wing_movement.listOverrideAnglesX_Gs_to_Wn_ixyz, (1.0,))
        self.assertEqual(copied.listOverrideAnglesX_Gs_to_Wn_ixyz, (1.0, 2.0))


class TestFlapDrivenAeroelasticUnsteadyProblem(unittest.TestCase):
    """This is a class with functions to test FlapDrivenAeroelasticUnsteadyProblems."""

    problem: (
        ps.flap_driven_aeroelastic_unsteady_problem.FlapDrivenAeroelasticUnsteadyProblem
    )

    def setUp(self) -> None:
        """Set up a problem that has not been run."""
        self.problem = (
            flap_drive_fixtures.make_flap_driven_aeroelastic_problem_fixture()
        )

    def _wing_movements(self) -> tuple[Any, ...]:
        """Returns the first Airplane's Wing movements."""
        movements: Any = self.problem.movement.airplane_movements[0].wing_movements
        return tuple(movements)

    def _run(
        self,
        problem: ps.flap_driven_aeroelastic_unsteady_problem.FlapDrivenAeroelasticUnsteadyProblem,
    ) -> None:
        """Solves a problem.

        :param problem: The problem to solve.
        """
        solver = ps.aeroelastic_unsteady_ring_vortex_lattice_method.AeroelasticUnsteadyRingVortexLatticeMethodSolver(
            aeroelastic_unsteady_problem=problem
        )
        solver.run(show_progress=False, calculate_streamlines=False)

    # --- Construction ---
    def test_is_an_aeroelastic_unsteady_problem(self) -> None:
        """Test the subclass relationship and the stored drives."""
        self.assertIsInstance(self.problem, ps.problems.AeroelasticUnsteadyProblem)
        self.assertEqual(len(self.problem.flap_drives), 1)

    def test_uses_analytical_spring_solver_by_default(self) -> None:
        """Test that a FlapDrivenAeroelasticUnsteadyProblem solves its torsional springs
        analytically unless told otherwise."""
        self.assertTrue(self.problem.use_analytical_spring_solver)

    def test_analytical_spring_solver_can_be_turned_off(self) -> None:
        """Test that use_analytical_spring_solver=False is passed to the parent, so the
        torsional springs are integrated numerically."""
        problem = ps.flap_driven_aeroelastic_unsteady_problem.FlapDrivenAeroelasticUnsteadyProblem(
            movement=flap_drive_fixtures.make_flap_driven_aeroelastic_movement_fixture(),
            wing_density=0.5,
            spring_constant_rad=20.0,
            damping_constant_rad=1.0,
            flap_drives=[flap_drive_fixtures.make_tracking_flap_drive_fixture()],
            use_analytical_spring_solver=False,
        )
        self.assertFalse(problem.use_analytical_spring_solver)

    def test_spring_forcing_samples_match_flap_acceleration_samples(self) -> None:
        """Test that every flap acceleration sample time is also a sample time of the
        analytical spring solver, so its solve is exact for the FlapDrives."""
        spring_samples = ps.problems._SPRING_FORCING_SAMPLES
        flap_samples = (
            ps.flap_driven_aeroelastic_unsteady_problem._FLAP_ACCELERATION_SAMPLES
        )
        self.assertEqual((spring_samples - 1) % (flap_samples - 1), 0)

    def test_analytical_and_numerical_runs_agree(self) -> None:
        """Test that solving the same flap driven run with the analytical and the
        numerical spring solvers gives the same Wing deformations, to within the
        numerical solver's tolerance."""
        histories = []
        for use_analytical in (True, False):
            problem = ps.flap_driven_aeroelastic_unsteady_problem.FlapDrivenAeroelasticUnsteadyProblem(
                movement=flap_drive_fixtures.make_flap_driven_aeroelastic_movement_fixture(
                    8
                ),
                wing_density=0.5,
                spring_constant_rad=20.0,
                damping_constant_rad=1.0,
                flap_drives=[flap_drive_fixtures.make_tracking_flap_drive_fixture()],
                step_discards=1,
                use_analytical_spring_solver=use_analytical,
            )
            self._run(problem)
            histories.append(
                np.array(problem.listDeformationAnglesYRad_Wcsp_to_Wcs_ixyz[0])
            )
        self.assertGreater(np.max(np.abs(histories[1])), 0.0)
        npt.assert_allclose(
            histories[0],
            histories[1],
            rtol=1e-5,
            atol=1e-5 * np.max(np.abs(histories[1])),
        )

    def test_slots(self) -> None:
        """Test that instances have no dict and reject new attributes."""
        self.assertFalse(hasattr(self.problem, "__dict__"))
        with self.assertRaises(AttributeError):
            setattr(self.problem, "new_attribute", 1.0)

    def test_properties_are_read_only(self) -> None:
        """Test that the configuration can't be reassigned."""
        with self.assertRaises(AttributeError):
            setattr(self.problem, "flap_drives", None)

    def test_history_is_seeded_with_the_initial_state(self) -> None:
        """Test that each history list starts with the first time step's entry."""
        self.assertEqual(len(self.problem.flap_actuator_positions), 1)
        self.assertEqual(len(self.problem.flap_actuator_positions[0]), 1)
        self.assertAlmostEqual(
            self.problem.flap_angles_rad[0][0],
            np.deg2rad(flap_drive_fixtures.sine_flap_angle_deg(0.0)),
        )
        self.assertEqual(self.problem.flap_fluid_torques[0], [0.0])

    def test_initial_wings_use_the_drive_start_angle(self) -> None:
        """Test that the Wings start at the angle the drive starts at."""
        for wing_movement in self._wing_movements():
            self.assertEqual(len(wing_movement.listOverrideAnglesX_Gs_to_Wn_ixyz), 1)
        wing = self.problem.steady_problems[0].airplanes[0].wings[0]
        self.assertAlmostEqual(
            wing.angles_Gs_to_Wn_ixyz[0],
            flap_drive_fixtures.sine_flap_angle_deg(0.0),
            places=6,
        )

    def test_construction_validates_arguments(self) -> None:
        """Test that bad flap_drives and movements are rejected."""
        movement = flap_drive_fixtures.make_flap_driven_aeroelastic_movement_fixture()
        drive = flap_drive_fixtures.make_tracking_flap_drive_fixture()

        def make(**overrides: Any) -> Any:
            arguments: dict[str, Any] = {
                "movement": movement,
                "wing_density": 0.5,
                "spring_constant_rad": 20.0,
                "damping_constant_rad": 1.0,
                "flap_drives": [drive],
            }
            arguments.update(overrides)
            return ps.flap_driven_aeroelastic_unsteady_problem.FlapDrivenAeroelasticUnsteadyProblem(
                **arguments
            )

        with self.assertRaises(ValueError):
            make(flap_drives=[])
        for bad in ("drive", [1.0], 3):
            with self.assertRaises(TypeError):
                make(flap_drives=bad)
        not_a_movement: Any = "movement"
        with self.assertRaises(TypeError):
            make(movement=not_a_movement)

    def test_wing_index_out_of_range_is_rejected(self) -> None:
        """Test that a drive can't name a Wing the Airplane doesn't have."""
        drive = flap_drive.FlapDrive(
            wing_indices=(0, 5),
            transmission=flap_drive.FlapTransmission.identity(),
            actuator=lambda inputs: 0.0,
            actuator_inertia=1e-3,
            reference=flap_drive_fixtures.make_sine_flap_reference_fixture(),
            use_position_lookup=False,
        )
        with self.assertRaises(ValueError):
            ps.flap_driven_aeroelastic_unsteady_problem.FlapDrivenAeroelasticUnsteadyProblem(
                movement=flap_drive_fixtures.make_flap_driven_aeroelastic_movement_fixture(),
                wing_density=0.5,
                spring_constant_rad=20.0,
                damping_constant_rad=1.0,
                flap_drives=[drive],
            )

    def test_wing_in_two_drives_is_rejected(self) -> None:
        """Test that a Wing can belong to only one FlapDrive."""
        with self.assertRaises(ValueError):
            ps.flap_driven_aeroelastic_unsteady_problem.FlapDrivenAeroelasticUnsteadyProblem(
                movement=flap_drive_fixtures.make_flap_driven_aeroelastic_movement_fixture(),
                wing_density=0.5,
                spring_constant_rad=20.0,
                damping_constant_rad=1.0,
                flap_drives=[
                    flap_drive_fixtures.make_tracking_flap_drive_fixture(),
                    flap_drive_fixtures.make_tracking_flap_drive_fixture(),
                ],
            )

    def test_wing_without_flap_driven_movement_is_rejected(self) -> None:
        """Test that a driven Wing needs a FlapDrivenAeroelasticWingMovement."""
        plain = ps.movements.aeroelastic_wing_movement
        flap_driven = (
            flap_drive_fixtures.make_flap_driven_aeroelastic_movement_fixture()
        )
        original = flap_driven.airplane_movements[0].wing_movements
        plain_movements: list[
            ps.movements.aeroelastic_wing_movement.AeroelasticWingMovement
            | ps.movements.wing_movement.WingMovement
        ] = [
            plain.AeroelasticWingMovement(
                base_wing=wing_movement.base_wing,
                wing_cross_section_movements=_cross_section_movements(wing_movement),
                ampAngles_Gs_to_Wn_ixyz=(15.0, 0.0, 0.0),
                periodAngles_Gs_to_Wn_ixyz=(1.0, 0.0, 0.0),
                phaseAngles_Gs_to_Wn_ixyz=(1.0, 0.0, 0.0),
            )
            for wing_movement in original
        ]
        airplane_movement = (
            ps.movements.aeroelastic_airplane_movement.AeroelasticAirplaneMovement(
                base_airplane=flap_driven.airplane_movements[0].base_airplane,
                wing_movements=plain_movements,
            )
        )
        movement = ps.movements.aeroelastic_movement.AeroelasticMovement(
            airplane_movements=[airplane_movement],
            operating_point_movement=flap_driven.operating_point_movement,
            delta_time=0.05,
            num_steps=6,
        )
        with self.assertRaises(TypeError):
            ps.flap_driven_aeroelastic_unsteady_problem.FlapDrivenAeroelasticUnsteadyProblem(
                movement=movement,
                wing_density=0.5,
                spring_constant_rad=20.0,
                damping_constant_rad=1.0,
                flap_drives=[flap_drive_fixtures.make_tracking_flap_drive_fixture()],
            )

    def test_initial_actuator_position_sets_the_start_angle(self) -> None:
        """Test that a drive's own initial actuator position, not its reference's start,
        sets the first time step's flap angle."""
        drive = flap_drive.FlapDrive(
            wing_indices=(0, 1),
            transmission=flap_drive.FlapTransmission.identity(),
            actuator=lambda inputs: 0.0,
            actuator_inertia=1e-3,
            initial_actuator_position=0.5,
            reference=flap_drive_fixtures.make_sine_flap_reference_fixture(),
            use_position_lookup=False,
        )
        problem = ps.flap_driven_aeroelastic_unsteady_problem.FlapDrivenAeroelasticUnsteadyProblem(
            movement=flap_drive_fixtures.make_flap_driven_aeroelastic_movement_fixture(),
            wing_density=0.5,
            spring_constant_rad=20.0,
            damping_constant_rad=1.0,
            flap_drives=[drive],
        )
        self.assertEqual(problem.flap_actuator_positions[0][0], 0.5)
        wing = problem.steady_problems[0].airplanes[0].wings[0]
        self.assertAlmostEqual(wing.angles_Gs_to_Wn_ixyz[0], np.rad2deg(0.5))

    # --- Period and cycle averaging ---
    def test_reference_period_sets_the_movement_periods(self) -> None:
        """Test that the reference's period reaches the Wings' movements, the movement's
        LCM period, its static flag, and the first averaging step."""
        movement = self.problem.movement
        for wing_movement in self._wing_movements():
            self.assertEqual(
                wing_movement.flap_period, flap_drive_fixtures.SINE_FLAP_PERIOD
            )
        self.assertEqual(movement.lcm_period, flap_drive_fixtures.SINE_FLAP_PERIOD)
        self.assertEqual(movement.max_period, flap_drive_fixtures.SINE_FLAP_PERIOD)
        self.assertEqual(movement.min_period, flap_drive_fixtures.SINE_FLAP_PERIOD)
        self.assertFalse(movement.static)
        self.assertEqual(
            self.problem.first_averaging_step,
            max(
                0,
                int(
                    np.floor(
                        movement.num_steps
                        - flap_drive_fixtures.SINE_FLAP_PERIOD / movement.delta_time
                    )
                ),
            ),
        )

    def test_periods_read_before_the_problem_are_refreshed(self) -> None:
        """Test that a movement whose periods were read, and cached, before the problem
        existed reports the reference's period afterwards."""
        movement = flap_drive_fixtures.make_flap_driven_aeroelastic_movement_fixture(
            num_steps=30
        )
        self.assertEqual(movement.lcm_period, 0.0)
        self.assertTrue(movement.static)
        self.assertEqual(movement.airplane_movements[0].all_periods, ())
        reference = flap_drive_fixtures.make_sine_flap_reference_fixture(period=0.5)
        problem = ps.flap_driven_aeroelastic_unsteady_problem.FlapDrivenAeroelasticUnsteadyProblem(
            movement=movement,
            wing_density=0.5,
            spring_constant_rad=20.0,
            damping_constant_rad=1.0,
            flap_drives=[
                flap_drive_fixtures.make_tracking_flap_drive_fixture(None, reference)
            ],
        )
        self.assertEqual(movement.lcm_period, 0.5)
        self.assertFalse(movement.static)
        self.assertEqual(movement.airplane_movements[0].all_periods, (0.5, 0.5))
        self.assertEqual(problem.first_averaging_step, 20)

    def test_reference_without_a_period_warns(self) -> None:
        """Test that a reference with no period logs the inaccurate cycle average
        warning, and leaves a movement with no other motion static."""
        reference = flap_drive_fixtures.make_sine_flap_reference_fixture(period=None)
        with self.assertLogs(
            "pterasoftware.flap_driven_aeroelastic_unsteady_problem", level="WARNING"
        ) as logs:
            problem = ps.flap_driven_aeroelastic_unsteady_problem.FlapDrivenAeroelasticUnsteadyProblem(
                movement=flap_drive_fixtures.make_flap_driven_aeroelastic_movement_fixture(),
                wing_density=0.5,
                spring_constant_rad=20.0,
                damping_constant_rad=1.0,
                flap_drives=[
                    flap_drive_fixtures.make_tracking_flap_drive_fixture(
                        None, reference
                    )
                ],
            )
        self.assertEqual(len(logs.records), 1)
        self.assertIn("no period", logs.output[0])
        self.assertIn("treated as static", logs.output[0])
        self.assertTrue(problem.movement.static)
        self.assertEqual(problem.first_averaging_step, problem.num_steps - 1)

    def test_period_cache_names_exist(self) -> None:
        """Test that the private period caches that the problem clears still exist on
        the core movement classes, so a rename in them fails here instead of leaving
        stale periods."""
        movement = self.problem.movement
        for name in ("_lcm_period", "_max_period", "_min_period", "_static"):
            self.assertTrue(hasattr(movement, name), name)
        for name in ("_all_periods", "_max_period"):
            self.assertTrue(hasattr(movement.airplane_movements[0], name), name)

    # --- Behavior over a run ---
    def test_run_records_every_step(self) -> None:
        """Test that a solved problem has one history entry per time step."""
        self._run(self.problem)
        for series in (
            self.problem.flap_actuator_positions[0],
            self.problem.flap_actuator_velocities[0],
            self.problem.flap_actuator_forces[0],
            self.problem.flap_fluid_torques[0],
            self.problem.flap_transmission_torques[0],
            self.problem.flap_angles_rad[0],
            self.problem.flap_angle_derivatives_rad[0],
        ):
            self.assertEqual(len(series), self.problem.num_steps)
            self.assertTrue(np.isfinite(series).all())
        for wing_movement in self._wing_movements():
            self.assertEqual(
                len(wing_movement.listOverrideAnglesX_Gs_to_Wn_ixyz),
                self.problem.num_steps,
            )

    def test_wings_use_the_driven_flap_angle(self) -> None:
        """Test that each solved time step's Wings have the angle the drive reached."""
        self._run(self.problem)
        for step, steady_problem in enumerate(self.problem.steady_problems):
            expected = np.rad2deg(self.problem.flap_angles_rad[0][step])
            for wing in steady_problem.airplanes[0].wings:
                self.assertAlmostEqual(wing.angles_Gs_to_Wn_ixyz[0], expected, places=6)

    def test_unlimited_drive_tracks_the_reference(self) -> None:
        """Test that a drive with feedforward and feedback follows the schedule."""
        self._run(self.problem)
        times = np.arange(self.problem.num_steps) * self.problem.delta_time
        desired = np.array([flap_drive_fixtures.sine_flap_angle_deg(t) for t in times])
        actual = np.rad2deg(self.problem.flap_angles_rad[0])
        self.assertLess(float(np.max(np.abs(actual - desired))), 1.0)

    def test_torque_limited_drive_lags_the_reference(self) -> None:
        """Test that a drive that can't supply enough torque falls behind, which is the
        reason for the whole model."""
        limited = flap_drive_fixtures.make_flap_driven_aeroelastic_problem_fixture(
            torque_limit=0.0005
        )
        self._run(self.problem)
        self._run(limited)
        times = np.arange(self.problem.num_steps) * self.problem.delta_time
        desired = np.array([flap_drive_fixtures.sine_flap_angle_deg(t) for t in times])
        free_error = np.max(
            np.abs(np.rad2deg(self.problem.flap_angles_rad[0]) - desired)
        )
        limited_error = np.max(np.abs(np.rad2deg(limited.flap_angles_rad[0]) - desired))
        self.assertGreater(float(limited_error), 3.0 * float(free_error))
        self.assertLessEqual(
            float(np.max(np.abs(limited.flap_actuator_forces[0]))), 0.0005 + 1e-12
        )

    def test_mirrored_wings_have_the_same_fluid_torque_sign(self) -> None:
        """Test that the main and reflected Wings contribute torque of one sign, so a
        shared drive adds them instead of cancelling them."""
        solver = ps.aeroelastic_unsteady_ring_vortex_lattice_method.AeroelasticUnsteadyRingVortexLatticeMethodSolver(
            aeroelastic_unsteady_problem=self.problem
        )
        solver.run(show_progress=False, calculate_streamlines=False)
        main = self.problem._calculate_flap_fluid_torque(solver, 0)
        reflected = self.problem._calculate_flap_fluid_torque(solver, 1)
        self.assertNotEqual(main, 0.0)
        self.assertAlmostEqual(reflected / main, 1.0, places=4)

    def test_array_path_fluid_torque_matches_the_object_path(self) -> None:
        """Test that, at every solved time step of an array path run, the array path's
        fluid torque (from the solver's kept loads and the recorded step parameters) is
        exactly the object path's on the same time step (from the Wing's Panels and its
        transformation), for the main and the mirror only Wing."""
        problem_class = (
            ps.flap_driven_aeroelastic_unsteady_problem.FlapDrivenAeroelasticUnsteadyProblem
        )
        original = problem_class._calculate_flap_fluid_torque
        torques: list[tuple[int, float, float]] = []

        def both_paths(
            problem: Any,
            solver: Any,
            wing_index: int,
        ) -> float:
            array_torque = original(problem, solver, wing_index)
            # The array path builds no Airplanes during the run, so the object branch
            # reads one built from the same time step's step parameters, with the same
            # loads written onto it, as the object path's Airplane carries them.
            loaded_airplane = problem._build_loaded_airplane(
                solver._current_step, solver._current_step_loads
            )
            current_airplanes = solver.current_airplanes
            solver.current_airplanes = (loaded_airplane,)
            solver._use_arrays = False
            try:
                object_torque = original(problem, solver, wing_index)
            finally:
                solver._use_arrays = True
                solver.current_airplanes = current_airplanes
            torques.append((wing_index, array_torque, object_torque))
            return array_torque

        # Discard only the first time step, so the torque is found at most time steps.
        # The rotation point is off the leading edge root point, so the torque depends
        # on the Wing's transformation at the time step, which moves it.
        problem = problem_class(
            movement=flap_drive_fixtures.make_flap_driven_aeroelastic_movement_fixture(
                8, rotationPointOffset_Gs_Ler=(0.1, 0.05, 0.02)
            ),
            wing_density=0.5,
            spring_constant_rad=20.0,
            damping_constant_rad=1.0,
            flap_drives=[flap_drive_fixtures.make_tracking_flap_drive_fixture()],
            step_discards=1,
        )
        solver = ps.aeroelastic_unsteady_ring_vortex_lattice_method.AeroelasticUnsteadyRingVortexLatticeMethodSolver(
            aeroelastic_unsteady_problem=problem
        )
        with mock.patch.object(
            problem_class, "_calculate_flap_fluid_torque", both_paths
        ):
            solver.run(
                show_progress=False, calculate_streamlines=False, use_arrays=True
            )
        self.assertTrue(solver.use_arrays)
        self.assertEqual({wing_index for wing_index, _, _ in torques}, {0, 1})
        self.assertTrue(any(array_torque != 0.0 for _, array_torque, _ in torques))
        for wing_index, array_torque, object_torque in torques:
            self.assertEqual(array_torque, object_torque, msg=f"Wing {wing_index}")

    def test_inertial_moment_uses_the_actual_acceleration(self) -> None:
        """Test that a recorded acceleration is the flap acceleration in the torsional
        spring's forcing, and that without one the forcing is zero, because the movement
        prescribes no flap motion."""
        wing_movement = _make_wing_movement()
        cls = (
            ps.flap_driven_aeroelastic_unsteady_problem.FlapDrivenAeroelasticUnsteadyProblem
        )
        unrecorded = cls._generate_inertial_moment_function(0.2, wing_movement)
        self.assertEqual(unrecorded(0.1), 0.0)
        wing_movement.record_flap_step(0, 0.0, lambda time: 3.0)
        actual = cls._generate_inertial_moment_function(0.2, wing_movement)
        self.assertAlmostEqual(actual(0.1), 0.2 * 3.0)

    @staticmethod
    def _record_saturated_acceleration(wing_movement: Any, step: int) -> None:
        """Records a flap acceleration for a step, sampled as _advance_flap_drives
        samples it, with a clipped stretch in the middle like a saturated motor's.

        :param wing_movement: The FlapDrivenAeroelasticWingMovement to record on.
        :param step: The time step whose acceleration to record.
        """
        dt = 0.0025
        times = np.linspace(dt * (step - 1), dt * step, 100)
        accelerations = np.clip(
            4000.0 * np.sin(2.0 * np.pi * 10.0 * times + 0.3), -2500.0, 2500.0
        )
        wing_movement.record_flap_step(
            step,
            0.0,
            ps.flap_driven_aeroelastic_unsteady_problem._make_interpolated_function(
                times, accelerations
            ),
        )

    def test_batched_spring_solve_matches_per_strip_with_recorded_acceleration(
        self,
    ) -> None:
        """Test that the batched analytical spring solve matches solving each strip on
        its own, byte for byte, when a FlapDrive's recorded acceleration (including a
        saturated stretch) drives the inertial forcing."""
        self.assertTrue(self.problem.use_analytical_spring_solver)
        wing_movement = _make_wing_movement()
        rng = np.random.default_rng(0)
        num_strips = 6
        dt = 0.0025
        torsional_inertias = 10 ** rng.uniform(-9.0, -4.0, num_strips)
        flapping_axis_inertias = 10 ** rng.uniform(-8.0, -3.0, num_strips)
        aeroMoments_GP1_Slep = rng.normal(0.0, 1e-3, (5, num_strips, 3))
        theta = rng.normal(0.0, 0.05, num_strips)
        theta_derivative = rng.normal(0.0, 1.0, num_strips)
        # Steps are recorded in order from 0, which has no preceding interval.
        wing_movement.record_flap_step(0, 0.0, None)
        for step in range(1, 4):
            self._record_saturated_acceleration(wing_movement, step)
            batched = self.problem._solve_strips_analytically(
                dt=dt,
                step=step,
                torsional_inertias=torsional_inertias,
                flapping_axis_inertias=flapping_axis_inertias,
                lastDeformationAnglesYRad_Wcsp_to_Wcs_ixyz=theta,
                lastDeformationAnglesDerivativeYRad_Wcsp_to_Wcs_ixyz=theta_derivative,
                aeroMoments_GP1_Slep=aeroMoments_GP1_Slep,
                wing_movement=wing_movement,
            )
            reference = [
                self.problem._calculate_torsional_spring_moment(
                    dt,
                    torsional_inertia=torsional_inertias[strip],
                    lastDeformationAngleYRad_Wcsp_to_Wcs_ixyz=theta[strip],
                    lastDeformationAngleDerivativeYRad_Wcsp_to_Wcs_ixyz=(
                        theta_derivative[strip]
                    ),
                    aeroMomentY_GP1_Slep=np.sum(aeroMoments_GP1_Slep[:, strip, 1]),
                    step=step,
                    flapping_axis_inertia=flapping_axis_inertias[strip],
                    wing_movement=wing_movement,
                )
                for strip in range(num_strips)
            ]
            theta = np.array([state[0] for state in reference], dtype=float)
            theta_derivative = np.array([state[1] for state in reference], dtype=float)
            with self.subTest(step=step):
                # Comparing the raw bytes also tells 0.0 from -0.0.
                self.assertEqual(batched[0].tobytes(), theta.tobytes())
                self.assertEqual(batched[1].tobytes(), theta_derivative.tobytes())

    def test_recorded_acceleration_inertial_moment_contract(self) -> None:
        """Test that the FlapDrive override applies the flapping-axis inertia as a
        single multiplication, so I * f_1(t) equals f_I(t) byte for byte, which the
        batched analytical spring solve relies on."""
        wing_movement = _make_wing_movement()
        # Steps are recorded in order from 0, which has no preceding interval.
        wing_movement.record_flap_step(0, 0.0, None)
        self._record_saturated_acceleration(wing_movement, 1)
        cls = (
            ps.flap_driven_aeroelastic_unsteady_problem.FlapDrivenAeroelasticUnsteadyProblem
        )
        sample_times = np.linspace(0.0, 0.0025, 100)
        unit = cls._generate_inertial_moment_function(1.0, wing_movement)
        for inertia in 10 ** np.random.default_rng(1).uniform(-8.0, 0.0, 20):
            scaled = cls._generate_inertial_moment_function(
                float(inertia), wing_movement
            )
            with self.subTest(inertia=inertia):
                expected = np.array(
                    [float(scaled(float(time))) for time in sample_times]
                )
                actual = inertia * np.array(
                    [float(unit(float(time))) for time in sample_times]
                )
                self.assertEqual(actual.tobytes(), expected.tobytes())

    def test_prescribed_wings_keep_the_parent_inertial_moment(self) -> None:
        """Test that a Wing with no recorded acceleration behaves as in the parent."""
        wing_movement = _make_wing_movement()
        cls = (
            ps.flap_driven_aeroelastic_unsteady_problem.FlapDrivenAeroelasticUnsteadyProblem
        )
        self.assertEqual(
            cls._generate_inertial_moment_function(0.2, wing_movement)(0.1),
            ps.problems.AeroelasticUnsteadyProblem._generate_inertial_moment_function(
                0.2, wing_movement
            )(0.1),
        )

    def test_a_new_problem_clears_earlier_flap_state(self) -> None:
        """Test that building a problem from a used movement starts the record over."""
        movement = flap_drive_fixtures.make_flap_driven_aeroelastic_movement_fixture()
        wing_movement = movement.airplane_movements[0].wing_movements[0]
        assert isinstance(
            wing_movement,
            ps.movements.flap_driven_aeroelastic_wing_movement.FlapDrivenAeroelasticWingMovement,
        )
        wing_movement.record_flap_step(0, 3.0, None)
        wing_movement.record_flap_step(1, 4.0, None)
        problem = ps.flap_driven_aeroelastic_unsteady_problem.FlapDrivenAeroelasticUnsteadyProblem(
            movement=movement,
            wing_density=0.5,
            spring_constant_rad=20.0,
            damping_constant_rad=1.0,
            flap_drives=[flap_drive_fixtures.make_tracking_flap_drive_fixture()],
        )
        self.assertEqual(len(wing_movement.listOverrideAnglesX_Gs_to_Wn_ixyz), 1)
        self.assertEqual(len(problem.flap_angles_rad[0]), 1)

    def test_save_and_load_round_trip(self) -> None:
        """Test that a solved problem saves and loads with its flap history intact."""
        import tempfile
        from pathlib import Path

        self._run(self.problem)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flap_driven.psz"
            ps.save(path, self.problem)
            loaded: Any = ps.load(path)
        self.assertIsInstance(
            loaded,
            ps.flap_driven_aeroelastic_unsteady_problem.FlapDrivenAeroelasticUnsteadyProblem,
        )
        npt.assert_allclose(loaded.flap_angles_rad[0], self.problem.flap_angles_rad[0])
        npt.assert_allclose(
            loaded.flap_fluid_torques[0], self.problem.flap_fluid_torques[0]
        )


class TestFlapDrivenAeroelasticWingMovementStepParameters(unittest.TestCase):
    """This is a class with functions to test the per time step parameters of
    FlapDrivenAeroelasticWingMovements."""

    def test_parameters_use_the_recorded_flap_angle(self) -> None:
        """Test that the parameters use the flap angle recorded for a time step, that an
        override wins over it, that a time step with no record uses the base Wing's flap
        angle, and that the parameters match the Wing built at the same time step."""
        wing_movement = _make_wing_movement()
        recorded = (10.0, 20.0, 30.0)
        for step, angle in enumerate(recorded):
            wing_movement.record_flap_step(step, angle, None)

        for step, angle in enumerate(recorded):
            with self.subTest(step=step):
                parameters = wing_movement._generate_wing_parameters_at_time_step(
                    step, 0.01
                )
                self.assertEqual(parameters.angles_Gs_to_Wn_ixyz[0], angle)
                wing = wing_movement.generate_wing_at_time_step(step, 0.01)
                self.assertTrue(
                    np.array_equal(
                        parameters.angles_Gs_to_Wn_ixyz, wing.angles_Gs_to_Wn_ixyz
                    )
                )
                self.assertTrue(np.array_equal(parameters.Ler_Gs_Cgs, wing.Ler_Gs_Cgs))

        overridden = wing_movement._generate_wing_parameters_at_time_step(
            1, 0.01, overrideAngleX_Gs_to_Wn_ixyz=-5.0
        )
        self.assertEqual(overridden.angles_Gs_to_Wn_ixyz[0], -5.0)

        unrecorded = wing_movement._generate_wing_parameters_at_time_step(5, 0.01)
        self.assertEqual(
            unrecorded.angles_Gs_to_Wn_ixyz[0],
            wing_movement.base_wing.angles_Gs_to_Wn_ixyz[0],
        )


if __name__ == "__main__":
    unittest.main()
