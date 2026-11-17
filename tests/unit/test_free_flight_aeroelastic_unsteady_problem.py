"""This module contains classes to test the FreeFlightAeroelasticUnsteadyProblem
class."""

import unittest
from typing import Any

import numpy as np

import pterasoftware as ps
from tests.unit.fixtures import movement_fixtures

_INERTIA = ((0.1, 0.0, 0.0), (0.0, 0.2, 0.0), (0.0, 0.0, 0.3))


def _make_movement(
    g_E: tuple[float, float, float] = (0.0, 0.0, 0.0),
    externalFX_W: float = 0.0,
) -> ps.movements.free_flight_movement.FreeFlightAeroelasticMovement:
    """Makes a FreeFlightAeroelasticMovement for the basic aeroelastic movement
    fixture's Airplane, whose weight is zero.

    :param g_E: The initial OperatingPoint's gravitational acceleration (m/s^2). The
        default is (0.0, 0.0, 0.0).
    :param externalFX_W: The initial OperatingPoint's externalFX_W (N). The default is
        0.0.
    :return: The FreeFlightAeroelasticMovement, with 2 prescribed and 2 free time steps.
    """
    aeroelastic_movement = movement_fixtures.make_basic_aeroelastic_movement_fixture()
    airplane_movement = aeroelastic_movement.airplane_movements[0]
    assert isinstance(
        airplane_movement,
        ps.movements.aeroelastic_airplane_movement.AeroelasticAirplaneMovement,
    )
    return ps.movements.free_flight_movement.FreeFlightAeroelasticMovement(
        airplane_movements=[airplane_movement],
        operating_point_movement=ps.movements.free_flight_operating_point_movement.FreeFlightOperatingPointMovement(
            base_operating_point=ps.operating_point.OperatingPoint(
                rho=1.225, vCg__E=10.0, alpha=0.0, externalFX_W=externalFX_W, g_E=g_E
            )
        ),
        delta_time=0.1,
        prescribed_num_steps=2,
        free_num_steps=2,
    )


def _make_problem(**overrides: Any) -> ps.problems.FreeFlightAeroelasticUnsteadyProblem:
    """Makes a FreeFlightAeroelasticUnsteadyProblem, with any argument overridden.

    :param overrides: The arguments to override.
    :return: The FreeFlightAeroelasticUnsteadyProblem.
    """
    arguments: dict[str, Any] = {
        "movement": _make_movement(),
        "wing_density": 0.01,
        "spring_constant_rad": 10.0,
        "damping_constant_rad": 0.5,
        "mass": 1.0,
        "I_BP1_CgP1": _INERTIA,
    }
    arguments.update(overrides)
    return ps.problems.FreeFlightAeroelasticUnsteadyProblem(**arguments)


class TestFreeFlightAeroelasticUnsteadyProblem(unittest.TestCase):
    """This class contains unit tests for the FreeFlightAeroelasticUnsteadyProblem
    class."""

    def setUp(self) -> None:
        """Set up a FreeFlightAeroelasticUnsteadyProblem."""
        self.problem = _make_problem()

    def test_is_an_aeroelastic_unsteady_problem(self) -> None:
        """Test that the class is an AeroelasticUnsteadyProblem, so it inherits the
        torsional spring model."""
        self.assertIsInstance(self.problem, ps.problems.AeroelasticUnsteadyProblem)

    def test_num_steps_spans_both_phases(self) -> None:
        """Test that num_steps covers the prescribed and the free time steps."""
        self.assertEqual(self.problem.num_steps, 4)

    def test_structural_parameters_are_passed_to_the_parent(self) -> None:
        """Test that the torsional spring parameters and step_discards reach
        AeroelasticUnsteadyProblem."""
        problem = _make_problem(step_discards=3)
        self.assertEqual(problem.wing_density, 0.01)
        self.assertEqual(problem.spring_constant_rad, 10.0)
        self.assertEqual(problem.damping_constant_rad, 0.5)
        self.assertEqual(problem.step_discards, 3)

    def test_uses_numerical_spring_solver_by_default(self) -> None:
        """Test that use_analytical_spring_solver defaults to False and can be set."""
        self.assertFalse(self.problem.use_analytical_spring_solver)
        self.assertTrue(
            _make_problem(
                use_analytical_spring_solver=True
            ).use_analytical_spring_solver
        )

    def test_mass_and_inertia_are_stored(self) -> None:
        """Test that the mass and the inertia matrix are stored as floats."""
        self.assertEqual(self.problem.mass, 1.0)
        self.assertIsInstance(self.problem.mass, float)
        np.testing.assert_array_equal(self.problem.I_BP1_CgP1, np.array(_INERTIA))
        self.assertEqual(self.problem.I_BP1_CgP1.dtype, np.dtype(float))

    def test_inertia_is_read_only(self) -> None:
        """Test that the stored inertia matrix can't be changed."""
        with self.assertRaises(ValueError):
            self.problem.I_BP1_CgP1[0, 0] = 5.0
        with self.assertRaises(AttributeError):
            setattr(self.problem, "I_BP1_CgP1", np.eye(3, dtype=float))

    def test_k_max_default_and_override(self) -> None:
        """Test that k_max defaults to 20 and can be set."""
        self.assertEqual(self.problem.k_max, 20)
        self.assertEqual(_make_problem(k_max=7).k_max, 7)

    def test_slots(self) -> None:
        """Test that instances have no dict and reject new attributes."""
        self.assertFalse(hasattr(self.problem, "__dict__"))
        with self.assertRaises(AttributeError):
            setattr(self.problem, "new_attribute", 1.0)

    def test_rejects_non_free_flight_movement(self) -> None:
        """Test that a movement that isn't a FreeFlightAeroelasticMovement raises a
        TypeError."""
        with self.assertRaises(TypeError):
            _make_problem(
                movement=movement_fixtures.make_basic_aeroelastic_movement_fixture()
            )

    def test_rejects_non_positive_mass(self) -> None:
        """Test that a zero or negative mass raises a ValueError."""
        for mass in (0.0, -1.0):
            with self.subTest(mass=mass):
                with self.assertRaises(ValueError):
                    _make_problem(mass=mass)

    def test_rejects_asymmetric_inertia(self) -> None:
        """Test that an inertia matrix that isn't symmetric raises a ValueError."""
        with self.assertRaises(ValueError):
            _make_problem(
                I_BP1_CgP1=((0.1, 0.01, 0.0), (0.0, 0.2, 0.0), (0.0, 0.0, 0.3))
            )

    def test_rejects_weight_inconsistent_with_mass_and_gravity(self) -> None:
        """Test that a mass that doesn't match the Airplane's weight under the
        OperatingPoint's gravity raises a ValueError.

        The fixture Airplane weighs nothing, so any mass under gravity disagrees with
        it.
        """
        with self.assertRaises(ValueError):
            _make_problem(movement=_make_movement(g_E=(0.0, 0.0, 9.80665)))

    def test_rejects_nonzero_external_force(self) -> None:
        """Test that a nonzero externalFX_W raises a ValueError, because free flight
        never applies it."""
        with self.assertRaises(ValueError):
            _make_problem(movement=_make_movement(externalFX_W=5.0))

    def test_rejects_non_callable_external_loads(self) -> None:
        """Test that an external_loads_fn that isn't callable raises a TypeError."""
        with self.assertRaises(TypeError):
            _make_problem(external_loads_fn=3.0)

    def test_rejects_unknown_integrator(self) -> None:
        """Test that an integrator that MuJoCo doesn't have raises a ValueError."""
        with self.assertRaises(ValueError):
            _make_problem(integrator="Leapfrog")
