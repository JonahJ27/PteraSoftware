"""This module contains classes to test the FreeFlightAeroelasticUnsteadyProblem
class."""

import unittest
from typing import Any, cast
from unittest import mock

import numpy as np

import pterasoftware as ps
from tests.unit.fixtures import coupled_unsteady_problem_fixtures, movement_fixtures

_INERTIA = ((0.1, 0.0, 0.0), (0.0, 0.2, 0.0), (0.0, 0.0, 0.3))


def _make_movement(
    g_E: tuple[float, float, float] = (0.0, 0.0, 0.0),
    externalFX_W: float = 0.0,
    CgP1_E_Eo: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> ps.movements.free_flight_movement.FreeFlightAeroelasticMovement:
    """Makes a FreeFlightAeroelasticMovement for the basic aeroelastic movement
    fixture's Airplane, whose weight is zero.

    :param g_E: The initial OperatingPoint's gravitational acceleration (m/s^2). The
        default is (0.0, 0.0, 0.0).
    :param externalFX_W: The initial OperatingPoint's externalFX_W (N). The default is
        0.0.
    :param CgP1_E_Eo: The initial OperatingPoint's CgP1_E_Eo (m). The default is (0.0,
        0.0, 0.0).
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
                rho=1.225,
                vCg__E=10.0,
                alpha=0.0,
                externalFX_W=externalFX_W,
                g_E=g_E,
                CgP1_E_Eo=CgP1_E_Eo,
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

    def test_approximations_default_override_and_validation(self) -> None:
        """Test that approximations defaults to an exact Approximations, stores a given
        one, and rejects anything else."""
        self.assertFalse(self.problem.approximations.any_enabled)
        fast = ps.approximations.Approximations.fast()
        self.assertIs(_make_problem(approximations=fast).approximations, fast)
        with self.assertRaises(TypeError):
            _make_problem(approximations=True)

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


class TestFreeFlightAeroelasticUnsteadyProblemArrayPath(unittest.TestCase):
    """This is a class with functions to test that a
    FreeFlightAeroelasticUnsteadyProblem's interval loads on the array path, read from
    the solver's kept loads, are exactly those read from the Airplane."""

    def test_interval_loads_from_step_loads_match_the_airplane(self) -> None:
        """Test that, at every call during an array path run (the snapshot, each
        coupling trial, and the prescribed phase), _assemble_interval_loads given the
        solver's StepLoads returns exactly what it returns reading the same Airplane,
        which the array path also writes the loads onto."""
        problem_class = ps.problems.FreeFlightAeroelasticUnsteadyProblem
        original = problem_class._assemble_interval_loads
        results: list[tuple[Any, Any]] = []

        def both_paths(
            problem: Any,
            operating_point: Any,
            airplane: Any,
            step: int,
            loads: Any = None,
            airplane_step: Any = None,
        ) -> Any:
            self.assertIsNotNone(loads)
            from_loads = original(
                problem,
                operating_point,
                airplane,
                step,
                loads=loads,
                airplane_step=airplane_step,
            )
            # The array path passes no Airplane, so the object branch reads one built
            # from the same time step's step parameters with the same loads, as the
            # object path's Airplane carries them.
            loaded_airplane = problem._build_loaded_airplane(airplane_step, loads)
            from_airplane = original(problem, operating_point, loaded_airplane, step)
            results.append((from_loads, from_airplane))
            return from_loads

        problem = _make_problem()
        solver = ps.free_flight_aeroelastic_unsteady_ring_vortex_lattice_method.FreeFlightAeroelasticUnsteadyRingVortexLatticeMethodSolver(
            problem
        )
        with mock.patch.object(problem_class, "_assemble_interval_loads", both_paths):
            solver.run(show_progress=False, use_arrays=True)
        self.assertTrue(solver.use_arrays)
        self.assertTrue(any(from_loads is not None for from_loads, _ in results))
        for from_loads, from_airplane in results:
            if from_airplane is None:
                self.assertIsNone(from_loads)
                continue
            assert from_loads is not None
            for actual, expected in zip(from_loads, from_airplane):
                self.assertTrue(np.array_equal(actual, expected))


class TestFreeFlightAeroelasticUnsteadyProblemStepParameters(unittest.TestCase):
    """This is a class with functions to test that a
    FreeFlightAeroelasticUnsteadyProblem records, on the array path, the step parameters
    of every time step's Airplane."""

    def test_records_every_time_steps_parameters(self) -> None:
        """Test that, after an array path run, every time step's recorded step
        parameters rebuild exactly the Airplane the problem solved there."""
        problem: Any = _make_problem()
        solver = ps.free_flight_aeroelastic_unsteady_ring_vortex_lattice_method.FreeFlightAeroelasticUnsteadyRingVortexLatticeMethodSolver(
            problem
        )
        solver.run(show_progress=False, use_arrays=True)
        coupled_unsteady_problem_fixtures.assert_recorded_step_parameters_match_the_airplanes(
            self,
            problem,
            problem.movement.airplane_movements[0],
            range(problem.num_steps),
        )


class TestFreeFlightAeroelasticUnsteadyProblemExternalLoadsOnTheArrayPath(
    unittest.TestCase
):
    """This is a class with functions to test that the external_loads_fn gets the same
    Airplanes on both paths."""

    def test_external_loads_fn_gets_the_object_paths_airplanes(self) -> None:
        """Test that, on the array path, which builds no Airplanes during the run, the
        external_loads_fn is called as often as on the object path, with Airplanes
        carrying the same deformed Panel corners and the same loads, to round off."""
        records: dict[bool, list[dict[str, Any]]] = {}
        for use_arrays in (False, True):
            calls: list[dict[str, Any]] = []

            def external_loads_fn(
                operating_point: ps.operating_point.OperatingPoint,
                airplane: ps.geometry.airplane.Airplane,
            ) -> tuple[np.ndarray, np.ndarray]:
                panels = [
                    panel
                    for wing in airplane.wings
                    for panel in np.ravel(cast(np.ndarray, wing.panels))
                ]
                calls.append(
                    {
                        "forces_W": (
                            None
                            if airplane.forces_W is None
                            else np.copy(airplane.forces_W)
                        ),
                        "Frpp_G_Cg": np.array([panel.Frpp_G_Cg for panel in panels]),
                    }
                )
                return np.array([0.1, 0.0, -0.2]), np.array([0.0, 0.01, 0.0])

            problem = _make_problem(external_loads_fn=external_loads_fn)
            solver = ps.free_flight_aeroelastic_unsteady_ring_vortex_lattice_method.FreeFlightAeroelasticUnsteadyRingVortexLatticeMethodSolver(
                problem
            )
            solver.run(show_progress=False, use_arrays=use_arrays)
            records[use_arrays] = calls

        self.assertGreater(len(records[False]), 1)
        self.assertEqual(len(records[True]), len(records[False]))
        for call_id, (array_call, object_call) in enumerate(
            zip(records[True], records[False])
        ):
            with self.subTest(call=call_id):
                if object_call["forces_W"] is None:
                    self.assertIsNone(array_call["forces_W"])
                else:
                    np.testing.assert_allclose(
                        array_call["forces_W"],
                        object_call["forces_W"],
                        rtol=1.0e-9,
                        atol=1.0e-9 * float(np.max(np.abs(object_call["forces_W"]))),
                    )
                np.testing.assert_allclose(
                    array_call["Frpp_G_Cg"],
                    object_call["Frpp_G_Cg"],
                    rtol=0.0,
                    atol=1.0e-12,
                )


class TestFreeFlightAeroelasticUnsteadyProblemInitialPosition(unittest.TestCase):
    """This is a class with functions to test that a
    FreeFlightAeroelasticUnsteadyProblem's body starts at its initial OperatingPoint's
    position."""

    def test_body_starts_at_the_initial_operating_points_position(self) -> None:
        """Test that MuJoCo starts the body at the initial OperatingPoint's CgP1_E_Eo
        rather than at the Earth origin."""
        CgP1_E_Eo = (3.0, -1.0, -20.0)
        problem = _make_problem(movement=_make_movement(CgP1_E_Eo=CgP1_E_Eo))
        np.testing.assert_array_equal(
            problem._mujoco_model.get_state()["position_E_Eo"], np.array(CgP1_E_Eo)
        )
