"""This module contains classes to test FreeFlightMovements."""

import unittest
from typing import Any

import numpy as np

import pterasoftware as ps
from tests.unit.fixtures import (
    aeroelastic_airplane_movement_fixtures,
    airplane_movement_fixtures,
    flap_drive_fixtures,
    free_flight_movement_fixtures,
    free_flight_operating_point_movement_fixtures,
    geometry_fixtures,
    operating_point_fixtures,
)


class TestFreeFlightMovement(unittest.TestCase):
    """This is a class with functions to test FreeFlightMovements."""

    basic_free_flight_movement: ps.movements.free_flight_movement.FreeFlightMovement
    static_free_flight_movement: ps.movements.free_flight_movement.FreeFlightMovement
    free_flight_movement_with_multiple_airplanes: (
        ps.movements.free_flight_movement.FreeFlightMovement
    )

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test fixtures once for all FreeFlightMovement tests."""
        cls.basic_free_flight_movement = (
            free_flight_movement_fixtures.make_basic_free_flight_movement_fixture()
        )
        cls.static_free_flight_movement = (
            free_flight_movement_fixtures.make_static_free_flight_movement_fixture()
        )
        cls.free_flight_movement_with_multiple_airplanes = (
            free_flight_movement_fixtures.make_free_flight_movement_with_multiple_airplanes_fixture()
        )

    def test_is_subclass_of_core_movement(self) -> None:
        """Test that FreeFlightMovement is a subclass of CoreMovement."""
        self.assertTrue(
            issubclass(
                ps.movements.free_flight_movement.FreeFlightMovement,
                ps._core.CoreMovement,
            )
        )

    def test_instantiation_returns_correct_type(self) -> None:
        """Test that FreeFlightMovement instantiation returns a FreeFlightMovement."""
        self.assertIsInstance(
            self.basic_free_flight_movement,
            ps.movements.free_flight_movement.FreeFlightMovement,
        )

    def test_rejects_non_airplane_movement_children(self) -> None:
        """Test that FreeFlightMovement rejects children that are not
        AirplaneMovements."""
        operating_point_movement = (
            free_flight_operating_point_movement_fixtures.make_basic_free_flight_operating_point_movement_fixture()
        )

        bad_airplane_movements: Any = ["not an AirplaneMovement"]
        with self.assertRaises(TypeError):
            ps.movements.free_flight_movement.FreeFlightMovement(
                airplane_movements=bad_airplane_movements,
                operating_point_movement=operating_point_movement,
                delta_time=0.1,
                prescribed_num_steps=2,
                free_num_steps=3,
            )

    def test_rejects_non_free_flight_operating_point_movement(self) -> None:
        """Test that FreeFlightMovement rejects OperatingPointMovement instances that
        are not FreeFlightOperatingPointMovements."""
        airplane_movements = [
            airplane_movement_fixtures.make_basic_airplane_movement_fixture()
        ]
        operating_point_movement: Any = (
            ps.movements.operating_point_movement.OperatingPointMovement(
                base_operating_point=operating_point_fixtures.make_basic_operating_point_fixture()
            )
        )

        with self.assertRaises(TypeError):
            ps.movements.free_flight_movement.FreeFlightMovement(
                airplane_movements=airplane_movements,
                operating_point_movement=operating_point_movement,
                delta_time=0.1,
                prescribed_num_steps=2,
                free_num_steps=3,
            )

    def test_prescribed_num_steps_validation(self) -> None:
        """Test prescribed_num_steps parameter validation."""
        airplane_movements = [
            airplane_movement_fixtures.make_basic_airplane_movement_fixture()
        ]
        operating_point_movement = (
            free_flight_operating_point_movement_fixtures.make_basic_free_flight_operating_point_movement_fixture()
        )

        invalid_values: list[Any] = [0, -5, 2.5, "three"]
        for invalid_value in invalid_values:
            with self.subTest(invalid_value=invalid_value):
                # noinspection PyTypeChecker
                with self.assertRaises((ValueError, TypeError)):
                    ps.movements.free_flight_movement.FreeFlightMovement(
                        airplane_movements=airplane_movements,
                        operating_point_movement=operating_point_movement,
                        delta_time=0.1,
                        prescribed_num_steps=invalid_value,
                        free_num_steps=3,
                    )

    def test_free_num_steps_validation(self) -> None:
        """Test free_num_steps parameter validation."""
        airplane_movements = [
            airplane_movement_fixtures.make_basic_airplane_movement_fixture()
        ]
        operating_point_movement = (
            free_flight_operating_point_movement_fixtures.make_basic_free_flight_operating_point_movement_fixture()
        )

        invalid_values: list[Any] = [0, -5, 2.5, "three"]
        for invalid_value in invalid_values:
            with self.subTest(invalid_value=invalid_value):
                # noinspection PyTypeChecker
                with self.assertRaises((ValueError, TypeError)):
                    ps.movements.free_flight_movement.FreeFlightMovement(
                        airplane_movements=airplane_movements,
                        operating_point_movement=operating_point_movement,
                        delta_time=0.1,
                        prescribed_num_steps=2,
                        free_num_steps=invalid_value,
                    )

    def test_delta_time_validation(self) -> None:
        """Test that delta_time must be a positive number."""
        airplane_movements = [
            airplane_movement_fixtures.make_basic_airplane_movement_fixture()
        ]
        operating_point_movement = (
            free_flight_operating_point_movement_fixtures.make_basic_free_flight_operating_point_movement_fixture()
        )

        invalid_values: list[Any] = [0.0, -0.01, "three"]
        for invalid_value in invalid_values:
            with self.subTest(invalid_value=invalid_value):
                # noinspection PyTypeChecker
                with self.assertRaises((ValueError, TypeError)):
                    ps.movements.free_flight_movement.FreeFlightMovement(
                        airplane_movements=airplane_movements,
                        operating_point_movement=operating_point_movement,
                        delta_time=invalid_value,
                        prescribed_num_steps=2,
                        free_num_steps=3,
                    )

    def test_num_steps_equals_sum_of_prescribed_and_free(self) -> None:
        """Test that num_steps equals prescribed_num_steps plus free_num_steps."""
        self.assertEqual(self.basic_free_flight_movement.num_steps, 5)
        self.assertEqual(
            self.basic_free_flight_movement.num_steps,
            self.basic_free_flight_movement.prescribed_num_steps
            + self.basic_free_flight_movement.free_num_steps,
        )

    def test_prescribed_num_steps_property(self) -> None:
        """Test that prescribed_num_steps is stored and accessible via the property."""
        self.assertEqual(self.basic_free_flight_movement.prescribed_num_steps, 2)

    def test_free_num_steps_property(self) -> None:
        """Test that free_num_steps is stored and accessible via the property."""
        self.assertEqual(self.basic_free_flight_movement.free_num_steps, 3)

    def test_airplane_movements_returns_tuple_of_airplane_movements(self) -> None:
        """Test that airplane_movements returns a tuple of AirplaneMovements."""
        airplane_movements = self.basic_free_flight_movement.airplane_movements
        self.assertIsInstance(airplane_movements, tuple)
        for airplane_movement in airplane_movements:
            self.assertIsInstance(
                airplane_movement,
                ps.movements.airplane_movement.AirplaneMovement,
            )

    def test_operating_point_movement_returns_free_flight_operating_point_movement(
        self,
    ) -> None:
        """Test that operating_point_movement returns a
        FreeFlightOperatingPointMovement."""
        self.assertIsInstance(
            self.basic_free_flight_movement.operating_point_movement,
            ps.movements.free_flight_operating_point_movement.FreeFlightOperatingPointMovement,
        )

    def test_airplanes_structure(self) -> None:
        """Test that airplanes is a tuple of tuples of Airplanes with the correct
        dimensions."""
        airplanes = self.basic_free_flight_movement.airplanes
        self.assertIsInstance(airplanes, tuple)

        # The first index runs over the AirplaneMovements.
        self.assertEqual(
            len(airplanes),
            len(self.basic_free_flight_movement.airplane_movements),
        )

        # The second index runs over the time steps.
        for airplane_list in airplanes:
            self.assertIsInstance(airplane_list, tuple)
            self.assertEqual(
                len(airplane_list), self.basic_free_flight_movement.num_steps
            )
            for airplane in airplane_list:
                self.assertIsInstance(airplane, ps.geometry.airplane.Airplane)

    def test_airplanes_structure_with_multiple_airplanes(self) -> None:
        """Test that airplanes has one inner tuple per AirplaneMovement."""
        airplanes = self.free_flight_movement_with_multiple_airplanes.airplanes
        self.assertEqual(len(airplanes), 2)
        for airplane_list in airplanes:
            self.assertEqual(
                len(airplane_list),
                self.free_flight_movement_with_multiple_airplanes.num_steps,
            )

    def test_static_property_for_static_free_flight_movement(self) -> None:
        """Test that static returns True for a static FreeFlightMovement."""
        self.assertTrue(self.static_free_flight_movement.static)

    def test_static_property_for_non_static_free_flight_movement(self) -> None:
        """Test that static returns False for a non static FreeFlightMovement."""
        self.assertFalse(self.basic_free_flight_movement.static)

    def test_max_period_for_static_free_flight_movement(self) -> None:
        """Test that max_period returns 0.0 for a static FreeFlightMovement."""
        self.assertEqual(self.static_free_flight_movement.max_period, 0.0)

    def test_max_period_for_non_static_free_flight_movement(self) -> None:
        """Test that max_period returns the correct value for a non static
        FreeFlightMovement."""
        # The basic FreeFlightMovement's prescribed motion has a period of 2.0.
        self.assertEqual(self.basic_free_flight_movement.max_period, 2.0)

    def test_min_period_for_non_static_free_flight_movement(self) -> None:
        """Test that min_period returns the correct value for a non static
        FreeFlightMovement."""
        # The basic FreeFlightMovement's prescribed motion has a period of 2.0.
        self.assertEqual(self.basic_free_flight_movement.min_period, 2.0)

    def test_lcm_period_for_non_static_free_flight_movement(self) -> None:
        """Test that lcm_period returns the correct value for a non static
        FreeFlightMovement."""
        # The basic FreeFlightMovement's prescribed motion has a period of 2.0, so the
        # LCM of its identical periods is that period.
        self.assertEqual(self.basic_free_flight_movement.lcm_period, 2.0)

    def test_max_wake_rows_default_none(self) -> None:
        """Test that max_wake_rows defaults to None."""
        self.assertIsNone(self.basic_free_flight_movement.max_wake_rows)

    def test_max_wake_rows_stored_and_accessible(self) -> None:
        """Test that max_wake_rows is stored and accessible via the property."""
        airplane_movements = [
            airplane_movement_fixtures.make_basic_airplane_movement_fixture()
        ]
        operating_point_movement = (
            free_flight_operating_point_movement_fixtures.make_basic_free_flight_operating_point_movement_fixture()
        )

        free_flight_movement = ps.movements.free_flight_movement.FreeFlightMovement(
            airplane_movements=airplane_movements,
            operating_point_movement=operating_point_movement,
            delta_time=0.1,
            prescribed_num_steps=2,
            free_num_steps=3,
            max_wake_rows=5,
        )
        self.assertEqual(free_flight_movement.max_wake_rows, 5)

    def test_max_wake_rows_validation(self) -> None:
        """Test that max_wake_rows must be a positive int."""
        airplane_movements = [
            airplane_movement_fixtures.make_basic_airplane_movement_fixture()
        ]
        operating_point_movement = (
            free_flight_operating_point_movement_fixtures.make_basic_free_flight_operating_point_movement_fixture()
        )

        invalid_values: list[Any] = [0, -1, 2.5, "three"]
        for invalid_value in invalid_values:
            with self.subTest(invalid_value=invalid_value):
                # noinspection PyTypeChecker
                with self.assertRaises((ValueError, TypeError)):
                    ps.movements.free_flight_movement.FreeFlightMovement(
                        airplane_movements=airplane_movements,
                        operating_point_movement=operating_point_movement,
                        delta_time=0.1,
                        prescribed_num_steps=2,
                        free_num_steps=3,
                        max_wake_rows=invalid_value,
                    )

    def test_symmetry_change_raises_error(self) -> None:
        """Test that a Wing transitioning between symmetry types raises an error."""
        # Create a type 4 Wing (symmetric=True, coincident symmetry plane).
        base_wing = geometry_fixtures.make_type_4_wing_fixture()

        # Create an Airplane with the base Wing first, so it processes symmetry. The
        # base Airplane is the first in a simulation, so its Cg_GP1_CgP1 is all zeros.
        base_airplane = ps.geometry.airplane.Airplane(
            wings=[base_wing],
            name="Test Airplane",
            Cg_GP1_CgP1=(0.0, 0.0, 0.0),
        )

        # Reference the Wing from the Airplane (after symmetry processing).
        processed_wing = base_airplane.wings[0]

        # Create WingCrossSectionMovements using the actual WingCrossSections.
        wing_cross_section_movements = [
            ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
                base_wing_cross_section=wing_cross_section,
            )
            for wing_cross_section in processed_wing.wing_cross_sections
        ]

        # Create a WingMovement with rotation that will cause the symmetry plane to
        # become non coincident (a type 4 to type 5 transition).
        wing_movement = ps.movements.wing_movement.WingMovement(
            base_wing=processed_wing,
            wing_cross_section_movements=wing_cross_section_movements,
            ampAngles_Gs_to_Wn_ixyz=(15.0, 0.0, 0.0),
            periodAngles_Gs_to_Wn_ixyz=(1.0, 0.0, 0.0),
        )

        airplane_movement = ps.movements.airplane_movement.AirplaneMovement(
            base_airplane=base_airplane,
            wing_movements=[wing_movement],
        )

        operating_point_movement = (
            free_flight_operating_point_movement_fixtures.make_basic_free_flight_operating_point_movement_fixture()
        )

        # Attempting to create a FreeFlightMovement should raise a ValueError. With a
        # delta_time of 0.25 and a period of 1.0, the second time step lands at the peak
        # of the rotation oscillation, so the symmetry plane is non coincident there.
        with self.assertRaises(ValueError) as context:
            ps.movements.free_flight_movement.FreeFlightMovement(
                airplane_movements=[airplane_movement],
                operating_point_movement=operating_point_movement,
                delta_time=0.25,
                prescribed_num_steps=1,
                free_num_steps=1,
            )

        # Verify the error message mentions symmetry.
        self.assertIn("symmetry", str(context.exception).lower())

    def test_static_motion_with_symmetric_wing_succeeds(self) -> None:
        """Test that a FreeFlightMovement with a symmetric Wing but no motion
        succeeds."""
        # Create a type 4 Wing (symmetric=True, coincident symmetry plane).
        base_wing = geometry_fixtures.make_type_4_wing_fixture()

        # Create an Airplane with the base Wing first, so it processes symmetry.
        base_airplane = ps.geometry.airplane.Airplane(
            wings=[base_wing],
            name="Test Airplane",
            Cg_GP1_CgP1=(0.0, 0.0, 0.0),
        )

        # Reference the Wing from the Airplane (after symmetry processing).
        processed_wing = base_airplane.wings[0]

        # Create static WingCrossSectionMovements using the actual WingCrossSections.
        wing_cross_section_movements = [
            ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
                base_wing_cross_section=wing_cross_section,
            )
            for wing_cross_section in processed_wing.wing_cross_sections
        ]

        # Create a static WingMovement (no rotation or translation).
        wing_movement = ps.movements.wing_movement.WingMovement(
            base_wing=processed_wing,
            wing_cross_section_movements=wing_cross_section_movements,
        )

        airplane_movement = ps.movements.airplane_movement.AirplaneMovement(
            base_airplane=base_airplane,
            wing_movements=[wing_movement],
        )

        operating_point_movement = (
            free_flight_operating_point_movement_fixtures.make_basic_free_flight_operating_point_movement_fixture()
        )

        # Creating a FreeFlightMovement should succeed without raising an error.
        free_flight_movement = ps.movements.free_flight_movement.FreeFlightMovement(
            airplane_movements=[airplane_movement],
            operating_point_movement=operating_point_movement,
            delta_time=0.1,
            prescribed_num_steps=2,
            free_num_steps=3,
        )

        self.assertIsInstance(
            free_flight_movement,
            ps.movements.free_flight_movement.FreeFlightMovement,
        )


class TestFreeFlightMovementImmutability(unittest.TestCase):
    """Tests for FreeFlightMovement attribute immutability."""

    basic_free_flight_movement: ps.movements.free_flight_movement.FreeFlightMovement

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test fixtures once for all immutability tests."""
        cls.basic_free_flight_movement = (
            free_flight_movement_fixtures.make_basic_free_flight_movement_fixture()
        )

    def test_immutable_prescribed_num_steps_property(self) -> None:
        """Test that prescribed_num_steps property is read only."""
        with self.assertRaises(AttributeError):
            setattr(self.basic_free_flight_movement, "prescribed_num_steps", 100)

    def test_immutable_free_num_steps_property(self) -> None:
        """Test that free_num_steps property is read only."""
        with self.assertRaises(AttributeError):
            setattr(self.basic_free_flight_movement, "free_num_steps", 100)

    def test_immutable_airplanes_property(self) -> None:
        """Test that airplanes property is read only."""
        with self.assertRaises(AttributeError):
            setattr(self.basic_free_flight_movement, "airplanes", ())

    def test_immutable_airplane_movements_property(self) -> None:
        """Test that airplane_movements property is read only."""
        with self.assertRaises(AttributeError):
            setattr(self.basic_free_flight_movement, "airplane_movements", ())


class TestFreeFlightAeroelasticMovementStepParameters(unittest.TestCase):
    """This is a class with functions to test the per time step parameters of
    FreeFlightAeroelasticMovements."""

    def test_parameters_delegate_with_internal_delta_time(self) -> None:
        """Test that the parameters equal those of the Airplane's movement at the
        movement's own delta_time, with the deformation passed through."""
        movement = (
            flap_drive_fixtures.make_free_flight_flap_driven_aeroelastic_movement_fixture()
        )
        airplane_movement = movement.airplane_movements[0]
        assert isinstance(
            airplane_movement,
            ps.movements.aeroelastic_airplane_movement.AeroelasticAirplaneMovement,
        )
        deformations: list[np.ndarray | None] = []
        for wing_movement in airplane_movement.wing_movements:
            deformation = np.full(
                (len(wing_movement.wing_cross_section_movements), 3), 0.2, dtype=float
            )
            # A Wing's root WingCrossSection must keep zero angles, so it never deforms.
            deformation[0] = 0.0
            deformations.append(deformation)
        for deformation_list in (None, deformations):
            for step in (0, 1, 3):
                with self.subTest(deformation_list=deformation_list, step=step):
                    parameters = movement._generate_airplane_parameters_at_time_step(
                        0, step, deformationAngles_Wcsp_to_Wcs_ixyz=deformation_list
                    )
                    expected = (
                        airplane_movement._generate_airplane_parameters_at_time_step(
                            step,
                            movement.delta_time,
                            deformationAngles_Wcsp_to_Wcs_ixyz=deformation_list,
                        )
                    )
                    self.assertTrue(
                        np.array_equal(parameters.Cg_GP1_CgP1, expected.Cg_GP1_CgP1)
                    )
                    for wing_parameters, expected_wing in zip(
                        parameters.wings, expected.wings
                    ):
                        for array, expected_array in zip(
                            wing_parameters, expected_wing
                        ):
                            self.assertTrue(np.array_equal(array, expected_array))

    def test_parameters_use_the_movements_own_delta_time(self) -> None:
        """Test that, for moving geometry and a delta_time of 0.037 s, the parameters
        equal those of the Airplane's movement at the movement's own delta_time and
        differ from those at another delta_time."""
        movement = (
            free_flight_movement_fixtures.make_moving_free_flight_aeroelastic_movement_fixture()
        )
        airplane_movement = movement.airplane_movements[0]
        assert isinstance(
            airplane_movement,
            ps.movements.aeroelastic_airplane_movement.AeroelasticAirplaneMovement,
        )
        self.assertEqual(movement.delta_time, 0.037)
        for step in (1, 2, 3):
            with self.subTest(step=step):
                parameters = movement._generate_airplane_parameters_at_time_step(
                    0, step
                )
                expected = airplane_movement._generate_airplane_parameters_at_time_step(
                    step, 0.037
                )
                elsewhere = (
                    airplane_movement._generate_airplane_parameters_at_time_step(
                        step, 0.1
                    )
                )
                self.assertTrue(
                    np.array_equal(
                        parameters.wings[0].angles_Gs_to_Wn_ixyz,
                        expected.wings[0].angles_Gs_to_Wn_ixyz,
                    )
                )
                self.assertFalse(
                    np.array_equal(
                        parameters.wings[0].angles_Gs_to_Wn_ixyz,
                        elsewhere.wings[0].angles_Gs_to_Wn_ixyz,
                    )
                )


class TestFreeFlightAeroelasticMovementPrescribedAirplanes(unittest.TestCase):
    """This is a class with functions to test that FreeFlightAeroelasticMovements build
    their prescribed Airplanes only when they are read, and still check every time
    step's symmetry types when they are created."""

    def test_no_airplanes_are_built_at_construction(self) -> None:
        """Test that creating the movement builds no Airplanes."""
        movement = (
            free_flight_movement_fixtures.make_moving_free_flight_aeroelastic_movement_fixture()
        )
        self.assertIsNone(movement._airplanes)

    def test_airplanes_match_the_movements_prescribed_geometry(self) -> None:
        """Test that the airplanes property holds, for every time step, the Airplane the
        Airplane's movement builds at that time step, that it is built once, and that
        the base Airplane's reference chord (which the problem reads) equals the first
        Airplane's."""
        movement = (
            free_flight_movement_fixtures.make_moving_free_flight_aeroelastic_movement_fixture()
        )
        airplanes = movement.airplanes
        self.assertIs(movement.airplanes, airplanes)
        airplane_movement = movement.airplane_movements[0]
        num_steps = movement.prescribed_num_steps + movement.free_num_steps
        self.assertEqual(len(airplanes), 1)
        self.assertEqual(len(airplanes[0]), num_steps)
        for step in range(num_steps):
            with self.subTest(step=step):
                expected = airplane_movement.generate_airplane_at_time_step(
                    step, movement.delta_time
                )
                airplane = airplanes[0][step]
                self.assertTrue(
                    np.array_equal(airplane.Cg_GP1_CgP1, expected.Cg_GP1_CgP1)
                )
                self.assertEqual(len(airplane.wings), len(expected.wings))
                for wing, expected_wing in zip(airplane.wings, expected.wings):
                    self.assertTrue(
                        np.array_equal(wing.Ler_Gs_Cgs, expected_wing.Ler_Gs_Cgs)
                    )
                    self.assertTrue(
                        np.array_equal(
                            wing.angles_Gs_to_Wn_ixyz,
                            expected_wing.angles_Gs_to_Wn_ixyz,
                        )
                    )
        self.assertEqual(airplane_movement.base_airplane.c_ref, airplanes[0][0].c_ref)

    def test_out_of_range_motion_is_rejected_at_construction(self) -> None:
        """Test that a movement whose Wing's or tip WingCrossSection's angle leaves
        [-90.0, 90.0] degrees at a later time step is rejected when it is created, with
        the error building that time step's Airplane raises, as when the movement built
        every Airplane up front, and that the same motion within range is accepted."""
        cases = (
            (dict(ampAngleY_Gs_to_Wn=100.0), True),
            (dict(ampTipAngleY_Wcsp_to_Wcs=100.0), True),
            (dict(ampAngleY_Gs_to_Wn=80.0), False),
        )
        for amplitudes, should_raise in cases:
            with self.subTest(amplitudes=amplitudes):

                def make() -> None:
                    ps.movements.free_flight_movement.FreeFlightAeroelasticMovement(
                        airplane_movements=[
                            aeroelastic_airplane_movement_fixtures.make_out_of_range_aeroelastic_airplane_movement_fixture(
                                **amplitudes
                            )
                        ],
                        operating_point_movement=free_flight_operating_point_movement_fixtures.make_basic_free_flight_operating_point_movement_fixture(),
                        delta_time=0.1,
                        prescribed_num_steps=3,
                        free_num_steps=2,
                    )

                if not should_raise:
                    make()
                    continue

                # The angle is out of range at time step 1 (a quarter period).
                airplane_movement = aeroelastic_airplane_movement_fixtures.make_out_of_range_aeroelastic_airplane_movement_fixture(
                    **amplitudes
                )
                with self.assertRaises(ValueError) as expected:
                    airplane_movement.generate_airplane_at_time_step(1, 0.1)
                with self.assertRaises(ValueError) as context:
                    make()
                self.assertEqual(str(context.exception), str(expected.exception))

    def test_a_symmetry_type_change_is_rejected_at_construction(self) -> None:
        """Test that a movement whose Wing changes symmetry type at a later time step is
        rejected when it is created, and that the same movement without the motion is
        accepted."""
        for ampLer_y, should_raise in ((0.1, True), (0.0, False)):
            with self.subTest(ampLer_y=ampLer_y):

                def make() -> None:
                    ps.movements.free_flight_movement.FreeFlightAeroelasticMovement(
                        airplane_movements=[
                            aeroelastic_airplane_movement_fixtures.make_symmetry_changing_aeroelastic_airplane_movement_fixture(
                                ampLer_y
                            )
                        ],
                        operating_point_movement=free_flight_operating_point_movement_fixtures.make_basic_free_flight_operating_point_movement_fixture(),
                        delta_time=0.1,
                        prescribed_num_steps=3,
                        free_num_steps=2,
                    )

                if should_raise:
                    with self.assertRaisesRegex(ValueError, "changed from type 4"):
                        make()
                else:
                    make()
