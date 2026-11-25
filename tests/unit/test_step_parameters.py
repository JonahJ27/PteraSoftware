"""This module contains classes to test the per time step parameter records'
validation."""

import unittest
from typing import Any

import numpy as np

import pterasoftware as ps

# noinspection PyProtectedMember
from pterasoftware import _mesh_arrays, _step_parameters
from tests.unit.fixtures import (
    airplane_movement_fixtures,
    geometry_fixtures,
    mesh_arrays_fixtures,
)


def _validate_wing_step_parameters(
    plan: _mesh_arrays.MeshPlan, parameters: _step_parameters.WingStepParameters
) -> None:
    """Calls validate_wing_step_parameters with mesh_wing_arrays' signature, so the
    tests can check both the same way.

    :param plan: Unused.
    :param parameters: The WingStepParameters to validate.
    :return: None
    """
    _step_parameters.validate_wing_step_parameters(parameters)


def _object_path_error(
    wing: ps.geometry.wing.Wing,
    parameters: _step_parameters.WingStepParameters,
) -> Exception | None:
    """Builds a Wing from WingStepParameters the way the object path does (a
    WingMovement's _wing_from_parameters) and returns the error it raises, if any.

    :param wing: The processed Wing, used as the WingMovement's base Wing.
    :param parameters: The WingStepParameters to build the Wing from.
    :return: The raised exception, or None if the Wing was built.
    """
    wing_movement = ps.movements.wing_movement.WingMovement(
        base_wing=wing,
        wing_cross_section_movements=[
            ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
                base_wing_cross_section=wing_cross_section
            )
            for wing_cross_section in wing.wing_cross_sections
        ],
    )
    try:
        # noinspection PyProtectedMember
        wing_movement._wing_from_parameters(parameters)
    except (ValueError, TypeError) as error:
        return error
    return None


def _changed_parameters(
    parameters: _step_parameters.WingStepParameters,
    Ler_Gs_Cgs: np.ndarray | None = None,
    angles_Gs_to_Wn_ixyz: np.ndarray | None = None,
    changes: tuple[tuple[str, int, int, float], ...] = (),
) -> _step_parameters.WingStepParameters:
    """Returns a copy of WingStepParameters with some values replaced.

    :param parameters: The WingStepParameters to copy.
    :param Ler_Gs_Cgs: The new Ler_Gs_Cgs, or None to keep it.
    :param angles_Gs_to_Wn_ixyz: The new angles_Gs_to_Wn_ixyz, or None to keep it.
    :param changes: Entries of a stacked WingCrossSection value to replace, each as
        ("Lp" or "angles", WingCrossSection index, component, new value).
    :return: The new WingStepParameters.
    """
    stacks = {
        "Lp": np.array(parameters.stackLp_Wcsp_Lpp, dtype=float),
        "angles": np.array(parameters.stackAngles_Wcsp_to_Wcs_ixyz, dtype=float),
    }
    for name, index, component, value in changes:
        stacks[name][index, component] = value
    return _step_parameters.WingStepParameters(
        Ler_Gs_Cgs=(
            np.array(parameters.Ler_Gs_Cgs, dtype=float)
            if Ler_Gs_Cgs is None
            else Ler_Gs_Cgs
        ),
        angles_Gs_to_Wn_ixyz=(
            np.array(parameters.angles_Gs_to_Wn_ixyz, dtype=float)
            if angles_Gs_to_Wn_ixyz is None
            else angles_Gs_to_Wn_ixyz
        ),
        stackLp_Wcsp_Lpp=stacks["Lp"],
        stackAngles_Wcsp_to_Wcs_ixyz=stacks["angles"],
    )


class TestValidateWingStepParameters(unittest.TestCase):
    """This is a class with functions to test that validate_wing_step_parameters raises
    exactly the errors the object path raises when it builds the same Wing."""

    def _assert_same_error_as_object_path(
        self,
        wing: ps.geometry.wing.Wing,
        plan: _mesh_arrays.MeshPlan,
        parameters: _step_parameters.WingStepParameters,
    ) -> None:
        """Asserts that validate_wing_step_parameters and mesh_wing_arrays raise the
        same exception type and message as the object path, or that all three accept the
        values.

        :param wing: The processed Wing.
        :param plan: Its MeshPlan.
        :param parameters: The WingStepParameters to check.
        :return: None
        """
        expected = _object_path_error(wing, parameters)
        for function in (
            _validate_wing_step_parameters,
            _mesh_arrays.mesh_wing_arrays,
        ):
            if expected is None:
                function(plan, parameters)
                continue
            with self.assertRaises(type(expected)) as context:
                function(plan, parameters)
            self.assertEqual(str(context.exception), str(expected))

    def test_accepts_what_the_object_path_accepts(self) -> None:
        """Test that every fixture's and random Wing's own values, and values exactly at
        the limits, are accepted, as the object path accepts them."""
        wings = [
            wing
            for make in (
                geometry_fixtures.make_type_1_wing_fixture,
                geometry_fixtures.make_type_2_wing_fixture,
                geometry_fixtures.make_type_3_wing_fixture,
                geometry_fixtures.make_type_4_wing_fixture,
                geometry_fixtures.make_type_5_wing_fixture,
                geometry_fixtures.make_three_section_wing_fixture,
            )
            for wing in mesh_arrays_fixtures.make_processed_wings_fixture(make())
        ] + mesh_arrays_fixtures.make_random_processed_wings_fixture(seed=1, count=60)
        for wing in wings:
            with self.subTest(wing=wing.name):
                plan = _mesh_arrays.build_mesh_plan(wing)
                parameters = mesh_arrays_fixtures.make_wing_step_parameters_fixture(
                    wing
                )
                self.assertIsNone(_object_path_error(wing, parameters))
                _step_parameters.validate_wing_step_parameters(parameters)

        # Values at the limits: angles of exactly +/-90 degrees and a root offset inside
        # np.allclose's tolerance.
        wing = mesh_arrays_fixtures.make_processed_wings_fixture(
            geometry_fixtures.make_three_section_wing_fixture()
        )[0]
        plan = _mesh_arrays.build_mesh_plan(wing)
        parameters = mesh_arrays_fixtures.make_wing_step_parameters_fixture(wing)
        for limit_parameters in (
            _changed_parameters(parameters, changes=(("angles", 1, 0, 90.0),)),
            _changed_parameters(parameters, changes=(("angles", 2, 2, -90.0),)),
            _changed_parameters(parameters, changes=(("Lp", 0, 0, 5.0e-9),)),
            _changed_parameters(
                parameters, angles_Gs_to_Wn_ixyz=np.array([90.0, -90.0, 0.0])
            ),
        ):
            with self.subTest(parameters=limit_parameters):
                self._assert_same_error_as_object_path(wing, plan, limit_parameters)

    def test_raises_the_object_path_errors(self) -> None:
        """Test that each kind of invalid value raises the same exception type and
        message as building the Wing on the object path, including when more than one
        value is invalid (the object path's first error)."""
        cases: dict[str, dict[str, Any]] = {
            "WingCrossSection angle above 90": dict(changes=(("angles", 1, 0, 95.0),)),
            "WingCrossSection angle below -90": dict(
                changes=(("angles", 2, 1, -91.0),)
            ),
            "non root Lp_Wcsp_Lpp[1] zero": dict(changes=(("Lp", 1, 1, 0.0),)),
            "non root Lp_Wcsp_Lpp[1] negative": dict(changes=(("Lp", 2, 1, -0.1),)),
            "root Lp_Wcsp_Lpp not zero": dict(changes=(("Lp", 0, 0, 0.1),)),
            "root Lp_Wcsp_Lpp[1] slightly negative": dict(
                changes=(("Lp", 0, 1, -1.0e-10),)
            ),
            "root angles not zero": dict(changes=(("angles", 0, 1, 5.0),)),
            "nan in Lp_Wcsp_Lpp": dict(changes=(("Lp", 1, 2, np.nan),)),
            "inf in angles_Wcsp_to_Wcs_ixyz": dict(changes=(("angles", 1, 2, np.inf),)),
            "nan in Ler_Gs_Cgs": dict(Ler_Gs_Cgs=np.array([0.0, np.nan, 0.0])),
            "nan in angles_Gs_to_Wn_ixyz": dict(
                angles_Gs_to_Wn_ixyz=np.array([np.nan, 0.0, 0.0])
            ),
            "Wing angle above 90": dict(
                angles_Gs_to_Wn_ixyz=np.array([0.0, 100.0, 0.0])
            ),
            "two invalid values": dict(
                angles_Gs_to_Wn_ixyz=np.array([0.0, 100.0, 0.0]),
                changes=(("Lp", 2, 1, 0.0), ("angles", 1, 0, 95.0)),
            ),
        }
        wing = mesh_arrays_fixtures.make_processed_wings_fixture(
            geometry_fixtures.make_three_section_wing_fixture()
        )[0]
        plan = _mesh_arrays.build_mesh_plan(wing)
        parameters = mesh_arrays_fixtures.make_wing_step_parameters_fixture(wing)
        for name, case in cases.items():
            with self.subTest(case=name):
                invalid_parameters = _changed_parameters(parameters, **case)
                self.assertIsNotNone(_object_path_error(wing, invalid_parameters))
                self._assert_same_error_as_object_path(wing, plan, invalid_parameters)


class TestValidateAirplaneStepParameters(unittest.TestCase):
    """This is a class with functions to test that validate_airplane_step_parameters
    raises exactly the errors the object path raises when it builds the same
    Airplane."""

    def test_matches_the_object_path(self) -> None:
        """Test that valid AirplaneStepParameters are accepted, and that a non finite
        Cg_GP1_CgP1, an invalid Wing value, and both together raise the same exception
        type and message as building the Airplane (the Wing's error first, since the
        object path builds the Wings before the Airplane)."""
        airplane_movement = (
            airplane_movement_fixtures.make_basic_airplane_movement_fixture()
        )
        # noinspection PyProtectedMember
        parameters = airplane_movement._generate_airplane_parameters_at_time_step(
            1, 0.1
        )
        wing_parameters = parameters.wings[0]
        invalid_wing_parameters = _step_parameters.WingStepParameters(
            Ler_Gs_Cgs=wing_parameters.Ler_Gs_Cgs,
            angles_Gs_to_Wn_ixyz=np.array([0.0, 100.0, 0.0], dtype=float),
            stackLp_Wcsp_Lpp=wing_parameters.stackLp_Wcsp_Lpp,
            stackAngles_Wcsp_to_Wcs_ixyz=wing_parameters.stackAngles_Wcsp_to_Wcs_ixyz,
        )
        nan_Cg_GP1_CgP1 = np.array([np.nan, 0.0, 0.0], dtype=float)
        cases = {
            "valid": parameters,
            "nan Cg_GP1_CgP1": parameters._replace(Cg_GP1_CgP1=nan_Cg_GP1_CgP1),
            "invalid Wing": parameters._replace(
                wings=(invalid_wing_parameters,) + parameters.wings[1:]
            ),
            "both": parameters._replace(
                Cg_GP1_CgP1=nan_Cg_GP1_CgP1,
                wings=(invalid_wing_parameters,) + parameters.wings[1:],
            ),
        }
        for name, case_parameters in cases.items():
            with self.subTest(case=name):
                expected: Exception | None = None
                try:
                    # noinspection PyProtectedMember
                    airplane_movement._airplane_from_parameters(case_parameters)
                except (ValueError, TypeError) as error:
                    expected = error

                if expected is None:
                    _step_parameters.validate_airplane_step_parameters(case_parameters)
                    continue
                with self.assertRaises(type(expected)) as context:
                    _step_parameters.validate_airplane_step_parameters(case_parameters)
                self.assertEqual(str(context.exception), str(expected))


class TestWingStepParametersKey(unittest.TestCase):
    """This is a class with functions to test wing_step_parameters_key, the exact key of
    the array path's mesh cache."""

    def setUp(self) -> None:
        """Set up a WingStepParameters with a zero in each of its arrays."""
        self.parameters = _step_parameters.WingStepParameters(
            Ler_Gs_Cgs=np.array([0.0, 0.5, -0.1]),
            angles_Gs_to_Wn_ixyz=np.array([12.5, 0.0, 3.0]),
            stackLp_Wcsp_Lpp=np.array([[0.0, 0.0, 0.0], [0.1, 0.4, 0.02]]),
            stackAngles_Wcsp_to_Wcs_ixyz=np.array([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        )

    def _changed(self, field: str, value: np.ndarray) -> Any:
        """Returns the WingStepParameters with one field replaced.

        :param field: The field's name.
        :param value: Its new value.
        :return: The WingStepParameters.
        """
        return self.parameters._replace(**{field: value})

    def test_equal_values_give_equal_keys(self) -> None:
        """Test that copies of the same values give the same key, and that the key is
        hashable."""
        copied = _step_parameters.WingStepParameters(
            *(np.array(value, copy=True) for value in self.parameters)
        )
        key = _step_parameters.wing_step_parameters_key(self.parameters)
        self.assertEqual(_step_parameters.wing_step_parameters_key(copied), key)
        self.assertEqual(hash(key), hash(key))

    def test_any_change_gives_a_different_key(self) -> None:
        """Test that one unit in the last place in any element, the sign of a zero, a
        nan, and a different shape each change the key."""
        key = _step_parameters.wing_step_parameters_key(self.parameters)
        for field in self.parameters._fields:
            value = getattr(self.parameters, field)
            for index in range(value.size):
                for label, new_element in (
                    ("nudged", np.nextafter(value.flat[index], np.inf)),
                    ("negated", -value.flat[index]),
                ):
                    if label == "negated" and value.flat[index] != 0.0:
                        continue
                    changed = np.array(value, copy=True)
                    changed.flat[index] = new_element
                    with self.subTest(field=field, index=index, change=label):
                        self.assertNotEqual(
                            _step_parameters.wing_step_parameters_key(
                                self._changed(field, changed)
                            ),
                            key,
                        )
            with self.subTest(field=field, change="shape"):
                self.assertNotEqual(
                    _step_parameters.wing_step_parameters_key(
                        self._changed(field, value.reshape(1, -1))
                    ),
                    key,
                )

    def test_nan_matches_only_the_same_nan(self) -> None:
        """Test that a nan gives the same key as the same nan (which == wouldn't), and
        a different key from a nan with another payload."""
        with_nan = np.array(self.parameters.Ler_Gs_Cgs, copy=True)
        with_nan[0] = np.nan
        other_nan = np.array(with_nan, copy=True)
        other_nan.view(np.uint64)[0] ^= 1
        self.assertTrue(np.isnan(other_nan[0]))
        key = _step_parameters.wing_step_parameters_key(
            self._changed("Ler_Gs_Cgs", with_nan)
        )
        self.assertEqual(
            _step_parameters.wing_step_parameters_key(
                self._changed("Ler_Gs_Cgs", np.array(with_nan, copy=True))
            ),
            key,
        )
        self.assertNotEqual(
            _step_parameters.wing_step_parameters_key(
                self._changed("Ler_Gs_Cgs", other_nan)
            ),
            key,
        )

    def test_changing_an_array_later_does_not_change_the_key(self) -> None:
        """Test that the key copies the values, so writing into an array after the key
        is made doesn't change it."""
        writable = np.array(self.parameters.angles_Gs_to_Wn_ixyz, copy=True)
        parameters = self._changed("angles_Gs_to_Wn_ixyz", writable)
        key = _step_parameters.wing_step_parameters_key(parameters)
        writable[0] += 1.0
        self.assertNotEqual(_step_parameters.wing_step_parameters_key(parameters), key)
        self.assertEqual(
            key, _step_parameters.wing_step_parameters_key(self.parameters)
        )


if __name__ == "__main__":
    unittest.main()
