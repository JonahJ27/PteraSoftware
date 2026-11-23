"""This module contains functions to collect the geometry values that the movement
classes produce, for comparing them against the stored reference in
tests/references/movement_reference.npz."""

import importlib
import inspect
from collections.abc import Callable

import numpy as np

import pterasoftware as ps

# The fixture modules whose movement fixtures are exercised, and the time steps and time
# step length at which every movement is evaluated. The time step length divides the
# fixtures' 0.1 s and 0.2 s periods, so generate_airplanes takes its periodic geometry
# shortcut where it can.
_FIXTURE_MODULES = (
    "core_wing_cross_section_movement_fixtures",
    "wing_cross_section_movement_fixtures",
    "aeroelastic_wing_cross_section_movement_fixtures",
    "core_wing_movement_fixtures",
    "wing_movement_fixtures",
    "aeroelastic_wing_movement_fixtures",
    "core_airplane_movement_fixtures",
    "airplane_movement_fixtures",
    "aeroelastic_airplane_movement_fixtures",
    "movement_fixtures",
    "free_flight_movement_fixtures",
    "flap_drive_fixtures",
)
_STEPS = (0, 1, 7, 23, 61)
_DELTA_TIME = 0.01


def _deformation_for(
    wing_movement: ps._core.CoreWingMovement,
    scale: float,
) -> np.ndarray:
    """Returns a deformation for a Wing's movement, with one row per WingCrossSection
    and a zero first row (a Wing's root WingCrossSection must keep zero angles).

    :param wing_movement: The Wing's movement class.
    :param scale: The largest absolute deformation angle. Its units are in degrees.
    :return: A (K,3) ndarray of floats representing the deformation angles. Its units
        are in degrees.
    """
    num = len(wing_movement.wing_cross_section_movements)
    deformation = np.linspace(-scale, scale, 3 * num, dtype=float).reshape(num, 3)
    deformation[0] = 0.0
    return deformation


def _record_wing_cross_section(
    values: dict[str, np.ndarray],
    key: str,
    wing_cross_section: ps.geometry.wing_cross_section.WingCrossSection,
) -> None:
    """Records a WingCrossSection's position and angles.

    :param values: The dict of recorded values, keyed by name.
    :param key: The name prefix for this WingCrossSection's values.
    :param wing_cross_section: The WingCrossSection.
    :return: None
    """
    values[f"{key}.Lp"] = np.array(wing_cross_section.Lp_Wcsp_Lpp, dtype=float)
    values[f"{key}.angles"] = np.array(
        wing_cross_section.angles_Wcsp_to_Wcs_ixyz, dtype=float
    )


def _record_wing(
    values: dict[str, np.ndarray], key: str, wing: ps.geometry.wing.Wing
) -> None:
    """Records a Wing's position and angles, and those of each of its WingCrossSections.

    :param values: The dict of recorded values, keyed by name.
    :param key: The name prefix for this Wing's values.
    :param wing: The Wing.
    :return: None
    """
    values[f"{key}.Ler"] = np.array(wing.Ler_Gs_Cgs, dtype=float)
    values[f"{key}.angles"] = np.array(wing.angles_Gs_to_Wn_ixyz, dtype=float)
    for index, wing_cross_section in enumerate(wing.wing_cross_sections):
        _record_wing_cross_section(values, f"{key}.wcs{index}", wing_cross_section)


def _record_parameters(
    values: dict[str, np.ndarray],
    key: str,
    parameters: (
        ps._step_parameters.WingCrossSectionStepParameters
        | ps._step_parameters.WingStepParameters
        | ps._step_parameters.AirplaneStepParameters
    ),
) -> None:
    """Records every array in a set of step parameters (a
    WingCrossSectionStepParameters, WingStepParameters, or AirplaneStepParameters),
    including each Wing's parameters in an AirplaneStepParameters.

    :param values: The dict of recorded values, keyed by name.
    :param key: The name prefix for these parameters' values.
    :param parameters: The step parameters.
    :return: None
    """
    if isinstance(parameters, ps._step_parameters.AirplaneStepParameters):
        values[f"{key}.Cg_GP1_CgP1"] = np.array(parameters.Cg_GP1_CgP1, dtype=float)
        for index, wing_parameters in enumerate(parameters.wings):
            _record_parameters(values, f"{key}.wing{index}", wing_parameters)
        return
    for field, value in zip(parameters._fields, parameters):
        values[f"{key}.{field}"] = np.array(value, dtype=float)


def _record_airplane(
    values: dict[str, np.ndarray], key: str, airplane: ps.geometry.airplane.Airplane
) -> None:
    """Records an Airplane's CG, and the values of each of its Wings after symmetry
    processing.

    :param values: The dict of recorded values, keyed by name.
    :param key: The name prefix for this Airplane's values.
    :param airplane: The Airplane.
    :return: None
    """
    values[f"{key}.Cg"] = np.array(airplane.Cg_GP1_CgP1, dtype=float)
    values[f"{key}.num_wings"] = np.array(len(airplane.wings), dtype=float)
    for index, wing in enumerate(airplane.wings):
        _record_wing(values, f"{key}.wing{index}", wing)


def _exercise(values: dict[str, np.ndarray], key: str, make: Callable) -> None:
    """Builds a fixture and, if it is a movement, records the geometry it produces at
    every reference time step, with and without deformation and overrides.

    :param values: The dict of recorded values, keyed by name.
    :param key: The name prefix for this fixture's values.
    :param make: The fixture's factory function, which takes no arguments.
    :return: None
    """
    movement = make()

    # Whether the movement has the per time step parameter methods. A package from
    # before they existed can still produce the object values, so the stored reference
    # can be checked against it.
    has_parameters = (
        hasattr(movement, "_generate_wing_cross_section_parameters_at_time_step")
        or hasattr(movement, "_generate_wing_parameters_at_time_step")
        or hasattr(movement, "_generate_airplane_parameters_at_time_step")
    )
    aeroelastic_wing_cross_section = (
        ps.movements.aeroelastic_wing_cross_section_movement.AeroelasticWingCrossSectionMovement
    )
    aeroelastic_wing = ps.movements.aeroelastic_wing_movement.AeroelasticWingMovement
    flap_driven_wing = (
        ps.movements.flap_driven_aeroelastic_wing_movement.FlapDrivenAeroelasticWingMovement
    )
    aeroelastic_airplane = (
        ps.movements.aeroelastic_airplane_movement.AeroelasticAirplaneMovement
    )

    if isinstance(movement, ps._core.CoreWingCrossSectionMovement):
        for step in _STEPS:
            _record_wing_cross_section(
                values,
                f"{key}.s{step}",
                movement.generate_wing_cross_section_at_time_step(step, _DELTA_TIME),
            )
            if has_parameters:
                _record_parameters(
                    values,
                    f"{key}.s{step}.parameters",
                    movement._generate_wing_cross_section_parameters_at_time_step(
                        step, _DELTA_TIME
                    ),
                )
            if isinstance(movement, aeroelastic_wing_cross_section):
                if has_parameters:
                    _record_parameters(
                        values,
                        f"{key}.s{step}.parameters.deformed",
                        movement._generate_wing_cross_section_parameters_at_time_step(
                            step,
                            _DELTA_TIME,
                            deformationAngles_Wcsp_to_Wcs_ixyz=np.array(
                                [0.3, -0.2, 0.1], dtype=float
                            ),
                        ),
                    )
                _record_wing_cross_section(
                    values,
                    f"{key}.s{step}.deformed",
                    movement.generate_wing_cross_section_at_time_step(
                        step,
                        _DELTA_TIME,
                        deformationAngles_Wcsp_to_Wcs_ixyz=np.array(
                            [0.3, -0.2, 0.1], dtype=float
                        ),
                    ),
                )
    elif isinstance(movement, ps._core.CoreWingMovement):
        if isinstance(movement, flap_driven_wing):
            for step in range(20):
                movement.record_flap_step(step, 30.0 * np.sin(0.4 * step), None)
        for step in _STEPS:
            _record_wing(
                values,
                f"{key}.s{step}",
                movement.generate_wing_at_time_step(step, _DELTA_TIME),
            )
            if has_parameters:
                _record_parameters(
                    values,
                    f"{key}.s{step}.parameters",
                    movement._generate_wing_parameters_at_time_step(step, _DELTA_TIME),
                )
            if isinstance(movement, aeroelastic_wing):
                deformation = _deformation_for(movement, 0.4)
                if has_parameters:
                    _record_parameters(
                        values,
                        f"{key}.s{step}.parameters.override",
                        movement._generate_wing_parameters_at_time_step(
                            step,
                            _DELTA_TIME,
                            deformationAngles_Wcsp_to_Wcs_ixyz=deformation,
                            overrideAngleX_Gs_to_Wn_ixyz=-11.5,
                        ),
                    )
                _record_wing(
                    values,
                    f"{key}.s{step}.deformed",
                    movement.generate_wing_at_time_step(
                        step,
                        _DELTA_TIME,
                        deformationAngles_Wcsp_to_Wcs_ixyz=deformation,
                    ),
                )
                _record_wing(
                    values,
                    f"{key}.s{step}.override",
                    movement.generate_wing_at_time_step(
                        step,
                        _DELTA_TIME,
                        deformationAngles_Wcsp_to_Wcs_ixyz=deformation,
                        overrideAngleX_Gs_to_Wn_ixyz=-11.5,
                    ),
                )
    elif isinstance(movement, ps._core.CoreAirplaneMovement):
        for step in _STEPS:
            _record_airplane(
                values,
                f"{key}.s{step}",
                movement.generate_airplane_at_time_step(step, _DELTA_TIME),
            )
            if has_parameters:
                _record_parameters(
                    values,
                    f"{key}.s{step}.parameters",
                    movement._generate_airplane_parameters_at_time_step(
                        step, _DELTA_TIME
                    ),
                )
            if isinstance(movement, aeroelastic_airplane):
                if has_parameters:
                    _record_parameters(
                        values,
                        f"{key}.s{step}.parameters.deformed",
                        movement._generate_airplane_parameters_at_time_step(
                            step,
                            _DELTA_TIME,
                            deformationAngles_Wcsp_to_Wcs_ixyz=[
                                _deformation_for(wing_movement, 0.2 * (index + 1))
                                for index, wing_movement in enumerate(
                                    movement.wing_movements
                                )
                            ],
                        ),
                    )
                _record_airplane(
                    values,
                    f"{key}.s{step}.deformed",
                    movement.generate_airplane_at_time_step(
                        step,
                        _DELTA_TIME,
                        deformationAngles_Wcsp_to_Wcs_ixyz=[
                            _deformation_for(wing_movement, 0.2 * (index + 1))
                            for index, wing_movement in enumerate(
                                movement.wing_movements
                            )
                        ],
                    ),
                )
        # The static and periodic geometry shortcuts of generate_airplanes.
        airplanes = movement.generate_airplanes(num_steps=70, delta_time=_DELTA_TIME)
        for step in _STEPS:
            _record_airplane(values, f"{key}.generate_airplanes{step}", airplanes[step])
    elif isinstance(movement, ps._core.CoreMovement):
        airplanes_by_movement = getattr(movement, "airplanes", None)
        if airplanes_by_movement is not None:
            for movement_index, airplanes in enumerate(airplanes_by_movement):
                for step, airplane in enumerate(airplanes):
                    if step in _STEPS:
                        _record_airplane(
                            values,
                            f"{key}.airplanes{movement_index}.s{step}",
                            airplane,
                        )
        if hasattr(movement, "_generate_airplane_parameters_at_time_step"):
            airplane_movement = movement.airplane_movements[0]

            # Record a flap history on every flap driven Wing's movement, so the
            # parameters use recorded flap angles rather than the base Wings' ones.
            for wing_movement in airplane_movement.wing_movements:
                if isinstance(wing_movement, flap_driven_wing):
                    for step in range(max(_STEPS) + 1):
                        wing_movement.record_flap_step(
                            step, 30.0 * np.sin(0.4 * step), None
                        )
            for step in _STEPS:
                if has_parameters:
                    _record_parameters(
                        values,
                        f"{key}.parameters.s{step}",
                        movement._generate_airplane_parameters_at_time_step(
                            0,
                            step,
                            deformationAngles_Wcsp_to_Wcs_ixyz=[
                                _deformation_for(wing_movement, 0.3)
                                for wing_movement in airplane_movement.wing_movements
                            ],
                        ),
                    )


def pack_values(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Packs a dict of recorded values into four arrays for storing in one file: the
    keys, each value's shape and its start in the flat values, and the flat values.

    Storing a few thousand small arrays separately costs more in file overhead than the
    values themselves, which is why they are packed.

    :param values: The dict of recorded values, keyed by name.
    :return: A dict of the four arrays, keyed "keys", "shapes", "offsets", and "flat".
    """
    keys = sorted(values)
    # Each row holds a value's number of dimensions, then its shape, padded with zeros.
    # Every recorded value has at most two dimensions.
    shapes = np.zeros((len(keys), 3), dtype=int)
    offsets = np.zeros(len(keys) + 1, dtype=int)
    for index, key in enumerate(keys):
        shape = values[key].shape
        assert len(shape) <= 2, key
        shapes[index, 0] = len(shape)
        shapes[index, 1 : 1 + len(shape)] = shape
        offsets[index + 1] = offsets[index] + values[key].size
    flat = np.concatenate([values[key].ravel() for key in keys]).astype(float)
    return {
        "keys": np.array(keys, dtype=str),
        "shapes": shapes,
        "offsets": offsets,
        "flat": flat,
    }


def unpack_values(packed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Unpacks values that pack_values packed.

    :param packed: A mapping holding the four arrays pack_values returns.
    :return: A dict of the recorded values, keyed by name.
    """
    values = {}
    for index, key in enumerate(packed["keys"]):
        start, end = packed["offsets"][index], packed["offsets"][index + 1]
        num_dimensions = packed["shapes"][index, 0]
        shape = tuple(int(n) for n in packed["shapes"][index, 1 : 1 + num_dimensions])
        values[str(key)] = packed["flat"][start:end].reshape(shape)
    return values


def collect_movement_reference_values() -> dict[str, np.ndarray]:
    """Collects the geometry values that every movement fixture produces at the
    reference time steps, keyed by fixture, time step, and value.

    Fixtures that aren't movements are recorded as nothing, and fixtures that need
    arguments are skipped.

    :return: A dict of the recorded values, each a ndarray of floats.
    """
    values: dict[str, np.ndarray] = {}
    for module_name in _FIXTURE_MODULES:
        module = importlib.import_module(f"tests.unit.fixtures.{module_name}")
        for name in sorted(dir(module)):
            if not name.startswith("make_"):
                continue
            make = getattr(module, name)

            # Skip fixtures that need arguments.
            if any(
                parameter.default is inspect.Parameter.empty
                for parameter in inspect.signature(make).parameters.values()
            ):
                continue
            _exercise(values, f"{module_name}.{name}", make)
    return values
