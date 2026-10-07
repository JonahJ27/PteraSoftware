"""Contains the FreeFlightMovement class."""

from __future__ import annotations

from typing import cast

from .. import _core, _parameter_validation, geometry
from . import airplane_movement as airplane_movement_mod
from . import aeroelastic_airplane_movement as aeroelastic_airplane_movement_mod
from .. import operating_point as operating_point_mod

from . import (
    free_flight_operating_point_movement as free_flight_operating_point_movement_mod,
)


class FreeFlightMovement(_core.CoreMovement):
    """A class used to contain a FreeFlightUnsteadyProblem's movement.

    In free flight, airplane geometry is prescribed (flapping, CG oscillation, etc.) but
    OperatingPoints are dynamically determined by the solver as it integrates rigid body
    dynamics at each time step. FreeFlightMovement pre generates all Airplanes upfront
    and provides a FreeFlightOperatingPointMovement whose mutable operating_points list
    the solver populates during simulation.

    The simulation is divided into two phases. During the prescribed phase, the solver
    uses the operating conditions from the initial OperatingPoint. During the free
    flight phase, the solver integrates rigid body dynamics using MuJoCo and creates new
    OperatingPoints from the resulting state at each time step.
    """

    __slots__ = (
        "_prescribed_num_steps",
        "_free_num_steps",
        "_airplanes",
    )

    def __init__(
        self,
        airplane_movements: list[airplane_movement_mod.AirplaneMovement],
        operating_point_movement: free_flight_operating_point_movement_mod.FreeFlightOperatingPointMovement,
        delta_time: float | int,
        prescribed_num_steps: int,
        free_num_steps: int,
        max_wake_rows: int | None = None,
    ) -> None:
        """The initialization method.

        This method checks that all Wings maintain their symmetry type across all time
        steps. See the WingMovement class documentation for more details on this
        requirement. See the Wing class documentation for more information on symmetry
        types.

        :param airplane_movements: A list of the AirplaneMovements associated with each
            of the FreeFlightUnsteadyProblem's Airplanes.
        :param operating_point_movement: A FreeFlightOperatingPointMovement holding the
            initial OperatingPoint. The solver populates its mutable operating_points
            list during simulation.
        :param delta_time: The time, in seconds, between each time step. It must be a
            positive number (int or float). It will be converted internally to a float.
        :param prescribed_num_steps: The number of prescribed flight time steps to
            simulate before the free flight time steps. It must be a positive int.
        :param free_num_steps: The number of free flight time steps to simulate after
            the prescribed time steps. It must be a positive int.
        :param max_wake_rows: The maximum number of chordwise wake ring vortex rows per
            Wing. Must be a positive int if set. The default is None (no truncation).
        :return: None
        """
        # Validate that every element is an AirplaneMovement. CoreMovement.__init__()
        # validates at the Core level, but FreeFlightMovement enforces the concrete
        # type.
        for airplane_movement in airplane_movements:
            if not isinstance(
                airplane_movement,
                airplane_movement_mod.AirplaneMovement,
            ):
                raise TypeError(
                    "Every element in airplane_movements must be an "
                    "AirplaneMovement."
                )

        # Validate that operating_point_movement is a FreeFlightOperatingPointMovement.
        if not isinstance(
            operating_point_movement,
            free_flight_operating_point_movement_mod.FreeFlightOperatingPointMovement,
        ):
            raise TypeError(
                "operating_point_movement must be a "
                "FreeFlightOperatingPointMovement."
            )

        # Validate and store the phase step counts.
        prescribed_num_steps = _parameter_validation.int_in_range_return_int(
            prescribed_num_steps,
            "prescribed_num_steps",
            min_val=1,
            min_inclusive=True,
        )
        free_num_steps = _parameter_validation.int_in_range_return_int(
            free_num_steps,
            "free_num_steps",
            min_val=1,
            min_inclusive=True,
        )
        num_steps = prescribed_num_steps + free_num_steps

        # --- Initialize CoreMovement ---
        super().__init__(
            airplane_movements=airplane_movements,
            operating_point_movement=operating_point_movement,
            delta_time=delta_time,
            num_steps=num_steps,
            max_wake_rows=max_wake_rows,
        )

        # --- Store FreeFlightMovement only attributes ---
        self._prescribed_num_steps = prescribed_num_steps
        self._free_num_steps = free_num_steps

        # --- Batch generate Airplanes ---
        # Generate a list of lists of Airplanes that are the steps through each
        # AirplaneMovement. The first index identifies the AirplaneMovement, and the
        # second index identifies the time step.
        airplanes_temp: list[list[geometry.airplane.Airplane]] = []
        for airplane_movement in self.airplane_movements:
            airplanes_temp.append(
                airplane_movement.generate_airplanes(
                    num_steps=self._num_steps, delta_time=self._delta_time
                )
            )

        # Validate that all Wings maintain their symmetry type across all time steps.
        for airplane_movement_id, airplane_list in enumerate(airplanes_temp):
            # Get the base Airplane (first time step).
            base_airplane = airplane_list[0]

            # Store the symmetry types of the base Wings.
            base_wing_symmetry_types = []
            for wing in base_airplane.wings:
                base_wing_symmetry_types.append(wing.symmetry_type)

            # Validate all subsequent time steps.
            for step_id, airplane in enumerate(airplane_list):
                # Check that Wings maintain their symmetry types.
                for wing_id, wing in enumerate(airplane.wings):
                    base_symmetry_type = base_wing_symmetry_types[wing_id]
                    if wing.symmetry_type != base_symmetry_type:
                        raise ValueError(
                            f"Wing {wing_id} in AirplaneMovement "
                            f"{airplane_movement_id} changed from type "
                            f"{base_symmetry_type} symmetry at time step 0 "
                            f"to type {wing.symmetry_type} symmetry at time "
                            f"step {step_id}. Wings cannot undergo motion "
                            f"that changes their symmetry type. This happens "
                            f"when a symmetric Wing moves such that its "
                            f"symmetry plane is no longer coincident with "
                            f"the wing axes' yz plane or vice versa."
                        )

        # Store as tuple of tuples to prevent external mutation.
        self._airplanes: tuple[tuple[geometry.airplane.Airplane, ...], ...] = tuple(
            tuple(airplane_list) for airplane_list in airplanes_temp
        )

    # --- Immutable: read only properties ---
    @property
    def operating_point_movement(
        self,
    ) -> free_flight_operating_point_movement_mod.FreeFlightOperatingPointMovement:
        assert isinstance(
            self._operating_point_movement,
            free_flight_operating_point_movement_mod.FreeFlightOperatingPointMovement,
        )
        return self._operating_point_movement

    @property
    def airplane_movements(
        self,
    ) -> tuple[airplane_movement_mod.AirplaneMovement, ...]:
        return cast(
            tuple[airplane_movement_mod.AirplaneMovement, ...],
            self._airplane_movements,
        )

    @property
    def prescribed_num_steps(self) -> int:
        return self._prescribed_num_steps

    @property
    def free_num_steps(self) -> int:
        return self._free_num_steps

    @property
    def airplanes(self) -> tuple[tuple[geometry.airplane.Airplane, ...], ...]:
        return self._airplanes





class FreeFlightAeroelasticMovement(_core.CoreMovement):
    """A class used to contain a FreeFlightAeroelasticUnsteadyProblem's movement.

    In free flight, airplane geometry is prescribed (flapping, CG oscillation, etc.) but
    OperatingPoints are dynamically determined by the solver as it integrates rigid body
    dynamics at each time step. FreeFlightMovement pre generates all Airplanes upfront
    and provides a FreeFlightOperatingPointMovement whose mutable operating_points list
    the solver populates during simulation.

    The simulation is divided into two phases. During the prescribed phase, the solver
    uses the operating conditions from the initial OperatingPoint. During the free
    flight phase, the solver integrates rigid body dynamics using MuJoCo and creates new
    OperatingPoints from the resulting state at each time step.
    """

    __slots__ = (
        "_prescribed_num_steps",
        "_free_num_steps",
        "_airplanes",
        "_operating_points",
    )

    def __init__(
        self,
        airplane_movements: list[aeroelastic_airplane_movement_mod.AeroelasticAirplaneMovement],
        operating_point_movement: free_flight_operating_point_movement_mod.FreeFlightOperatingPointMovement,
        delta_time: float | int,
        prescribed_num_steps: int,
        free_num_steps: int,
        max_wake_rows: int | None = None,
    ) -> None:
        """The initialization method.

        This method checks that all Wings maintain their symmetry type across all time
        steps. See the WingMovement class documentation for more details on this
        requirement. See the Wing class documentation for more information on symmetry
        types.

        :param airplane_movements: A list of the AirplaneMovements associated with each
            of the FreeFlightUnsteadyProblem's Airplanes.
        :param operating_point_movement: A FreeFlightOperatingPointMovement holding the
            initial OperatingPoint. The solver populates its mutable operating_points
            list during simulation.
        :param delta_time: The time, in seconds, between each time step. It must be a
            positive number (int or float). It will be converted internally to a float.
        :param prescribed_num_steps: The number of prescribed flight time steps to
            simulate before the free flight time steps. It must be a positive int.
        :param free_num_steps: The number of free flight time steps to simulate after
            the prescribed time steps. It must be a positive int.
        :param max_wake_rows: The maximum number of chordwise wake ring vortex rows per
            Wing. Must be a positive int if set. The default is None (no truncation).
        :return: None
        """
        # Validate that every element is an AirplaneMovement. CoreMovement.__init__()
        # validates at the Core level, but FreeFlightMovement enforces the concrete
        # type.
        for airplane_movement in airplane_movements:
            if not isinstance(
                airplane_movement,
                aeroelastic_airplane_movement_mod.AeroelasticAirplaneMovement,
            ):
                raise TypeError(
                    "Every element in aeroelastic_airplane_movements must be an "
                    "AeroelasticAirplaneMovement."
                )

        # Validate that operating_point_movement is a FreeFlightOperatingPointMovement.
        if not isinstance(
            operating_point_movement,
            free_flight_operating_point_movement_mod.FreeFlightOperatingPointMovement,
        ):
            raise TypeError(
                "operating_point_movement must be a "
                "FreeFlightOperatingPointMovement."
            )

        # Validate and store the phase step counts.
        prescribed_num_steps = _parameter_validation.int_in_range_return_int(
            prescribed_num_steps,
            "prescribed_num_steps",
            min_val=1,
            min_inclusive=True,
        )
        free_num_steps = _parameter_validation.int_in_range_return_int(
            free_num_steps,
            "free_num_steps",
            min_val=1,
            min_inclusive=True,
        )
        num_steps = prescribed_num_steps + free_num_steps

        # --- Initialize CoreMovement ---
        super().__init__(
            airplane_movements=airplane_movements,
            operating_point_movement=operating_point_movement,
            delta_time=delta_time,
            num_steps=num_steps,
            max_wake_rows=max_wake_rows,
        )

        # --- Store FreeFlightMovement only attributes ---
        self._prescribed_num_steps = prescribed_num_steps
        self._free_num_steps = free_num_steps

        # --- Batch generate Airplanes ---
        # Generate a list of lists of Airplanes that are the steps through each
        # AirplaneMovement. The first index identifies the AirplaneMovement, and the
        # second index identifies the time step.
        airplanes_temp: list[list[geometry.airplane.Airplane]] = []
        for airplane_movement in self.airplane_movements:
            airplanes_temp.append(
                airplane_movement.generate_airplanes(
                    num_steps=self._num_steps, delta_time=self._delta_time
                )
            )

        # Validate that all Wings maintain their symmetry type across all time steps.
        for airplane_movement_id, airplane_list in enumerate(airplanes_temp):
            # Get the base Airplane (first time step).
            base_airplane = airplane_list[0]

            # Store the symmetry types of the base Wings.
            base_wing_symmetry_types = []
            for wing in base_airplane.wings:
                base_wing_symmetry_types.append(wing.symmetry_type)

            # Validate all subsequent time steps.
            for step_id, airplane in enumerate(airplane_list):
                # Check that Wings maintain their symmetry types.
                for wing_id, wing in enumerate(airplane.wings):
                    base_symmetry_type = base_wing_symmetry_types[wing_id]
                    if wing.symmetry_type != base_symmetry_type:
                        raise ValueError(
                            f"Wing {wing_id} in AirplaneMovement "
                            f"{airplane_movement_id} changed from type "
                            f"{base_symmetry_type} symmetry at time step 0 "
                            f"to type {wing.symmetry_type} symmetry at time "
                            f"step {step_id}. Wings cannot undergo motion "
                            f"that changes their symmetry type. This happens "
                            f"when a symmetric Wing moves such that its "
                            f"symmetry plane is no longer coincident with "
                            f"the wing axes' yz plane or vice versa."
                        )

        # Store as tuple of tuples to prevent external mutation.
        self._airplanes: tuple[tuple[geometry.airplane.Airplane, ...], ...] = tuple(
            tuple(airplane_list) for airplane_list in airplanes_temp
        )
        # --- Batch generate OperatingPoints ---
        # OperatingPoints are prescribed in aeroelastic simulations, so generate them
        # all upfront.
        operating_points_list = operating_point_movement.generate_operating_points(
            num_steps=self._num_steps, delta_time=self._delta_time
        )
        self._operating_points: tuple[operating_point_mod.OperatingPoint, ...] = tuple(
            operating_points_list
        )

    # --- Immutable: read only properties ---
    @property
    def operating_point_movement(
        self,
    ) -> free_flight_operating_point_movement_mod.FreeFlightOperatingPointMovement:
        assert isinstance(
            self._operating_point_movement,
            free_flight_operating_point_movement_mod.FreeFlightOperatingPointMovement,
        )
        return self._operating_point_movement

    @property
    def airplane_movements(
        self,
    ) -> tuple[airplane_movement_mod.AirplaneMovement, ...]:
        return cast(
            tuple[airplane_movement_mod.AirplaneMovement, ...],
            self._airplane_movements,
        )

    @property
    def prescribed_num_steps(self) -> int:
        return self._prescribed_num_steps

    @property
    def free_num_steps(self) -> int:
        return self._free_num_steps

    @property
    def airplanes(self) -> tuple[tuple[geometry.airplane.Airplane, ...], ...]:
        return self._airplanes
    @property
    def operating_points(self) -> tuple[operating_point_mod.OperatingPoint, ...]:
        return self._operating_points
    def generate_airplane_at_time_step(
            self,
            airplane_movement_index: int,
            step: int,
            deformationAngles_Wcsp_to_Wcs_ixyz: list[np.ndarray | None] | None = None,
        ) -> geometry.airplane.Airplane:
            """Creates the Airplane at a single time step for a given
            AeroelasticAirplaneMovement, applying deformation from the solver's structural
            response.
    
            This is the method the aeroelastic solver calls at each time step to get the
            deformed Airplane geometry.
    
            :param airplane_movement_index: The index of the AeroelasticAirplaneMovement in
                this AeroelasticMovement's airplane_movements tuple.
            :param step: The time step index. Must be a non negative int.
            :param deformationAngles_Wcsp_to_Wcs_ixyz: A list of ndarrays of floats, one per
                Wing, each with one row per WingCrossSection in that Wing. Each row is a
                (3,) deformation angle vector that perturbs the corresponding
                WingCrossSection's angles_Wcsp_to_Wcs_ixyz, using an intrinsic xy'z"
                sequence. The units are in degrees. When None, no deformation is applied.
                The default is None.
            :return: The Airplane at this time step, with structural deformation applied if
                provided.
            """
            return self.airplane_movements[
                airplane_movement_index
            ].generate_airplane_at_time_step(
                step,
                self._delta_time,
                deformationAngles_Wcsp_to_Wcs_ixyz=deformationAngles_Wcsp_to_Wcs_ixyz,
            )
    


