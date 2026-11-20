"""Contains the FreeFlightFlapDrivenAeroelasticUnsteadyProblem class."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from . import approximations as approximations_mod
from . import flap_drive as flap_drive_mod
from . import flap_driven_aeroelastic_unsteady_problem as flap_problem_mod
from . import geometry
from . import operating_point as operating_point_mod
from . import problems
from .movements import free_flight_movement as free_flight_movement_mod


class FreeFlightFlapDrivenAeroelasticUnsteadyProblem(
    flap_problem_mod._FlapDriveCoupling, problems.FreeFlightAeroelasticUnsteadyProblem
):
    """A subclass of FreeFlightAeroelasticUnsteadyProblem whose Wings' flap angles come
    from FlapDrives instead of from the movement's prescribed oscillation.

    VALIDATED CORRECT

    It combines the FlapDrives of FlapDrivenAeroelasticUnsteadyProblem with the free
    flight of FreeFlightAeroelasticUnsteadyProblem: the Wings are flapped by actuators
    through their transmissions, they deform under the torsional spring model, and the
    vehicle's rigid body state comes from MuJoCo. Both share the flap drive logic
    through the _FlapDriveCoupling mixin, so a FlapDrive behaves the same way in both.
    See FlapDrivenAeroelasticUnsteadyProblem for what the FlapDrives do each time step,
    their histories, and how their references set the cycle averaged results. It is
    solved by the same FreeFlightAeroelasticUnsteadyRingVortexLatticeMethodSolver.

    **The order of one time step:**

    The solver solves time step s, then calls initialize_next_problem(solver, s) to
    build time step s + 1, which happens in three stages. First, the FlapDrives advance:
    each one integrates its actuator over the interval with the fluid torque on its
    Wings from time step s's solution held constant, and records the flap angle at s + 1
    and the flap acceleration over the interval. Second, the torsional springs advance
    over the same interval, forced by those flap accelerations, and time step s + 1's
    Airplane is built at the recorded flap angles with the new deformation. Third, the
    rigid body advances: during the prescribed phase it coasts at its initial state, and
    during the free flight phase the strongly coupled sub-iteration drives the
    aerodynamic loads on that Airplane and the rigid body state to agreement. The first
    two stages run once per time step, before the sub-iteration, and the Airplane's
    geometry is held fixed while it runs.

    **Coupling:**

    The rigid body is coupled strongly: within each time step, the loads and the body
    state are iterated to agreement. The FlapDrives and the torsional springs are
    coupled loosely: they use the fluid torque from the time step just solved, so the
    fluid's response to their motion acts one time step late. Because they run once per
    time step, no flap state has to be saved and restored inside the sub-iteration. The
    planned fix for the lag, which would merge the flap into the rigid body's sub-
    iteration, is described in docs/FLAP_DRIVE.md under "Planned: Strongly Coupling the
    Flap".

    **What the rigid body dynamics don't include:**

    MuJoCo models the vehicle as one rigid body with the constant mass and inertia given
    to this problem. The FlapDrives' dynamics don't reach it. So the rigid body feels
    the aerodynamic loads on the flapping Wings, gravity, and any external loads, but
    not the following:

    - The actuators' reaction torques on the airframe. Each actuator pushes back on the
    airframe with its force (and a spinning rotor adds a gyroscopic torque), which is
    not applied. For a left and right pair of FlapDrives the reactions largely cancel,
    but not when one saturates or they differ.

    - The flapping Wings' inertial loads on the airframe. Swinging the Wings' mass back
    and forth shakes a real vehicle at the flapping frequency, which is not applied.

    - The changing mass distribution. The vehicle's CG and inertia stay fixed while the
    Wings flap, so give mass and I_BP1_CgP1 that include the Wings and the actuators,
    for example at the mean flap angle.

    docs/FLAP_DRIVE.md describes how these could be added later.

    **Using a blade element theory model:**

    The OperatingPoint that a FlapDrive's aerodynamic model receives in its
    AerodynamicModelInputs is the vehicle's actual state from the rigid body dynamics at
    the time step just solved, so a blade element theory model passed as
    feedforward_aerodynamic sees the airspeed, the angle of attack, and the body rates
    change as the vehicle flies.
    """

    __slots__ = flap_problem_mod._FLAP_DRIVE_SLOTS

    def __init__(
        self,
        movement: free_flight_movement_mod.FreeFlightAeroelasticMovement,
        wing_density: float,
        spring_constant_rad: float,
        damping_constant_rad: float,
        mass: float | int,
        I_BP1_CgP1: np.ndarray | Sequence[Sequence[float | int]],
        flap_drives: Sequence[flap_drive_mod.FlapDrive],
        external_loads_fn: (
            Callable[
                [
                    operating_point_mod.OperatingPoint,
                    geometry.airplane.Airplane,
                ],
                tuple[np.ndarray, np.ndarray],
            ]
            | None
        ) = None,
        k_max: int = 20,
        integrator: str = "RK4",
        extra_xml: dict[str, str] | None = None,
        mujoco_assets: dict[str, bytes] | None = None,
        step_discards: int = 5,
        use_analytical_spring_solver: bool | np.bool = True,
        approximations: approximations_mod.Approximations | None = None,
    ) -> None:
        """The initialization method.

        See FreeFlightAeroelasticUnsteadyProblem's initialization method for the
        parameters that share its names, and FlapDrivenAeroelasticUnsteadyProblem's for
        flap_drives.

        :param movement: A FreeFlightAeroelasticMovement containing the prescribed
            motion and the initial OperatingPoint. The Wings that a FlapDrive flaps must
            be moved by FlapDrivenAeroelasticWingMovements in its AirplaneMovement.
            Their recorded flap states are cleared and their flap periods are set from
            the FlapDrives' references, so use a new movement for each problem.
        :param wing_density: The mass per unit span area of the wing (kg/m^2), for the
            torsional spring model and, unless a FlapDrive gives its own, the Wings'
            inertia about the flap axis.
        :param spring_constant_rad: The torsional spring stiffness (N*m/rad).
        :param damping_constant_rad: The torsional damping coefficient (N*m*s/rad).
        :param mass: The vehicle's mass (kg), including the Wings and the actuators. See
            FreeFlightAeroelasticUnsteadyProblem.
        :param I_BP1_CgP1: The vehicle's inertia matrix (kg*m^2), including the Wings
            and the actuators, which stays constant. See
            FreeFlightAeroelasticUnsteadyProblem.
        :param flap_drives: A sequence of FlapDrives. Can be a tuple or list. It must be
            non empty. Each FlapDrive's wing_indices index the first Airplane's Wings,
            every one of those Wings must be moved by a
            FlapDrivenAeroelasticWingMovement, and no Wing can belong to two FlapDrives.
            Each FlapDrive's reference must cover the whole simulation.
        :param external_loads_fn: See FreeFlightAeroelasticUnsteadyProblem. The default
            is None.
        :param k_max: See FreeFlightAeroelasticUnsteadyProblem. The default is 20.
        :param integrator: See FreeFlightAeroelasticUnsteadyProblem. The default is
            "RK4".
        :param extra_xml: See FreeFlightAeroelasticUnsteadyProblem. The default is None.
        :param mujoco_assets: See FreeFlightAeroelasticUnsteadyProblem. The default is
            None.
        :param step_discards: The number of initial time steps to discard for numerical
            stability (there are inconsistent startup effects from the UVLM solver).
            During these time steps, the solver will run but the results will not be
            applied to the deformation of the Wings, and the FlapDrives act as if the
            fluid applied no torque. It must be a non negative int. The default is 5.
        :param use_analytical_spring_solver: See FlapDrivenAeroelasticUnsteadyProblem.
            Can be a bool or a numpy bool. The default is True.
        :param approximations: See FreeFlightAeroelasticUnsteadyProblem. The default is
            None, which turns every approximation off.
        :return: None
        """
        if not isinstance(
            movement, free_flight_movement_mod.FreeFlightAeroelasticMovement
        ):
            raise TypeError("movement must be a FreeFlightAeroelasticMovement.")

        # Validate the FlapDrives and record their starting flap angles and periods on
        # the movements, before the parent builds the first time step's Airplane and
        # reads the movement's periods.
        validated_flap_drives, initial_states = self._prepare_flap_drives(
            movement, flap_drives
        )

        super().__init__(
            movement=movement,
            wing_density=wing_density,
            spring_constant_rad=spring_constant_rad,
            damping_constant_rad=damping_constant_rad,
            mass=mass,
            I_BP1_CgP1=I_BP1_CgP1,
            external_loads_fn=external_loads_fn,
            k_max=k_max,
            integrator=integrator,
            extra_xml=extra_xml,
            mujoco_assets=mujoco_assets,
            step_discards=step_discards,
            use_analytical_spring_solver=use_analytical_spring_solver,
            approximations=approximations,
        )

        # Seed the FlapDrives' histories from the first time step.
        self._initialize_flap_drives(validated_flap_drives, initial_states)
