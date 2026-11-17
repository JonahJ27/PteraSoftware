"""This module contains functions to create the flap driven problems that the flap drive
integration tests solve."""

import numpy as np

import pterasoftware as ps
from pterasoftware import flap_drive

# These are the parameters of the sine flap that the fixtures' FlapDrives track and that
# the prescribed problem fixture flaps through directly: the flap angle is
# FLAP_AMPLITUDE
# * sin(2 * pi * time / FLAP_PERIOD + FLAP_PHASE), in degrees.
FLAP_AMPLITUDE = 15.0
FLAP_PERIOD = 1.0
FLAP_PHASE = 169.0

# This is the fixtures' time step (s), which gives 20 time steps per flap.
DELTA_TIME = 0.05

# These are the fixtures' torsional spring parameters. The structural results are taken
# after STEP_DISCARDS time steps, so most of a 20 time step run deforms.
WING_DENSITY = 0.5
SPRING_CONSTANT_RAD = 20.0
DAMPING_CONSTANT_RAD = 1.0
STEP_DISCARDS = 2

# These are the tracking FlapDrive fixture's actuator inertia (kg*m^2) and feedback
# gains (N*m/rad and N*m*s/rad) for a FlapDrive that flaps both Wings.
ACTUATOR_INERTIA = 1e-3
PROPORTIONAL_GAIN = 400.0
DERIVATIVE_GAIN = 20.0


def _make_airplane() -> ps.geometry.airplane.Airplane:
    """Builds the Airplane that every fixture flaps: a symmetric main Wing with two
    strips per side, whose root is off the plane of symmetry so that the Airplane splits
    it into the main Wing (index 0) and a reflected Wing (index 1).

    :return: The Airplane.
    """
    airfoil = ps.geometry.airfoil.Airfoil(name="naca2412")
    wing_cross_sections = [
        ps.geometry.wing_cross_section.WingCrossSection(
            num_spanwise_panels=num_spanwise_panels,
            chord=chord,
            Lp_Wcsp_Lpp=(0.0, span, 0.0),
            angles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
            control_surface_symmetry_type="symmetric",
            control_surface_hinge_point=0.75,
            control_surface_deflection=0.0,
            spanwise_spacing=spanwise_spacing,
            airfoil=airfoil,
        )
        for num_spanwise_panels, chord, span, spanwise_spacing in (
            (1, 1.0, 0.0, "uniform"),
            (1, 0.8, 0.5, "uniform"),
            (None, 0.6, 0.5, None),
        )
    ]
    main_wing = ps.geometry.wing.Wing(
        wing_cross_sections=wing_cross_sections,
        name="Main Wing",
        Ler_Gs_Cgs=(0.0, 0.1, 0.0),
        angles_Gs_to_Wn_ixyz=(0.0, 0.0, 0.0),
        symmetric=True,
        mirror_only=False,
        symmetryNormal_G=(0.0, 1.0, 0.0),
        symmetryPoint_G_Cg=(0.0, 0.0, 0.0),
        explode_into_strips=False,
        num_chordwise_panels=2,
        chordwise_spacing="uniform",
    )
    return ps.geometry.airplane.Airplane(
        wings=[main_wing],
        name="Flap Drive Test Airplane",
        Cg_GP1_CgP1=(0.0, 0.0, 0.0),
        weight=0.0,
    )


def _make_airplane_movement(
    flap_driven: bool,
) -> ps.movements.aeroelastic_airplane_movement.AeroelasticAirplaneMovement:
    """Builds the AeroelasticAirplaneMovement for the fixture Airplane.

    :param flap_driven: If True, each Wing is moved by a
        FlapDrivenAeroelasticWingMovement, so a FlapDrive must flap it. If False, each
        Wing is moved by an AeroelasticWingMovement that prescribes the sine flap.
    :return: The AeroelasticAirplaneMovement.
    """
    airplane = _make_airplane()
    wing_movements: list[
        ps.movements.aeroelastic_wing_movement.AeroelasticWingMovement
        | ps.movements.wing_movement.WingMovement
    ] = []
    for wing in airplane.wings:
        wing_cross_section_movements = [
            ps.movements.aeroelastic_wing_cross_section_movement.AeroelasticWingCrossSectionMovement(
                base_wing_cross_section=wing_cross_section
            )
            for wing_cross_section in wing.wing_cross_sections
        ]
        if flap_driven:
            wing_movements.append(
                ps.movements.flap_driven_aeroelastic_wing_movement.FlapDrivenAeroelasticWingMovement(
                    base_wing=wing,
                    wing_cross_section_movements=wing_cross_section_movements,
                )
            )
        else:
            wing_movements.append(
                ps.movements.aeroelastic_wing_movement.AeroelasticWingMovement(
                    base_wing=wing,
                    wing_cross_section_movements=wing_cross_section_movements,
                    ampAngles_Gs_to_Wn_ixyz=(FLAP_AMPLITUDE, 0.0, 0.0),
                    periodAngles_Gs_to_Wn_ixyz=(FLAP_PERIOD, 0.0, 0.0),
                    spacingAngles_Gs_to_Wn_ixyz=("sine", "sine", "sine"),
                    phaseAngles_Gs_to_Wn_ixyz=(FLAP_PHASE, 0.0, 0.0),
                )
            )
    return ps.movements.aeroelastic_airplane_movement.AeroelasticAirplaneMovement(
        base_airplane=airplane, wing_movements=wing_movements
    )


def _make_operating_point() -> ps.operating_point.OperatingPoint:
    """Builds the fixtures' OperatingPoint: 10 m/s, level, with no gravity.

    :return: The OperatingPoint.
    """
    return ps.operating_point.OperatingPoint(
        rho=1.225, vCg__E=10.0, alpha=0.0, beta=0.0, externalFX_W=0.0, nu=15.06e-6
    )


def make_aeroelastic_movement(
    flap_driven: bool, num_steps: int = 20
) -> ps.movements.aeroelastic_movement.AeroelasticMovement:
    """This method makes a fixture that is an AeroelasticMovement for the fixture
    Airplane.

    :param flap_driven: If True, the Wings are moved by
        FlapDrivenAeroelasticWingMovements. If False, they flap through the prescribed
        sine flap.
    :param num_steps: The number of time steps. The default is 20, one flap.
    :return aeroelastic_movement: AeroelasticMovement This is the movement.
    """
    aeroelastic_movement = ps.movements.aeroelastic_movement.AeroelasticMovement(
        airplane_movements=[_make_airplane_movement(flap_driven)],
        operating_point_movement=ps.movements.operating_point_movement.OperatingPointMovement(
            base_operating_point=_make_operating_point()
        ),
        delta_time=DELTA_TIME,
        num_steps=num_steps,
    )

    return aeroelastic_movement


def make_sine_reference(duration: float = 3.0) -> flap_drive.FlapReference:
    """This method makes a fixture that is a FlapReference for a direct drive that
    follows the sine flap.

    :param duration: The time the reference covers (s). The default is 3.0.
    :return sine_reference: FlapReference This is the FlapReference, built through an
        identity transmission, so its actuator positions are the flap angles in radians.
    """
    times = np.arange(int(round(duration / 1e-4)) + 1, dtype=float) * 1e-4
    amplitude_rad = np.deg2rad(FLAP_AMPLITUDE)
    frequency_rad = 2.0 * np.pi / FLAP_PERIOD
    phases_rad = frequency_rad * times + np.deg2rad(FLAP_PHASE)
    sine_reference = flap_drive.FlapReference.from_flap_trajectory(
        flap_drive.FlapTransmission.identity(),
        times,
        amplitude_rad * np.sin(phases_rad),
        amplitude_rad * frequency_rad * np.cos(phases_rad),
        -amplitude_rad * frequency_rad**2 * np.sin(phases_rad),
        period=FLAP_PERIOD,
    )

    return sine_reference


def make_tracking_drive(
    wing_indices: tuple[int, ...] = (0, 1),
    share: float = 1.0,
    torque_limit: float | None = None,
) -> flap_drive.FlapDrive:
    """This method makes a fixture that is a direct drive FlapDrive that tracks the sine
    flap with feedforward plus feedback.

    :param wing_indices: The indices of the Wings it flaps. The default is (0, 1).
    :param share: The fraction of the full actuator it is: its actuator inertia and its
        feedback gains are this fraction of a FlapDrive that flaps both Wings. Two
        FlapDrives with a share of 0.5, one per Wing, together match one FlapDrive with
        a share of 1.0 for both Wings. The default is 1.0.
    :param torque_limit: None, or the largest magnitude of the actuator's torque (N*m).
        The default is None, which leaves the actuator unlimited.
    :return tracking_drive: FlapDrive This is the FlapDrive.
    """

    def actuator(inputs: flap_drive.FlapDriveInputs) -> float:
        force = (
            inputs.feedforward_force
            + share
            * PROPORTIONAL_GAIN
            * (inputs.desired_actuator_position - inputs.actuator_position)
            + share
            * DERIVATIVE_GAIN
            * (inputs.desired_actuator_velocity - inputs.actuator_velocity)
        )
        if torque_limit is not None:
            force = float(np.clip(force, -torque_limit, torque_limit))
        return force

    tracking_drive = flap_drive.FlapDrive(
        wing_indices=wing_indices,
        transmission=flap_drive.FlapTransmission.identity(),
        actuator=actuator,
        actuator_inertia=share * ACTUATOR_INERTIA,
        reference=make_sine_reference(),
        use_position_lookup=False,
    )

    return tracking_drive


def make_flap_driven_problem(
    flap_drives: list[flap_drive.FlapDrive] | None = None,
    num_steps: int = 20,
    use_analytical_spring_solver: bool = True,
) -> ps.flap_driven_aeroelastic_unsteady_problem.FlapDrivenAeroelasticUnsteadyProblem:
    """This method makes a fixture that is a FlapDrivenAeroelasticUnsteadyProblem.

    :param flap_drives: The FlapDrives, or None for one tracking FlapDrive that flaps
        both Wings. The default is None.
    :param num_steps: The number of time steps. The default is 20, one flap.
    :param use_analytical_spring_solver: Whether the torsional springs are solved
        analytically. The default is True.
    :return flap_driven_problem: FlapDrivenAeroelasticUnsteadyProblem This is the
        problem.
    """
    flap_driven_problem = ps.flap_driven_aeroelastic_unsteady_problem.FlapDrivenAeroelasticUnsteadyProblem(
        movement=make_aeroelastic_movement(True, num_steps),
        wing_density=WING_DENSITY,
        spring_constant_rad=SPRING_CONSTANT_RAD,
        damping_constant_rad=DAMPING_CONSTANT_RAD,
        flap_drives=(
            flap_drives if flap_drives is not None else [make_tracking_drive()]
        ),
        step_discards=STEP_DISCARDS,
        use_analytical_spring_solver=use_analytical_spring_solver,
    )

    return flap_driven_problem


def make_prescribed_problem(
    num_steps: int = 20,
) -> ps.problems.AeroelasticUnsteadyProblem:
    """This method makes a fixture that is an AeroelasticUnsteadyProblem that flaps the
    fixture Airplane through the prescribed sine flap, with the same structure as the
    flap driven problem fixture.

    :param num_steps: The number of time steps. The default is 20, one flap.
    :return prescribed_problem: AeroelasticUnsteadyProblem This is the problem.
    """
    prescribed_problem = ps.problems.AeroelasticUnsteadyProblem(
        movement=make_aeroelastic_movement(False, num_steps),
        wing_density=WING_DENSITY,
        spring_constant_rad=SPRING_CONSTANT_RAD,
        damping_constant_rad=DAMPING_CONSTANT_RAD,
        step_discards=STEP_DISCARDS,
    )

    return prescribed_problem


def make_free_flight_flap_driven_problem(
    prescribed_num_steps: int = 4,
    free_num_steps: int = 12,
    use_analytical_spring_solver: bool = True,
) -> (
    ps.free_flight_flap_driven_aeroelastic_unsteady_problem.FreeFlightFlapDrivenAeroelasticUnsteadyProblem
):
    """This method makes a fixture that is a
    FreeFlightFlapDrivenAeroelasticUnsteadyProblem, with one tracking FlapDrive for both
    Wings, a 1.0 kg vehicle, and no gravity.

    :param prescribed_num_steps: The number of prescribed time steps. The default is 4.
    :param free_num_steps: The number of free flight time steps. The default is 12.
    :param use_analytical_spring_solver: Whether the torsional springs are solved
        analytically. The default is True.
    :return free_flight_flap_driven_problem:
        FreeFlightFlapDrivenAeroelasticUnsteadyProblem This is the problem.
    """
    movement = ps.movements.free_flight_movement.FreeFlightAeroelasticMovement(
        airplane_movements=[_make_airplane_movement(True)],
        operating_point_movement=ps.movements.free_flight_operating_point_movement.FreeFlightOperatingPointMovement(
            base_operating_point=_make_operating_point()
        ),
        delta_time=DELTA_TIME,
        prescribed_num_steps=prescribed_num_steps,
        free_num_steps=free_num_steps,
    )
    free_flight_flap_driven_problem = ps.free_flight_flap_driven_aeroelastic_unsteady_problem.FreeFlightFlapDrivenAeroelasticUnsteadyProblem(
        movement=movement,
        wing_density=WING_DENSITY,
        spring_constant_rad=SPRING_CONSTANT_RAD,
        damping_constant_rad=DAMPING_CONSTANT_RAD,
        mass=1.0,
        I_BP1_CgP1=((0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1)),
        flap_drives=[make_tracking_drive()],
        step_discards=STEP_DISCARDS,
        use_analytical_spring_solver=use_analytical_spring_solver,
    )

    return free_flight_flap_driven_problem
