"""This module contains functions to create FlapTransmissions, FlapDrives, and the flap
driven classes that use them, for use in tests."""

import numpy as np

import pterasoftware as ps
from pterasoftware import flap_drive

# These are the parameters of the nonlinear transmission fixture, a stand-in for a
# crankshaft linkage: the flap angle is a sinusoid of the actuator position plus a
# linear term. They are exposed so that tests can compare against the analytic values.
NONLINEAR_SINE_AMPLITUDE_RAD = 0.5
NONLINEAR_LINEAR_COEFFICIENT = 0.1


def nonlinear_angle(actuator_position: float) -> float:
    """Returns the nonlinear fixture transmission's flap angle.

    :param actuator_position: The actuator position (rad).
    :return: The flap angle (rad).
    """
    return float(
        NONLINEAR_SINE_AMPLITUDE_RAD * np.sin(actuator_position)
        + NONLINEAR_LINEAR_COEFFICIENT * actuator_position
    )


def nonlinear_jacobian(actuator_position: float) -> float:
    """Returns the analytic Jacobian of the nonlinear fixture transmission.

    :param actuator_position: The actuator position (rad).
    :return: The first derivative of the flap angle with respect to the actuator
        position.
    """
    return float(
        NONLINEAR_SINE_AMPLITUDE_RAD * np.cos(actuator_position)
        + NONLINEAR_LINEAR_COEFFICIENT
    )


def nonlinear_hessian(actuator_position: float) -> float:
    """Returns the analytic Hessian of the nonlinear fixture transmission.

    :param actuator_position: The actuator position (rad).
    :return: The second derivative of the flap angle with respect to the actuator
        position.
    """
    return float(-NONLINEAR_SINE_AMPLITUDE_RAD * np.sin(actuator_position))


def unused_trajectory(
    times: np.ndarray,
    angles_rad: np.ndarray,
    angle_derivatives_rad: np.ndarray,
    angle_second_derivatives_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A placeholder inverse kinematics for fixture transmissions whose tests only
    exercise the equation of motion and never convert a flap trajectory.

    A FlapTransmission requires an inverse kinematics function. The nonlinear and
    singular fixture transmissions have none in closed form, so they get this one, which
    fails loudly if a test ever does convert a trajectory through them.

    :param times: The sample times (s). It is unused.
    :param angles_rad: The flap angles (rad). It is unused.
    :param angle_derivatives_rad: The flap angles' time derivatives (rad/s). It is
        unused.
    :param angle_second_derivatives_rad: The flap angles' second time derivatives
        (rad/s^2). It is unused.
    :return: Nothing. It always raises a NotImplementedError.
    """
    raise NotImplementedError(
        "This fixture transmission has no inverse kinematics, so its tests must not "
        "convert a flap trajectory through it."
    )


def constant_actuator(inputs: flap_drive.FlapDriveInputs) -> float:
    """An actuator function that applies a constant generalized force.

    :param inputs: The FlapDriveInputs passed by the FlapDrive. It is unused.
    :return: 0.7
    """
    return 0.7


def make_hold_reference_fixture() -> flap_drive.FlapReference:
    """This method makes a fixture that is a FlapReference that holds the actuator at
    0.0.

    :return hold_reference_fixture: FlapReference This is the FlapReference whose
        desired actuator position, velocity, and acceleration are 0.0 from 0.0 s to
        1000.0 s. It has no period. It is the reference for tests that only exercise the
        equation of motion and don't track anything.
    """
    hold_reference_fixture = flap_drive.FlapReference(
        (0.0, 1000.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), period=None
    )

    return hold_reference_fixture


def make_nonlinear_transmission_fixture() -> flap_drive.FlapTransmission:
    """This method makes a fixture that is a nonlinear FlapTransmission with analytic
    derivatives.

    :return nonlinear_transmission_fixture: FlapTransmission This is the
        FlapTransmission whose flap angle is 0.5 * sin(q) + 0.1 * q, which has a
        Jacobian that varies with the actuator position and a nonzero Hessian.
    """
    nonlinear_transmission_fixture = flap_drive.FlapTransmission(
        nonlinear_angle, nonlinear_jacobian, nonlinear_hessian, unused_trajectory
    )

    return nonlinear_transmission_fixture


def make_basic_flap_drive_fixture() -> flap_drive.FlapDrive:
    """This method makes a fixture that is a FlapDrive with a direct drive transmission
    and no optional features.

    :return basic_flap_drive_fixture: FlapDrive This is the FlapDrive for general
        testing. It flaps the first two Wings through an identity transmission, with a
        constant actuator force and a small actuator inertia.
    """
    basic_flap_drive_fixture = flap_drive.FlapDrive(
        wing_indices=(0, 1),
        transmission=flap_drive.FlapTransmission.identity(),
        actuator=constant_actuator,
        actuator_inertia=0.02,
        reference=make_hold_reference_fixture(),
        use_position_lookup=False,
    )

    return basic_flap_drive_fixture


def make_nonlinear_flap_drive_fixture() -> flap_drive.FlapDrive:
    """This method makes a fixture that is a FlapDrive with a nonlinear transmission,
    damping, and both passive elements.

    :return nonlinear_flap_drive_fixture: FlapDrive This is the FlapDrive that exercises
        every term in the equation of motion. It uses the nonlinear transmission, an
        actuator inertia of 0.02, a damping coefficient of 0.05, a passive Wing torque
        of -0.03, and a passive actuator force of 0.2.
    """
    nonlinear_flap_drive_fixture = flap_drive.FlapDrive(
        wing_indices=[0, 1],
        transmission=make_nonlinear_transmission_fixture(),
        actuator=constant_actuator,
        actuator_inertia=0.02,
        actuator_damping=0.05,
        initial_actuator_position=0.3,
        initial_actuator_velocity=0.4,
        passive_wing_torque_function=lambda angle_rad, angle_derivative_rad: -0.03,
        passive_actuator_force_function=lambda position, velocity: 0.2,
        reference=make_hold_reference_fixture(),
        use_position_lookup=False,
    )

    return nonlinear_flap_drive_fixture


def make_singular_flap_drive_fixture(
    actuator_inertia: float, reference: flap_drive.FlapReference | None = None
) -> flap_drive.FlapDrive:
    """This method makes a fixture that is a FlapDrive whose transmission has a zero
    Jacobian at an actuator position of 0.0.

    :param actuator_inertia: The actuator's own inertia. A value of 0.0 leaves the
        FlapDrive with no inertia at the singular position.
    :param reference: None, or the FlapReference to follow. If None, the FlapDrive
        follows the hold reference fixture. The default is None.
    :return singular_flap_drive_fixture: FlapDrive This is the FlapDrive with a flap
        angle of 0.5 * q^2, a Jacobian of q, and a Hessian of 1.0.
    """
    singular_transmission = flap_drive.FlapTransmission(
        lambda position: 0.5 * position**2,
        lambda position: position,
        lambda position: 1.0,
        unused_trajectory,
    )
    singular_flap_drive_fixture = flap_drive.FlapDrive(
        wing_indices=(0,),
        transmission=singular_transmission,
        actuator=lambda inputs: 1.0,
        actuator_inertia=actuator_inertia,
        reference=reference if reference is not None else make_hold_reference_fixture(),
        use_position_lookup=False,
    )

    return singular_flap_drive_fixture


# These are the parameters of the sine flap trajectory that the flap driven problem
# fixtures' references follow. The amplitude and phase are in degrees and the period is
# in seconds.
SINE_FLAP_AMPLITUDE = 15.0
SINE_FLAP_PERIOD = 1.0
SINE_FLAP_PHASE = 1.0
FIXTURE_DELTA_TIME = 0.05


def sine_flap_angle_deg(time: float) -> float:
    """Returns the flap angle of the fixtures' sine flap trajectory.

    :param time: The time (s).
    :return: The flap angle (degrees).
    """
    return float(
        SINE_FLAP_AMPLITUDE
        * np.sin(2.0 * np.pi / SINE_FLAP_PERIOD * time + np.deg2rad(SINE_FLAP_PHASE))
    )


def make_sine_flap_reference_fixture(
    duration: float = 2.0, period: float | None = SINE_FLAP_PERIOD
) -> flap_drive.FlapReference:
    """This method makes a fixture that is a FlapReference for a direct drive that
    follows the sine flap trajectory.

    :param duration: The time the reference covers (s). The default is 2.0.
    :param period: The period the reference reports (s), or None. The default is the
        sine's period.
    :return sine_flap_reference_fixture: FlapReference This is the FlapReference, built
        through an identity transmission, so its actuator positions are the flap angles
        in radians.
    """
    times = np.arange(int(round(duration / 1e-4)) + 1, dtype=float) * 1e-4
    amplitude_rad = np.deg2rad(SINE_FLAP_AMPLITUDE)
    frequency_rad = 2.0 * np.pi / SINE_FLAP_PERIOD
    phases_rad = frequency_rad * times + np.deg2rad(SINE_FLAP_PHASE)
    sine_flap_reference_fixture = flap_drive.FlapReference.from_flap_trajectory(
        flap_drive.FlapTransmission.identity(),
        times,
        amplitude_rad * np.sin(phases_rad),
        amplitude_rad * frequency_rad * np.cos(phases_rad),
        -amplitude_rad * frequency_rad**2 * np.sin(phases_rad),
        period=period,
    )

    return sine_flap_reference_fixture


def _make_flap_driven_airplane_movement(
    rotationPointOffset_Gs_Ler: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> ps.movements.aeroelastic_airplane_movement.AeroelasticAirplaneMovement:
    """Builds the AeroelasticAirplaneMovement that the flap driven movement fixtures
    share: a symmetric two-Wing Airplane with no weight, whose Wings are moved by
    FlapDrivenAeroelasticWingMovements that prescribe no flap motion.

    :param rotationPointOffset_Gs_Ler: Each Wing movement's rotation point offset (m).
        The default is (0.0, 0.0, 0.0).
    :return: The AeroelasticAirplaneMovement.
    """
    airfoil = ps.geometry.airfoil.Airfoil(name="naca2412")
    wing_cross_sections = [
        ps.geometry.wing_cross_section.WingCrossSection(
            num_spanwise_panels=num_panels,
            chord=chord,
            Lp_Wcsp_Lpp=(0.0, span, 0.0),
            angles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
            control_surface_symmetry_type="symmetric",
            control_surface_hinge_point=0.75,
            control_surface_deflection=0.0,
            spanwise_spacing=spacing,
            airfoil=airfoil,
        )
        for num_panels, chord, span, spacing in (
            (1, 1.0, 0.0, "uniform"),
            (None, 0.6, 0.5, None),
        )
    ]
    main_wing = ps.geometry.wing.Wing(
        wing_cross_sections=wing_cross_sections,
        name="Main Wing",
        Ler_Gs_Cgs=(0.0, 0.1, 0.0),
        symmetric=True,
        symmetryNormal_G=(0.0, 1.0, 0.0),
        symmetryPoint_G_Cg=(0.0, 0.0, 0.0),
        num_chordwise_panels=2,
        chordwise_spacing="uniform",
    )
    airplane = ps.geometry.airplane.Airplane(
        wings=[main_wing], name="Airplane", Cg_GP1_CgP1=(0.0, 0.0, 0.0), weight=0.0
    )

    wing_movements: list[
        ps.movements.aeroelastic_wing_movement.AeroelasticWingMovement
        | ps.movements.wing_movement.WingMovement
    ] = []
    for wing in airplane.wings:
        wing_movements.append(
            ps.movements.flap_driven_aeroelastic_wing_movement.FlapDrivenAeroelasticWingMovement(
                base_wing=wing,
                wing_cross_section_movements=[
                    ps.movements.aeroelastic_wing_cross_section_movement.AeroelasticWingCrossSectionMovement(
                        base_wing_cross_section=wing_cross_section
                    )
                    for wing_cross_section in wing.wing_cross_sections
                ],
                rotationPointOffset_Gs_Ler=rotationPointOffset_Gs_Ler,
            )
        )

    return ps.movements.aeroelastic_airplane_movement.AeroelasticAirplaneMovement(
        base_airplane=airplane, wing_movements=wing_movements
    )


def make_flap_driven_aeroelastic_movement_fixture(
    num_steps: int = 6,
    rotationPointOffset_Gs_Ler: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> ps.movements.aeroelastic_movement.AeroelasticMovement:
    """This method makes a fixture that is an AeroelasticMovement whose Wings are moved
    by FlapDrivenAeroelasticWingMovements.

    :param num_steps: The number of time steps. The default is 6.
    :param rotationPointOffset_Gs_Ler: Each Wing movement's rotation point offset (m).
        The default is (0.0, 0.0, 0.0).
    :return flap_driven_aeroelastic_movement_fixture: AeroelasticMovement This is the
        AeroelasticMovement for a symmetric two-Wing Airplane whose movements prescribe
        no flap motion, so a FlapDrive must flap it.
    """
    return ps.movements.aeroelastic_movement.AeroelasticMovement(
        airplane_movements=[
            _make_flap_driven_airplane_movement(rotationPointOffset_Gs_Ler)
        ],
        operating_point_movement=ps.movements.operating_point_movement.OperatingPointMovement(
            base_operating_point=ps.operating_point.OperatingPoint(
                rho=1.225, vCg__E=10.0, alpha=0.0
            )
        ),
        delta_time=FIXTURE_DELTA_TIME,
        num_steps=num_steps,
    )


def make_tracking_flap_drive_fixture(
    torque_limit: float | None = None,
    reference: flap_drive.FlapReference | None = None,
) -> flap_drive.FlapDrive:
    """This method makes a fixture that is a direct drive FlapDrive for both Wings that
    tracks the sine flap trajectory with feedforward plus feedback.

    :param torque_limit: None, or the largest magnitude of the actuator's torque (N*m).
        The default is None, which leaves the actuator unlimited.
    :param reference: None, or the FlapReference to follow. If None, the FlapDrive
        follows the sine flap reference fixture. The default is None.
    :return tracking_flap_drive_fixture: FlapDrive This is the FlapDrive.
    """

    def actuator(inputs: flap_drive.FlapDriveInputs) -> float:
        force = (
            inputs.feedforward_force
            + 400.0 * (inputs.desired_actuator_position - inputs.actuator_position)
            + 20.0 * (inputs.desired_actuator_velocity - inputs.actuator_velocity)
        )
        if torque_limit is not None:
            force = float(np.clip(force, -torque_limit, torque_limit))
        return force

    tracking_flap_drive_fixture = flap_drive.FlapDrive(
        wing_indices=(0, 1),
        transmission=flap_drive.FlapTransmission.identity(),
        actuator=actuator,
        actuator_inertia=1e-3,
        reference=(
            reference if reference is not None else make_sine_flap_reference_fixture()
        ),
        use_position_lookup=False,
    )

    return tracking_flap_drive_fixture


def make_flap_driven_aeroelastic_problem_fixture(
    torque_limit: float | None = None,
    num_steps: int = 6,
) -> ps.flap_driven_aeroelastic_unsteady_problem.FlapDrivenAeroelasticUnsteadyProblem:
    """This method makes a fixture that is a FlapDrivenAeroelasticUnsteadyProblem.

    :param torque_limit: None, or the largest magnitude of the FlapDrive's torque (N*m).
        The default is None.
    :param num_steps: The number of time steps. The default is 6.
    :return flap_driven_aeroelastic_problem_fixture:
        FlapDrivenAeroelasticUnsteadyProblem This is the problem.
    """
    flap_driven_aeroelastic_problem_fixture = ps.flap_driven_aeroelastic_unsteady_problem.FlapDrivenAeroelasticUnsteadyProblem(
        movement=make_flap_driven_aeroelastic_movement_fixture(num_steps),
        wing_density=0.5,
        spring_constant_rad=20.0,
        damping_constant_rad=1.0,
        flap_drives=[make_tracking_flap_drive_fixture(torque_limit)],
    )

    return flap_driven_aeroelastic_problem_fixture


def make_free_flight_flap_driven_aeroelastic_movement_fixture(
    prescribed_num_steps: int = 2,
    free_num_steps: int = 4,
) -> ps.movements.free_flight_movement.FreeFlightAeroelasticMovement:
    """This method makes a fixture that is a FreeFlightAeroelasticMovement whose Wings
    are moved by FlapDrivenAeroelasticWingMovements.

    :param prescribed_num_steps: The number of prescribed time steps. The default is 2.
    :param free_num_steps: The number of free flight time steps. The default is 4.
    :return free_flight_flap_driven_aeroelastic_movement_fixture:
        FreeFlightAeroelasticMovement This is the movement, for the same Airplane as the
        AeroelasticMovement fixture. It has no gravity and the Airplane has no weight.
    """
    free_flight_flap_driven_aeroelastic_movement_fixture = ps.movements.free_flight_movement.FreeFlightAeroelasticMovement(
        airplane_movements=[_make_flap_driven_airplane_movement()],
        operating_point_movement=ps.movements.free_flight_operating_point_movement.FreeFlightOperatingPointMovement(
            base_operating_point=ps.operating_point.OperatingPoint(
                rho=1.225, vCg__E=10.0, alpha=0.0
            )
        ),
        delta_time=FIXTURE_DELTA_TIME,
        prescribed_num_steps=prescribed_num_steps,
        free_num_steps=free_num_steps,
    )

    return free_flight_flap_driven_aeroelastic_movement_fixture


def make_free_flight_flap_driven_aeroelastic_problem_fixture(
    prescribed_num_steps: int = 2,
    free_num_steps: int = 4,
    reference: flap_drive.FlapReference | None = None,
    step_discards: int = 5,
    drive: flap_drive.FlapDrive | None = None,
    approximations: ps.approximations.Approximations | None = None,
) -> (
    ps.free_flight_flap_driven_aeroelastic_unsteady_problem.FreeFlightFlapDrivenAeroelasticUnsteadyProblem
):
    """This method makes a fixture that is a
    FreeFlightFlapDrivenAeroelasticUnsteadyProblem.

    :param prescribed_num_steps: The number of prescribed time steps. The default is 2.
    :param free_num_steps: The number of free flight time steps. The default is 4.
    :param reference: None, or the FlapReference the FlapDrive follows. If None, it
        follows the sine flap reference fixture. It is ignored if drive is given. The
        default is None.
    :param step_discards: The problem's step_discards. The default is 5.
    :param drive: None, or the FlapDrive to use. If None, the tracking FlapDrive fixture
        is used. The default is None.
    :param approximations: The problem's Approximations, or None for none. The default
        is None.
    :return free_flight_flap_driven_aeroelastic_problem_fixture:
        FreeFlightFlapDrivenAeroelasticUnsteadyProblem This is the problem, with a
        direct drive FlapDrive for both Wings, a 1.0 kg vehicle, and no gravity.
    """
    free_flight_flap_driven_aeroelastic_problem_fixture = ps.free_flight_flap_driven_aeroelastic_unsteady_problem.FreeFlightFlapDrivenAeroelasticUnsteadyProblem(
        movement=make_free_flight_flap_driven_aeroelastic_movement_fixture(
            prescribed_num_steps, free_num_steps
        ),
        wing_density=0.5,
        spring_constant_rad=20.0,
        damping_constant_rad=1.0,
        mass=1.0,
        I_BP1_CgP1=((0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1)),
        flap_drives=[
            (
                drive
                if drive is not None
                else make_tracking_flap_drive_fixture(reference=reference)
            )
        ],
        step_discards=step_discards,
        approximations=approximations,
    )

    return free_flight_flap_driven_aeroelastic_problem_fixture
