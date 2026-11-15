"""Demonstrates running Ptera Software's
AeroelasticUnsteadyRingVortexLatticeMethodSolver with a 5 meter span flapping wing that
is driven by motors, gearboxes, and crank linkages instead of by a prescribed flap
angle.

The crank linkage is a crank_linkage.CrankLinkage, a powertrain with its own class that
conforms to the flap_drive.Transmission protocol, so it is passed to each FlapDrive
directly. The companion flap_driven_aeroelastic_unsteady example shows the other way to
supply a transmission: as your own functions, wrapped in a flap_drive.FlapTransmission.

Each wing has its own motor. The motor's torque-speed curve, its peak and continuous
torques, its motor controller's peak and continuous currents, its brake current, and its
gearbox all limit the torque that it can apply, so the wing can fall behind its desired
flap trajectory.
"""

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np

# First, import the software's main package. Note that if you wished to import this
# software into another package, you would first install it by running "pip install
# pterasoftware" in your terminal.
import pterasoftware as ps

# Define the flight condition, the flap frequency, and the simulation's time steps.
FLAP_FREQUENCY = 2.0
FLAP_PERIOD = 1.0 / FLAP_FREQUENCY
AIR_SPEED = 10.0
ANGLE_OF_ATTACK = 10.0
DELTA_TIME = 0.01
NUM_STEPS = 150
DURATION = NUM_STEPS * DELTA_TIME

# Define the powertrain. A motor turns a gearbox, whose output turns the gearbox crank
# of a crank linkage. A rigid link joins the gearbox crank to a wing crank that turns
# with the Wing. The linkage's geometry is given in the plane of the flap.
#
# A FlapDrive doesn't know about cranks. It only uses its transmission's flap angle,
# Jacobian, Hessian, and trajectory conversion, which CrankLinkage provides in terms of
# the motor's angle, so the CrankLinkage is passed to the FlapDrives directly. Any other
# powertrain class with the same methods (see flap_drive.Transmission) works the same
# way.
#
# TRAJECTORY_CONVERSION sets how a flap trajectory is converted into the motor
# trajectory: "analytical" uses the inverse kinematics' analytical Jacobian and Hessian,
# and "numerical" differentiates the motor position in time. It doesn't affect the
# "constant_speed" trajectory below, which is already a motor trajectory.
GEAR_RATIO = 15.0
TRAJECTORY_CONVERSION = "analytical"
transmission = ps.crank_linkage.CrankLinkage(
    gearbox_to_shoulder_y=0.11,
    gearbox_to_shoulder_z=-0.05,
    wing_crank_length=0.065,
    wing_crank_angle_offset_rad=np.pi / 2,
    gearbox_crank_length=0.032,
    linkage_length=0.11,
    gear_ratio=GEAR_RATIO,
    derivative_method=TRAJECTORY_CONVERSION,
)

# Find the linkage's stroke by sampling its forward kinematics over one turn of the
# gearbox crank. The wing beat amplitude is the stroke's peak to peak size, and the
# dihedral is its middle.
min_flap_angle_rad, max_flap_angle_rad = transmission.angle_range()
WING_BEAT_AMPLITUDE_RAD = max_flap_angle_rad - min_flap_angle_rad
DIHEDRAL_RAD = (max_flap_angle_rad + min_flap_angle_rad) / 2.0

# Define the trajectory that each motor follows. "constant_speed" turns the motor at a
# constant speed, starting with the gearbox crank at a quarter turn, so the wing's
# turnarounds land at the linkage's dead centers. "sine" and "square" are flap angle
# trajectories that are converted into motor trajectories through the linkage's inverse
# kinematics. Both cover the linkage's full stroke. "square" reverses the flap instantly
# at each dead center, which needs an unbounded motor speed there, so its motor
# trajectory has very large spikes at the turnarounds. The reference's period is the
# flap's period, one turn of the gearbox crank, which is GEAR_RATIO turns of the motor.
FLAP_TRAJECTORY = "constant_speed"
if FLAP_TRAJECTORY == "constant_speed":
    flap_reference = ps.flap_drive.FlapReference.constant_velocity(
        start_position=GEAR_RATIO * np.pi / 2.0,
        velocity=GEAR_RATIO * 2.0 * np.pi * FLAP_FREQUENCY,
        duration=DURATION,
        period=FLAP_PERIOD,
    )
else:
    flap_reference = ps.flap_drive.FlapReference.from_flap_trajectory(
        transmission,
        *ps.flap_drive.generate_flap_trajectory(
            style=FLAP_TRAJECTORY,
            flapping_frequency=FLAP_FREQUENCY,
            wing_beat_amplitude_rad=WING_BEAT_AMPLITUDE_RAD,
            dihedral_rad=DIHEDRAL_RAD,
            min_angle_rad=min_flap_angle_rad,
            max_angle_rad=max_flap_angle_rad,
            num_cycles=int(np.ceil(DURATION * FLAP_FREQUENCY)),
        ),
        period=FLAP_PERIOD,
    )

# Define the wing's planform from its leading and trailing edge points (in millimeters,
# for a wingspan of 1.2 meters). The x axis points aft and the y axis points out along
# the span. The points are scaled to the desired wingspan below. Each WingCrossSection
# sits at a station where a trailing edge point is defined, and its leading edge is
# interpolated from the leading edge points.
WINGSPAN = 5.0
SCALE_M_PER_MM = WINGSPAN / 5.0 * 5.0 / 1.2 / 1000.0
leading_edge_y_mm = [0, 22, 71, 120, 170, 245, 318, 366, 415, 464, 512, 563, 611]
leading_edge_x_mm = [
    0,
    -20,
    -25.6,
    -29,
    -32,
    -34,
    -35,
    -34.5,
    -33.4,
    -31.7,
    -29.7,
    -27.5,
    -25,
]
station_y_mm = [52, 148, 255, 367, 494, 611]
station_trailing_edge_x_mm = [208, 175, 123, 73, 23, -16]
station_leading_edge_x_mm = np.interp(
    station_y_mm, leading_edge_y_mm, leading_edge_x_mm
)

NUM_SPANWISE_PANELS = 2
wing_cross_sections = []
for i, y_mm in enumerate(station_y_mm):
    chord = (station_trailing_edge_x_mm[i] - station_leading_edge_x_mm[i]) * (
        SCALE_M_PER_MM
    )
    if i == 0:
        Lp_Wcsp_Lpp = (0.0, 0.0, 0.0)
    else:
        Lp_Wcsp_Lpp = (
            (station_leading_edge_x_mm[i] - station_leading_edge_x_mm[i - 1])
            * SCALE_M_PER_MM,
            (y_mm - station_y_mm[i - 1]) * SCALE_M_PER_MM,
            0.0,
        )
    last = i == len(station_y_mm) - 1
    wing_cross_sections.append(
        ps.geometry.wing_cross_section.WingCrossSection(
            num_spanwise_panels=None if last else NUM_SPANWISE_PANELS,
            chord=chord,
            Lp_Wcsp_Lpp=Lp_Wcsp_Lpp,
            angles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
            control_surface_symmetry_type="symmetric",
            control_surface_hinge_point=0.75,
            control_surface_deflection=0.0,
            spanwise_spacing=None if last else "uniform",
            airfoil=ps.geometry.airfoil.Airfoil(name="naca0004"),
        )
    )

# The wing's root starts 52 mm (scaled) out from the shoulder. The Wing flaps about the
# shoulder, so the rotation point offset below moves the flap axis back to the
# airplane's plane of symmetry. The Wing's base flap angle is the stroke's middle. The
# first time step's Wings start at the flap angle of the drives' starting motor
# positions instead.
root_offset_y = station_y_mm[0] * SCALE_M_PER_MM
example_airplane = ps.geometry.airplane.Airplane(
    wings=[
        ps.geometry.wing.Wing(
            wing_cross_sections=wing_cross_sections,
            name="Main Wing",
            Ler_Gs_Cgs=(
                station_leading_edge_x_mm[0] * SCALE_M_PER_MM,
                root_offset_y,
                0.0,
            ),
            angles_Gs_to_Wn_ixyz=(np.rad2deg(DIHEDRAL_RAD), 0.0, 0.0),
            symmetric=True,
            mirror_only=False,
            symmetryNormal_G=(0.0, 1.0, 0.0),
            symmetryPoint_G_Cg=(0.0, 0.0, 0.0),
            explode_into_strips=True,
            num_chordwise_panels=6,
            chordwise_spacing="uniform",
        ),
    ],
    name="Example Airplane",
    Cg_GP1_CgP1=(0.0, 0.0, 0.0),
    weight=0.0,
)

# The main Wing has type 5 symmetry, so the Airplane split it into the Wing and a
# reflected Wing. Each needs its own movement, built around its own WingCrossSections.
wing_movements: list[
    ps.movements.aeroelastic_wing_movement.AeroelasticWingMovement
    | ps.movements.wing_movement.WingMovement
] = []
for wing in example_airplane.wings:
    wing_cross_section_movements = [
        ps.movements.aeroelastic_wing_cross_section_movement.AeroelasticWingCrossSectionMovement(
            base_wing_cross_section=wing_cross_section,
        )
        for wing_cross_section in wing.wing_cross_sections
    ]

    # A FlapDrivenAeroelasticWingMovement can't prescribe a flap motion: its angular
    # motion parameters (such as ampAnglesYZ_Gs_to_Wn_ixyz) take only the y and z
    # components, and the x (flap) component is held static. The FlapDrives' references
    # are the only definition of the flap trajectory, and their period sets the window
    # of the cycle averaged results.
    wing_movements.append(
        ps.movements.flap_driven_aeroelastic_wing_movement.FlapDrivenAeroelasticWingMovement(
            base_wing=wing,
            wing_cross_section_movements=wing_cross_section_movements,
            rotationPointOffset_Gs_Ler=(0.0, -root_offset_y, 0.0),
        )
    )

example_airplane_movement = (
    ps.movements.aeroelastic_airplane_movement.AeroelasticAirplaneMovement(
        base_airplane=example_airplane,
        wing_movements=wing_movements,
    )
)

example_operating_point = ps.operating_point.OperatingPoint(
    rho=1.225,
    vCg__E=AIR_SPEED,
    alpha=ANGLE_OF_ATTACK,
    beta=0.0,
    externalFX_W=0.0,
    nu=15.06e-6,
)
example_operating_point_movement = (
    ps.movements.operating_point_movement.OperatingPointMovement(
        base_operating_point=example_operating_point,
    )
)

example_movement = ps.movements.aeroelastic_movement.AeroelasticMovement(
    airplane_movements=[example_airplane_movement],
    operating_point_movement=example_operating_point_movement,
    delta_time=DELTA_TIME,
    num_steps=NUM_STEPS,
)

# Define the motor. This is the MAD M13 EEE KV90 (motor 9) at 48 V. Its torque-speed
# line and its torque-current line are fit to the same two measured points as in the
# blade element theory motor limit simulation, with each value divided by the test
# voltage before fitting, and the torque-speed line's intercept is then scaled back up
# to the supply voltage.
TEST_VOLTAGE = 48.0
SUPPLY_VOLTAGE = 48.0
measured_speeds = np.array([4300.0, 3424.0]) / TEST_VOLTAGE / 60.0 * 2.0 * np.pi
measured_torques = np.array([0.0, 8.85]) / TEST_VOLTAGE
measured_currents = np.array([2.0 * 48.0 / 20.0, 88.0]) / TEST_VOLTAGE
speed_torque_coefficients = np.polyfit(measured_speeds, measured_torques, 1)
speed_torque_coefficients[1] *= SUPPLY_VOLTAGE
torque_current_coefficients = np.polyfit(measured_currents, measured_torques, 1)
ROTOR_INERTIA = 0.0004158
MOTOR_CONTINUOUS_TORQUE = 8.8
MOTOR_PEAK_DURATION = 0.5

# Define the motor controller (an ODrive Pro) and the gearbox.
ESC_PEAK_CURRENT = 150.0
ESC_CONTINUOUS_CURRENT = 100.0
ESC_PEAK_DURATION = 0.1
ESC_MAX_BRAKE_CURRENT = 1000.0
GEARBOX_MAX_TORQUE = 225.0

# Define the controller's gains, the motor's damping, and each wing's measured inertia
# about the flap axis.
PROPORTIONAL_GAIN = 5.0
DERIVATIVE_GAIN = 0.5
MOTOR_DAMPING = 1e-3
WING_FLAP_INERTIA = 1.461


def make_motor() -> tuple[
    ps.motor_saturation.MotorSaturationController,
    Callable[[ps.flap_drive.FlapDriveInputs], float],
    dict[float, float],
]:
    """Builds one motor's saturation controller, the actuator function that uses it, and
    a dictionary that the actuator function fills with the largest torque the motor
    could apply, keyed by time."""
    max_torque_possible_log: dict[float, float] = {}
    saturation_controller = ps.motor_saturation.MotorSaturationController(
        speed_torque_coefficients=speed_torque_coefficients,
        torque_current_coefficients=torque_current_coefficients,
        motor_continuous_torque=MOTOR_CONTINUOUS_TORQUE,
        motor_peak_duration=MOTOR_PEAK_DURATION,
        esc_peak_current=ESC_PEAK_CURRENT,
        esc_continuous_current=ESC_CONTINUOUS_CURRENT,
        esc_peak_duration=ESC_PEAK_DURATION,
        gearbox_max_torque=GEARBOX_MAX_TORQUE,
        gear_ratio=GEAR_RATIO,
        esc_max_brake_current=ESC_MAX_BRAKE_CURRENT,
    )

    def motor(inputs: ps.flap_drive.FlapDriveInputs) -> float:
        """The torque that the motor applies: a feedforward torque plus feedback on the
        motor's position and velocity, limited by the saturation controller."""
        # NO BLADE ELEMENT THEORY MODEL IS USED IN THIS EXAMPLE. The feedforward's
        # aerodynamic part is the previous time step's UVLM torque at the wing's actual
        # motion (feedforward_aerodynamic=True below), so the controller and the wing
        # see the same UVLM torque. The blade element theory code's feedforward instead
        # predicts it at the desired wing velocity. To do that here, pass a blade
        # element theory model as the FlapDrive's feedforward_aerodynamic below. The
        # FlapDrive's feedforward_inertial, feedforward_damping, feedforward_passive,
        # and feedforward_aerodynamic switches turn each term of the feedforward on or
        # off.
        torque_desired = (
            inputs.feedforward_force
            + PROPORTIONAL_GAIN
            * (inputs.desired_actuator_position - inputs.actuator_position)
            + DERIVATIVE_GAIN
            * (inputs.desired_actuator_velocity - inputs.actuator_velocity)
        )
        torque_applied, saturation_info = saturation_controller.clip_torque(
            torque_desired, inputs.actuator_velocity, inputs.time
        )

        # The solver calls this function many times per time step while it integrates
        # the motor's dynamics, and its last call for each time step is at that time
        # step's instant. Keying the log by time, so later calls overwrite earlier ones,
        # leaves the limit at each time step's instant.
        max_torque_possible_log[round(inputs.time, 9)] = (
            saturation_info.max_torque_possible
        )
        return torque_applied

    return saturation_controller, motor, max_torque_possible_log


# Define one FlapDrive for each half of the main Wing, so each Wing has its own motor
# and its own saturation controller, which keeps track of that motor's peak allowances.
# Each drive starts at its reference's first motor position and velocity. Each drive's
# feedforward reads the reference at the entry nearest the motor's actual position
# (use_position_lookup=True), like the blade element theory code's use_closest_pos,
# which works because the motor keeps turning one way, so the reference's positions are
# monotonic. Set it to False to read the reference at the current time instead.
#
# The aerodynamic model is the UVLM only. No blade element theory model is used yet.
# feedforward_aerodynamic=True makes the controller's feedforward cancel the previous
# time step's UVLM torque. The wing itself always feels the UVLM torque, whatever this
# is set to. To make the controller use a blade element theory model instead, replace
# True with a function of the flap angle (rad) and the desired flap rate (rad/s) that
# returns the predicted fluid torque on the wing (N*m, positive toward increasing flap
# angle). Only the controller's feedforward would change.
saturation_controllers = []
max_torque_possible_logs = []
example_flap_drives = []
for wing_index in range(2):
    saturation_controller, motor, max_torque_possible_log = make_motor()
    saturation_controllers.append(saturation_controller)
    max_torque_possible_logs.append(max_torque_possible_log)
    example_flap_drives.append(
        ps.flap_drive.FlapDrive(
            wing_indices=(wing_index,),
            transmission=transmission,
            actuator=motor,
            actuator_inertia=ROTOR_INERTIA,
            actuator_damping=MOTOR_DAMPING,
            wing_flap_inertia=WING_FLAP_INERTIA,
            reference=flap_reference,
            use_position_lookup=True,
            # The UVLM torque, not a blade element theory model.
            feedforward_aerodynamic=True,
        )
    )

# Define the problem. It is an AeroelasticUnsteadyProblem that also takes the
# FlapDrives.
example_problem = (
    ps.flap_driven_aeroelastic_unsteady_problem.FlapDrivenAeroelasticUnsteadyProblem(
        movement=example_movement,
        wing_density=5.5,  # 0.055,
        spring_constant_rad=1000.0,
        damping_constant_rad=1000.0,
        flap_drives=example_flap_drives,
        step_discards=5,
    )
)

example_solver = ps.aeroelastic_unsteady_ring_vortex_lattice_method.AeroelasticUnsteadyRingVortexLatticeMethodSolver(
    aeroelastic_unsteady_problem=example_problem,
)

example_solver.run(prescribed_wake=False)
print("The simulation has finished.")

# The plots below mirror the blade element theory code's runSim figures, for the first
# Wing's motor. The desired trajectory is plotted on the reference's fine time grid
# (time_vec there), and the actual trajectory at each time step.
step_times = np.arange(NUM_STEPS) * DELTA_TIME
on_steps = flap_reference.times <= step_times[-1]
reference_times = flap_reference.times[on_steps]

# Get the desired motor trajectory (motor_traj_des there) from the reference.
motor_pos_des = flap_reference.positions[on_steps]
motor_velo_des = flap_reference.velocities[on_steps]
motor_accel_des = flap_reference.accelerations[on_steps]

# Get the desired wing trajectory (phi_des, phi_dot_des, and phi_ddot_des there) from
# the desired motor trajectory through the transmission: phi = T(q), phi' = J * q', and
# phi'' = H * q'^2 + J * q''.
phi_des = np.array([transmission.evaluate_angle(q) for q in motor_pos_des])
jacobian_des = np.array([transmission.evaluate_jacobian(q) for q in motor_pos_des])
hessian_des = np.array([transmission.evaluate_hessian(q) for q in motor_pos_des])
phi_dot_des = jacobian_des * motor_velo_des
phi_ddot_des = hessian_des * motor_velo_des**2 + jacobian_des * motor_accel_des

# Get the actual trajectories that the problem recorded at each time step.
phi_actual = np.array(example_problem.flap_angles_rad[0])
phi_dot_actual = np.array(example_problem.flap_angle_derivatives_rad[0])
phi_ddot_actual = np.array(example_problem.flap_angle_second_derivatives_rad[0])
motor_pos_actual = np.array(example_problem.flap_actuator_positions[0])
motor_velo_actual = np.array(example_problem.flap_actuator_velocities[0])
motor_accel_actual = np.array(example_problem.flap_actuator_accelerations[0])
motor_torque_applied = np.array(example_problem.flap_actuator_forces[0])
max_motor_torque_possible = np.array(
    [max_torque_possible_logs[0][round(t, 9)] for t in step_times]
)

# Plot the wing kinematics (the "Wing Kinematics" figure there).
wing_figure, wing_axes = plt.subplots(
    3, 1, figsize=(10, 9), sharex=True, num="Wing Kinematics"
)
for axis, desired, actual, color, label in (
    (wing_axes[0], phi_des, phi_actual, "b", "Wing Position [rad]"),
    (wing_axes[1], phi_dot_des, phi_dot_actual, "r", "Wing Velocity [rad/sec]"),
    (
        wing_axes[2],
        phi_ddot_des,
        phi_ddot_actual,
        "g",
        "Wing Acceleration [rad/sec^2]",
    ),
):
    # Draw the desired trajectory on top, thicker, and in black, so that it stays
    # visible where the actual trajectory follows it closely.
    axis.plot(step_times, actual, color + "-", linewidth=3.0, alpha=0.5, label="Actual")
    axis.plot(reference_times, desired, "k:", linewidth=1.5, zorder=3, label="Desired")
    axis.set_ylabel(label)
    axis.legend(loc="best")
    axis.grid(True)
wing_axes[0].set_title("Wing Kinematics")
wing_axes[2].set_xlabel("Time [s]")
wing_figure.tight_layout()

# Plot the motor dynamics (the "Motor Dynamics" figure there).
motor_figure, motor_axes = plt.subplots(
    4, 1, figsize=(10, 12), sharex=True, num="Motor Dynamics"
)
for axis, desired, actual, color, label in (
    (motor_axes[0], motor_pos_des, motor_pos_actual, "b", "Motor Position [rad]"),
    (
        motor_axes[1],
        motor_velo_des,
        motor_velo_actual,
        "r",
        "Motor Velocity [rad/sec]",
    ),
    (
        motor_axes[2],
        motor_accel_des,
        motor_accel_actual,
        "g",
        "Motor Acceleration [rad/sec^2]",
    ),
):
    # Draw the desired trajectory on top, thicker, and in black, so that it stays
    # visible where the actual trajectory follows it closely.
    axis.plot(step_times, actual, color + "-", linewidth=3.0, alpha=0.5, label="Actual")
    axis.plot(reference_times, desired, "k:", linewidth=1.5, zorder=3, label="Desired")
    axis.set_ylabel(label)
    axis.legend(loc="best")
    axis.grid(True)
motor_axes[0].set_title("Motor Trajectory")
motor_axes[3].plot(
    step_times,
    motor_torque_applied,
    color=(0.47, 0.67, 0.19),
    linewidth=1.5,
    label="Applied Torque",
)
motor_axes[3].plot(
    step_times,
    max_motor_torque_possible,
    "k",
    linewidth=1.0,
    label="Max Possible Torque",
)
motor_axes[3].plot(step_times, -max_motor_torque_possible, "k", linewidth=1.0)
motor_axes[3].set_ylabel("Motor Torque [Nm]")
motor_axes[3].set_xlabel("Time [s]")
motor_axes[3].legend(loc="best")
motor_axes[3].grid(True)
motor_figure.tight_layout()

# Plot the motor's operating points against its peak and continuous torque-speed limits.
limits_figure, limits_axis = plt.subplots(figsize=(10, 5), num="Motor Operating Points")
no_load_speed = -speed_torque_coefficients[1] / speed_torque_coefficients[0]
speeds = np.linspace(0.0, 1.2 * no_load_speed, 200)
first_controller = saturation_controllers[0]
limits_axis.plot(
    speeds,
    [first_controller.max_torque_possible(s, False, False) for s in speeds],
    "k--",
    label="Peak limit",
)
limits_axis.plot(
    speeds,
    [first_controller.max_torque_continuous(s) for s in speeds],
    "k:",
    label="Continuous limit",
)
limits_axis.plot(
    np.abs(motor_velo_actual),
    np.abs(motor_torque_applied),
    ".",
    label="Operating points",
)
limits_axis.set_xlabel("Motor Speed [rad/sec]")
limits_axis.set_ylabel("Motor Torque [Nm]")
limits_axis.legend(loc="best")
limits_axis.grid(True)
limits_figure.tight_layout()

plt.show()
