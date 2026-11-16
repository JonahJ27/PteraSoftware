"""Demonstrates running Ptera Software's
AeroelasticUnsteadyRingVortexLatticeMethodSolver with the blade element theory code's
model 5 (4.3 meter span) flapping wing that is driven by motors, gearboxes, and crank
linkages instead of by a prescribed flap angle.

The crank linkage is a crank_linkage.CrankLinkage, a powertrain with its own class that
conforms to the flap_drive.Transmission protocol, so it is passed to each FlapDrive
directly. The companion flap_driven_aeroelastic_unsteady example shows the other way to
supply a transmission: as your own functions, wrapped in a flap_drive.FlapTransmission.

Each wing has its own motor. The motor's torque-speed curve, its peak and continuous
torques, its motor controller's peak and continuous currents, its brake current, and its
gearbox all limit the torque that it can apply, so the wing can fall behind its desired
flap trajectory.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# First, import the software's main package. Note that if you wished to import this
# software into another package, you would first install it by running "pip install
# pterasoftware" in your terminal.
import pterasoftware as ps

# Load the hardware's parameters from the JSON files in the flap_drive_parameters
# directory next to this script, which the flap driven examples share: the model 5
# wing's outline and mass (model_5_wing.json), the powertrain's motor, motor controller,
# gearbox, and feedback gains (mad_m13_kv90_powertrain.json), and the crank linkage's
# geometry (crank_linkage.json). Edit those files to change the hardware. The flight
# condition and the simulation's settings are defined in this script, below.
PARAMETERS_DIRECTORY = Path(__file__).resolve().parent / "flap_drive_parameters"


def load_parameters(file_name: str) -> dict[str, Any]:
    """Loads one of the JSON files of parameters in PARAMETERS_DIRECTORY."""
    with open(PARAMETERS_DIRECTORY / file_name, encoding="utf-8") as file:
        parameters: dict[str, Any] = json.load(file)
    return parameters


wing_parameters = load_parameters("model_5_wing.json")
powertrain_parameters = load_parameters("mad_m13_kv90_powertrain.json")
motor_parameters = powertrain_parameters["motor"]
motor_controller_parameters = powertrain_parameters["motor_controller"]
crank_parameters = load_parameters("crank_linkage.json")

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
GEAR_RATIO = powertrain_parameters["gearbox"]["gear_ratio"]
TRAJECTORY_CONVERSION = "analytical"
transmission = ps.crank_linkage.CrankLinkage(
    gearbox_to_shoulder_y=crank_parameters["gearbox_to_shoulder_y_m"],
    gearbox_to_shoulder_z=crank_parameters["gearbox_to_shoulder_z_m"],
    wing_crank_length=crank_parameters["wing_crank_length_m"],
    wing_crank_angle_offset_rad=np.deg2rad(
        crank_parameters["wing_crank_angle_offset_deg"]
    ),
    gearbox_crank_length=crank_parameters["gearbox_crank_length_m"],
    linkage_length=crank_parameters["linkage_length_m"],
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

# Define the main Wing as the blade element theory code's model 5 wing (calcWingGeom
# with model_type = 5), whose outline and mass are in model_5_wing.json. calcWingGeom
# gives the wing's leading and trailing edge points in millimeters, for a wing of the
# outline reference length (1.2 meters), with the shoulder at the origin, the x axis
# pointing aft, and the y axis pointing out along the span. That code scales them to its
# wingspan of 2 * (spar length + shoulder offset) meters (the 2.09 meter spar plus its
# 7.5 centimeter offset from the shoulder, per side), so SCALE_M_PER_MM converts them to
# meters at that wingspan. The trailing edge runs from the shoulder to the wing's
# connection point and then out to the tip, and the tip is closed by a chordwise edge
# from the last trailing edge point to the last leading edge point.
MODEL_5_WINGSPAN = 2.0 * (
    wing_parameters["spar_length_m"] + wing_parameters["shoulder_offset_m"]
)
SCALE_M_PER_MM = (
    MODEL_5_WINGSPAN / wing_parameters["outline_reference_length_m"] / 1000.0
)
leading_edge_x_mm = wing_parameters["leading_edge_x_mm"]
leading_edge_y_mm = wing_parameters["leading_edge_y_mm"]
trailing_edge_x_mm = wing_parameters["trailing_edge_x_mm"]
trailing_edge_y_mm = wing_parameters["trailing_edge_y_mm"]

# Put a WingCrossSection at every spanwise station where the leading or the trailing
# edge has a point, so that the Wing's straight edges between WingCrossSections trace
# the model 5 outline exactly. The leading edge points at 71 and 366 millimeters are 1
# millimeter from the trailing edge points at 70 and 367 millimeters, so they are merged
# into those stations instead of making WingCrossSections a few millimeters wide. That
# moves the leading edge by at most 0.1 millimeters (0.4 millimeters at full scale)
# there.
MERGE_TOLERANCE_MM = wing_parameters["meshing"]["merge_tolerance_mm"]
station_y_mm = sorted(
    set(trailing_edge_y_mm)
    | {
        y
        for y in leading_edge_y_mm
        if min(abs(y - other) for other in trailing_edge_y_mm) > MERGE_TOLERANCE_MM
    }
)
station_leading_edge_x_mm = np.interp(
    station_y_mm, leading_edge_y_mm, leading_edge_x_mm
)
station_trailing_edge_x_mm = np.interp(
    station_y_mm, trailing_edge_y_mm, trailing_edge_x_mm
)

# The model 5 wing's chord is zero at the shoulder, where its leading and trailing edges
# meet, but a WingCrossSection's chord must be positive, so the root WingCrossSection
# gets a small chord instead. Each section between WingCrossSections gets spanwise
# panels about SPANWISE_PANEL_WIDTH wide.
ROOT_CHORD = wing_parameters["meshing"]["root_chord_m"]
SPANWISE_PANEL_WIDTH = wing_parameters["meshing"]["spanwise_panel_width_m"]
wing_cross_sections = []
for i, y_mm in enumerate(station_y_mm):
    chord = max(
        (station_trailing_edge_x_mm[i] - station_leading_edge_x_mm[i]) * SCALE_M_PER_MM,
        ROOT_CHORD,
    )
    last = i == len(station_y_mm) - 1
    if i == 0:
        Lp_Wcsp_Lpp = (0.0, 0.0, 0.0)
    else:
        Lp_Wcsp_Lpp = (
            (station_leading_edge_x_mm[i] - station_leading_edge_x_mm[i - 1])
            * SCALE_M_PER_MM,
            (y_mm - station_y_mm[i - 1]) * SCALE_M_PER_MM,
            0.0,
        )
    num_spanwise_panels = None
    if not last:
        section_width = (station_y_mm[i + 1] - y_mm) * SCALE_M_PER_MM
        num_spanwise_panels = max(1, round(section_width / SPANWISE_PANEL_WIDTH))
    wing_cross_sections.append(
        ps.geometry.wing_cross_section.WingCrossSection(
            num_spanwise_panels=num_spanwise_panels,
            chord=chord,
            Lp_Wcsp_Lpp=Lp_Wcsp_Lpp,
            angles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
            control_surface_symmetry_type="symmetric",
            control_surface_hinge_point=0.75,
            control_surface_deflection=0.0,
            spanwise_spacing=None if last else "uniform",
            airfoil=ps.geometry.airfoil.Airfoil(name=wing_parameters["airfoil"]),
        )
    )

# Define the model 5 wing's mass properties, from the blade element theory code's
# precomputeWingGeometry with runSim's settings: a carbon fiber leading edge tube
# (leading_edge_shape = 6, a 2.5 millimeter wall, with a 36 millimeter outer diameter
# and a density of 1500.0 kg/m^3) along the leading edge points, and no membrane mass
# (include_membrane_mass = false). WING_FLAP_INERTIA is the tube's inertia about the
# flap axis at the shoulder (its Ixx), which is what that code's
# inertialBladeElementTheoryCalc uses. The torsional spring model spreads a Wing's mass
# over its panels by area instead, so its wing_density is the tube's mass divided by one
# Wing's planform area. See docs/FLAP_DRIVE_TODO.md about the two inertias.
WING_FLAP_INERTIA = wing_parameters["flap_inertia_kg_m2"]
WING_MASS = wing_parameters["mass_kg"]
station_chords = (
    station_trailing_edge_x_mm - station_leading_edge_x_mm
) * SCALE_M_PER_MM
station_widths = np.diff(station_y_mm) * SCALE_M_PER_MM
wing_planform_area = float(
    np.sum(0.5 * (station_chords[:-1] + station_chords[1:]) * station_widths)
)
WING_DENSITY = WING_MASS / wing_planform_area

# The model 5 wing flaps about an axis through its shoulder, the origin of its points,
# parallel to the x axis. As in the CAD model, the two Wings' roots, and so their
# shoulders, are 410 millimeters apart, so each root is ROOT_OFFSET_Y from the plane of
# symmetry, and each Wing flaps about its own root's leading edge point (the default
# rotation point). Because the root is off the plane, the Airplane splits the symmetric
# Wing into two Wings that can flap separately. Don't move the roots much closer
# together: the two roots' bound vortices spin in opposite directions, and within a few
# centimeters of each other they throw the free wake's root vertices far out of place.
ROOT_OFFSET_Y = wing_parameters["meshing"]["root_offset_y_m"]

# The Wing's base flap angle is the stroke's middle. The first time step's Wings start
# at the flap angle of the drives' starting motor positions instead.
example_airplane = ps.geometry.airplane.Airplane(
    wings=[
        ps.geometry.wing.Wing(
            wing_cross_sections=wing_cross_sections,
            name="Main Wing",
            Ler_Gs_Cgs=(
                station_leading_edge_x_mm[0] * SCALE_M_PER_MM,
                ROOT_OFFSET_Y,
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
# to the supply voltage. The motor's, motor controller's, and gearbox's values are in
# mad_m13_kv90_powertrain.json.
TEST_VOLTAGE = motor_parameters["test_voltage_V"]
SUPPLY_VOLTAGE = motor_parameters["supply_voltage_V"]
measured_speeds = (
    np.array(motor_parameters["measured_speeds_rpm"])
    / TEST_VOLTAGE
    / 60.0
    * 2.0
    * np.pi
)
measured_torques = np.array(motor_parameters["measured_torques_N_m"]) / TEST_VOLTAGE
measured_currents = np.array(motor_parameters["measured_currents_A"]) / TEST_VOLTAGE
speed_torque_coefficients = np.polyfit(measured_speeds, measured_torques, 1)
speed_torque_coefficients[1] *= SUPPLY_VOLTAGE
torque_current_coefficients = np.polyfit(measured_currents, measured_torques, 1)
ROTOR_INERTIA = motor_parameters["rotor_inertia_kg_m2"]
MOTOR_CONTINUOUS_TORQUE = motor_parameters["continuous_torque_N_m"]
MOTOR_PEAK_DURATION = motor_parameters["peak_duration_s"]
ESC_PEAK_CURRENT = motor_controller_parameters["peak_current_A"]
ESC_CONTINUOUS_CURRENT = motor_controller_parameters["continuous_current_A"]
ESC_PEAK_DURATION = motor_controller_parameters["peak_duration_s"]
ESC_MAX_BRAKE_CURRENT = motor_controller_parameters["max_brake_current_A"]
GEARBOX_MAX_TORQUE = powertrain_parameters["gearbox"]["max_torque_N_m"]

# Define the controller's gains and the motor's damping. Each wing's inertia about the
# flap axis, WING_FLAP_INERTIA, is the model 5 wing's, defined with the wing above. It
# is one Wing's inertia because each FlapDrive below flaps one Wing. A FlapDrive's
# wing_flap_inertia is the combined inertia of all of its Wings, so a single motor that
# flaps both Wings would get 2 * WING_FLAP_INERTIA, and that motor would also carry both
# Wings' fluid torques, so size it for both. See the
# flap_driven_aeroelastic_unsteady_crank_shared_motor example.
PROPORTIONAL_GAIN = powertrain_parameters["feedback_gains"]["proportional"]
DERIVATIVE_GAIN = powertrain_parameters["feedback_gains"]["derivative"]
MOTOR_DAMPING = motor_parameters["damping_N_m_s_per_rad"]


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
        wing_density=WING_DENSITY,
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
