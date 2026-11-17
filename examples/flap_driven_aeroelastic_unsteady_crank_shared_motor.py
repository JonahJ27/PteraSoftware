"""Demonstrates running Ptera Software's
AeroelasticUnsteadyRingVortexLatticeMethodSolver with the FAAV's small (0.86 meter span)
flapping wing whose two halves are both driven by one motor, gearbox, and crank linkage
instead of by a prescribed flap angle, as on the FAAV.

This example is like the flap_driven_aeroelastic_unsteady_crank example, with two
changes. First, one FlapDrive flaps both Wings, instead of one FlapDrive per Wing. A
shared FlapDrive has one actuator state, so both Wings always have the same flap angle.
Its motor carries both Wings: the problem sums the two Wings' fluid torques about the
flap axis, and the FlapDrive's wing_flap_inertia is the two Wings' combined inertia.
Second, the hardware is the FAAV's (its wing, powertrain, and crank linkage) instead of
the pterosaur's.

The crank linkage is a crank_linkage.CrankLinkage, a powertrain with its own class that
conforms to the flap_drive.Transmission protocol, so it is passed to each FlapDrive
directly. The companion flap_driven_aeroelastic_unsteady example shows the other way to
supply a transmission: as your own functions, wrapped in a flap_drive.FlapTransmission.

The motor is the FAAV's T-Motor AT2303 1500KV. Its torque-speed curve, its peak and
continuous torques, its motor controller's peak and continuous currents, its brake
current, and its gearbox all limit the torque that it can apply, so the Wings can fall
behind their desired flap trajectory.
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
# directory next to this script, which the flap driven examples share: the FAAV small
# wing's outline and mass (faav_small_wing.json), the powertrain's motor, motor
# controller, gearbox, and feedback gains (faav_at2303_1500kv_powertrain.json), and the
# crank linkage's geometry (faav_crank_linkage.json). Edit those files to change the
# hardware, or load faav_large_wing.json for the FAAV's 530 millimeter wing. The flight
# condition and the simulation's settings are defined in this script, below.
PARAMETERS_DIRECTORY = Path(__file__).resolve().parent / "flap_drive_parameters"


def load_parameters(file_name: str) -> dict[str, Any]:
    """Loads one of the JSON files of parameters in PARAMETERS_DIRECTORY."""
    with open(PARAMETERS_DIRECTORY / file_name, encoding="utf-8") as file:
        parameters: dict[str, Any] = json.load(file)
    return parameters


wing_parameters = load_parameters("faav_small_wing.json")
powertrain_parameters = load_parameters("faav_at2303_1500kv_powertrain.json")
motor_parameters = powertrain_parameters["motor"]
motor_controller_parameters = powertrain_parameters["motor_controller"]
crank_parameters = load_parameters("faav_crank_linkage.json")

# Define the flight condition, the flap frequency, and the simulation's time steps. The
# time step is set by the number of time steps per flap, STEPS_PER_FLAP, so the flap
# stays equally well resolved at any flap frequency. Keep it at about 20 or more: each
# time step sheds one row of wake panels, and the torsional springs' sampled forcing
# loses accuracy as the time step grows relative to the flap period. The simulation runs
# for NUM_FLAPS flaps.
FLAP_FREQUENCY = 5.0
FLAP_PERIOD = 1.0 / FLAP_FREQUENCY
AIR_SPEED = 6.0
ANGLE_OF_ATTACK = 10.0
STEPS_PER_FLAP = 40
NUM_FLAPS = 3
DELTA_TIME = FLAP_PERIOD / STEPS_PER_FLAP
NUM_STEPS = NUM_FLAPS * STEPS_PER_FLAP
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

# Define the trajectory that the motor follows. "constant_speed" turns the motor at a
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

# Define the main Wing as the FAAV's small wing (the blade element theory code's
# calcFAAVWingGeom), whose outline and mass are in faav_small_wing.json. Its leading and
# trailing edge points are in millimeters, for the code's base wing of the outline
# reference length (two 330 millimeter wings), with the shoulder at the origin, the x
# axis pointing aft, and the y axis pointing out along the span. That code scales them
# to a wingspan of 2 * (spar length + shoulder offset) meters (here a 430 millimeter
# spar per side, with no offset from the shoulder), so SCALE_M_PER_MM converts them to
# meters at that wingspan. The leading edge is straight. The trailing edge starts half
# the wing's length aft of the shoulder, runs straight out to half the wing's length,
# and then curves around a quarter circle to the tip, where it meets the leading edge.
WINGSPAN = 2.0 * (
    wing_parameters["spar_length_m"] + wing_parameters["shoulder_offset_m"]
)
SCALE_M_PER_MM = WINGSPAN / wing_parameters["outline_reference_length_m"] / 1000.0
leading_edge_x_mm = wing_parameters["leading_edge_x_mm"]
leading_edge_y_mm = wing_parameters["leading_edge_y_mm"]
trailing_edge_x_mm = wing_parameters["trailing_edge_x_mm"]
trailing_edge_y_mm = wing_parameters["trailing_edge_y_mm"]

# Put a WingCrossSection at every spanwise station where the leading or the trailing
# edge has a point, so that the Wing's straight edges between WingCrossSections trace
# the outline's points exactly. A leading edge point within MERGE_TOLERANCE_MM of a
# trailing edge point is merged into that point's station instead of making a
# WingCrossSection a few millimeters wide. (The FAAV wing's only leading edge points are
# at its root and tip, which are also trailing edge points.)
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

# The FAAV wing's chord is zero at the tip, where its leading and trailing edges meet,
# but a WingCrossSection's chord must be positive, so the tip WingCrossSection gets a
# small chord (ROOT_CHORD, the smallest chord a station can have) instead. Each section
# between WingCrossSections gets spanwise panels about SPANWISE_PANEL_WIDTH wide.
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

# Define the FAAV wing's mass properties, which have two parts (see
# faav_small_wing.json). The leading edge is a flat carbon fiber plate (the blade
# element theory code's leading_edge_shape = 1), and the membrane has the FAAV membrane
# areal density from that code's BET_plotter. Both parts' masses and inertias about the
# shoulder are computed as in that code's precomputeWingGeometry. The motor flaps the
# whole wing, so each Wing's inertia, WING_FLAP_INERTIA, is the sum of the plate's and
# the membrane's. The torsional springs twist each strip about its leading edge, where
# the plate sits, so the plate adds almost nothing to their twist inertia, and the
# springs get only the membrane: wing_density is the membrane's areal density. (The
# torsional spring model currently triples every strip's mass. See item 3 of
# docs/FLAP_DRIVE_TODO.md.)
leading_edge_parameters = wing_parameters["leading_edge"]
membrane_parameters = wing_parameters["membrane"]
WING_FLAP_INERTIA = (
    leading_edge_parameters["flap_inertia_kg_m2"]
    + membrane_parameters["flap_inertia_kg_m2"]
)
WING_DENSITY = membrane_parameters["areal_density_kg_m2"]

# The FAAV wing flaps about an axis through its shoulder, the origin of its points,
# parallel to the x axis. As on the FAAV, the two Wings' roots, and so their shoulders,
# are 25.3 millimeters apart (center to center), so each root is ROOT_OFFSET_Y from the
# plane of symmetry, and each Wing flaps about its own root's leading edge point (the
# default rotation point). Because the root is off the plane, the Airplane splits the
# symmetric Wing into two Wings that can flap separately. The two roots' bound vortices
# spin in opposite directions, and close together they can throw the free wake's root
# vertices out of place (see item 24 of docs/FLAP_DRIVE_TODO.md), so check the wake near
# the plane of symmetry if you move the roots closer together.
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

# Define the motor. This is the T-Motor AT2303 1500KV (motor 4), measured at 7.2 V and
# run at 16 V. Its torque-speed line and its torque-current line are fit to the same two
# measured points as in the blade element theory motor limit simulation, with each value
# divided by the test voltage before fitting, and the torque-speed line's intercept is
# then scaled up to the supply voltage. The motor's, motor controller's, and gearbox's
# values are in faav_at2303_1500kv_powertrain.json.
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
# flap axis, WING_FLAP_INERTIA, is the FAAV wing's, defined with the wing above. The
# motor drives both Wings, so its FlapDrive gets their combined inertia below.
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
        # The desired torque is the feedforward torque plus feedback on the motor's
        # position and velocity errors. The FlapDrive computes the feedforward torque,
        # inputs.feedforward_force, as the motor torque needed to follow the desired
        # trajectory. It is the sum of four terms, each of which the FlapDrive switch of
        # the same name turns on or off:
        #
        #   1. inertial (feedforward_inertial): accelerates the rotor and the Wings
        #   2. damping (feedforward_damping): cancels the motor's damping
        #   3. passive (feedforward_passive): cancels any passive springs (none here)
        #   4. aerodynamic (feedforward_aerodynamic): cancels the fluid torque on the
        #      Wings, which comes from one of two sources:
        #      - True (used here): the previous time step's UVLM torque at the Wings'
        #        actual motion. No blade element theory model is used.
        #      - A callable, such as a blade element theory model: it predicts the
        #        torque, for example at the desired wing velocity, like the blade
        #        element theory code's feedforward.
        #
        # Each term is also available on its own, for example as
        # inputs.feedforward_aerodynamic_force. Whichever aerodynamic source is chosen
        # only changes what the controller asks for. The Wings always feel the UVLM
        # torque.
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


# Define one FlapDrive for both halves of the main Wing, so one motor and one saturation
# controller, which keeps track of the motor's peak allowances, flap both Wings. The
# Airplane split the symmetric main Wing into the Wing (index 0) and its reflected Wing
# (index 1), so the drive's wing_indices are (0, 1). The problem sums the two Wings'
# fluid torques about the flap axis into the one torque that loads the motor. The
# reflected Wing's torque is measured in its own (mirrored) flap direction, so for a
# symmetric flap the two torques have the same sign and add instead of cancelling.
#
# wing_flap_inertia is the combined inertia of all of the drive's Wings, so it is twice
# one Wing's. It reflects through the linkage to the motor as J^2 * wing_flap_inertia,
# so the motor sees twice the reflected inertia that one Wing would give it.
#
# The drive starts at its reference's first motor position and velocity. Its feedforward
# reads the reference at the entry nearest the motor's actual position
# (use_position_lookup=True), like the blade element theory code's use_closest_pos,
# which works because the motor keeps turning one way, so the reference's positions are
# monotonic. Set it to False to read the reference at the current time instead.
#
# The aerodynamic model is the UVLM only. No blade element theory model is used yet.
# feedforward_aerodynamic=True makes the controller's feedforward cancel the previous
# time step's UVLM torque, summed over both Wings. The Wings themselves always feel the
# UVLM torque, whatever this is set to. To make the controller use a blade element
# theory model instead, replace True with a function that takes a
# flap_drive.AerodynamicModelInputs and returns the predicted fluid torque on both Wings
# together (N*m, positive toward increasing flap angle). Only the controller's
# feedforward would change.
SHARED_WING_INDICES = (0, 1)
saturation_controller, motor, max_torque_possible_log = make_motor()
saturation_controllers = [saturation_controller]
max_torque_possible_logs = [max_torque_possible_log]
example_flap_drives = [
    ps.flap_drive.FlapDrive(
        wing_indices=SHARED_WING_INDICES,
        transmission=transmission,
        actuator=motor,
        actuator_inertia=ROTOR_INERTIA,
        actuator_damping=MOTOR_DAMPING,
        wing_flap_inertia=len(SHARED_WING_INDICES) * WING_FLAP_INERTIA,
        reference=flap_reference,
        use_position_lookup=True,
        # The UVLM torque, not a blade element theory model.
        feedforward_aerodynamic=True,
    )
]

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

# Both Wings share one actuator, so every time step's Wings have the same flap angle,
# the one the drive reached.
for step, steady_problem in enumerate(example_problem.steady_problems):
    drive_angle_deg = np.rad2deg(example_problem.flap_angles_rad[0][step])
    for wing in steady_problem.airplanes[0].wings:
        assert abs(wing.angles_Gs_to_Wn_ixyz[0] - drive_angle_deg) < 1e-6
print("Both Wings followed the shared motor at every time step.")

# The plots below mirror the blade element theory code's runSim figures, for the one
# motor, whose histories are the first (and only) entries of the problem's flap
# histories. The desired trajectory is plotted on the reference's fine time grid
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
