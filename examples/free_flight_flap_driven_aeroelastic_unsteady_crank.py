"""Demonstrates running Ptera Software's
FreeFlightAeroelasticUnsteadyRingVortexLatticeMethodSolver with the blade element theory
code's model 5 (4.3 meter span) flapping wing that is driven by motors, gearboxes, and
crank linkages, while the vehicle flies freely.

This is the free flight version of the flap_driven_aeroelastic_unsteady_crank example.
The wing, the crank linkage, the motors, their limits, and the controller are the same.
What changes is that the vehicle is no longer held at a fixed flight condition: after a
prescribed phase that lets the wake develop, MuJoCo integrates its six-degree-of-freedom
rigid body dynamics under the aerodynamic loads, gravity, and its mass and inertia. A
V-tail is added for pitch stability.

The airframe's mass and inertia below are placeholders, scaled from the
free_flight_aeroelastic_unsteady_first_order_deformation_flapping example, and the
initial flight condition is not trimmed, so the vehicle will climb, sink, or pitch.
Replace them with the robot's measured values.

What the rigid body dynamics don't include: MuJoCo models the vehicle as one rigid body
with constant mass and inertia, so the motors' reaction torques on the airframe, the
flapping wings' inertial loads on the airframe, and the movement of the CG as the wings
flap are not modeled. See FreeFlightFlapDrivenAeroelasticUnsteadyProblem and
docs/FLAP_DRIVE.md.

The script will likely take a few minutes to run.
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
# wing's outline and mass (pterosaur_model_5_wing.json), the powertrain's motor, motor
# controller, gearbox, and feedback gains (pterosaur_mad_m13_kv90_powertrain.json), and
# the crank linkage's geometry (pterosaur_crank_linkage.json). Edit those files to
# change the hardware. The flight condition and the simulation's settings are defined in
# this script, below.
PARAMETERS_DIRECTORY = Path(__file__).resolve().parent / "flap_drive_parameters"


def load_parameters(file_name: str) -> dict[str, Any]:
    """Loads one of the JSON files of parameters in PARAMETERS_DIRECTORY."""
    with open(PARAMETERS_DIRECTORY / file_name, encoding="utf-8") as file:
        parameters: dict[str, Any] = json.load(file)
    return parameters


wing_parameters = load_parameters("pterosaur_model_5_wing.json")
powertrain_parameters = load_parameters("pterosaur_mad_m13_kv90_powertrain.json")
motor_parameters = powertrain_parameters["motor"]
motor_controller_parameters = powertrain_parameters["motor_controller"]
crank_parameters = load_parameters("pterosaur_crank_linkage.json")

# Define the initial flight condition, the flap frequency, and the simulation's time
# steps. The vehicle holds its initial state for PRESCRIBED_NUM_STEPS time steps (one
# wingbeat), so the wake can develop, and then flies freely for the rest.
FLAP_FREQUENCY = 3.5
FLAP_PERIOD = 1.0 / FLAP_FREQUENCY
AIR_SPEED = 20.0
ANGLE_OF_ATTACK = 10.0
DELTA_TIME = 0.01
PRESCRIBED_NUM_STEPS = 50
FREE_NUM_STEPS = 100
NUM_STEPS = PRESCRIBED_NUM_STEPS + FREE_NUM_STEPS
DURATION = NUM_STEPS * DELTA_TIME

# Define the airframe's mass properties. These are placeholders: the inertia is the
# free_flight_aeroelastic_unsteady_first_order_deformation_flapping example's, scaled by
# this vehicle's mass and by the square of its smaller size. The rigid body keeps them
# constant, so they should include the wings and the motors, for example at the wings'
# mean flap angle.
GRAVITY = 9.80665
ROBOT_MASS = 30.0
ROBOT_WEIGHT = ROBOT_MASS * GRAVITY
I_BP1_CgP1 = (
    (27.0, 0.0, -8.0),
    (0.0, 70.0, 0.0),
    (-8.0, 0.0, 89.0),
)

# Define the powertrain. A motor turns a gearbox, whose output turns the gearbox crank
# of a crank linkage. A rigid link joins the gearbox crank to a wing crank that turns
# with the Wing. The linkage's geometry is given in the plane of the flap. See the
# flap_driven_aeroelastic_unsteady_crank example for more about the options here.
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
# gearbox crank.
min_flap_angle_rad, max_flap_angle_rad = transmission.angle_range()
WING_BEAT_AMPLITUDE_RAD = max_flap_angle_rad - min_flap_angle_rad
DIHEDRAL_RAD = (max_flap_angle_rad + min_flap_angle_rad) / 2.0

# Define the trajectory that each motor follows: the motor turns at a constant speed,
# starting with the gearbox crank at a quarter turn, so the wing's turnarounds land at
# the linkage's dead centers. The reference's period is the flap's period, one turn of
# the gearbox crank, which is GEAR_RATIO turns of the motor. It sets the window of the
# cycle averaged results.
flap_reference = ps.flap_drive.FlapReference.constant_velocity(
    start_position=GEAR_RATIO * np.pi / 2.0,
    velocity=GEAR_RATIO * 2.0 * np.pi * FLAP_FREQUENCY,
    duration=DURATION,
    period=FLAP_PERIOD,
)

# Define the main Wing as the blade element theory code's model 5 wing (calcWingGeom
# with model_type = 5), whose outline and mass are in pterosaur_model_5_wing.json.
# calcWingGeom gives the wing's leading and trailing edge points in millimeters, for a
# wing of the outline reference length (1.2 meters), with the shoulder at the origin,
# the x axis pointing aft, and the y axis pointing out along the span. That code scales
# them to its wingspan of 2 * (spar length + shoulder offset) meters (the 2.09 meter
# spar plus its 7.5 centimeter offset from the shoulder, per side), so SCALE_M_PER_MM
# converts them to meters at that wingspan. The trailing edge runs from the shoulder to
# the wing's connection point and then out to the tip, and the tip is closed by a
# chordwise edge from the last trailing edge point to the last leading edge point.
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

# Define the model 5 wing's mass properties, which have two parts (see
# pterosaur_model_5_wing.json). The leading edge is a carbon fiber tube (the blade
# element theory code's leading_edge_shape = 6), and the membrane has the measured areal
# density from that code's runSim. Both parts' masses and inertias about the shoulder
# are computed as in that code's precomputeWingGeometry. The motor flaps the whole wing,
# so each FlapDrive's wing inertia, WING_FLAP_INERTIA, is the sum of the tube's and the
# membrane's. The torsional springs twist each strip about its leading edge, where the
# tube sits, so the tube adds almost nothing to their twist inertia, and the springs get
# only the membrane: wing_density is the membrane's areal density. (The torsional spring
# model currently triples every strip's mass. See item 3 of docs/FLAP_DRIVE_TODO.md.)
leading_edge_parameters = wing_parameters["leading_edge"]
membrane_parameters = wing_parameters["membrane"]
WING_FLAP_INERTIA = (
    leading_edge_parameters["flap_inertia_kg_m2"]
    + membrane_parameters["flap_inertia_kg_m2"]
)
WING_DENSITY = membrane_parameters["areal_density_kg_m2"]
# The model 5 wing flaps about an axis through its shoulder, the origin of its points,
# parallel to the x axis. As in the CAD model, the two Wings' roots, and so their
# shoulders, are 410 millimeters apart, so each root is ROOT_OFFSET_Y from the plane of
# symmetry, and each Wing flaps about its own root's leading edge point (the default
# rotation point). Because the root is off the plane, the Airplane splits the symmetric
# Wing into two Wings that can flap separately. Don't move the roots much closer
# together: the two roots' bound vortices spin in opposite directions, and within a few
# centimeters of each other they throw the free wake's root vertices far out of place.
ROOT_OFFSET_Y = wing_parameters["meshing"]["root_offset_y_m"]

# Define the V-tail, a rigid tail for pitch stability, 3 meters behind the wing's root.
# It sits on the plane of symmetry, so it stays one Wing that is mirrored about that
# plane, and it isn't split into strips, because it doesn't deform.
v_tail = ps.geometry.wing.Wing(
    wing_cross_sections=[
        ps.geometry.wing_cross_section.WingCrossSection(
            num_spanwise_panels=6,
            chord=0.6,
            Lp_Wcsp_Lpp=(0.0, 0.0, 0.0),
            angles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
            control_surface_symmetry_type="symmetric",
            control_surface_hinge_point=0.75,
            control_surface_deflection=0.0,
            spanwise_spacing="uniform",
            airfoil=ps.geometry.airfoil.Airfoil(name="naca0012"),
        ),
        ps.geometry.wing_cross_section.WingCrossSection(
            num_spanwise_panels=None,
            chord=0.4,
            Lp_Wcsp_Lpp=(0.2, 0.8, 0.0),
            angles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
            control_surface_symmetry_type="symmetric",
            control_surface_hinge_point=0.75,
            control_surface_deflection=0.0,
            spanwise_spacing=None,
            airfoil=ps.geometry.airfoil.Airfoil(name="naca0012"),
        ),
    ],
    name="V-Tail",
    Ler_Gs_Cgs=(3.0, 0.0, 0.0),
    angles_Gs_to_Wn_ixyz=(0.0, -5.0, 0.0),
    symmetric=True,
    mirror_only=False,
    symmetryNormal_G=(0.0, 1.0, 0.0),
    symmetryPoint_G_Cg=(0.0, 0.0, 0.0),
    explode_into_strips=False,
    num_chordwise_panels=4,
    chordwise_spacing="uniform",
)

# The Airplane's weight must equal its mass times the gravitational acceleration, which
# the problem checks.
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
        v_tail,
    ],
    name="Example Airplane",
    Cg_GP1_CgP1=(0.0, 0.0, 0.0),
    weight=ROBOT_WEIGHT,
)

# The main Wing has type 5 symmetry, so the Airplane split it into the Wing and a
# reflected Wing (indices 0 and 1). The V-tail is index 2. Each main Wing gets a
# FlapDrivenAeroelasticWingMovement, built around its own WingCrossSections. A
# FlapDrivenAeroelasticWingMovement can't prescribe a flap motion: its angular motion
# parameters (such as ampAnglesYZ_Gs_to_Wn_ixyz) take only the y and z components, and
# the x (flap) component is held static. The FlapDrives' references are the only
# definition of the flap trajectory.
main_wing_indices = (0, 1)
wing_movements: list[
    ps.movements.aeroelastic_wing_movement.AeroelasticWingMovement
    | ps.movements.wing_movement.WingMovement
] = []
for wing_index in main_wing_indices:
    wing = example_airplane.wings[wing_index]
    wing_movements.append(
        ps.movements.flap_driven_aeroelastic_wing_movement.FlapDrivenAeroelasticWingMovement(
            base_wing=wing,
            wing_cross_section_movements=[
                ps.movements.aeroelastic_wing_cross_section_movement.AeroelasticWingCrossSectionMovement(
                    base_wing_cross_section=wing_cross_section,
                )
                for wing_cross_section in wing.wing_cross_sections
            ],
        )
    )

# The V-tail doesn't move or deform, so it gets a plain WingMovement.
v_tail_wing = example_airplane.wings[2]
wing_movements.append(
    ps.movements.wing_movement.WingMovement(
        base_wing=v_tail_wing,
        wing_cross_section_movements=[
            ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
                base_wing_cross_section=wing_cross_section,
            )
            for wing_cross_section in v_tail_wing.wing_cross_sections
        ],
    )
)

example_airplane_movement = (
    ps.movements.aeroelastic_airplane_movement.AeroelasticAirplaneMovement(
        base_airplane=example_airplane,
        wing_movements=wing_movements,
    )
)

# Define the initial flight condition. The body is pitched up by the angle of attack, so
# the vehicle starts in level flight. Gravity points down the Earth axes' z axis. Free
# flight never applies externalFX_W, so it must be zero.
example_operating_point = ps.operating_point.OperatingPoint(
    rho=1.225,
    vCg__E=AIR_SPEED,
    alpha=ANGLE_OF_ATTACK,
    beta=0.0,
    angles_E_to_BP1_izyx=(0.0, ANGLE_OF_ATTACK, 0.0),
    externalFX_W=0.0,
    nu=15.06e-6,
    g_E=(0.0, 0.0, GRAVITY),
)

# The FreeFlightOperatingPointMovement holds only the initial OperatingPoint. The solver
# adds one OperatingPoint per time step from the rigid body dynamics.
example_operating_point_movement = (
    ps.movements.free_flight_operating_point_movement.FreeFlightOperatingPointMovement(
        base_operating_point=example_operating_point,
    )
)

example_movement = ps.movements.free_flight_movement.FreeFlightAeroelasticMovement(
    airplane_movements=[example_airplane_movement],
    operating_point_movement=example_operating_point_movement,
    delta_time=DELTA_TIME,
    prescribed_num_steps=PRESCRIBED_NUM_STEPS,
    free_num_steps=FREE_NUM_STEPS,
)

# Define the motor (the MAD M13 EEE KV90 at 48 V), its motor controller (an ODrive Pro),
# and its gearbox, as in the flap_driven_aeroelastic_unsteady_crank example. The
# motor's, motor controller's, and gearbox's values are in
# pterosaur_mad_m13_kv90_powertrain.json.
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
        # see the same UVLM torque.
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
        # leaves the limit at each time step's instant. The motors advance once per time
        # step, before the rigid body's sub-iteration, so the sub-iteration doesn't call
        # this function.
        max_torque_possible_log[round(inputs.time, 9)] = (
            saturation_info.max_torque_possible
        )
        return torque_applied

    return saturation_controller, motor, max_torque_possible_log


# Define one FlapDrive for each half of the main Wing, so each Wing has its own motor
# and its own saturation controller. Each drive's feedforward reads the reference at the
# entry nearest the motor's actual position (use_position_lookup=True).
#
# The aerodynamic model is the UVLM only. No blade element theory model is used yet.
# feedforward_aerodynamic=True makes the controller's feedforward cancel the previous
# time step's UVLM torque. The wing itself always feels the UVLM torque, whatever this
# is set to. To make the controller use a blade element theory model instead, replace
# True with a function that accepts a ps.flap_drive.AerodynamicModelInputs and returns
# the predicted fluid torque on the wing (N*m, positive toward increasing flap angle).
# Its operating_point field is the vehicle's actual state from the rigid body dynamics,
# so the model sees the airspeed and body rates change as the vehicle flies. For
# example:
#
#     def blade_element_torque(inputs: ps.flap_drive.AerodynamicModelInputs) -> float:
#         air_speed = inputs.operating_point.vCg__E
#         flap_angle = inputs.angleXRad_Gs_to_Wn_ixyz
#         desired_flap_rate = inputs.desiredAngleDerivativeXRad_Gs_to_Wn_ixyz
#         ...
#         return predicted_torque
saturation_controllers = []
max_torque_possible_logs = []
example_flap_drives = []
for wing_index in main_wing_indices:
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

# Define the problem. It is a FreeFlightAeroelasticUnsteadyProblem that also takes the
# FlapDrives. The FlapDrives and the torsional springs advance once per time step from
# the previous time step's loads, and the rigid body is then iterated to agreement with
# the loads on the resulting geometry. The rigid body is one body with the constant mass
# and inertia given here, so the motors' reaction torques and the wings' inertial loads
# don't act on it.
example_problem = ps.free_flight_flap_driven_aeroelastic_unsteady_problem.FreeFlightFlapDrivenAeroelasticUnsteadyProblem(
    movement=example_movement,
    wing_density=WING_DENSITY * 1000,
    spring_constant_rad=1000.0,
    damping_constant_rad=1000.0,
    mass=ROBOT_MASS,
    I_BP1_CgP1=I_BP1_CgP1,
    flap_drives=example_flap_drives,
    external_loads_fn=None,
    step_discards=5,
)

example_solver = ps.free_flight_aeroelastic_unsteady_ring_vortex_lattice_method.FreeFlightAeroelasticUnsteadyRingVortexLatticeMethodSolver(
    free_flight_unsteady_problem=example_problem,
)

example_solver.run(prescribed_wake=False, show_progress=True)
print("The simulation has finished.")

# Log the loads, and the vehicle's initial and final six-degree-of-freedom state.
ps.output.log_results(solver=example_solver)

# The plots below mirror the blade element theory code's runSim figures, for the first
# Wing's motor, as in the flap_driven_aeroelastic_unsteady_crank example.
step_times = np.arange(NUM_STEPS) * DELTA_TIME
on_steps = flap_reference.times <= step_times[-1]
reference_times = flap_reference.times[on_steps]

motor_pos_des = flap_reference.positions[on_steps]
motor_velo_des = flap_reference.velocities[on_steps]
motor_accel_des = flap_reference.accelerations[on_steps]

phi_des = np.array([transmission.evaluate_angle(q) for q in motor_pos_des])
jacobian_des = np.array([transmission.evaluate_jacobian(q) for q in motor_pos_des])
hessian_des = np.array([transmission.evaluate_hessian(q) for q in motor_pos_des])
phi_dot_des = jacobian_des * motor_velo_des
phi_ddot_des = hessian_des * motor_velo_des**2 + jacobian_des * motor_accel_des

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

# Plot the wing kinematics.
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
    axis.plot(step_times, actual, color + "-", linewidth=3.0, alpha=0.5, label="Actual")
    axis.plot(reference_times, desired, "k:", linewidth=1.5, zorder=3, label="Desired")
    axis.axvline(PRESCRIBED_NUM_STEPS * DELTA_TIME, color="0.5", linestyle="--")
    axis.set_ylabel(label)
    axis.legend(loc="best")
    axis.grid(True)
wing_axes[0].set_title("Wing Kinematics (dashed line: start of free flight)")
wing_axes[2].set_xlabel("Time [s]")
wing_figure.tight_layout()

# Plot the motor dynamics.
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

# Get the vehicle's state at every time step. In free flight, the solver adds one
# OperatingPoint per time step from the rigid body dynamics, which holds the CG's
# position (in Earth axes, relative to the Earth origin, in meters), the speed, the
# aerodynamic angles, the attitude (in degrees), and the body rates (in degrees per
# second). The Earth axes' z axis points down, so the altitude is -z.
operating_points = example_movement.operating_point_movement.operating_points
state_times = np.arange(len(operating_points)) * DELTA_TIME
positions_E_Eo = np.array(
    [operating_point.CgP1_E_Eo for operating_point in operating_points]
)
downrange = positions_E_Eo[:, 0]
altitude = -positions_E_Eo[:, 2]
speeds = np.array([operating_point.vCg__E for operating_point in operating_points])
alphas = np.array([operating_point.alpha for operating_point in operating_points])
pitch_angles = np.array(
    [operating_point.angles_E_to_BP1_izyx[1] for operating_point in operating_points]
)
omegas_BP1__E = np.array(
    [operating_point.omegas_BP1__E for operating_point in operating_points]
)
free_flight_start_time = PRESCRIBED_NUM_STEPS * DELTA_TIME

# Plot the vehicle's trajectory next to the flap motion that produced it. The left
# column shows the flight path and the speed and altitude, the right column shows the
# flap angles and the attitude. The dashed line marks the start of free flight, before
# which the body holds its initial state.
trajectory_figure, trajectory_axes = plt.subplots(
    3, 2, figsize=(14, 11), num="Vehicle Trajectory"
)

path_axis = trajectory_axes[0, 0]
path_axis.plot(downrange, altitude, "b-", linewidth=2.0)
path_axis.plot(downrange[0], altitude[0], "go", label="Start")
path_axis.plot(
    downrange[PRESCRIBED_NUM_STEPS],
    altitude[PRESCRIBED_NUM_STEPS],
    "ks",
    label="Free flight starts",
)
path_axis.plot(downrange[-1], altitude[-1], "ro", label="End")
path_axis.set_xlabel("Downrange (Earth x) [m]")
path_axis.set_ylabel("Altitude (Earth -z) [m]")
path_axis.set_title("Flight Path")
path_axis.legend(loc="best")
path_axis.grid(True)

flap_axis = trajectory_axes[0, 1]
for drive_position, wing_index in enumerate(main_wing_indices):
    flap_axis.plot(
        step_times,
        np.rad2deg(example_problem.flap_angles_rad[drive_position]),
        label=f"Wing {wing_index} actual",
    )
flap_axis.plot(reference_times, np.rad2deg(phi_des), "k:", label="Desired")
flap_axis.set_ylabel("Flap Angle [deg]")
flap_axis.set_title("Flap Angles")
flap_axis.legend(loc="best")

speed_axis = trajectory_axes[1, 0]
speed_axis.plot(state_times, speeds, "r-")
speed_axis.set_ylabel("Speed [m/s]")
speed_axis.set_title("Speed")

altitude_axis = trajectory_axes[1, 1]
altitude_axis.plot(state_times, altitude, "b-")
altitude_axis.set_ylabel("Altitude [m]")
altitude_axis.set_title("Altitude")

angle_axis = trajectory_axes[2, 0]
angle_axis.plot(state_times, alphas, "g-", label="Angle of attack")
angle_axis.plot(state_times, pitch_angles, "m-", label="Pitch angle")
angle_axis.set_ylabel("Angle [deg]")
angle_axis.set_title("Angle of Attack and Pitch Angle")
angle_axis.legend(loc="best")

rate_axis = trajectory_axes[2, 1]
for component, name in enumerate(("Roll", "Pitch", "Yaw")):
    rate_axis.plot(state_times, omegas_BP1__E[:, component], label=f"{name} rate")
rate_axis.set_ylabel("Body Rate [deg/s]")
rate_axis.set_title("Body Rates")
rate_axis.legend(loc="best")

for axis in (flap_axis, speed_axis, altitude_axis, angle_axis, rate_axis):
    axis.axvline(free_flight_start_time, color="0.5", linestyle="--")
    axis.set_xlabel("Time [s]")
    axis.grid(True)
trajectory_figure.tight_layout()

# Draw the vehicle at its final pose. For a free flight solver, the geometry is drawn in
# Earth axes where and how the vehicle ended up after the final time step. Press any key
# to close the plotter after it draws the output.
ps.output.draw(
    solver=example_solver,
    scalar_type="lift",
    show_streamlines=True,
    show_wake_vortices=False,
    save=True,
)

# Animate the vehicle flying along its computed trajectory, producing an animated WebP
# saved in the same directory as this script. Press any key, after orienting the view,
# to begin the animation.
ps.output.animate(
    unsteady_solver=example_solver,
    scalar_type="lift",
    show_wake_vortices=True,
    save=True,
)

# Plot the loads and the vehicle's six-degree-of-freedom state against time, as the
# other free flight examples do: the first Airplane's position, velocity, orientation,
# angular velocity, and aerodynamic angles. With show=True, this also shows the figures
# above, all at once.
ps.output.plot_results_versus_time(
    unsteady_solver=example_solver,
    show=True,
    save=True,
)
