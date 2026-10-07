"""Demonstrates running Ptera Software's FreeFlightAeroelasticUnsteadyRingVortexLatticeMethodSolver
with a flapping-wing airplane whose main wing deforms under its own aerodynamic loads

The main wing flaps symmetrically while the unsteady aerodynamics are coupled to
MuJoCo's rigid body dynamics, so the airplane flies a free six-degree-of-freedom
trajectory through the scene under the loads produced by its own flapping motion.

The script will likely take several minutes to run, and will log simulation progress and
results in a log file.
"""

import logging

# First, import the software's main package. Note that if you wished to import this
# software into another package, you would first install it by running "pip install
# pterasoftware" in your terminal.
import pterasoftware as ps

# Configure logging to write info level messages to a file. To display log messages on
# the console alongside progress bars instead, omit the handler argument.
ps.set_up_logging(level="Info", handler=logging.FileHandler("example_solver.log"))


# Create an Airplane with our custom geometry. I am going to declare every parameter for
# Airplane, even though most of them have usable default values. This is for educational
# purposes, but keep in mind that it makes the code much longer than it needs to be. For
# details about each parameter, read the detailed class docstring. The same caveats
# apply to the other classes, methods, and functions I call in this script.

# Initialize the WingCrossSection parameters. Define the offsets for the spacing.
NUM_SPANWISE_PANELS = 2
Lp_Wcsp_Lpp_Offsets = (0.1, 0.5, 0.0)
CROSS_SECTION_CHORDS = [1.75, 1.75, 1.75, 1.75, 1.65, 1.55, 1.4, 1.2, 1.0]
wing_cross_sections = []

#DEFINE ROBOT WEIGHT
ROBOT_WEIGHT = 420 #in newtons

#DEFINE FLAP PARAMS
FLAP_PERIOD = 1.0
FLAP_AMPLITUDE = 30
FLAP_CYCLES = 6
dt_des = .03
PRESCRIBED_RIGID_STEPS = 10

#WING PARAMS
WING_DENSITY = 6
SPRING_CONSTANT_RAD = 1000.0
DAMPING_CONSTANT_RAD =  1000.0

# Define WingCrossSections with a variable set of chords. All WingCrossSections for the
# deformation simulation have num_spanwise_panels = 2 (except the wing tip, which is
# always None). This is because each strip of WingCrossSection deforms independently as
# a torsional spring, and that model only works well if the strips are thin. To go
# thinner for the same base definition, increase the number of spanwise panels and set
# the explode_into_strips parameter to True on the Wing, which splits it back into
# single strips for deformation.
for i in range(len(CROSS_SECTION_CHORDS)):
    wing_cross_sections.append(
        ps.geometry.wing_cross_section.WingCrossSection(
            num_spanwise_panels=(
                NUM_SPANWISE_PANELS if i < len(CROSS_SECTION_CHORDS) - 1 else None
            ),
            chord=CROSS_SECTION_CHORDS[i],
            Lp_Wcsp_Lpp=Lp_Wcsp_Lpp_Offsets if i > 0 else (0.0, 0.0, 0.0),
            angles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
            control_surface_symmetry_type="symmetric",
            control_surface_hinge_point=0.75,
            control_surface_deflection=0.0,
            spanwise_spacing="uniform" if i < len(CROSS_SECTION_CHORDS) - 1 else None,
            airfoil=ps.geometry.airfoil.Airfoil(
                name="naca2412",
                outline_A_Lp=None,
                resample=True,
                n_points_per_side=400,
            ),
        )
    )


# Define the primary Wing. Note that the explode_into_strips parameter is set to True,
# which means that the Wing will be split into strips for deformation, and each strip
# will be modeled as a torsional spring.
wing_1 = ps.geometry.wing.Wing(
    wing_cross_sections=wing_cross_sections,
    name="Main Wing",
    Ler_Gs_Cgs=(0.0, 0.5, 0.0),
    angles_Gs_to_Wn_ixyz=(0.0, 0.0, 0.0),
    symmetric=True,
    mirror_only=False,
    symmetryNormal_G=(0.0, 1.0, 0.0),
    symmetryPoint_G_Cg=(0.0, 0.0, 0.0),
    explode_into_strips=True,
    num_chordwise_panels=6,
    chordwise_spacing="uniform",
)

# Actually generate the Airplane. The V-tail is added as a second lifting surface but is
# not split into deformation strips, because it follows prescribed rigid motion rather
# than deforming. The solver deforms each Wing from its own aerodynamic loads, but only
# Wings backed by an AeroelasticWingMovement. The V-tail uses a standard WingMovement
# and stays rigid. Leaving the tail rigid also keeps its movement definition simpler,
# since it then needs no per-strip WingCrossSections.
example_airplane = ps.geometry.airplane.Airplane(
    wings=[
        wing_1,
        ps.geometry.wing.Wing(
            wing_cross_sections=[
                ps.geometry.wing_cross_section.WingCrossSection(
                    num_spanwise_panels=8,
                    chord=1.5,
                    Lp_Wcsp_Lpp=(0.0, 0.0, 0.0),
                    angles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
                    control_surface_symmetry_type="symmetric",
                    control_surface_hinge_point=0.75,
                    control_surface_deflection=0.0,
                    spanwise_spacing="uniform",
                    airfoil=ps.geometry.airfoil.Airfoil(
                        name="naca0012",
                        outline_A_Lp=None,
                        resample=True,
                        n_points_per_side=400,
                    ),
                ),
                ps.geometry.wing_cross_section.WingCrossSection(
                    num_spanwise_panels=None,
                    chord=1.0,
                    Lp_Wcsp_Lpp=(0.5, 2.0, 0.0),
                    angles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
                    control_surface_symmetry_type="symmetric",
                    control_surface_hinge_point=0.75,
                    control_surface_deflection=0.0,
                    spanwise_spacing=None,
                    airfoil=ps.geometry.airfoil.Airfoil(
                        name="naca0012",
                        outline_A_Lp=None,
                        resample=True,
                        n_points_per_side=400,
                    ),
                ),
            ],
            name="V-Tail",
            Ler_Gs_Cgs=(5.0, 0.0, 0.0),
            angles_Gs_to_Wn_ixyz=(0.0, -5.0, 0.0),
            symmetric=True,
            mirror_only=False,
            symmetryNormal_G=(0.0, 1.0, 0.0),
            symmetryPoint_G_Cg=(0.0, 0.0, 0.0),
            explode_into_strips=False,
            num_chordwise_panels=6,
            chordwise_spacing="uniform",
        ),
    ],
    name="Example Airplane",
    Cg_GP1_CgP1=(0.0, 0.0, 0.0),
    weight=ROBOT_WEIGHT,
    c_ref=None,
    b_ref=None,
)



# The main Wing was defined to have symmetric = True, mirror_only = False, and with a
# symmetry plane offset non-coincident with the Wing's axes yz plane. Therefore, that
# Wing had type 5 symmetry (see the Wing class documentation for more details on
# symmetry types). Therefore, it was actually split into two Wings, with the second Wing
# being a reflected version of the first. Therefore, we need to define a WingMovement
# for this reflected Wing. To start, we'll first define the reflected main Wing's root
# and tip WingCrossSections' WingCrossSectionMovements.

# Define the WingCrossSectionMovement parameters.
DEPHASE_X = 0.0
PERIOD_X = 0.0
AMPLITUDE_X = 0.0

DEPHASE_Y = 0.0
PERIOD_Y = 0.0
AMPLITUDE_Y = 0.0

DEPHASE_Z = 0.0
PERIOD_Z = 0.0
AMPLITUDE_Z = 0.0

# Create a list of WingCrossSectionMovements for the main Wing.
main_wing_cross_section_movements_list: list[
    ps.movements.aeroelastic_wing_cross_section_movement.AeroelasticWingCrossSectionMovement
] = []

# Create a list of WingCrossSectionMovements for the reflected Wing.
reflected_wing_cross_section_movements_list: list[
    ps.movements.aeroelastic_wing_cross_section_movement.AeroelasticWingCrossSectionMovement
] = []

# Define the movement for the main Wing's and its reflected counterpart's
# WingCrossSections. Each WingCrossSection has its own
# AeroelasticWingCrossSectionMovement, which allows the solver to apply deformation
# angles at each time step based on the aerodynamic loads. The reflected Wing has its
# own WingCrossSections, so its movements must be built around those rather than reusing
# the main Wing's movements, even though both halves move identically.
for wing_cross_section_movements_list, wing in (
    (main_wing_cross_section_movements_list, example_airplane.wings[0]),
    (reflected_wing_cross_section_movements_list, example_airplane.wings[1]),
):
    for i in range(len(wing.wing_cross_sections)):
        if i == 0:
            wing_cross_section_movement = ps.movements.aeroelastic_wing_cross_section_movement.AeroelasticWingCrossSectionMovement(
                base_wing_cross_section=wing.wing_cross_sections[i],
            )
            wing_cross_section_movements_list.append(wing_cross_section_movement)

        else:
            wing_cross_section_movement = ps.movements.aeroelastic_wing_cross_section_movement.AeroelasticWingCrossSectionMovement(
                base_wing_cross_section=wing.wing_cross_sections[i],
                ampLp_Wcsp_Lpp=(0.0, 0.0, 0.0),
                periodLp_Wcsp_Lpp=(0.0, 0.0, 0.0),
                spacingLp_Wcsp_Lpp=("sine", "sine", "sine"),
                phaseLp_Wcsp_Lpp=(0.0, 0.0, 0.0),
                ampAngles_Wcsp_to_Wcs_ixyz=(AMPLITUDE_X, AMPLITUDE_Y, AMPLITUDE_Z),
                periodAngles_Wcsp_to_Wcs_ixyz=(PERIOD_X, PERIOD_Y, PERIOD_Z),
                spacingAngles_Wcsp_to_Wcs_ixyz=("sine", "sine", "sine"),
                phaseAngles_Wcsp_to_Wcs_ixyz=(DEPHASE_X, DEPHASE_Y, DEPHASE_Z),
            )
            wing_cross_section_movements_list.append(wing_cross_section_movement)


# Now define the V-tail's root and tip WingCrossSections' WingCrossSectionMovements. The
# V-tail is not an aeroelastic surface, so we use standard WingCrossSectionMovements and
# a standard WingMovement. This keeps the V-tail mesh consistent across all time steps
# without applying any structural deformation.
v_tail_root_wing_cross_section_movement = (
    ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
        base_wing_cross_section=example_airplane.wings[2].wing_cross_sections[0],
        ampLp_Wcsp_Lpp=(0.0, 0.0, 0.0),
        periodLp_Wcsp_Lpp=(0.0, 0.0, 0.0),
        spacingLp_Wcsp_Lpp=("sine", "sine", "sine"),
        phaseLp_Wcsp_Lpp=(0.0, 0.0, 0.0),
        ampAngles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
        periodAngles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
        spacingAngles_Wcsp_to_Wcs_ixyz=("sine", "sine", "sine"),
        phaseAngles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
    )
)
v_tail_tip_wing_cross_section_movement = (
    ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
        base_wing_cross_section=example_airplane.wings[2].wing_cross_sections[1],
        ampLp_Wcsp_Lpp=(0.0, 0.0, 0.0),
        periodLp_Wcsp_Lpp=(0.0, 0.0, 0.0),
        spacingLp_Wcsp_Lpp=("sine", "sine", "sine"),
        phaseLp_Wcsp_Lpp=(0.0, 0.0, 0.0),
        ampAngles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
        periodAngles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
        spacingAngles_Wcsp_to_Wcs_ixyz=("sine", "sine", "sine"),
        phaseAngles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
    )
)
# Reflected V-tail WingCrossSectionMovements reuse the same static motion as the
# original V-tail. Both halves are symmetric and neither deforms.
# This dephase parameter is used to make the Wing start in a flat position.
DEPHASE = 169.0

# Now define the main Wing's AeroelasticWingMovement, the reflected main Wing's
# AeroelasticWingMovement, and the V-tail's AeroelasticWingMovement.
main_wing_movement = ps.movements.aeroelastic_wing_movement.AeroelasticWingMovement(
    base_wing=example_airplane.wings[0],
    wing_cross_section_movements=main_wing_cross_section_movements_list,
    ampLer_Gs_Cgs=(0.0, 0.0, 0.0),
    periodLer_Gs_Cgs=(0.0, 0.0, 0.0),
    spacingLer_Gs_Cgs=("sine", "sine", "sine"),
    phaseLer_Gs_Cgs=(0.0, 0.0, 0.0),
    ampAngles_Gs_to_Wn_ixyz=(FLAP_AMPLITUDE, 0.0, 0.0),
    periodAngles_Gs_to_Wn_ixyz=(FLAP_PERIOD, 0.0, 0.0),
    spacingAngles_Gs_to_Wn_ixyz=("sine", "sine", "sine"),
    phaseAngles_Gs_to_Wn_ixyz=(DEPHASE, 0.0, 0.0),
)

reflected_main_wing_movement = (
    ps.movements.aeroelastic_wing_movement.AeroelasticWingMovement(
        base_wing=example_airplane.wings[1],
        wing_cross_section_movements=reflected_wing_cross_section_movements_list,
        ampLer_Gs_Cgs=(0.0, 0.0, 0.0),
        periodLer_Gs_Cgs=(0.0, 0.0, 0.0),
        spacingLer_Gs_Cgs=("sine", "sine", "sine"),
        phaseLer_Gs_Cgs=(0.0, 0.0, 0.0),
        ampAngles_Gs_to_Wn_ixyz=(FLAP_AMPLITUDE, 0.0, 0.0),
        periodAngles_Gs_to_Wn_ixyz=(FLAP_PERIOD, 0.0, 0.0),
        spacingAngles_Gs_to_Wn_ixyz=("sine", "sine", "sine"),
        phaseAngles_Gs_to_Wn_ixyz=(DEPHASE, 0.0, 0.0),
    )
)

v_tail_wing_movement = ps.movements.wing_movement.WingMovement(
    base_wing=example_airplane.wings[2],
    wing_cross_section_movements=[
        v_tail_root_wing_cross_section_movement,
        v_tail_tip_wing_cross_section_movement,
    ],
    ampLer_Gs_Cgs=(0.0, 0.0, 0.0),
    periodLer_Gs_Cgs=(0.0, 0.0, 0.0),
    spacingLer_Gs_Cgs=("sine", "sine", "sine"),
    phaseLer_Gs_Cgs=(0.0, 0.0, 0.0),
    ampAngles_Gs_to_Wn_ixyz=(0.0, 0.0, 0.0),
    periodAngles_Gs_to_Wn_ixyz=(0.0, 0.0, 0.0),
    spacingAngles_Gs_to_Wn_ixyz=("sine", "sine", "sine"),
    phaseAngles_Gs_to_Wn_ixyz=(0.0, 0.0, 0.0),
)

# Delete the extraneous pointers to the WingCrossSectionMovements, as these are now
# contained within the WingMovements. This is optional, but it can make debugging
# easier.
del v_tail_root_wing_cross_section_movement
del v_tail_tip_wing_cross_section_movement

# Now define the example Airplane's AeroelasticAirplaneMovement.
example_airplane_movement = (
    ps.movements.aeroelastic_airplane_movement.AeroelasticAirplaneMovement(
        base_airplane=example_airplane,
        wing_movements=[
            main_wing_movement,
            reflected_main_wing_movement,
            v_tail_wing_movement,
        ],
        ampCg_GP1_CgP1=(0.0, 0.0, 0.0),
        periodCg_GP1_CgP1=(0.0, 0.0, 0.0),
        spacingCg_GP1_CgP1=("sine", "sine", "sine"),
        phaseCg_GP1_CgP1=(0.0, 0.0, 0.0),
    )
)

del main_wing_movement
del reflected_main_wing_movement
del v_tail_wing_movement

# Define a new OperatingPoint.
example_operating_point = ps.operating_point.OperatingPoint(
    rho=1.225, 
    vCg__E=10.0, 
    alpha=3.3, 
    beta=0.0, 
    angles_E_to_BP1_izyx=(0.0, 3.3, 0.0),
    externalFX_W=0.0, nu=15.06e-6,
    g_E=(0.0, 0.0, 9.80665),
)


# Define the OperatingPoint's FreeFlightOperatingPointMovement. It holds only the
# initial OperatingPoint. The solver populates its operating_points list with the body
# state from the dynamics integration at each time step.
operating_point_movement = (
    ps.movements.free_flight_operating_point_movement.FreeFlightOperatingPointMovement(
        base_operating_point=example_operating_point,
    )
)

# Delete the extraneous pointer.
del example_operating_point



# Define the FreeFlightMovement. This contains the AirplaneMovement and the
# FreeFlightOperatingPointMovement. The Airplane first holds its initial flight
# condition for prescribed_num_steps time steps so the wake can develop, then the solver
# releases the rigid body dynamics for the remaining free_num_steps time steps.
#TODO: NEED AEROELASTIC FREE FLIGHT MOVEMENT
movement = ps.movements.free_flight_movement.FreeFlightAeroelasticMovement(
    airplane_movements=[example_airplane_movement],
    operating_point_movement=operating_point_movement,
    delta_time=dt_des,
    prescribed_num_steps=PRESCRIBED_RIGID_STEPS,
    free_num_steps=int(FLAP_PERIOD*FLAP_CYCLES/dt_des) - PRESCRIBED_RIGID_STEPS,
)


# Define the FreeFlightAeroelasticUnsteadyProblem. The inertia matrix holds representative mass
# properties reused from the glider example. This airframe is a coupling demonstration
# and is not separately tuned or verified for static stability. It is expressed in the
# first Airplane's body axes relative to the first Airplane's center of gravity, which
# is at the geometry origin. The off-diagonal terms are the body-axes products of
# inertia. The mass equals the Airplane's weight (420.0 N) divided by the gravitational
# acceleration magnitude (9.80665 m/s^2), which keeps the weight, mass, and gravity
# consistent. The solver applies the gravitational force as mass * g_E. No external
# loads are applied (external_loads_fn = None), so the Airplane is driven only by its
# flapping aerodynamics, gravity, and inertia.
#TODO: NEED AEROELASTIC FREE FLIGHT UNSTEADY PROBLEM
example_problem = ps.problems.FreeFlightAeroelasticUnsteadyProblem(
    movement=movement,
    wing_density=WING_DENSITY,
    spring_constant_rad=SPRING_CONSTANT_RAD,
    damping_constant_rad=DAMPING_CONSTANT_RAD,
    mass=ROBOT_WEIGHT / 9.80665,
    I_BP1_CgP1=(
        (155.614, 0.0, -45.658),
        (0.0, 398.513, 0.0),
        (-45.658, 0.0, 508.699),
    ),
    external_loads_fn=None,
)

# Define a new solver. We'll create a FreeFlightUnsteadyRingVortexLatticeMethodSolver,
# which requires a FreeFlightUnsteadyProblem.
#TODO: NEED AEROELASTIC FREE FLIGHT ULVM SOLVER
example_solver = ps.free_flight_aeroelastic_unsteady_ring_vortex_lattice_method.FreeFlightAeroelasticUnsteadyRingVortexLatticeMethodSolver(
    free_flight_unsteady_problem=example_problem,
)

# Run the solver.
example_solver.run(
    prescribed_wake=False,
    show_progress=True,
)

# Save the solved solver to a .psz file. This allows us to load the results later
# without re-running the simulation.
ps.save("example_solver.psz", example_solver)

# Load the saved solver. The loaded object is identical to the original and can be
# passed to any output function.
loaded_solver = ps.load("example_solver.psz")


# The load function is annotated as returning object because a saved file can hold any
# Ptera Software object. This assert narrows the type for type checkers and guards
# against loading the wrong file.
assert isinstance(
    loaded_solver,
    ps.free_flight_aeroelastic_unsteady_ring_vortex_lattice_method.FreeFlightAeroelasticUnsteadyRingVortexLatticeMethodSolver,
)

# Log the loaded solver's loads. For a free flight solver, this also logs the first
# Airplane's initial and final six-degree-of-freedom state: its position, velocity,
# orientation, angular velocity, and aerodynamic angles.
ps.output.log_results(solver=loaded_solver)

# Call the draw function on the loaded solver. For a free flight solver, the geometry is
# drawn in Earth axes at the final time step's true flight pose, so the Airplane appears
# where and how it ended up after the final time step. Press any key to close the
# plotter after it draws the output.
ps.output.draw(
    solver=loaded_solver,
    scalar_type="lift",
    show_streamlines=True,
    show_wake_vortices=False,
    save=True,
)

# Call the animate function on the loaded solver. For a free flight solver, this flies
# the Airplane through the scene along its computed trajectory, producing an animated
# WebP saved in the same directory as this script. Press any key, after orienting the
# view, to begin the animation.
ps.output.animate(
    unsteady_solver=loaded_solver,
    scalar_type="lift",
    show_wake_vortices=True,
    save=True,
)

# Call the plotting function on the solver. This produces graphs of the loads with
# respect to time. For a free flight solver, it also plots the first Airplane's
# six-degree-of-freedom state history.
ps.output.plot_results_versus_time(
    unsteady_solver=loaded_solver,
    show=True,
    save=True,
)
