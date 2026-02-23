"""
This example shows how to use the UnsteadyRingVortexLatticeMethodSolver with OptiTrack data
"""

import pterasoftware as ps
import numpy as np

# Define all the necessary parameters for loading the OptiTrack data.

optitrack_file = r"C:\Users\henri\Documents\MIT\PteraSoftware\optitrack_data\five_hz.csv"

# This list is adapted for our robot, you may need to change the list according to your setup.
list_trackers = [
    "A1", "A2", "A3", "B1", "B2", "B3", "B4", "B5",
    "C1", "C2", "C3", "C4", "C5", "D1", "D2", "D3", "D4", "D5",
    "E1", "E2", "E3", "E4", "E5", "F1", "F2", "F3", "F4", "F5", 
    "G1", "G2", "G3", "G4", "G5", "H1", "H2", "H3", "H4",
    "I1", "I2"
]

frequency = 4  # Frequency of the flapping-cycle in Hz.

columns = ps.geometry.airfoil_creation.extract_columns(list_trackers)

# Load the OptiTrack data.
data = ps.geometry.airfoil_creation.load_data(optitrack_file, list_trackers, right = False) *10**-3  # Convert from mm to m

# Define the airplane geometry using the airfoils created from the OptiTrack data.
example_airplane = ps.geometry.airplane.Airplane(
    wings=[
        ps.geometry.wing.Wing(
            wing_cross_sections=ps.geometry.airfoil_creation.creation_wing_cross_sections(data, list_trackers, columns, frequency),
            name="Main Wing",
            Ler_Gs_Cgs= [0.0, 0.025, 0.0], # position of the wing's rotation axes
            angles_Gs_to_Wn_ixyz= [4, 0.0, 0.0], # angle offset
            symmetric=True,
            mirror_only=False,
            symmetryNormal_G=(0.0, 1.0, 0.0),
            symmetryPoint_G_Cg=(0.0, 0.0, 0.0),
            num_chordwise_panels=6,
            chordwise_spacing="uniform",
        ),
        ps.geometry.wing.Wing(
            wing_cross_sections=[
                ps.geometry.wing_cross_section.WingCrossSection(
                    num_spanwise_panels=8,
                    chord=0.1,
                    Lp_Wcsp_Lpp=(0.0, 0.0, 0.0),
                    angles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
                    control_surface_symmetry_type="symmetric",
                    control_surface_hinge_point=0.75,
                    control_surface_deflection=0.0,
                    spanwise_spacing="uniform",
                    airfoil=ps.geometry.airfoil.Airfoil(
                        name="naca0012",
                        outline_A_lp=None,
                        resample=True,
                        n_points_per_side=400,
                    ),
                ),
                ps.geometry.wing_cross_section.WingCrossSection(
                    num_spanwise_panels=None,
                    chord=0.01,
                    Lp_Wcsp_Lpp=(0.09, 0.1, 0),
                    angles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
                    control_surface_symmetry_type="symmetric",
                    control_surface_hinge_point=0.75,
                    control_surface_deflection=0.0,
                    spanwise_spacing=None,
                    airfoil=ps.geometry.airfoil.Airfoil(
                        name="naca0012",
                        outline_A_lp=None,
                        resample=True,
                        n_points_per_side=400,
                    ),
                ),
            ],
            name="V-Tail",
            Ler_Gs_Cgs=(0.3, 0.0, 0.0),
            angles_Gs_to_Wn_ixyz=(0.0, 0.0, 0.0),
            symmetric=True,
            mirror_only=False,
            symmetryNormal_G=(0.0, 1.0, 0.0),
            symmetryPoint_G_Cg=(0.0, 0.0, 0.0),
            num_chordwise_panels=6,
            chordwise_spacing="uniform",
        ),
    ],
    name="Example Airplane",
    Cg_GP1_CgP1=(0.0, 0.0, 0.0), # center of gravity of your airplane (the moments will be calculated at this point)
    weight=5,
    s_ref=None,
    c_ref=None,
    b_ref=None,
)

# Define the airplane's WingCrossSectionMovements using the OptiTrack data.
main_wing_cross_section_movement=ps.geometry.airfoil_creation.wing_cross_sections_movement(example_airplane.wings[0], columns)

reflected_main_wing_cross_section_movement=ps.geometry.airfoil_creation.wing_cross_sections_movement(example_airplane.wings[1], columns)


# Now define the v tail's root and tip WingCrossSections' WingCrossSectionMovements.
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

# Define the airplane's WingMovements. It's necessary but the simulation won't use it.
main_wing_movement = ps.movements.wing_movement.WingMovement(
    base_wing=example_airplane.wings[0],
    wing_cross_section_movements= main_wing_cross_section_movement,
    ampLer_Gs_Cgs=(0.0, 0.0, 0.0),
    periodLer_Gs_Cgs=(0.0, 0.0, 0.0),
    spacingLer_Gs_Cgs=("sine", "sine", "sine"),
    phaseLer_Gs_Cgs=(0.0, 0.0, 0.0),
    ampAngles_Gs_to_Wn_ixyz=(0.0, 0.0, 0.0), 
    periodAngles_Gs_to_Wn_ixyz=(0.0, 0.0, 0.0), 
    spacingAngles_Gs_to_Wn_ixyz=("sine", "sine", "sine"),
    phaseAngles_Gs_to_Wn_ixyz=(0.0, 0.0, 0.0),
    optitrack = True  # Put to True to use OptiTrack data. The other parameters will be ignored except for base_wing
)
reflected_main_wing_movement = ps.movements.wing_movement.WingMovement(
    base_wing=example_airplane.wings[1],
    wing_cross_section_movements=reflected_main_wing_cross_section_movement,
    ampLer_Gs_Cgs=(0.0, 0.0, 0.0),
    periodLer_Gs_Cgs=(0.0, 0.0, 0.0),
    spacingLer_Gs_Cgs=("sine", "sine", "sine"),
    phaseLer_Gs_Cgs=(0.0, 0.0, 0.0),
    ampAngles_Gs_to_Wn_ixyz=(0.0, 0.0, 0.0),  
    periodAngles_Gs_to_Wn_ixyz=(0.0, 0.0, 0.0), 
    spacingAngles_Gs_to_Wn_ixyz=("sine", "sine", "sine"),
    phaseAngles_Gs_to_Wn_ixyz=(0.0, 0.0, 0.0),
    optitrack = True  # Put to True to use OptiTrack data. The other parameters will be ignored except for base_wing
)

v_tail_movement = ps.movements.wing_movement.WingMovement(
    base_wing=example_airplane.wings[2],
    wing_cross_section_movements=[v_tail_root_wing_cross_section_movement, v_tail_tip_wing_cross_section_movement],
    ampLer_Gs_Cgs=(0.0, 0.0, 0.0),
    periodLer_Gs_Cgs=(0.0, 0.0, 0.0),
    spacingLer_Gs_Cgs=("sine", "sine", "sine"),
    phaseLer_Gs_Cgs=(0.0, 0.0, 0.0),
    ampAngles_Gs_to_Wn_ixyz=(0.0, 0.0, 0.0),
    periodAngles_Gs_to_Wn_ixyz=(0.0, 0.0, 0.0),
    spacingAngles_Gs_to_Wn_ixyz=("sine", "sine", "sine"),
    phaseAngles_Gs_to_Wn_ixyz=(0.0, 0.0, 0.0),
)

# Now define the example airplane's AirplaneMovement. For now, no movement of the airplane is possible.  
example_airplane_movement = ps.movements.airplane_movement.AirplaneMovement(
    base_airplane=example_airplane,
    wing_movements=[main_wing_movement, reflected_main_wing_movement, v_tail_movement],
    ampCg_GP1_CgP1=(0.0, 0.0, 0.0),
    periodCg_GP1_CgP1=(0.0, 0.0, 0.0),
    spacingCg_GP1_CgP1=("sine", "sine", "sine"),
    phaseCg_GP1_CgP1=(0.0, 0.0, 0.0),
)

# Define a new OperatingPoint.
example_operating_point = ps.operating_point.OperatingPoint(
    rho=1.225, vCg__E=5.0, alpha=12.0, beta=0.0, externalFX_W=0.0, nu=15.06e-6
)

# Define the operating point's OperatingPointMovement.
operating_point_movement = ps.movements.operating_point_movement.OperatingPointMovement(
    base_operating_point=example_operating_point)

# Define the Movement. This contains the AirplaneMovement and the
# OperatingPointMovement.
movement = ps.movements.movement.Movement(
    airplane_movements=[example_airplane_movement],
    operating_point_movement=operating_point_movement,
    delta_time=1/360,   # Time step between two frames in the OptiTrack data
    num_cycles=None,
    num_chords=None,
    num_steps=200,
)

# Define the UnsteadyProblem.
example_problem = ps.problems.UnsteadyProblem(
    movement=movement, 
)

# Define a new solver. The available solver classes are 
# SteadyHorseshoeVortexLatticeMethodSolver, SteadyRingVortexLatticeMethodSolver,
# and UnsteadyRingVortexLatticeMethodSolver. We'll create an
# UnsteadyRingVortexLatticeMethodSolver, which requires a UnsteadyProblem.
example_solver = (
    ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(
        unsteady_problem=example_problem,
    )
)

# WingKinematicsComparison = ps.optitrack_validation.WingKinematicsComparison(example_solver)
# simulated_solver = WingKinematicsComparison.simulated_solver
# simulated_solver.run(
#     prescribed_wake=True,
#     show_progress=True,
#     wing_density=0.05,
#     damping_constant=0.1,
#     spring_constant=100
# )
# WingKinematicsComparison.dynamic_wing()
# Run the solver.
example_solver.run(
    prescribed_wake=True,
    show_progress=True,
)

# simulated_solver.run(
#     prescribed_wake=True,
#     show_progress=True,
# )
# Call the animate function on the solver. This produces a GIF of the wake being
# shed. The GIF is saved in the same directory as this script. Press "q",
# after orienting the view, to begin the animation.

# ps.output.plot_results_versus_time(example_solver)
# ps.output.plot_wing_loads_versus_time(example_solver)


# ps.output.print_results(example_solver)
# print(movement.static)

# ps.output.animate(
#     unsteady_solver=example_solver,
#     scalar_type="lift",     
#     show_wake_vortices=True,
#     save=True,
#  )

ps.output.plot_theta(example_solver, wing_cross_section_index=5)
