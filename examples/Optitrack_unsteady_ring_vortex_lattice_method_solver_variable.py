"""
This example shows how to use the UnsteadyRingVortexLatticeMethodSolver withOptiTrack data
"""

import pterasoftware as ps

# Define all the necessary parameters for loading the OptiTrack data.


optitrack_file = r"C:\Users\henri\Documents\MIT\Travail\Optitrack\Test_2\trois_hz.csv"

# Don't forget to add the Z1 and Z2 trackers, which are virtual trackers defining the wing root chord line. 
# Their definition is in airfoil_creation.py adapted for our robot, you may need to change the position according to your setup.
list_trackers = [
    "Z1", "Z2", "A1", "A2", "A3", "B1", "B2", "B3", "B4", "B5",
    "C1", "C2", "C3", "C4", "C5", "D1", "D2", "D3", "D4", "D5",
    "E1", "E2", "E3", "E4", "E5", "F1", "F2", "F3", "F4", "F5", 
    "G1", "G2", "G3", "G4", "G5", "H1", "H2", "H3", "H4",
    "I1", "I2"
]

columns = ps.geometry.airfoil_creation.extract_columns(list_trackers)

Nb_columns=len(columns)

# Load the OptiTrack data.
data = ps.geometry.airfoil_creation.load_data(optitrack_file, list_trackers, right = False) *10**-3  # Convert from mm to m

# Create the airfoils at step 0.
airfoils_0 = {}

for column in columns:
    airfoils_0[column] = ps.geometry.airfoil_creation.Real_Airfoil(
        data=data,
        step=0,
        column=column,
        list_trackers=list_trackers,
    )

# Define the airplane geometry using the airfoils created from the OptiTrack data.
example_airplane = ps.geometry.airplane.Airplane(
    wings=[
        ps.geometry.wing.Wing(
            wing_cross_sections=[
                ps.geometry.wing_cross_section.WingCrossSection(
                    num_spanwise_panels = None if column == columns[-1] else 1, # Last wing cross-section has no panels 
                    chord=airfoils_0[column].get_chord_length(), 
                    Lp_Wcsp_Lpp=airfoils_0[column].get_position()[0],
                    angles_Wcsp_to_Wcs_ixyz=airfoils_0[column].get_position()[1],
                    control_surface_symmetry_type="symmetric",
                    control_surface_hinge_point=0.75,
                    control_surface_deflection=0.0,
                    spanwise_spacing=None if column == columns[-1] else "uniform", # Last wing cross-section has no panels
                    airfoil=ps.geometry.airfoil.Airfoil(
                        name=f"column_{column}_airfoil",  
                        outline_A_lp=airfoils_0[column].get_airfoil_shape(),
                        resample=True,
                        n_points_per_side=400,
                        data=data,    # Pass the data to the Airfoil
                        column=column,        # Pass the column to the Airfoil
                        list_trackers=list_trackers,       # Pass the list of trackers to the Airfoil
                    ),
                )
                for column in columns
            ],
            name="Main Wing",
            Ler_Gs_Cgs=(0.0, 0.0005, 0.0),
            angles_Gs_to_Wn_ixyz=(0.0, 0.0, 0.0),
            symmetric=True,
            mirror_only=False,
            symmetryNormal_G=(0.0, 0.0001, 0.0),
            symmetryPoint_G_Cg=(0.0, 0.0, 0.0),
            num_chordwise_panels=6,
            chordwise_spacing="uniform",
        )
    ],
    name="Example Airplane",
    Cg_E_CgP1=(0.0, 0.0, 0.0),
    angles_E_to_B_izyx=(0.0, 0.0, 0.0),
    weight=5,
    s_ref=None,
    c_ref=None,
    b_ref=None,
)


# Define the airplane's AirplaneMovement.
main_wing_cross_section_movement=[None]*Nb_columns

for i in range (Nb_columns):
    main_wing_cross_section_movement[i] = (
        ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
            base_wing_cross_section=example_airplane.wings[0].wing_cross_sections[i],
            ampLp_Wcsp_Lpp=(0.0, 0.0, 0.0),
            periodLp_Wcsp_Lpp=(0.0, 0.0, 0.0),
            spacingLp_Wcsp_Lpp=("sine", "sine", "sine"),
            phaseLp_Wcsp_Lpp=(0.0, 0.0, 0.0),
            ampAngles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
            periodAngles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
            spacingAngles_Wcsp_to_Wcs_ixyz=("sine", "sine", "sine"),
            phaseAngles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
            optitrack = True  # Put to True to use OptiTrack data. The other parameters will be ignored except for base_wing_cross_section
        )
    )

reflected_main_wing_cross_section_movement=[None] *Nb_columns

for i in range (Nb_columns):
    reflected_main_wing_cross_section_movement[i] = (
        ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
            base_wing_cross_section=example_airplane.wings[1].wing_cross_sections[i],
            ampLp_Wcsp_Lpp=(0.0, 0.0, 0.0),
            periodLp_Wcsp_Lpp=(0.0, 0.0, 0.0),
            spacingLp_Wcsp_Lpp=("sine", "sine", "sine"),
            phaseLp_Wcsp_Lpp=(0.0, 0.0, 0.0),
            ampAngles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
            periodAngles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
            spacingAngles_Wcsp_to_Wcs_ixyz=("sine", "sine", "sine"),
            phaseAngles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
            optitrack = True  # Put to True to use OptiTrack data. The other parameters will be ignored except for base_wing_cross_section
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
)

# Now define the example airplane's AirplaneMovement. For now, no movement of the airplane is possible.  
airplane_movement = ps.movements.airplane_movement.AirplaneMovement(
    base_airplane=example_airplane,
    wing_movements=[main_wing_movement, reflected_main_wing_movement],
    ampCg_E_CgP1=(0.0, 0.0, 0.0),
    periodCg_E_CgP1=(0.0, 0.0, 0.0),
    spacingCg_E_CgP1=("sine", "sine", "sine"),
    phaseCg_E_CgP1=(0.0, 0.0, 0.0),
    ampAngles_E_to_B_izyx=(0.0, 0.0, 0.0),
    periodAngles_E_to_B_izyx=(0.0, 0.0, 0.0),
    spacingAngles_E_to_B_izyx=("sine", "sine", "sine"),
    phaseAngles_E_to_B_izyx=(0.0, 0.0, 0.0),
)



# Define a new OperatingPoint.
example_operating_point = ps.operating_point.OperatingPoint(
    rho=1.225, vCg__E=1.0, alpha=1.0, beta=0.0, externalFX_W=0.0, nu=15.06e-6
)

# Define the operating point's OperatingPointMovement.
operating_point_movement = ps.movements.operating_point_movement.OperatingPointMovement(
    base_operating_point=example_operating_point)

# Define the Movement. This contains the AirplaneMovement and the
# OperatingPointMovement.
movement = ps.movements.movement.Movement(
    airplane_movements=[airplane_movement],
    operating_point_movement=operating_point_movement,
    delta_time=1/360,   # Time step between two frames in the OptiTrack data
    num_cycles=None,
    num_chords=None,
    num_steps=100,
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


# Run the solver.
example_solver.run(
    logging_level="Warning",
    prescribed_wake=True,
)


# ps.output.plot_results_versus_time(
#     unsteady_solver=example_solver,
#     show=True,
#     save=False,
# )


# Call the animate function on the solver. This produces a GIF of the wake being
# shed. The GIF is saved in the same directory as this script. Press "q",
# after orienting the view, to begin the animation.

# ps.output.animate(
#     unsteady_solver=example_solver,
#     scalar_type="lift",
#     show_wake_vortices=True,
#     save=True,
# )

# You can creat a simulated airplane with the same geometry by calling differential_measures.Analysis. 
# You will be able to use output for this new simulation and compare the results with the OptiTrack based simulation.

# analysis = ps.differential_measures.Analysis(example_solver,3)



