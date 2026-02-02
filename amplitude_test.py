import pterasoftware as ps
import numpy as np
import csv


optitrack_file = r"C:\Users\henri\Documents\MIT\PteraSoftware\optitrack_data\five_hz.csv"

list_trackers = [
    "A1", "A2", "A3", "B1", "B2", "B3", "B4", "B5",
    "C1", "C2", "C3", "C4", "C5", "D1", "D2", "D3", "D4", "D5",
    "E1", "E2", "E3", "E4", "E5", "F1", "F2", "F3", "F4", "F5", 
    "G1", "G2", "G3", "G4", "G5", "H1", "H2", "H3", "H4",
    "I1", "I2"
]

frequency = 5  

columns = ps.geometry.airfoil_creation.extract_columns(list_trackers)

data = ps.geometry.airfoil_creation.load_data(optitrack_file, list_trackers, right = False) *10**-3 


example_airplane = ps.geometry.airplane.Airplane(
    wings=[
        ps.geometry.wing.Wing(
            wing_cross_sections=ps.geometry.airfoil_creation.creation_wing_cross_sections(data, list_trackers, columns, frequency),
            name="Main Wing",
            Ler_Gs_Cgs= [0.0, 0.025, 0.0],
            angles_Gs_to_Wn_ixyz= [4, 0.0, 0.0], 
            symmetric=True,
            mirror_only=False,
            symmetryNormal_G=(0.0, 1.0, 0.0),
            symmetryPoint_G_Cg=(0.0, 0.0, 0.0),
            num_chordwise_panels=6,
            chordwise_spacing="uniform",
        ),
    ],
    name="Example Airplane",
    Cg_GP1_CgP1=(0.0, 0.025, 0.0), 
    weight=5,
    s_ref=None,
    c_ref=None,
    b_ref=None,
)

simulated_airplane = ps.optitrack_validation.WingKinematicsComparison(example_airplane).simulated_airplane

example_operating_point = ps.operating_point.OperatingPoint(
    rho=1.225, vCg__E=6.0, alpha=14.0, beta=0.0, externalFX_W=0.0, nu=15.06e-6
)

operating_point_movement = ps.movements.operating_point_movement.OperatingPointMovement(
    base_operating_point=example_operating_point)

wing_movements = []
for wing in simulated_airplane.wings:
    wing_cross_section_movements = []
    for wing_cross_section in wing.wing_cross_sections:
        wing_cross_section_movements.append(ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
            base_wing_cross_section=wing_cross_section)
        )
    wing_movements.append(wing_cross_section_movements)

output_csv = r"C:\Users\henri\Documents\MIT\PteraSoftware\results.csv"

with open(output_csv, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["freq", "amplitude", "lift", "drag", "roll", "power"])

for freq in range (3,11):
    for Amplitude in [20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0]:

        main_wing_movement = ps.movements.wing_movement.WingMovement(
            base_wing=simulated_airplane.wings[0],
            wing_cross_section_movements=wing_movements[0],
            ampLer_Gs_Cgs=(0.0, 0.0, 0.0),
            periodLer_Gs_Cgs=(0.0, 0.0, 0.0),
            spacingLer_Gs_Cgs=("sine", "sine", "sine"),
            phaseLer_Gs_Cgs=(0.0, 0.0, 0.0),
            ampAngles_Gs_to_Wn_ixyz=(Amplitude, 0.0, 0.0), 
            periodAngles_Gs_to_Wn_ixyz=(1/freq, 0.0, 0.0), 
            spacingAngles_Gs_to_Wn_ixyz=("sine", "sine", "sine"),
            phaseAngles_Gs_to_Wn_ixyz=(0.0, 0.0, 0.0),
        )
        
        reflected_main_wing_movement = ps.movements.wing_movement.WingMovement(
            base_wing=simulated_airplane.wings[1],
            wing_cross_section_movements=wing_movements[1],
            ampLer_Gs_Cgs=(0.0, 0.0, 0.0),
            periodLer_Gs_Cgs=(0.0, 0.0, 0.0),
            spacingLer_Gs_Cgs=("sine", "sine", "sine"),
            phaseLer_Gs_Cgs=(0.0, 0.0, 0.0),
            ampAngles_Gs_to_Wn_ixyz=(Amplitude, 0.0, 0.0),  
            periodAngles_Gs_to_Wn_ixyz=(1/freq, 0.0, 0.0), 
            spacingAngles_Gs_to_Wn_ixyz=("sine", "sine", "sine"),
            phaseAngles_Gs_to_Wn_ixyz=(0.0, 0.0, 0.0),
        )

        simulated_airplane_movement = ps.movements.airplane_movement.AirplaneMovement(
            base_airplane=simulated_airplane,
            wing_movements=[main_wing_movement, reflected_main_wing_movement],
            ampCg_GP1_CgP1=(0.0, 0.0, 0.0),
            periodCg_GP1_CgP1=(0.0, 0.0, 0.0),
            spacingCg_GP1_CgP1=("sine", "sine", "sine"),
            phaseCg_GP1_CgP1=(0.0, 0.0, 0.0),
        )

        movement = ps.movements.movement.Movement(
            airplane_movements=[simulated_airplane_movement],
            operating_point_movement=operating_point_movement,
            delta_time=None,
            num_cycles=5,
            num_chords=None,
            num_steps=None,
        )

        example_problem = ps.problems.UnsteadyProblem(
            movement=movement, 
        )


        example_solver = (
            ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(
                unsteady_problem=example_problem,
            )
        )

        example_solver.run(
            prescribed_wake=True,
            show_progress=True,
        )

        results = ps.output.amplitude_exp(example_solver, freq, Amplitude)

        lift, drag, roll, power = results

        with open(output_csv, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([freq, Amplitude, lift, drag, roll, power])



