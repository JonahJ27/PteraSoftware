"""
Tools for extracting real wing kinematics from OptiTrack data and
comparing them with a URVLM-based simulated reconstruction (PteraSoftware).

Main class
----------
WingKinematicsComparison
    Extracts real wing motion, reconstructs an equivalent simulated motion,
    and compares real and simulated trajectories and aerodynamic forces.

Public methods
--------------
__init__
    Initialize the analysis pipeline and build the simulated model.

extract_movement_data
    Extract flapping amplitudes and phases from tracker data.

build_simulated_airplane
    Build the simulated airplane geometry.

build_simulated_movement
    Reconstruct a periodic wing motion from extracted parameters.

build_simulated_problem
    Build the unsteady aerodynamic problem.

build_simulated_solver
    Build and run the unsteady solver.

get_coordinates
    Return the 3D coordinates of a wing point at a given time step.

get_full_trajectory
    Return the full trajectory of a wing point.

compare_trajectories
    Compute real–simulated trajectory differences.

compare_forces
    Compute real–simulated aerodynamic force differences.

plot_trajectory_3d
    Plot real and simulated 3D trajectories.

plot_forces
    Plot forces, force coefficients, and moments.
"""


from matplotlib.pylab import real
import numpy as np
import matplotlib.pyplot as plt
import pterasoftware as ps
from scipy.spatial import Delaunay
from matplotlib.animation import FuncAnimation, PillowWriter

class WingKinematicsComparison:
   
    def __init__(self, object, delta_time = None):
        """Initializes the class and builds simulated equivalents.

        :param object: Either a UnsteadyRingVortexLatticeMethodSolver object or an Airplane_movement object.    
        :return: None.
        """
        if isinstance(object, ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver):
            self.solver = object
            self.problem = self.solver.unsteady_problem
            self.movement = self.problem.movement
            if len(self.movement.airplane_movements) != 1:
                raise ValueError("Analysis currently only supports a single AirplaneMovement.")
            self.airplane_movements = self.movement.airplane_movements[0]
            self.num_steps = self.movement.num_steps
            self.delta_time = self.movement.delta_time
            self.base_airplane = self.airplane_movements.base_airplane
            airfoil = self.base_airplane.wings[0].wing_cross_sections[0].airfoil

            self.data = airfoil.data
            self.list_trackers = airfoil.list_trackers

        
            self.frequency = airfoil.frequency
            self.period=1/self.frequency
          
            self.extract_movement_data()

            self.simulated_airplane = self.build_simulated_airplane()
            self.simulated_airplane_movement = self.build_simulated_airplane_movement()
            self.simulated_problem = self.build_simulated_problem()
            self.simulated_solver = self.build_simulated_solver()

        elif isinstance(object, ps.geometry.airplane.Airplane):
            self.base_airplane = object
            airfoil = self.base_airplane.wings[0].wing_cross_sections[0].airfoil

            self.data = airfoil.data
            self.list_trackers = airfoil.list_trackers

            self.frequency = airfoil.frequency
            self.period=1/self.frequency
      

            self.extract_movement_data()

            self.simulated_airplane = self.build_simulated_airplane()

        else :
            raise ValueError("Object must be either a UnsteadyRingVortexLatticeMethodSolver object, an Airplane_movement object, or an Airplane object.")


    def build_simulated_airplane(self):
        """Builds a simulated Airplane object that reproduces the geometry and structure of
        the real airplane while allowing controlled simulated motion.

        :param None:

        :return: A simulated Airplane object with identical geometry to the real model.
        """

        real_wing = self.base_airplane.wings[0]
        

        simulated_wing = ps.geometry.wing.Wing(
            wing_cross_sections=[
                ps.geometry.wing_cross_section.WingCrossSection(
                    num_spanwise_panels=cs.num_spanwise_panels,
                    chord=cs.chord,
                    Lp_Wcsp_Lpp=cs.Lp_Wcsp_Lpp,
                    angles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
                    control_surface_symmetry_type="symmetric",
                    control_surface_hinge_point=0.75,
                    control_surface_deflection=0.0,
                    airfoil=cs.airfoil,
                )
                for cs in real_wing.wing_cross_sections
            ],
            name="simulatedWing",
            Ler_Gs_Cgs=real_wing.Ler_Gs_Cgs,
            angles_Gs_to_Wn_ixyz=real_wing.angles_Gs_to_Wn_ixyz,
            symmetric=True if len(self.base_airplane.wings) > 1 else False,    # We can't use real_wing.symmetric because after the creation of the reflected wing it becomes False (geometry.airplane, line 763)
            mirror_only=False,     # We can't use real_wing.mirror_only because after the creation of the reflected wing it becomes False (geometry.airplane, line 764)
            symmetryNormal_G=(0.0, 1.0, 0.0) if len(self.base_airplane.wings) > 1 else None,    # We can't use real_wing.symmetryNormal_G because after the creation of the reflected wing it becomes None (geometry.airplane, line 765)
            symmetryPoint_G_Cg=(0.0, 0.0, 0.0) if len(self.base_airplane.wings) > 1 else None,      # We can't use real_wing.symmetryPoint_G_Cg because after the creation of the reflected wing it becomes None (geometry.airplane, line 766)
            num_chordwise_panels=real_wing.num_chordwise_panels,
            chordwise_spacing=real_wing.chordwise_spacing,
        )

        return ps.geometry.airplane.Airplane(
            wings=[simulated_wing, self.base_airplane.wings[2]] if len(self.base_airplane.wings) > 2 else [simulated_wing],
            name="simulatedAirplane",
            Cg_GP1_CgP1=(0.0, 0.0, 0.0),
            weight=self.base_airplane.weight,
            s_ref=self.base_airplane.s_ref,
            c_ref=self.base_airplane.c_ref,
            b_ref=self.base_airplane.b_ref,
        )

    def extract_movement_data(self):
        """Extracts motion parameters (max amplitude, min amplitude, phase shift, status
        phase) from the tracked airfoil markers and computes their mean values over all
        trackers.

        :param None:

        :return: None.
        """
        trackers_1 = [t for t in self.list_trackers if t.endswith("1")]

        amplitudes_max = []
        amplitudes_min = []
        phase_shift = []
        status_phases = []

        for tracker in trackers_1[3:]:
            col = self.list_trackers.index(tracker)
            tracker_data = self.data[:, col]  # shape (num_steps, 3)

            z_max = np.argmax(tracker_data[:, 2])
            z_min = np.argmin(tracker_data[:, 2])

            p_max = tracker_data[z_max]
            p_min = tracker_data[z_min]
            p0 = tracker_data[0]
            p1 = tracker_data[1]


            amplitudes_max.append(np.arctan2(p_max[2], p_max[1] - 25*10**-3))  # 25 is half the width of the robot body (in mm), change it for each robot
            amplitudes_min.append(np.arctan2(p_min[2], p_min[1] - 25*10**-3))
            phase_shift.append(np.arctan2(p0[2], p0[1]))
            status_phases.append(np.arctan2(p1[2], p1[1]))


        self.amplitude_max = float(np.mean(amplitudes_max)) 
        self.amplitude_min = float(np.mean(amplitudes_min)) 
        self.phase_shift = float(np.mean(phase_shift))
        self.status_phase = float(np.mean(status_phases))

    def build_simulated_airplane_movement(self):
        """Reconstructs a synthetic, periodic movement model from the extracted motion
        parameters, generating a simulated wing oscillation compatible with PteraSoftware.

        :param None:

        :return: An AirplaneMovement object representing the simulated periodic motion.
        """
        simulated_airplane = self.simulated_airplane

        A = abs((self.amplitude_max - self.amplitude_min)/2)

        theta0 = self.phase_shift
        theta1 = self.status_phase

        s = np.clip(theta0/ A, -1.0, 1.0)

        dt = self.delta_time
        omega = 2*np.pi*self.frequency

        dtheta = (theta1 - theta0) / dt
        c = np.clip(dtheta / (A * omega), -1.0, 1.0)

        phi = np.arctan2(s, c)

        phi = np.degrees(phi)
        A = np.degrees(A)

        main_wing_cross_section_movement = []
        main_wing_cross_section_movement_single_step = []
        reflected_wing_cross_section_movement = []
        reflected_wing_cross_section_movement_single_step = []

        for i, cs in enumerate(simulated_airplane.wings[0].wing_cross_sections):

            movement = ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
                base_wing_cross_section=cs,
                ampLp_Wcsp_Lpp=(0.0, 0.0, 0.0),
                periodLp_Wcsp_Lpp=(0.0, 0.0, 0.0),
                spacingLp_Wcsp_Lpp=("sine", "sine", "sine"),
                phaseLp_Wcsp_Lpp=(0.0, 0.0, 0.0),
                ampAngles_Wcsp_to_Wcs_ixyz=(0, 0.0, 0.0),
                periodAngles_Wcsp_to_Wcs_ixyz=(0, 0.0, 0.0),
                spacingAngles_Wcsp_to_Wcs_ixyz=("sine", "sine", "sine"),
                phaseAngles_Wcsp_to_Wcs_ixyz=(0, 0.0, 0.0),
            )

            single_step_movement = ps.movements.single_step.single_step_wing_cross_section_movement.SingleStepWingCrossSectionMovement()

            main_wing_cross_section_movement.append(movement)
            main_wing_cross_section_movement_single_step.append(single_step_movement)
            reflected_wing_cross_section_movement.append(movement)
            reflected_wing_cross_section_movement_single_step.append(single_step_movement)

        if len(self.base_airplane.wings) > 2:
            v_tail_root_wing_cross_section_movement = (
                ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
                    base_wing_cross_section=simulated_airplane.wings[2].wing_cross_sections[0], 
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
                    base_wing_cross_section=simulated_airplane.wings[2].wing_cross_sections[1],
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

        maine_wing_movement = ps.movements.wing_movement.WingMovement(
                base_wing=simulated_airplane.wings[0],
                wing_cross_section_movements=main_wing_cross_section_movement,
                ampLer_Gs_Cgs=(0.0, 0.0, 0.0),
                periodLer_Gs_Cgs=(0.0, 0.0, 0.0),
                spacingLer_Gs_Cgs=("sine", "sine", "sine"),
                phaseLer_Gs_Cgs=(0.0, 0.0, 0.0),
                ampAngles_Gs_to_Wn_ixyz=(A, 0.0, 0.0),
                periodAngles_Gs_to_Wn_ixyz=(self.period, 0.0, 0.0),
                spacingAngles_Gs_to_Wn_ixyz=("sine", "sine", "sine"),
                phaseAngles_Gs_to_Wn_ixyz=(phi, 0.0, 0.0),
            )
        
        single_step_main_wing_movement = (
            ps.movements.single_step.single_step_wing_movement.SingleStepWingMovement(
                single_step_wing_cross_section_movements=main_wing_cross_section_movement_single_step,
                ampLer_Gs_Cgs=(0.0, 0.0, 0.0),
                periodLer_Gs_Cgs=(0.0, 0.0, 0.0),
                spacingLer_Gs_Cgs=("sine", "sine", "sine"),
                phaseLer_Gs_Cgs=(0.0, 0.0, 0.0),
                ampAngles_Gs_to_Wn_ixyz=(A, 0.0, 0.0),
                periodAngles_Gs_to_Wn_ixyz=(self.period, 0.0, 0.0),
                spacingAngles_Gs_to_Wn_ixyz=("sine", "sine", "sine"),
                phaseAngles_Gs_to_Wn_ixyz=(phi, 0.0, 0.0),
            )
        )
    
        reflected_main_wing_movement = ps.movements.wing_movement.WingMovement(
            base_wing=simulated_airplane.wings[1],
            wing_cross_section_movements=reflected_wing_cross_section_movement,
            ampLer_Gs_Cgs=(0.0, 0.0, 0.0),
            periodLer_Gs_Cgs=(0.0, 0.0, 0.0),
            spacingLer_Gs_Cgs=("sine", "sine", "sine"),
            phaseLer_Gs_Cgs=(0.0, 0.0, 0.0),
            ampAngles_Gs_to_Wn_ixyz=(A, 0.0, 0.0),
            periodAngles_Gs_to_Wn_ixyz=(self.period, 0.0, 0.0),
            spacingAngles_Gs_to_Wn_ixyz=("sine", "sine", "sine"),
            phaseAngles_Gs_to_Wn_ixyz=(phi, 0.0, 0.0),
        )

        single_step_reflected_main_wing_movement = (
            ps.movements.single_step.single_step_wing_movement.SingleStepWingMovement(
                single_step_wing_cross_section_movements=reflected_wing_cross_section_movement_single_step,
                ampLer_Gs_Cgs=(0.0, 0.0, 0.0),
                periodLer_Gs_Cgs=(0.0, 0.0, 0.0),
                spacingLer_Gs_Cgs=("sine", "sine", "sine"),
                phaseLer_Gs_Cgs=(0.0, 0.0, 0.0),
                ampAngles_Gs_to_Wn_ixyz=(A, 0.0, 0.0),
                periodAngles_Gs_to_Wn_ixyz=(self.period, 0.0, 0.0),
                spacingAngles_Gs_to_Wn_ixyz=("sine", "sine", "sine"),
                phaseAngles_Gs_to_Wn_ixyz=(phi, 0.0, 0.0),
            )
        )

        if len(self.base_airplane.wings) > 2:
            v_tail_movement = ps.movements.wing_movement.WingMovement(
                base_wing=simulated_airplane.wings[2],
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

        wing_movements = []
        if len(self.base_airplane.wings) > 2:
            wing_movements = [maine_wing_movement, reflected_main_wing_movement, v_tail_movement]
        else :
            wing_movements = [maine_wing_movement, reflected_main_wing_movement]

        airplane_movement = ps.movements.airplane_movement.AirplaneMovement(
            base_airplane=simulated_airplane,
            wing_movements=wing_movements,
            ampCg_GP1_CgP1=(0.0, 0.0, 0.0),
            periodCg_GP1_CgP1=(0.0, 0.0, 0.0),
            spacingCg_GP1_CgP1=("sine", "sine", "sine"),
            phaseCg_GP1_CgP1=(0.0, 0.0, 0.0),
        )

        single_step_airplane_movement = (
            ps.movements.single_step.single_step_airplane_movement.SingleStepAirplaneMovement(
                single_step_wing_movements=[
                    single_step_main_wing_movement,
                    single_step_reflected_main_wing_movement,
                ],
                ampCg_GP1_CgP1=(0.0, 0.0, 0.0),
                periodCg_GP1_CgP1=(0.0, 0.0, 0.0),
                spacingCg_GP1_CgP1=("sine", "sine", "sine"),
                phaseCg_GP1_CgP1=(0.0, 0.0, 0.0),
            )
        )

        return [airplane_movement, single_step_airplane_movement]
    
    def build_simulated_problem(self):
        """Builds a simulated unsteady aerodynamic problem using the same operating-point
        movement as the real case but using the simulated Movement created previously.

        :param None:

        :return: A simulated UnsteadyProblem object.
        """
        # real_op_mov = self.movement.operating_point_movement

        simulated_operating_point_movement = self.movement.operating_point_movement
        single_step_operating_point_movement = (
            ps.movements.single_step.single_step_operating_point_movement.SingleStepOperatingPointMovement(
                ampVCg__E=0.0, periodVCg__E=simulated_operating_point_movement.periodVCg__E, spacingVCg__E=simulated_operating_point_movement.spacingVCg__E
            )
        )
        # ps.movements.operating_point_movement.OperatingPointMovement(
        #     base_operating_point = real_op_mov.base_operating_point,
        #     periodVCg__E        = real_op_mov.periodVCg__E,
        #     spacingVCg__E       = real_op_mov.spacingVCg__E,
        # )

        simulated_movement = ps.movements.movement.Movement(
            airplane_movements       = [self.simulated_airplane_movement[0]],
            operating_point_movement = simulated_operating_point_movement,
            delta_time               = self.delta_time,
            num_steps                = self.num_steps,
        )

        single_step_movement = ps.movements.single_step.single_step_movement.SingleStepMovement(
            single_step_airplane_movements=[self.simulated_airplane_movement[1]],
            single_step_operating_point_movement=single_step_operating_point_movement,
            delta_time=self.delta_time,
            num_steps=self.num_steps,
        )

        return ps.problems.BetterAeroelasticUnsteadyProblem(
            movement=simulated_movement,
            single_step_movement=single_step_movement,
            )   


    def build_simulated_solver(self):
        """Creates and runs a simulated unsteady vortex-lattice solver corresponding to the
        previously defined simulated problem.

        :param None:

        :return: A simulated UnsteadyRingVortexLatticeMethodSolver object.
        """
        solver = ps.coupled_unsteady_ring_vortex_lattice_method.CoupledUnsteadyRingVortexLatticeMethodSolver(
            coupled_unsteady_problem=self.simulated_problem
        )
        return solver

    def _track_point(self, airplane, wing_index, x_norm, y_norm):
        """Determines which aerodynamic panel contains a normalized point (x_norm, y_norm)
        on a wing and returns the panel indices and local interpolation coordinates.

        :param airplane: The Airplane object to track the point on.
        :param wing_index: Index of the wing.
        :param x_norm: Normalized chordwise coordinate (0–1).
        :param y_norm: Normalized spanwise coordinate (0–1).

        :return: (i, j, u, v) giving panel indices and local coordinates.
        """
        wing = airplane.wings[wing_index]

        Nx = wing.num_chordwise_panels
        Ny = wing.num_spanwise_panels

        i = int(x_norm * Nx)
        j = int(y_norm * Ny)

        i = min(i, Nx - 1)
        j = min(j, Ny - 1)

        u = x_norm * Nx - i
        v = y_norm * Ny - j

        return i, j, u, v
    
    # spacial coordinates -------------------------------------------------------------------------------------------------------------------------------------------------

    def get_coordinates(self, step, x_norm, y_norm, real=True):
        """Retrieves the 3D position of a normalized point (x_norm, y_norm) on the wing at a
        given time step using bilinear interpolation.

        :param step: Time-step index.
        :param x_norm: Normalized chordwise position.
        :param y_norm: Normalized spanwise position.
        :param real: If True, use real data; otherwise use simulated data.

        :return: A (3,) array with the interpolated 3D coordinate.
        """
        
        if real:
            airplane = self.solver.steady_problems[step].airplanes[0]
        else:
            airplane = self.simulated_solver.steady_problems[step].airplanes[0]
    
        i, j, u, v = self._track_point(airplane, 0, x_norm, y_norm)
        panel = airplane.wings[0].panels[i, j]

        p00 = np.asarray(panel.Flpp_G_Cg).reshape(-1)
        p10 = np.asarray(panel.Frpp_G_Cg).reshape(-1)
        p01 = np.asarray(panel.Blpp_G_Cg).reshape(-1)
        p11 = np.asarray(panel.Brpp_G_Cg).reshape(-1)

        p00 = p00[:3]
        p10 = p10[:3]
        p01 = p01[:3]
        p11 = p11[:3]

        P = (
            (1 - u) * (1 - v) * p00 +
            u       * (1 - v) * p10 +
            (1 - u) * v       * p01 +
            u       * v       * p11
        )

        return np.asarray(P).reshape(3,)

    def get_full_trajectory(self, x_norm, y_norm, real=True):
        """Returns the full time history of the 3D coordinates of a point on the wing,
        evaluated at every time step.

        :param x_norm: Normalized chordwise position.
        :param y_norm: Normalized spanwise position.
        :param real: Whether to use real or simulated movement.

        :return: (num_steps, 3) array of coordinates over time.
        """
        return np.array([
            self.get_coordinates(k, x_norm, y_norm, real=real)
            for k in range(self.num_steps)
        ])
    
    def compare_trajectories(self, x_norm, y_norm):
        """Computes the difference between real and simulated trajectories for a point on
        the wing, evaluated at all time steps.

        :param x_norm: Normalized chordwise coordinate.
        :param y_norm: Normalized spanwise coordinate.

        :return: (num_steps, 3) array of trajectory differences.
        """
        real = self.get_full_trajectory(x_norm, y_norm, real=True)
        simulated = self.get_full_trajectory(x_norm, y_norm, real=False)
        return real - simulated

    def plot_trajectory_3d(self, x_norm, y_norm):
        """Plots the 3D trajectory of a point on the wing, comparing real vs simulated
        movement over time.

        :param x_norm: Normalized chordwise coordinate.
        :param y_norm: Normalized spanwise coordinate.

        :return: None.
        """
        traj_real = self.get_full_trajectory(x_norm, y_norm, real=True)
        traj_simulated = self.get_full_trajectory(x_norm, y_norm, real=False)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(traj_real[:, 0], traj_real[:, 1], traj_real[:, 2], 
                label="Real", linewidth=2)
        ax.plot(traj_simulated[:, 0], traj_simulated[:, 1], traj_simulated[:, 2], 
                label="Simulated", linestyle='--')

        ax.set_title("Trajectory Comparison")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()

        all_points = np.vstack([traj_real, traj_simulated])

        X = all_points[:, 0]
        Y = all_points[:, 1]
        Z = all_points[:, 2]

        min_val = min(X.min(), Y.min(), Z.min())
        max_val = max(X.max(), Y.max(), Z.max())

        X_center = (X.max() + X.min()) / 2
        Y_center = (Y.max() + Y.min()) / 2
        Z_center = (Z.max() + Z.min()) / 2

        center = (min_val + max_val) / 2
        radius = (max_val - min_val) / 2

        ax.set_xlim(X_center - radius, X_center + radius)
        ax.set_ylim(Y_center - radius, Y_center + radius)
        ax.set_zlim(Z_center - radius, Z_center + radius)

        plt.show()

    def plot_difference_position_versus_time(self):
        """Computes and plots the mean positional error between real and simulated movement
        by sampling many points across the wing surface.

        :param None:

        :return: None.
        """
        sum_dif = np.zeros(self.num_steps)

        x_values = np.linspace(0, 1, 100)
        y_values = np.linspace(0, 1, 100)

        for x in x_values:
            for y in y_values:
                diff = self.compare_trajectories(x, y)
                dist = np.linalg.norm(diff, axis=1) 
                sum_dif += dist 
        
        mean = sum_dif/(len(x_values) * len(y_values))

        t = np.arange(self.num_steps) * self.delta_time
        plt.plot(t, mean)
        plt.xlabel("Time (s)")
        plt.ylabel("mean of the distances (m)")
        plt.title("Mean error between actual and simulated trajectories")
        plt.show()

    def get_mean_difference_position_mae(self):
        """Computes and prints the mean positional error between real and simulated movement

        :param None:

        :return: None.
        """
        sum_dif = np.zeros(self.num_steps)

        x_values = np.linspace(0, 1, 100)
        y_values = np.linspace(0, 1, 100)

        for x in x_values:
            for y in y_values:
                diff = self.compare_trajectories(x, y)
                dist = np.linalg.norm(diff, axis=1) 
                sum_dif += dist 
        
        mean = sum_dif/(len(x_values) * len(y_values))
        
        print("Mean error between actual and simulated trajectories: " + str(np.mean(mean)) + " m")
        return np.mean(mean)

    def get_mean_difference_position_rms(self):
        """Computes and returns the RMS positional error between real and simulated movement."""

        sum_sq_dif = np.zeros(self.num_steps)

        x_values = np.linspace(0, 1, 100)
        y_values = np.linspace(0, 1, 100)

        for x in x_values:
            for y in y_values:
                diff = self.compare_trajectories(x, y)
                dist_sq = np.linalg.norm(diff, axis=1) ** 2
                sum_sq_dif += dist_sq

        mean_sq = sum_sq_dif / (len(x_values) * len(y_values))

        # RMS temporelle + spatiale
        rms = np.sqrt(np.mean(mean_sq))

        print(
            "RMS error between actual and simulated trajectories: "
            + str(rms)
            + " m"
        )

        return rms


    #sections ------------------------------------------------------------------------------------------------------------------------------------------------------------

    def get_section_by_column(self, col_index, real=True):
        """Retrieves the coordinates of all panel vertices belonging to a vertical section
        (column of panels) across all time steps.

        :param col_index: Index of the panel column.
        :param real: Whether to use real or simulated data.

        :return: List of section coordinates for each time-step.
        """
    
        if real:
            airplane = self.solver.steady_problems[step].airplanes[0]
        else:
            airplane = self.simulated_solver.steady_problems[step].airplanes[0]

        wing_index = 0
        coords_by_step = []

        for step in range(self.num_steps):

            if real:
                airplane = self.solver.steady_problems[step].airplanes[0]
            else:
                airplane = self.simulated_solver.steady_problems[step].airplanes[0]
            wing = airplane.wings[wing_index]

            panels_column = wing.panels[:, col_index]
            section_points = []

            for panel in panels_column:
                vertices = [
                    panel.Flpp_G_Cg,
                    panel.Frpp_G_Cg,
                    panel.Blpp_G_Cg,
                    panel.Brpp_G_Cg
                ]

                section_points.append([vertices[0][0], vertices[0][2]])
                section_points.append([vertices[2][0], vertices[2][2]])

            coords_by_step.append(section_points)

        return coords_by_step

    def plot_section(self, col_index, compare_simulated=True):
        """Plots the flapping motion of a wing section over half a period (upstroke and
        downstroke), and optionally compares real and simulated data.

        :param col_index: Column index of the wing panels to plot.
        :param compare_simulated: Whether to also plot simulated data.

        :return: None.
        """
        section_real = self.get_section_by_column(col_index, real=True)

        if compare_simulated:
            section_simulated = self.get_section_by_column(col_index, real=False)

        maximum_z = section_real[0][0][1]
        i_peak = 0
        if maximum_z < 0:
            while maximum_z <= section_real[i_peak + 1][0][1]:
                i_peak += 1
                maximum_z = section_real[i_peak][0][1]
        else:
            while maximum_z >= section_real[i_peak + 1][0][1]:
                i_peak += 1
                maximum_z = section_real[i_peak][0][1]

        period_steps = int(self.period / self.delta_time)

        half = period_steps // 2

        if section_real[i_peak][0][1] < 0:
            up_real = section_real[i_peak:i_peak + half]
            down_real = section_real[i_peak + half:i_peak + period_steps]

            if compare_simulated:
                up_simulated = section_simulated[i_peak:i_peak + half]
                down_simulated = section_simulated[i_peak + half:i_peak + period_steps]
        
        else :
            down_real = section_real[i_peak:i_peak + half]
            up_real = section_real[i_peak + half:i_peak + period_steps]

            if compare_simulated:
                down_simulated = section_simulated[i_peak:i_peak + half]
                up_simulated = section_simulated[i_peak + half:i_peak + period_steps]
              

        fig, (ax_down, ax_up) = plt.subplots(1, 2, figsize=(12, 5))

        color_real = 'tab:blue'
        color_simulated = 'tab:red'

        global_x, global_z = [], []

        def accumulate(section):
            for pts in section:
                for x, z in pts:
                    global_x.append(x)
                    global_z.append(z)

        accumulate(down_real)
        accumulate(up_real)
        if compare_simulated:
            accumulate(down_simulated)
            accumulate(up_simulated)

        xmin, xmax = min(global_x), max(global_x)
        zmin, zmax = min(global_z), max(global_z)

        for i in range(len(down_real)//3):
            x_r = [p[0] for p in down_real[i*3]]
            z_r = [p[1] for p in down_real[i*3]]
            ax_down.plot(x_r, z_r, color_real, alpha=0.8, label="Real" if i == 0 else "")
            if compare_simulated:
                x_f = [p[0] for p in down_simulated[i*3]]
                z_f = [p[1] for p in down_simulated[i*3]]
                ax_down.plot(x_f, z_f, color_simulated, linestyle="--", alpha=0.8, label="Simulated" if i == 0 else "")

        ax_down.set_xlim(xmin, xmax)
        ax_down.set_ylim(zmin, zmax)
        ax_down.set_aspect("equal", adjustable="box")
        ax_down.set_title("Down-stroke")
        ax_down.set_xlabel("X"); ax_down.set_ylabel("Z")
        ax_down.legend()

        for i in range(len(up_real)//3):
            x_r = [p[0] for p in up_real[i*3]]
            z_r = [p[1] for p in up_real[i*3]]
            ax_up.plot(x_r, z_r, color_real, alpha=0.8, label="Real" if i == 0 else "")
            if compare_simulated:
                x_f = [p[0] for p in up_simulated[i*3]]
                z_f = [p[1] for p in up_simulated[i*3]]
                ax_up.plot(x_f, z_f, color_simulated, linestyle="--", alpha=0.8, label="Simulated" if i == 0 else "")

        ax_up.set_xlim(xmin, xmax)
        ax_up.set_ylim(zmin, zmax)
        ax_up.set_aspect("equal", adjustable="box")
        ax_up.set_title("Up-stroke")
        ax_up.set_xlabel("X"); ax_up.set_ylabel("Z")
        ax_up.legend()

        plt.tight_layout()
        plt.show()

    #panel forces ------------------------------------------------------------------------------------------------------------------------------------------------
     
    def get_forces(self, step, x_norm, y_norm, scalar_type, real=True):
        """Returns the aerodynamic scalar force (lift, side force, or induced drag) at a
        specific point and time step.

        :param step: Time-step index.
        :param x_norm: Normalized chordwise coordinate.
        :param y_norm: Normalized spanwise coordinate.
        :param scalar_type: Type of force ("lift", "side force", "induced drag").
        :param real: Whether to use real or simulated data.

        :return: The scalar force value at that location and time.
        """
        if real == True:
            airplane = self.solver.steady_problems[step].airplanes
            qInf__E = self.solver.steady_problems[step].operating_point.qInf__E
            self.Nch, self.Nsp, u, v = self._track_point(airplane[0], 0, x_norm, y_norm)
        else:
            airplane = self.simulated_solver.steady_problems[step].airplanes
            qInf__E = self.simulated_solver.steady_problems[step].operating_point.qInf__E
            self.Nch, self.Nsp, u, v = self._track_point(airplane[0], 0, x_norm, y_norm)

        these_scalars = ps.output._get_scalars(airplane, scalar_type, qInf__E)
        index = self.Nch * self.base_airplane.wings[0].wing_cross_sections[0].num_spanwise_panels + self.Nsp
        this_scalar = these_scalars[index]

        return this_scalar
    
    def get_full_forces(self, x_norm, y_norm, scalar_type, real=True):
        """Computes the full time history of a scalar aerodynamic force at a specific point
        on the wing.

        :param x_norm: Normalized chordwise coordinate.
        :param y_norm: Normalized spanwise coordinate.
        :param scalar_type: Type of aerodynamic force.
        :param real: Whether to use real or simulated forces.

        :return: (num_steps,) array of force values.
        """
        return np.array([self.get_forces(k, x_norm, y_norm, real=real, scalar_type=scalar_type) for k in range(self.num_steps)])
    
    def compare_forces(self, x_norm, y_norm, scalar_type):
        """Returns the difference between real and simulated scalar forces at a given point
        for all time steps.

        :param x_norm: Normalized chordwise coordinate.
        :param y_norm: Normalized spanwise coordinate.
        :param scalar_type: Type of aerodynamic force.

        :return: (num_steps,) array of force differences.
        """
        real = self.get_full_forces(x_norm, y_norm, scalar_type, real=True)
        simulated = self.get_full_forces(x_norm, y_norm, scalar_type, real=False)
        return real - simulated
    
    def plot_panel_forces(self, x_norm, y_norm):
        """Plots the real and simulated scalar aerodynamic forces (lift, side force, induced
        drag) at a specific point on the wing.

        :param x_norm: Normalized chordwise coordinate.
        :param y_norm: Normalized spanwise coordinate.

        :return: None.
        """
        scalar_types = ["lift", "side force", "induced drag"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for ax, scalar_type in zip(axes, scalar_types):

            real = self.get_full_forces(x_norm, y_norm, scalar_type, real=True)
            simulated = self.get_full_forces(x_norm, y_norm, scalar_type, real=False)

            ax.plot(real, label="Real", linewidth=2)
            ax.plot(simulated, label="Simulated", linestyle='--')

            ax.set_title(scalar_type.capitalize())
            ax.set_xlabel("Time step")
            ax.set_ylabel(scalar_type)
            ax.legend()

        fig.suptitle("Panel forces " + str(self.Nch) + "x" + str(self.Nsp) + "", fontsize=16)
        plt.tight_layout()
        plt.show()

    #general forces ------------------------------------------------------------------------------------------------------------------------------------------------
    def compute_forces_over_time(self, real=True):
        """Retrieves the full time evolution of total forces, force coefficients, moments,
        and moment coefficients from either the real or simulated solver.

        :param real: Whether to use real or simulated forces.

        :return: Dictionary containing arrays of forces and moments over time.
        """
        solver = self.solver if real else self.simulated_solver
        num_steps = solver.num_steps
        forces = []
        forces_coeff = []
        moments = []
        moments_coeff = []

        for step in range(num_steps):
            airplane = solver.steady_problems[step].airplanes[0]

            forces.append(airplane.forces_W.copy())
            forces_coeff.append(airplane.forceCoefficients_W.copy())
            moments.append(airplane.moments_W_CgP1.copy())
            moments_coeff.append(airplane.momentCoefficients_W_CgP1.copy())

        return {
            "forces": np.array(forces),
            "forces_coeff": np.array(forces_coeff),
            "moments": np.array(moments),
            "moments_coeff": np.array(moments_coeff),
        }

    def plot_forces(self):
        """Plots time histories of forces, force coefficients, moments, and moment
        coefficients for both real and simulated solvers, grouped on multiple figures.

        :param None:

        :return: None.
        """
        _prism = [
            "#5F4690", "#1D6996", "#38A6A5", "#0F8554", "#73AF48",
            "#EDAD08", "#E17C05", "#CC503E", "#94346E", "#6F4070",
            "#994E95", "#666666",
        ]

        [
            _drag_color,
            _side_color,
            _lift_color,
            _roll_color,
            _pitch_color,
            _yaw_color,
        ] = _prism[3:9]

        _num_markers = 6
        _marker_size = 8
        _marker_spacing = 1.0 / _num_markers

        R = self.compute_forces_over_time(real=True)
        F = self.compute_forces_over_time(real=False)

        t = np.arange(len(R["forces"]))

        
        fig1, axs1 = plt.subplots(3, 1, figsize=(10, 12))
        fig1.suptitle("Forces (Real vs Simulated)", fontsize=16)

        force_names = ["Induced Drag", "Side Force", "Lift"]
        idx = [0, 1, 2]
        signs = [-1, 1, -1]
        colors = [_drag_color, _side_color, _lift_color]

        for ax, name, i, s, c in zip(axs1, force_names, idx, signs, colors):
            ax.plot(t, s * R["forces"][:, i], label=f"Real {name}", color=c,
                    marker=".", markevery=(_marker_spacing*0, _marker_spacing), markersize=_marker_size)
            ax.plot(t, s * F["forces"][:, i], "--", label=f"simulated {name}", color=c)
            ax.set_ylabel(name); ax.grid(True, alpha=0.3); ax.legend()

        axs1[-1].set_xlabel("Time step")
        plt.tight_layout()

        
        fig2, axs2 = plt.subplots(3, 1, figsize=(10, 12))
        fig2.suptitle("Force Coefficients (Real vs Simulated)", fontsize=16)

        for ax, name, i, s, c in zip(axs2, force_names, idx, signs, colors):
            ax.plot(t, s * R["forces_coeff"][:, i], label=f"Real {name} Coeff", color=c,
                    marker=".", markevery=(_marker_spacing*0, _marker_spacing), markersize=_marker_size)
            ax.plot(t, s * F["forces_coeff"][:, i], "--", label=f"simulated {name} Coeff", color=c)
            ax.set_ylabel(name + " Coeff"); ax.grid(True, alpha=0.3); ax.legend()

        axs2[-1].set_xlabel("Time step")
        plt.tight_layout()

        
        fig3, axs3 = plt.subplots(3, 1, figsize=(10, 12))
        fig3.suptitle("Moments (Real vs Simulated)", fontsize=16)

        moment_names = ["Roll", "Pitch", "Yaw"]
        idxm = [0, 1, 2]
        colors_m = [_roll_color, _pitch_color, _yaw_color]

        for ax, name, i, c in zip(axs3, moment_names, idxm, colors_m):
            ax.plot(t, R["moments"][:, i], label=f"Real {name}", color=c,
                    marker=".", markevery=(_marker_spacing*0, _marker_spacing), markersize=_marker_size)
            ax.plot(t, F["moments"][:, i], "--", label=f"simulated {name}", color=c)
            ax.set_ylabel(name); ax.grid(True, alpha=0.3); ax.legend()

        axs3[-1].set_xlabel("Time step")
        plt.tight_layout()

        
        fig4, axs4 = plt.subplots(3, 1, figsize=(10, 12))
        fig4.suptitle("Moment Coefficients (Real vs Simulated)", fontsize=16)

        for ax, name, i, c in zip(axs4, moment_names, idxm, colors_m):
            ax.plot(t, R["moments_coeff"][:, i], label=f"Real {name} Coeff", color=c,
                    marker=".", markevery=(_marker_spacing*0, _marker_spacing), markersize=_marker_size)
            ax.plot(t, F["moments_coeff"][:, i], "--", label=f"simulated {name} Coeff", color=c)
            ax.set_ylabel(name + " Coeff"); ax.grid(True, alpha=0.3); ax.legend()

        axs4[-1].set_xlabel("Time step")
        plt.tight_layout()

        plt.show()

    # Wing animation------------------------------------------------------------------------------------------------------------------------------------------------
    def get_wing_data(self, real=True, wing_index=0):
        """Extracts and returns the coordinates of all wing panels at each time step for
        either the real or simulated wing.

        :param real: Whether to extract real or simulated wing geometry.
        :param wing_index: Index of the wing.

        :return: Dictionary mapping each time step to an array of panel vertex coordinates.
        """

        coords={}

        for step in range(self.num_steps):
            coords_step = []

            if real:
                airplane = self.solver.steady_problems[step].airplanes[0]
            else:
                airplane = self.simulated_solver.steady_problems[step].airplanes[0]

            wing = airplane.wings[wing_index]
            panels = np.ravel(wing.panels)

            for panel in panels:
                for vertex_name in [
                    "Flpp_GP1_CgP1", "Frpp_GP1_CgP1",
                    "Brpp_GP1_CgP1", "Blpp_GP1_CgP1"
                ]:
                    coords_step.append(getattr(panel, vertex_name))

            coords[step] = np.array(coords_step)

        return coords


    def max_edge_len(self, triangle, points):
        """Computes the maximum edge length of a triangular face defined by three point
        indices.

        :param triangle: Indices of the triangle's three vertices.
        :param points: Array of 3D point coordinates.

        :return: Longest edge length of the triangle.
        """
        verts = points[triangle]
        edges = [np.linalg.norm(verts[i] - verts[j]) for i in range(3) for j in range(i + 1, 3)]
        return max(edges)

    def plot_wing(self, points):
        """Generates a filtered triangle mesh (Delaunay-based) representing the wing
        surface, removing triangles with excessively long edges.

        :param points: (N,3) array of surface points.

        :return: (M,3) array of triangle indices.
        """
        proj2 = points[:, :2] 
        delaunay = Delaunay(proj2)
        triangles = delaunay.simplices.copy()

        dists = []
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                dists.append(np.linalg.norm(points[i] - points[j]))

        max_allowed = np.percentile(dists, 60)
        filtered = [tri for tri in triangles if self.max_edge_len(tri, points) <= max_allowed]

        return np.array(filtered)

    def dynamic_wing(self, wing_index=0):
        """Creates an animated visualization comparing the 3D bisector mesh of the real
        wing and the simulated wing over time.

        :param wing_index: Index of the wing to animate.

        :return: A matplotlib FuncAnimation object containing the animation.
        """
        points_real_dict = self.get_wing_data(real=True, wing_index=wing_index)
        steps = sorted(points_real_dict.keys())
        points_real = np.array([points_real_dict[s] for s in steps])

        points_simulated_dict = self.get_wing_data(real=False, wing_index=wing_index + 1)
        points_simulated = np.array([points_simulated_dict[s] for s in steps])

        faces = self.plot_wing(points_real[0])

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlim(-0.500, 0.500)
        ax.set_ylim(-0.500, 0.500)
        ax.set_zlim(-0.500, 0.500)

        def update(frame):
            ax.cla()

            ax.set_xlim(-0.500, 0.500)
            ax.set_ylim(-0.500, 0.500)
            ax.set_zlim(-0.500, 0.500)

            
            P = points_real[frame]

            ax.plot_trisurf(
                P[:, 0], P[:, 1], P[:, 2],
                triangles=faces, color='cyan',
                edgecolor='k', linewidth=0.2, alpha=0.5
            )
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], color='b', s=12)

            
            F = points_simulated[frame].copy()
        

            ax.plot_trisurf(
                F[:, 0], F[:, 1], F[:, 2],
                triangles=faces, color='orange',
                edgecolor='k', linewidth=0.2, alpha=0.5
            )
            ax.scatter(F[:, 0], F[:, 1], F[:, 2], color='r', s=12)

            ax.set_title(f"Real (cyan/blue) vs simulated Mirrored (orange/red) — Frame {frame+1}/{len(points_real)}")
            return ax,

        ani = FuncAnimation(fig, update, frames=len(points_real), interval=80, blit=False)
        writer = PillowWriter(fps=12)
        ani.save("Dynamic.gif", writer=writer)
        plt.show()

