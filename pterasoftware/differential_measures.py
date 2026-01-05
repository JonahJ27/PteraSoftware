"""
This module provides a full analysis toolbox for comparing real wing kinematics
(with tracked markers) to a simulated reconstruction using the
Unsteady Ring Vortex Lattice Method (URVLM) from Ptera Software.

The module contains the following classes:
    - Analysis: A complete analysis pipeline for extracting real wing movement
      from tracker data, reconstructing an equivalent periodic motion, building
      a simulated airplane/movement/problem/solver, and comparing the real and
      simulated trajectories and aerodynamics.

This module contains the following main functionalities:

    Extraction of movement characteristics from tracker data:
        - amplitude_max, amplitude_min
        - phase_shift, status_phase
        - geometric offset (delta)

    Construction of a simulated (“simulated”) airplane and its associated movement
    designed to mimic the real measured kinematics.

    Automatic building and execution of a simulated unsteady solver for comparison.

    Trajectory analysis tools:
        - get_coordinates: Returns 3D coordinates of any normalized point.
        - get_full_trajectory: Returns time evolution of a point.
        - compare_trajectories: Real vs simulated difference.
        - plot_trajectory_3d: Visual comparison in 3D.
        - plot_difference_position_versus_time: Mean tracking error over time.

    Wing section extraction & visualization:
        - get_section_by_column: Extracts a vertical section of the wing.
        - plot_section: Down-stroke / Up-stroke comparison of real vs simulated.

    Panel force analysis:
        - get_forces: Returns lift, side-force, or induced drag for one panel.
        - get_full_forces: Time series of these forces.
        - compare_forces: Real vs simulated force differences.
        - plot_panel_forces: Time-series plot for lift, side force and drag.

    Global aerodynamic forces and moments:
        - compute_forces_over_time: Extracts forces, moments and coefficients
          at every time step for real or simulated solver.

This module integrates geometry, movement reconstruction, solver execution,
and multi-level comparison tools for in-depth validation of flapping wing
kinematics and their aerodynamic signatures.
"""


import numpy as np
import matplotlib.pyplot as plt
import pterasoftware as ps
from scipy.spatial import Delaunay
from matplotlib.animation import FuncAnimation, PillowWriter

class Analysis:
   
    def __init__(self, unsteady_solver,f):


        """
        Initialization method for the Analysis class.

        Parameters
        ----------
        unsteady_solver : UnsteadyRingVortexLatticeMethodSolver
            The solver object that will be used to extract the real movement data.
        f : float
            The frequency of the movement in Hz.

        Attributes
        ----------
        solver : UnsteadyRingVortexLatticeMethodSolver
            The solver object that was used to extract the real movement data.
        problem : UnsteadyProblem
            The unsteady problem object that was used to extract the real movement data.
        movement : Movement
            The movement object that was used to extract the real movement data.
        airplane_movements : AirplaneMovement
            The AirplaneMovement object that was used to extract the real movement data.
        num_steps : int
            The number of steps in the movement.
        delta_time : float
            The time step of the movement in seconds.
        base_airplane : Airplane
            The Airplane object that was used to extract the real movement data.
        data : dict
            A dictionary containing the data of the airfoil used in the
            Airplane object.
        list_trackers : list
            A list of the trackers used in the airfoil.
        f : float
            The frequency of the movement in Hz.
        periode : float
            The period of the movement in seconds.
        simulated_airplane : Airplane
            A simulated Airplane object that was used to generate the simulated
            movement data.
        simulated_movement : Movement
            A simulated Movement object that was used to generate the simulated
            movement data.
        simulated_problem : UnsteadyProblem
            A simulated UnsteadyProblem object that was used to generate the simulated
            movement data.
        simulated_solver : UnsteadyRingVortexLatticeMethodSolver
            A simulated UnsteadyRingVortexLatticeMethodSolver object that was used to generate
            the simulated movement data.
        """
        self.solver = unsteady_solver
        self.problem = self.solver.unsteady_problem
        self.movement = self.problem.movement
        self.airplane_movements = self.movement.airplane_movements[0]
        self.num_steps = self.movement.num_steps
        self.delta_time = self.movement.delta_time
        self.base_airplane = self.airplane_movements.base_airplane
        airfoil = self.base_airplane.wings[0].wing_cross_sections[0].airfoil

        self.data = airfoil.data
        self.list_trackers = airfoil.list_trackers
        self.f=f
        self.period=1/self.f
      

        self.extract_movement_data()

        self.simulated_airplane = self.build_simulated_airplane()
        self.simulated_movement = self.build_simulated_movement()
        self.simulated_problem = self.build_simulated_problem()
        self.simulated_solver = self.build_simulated_solver()


    def build_simulated_airplane(self):
        """
        Build a simulated Airplane object that is identical to the real Airplane.

        Attributes
        ----------
        simulated_airplane : Airplane
            The simulated Airplane object that was built.

        Returns
        -------
        Airplane
            The simulated Airplane object that was built.
        """
        real_wing = self.base_airplane.wings
        

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
                for cs in real_wing[0].wing_cross_sections
            ],
            name="simulatedWing",
            Ler_Gs_Cgs=real_wing[0].Ler_Gs_Cgs,
            angles_Gs_to_Wn_ixyz=(4, 0.0, 0.0),
            symmetric=True,
            mirror_only=False,
            symmetryNormal_G=(0.0, 1.0, 0.0),
            symmetryPoint_G_Cg=(0.0, 0.0, 0.0),
            num_chordwise_panels=real_wing[0].num_chordwise_panels,
            chordwise_spacing=real_wing[0].chordwise_spacing,
        )

        return ps.geometry.airplane.Airplane(
            wings=[simulated_wing],
            name="simulatedAirplane",
            Cg_E_CgP1=(0.0, 0.0, 0.0),
            angles_E_to_B_izyx=self.base_airplane.angles_E_to_B_izyx,
            weight=self.base_airplane.weight,
            s_ref=self.base_airplane.s_ref,
            c_ref=self.base_airplane.c_ref,
            b_ref=self.base_airplane.b_ref,
        )

    def extract_movement_data(self):

        """
        Extracts movement data from the given data.

        This method takes the data collected from the trackers and extracts the
        maximum and minimum amplitudes, dephasage, and status phase.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        This method assumes that the data is collected in the following format:
        tracker_1: (x, y, z)
        tracker_2: (x, y, z)
        ...
        tracker_n: (x, y, z)
        where (x, y, z) are the coordinates of the tracker at each time step.

        The method will calculate the maximum and minimum amplitudes, dephasage,
        and status phase for each tracker, and then calculate the mean of
        the maximum and minimum amplitudes, dephasage, and status phase for all
        trackers.

        The results are stored in the following attributes of the class:

        amplitude_max: float
            The mean of the maximum amplitudes for all trackers.
        amplitude_min: float
            The mean of the minimum amplitudes for all trackers.
        dephasage: float
            The mean of the dephasages for all trackers.
        status_phase: float
            The mean of the status phases for all trackers.
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

        print(self.amplitude_max,self.amplitude_min,self.phase_shift,self.status_phase)


    def build_simulated_movement(self):

        """
        Reconstruct a simulated movement from the tracked data points.

        Parameters
        ----------
        None

        Returns
        -------
        airplane_movement : AirplaneMovement
            The simulated movement is represented by an AirplaneMovement object.
        """
        simulated_airplane = self.simulated_airplane

        A = abs((self.amplitude_max - self.amplitude_min)/2)

        theta0 = self.phase_shift
        theta1 = self.status_phase

        s = np.clip(theta0/ A, -1.0, 1.0)

        dt = self.delta_time
        omega = 2*np.pi*self.f 

        dtheta = (theta1 - theta0) / dt
        c = np.clip(dtheta / (A * omega), -1.0, 1.0)

        phi = np.arctan2(s, c)

        phi = np.degrees(phi)
        A = np.degrees(A)


        main_wing_cross_section_movement = []
        reflected_wing_cross_section_movement = []

        for i, cs in enumerate(simulated_airplane.wings[0].wing_cross_sections):

            mov = ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
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

            main_wing_cross_section_movement.append(mov)

        if len(simulated_airplane.wings) >= 1:
            for i, cs in enumerate(simulated_airplane.wings[1].wing_cross_sections):

                mov = ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
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

                reflected_wing_cross_section_movement.append(mov)
        else:
            reflected_wing_cross_section_movement = []


 
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
        

        if reflected_wing_cross_section_movement != []:
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


        airplane_movement = ps.movements.airplane_movement.AirplaneMovement(
            base_airplane=simulated_airplane,
            wing_movements=[maine_wing_movement, reflected_main_wing_movement],
            ampCg_E_CgP1=(0.0, 0.0, 0.0),
            periodCg_E_CgP1=(0.0, 0.0, 0.0),
            spacingCg_E_CgP1=("sine", "sine", "sine"),
            phaseCg_E_CgP1=(0.0, 0.0, 0.0),

            ampAngles_E_to_B_izyx=(0.0, 0.0, 0.0),
            periodAngles_E_to_B_izyx=(0.0, 0.0, 0.0),
            spacingAngles_E_to_B_izyx=("sine", "sine", "sine"),
            phaseAngles_E_to_B_izyx=(0.0, 0.0, 0.0)
        )

        return airplane_movement
    
    def build_simulated_problem(self):

        """
        Builds a simulated UnsteadyProblem based on the self.movement attribute.

        The simulated UnsteadyProblem is based on the same operating point movement as
        the original, but uses the simulated movement generated by the
        build_simulated_movement method.

        Returns:
            UnsteadyProblem: The simulated UnsteadyProblem.
        """
        real_op_mov = self.movement.operating_point_movement

        simulated_op_mov = ps.movements.operating_point_movement.OperatingPointMovement(
            base_operating_point = real_op_mov.base_operating_point,
            periodVCg__E        = real_op_mov.periodVCg__E,
            spacingVCg__E       = real_op_mov.spacingVCg__E,
        )

        simulated_movement = ps.movements.movement.Movement(
            airplane_movements       = [self.simulated_movement],
            operating_point_movement = simulated_op_mov,
            delta_time               = self.delta_time,
            num_steps                = self.num_steps,
        )

        return ps.problems.UnsteadyProblem(simulated_movement)


    def build_simulated_solver(self):
        """
        Builds a simulated UnsteadyRingVortexLatticeMethodSolver based on the self.simulated_problem attribute.

        The simulated solver is based on the same UnsteadyProblem as the original, but uses
        the simulated UnsteadyProblem generated by the build_simulated_problem method.

        Returns:
            UnsteadyRingVortexLatticeMethodSolver: The simulated UnsteadyRingVortexLatticeMethodSolver.
        """
        solver = ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(
            unsteady_problem=self.simulated_problem
        )
        solver.run()
        return solver

    def _track_point(self, airplane, wing_index, x_norm, y_norm):
        """
        Private method to track the position of a point on the wing.

        Parameters
        ----------
        airplane : Airplane
            The Airplane object to track the point on.
        wing_index : int
            The index of the wing to track the point on.
        x_norm : float
            The normalized x-coordinate of the point to track (0 <= x_norm <= 1).
        y_norm : float
            The normalized y-coordinate of the point to track (0 <= y_norm <= 1).

        Returns
        -------
        i : int
            The index of the chordwise panel containing the point.
        j : int
            The index of the spanwise panel containing the point.
        u : float
            The normalized x-coordinate of the point within the panel (0 <= u <= 1).
        v : float
            The normalized y-coordinate of the point within the panel (0 <= v <= 1).
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
        """
        Retrieve a clean (3,) coordinate for a point on the wing.
        """

        airplane = (
            self.movement.airplanes[0][step]
            if real
            else self.simulated_solver.unsteady_problem.movement.airplanes[0][step]
        )

        i, j, u, v = self._track_point(airplane, 0, x_norm, y_norm)
        panel = airplane.wings[0].panels[i, j]

        p00 = np.asarray(panel.Flpp_G_Cg).reshape(-1)
        p10 = np.asarray(panel.Frpp_G_Cg).reshape(-1)
        p01 = np.asarray(panel.Blpp_G_Cg).reshape(-1)
        p11 = np.asarray(panel.Brpp_G_Cg).reshape(-1)

        # Keep only first 3 components if more are present
        p00 = p00[:3]
        p10 = p10[:3]
        p01 = p01[:3]
        p11 = p11[:3]

        # Bilinear interpolation
        P = (
            (1 - u) * (1 - v) * p00 +
            u       * (1 - v) * p10 +
            (1 - u) * v       * p01 +
            u       * v       * p11
        )

        return np.asarray(P).reshape(3,)

    def get_full_trajectory(self, x_norm, y_norm, real=True):
        """
        Retrieve the full trajectory of a point on the wing.

        :param x_norm: float
            Normalized x coordinate of the point.
        :param y_norm: float
            Normalized y coordinate of the point.
        :param real: bool, optional
            If True, use the real movement data. If False, use the simulated
            solver data. Default is True.

        :return: numpy array
            The full trajectory of the point, with shape (num_steps, 3).
        """
        return np.array([
            self.get_coordinates(k, x_norm, y_norm, real=real)
            for k in range(self.num_steps)
        ])
    
    def compare_trajectories(self, x_norm, y_norm):
        
        """
        Compare the full trajectory of a point on the wing between the real and simulated
        movement data.

        :param x_norm: float
            Normalized x coordinate of the point.
        :param y_norm: float
            Normalized y coordinate of the point.

        :return: numpy array
            The difference between the real and simulated trajectories, with shape (num_steps, 3).
        """
        real = self.get_full_trajectory(x_norm, y_norm, real=True)
        simulated = self.get_full_trajectory(x_norm, y_norm, real=False)
        return real - simulated

    def plot_trajectory_3d(self, x_norm, y_norm):
        
        """
        Plot the 3D trajectory of a point on the wing, comparing the real and simulated movement data.

        Parameters
        ----------
        x_norm : float
            Normalized x coordinate of the point.
        y_norm : float
            Normalized y coordinate of the point.

        Returns
        -------
        None
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

        """
        Plot the mean difference between the actual and simulated trajectories as a function of time.

        The mean difference is calculated by sampling the wing at 100x100 points and calculating the
        difference between the actual and simulated trajectories at each point. The mean of these
        differences is then plotted as a function of time.

        Parameters
        ----------
        None

        Returns
        -------
        None
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
        plt.ylabel("mean of the distances (mm)")
        plt.title("Mean error between actual and simulated trajectories")
        plt.show()



    
#sections ------------------------------------------------------------------------------------------------------------------------------------------------------------
    def get_section_by_column(self, col_index, real=True):

        """
        Retrieve the coordinates of a section of the wing by column index.

        :param col_index: int
            The index of the column of panels to retrieve.
        :param real: bool, optional
            If True, use the real movement data. If False, use the simulated
            solver data. Default is True.

        :return: list
            A list of coordinates of the section, with shape (num_steps, num_panels, 2, 2).
        """
        movement = (
            self.movement
            if real
            else self.simulated_solver.unsteady_problem.movement
        )

        wing_index = 0
        coords_by_step = []

        for step in range(movement.num_steps):

            airplane = movement.airplanes[0][step]
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

        """
        Plot the trajectory of a section of the wing.

        :param col_index: int
            The index of the column of panels to plot.
        :param compare_simulated: bool, optional
            If True, compare the real movement data with the simulated
            solver data. Default is True.
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

        # --- Plot UP ---
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
        """
        Return the scalar force of type scalar_type at the given step and point (x_norm, y_norm)
        on the wing of the given airplane.

        Parameters:
        step : int

            The step in time.

        x_norm : float

            The normalized x-coordinate of the point.

        y_norm : float

            The normalized y-coordinate of the point.

        scalar_type : str

            The type of the scalar force to return. Possible values are "lift", "side force", and "induced drag".

        real : bool, optional

            If True, the function will return the scalar force of the real solver.
            If False, the function will return the scalar force of the simulated solver. Default is True.

        Returns:
        scalar : float

            The scalar force of the given type at the given step and point.

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
        """
        Returns the scalar forces of a given type at a given point for all time steps.

        Parameters:
        x_norm : float

            The normalized x-coordinate of the point.

        y_norm : float

            The normalized y-coordinate of the point.

        scalar_type : str

            The type of the scalar force to return. Possible values are "lift", "side force", and "induced drag".

        real : bool, optional

            If True, the function will return the scalar force of the real solver.
            If False, the function will return the scalar force of the simulated solver. Default is True.

        Returns:
        scalar_forces : ndarray of shape (num_steps,) and dtype float

            The scalar forces of the given type at the given point for all time steps.
        """
        return np.array([self.get_forces(k, x_norm, y_norm, real=real, scalar_type=scalar_type) for k in range(self.num_steps)])
    
    def compare_forces(self, x_norm, y_norm, scalar_type):
        """
        Returns the difference between the scalar forces of the real solver and the simulated solver
        at a given point for all time steps.

        Parameters:
        x_norm : float

            The normalized x-coordinate of the point.

        y_norm : float

            The normalized y-coordinate of the point.

        scalar_type : str

            The type of the scalar force to return. Possible values are "lift", "side force", and "induced drag".

        Returns:
        scalar_force_difference : ndarray of shape (num_steps,) and dtype float

            The difference between the scalar forces of the real solver and the simulated solver
            at the given point for all time steps.
        """
        real = self.get_full_forces(x_norm, y_norm, scalar_type, real=True)
        simulated = self.get_full_forces(x_norm, y_norm, scalar_type, real=False)
        return real - simulated
    
    def plot_panel_forces(self, x_norm, y_norm):


        """
        Plots the comparison between the real and simulated scalar forces for a given point (x_norm, y_norm)
        on the wing of the given airplane.

        Parameters:
        x_norm : float

            The normalized x-coordinate of the point.

        y_norm : float

            The normalized y-coordinate of the point.

        Returns:
        None

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

        """
        Returns the forces, force coefficients, moments, and moment coefficients for all time steps.

        Parameters:
        real : bool, optional

            If True, the function will return the forces, force coefficients, moments, and moment coefficients of the real solver.
            If False, the function will return the forces, force coefficients, moments, and moment coefficients of the simulated solver. Default is True.

        Returns:
        dict

            A dictionary containing the following keys and values:
                - "forces": A (num_steps,3) array of forces in the geometry axes.
                - "forces_coeff": A (num_steps,3) array of force coefficients in the geometry axes.
                - "moments": A (num_steps,3) array of moments in the geometry axes, relative to the CG.
                - "moments_coeff": A (num_steps,3) array of moment coefficients in the geometry axes, relative to the CG.
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
        """
        Plots the forces, force coefficients, moments, and moment coefficients 
        of a RealSolver versus a simulatedSolver on 4 separate pages with 3 graphs per page.
        All figures open at the same time.
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

# Animation des ailes ------------------------------------------------------------------------------------------------------------------------------------------------
    def get_wing_data(self, real=True, wing_index=0):

        """
        Get the coordinates of the ailes at each time step.

        Parameters
        ----------
        real : bool, optional
            Whether to get the real or simulated data. Defaults to True.
        wing_index : int, optional
            The index of the wing from which to get the coordinates. Defaults to 0.

        Returns
        -------
        coords : dict
            A dictionary where the keys are the time steps and the values are the coordinates of the ailes at each time step.
        """
        movement = (
            self.movement
            if real
            else self.simulated_solver.unsteady_problem.movement
        )

        coords={}

        for step in range(movement.num_steps):
            coords_step=[]
            airplane = movement.airplanes[0][step]
            wing = airplane.wings[wing_index]
            panels = np.ravel(wing.panels)
            for panel_id, panel in enumerate(panels):
                for vertex_name in ["Flpp_GP1_CgP1", "Frpp_GP1_CgP1",
                                    "Brpp_GP1_CgP1", "Blpp_GP1_CgP1"]:
                    vertex = getattr(panel, vertex_name)
                    coords_step.append(vertex)
           
            coords[step] = np.array(coords_step)

        return coords


    def max_edge_len(self, triangle, points):
        """
        Compute the length of the longest edge of a triangle.

        Parameters
        ----------
        triangle : array of 3 ints
            The indices of the triangle's vertices in the points array.
        points : array of shape (n,3)
            The array of points.

        Returns
        -------
        float
            The length of the longest edge of the triangle.

        """

        verts = points[triangle]
        edges = [np.linalg.norm(verts[i] - verts[j]) for i in range(3) for j in range(i + 1, 3)]
        return max(edges)


    def plot_wing(self, points):
        """
        Compute the triangles of a bisector mesh, filtered by the length of its edges.

        Parameters
        ----------
        points : array of shape (n,3)
            The array of points.

        Returns
        -------
        array of shape (m,3)
            The filtered triangles.

        Notes
        -----
        The filtering is done by the 60th percentile of the distances between points, to
        avoid having too long edges in the bisector mesh.
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
        """
        Plot the bisector mesh of the real and simulated wings at every time step.

        Parameters
        ----------
        wing_index : int, optional
            The index of the wing for which to generate the bisector mesh.

        Returns
        -------
        matplotlib.animation.FuncAnimation
            The animation of the bisector mesh.

        Notes
        -----
        The bisector mesh of the real wing is plotted in cyan and blue, and the bisector mesh of the simulated wing is plotted in orange and red.
        The simulated wing is mirrored in Y.
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

