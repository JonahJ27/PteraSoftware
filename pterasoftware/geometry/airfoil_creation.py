"""Contains functions for creating airfoils.

**Contains the following classes:**

Real_Airfoil: A class to handle airfoil shape creation and analysis from OptiTrack data.

**Contains the following functions:**

load_data: Load and preprocess the OptiTrack data from an OptiTrack CSV file.

extract_columns: Extract the column identifiers corresponding to the airfoil sections.

Real_Airfoil.get_airfoil_shape: Create the normalized airfoil shape from OptiTrack tracker points.

Real_Airfoil.get_chord_length: Get the chord length of the current airfoil section.

Real_Airfoil.get_chord_unit: Get the unit vector along the chord direction.

Real_Airfoil.get_relative_transform: Get the relative position and orientation [angleX, angleY, angleZ] (in degrees) of this airfoil section relative to its parent or neighbor section.
"""

import numpy as np
import pandas as pd


def load_data(optitrack_file, list_trackers, right=True):

    """Loads and preprocesses OptiTrack 3D marker data from a CSV file. It returns a
    (n_frames, n_trackers, 3) array containing the cleaned and reoriented tracker
    coordinates, expressed in a consistent right-handed reference frame aligned with
    the airfoil longitudinal axis.

    The function performs several steps: (1) reading and converting OptiTrack CSV
    numeric columns, (2) reshaping the raw data into a tracker-by-frame structure,
    (3) computing a local reference frame based on the A1 and Af markers to define the
    chord direction, (4) constructing an orthonormal basis (X,Y,Z) depending on the
    `right` flag, (5) rotating all coordinates into this frame, and (6) recentering the
    system so that marker A1 corresponds to the origin. The output is a consistent,
    time-resolved dataset suitable for airfoil reconstruction.

    :param optitrack_file: The path to the OptiTrack CSV file. Only numerical
        marker columns are used; headers and units rows are ignored.
    :param list_trackers: A list of tracker names describing the order of
        markers in the file (e.g., ["A1","A2","B1",...]). Must match the column
        structure of the CSV file.
    :param right: Boolean flag indicating if the wing is the right wing (True) or
        the left wing (False). 
    :return: A (n_frames, n_trackers, 3) ndarray of floats containing the processed
        tracker coordinates expressed in the normalized right-handed reference frame.

    Notes
    -----
    - The A1 → Af vector defines the local X axis (chord direction).
    - The Y axis is constructed using the relative positions of the last tracked
      section markers and optionally flipped depending on `right`.
    - The Z axis is computed as the cross product ensuring orthonormality.
    - All coordinates are finally shifted so that A1 lies at the origin for all frames.
    """
    
    data = pd.read_csv(optitrack_file, skiprows=8).iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
    arr = data.to_numpy(dtype=np.float64)
    n_real_points = arr.shape[1] // 3
    arr = arr.reshape((arr.shape[0], n_real_points, 3))

    real_trackers = [t for t in list_trackers]
    index_map = {t: i for i, t in enumerate(real_trackers)}
    
    last_A = [t for t in real_trackers if "A" in t][-1]
    idx_A1, idx_Af = index_map["A1"], index_map[last_A]
    mean_A1, mean_Af = arr[:, idx_A1].mean(0), arr[:, idx_Af].mean(0)

    X = mean_Af - mean_A1
    X /= np.linalg.norm(X)

    last_t1 = [t for t in real_trackers if "1" in t][-1]
    mean_last = arr[:, index_map[last_t1]].mean(0)


    if right == True :
        Y =  - mean_A1  + mean_last
    else :
        Y =  - mean_last + mean_A1
    Z = np.cross(X, Y); Z /= np.linalg.norm(Z)
    Y = np.cross(Z, X); Y /= np.linalg.norm(Y)

    R = np.linalg.inv(np.column_stack((X, Y, Z)))

    data_real = (R @ arr.transpose(0, 2, 1)).transpose(0, 2, 1)
    if right == False :
        data_real[:, :, 1] = -data_real[:, :, 1]
    

    mean_A1, mean_Af = data_real[:, idx_A1].mean(0), data_real[:, idx_Af].mean(0)

    data_real -= np.array([mean_A1[0], mean_A1[1] , mean_A1[2]])  

    data = np.zeros((arr.shape[0], len(list_trackers), 3))
    for t, idx_r in index_map.items():
        data[:, list_trackers.index(t)] = data_real[:, idx_r]

    return data

def extract_columns(list_trackers):
    """Extracts the distinct airfoil section identifiers (A, B, C, …) from the tracker
    names and returns them in the order they appear.

    :param list_trackers: List of tracker names used in the experiment.

    :return: A list of unique section letters.
    """
    columns = []
    for tracker in list_trackers:
        letter = tracker[0]  
        if letter not in columns:
            columns.append(letter)
    return columns

class Real_Airfoil:
    """A class to handle airfoil shape creation and analysis from OptiTrack data."""

    def __init__(self, data, step, column, list_trackers):
        """Initializes an airfoil section by gathering all tracker points belonging to the
        specified column and computing its chord geometry at the selected time step.

        :param data: Processed OptiTrack array shaped (n_frames, n_trackers, 3).
        :param step: Time index from which to extract the airfoil geometry.
        :param column: Section identifier selecting which group of trackers to use.
        :param list_trackers: List of all tracker names in the dataset.

        :return: None.
        """
        self.data = data
        self.step = step
        self.column = column
        self.list_trackers = list_trackers

        self.points = np.array([data[step, i]
                                for i, t in enumerate(list_trackers)
                                if column in t])

        if len(self.points) < 2:
            raise ValueError(f"No point found for column '{column}'.")

        self.chord = self.points[-1] - self.points[0]
        self.chord_length = np.linalg.norm(self.chord)
        self.chord_unit = self.chord / self.chord_length

    def get_airfoil_shape(self):

        """Generates the 2D normalized airfoil shape for this section using the measured
        tracker coordinates, producing a closed profile scaled by chord length.

        :return: A (n_points, 2) ndarray containing the normalized airfoil contour.
        """


        trackers = self.points / self.chord_length
        origin = trackers[0]
        profil = [[np.dot(traker - origin, self.chord_unit),
                 np.linalg.norm((traker - origin) -
                                np.dot(traker - origin, self.chord_unit)*self.chord_unit)]
                for traker in trackers[::-1]]

        if len(profil) < 3:
            m = (np.array(profil[0]) + np.array(profil[1])) / 2
            profil.insert(1, m.tolist())

        prof = np.array(profil)
        profil_sym = np.c_[prof[::-1][1:-1, 0], prof[::-1][1:-1, 1]-0.0015]
        return np.vstack((prof, profil_sym, [prof[0, 0], prof[0, 1]-1e-5]))

    def get_chord_length(self):
        """Returns the chord length of the current airfoil section.

        :return: A float representing the chord length.
        """
        return self.chord_length

    def get_chord_unit(self):
        """Returns the unit vector pointing along the chord direction of this airfoil
        section.

        :return: A (3,) ndarray containing the chord unit vector.
        """
        return self.chord_unit

    def get_Lp(self):
        """Returns the 3D coordinates of the reference leading-point tracker for this
        section.

        :return: A (3,) ndarray giving the leading point position.
        """
        idx = self.list_trackers.index(self.column + "1")
        return self.data[self.step, idx]

    def get_relative_transform(self):
        """Computes the relative translation and orientation of this airfoil section with
        respect to its adjacent section in the wing structure.

        :return: A tuple (translation, angles) containing the relative position and the
        Euler rotation angles in degrees.
        """
        sections = ["A", "B", "C", "D", "E", "F", "G",
            "H", "I", "J", "K", "L", "M", "N",
            "O", "P", "Q", "R", "S", "T", "U",
            "V", "W", "X", "Y", "Z"]        
        idx = sections.index(self.column)


        if idx == 0 :  
            next_col = sections[idx + 1]

            next_cross_section = Real_Airfoil(self.data, self.step,
                                next_col, self.list_trackers)

            Xp = (1,0,0)
            Yp = (0,1,0)
            Zp = (0,0,1)
            Rp = np.column_stack((Xp, Yp, Zp))

            Xe = self.get_chord_unit()
            Ye = next_cross_section.get_Lp() - self.get_Lp()
            Ze = np.cross(Xe, Ye); Ze /= np.linalg.norm(Ze)
            Ye = np.cross(Ze, Xe); Ye /= np.linalg.norm(Ye)
            Re = np.column_stack((Xe, Ye, Ze))

            R = Rp.T @ Re

            angY = np.degrees(np.arcsin(-R[2, 0]))
            angX = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
            angZ = np.degrees(np.arctan2(R[1, 0], R[0, 0]))

            angles_Wcsp_to_Wcs_ixyz = np.array([angX, angY, angZ])

            Lp_Wcsp_Lpp = self.get_Lp() 
        
        
        elif idx == len(extract_columns(self.list_trackers)) - 1:
            parent_col = sections[idx - 1]

            parent = Real_Airfoil(self.data, self.step,
                                parent_col, self.list_trackers)

            Xp = parent.get_chord_unit()
            Yp = self.get_Lp() - parent.get_Lp()
            Zp = np.cross(Xp, Yp); Zp /= np.linalg.norm(Zp)
            Yp = np.cross(Zp, Xp); Yp /= np.linalg.norm(Yp)
            Rp = np.column_stack((Xp, Yp, Zp))

            Xe = self.get_chord_unit()
            Ye = self.get_Lp() - parent.get_Lp()
            Ze = np.cross(Xe, Ye); Ze /= np.linalg.norm(Ze)
            Ye = np.cross(Ze, Xe); Ye /= np.linalg.norm(Ye)
            Re = np.column_stack((Xe, Ye, Ze))

            R = Rp.T @ Re

            angY = np.degrees(np.arcsin(-R[2, 0]))
            angX = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
            angZ = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
            angles_Wcsp_to_Wcs_ixyz = np.array([angX, angY, angZ])

            Lp_Wcsp_Lpp = Rp.T @ (self.get_Lp() - parent.get_Lp())

        else :
            parent_col = sections[idx - 1]

            parent = Real_Airfoil(self.data, self.step,
                                parent_col, self.list_trackers)

            next_col = sections[idx + 1]

            next_cross_section = Real_Airfoil(self.data, self.step,
                                next_col, self.list_trackers)

            Xp = parent.get_chord_unit()
            Yp = self.get_Lp() - parent.get_Lp()
            Zp = np.cross(Xp, Yp); Zp /= np.linalg.norm(Zp)
            Yp = np.cross(Zp, Xp); Yp /= np.linalg.norm(Yp)
            Rp = np.column_stack((Xp, Yp, Zp))

            Xe = self.get_chord_unit()
            Ye = next_cross_section.get_Lp() - self.get_Lp() 
            Ze = np.cross(Xe, Ye); Ze /= np.linalg.norm(Ze)
            Ye = np.cross(Ze, Xe); Ye /= np.linalg.norm(Ye)
            Re = np.column_stack((Xe, Ye, Ze))

            R = Rp.T @ Re

            angY = np.degrees(np.arcsin(-R[2, 0]))
            angX = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
            angZ = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
            angles_Wcsp_to_Wcs_ixyz = np.array([angX, angY, angZ])

            Lp_Wcsp_Lpp = Rp.T @ (self.get_Lp() - parent.get_Lp())

        return Lp_Wcsp_Lpp, angles_Wcsp_to_Wcs_ixyz