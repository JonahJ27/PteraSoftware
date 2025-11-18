"""This module contains the functions for creating airfoils.

This module contains the following classes:
    Real_Airfoil: A class to handle airfoil shape creation and analysis from OptiTrack data.

This module contains the following functions:
    
    load_data: Load and preprocess the OptiTrack data.

    extract_column: Extract the airfoil columns.
        
    get_airfoil_shape: Create the airfoil shape from OptiTrack data.
    
    get_chord_length: Get the length of the chord of the airfoil.
    
    get_chord_unit: Get the chord unit vector for this airfoil section.
    
    get_Lp: Get the coordinates Lp, Wcsp, Lpp of the airfoil.
    
    parent_axes: Give the relative position and orientation [angleX, angleY, angleZ] in degrees,

    get_position: Give the relative position and orientation [angleX, angleY, angleZ] in degrees,
    
"""

import numpy as np
import pandas as pd


def load_data(optitrack_file, list_trackers, right=True):
    

    """
    Load and preprocess the OptiTrack data.

    Parameters
    ----------
    optitrack_file : str
        The path to the OptiTrack CSV file.

    list_trackers : list of str
        The list of trackers in the OptiTrack file.

    right : bool, optional
        Whether to define the positive x-axis direction as right or left. The default is True.

    Returns
    -------
    data : numpy.ndarray of shape (n_frames, n_trackers, 3)
        The preprocessed OptiTrack data.

    Notes
    -----
    The data is preprocessed by subtracting the mean of the A1 and Af trackers from the data, and then by applying a rotation matrix to align the positive x-axis direction to the right or left. The data is then shifted so that the mean of the A1 tracker is at (0, 0, 0) and the mean of the Af tracker is at (d, 0, 0), where d is the distance between the mean of the A1 and Af trackers. Finally, the data is shifted so that the Z1 tracker is at (0, 0, 0) and the Z2 tracker is at (d, 0, 0).

    The list of trackers should include all the real trackers (A1, A2, ..., An, B1, B2, ..., Bn, ..., 1, 2, ..., n). 

    The right parameter is used to define the positive x-axis direction. If right is True, the positive x-axis direction is defined as right. If right is False, the positive x-axis direction is defined as left. The default is True.
   
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
    """
    Extract the columns from the list of trackers.

    Parameters
    ----------
    list_trackers : list
        A list of the trackers used in the airfoil.

    Returns
    -------
    list
        A list of the columns used in the airfoil.
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
        """
        Initialize the Airfoil object.

        :param data: array
            Data of Optitrack.
        :param step: int
            Time step index to extract the airfoil shape.
        :param column: str
            Column index in the OptiTrack data file that contains the airfoil shape data.
        :param list_trackers: list of str
            List of tracker names to extract the airfoil shape.
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

        """
        Get the airfoil shape coordinates in the normalized frame.

        Returns
        -------
        numpy.ndarray of shape (n_points, 2)
            The coordinates of the airfoil shape in the normalized frame.
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
        """Get the length of the chord of the airfoil."""

        return self.chord_length

    def get_chord_unit(self):
        """ Get the chord unit vector for this airfoil section. """

        return self.chord_unit

    def get_Lp(self):
        """Get the coordinates Lp, Wcsp, Lpp of the airfoil."""

        idx = self.list_trackers.index(self.column + "1")
        return self.data[self.step, idx]

    def get_relative_transform(self):
        """
        Give the relative position and orientation [angleX, angleY, angleZ] in degrees,
        following the x-y'-z" convention, with values in the range [-90, 90].
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