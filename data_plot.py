from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import griddata

def plot_amplitude_test_data(file):
    data = pd.read_csv(file)

    freq = data['freq']
    amp = data['amplitude']
    lift = data['lift']
    drag = data['drag']
    power = data['power']

    grid_x, grid_y = np.mgrid[freq.min():freq.max():100j,
                              amp.min():amp.max():100j]

    lift_grid = griddata((freq, amp), lift, (grid_x, grid_y), method='cubic')
    drag_grid = griddata((freq, amp), drag, (grid_x, grid_y), method='cubic')
    power_grid = griddata((freq, amp), power, (grid_x, grid_y), method='cubic')

    fig = plt.figure(figsize=(16, 5))

    ax = fig.add_subplot(1, 3, 1, projection='3d')
    surf = ax.plot_surface(grid_x, grid_y, lift_grid, cmap=cm.coolwarm)
    ax.set_title('Lift')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude (deg)')
    ax.set_zlabel('Lift (N)')
    fig.colorbar(surf, ax=ax, shrink=0.5)

    ax = fig.add_subplot(1, 3, 2, projection='3d')
    surf = ax.plot_surface(grid_x, grid_y, drag_grid, cmap=cm.coolwarm)
    ax.set_title('Drag')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude (deg)')
    ax.set_zlabel('Drag (N)')
    fig.colorbar(surf, ax=ax, shrink=0.5)

    ax = fig.add_subplot(1, 3, 3, projection='3d')
    surf = ax.plot_surface(grid_x, grid_y, power_grid, cmap=cm.coolwarm)
    ax.set_title('Power')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude (deg)')
    ax.set_zlabel('Power (W)')
    fig.colorbar(surf, ax=ax, shrink=0.5)

    plt.tight_layout()
    plt.show()
