import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import h5py


def update(frame, data, sc):
    #fig = plt.figure(figsize=(10, 8))
    #ax = fig.add_subplot(111, projection='3d')
    #ax.clear()
    #ax.set_xlabel('X')
    #ax.set_ylabel('Y')
    #ax.set_zlabel('Concentrate')
    #ax.set_title(f'Frame: {frame}')

    # Set the x and y limits to be from 1 to 100
    #ax.set_xlim(1, 100)
    #ax.set_ylim(1, 100)

    x, y, z = data.shape

    x_coords = []
    y_coords = []
    z_coords = []

    for i in range(x):
        for j in range(y):
            x_coords.append(i + 1)  # Row index
            y_coords.append(j + 1)  # Column index
            z_coords.append(data[i, j, frame])  # Value at (i, j, frame)

    # Update the scatter plot for this frame
    sc._offsets3d = (x_coords, y_coords, z_coords)

    return sc


def scatter_3D_animation(data):
    fig = plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
    ax = fig.add_subplot(111, projection='3d')

    # Create an empty scatter plot
    sc = ax.scatter([], [], [], c='r', marker='o')

    ani = FuncAnimation(fig, update, frames=data.shape[2], fargs=(data, sc), blit=True)

    # Save animation as an mp4 file with higher resolution
    ani.save('/home/samani/Documents/sim/animation.mp4', writer='ffmpeg', dpi=200)

    plt.show()


# Example usage:
# scatter_3D_animation(np.random.rand(100, 100, 100))


sim_result = "/home/samani/Documents/sim/sims.h5"

with h5py.File(sim_result, 'r') as hdf_file:

    #expected_iM_data = hdf_file["expected_iM"]
    #expected_fM_data = hdf_file["expected_fM"]
    expected_ibM_data = hdf_file["expected_ibM"]
    #expected_bM_data = hdf_file["expected_bM"]
    #expected_cells_anker_all = hdf_file["expected_cells_anker_all"]
    #print(expected_bM_data[:])

    # Plot animations for different datasets
    #plot_animation(expected_iM_data, "iM")
    #plot_animation(expected_fM_data, "fM")
    scatter_3D_animation(expected_ibM_data)
    #plot_animation(expected_bM_data, "bM")
    #plot_animation(expected_cells_anker_all, "anchor cells")
    # plot_animation(im, "im")




