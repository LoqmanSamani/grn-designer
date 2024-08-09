from simulation import *
import os
import h5py
import time
from multi_heatmap import *


"""
tic = time.time()
pop = np.zeros((2, 7, 30, 30))
pop[:, 1, :, 0] = 1
pop[:, 3, :, -1] = 1

pop[:, -1, 0, :3] = [.09, .007, 1.1]
pop[:, -1, 2, :3] = [0.09, 0.006, 1.2]
pop[:, -1, -1, :5] = [2, 1, 500, 5, .01]
pop[:, -2, 0, 0:2] = [0, 2]
pop[:, -2, 1, 0:4] = [6, .01, 0.001, 1.3]

result, sp1, sp2, com = population_simulation(pop)


full_path = "/home/samani/Documents/sim/g"

if not os.path.exists(full_path):
    os.makedirs(full_path)

full_file_path = os.path.join(full_path, "result.h5")

with h5py.File(full_file_path, "w") as file:
    file.create_dataset("sp1", data=sp1)
    file.create_dataset("sp2", data=sp2)
    file.create_dataset("com", data=com)



keys = ["sp1", "com", "sp2"]

model = HeatMaps(
    data_path="/home/samani/Documents/sim/g/result.h5",
    video_directory="/home/samani/Documents/sim",
    video_name="heatmaps",
    title="Heat Maps",
    x_label="Number of Cells",
    y_label="Number of Cells",
    z_labels=["Free GFP", 'Inhibitor-GFP', "Free Inhibitor"],
    subplots=(1, 3),
    cmaps=["GreenBlack", "BlueGreenBlack", "BlueBlack"],
    title_size=14,
    label_size=12,
    fps=20,
    interval=50,
    writer='ffmpeg',
    fig_size=(30, 10),
    colorbar=True,
    grid=None,
    subplot_size=(8, 8),
    plot_margins=(0.2, 0.1),
    hide_axis=False,
    colorbar_axis=True,
    background_color='white',
    title_color='black',
    xlabel_color='black',
    ylabel_color='black'
)


model.heatmap_animation(keys)




pop_size = [20, 50, 100, 200, 500]

t = []

for p in pop_size:
    tic = time.time()
    pop = np.zeros((p, 7, 30, 30))
    pop[:, 1, :, 0] = 1
    pop[:, 3, :, -1] = 1

    pop[:, -1, 0, :3] = [.09, .007, 1.1]
    pop[:, -1, 2, :3] = [0.09, 0.006, 1.2]
    pop[:, -1, -1, :5] = [2, 1, 500, 5, .01]
    pop[:, -2, 0, 0:2] = [0, 2]
    pop[:, -2, 1, 0:4] = [6, .01, 0.001, 1.3]

    result = population_simulation(pop)
    toc = time.time()
    t.append(toc-tic)

print(t)
"""

tic = time.time()
pop = np.zeros((500, 7, 30, 30))
pop[:, 1, :, 0] = 1
pop[:, 3, :, -1] = 1

pop[:, -1, 0, :3] = [.09, .007, 1.1]
pop[:, -1, 2, :3] = [0.09, 0.006, 1.2]
pop[:, -1, -1, :5] = [2, 1, 500, 5, .01]
pop[:, -2, 0, 0:2] = [0, 2]
pop[:, -2, 1, 0:4] = [6, .01, 0.001, 1.3]

result = population_simulation(pop)
toc = time.time()

print(toc - tic)
