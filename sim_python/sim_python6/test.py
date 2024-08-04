from simulation import *
import os
import h5py
from heatmap import *


pop = np.zeros((2, 7, 10, 10))
pop[:, 1, :, 0] = 1
pop[:, 3, :, -1] = 1

pop[:, -1, 0, :3] = [.7, .2, 1.1]
pop[:, -1, 2, :3] = [.5, .1, .9]
pop[:, -1, -1, :5] = [2, 1, 50000, 20, .01]
pop[:, -2, 0, 0:2] = [1, 2]
pop[:, -2, 1, 0:4] = [.4, .1, .2, .7]

r, result = population_simulation(pop)

full_path = "/home/samani/Documents/sim/g"

if not os.path.exists(full_path):
    os.makedirs(full_path)

full_file_path = os.path.join(full_path, "result.h5")

with h5py.File(full_file_path, "w") as file:
    file.create_dataset("sp1", data=result[0])
    file.create_dataset("sp2", data=result[1])
    file.create_dataset("com", data=result[2])


model1 = HeatMap(
    data_path="/home/samani/Documents/sim/g/result.h5",
    video_directory="/home/samani/Documents/sim/g",
    video_name="this",
    title="Free GFP",
    x_label="Number of Cells",
    y_label="Number of Cells",
    c_map="GreenBlack",
    fps=10,
    interval=50,
    writer='ffmpeg',
    color_bar=True,
    norm=False
)

model1.heatmap_animation(key="sp1")
