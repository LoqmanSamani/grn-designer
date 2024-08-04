from simulation import *
import numpy as np
import time
import os
import h5py
from heatmap import *
from multi_heatmap import *



def run_simulation_with_timing():
    try:
        com_size = [10, 50, 100, 200, 500, 1000]
        com_time = []
        for c in com_size:
            tic = time.time()
            pop = np.zeros((7, c, c))
            pop[1, :, 0] = 10
            pop[3, :, -1] = 10

            pop[-1, 0, :3] = [.09, .007, 1.1]
            pop[-1, 2, :3] = [0.09, 0.006, 1.2]
            pop[-1, -1, :5] = [2, 1, 500, 5, .01]
            pop[-2, 0, 0:2] = [0, 2]
            pop[-2, 1, 0:4] = [6, .01, 0.001, 1.3]

            result = population_simulation(pop)
            toc = time.time()
            d = toc - tic
            com_time.append(d)

        max_s = [100, 500, 1000, 10000, 50000, 100000]
        dts = [.1, .02, .01, .001, 0.0002, 0.0001]
        sim_time = []
        for i in range(len(max_s)):
            tic = time.time()
            pop = np.zeros((7, 50, 50))
            pop[1, :, 0] = 10
            pop[3, :, -1] = 10

            pop[-1, 0, :3] = [.09, .007, 1.1]
            pop[-1, 2, :3] = [0.09, 0.006, 1.2]
            pop[-1, -1, :5] = [2, 1, max_s[i], 10, dts[i]]
            pop[-2, 0, 0:2] = [0, 2]
            pop[-2, 1, 0:4] = [6, .01, 0.001, 1.3]

            result = population_simulation(pop)
            toc = time.time()
            d = toc - tic
            sim_time.append(d)

        print("Compartment size times: ", com_time)
        print("Simulation time for different epochs and time steps: ", sim_time)

    except Exception as e:
        print(f"An error occurred: {e}")


run_simulation_with_timing()




tic = time.time()

for i in range(500):
    pop = np.zeros((7, 30, 30))
    pop[1, :, 0] = 10
    pop[3, :, -1] = 10

    pop[-1, 0, :3] = [.09, .007, 1.1]
    pop[-1, 2, :3] = [0.09, 0.006, 1.2]
    pop[-1, -1, :5] = [2, 1, 500, 5, .01]
    pop[-2, 0, 0:2] = [0, 2]
    pop[-2, 1, 0:4] = [6, .01, 0.001, 1.3]
    result = population_simulation(pop)

toc = time.time()
d = toc - tic
print(d)






pop = np.zeros((7, 20, 20))
pop[1, :, 0] = 1000
pop[3, :, -1] = 1000

pop[-1, 0, :3] = [.09, .007, 1.1]
pop[-1, 2, :3] = [0.09, 0.006, 1.2]
pop[-1, -1, :5] = [2, 1, 1000, 5, .01]
pop[-2, 0, 0:2] = [0, 2]
pop[-2, 1, 0:4] = [6, .01, 0.001, 1.3]
result, s1, s2, s3 = population_simulation(pop)

full_path = "/home/samani/Documents/sim"

if not os.path.exists(full_path):
    os.makedirs(full_path)

full_file_path = os.path.join(full_path, "sim_new.h5")

with h5py.File(full_file_path, "w") as file:
    file.create_dataset("sp1", data=s1)
    file.create_dataset("sp2", data=s2)
    file.create_dataset("com", data=s3)



model1 = HeatMap(
    data_path="/home/samani/Documents/sim/sim_new.h5",
    video_directory="/home/samani/Documents/sim/",
    video_name="sp1",
    title="Sp1",
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

model2 = HeatMap(
    data_path="/home/samani/Documents/sim/sim_new.h5",
    video_directory="/home/samani/Documents/sim/",
    video_name="sp2",
    title="Sp2",
    x_label="Number of Cells",
    y_label="Number of Cells",
    c_map="BlueBlack",
    fps=10,
    interval=50,
    writer='ffmpeg',
    color_bar=True,
    norm=False
)

model2.heatmap_animation(key="sp2")

model3 = HeatMap(
    data_path="/home/samani/Documents/sim/sim_new.h5",
    video_directory="/home/samani/Documents/sim/",
    video_name="com",
    title="Complex",
    x_label="Number of Cells",
    y_label="Number of Cells",
    c_map="BlueGreenBlack",
    fps=10,
    interval=50,
    writer='ffmpeg',
    color_bar=True,
    norm=False
)

model3.heatmap_animation(key="com")


keys = ["sp1", "com", "sp2"]

model = HeatMaps(
    data_path="/home/samani/Documents/sim/sim_new.h5",
    video_directory="/home/samani/Documents/sim",
    video_name="heatmaps",
    title="Heat Maps",
    x_label="Number of Cells",
    y_label="Number of Cells",
    z_labels=["Sp1", 'Complex', "Sp2"],
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

