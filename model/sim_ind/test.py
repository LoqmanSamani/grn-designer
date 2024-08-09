from simulation import *
import numpy as np
import time
import os
import h5py
# from heatmap import *
# from multi_heatmap import *
import matplotlib.pyplot as plt


"""
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

            result = individual_simulation(pop)
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

            result = individual_simulation(pop)
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
    result = individual_simulation(pop)

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
result, s1, s2, s3 = individual_simulation(pop)

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

"""







# Data Lists
com_size_x = [10, 50, 100, 200, 500, 1000]
com_size_y = [0.742, 3.519, 7.374, 18.195, 64.692, 231.442]
com_size_x1 = [10, 50, 100, 200, 500, 1000]
com_size_y1 = [0.939, 4.635, 10.665, 24.520, 78.512, 237.527]
numba_com_size_x = [10, 50, 100, 200, 500, 1000]
numba_com_size_y = [10.832, 0.116, 0.362, 2.039, 16.385, 107.511]
numba1_com_size_x = [10, 50, 100, 200, 500, 1000]
numba1_com_size_y = [26.764, 0.353, 1.379, 5.353, 35.197, 159.298]


sim_epochs_x = [100, 500, 1000, 10000, 50000, 100000]
sim_epochs_y = [0.686, 3.469, 6.919, 69.684, 351.579, 711.421]
sim_epochs_x1 = [100, 500, 1000, 10000, 50000, 100000]
sim_epochs_y1 = [0.929, 5.341, 10.591, 107.302, 532.303, 1083.512]
numba_sim_epochs_x = [100, 500, 1000, 10000, 50000, 100000]
numba_sim_epochs_y = [0.025, 0.116, 0.231, 2.310, 11.656, 23.539]
numba1_sim_epochs_x = [100, 500, 1000, 10000, 50000, 100000]
numba1_sim_epochs_y = [0.114, 0.403, 0.678, 7.273, 37.852, 75.112]


pop_size_x = [20, 50, 100, 200, 500]
pop_size_y = [150.991, 378.386, 752.464, 1492.955, 3762.638]
pop_size_x1 = [20, 50, 100, 200, 500]
pop_size_y1 = [29.117, 58.717, 144.363, 432.243, 1658.786]
numba_pop_size_x = [20, 50, 100, 200, 500]
numba_pop_size_y = [18.736, 18.984, 38.325, 76.573, 191.065]
numba1_pop_size_x = [20, 50, 100, 200, 500]
numba1_pop_size_y = [35.294, 32.226, 79.221, 222.690, 1065.238]


plt.figure(figsize=(30, 6))
plt.subplot(1, 3, 1)
plt.plot(com_size_x, com_size_y, label='Without Numba (individual_simulation())', marker='o')
plt.plot(com_size_x1, com_size_y1, label='Without Numba (population_simulation())', marker='o')
plt.plot(numba_com_size_x, numba_com_size_y, label='With Numba (individual_simulation())', marker='o')
plt.plot(numba1_com_size_x, numba1_com_size_y, label='With Numba (population_simulation())', marker='o')

plt.title('Performance vs. Compartment Size')
plt.xlabel("Compartment Size")
plt.ylabel("Time[s]")
plt.legend()
plt.grid(True)


plt.subplot(1, 3, 2)
plt.plot(sim_epochs_x, sim_epochs_y, label='Without Numba (individual_simulation())', marker='o')
plt.plot(sim_epochs_x1, sim_epochs_y1, label='Without Numba (population_simulation())', marker='o')
plt.plot(numba_sim_epochs_x, numba_sim_epochs_y, label='With Numba (individual_simulation())', marker='o')
plt.plot(numba1_sim_epochs_x, numba1_sim_epochs_y, label='With Numba (population_simulation())', marker='o')
plt.title('Performance vs. Simulation Epochs')
plt.xlabel("Number of Simulation Epochs")
plt.ylabel("Time[s]")
plt.legend()
plt.grid(True)


plt.subplot(1, 3, 3)
plt.plot(pop_size_x, pop_size_y, label='Without Numba (individual_simulation())', marker='o')
plt.plot(pop_size_x1, pop_size_y1, label='Without Numba (population_simulation())', marker='o')
plt.plot(numba_pop_size_x, numba_pop_size_y, label='With Numba (individual_simulation())', marker='o')
plt.plot(numba1_pop_size_x, numba1_pop_size_y, label='With Numba (population_simulation())', marker='o')
plt.title('Performance vs. Population Size')
plt.xlabel("Population Size")
plt.ylabel("Time[s]")
plt.legend()
plt.grid(True)

plt.show()






