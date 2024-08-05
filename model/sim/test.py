from simulation import *
import numpy as np
import time
import os
import h5py
# from heatmap import *
# from multi_heatmap import *
import matplotlib.pyplot as plt



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







# Benchmarking data
benchmarking_data = {
    "numba_com_size": {
        10: 10.832071781158447,
        50: 0.11558675765991211,
        100: 0.36243414878845215,
        200: 2.0385539531707764,
        500: 16.384703159332275,
        1000: 107.51107883453369,
    },
    "numba_sim_epochs": {
        100: 0.0251920223236084,
        500: 0.11624932289123535,
        1000: 0.2307727336883545,
        10000: 2.3095428943634033,
        50000: 11.656088590621948,
        100000: 23.539443016052246,
    },
    "numba_pop_size": {
        20: 18.735677242279053,
        50: 18.983787536621094,
        100: 38.32541823387146,
        200: 76.57288527488708,
        500: 191.06534576416016,
    },
    "com_size": {
        10: 0.7420475482940674,
        50: 3.518643617630005,
        100: 7.3738672733306885,
        200: 18.19509768486023,
        500: 64.69217228889465,
        1000: 231.44163966178894,
    },
    "sim_epochs": {
        100: 0.6864268779754639,
        500: 3.468745470046997,
        1000: 6.919107675552368,
        10000: 69.68381881713867,
        50000: 351.5794668197632,
        100000: 711.421303987503,
    },
    "pop_size": {
        20: 150.99141120910645,
        50: 378.38582491874695,
        100: 752.4636449813843,
        200: 1492.9554994106293,
        500: 3762.6383543014526
    }
}

# Extract data for plotting
x_com_size = sorted(benchmarking_data['com_size'].values())
y_com_size = [key for key in sorted(benchmarking_data['com_size'].keys())]

x_sim_epochs = sorted(benchmarking_data['sim_epochs'].values())
y_sim_epochs = [key for key in sorted(benchmarking_data['sim_epochs'].keys())]

x_pop_size = sorted(benchmarking_data['pop_size'].values())
y_pop_size = [key for key in sorted(benchmarking_data['pop_size'].keys())]

x_numba_com_size = sorted(benchmarking_data['numba_com_size'].values())
y_numba_com_size = [key for key in sorted(benchmarking_data['numba_com_size'].keys())]

x_numba_sim_epochs = sorted(benchmarking_data['numba_sim_epochs'].values())
y_numba_sim_epochs = [key for key in sorted(benchmarking_data['numba_sim_epochs'].keys())]

x_numba_pop_size = sorted(benchmarking_data['numba_pop_size'].values())
y_numba_pop_size = [key for key in sorted(benchmarking_data['numba_pop_size'].keys())]

# Create subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# Plot Com Size
axs[0].plot(y_com_size, x_com_size, label='Without Numba', marker='o', color='b')
axs[0].plot(y_numba_com_size, x_numba_com_size, label='With Numba', marker='o', color='r')
axs[0].set_title('Performance vs. Compartment Size')
axs[0].set_xlabel('Compartment Size')
axs[0].set_ylabel('Time (seconds)')
axs[0].legend()
axs[0].grid(True)

# Plot Sim Epochs
axs[1].plot(y_sim_epochs, x_sim_epochs, label='Without Numba', marker='o', color='b')
axs[1].plot(y_numba_sim_epochs, x_numba_sim_epochs, label='With Numba', marker='o', color='r')
axs[1].set_title('Performance vs. Maximum Number of Simulation Epoch')
axs[1].set_xlabel('Simulation Epochs')
axs[1].set_ylabel('Time (seconds)')
axs[1].legend()
axs[1].grid(True)

# Plot Pop Size
axs[2].plot(y_pop_size, x_pop_size, label='Without Numba', marker='o', color='b')
axs[2].plot(y_numba_pop_size, x_numba_pop_size, label='With Numba', marker='o', color='r')
axs[2].set_title('Performance vs. Population Size')
axs[2].set_xlabel('Population Size')
axs[2].set_ylabel('Time (seconds)')
axs[2].legend()
axs[2].grid(True)

# Show the plot
plt.tight_layout()
plt.show()
