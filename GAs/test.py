from genetic_algorithm import genetic_algorithm
import h5py
import numpy as np
from simulation import *
import os
import matplotlib.pyplot as plt



full_path = "/home/samani/Documents/sim"


sp1 = np.zeros((30, 30))
sp2 = np.zeros((30, 30))
sp1_cells = np.zeros((30, 30))
sp1_cells[:, 0] = 10
sp2_cells = np.zeros((30, 30))
sp2_cells[:, 12:14] = 5
params = np.array([[.4, .4, 0.5, 0.5, 0.01, 0.01]])
dt = 0.01
sim_start = 1
sim_stop = 20
epochs = 500
target_shape = (30, 30)
result = simulation(sp1, sp2, sp1_cells, sp2_cells, params, dt, sim_start, sim_stop, epochs, target_shape)
full_file_path = os.path.join(full_path, "sim.h5")



with h5py.File(full_file_path, "w") as file:
    file.create_dataset("sp2", data=result)


file = h5py.File("/home/samani/Documents/sim/sim.h5", "r")

print(file.keys())

sp2 = file["sp2"]


plt.figure(figsize=(10, 10))
plt.imshow(sp2, cmap="hot", interpolation="nearest")
plt.title("Target", fontsize=20)
plt.colorbar(shrink=0.9)
plt.axis("off")
plt.show()


precision_bits = {"sp1": (0, 10, 10), "sp2": (0, 10, 10), "sp1_cells": (0, 10, 10), "sp2_cells": (0, 10, 10), "params": (0, 10, 10)}



genetic_algorithm(
    population_size=100,
    specie_matrix_shape=(30, 30),
    precision_bits=precision_bits,
    num_params=6,
    max_generation=100,
    mutation_rates=[.01, .01, .01, .01, .01],
    crossover_rates=[.8, .8, .8, .8, .8],
    num_crossover_points=[1, 1, 1, 1, 1],
    target=sp2,
    target_precision_bits=(0, 10, 10),
    result_path="/home/samani/Documents/sim",
    selection_method="tournament",
    tournament_size=4,
    file_name="ga",
    dt=0.01,
    sim_start=1,
    sim_stop=20,
    epochs=500,
    fitness_trigger=False
)

"""
=====================================================================================
                             *** Genetic Algorithm ***                               
=====================================================================================
Generation 1; Best/Max Fitness: 7822/9000; Generation Duration: 9.718567848205566
Generation 2; Best/Max Fitness: 4820/9000; Generation Duration: 7.502077102661133
Generation 3; Best/Max Fitness: 4744/9000; Generation Duration: 7.482980489730835
Generation 4; Best/Max Fitness: 4771/9000; Generation Duration: 7.59636116027832
Generation 5; Best/Max Fitness: 4778/9000; Generation Duration: 7.478018760681152
Generation 6; Best/Max Fitness: 4995/9000; Generation Duration: 7.472658634185791
Generation 7; Best/Max Fitness: 5062/9000; Generation Duration: 7.475281476974487
Generation 8; Best/Max Fitness: 5096/9000; Generation Duration: 7.47855019569397
Generation 9; Best/Max Fitness: 7400/9000; Generation Duration: 7.460923433303833
Generation 10; Best/Max Fitness: 7374/9000; Generation Duration: 7.460189342498779
Generation 11; Best/Max Fitness: 8705/9000; Generation Duration: 7.4768431186676025
Generation 12; Best/Max Fitness: 8690/9000; Generation Duration: 8.009056806564331
Generation 13; Best/Max Fitness: 8721/9000; Generation Duration: 7.463405132293701
Generation 14; Best/Max Fitness: 8728/9000; Generation Duration: 7.457747936248779
Generation 15; Best/Max Fitness: 8234/9000; Generation Duration: 7.4603517055511475
Generation 16; Best/Max Fitness: 8728/9000; Generation Duration: 7.476712226867676
Generation 17; Best/Max Fitness: 7667/9000; Generation Duration: 7.478697776794434
Generation 18; Best/Max Fitness: 8728/9000; Generation Duration: 7.456556797027588
Generation 19; Best/Max Fitness: 8375/9000; Generation Duration: 7.4757184982299805
Generation 20; Best/Max Fitness: 8282/9000; Generation Duration: 7.470835447311401
Generation 21; Best/Max Fitness: 8728/9000; Generation Duration: 7.459960699081421
Generation 22; Best/Max Fitness: 8728/9000; Generation Duration: 7.45794677734375

...

Generation 87; Best/Max Fitness: 8728/9000; Generation Duration: 7.520690441131592
Generation 88; Best/Max Fitness: 8728/9000; Generation Duration: 7.5281476974487305
Generation 89; Best/Max Fitness: 8728/9000; Generation Duration: 7.529546022415161
Generation 90; Best/Max Fitness: 8728/9000; Generation Duration: 7.537861585617065
Generation 91; Best/Max Fitness: 8728/9000; Generation Duration: 7.546834230422974
Generation 92; Best/Max Fitness: 8728/9000; Generation Duration: 7.526763200759888
Generation 93; Best/Max Fitness: 8728/9000; Generation Duration: 7.5186378955841064
Generation 94; Best/Max Fitness: 8728/9000; Generation Duration: 7.527470350265503
Generation 95; Best/Max Fitness: 8728/9000; Generation Duration: 7.5226569175720215
Generation 96; Best/Max Fitness: 8728/9000; Generation Duration: 7.543120861053467
Generation 97; Best/Max Fitness: 8728/9000; Generation Duration: 7.524866580963135
Generation 98; Best/Max Fitness: 8728/9000; Generation Duration: 7.52039647102356
Generation 99; Best/Max Fitness: 8728/9000; Generation Duration: 7.5197718143463135
Generation 100; Best/Max Fitness: 8728/9000; Generation Duration: 7.525475025177002
                   -----------------------------------------------
                     Simulation Complete!
                     The best found fitness: 8729
                     Total Generations: 100
                     Average Fitness: 8391.85
                     Total Simulation Duration: 778 seconds
                   -----------------------------------------------
"""






full_path = "/home/samani/Documents/sim"

sp1 = np.zeros((5, 5))
sp2 = np.zeros((5, 5))
sp1_cells = np.zeros((5, 5))
sp1_cells[:, 0] = 1
sp2_cells = np.zeros((5, 5))
sp2_cells[:, -1] = 1
params = np.array([[.4, .4, 0.5, 0.5, 0.01, 0.01]])
dt = 0.01
sim_start = 1
sim_stop = 20
epochs = 500
target_shape = (5, 5)
result = simulation(sp1, sp2, sp1_cells, sp2_cells, params, dt, sim_start, sim_stop, epochs, target_shape)
full_file_path = os.path.join(full_path, "sim.h5")

with h5py.File(full_file_path, "w") as file:
    file.create_dataset("sp2", data=result)


precision_bits = {"sp1": (0, 20, 8), "sp2": (0, 100, 8), "sp1_cells": (0, 20, 8), "sp2_cells": (0, 20, 8), "params": (0, 10, 8)}

file = h5py.File("/home/samani/Documents/sim/sim.h5", "r")

sp2 = file["sp2"][:]

genetic_algorithm(
    population_size=100,
    specie_matrix_shape=(5, 5),
    precision_bits=precision_bits,
    num_params=6,
    max_generation=10000,
    mutation_rates=[.007, .007, .007, .007, .004],
    crossover_rates=[.70, .7, .7, .7, .65],
    num_crossover_points=[1, 1, 1, 1, 1],
    target=sp2,
    target_precision_bits=(0, 10, 8),
    result_path="/home/samani/Documents/sim",
    selection_method="tournament",
    tournament_size=4,
    file_name="ga1",
    dt=0.01,
    sim_start=1,
    sim_stop=20,
    epochs=500,
    fitness_trigger=False
)

"""
=====================================================================================
                             *** Genetic Algorithm ***                               
=====================================================================================
Generation 1; Best/Max Fitness: 136/200; Generation Duration: 2.4251630306243896
Generation 2; Best/Max Fitness: 116/200; Generation Duration: 0.20687317848205566
Generation 3; Best/Max Fitness: 113/200; Generation Duration: 0.20706439018249512
Generation 4; Best/Max Fitness: 130/200; Generation Duration: 0.20663762092590332
Generation 5; Best/Max Fitness: 118/200; Generation Duration: 0.20707273483276367
Generation 6; Best/Max Fitness: 122/200; Generation Duration: 0.20714735984802246
Generation 7; Best/Max Fitness: 119/200; Generation Duration: 0.20675992965698242

...

Generation 9996; Best/Max Fitness: 186/200; Generation Duration: 0.21660423278808594
Generation 9997; Best/Max Fitness: 187/200; Generation Duration: 0.21477150917053223
Generation 9998; Best/Max Fitness: 186/200; Generation Duration: 0.20869922637939453
Generation 9999; Best/Max Fitness: 186/200; Generation Duration: 0.20888233184814453
Generation 10000; Best/Max Fitness: 186/200; Generation Duration: 0.21114373207092285
                   -----------------------------------------------
                     Simulation Complete!
                     The best found fitness: 188
                     Total Generations: 10000
                     Average Fitness: 181.33
                     Total Simulation Duration: 2101 seconds
                   -----------------------------------------------
"""








# presentation 09.07.2024
"""

from genetic_algorithm import genetic_algorithm
import h5py
import numpy as np
from simulation import *
import os
import matplotlib.pyplot as plt



full_path = "/home/samani/Documents/sim"


sp1 = np.zeros((5, 5))
sp2 = np.zeros((5, 5))
sp1_cells = np.zeros((5, 5))
sp1_cells[:, 0] = 5
sp2_cells = np.zeros((5, 5))
sp2_cells[:, 3] = 2
params = np.array([[3, 3, 2, 2, 0.1, 0.1]])
dt = 0.01
sim_start = 1
sim_stop = 20
epochs = 500
target_shape = (5, 5)
result = simulation(sp1, sp2, sp1_cells, sp2_cells, params, dt, sim_start, sim_stop, epochs, target_shape)
full_file_path = os.path.join(full_path, "sim.h5")

with h5py.File(full_file_path, "w") as file:
    file.create_dataset("sp2", data=result)

file = h5py.File("/home/samani/Documents/sim/sim.h5", "r")
sp2 = file["sp2"]


plt.figure(figsize=(10, 10))
plt.imshow(sp2, cmap="hot", interpolation="nearest")
plt.title("Target", fontsize=20)
plt.colorbar(shrink=0.9)
plt.axis("off")
plt.show()


file = h5py.File("/home/samani/Documents/sim/sim.h5", "r")
sp2 = file["sp2"][:]




precision_bits = {"sp1": (0, 200, 8), "sp2": (0, 200, 8), "sp1_cells": (0, 200, 8), "sp2_cells": (0, 200, 8), "params": (0, 200, 8)}



genetic_algorithm(
    population_size=200,
    specie_matrix_shape=(30, 30),
    precision_bits=precision_bits,
    num_params=6,
    max_generation=500,
    mutation_rates=[.02, .02, .02, .02, .01],
    crossover_rates=[.85, .85, .85, .85, .85],
    num_crossover_points=[2, 2, 2, 2, 1],
    target=sp2,
    target_precision_bits=(0, 200, 8),
    result_path="/home/samani/Documents/sim",
    selection_method="tournament",
    tournament_size=10,
    file_name="gar",
    dt=0.01,
    sim_start=1,
    sim_stop=20,
    epochs=500,
    fitness_trigger=False
)



file = h5py.File("/home/samani/Documents/sim/gar.h5", "r")
print(file.keys())

plt.plot(file["best_fitness"][:])
plt.show()



file = h5py.File("/home/samani/Documents/sim/sim.h5", "r")
sp2 = file["sp2"][:]

precision_bits = {"sp1": (0, 200, 8), "sp2": (0, 200, 8), "sp1_cells": (0, 200, 8), "sp2_cells": (0, 200, 8), "params": (0, 200, 8)}




for i in range(10):

    genetic_algorithm(
        population_size=200,
        specie_matrix_shape=(30, 30),
        precision_bits=precision_bits,
        num_params=6,
        max_generation=100,
        mutation_rates=[.02, .02, .02, .02, .01],
        crossover_rates=[.85, .85, .85, .85, .85],
        num_crossover_points=[2, 2, 2, 2, 1],
        target=sp2,
        target_precision_bits=(0, 200, 8),
        result_path="/home/samani/Documents/sim",
        selection_method="tournament",
        tournament_size=10,
        file_name=f"g{i}",
        dt=0.01,
        sim_start=1,
        sim_stop=20,
        epochs=500,
        fitness_trigger=False
    )




file = h5py.File("/home/samani/Documents/sim/sim.h5", "r")
sp2 = file["sp2"][:]
precision_bits = {"sp1": (0, 200, 8), "sp2": (0, 200, 8), "sp1_cells": (0, 200, 8), "sp2_cells": (0, 200, 8), "params": (0, 200, 8)}

genetic_algorithm(
        population_size=300,
        specie_matrix_shape=(5, 5),
        precision_bits=precision_bits,
        num_params=6,
        max_generation=1000,
        mutation_rates=[.02, .02, .02, .02, .01],
        crossover_rates=[.85, .85, .85, .85, .85],
        num_crossover_points=[2, 2, 2, 2, 1],
        target=sp2,
        target_precision_bits=(0, 200, 8),
        result_path="/home/samani/Documents/sim",
        selection_method="tournament",
        tournament_size=20,
        file_name="new",
        dt=0.01,
        sim_start=1,
        sim_stop=20,
        epochs=500,
        fitness_trigger=False
    )


from heatmap import *


model1 = HeatMap(
    data_path="/home/samani/Documents/sim/new.h5",
    video_directory="/home/samani/Documents/sim/",
    video_name="GA-result",
    title="GA-result",
    x_label="X",
    y_label="Y",
    c_map="RedBlack",
    fps=10,
    interval=50,
    writer='ffmpeg',
    color_bar=True,
    norm=False
)

model1.heatmap_animation(key="best_results")



file = h5py.File("/home/samani/Documents/sim/sim.h5", "r")
sp2 = file["sp2"][:]
precision_bits = {"sp1": (0, 5, 8), "sp2": (0, 200, 8), "sp1_cells": (0, 5, 8), "sp2_cells": (0, 5, 8), "params": (0, 4, 8)}


genetic_algorithm(
        population_size=300,
        specie_matrix_shape=(5, 5),
        precision_bits=precision_bits,
        num_params=6,
        max_generation=200,
        mutation_rates=[.02, .02, .02, .02, .01],
        crossover_rates=[.85, .85, .85, .85, .85],
        num_crossover_points=[2, 2, 2, 2, 1],
        target=sp2,
        target_precision_bits=(0, 200, 8),
        result_path="/home/samani/Documents/sim",
        selection_method="tournament",
        tournament_size=20,
        file_name="new",
        dt=0.01,
        sim_start=1,
        sim_stop=20,
        epochs=500,
        fitness_trigger=False
    )


from heatmap import *

model1 = HeatMap(
    data_path="/home/samani/Documents/sim/new.h5",
    video_directory="/home/samani/Documents/sim/",
    video_name="GA",
    title="GA",
    x_label="X",
    y_label="Y",
    c_map="RedBlack",
    fps=10,
    interval=50,
    writer='ffmpeg',
    color_bar=True,
    norm=False
)

model1.heatmap_animation(key="best_results")


file = h5py.File("/home/samani/Documents/sim/new.h5", "r")

print(file["best_results"][:, :, 171])

file = h5py.File("/home/samani/Documents/sim/sim.h5", "r")
sp2 = file["sp2"]

print(sp2[:])
"""




