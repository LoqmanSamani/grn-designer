from genetic_algorithm import genetic_algorithm
import h5py
import numpy as np





full_path = "/home/samani/Documents/sim"


sp1 = np.zeros((20, 20))
sp2 = np.zeros((20, 20))
sp1_cells = np.zeros((20, 20))
sp2_cells = np.zeros((20, 20))
params = np.array([[.5, .5, 4, 4, .1, .1]])
dt = 0.01
sim_start = 1
sim_stop = 20
epochs = 500
target_shape = (20, 20)

file = h5py.File("/home/samani/Documents/sim/sim.h5", "r")
target = np.array(file["sp2"])

precision_bits = {"sp1": (0, 10, 10), "sp2": (0, 10, 10), "sp1_cells": (0, 10, 10), "sp2_cells": (0, 10, 10), "params": (0, 10, 10)}

genetic_algorithm(
    population_size=100,
    specie_matrix_shape=(20, 20),
    precision_bits=precision_bits,
    num_params=6,
    generations=100,
    mutation_rates=[.01, .01, .01, .01, .01],
    crossover_rates=[.8, .8, .8, .8, .8],
    num_crossover_points=[2, 2, 2, 2, 1],
    target=target,
    target_precision_bits=(0, 10, 10),
    result_path="/home/samani/Documents/sim",
    file_name="ga",
    dt=0.01,
    sim_start=1,
    sim_stop=20,
    epochs=500
)