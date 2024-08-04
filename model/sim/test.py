from simulation import *
import numpy as np
import time


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








