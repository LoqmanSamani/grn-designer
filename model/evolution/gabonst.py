import numpy as np
from numba import jit
from master_project.model.sim.sim_ind.simulation import *
from cost import *

def evolutionary_optimization(population, target, cost_alpha, cost_beta, cost_kernel_size, cost_method):

    _, y, x = population[0].shape
    m = len(population)
    predictions = np.zeros((m, y, x))
    delta_D = []

    for i in range(m):
        predictions[i, :, :], dd = individual_simulation(individual=population[i])
        delta_D.append(dd)

    costs = compute_cost(
        predictions=predictions,
        target=target,
        delta_D=delta_D,
        alpha=cost_alpha,
        beta=cost_beta,
        kernel_size=cost_kernel_size,
        method=cost_method
    )

    mean_cost = np.mean(costs)
    





