import numpy as np
from numba import jit
from master_project.model.sim.sim_ind.simulation import *
from initialization import *
from cost import *
from mutation import *
from crossover import *


def evolutionary_optimization(population, target, cost_alpha, cost_beta, cost_kernel_size, cost_method,
                              sim_mutation_rates, compartment_mutation_rate, parameter_mutation_rate,
                              insertion_mutation_rate, deletion_mutation_rate, sim_min_vals, sim_max_vals, sim_dtypes,
                              compartment_mean, compartment_std, compartment_min_val, compartment_max_val,
                              compartment_distribution, param_means, param_stds, param_min_vals, param_max_vals,
                              param_distribution, sim_mutation, compartment_mutation, param_mutation,
                              species_insertion_mutation, species_deletion_mutation, crossover_alpha, sim_crossover,
                              compartment_crossover, param_crossover
                              ):

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
    sorted_indices = np.argsort(costs)
    lowest_indices = sorted_indices[:5]

    low_cost_individuals = [population[i] for i in range(len(costs)) if costs[i] < mean_cost]
    high_cost_individuals = [population[i] for i in range(len(costs)) if costs[i] >= mean_cost]
    elite_individuals = [population[i] for i in lowest_indices]

    for i in range(len(low_cost_individuals)):

        low_cost_individuals[i] = apply_mutation(
            individual=low_cost_individuals[i],
            sim_mutation_rates=sim_mutation_rates,
            compartment_mutation_rate=compartment_mutation_rate,
            parameter_mutation_rate=parameter_mutation_rate,
            insertion_mutation_rate=insertion_mutation_rate,
            deletion_mutation_rate=deletion_mutation_rate,
            sim_min_vals=sim_min_vals,
            sim_max_vals=sim_max_vals,
            sim_dtypes=sim_dtypes,
            compartment_mean=compartment_mean,
            compartment_std=compartment_std,
            compartment_min_val=compartment_min_val,
            compartment_max_val=compartment_max_val,
            compartment_distribution=compartment_distribution,
            param_means=param_means,
            param_stds=param_stds,
            param_min_vals=param_min_vals,
            param_max_vals=param_max_vals,
            param_distribution=param_distribution,
            sim_mutation=sim_mutation,
            compartment_mutation=compartment_mutation,
            param_mutation=param_mutation,
            species_insertion_mutation=species_insertion_mutation,
            species_deletion_mutation=species_deletion_mutation
        )

    for i in range(len(high_cost_individuals)):

        high_cost_individuals[i] = apply_crossover(
            elite_individuals=elite_individuals,
            individual=high_cost_individuals[i],
            crossover_alpha=crossover_alpha,
            sim_crossover=sim_crossover,
            compartment_crossover=compartment_crossover,
            param_crossover=param_crossover
        )

    predictions1 = np.zeros((len(high_cost_individuals), y, x))
    delta_D1 = []

    for i in range(len(high_cost_individuals)):
        predictions1[i, :, :], dd1 = individual_simulation(individual=high_cost_individuals[i])
        delta_D1.append(dd1)

    costs1 = compute_cost(
        predictions=predictions1,
        target=target,
        delta_D=delta_D1,
        alpha=cost_alpha,
        beta=cost_beta,
        kernel_size=cost_kernel_size,
        method=cost_method
    )

    for i in range(len(costs1)):
        if costs1[i] < mean_cost:
            low_cost_individuals.append(high_cost_individuals[i])
            del high_cost_individuals[i]

    if len(high_cost_individuals) > 0:

        for i in range(len(high_cost_individuals)):

            high_cost_individuals[i] = apply_mutation(
                individual=high_cost_individuals[i],
                sim_mutation_rates=sim_mutation_rates,
                compartment_mutation_rate=compartment_mutation_rate,
                parameter_mutation_rate=parameter_mutation_rate,
                insertion_mutation_rate=insertion_mutation_rate,
                deletion_mutation_rate=deletion_mutation_rate,
                sim_min_vals=sim_min_vals,
                sim_max_vals=sim_max_vals,
                sim_dtypes=sim_dtypes,
                compartment_mean=compartment_mean,
                compartment_std=compartment_std,
                compartment_min_val=compartment_min_val,
                compartment_max_val=compartment_max_val,
                compartment_distribution=compartment_distribution,
                param_means=param_means,
                param_stds=param_stds,
                param_min_vals=param_min_vals,
                param_max_vals=param_max_vals,
                param_distribution=param_distribution,
                sim_mutation=sim_mutation,
                compartment_mutation=compartment_mutation,
                param_mutation=param_mutation,
                species_insertion_mutation=species_insertion_mutation,
                species_deletion_mutation=species_deletion_mutation
            )

    predictions2 = np.zeros((len(high_cost_individuals), y, x))
    delta_D2 = []

    for i in range(len(high_cost_individuals)):
        predictions2[i, :, :], dd2 = individual_simulation(individual=high_cost_individuals[i])
        delta_D2.append(dd2)

    costs2 = compute_cost(
        predictions=predictions2,
        target=target,
        delta_D=delta_D2,
        alpha=cost_alpha,
        beta=cost_beta,
        kernel_size=cost_kernel_size,
        method=cost_method
    )

    for i in range(len(costs2)):
        if costs2[i] < mean_cost:
            low_cost_individuals.append(high_cost_individuals[i])
            del high_cost_individuals[i]

    if len(high_cost_individuals) > 0:

        for i in range(len(high_cost_individuals)):
            low_cost_individuals.append(...)  # apply initialize individual


    return low_cost_individuals







    





