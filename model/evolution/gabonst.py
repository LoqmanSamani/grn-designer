import numpy as np
from numba import jit
from master_project.model.sim.sim_ind.simulation import *
from initialization import *
from cost import *
from mutation import *
from crossover import *


def evolutionary_optimization(
        population, target, cost_alpha, cost_beta, cost_kernel_size, cost_method, sim_mutation_rate,
        compartment_mutation_rate, parameter_mutation_rate, insertion_mutation_rate, deletion_mutation_rate,
        sim_means, sim_std_devs, sim_min_vals, sim_max_vals, compartment_mean, compartment_std, compartment_min_val,
        compartment_max_val, sim_distribution, compartment_distribution, species_param_means, species_param_stds,
        species_param_min_vals, species_param_max_vals, complex_param_means, complex_param_stds, complex_param_min_vals,
        complex_param_max_vals, param_distribution, sim_mutation, compartment_mutation, param_mutation,
        species_insertion_mutation, species_deletion_mutation, crossover_alpha, sim_crossover, compartment_crossover,
        param_crossover, num_elite_individuals, individual_fix_size, species_parameters, complex_parameters
):
    """
        Performs an evolutionary optimization process on a population of individuals, applying mutation and crossover
        operations to evolve the population towards a target state.

        This function simulates each individual, computes their costs relative to a target, and then applies mutations and
        crossovers. It retains the best-performing individuals (elite individuals) and reinitializes underperforming
        individuals if necessary.

        Parameters:
            - population (list of numpy.ndarray): The current population of individuals to be optimized, where each
              individual is represented as a 3D numpy array.
            - target (numpy.ndarray): The target state that the population aims to achieve, represented as a 2D numpy array.
            - cost_alpha (float): Weighting factor for the primary component of the cost function.
            - cost_beta (float): Weighting factor for the secondary component of the cost function.
            - cost_kernel_size (int): The size of the kernel used in the cost computation.
            - cost_method (str): The method used to compute the cost function (e.g., "mse", "mae").
            - sim_mutation_rate (float): The mutation rate for simulation parameters.
            - compartment_mutation_rate (float): The mutation rate for compartment parameters.
            - parameter_mutation_rate (float): The mutation rate for species and complex parameters.
            - insertion_mutation_rate (float): The mutation rate for species insertion operations.
            - deletion_mutation_rate (float): The mutation rate for species deletion operations.
            - sim_means (list of float): Mean values for the simulation parameters.
            - sim_std_devs (list of float): Standard deviation values for the simulation parameters.
            - sim_min_vals (list of float): Minimum values for the simulation parameters.
            - sim_max_vals (list of float): Maximum values for the simulation parameters.
            - compartment_mean (float): Mean value for the compartment parameters.
            - compartment_std (float): Standard deviation value for the compartment parameters.
            - compartment_min_val (float): Minimum value for the compartment parameters.
            - compartment_max_val (float): Maximum value for the compartment parameters.
            - sim_distribution (str): The distribution type for simulation mutations (e.g., "normal", "uniform").
            - compartment_distribution (str): The distribution type for compartment mutations (e.g., "normal", "uniform").
            - species_param_means (list of float): Mean values for the species parameters.
            - species_param_stds (list of float): Standard deviation values for the species parameters.
            - species_param_min_vals (list of float): Minimum values for the species parameters.
            - species_param_max_vals (list of float): Maximum values for the species parameters.
            - complex_param_means (list of float): Mean values for the complex parameters.
            - complex_param_stds (list of float): Standard deviation values for the complex parameters.
            - complex_param_min_vals (list of float): Minimum values for the complex parameters.
            - complex_param_max_vals (list of float): Maximum values for the complex parameters.
            - param_distribution (str): The distribution type for parameter mutations (e.g., "normal", "uniform").
            - sim_mutation (bool): Flag indicating whether to apply simulation parameter mutations.
            - compartment_mutation (bool): Flag indicating whether to apply compartment parameter mutations.
            - param_mutation (bool): Flag indicating whether to apply species and complex parameter mutations.
            - species_insertion_mutation (bool): Flag indicating whether to apply species insertion mutations.
            - species_deletion_mutation (bool): Flag indicating whether to apply species deletion mutations.
            - crossover_alpha (float): The weighting factor for crossover operations, determining the contribution of elite
              individuals.
            - sim_crossover (bool): Flag indicating whether to apply crossover to simulation variables.
            - compartment_crossover (bool): Flag indicating whether to apply crossover to compartment parameters.
            - param_crossover (bool): Flag indicating whether to apply crossover to species and complex parameters.
            - num_elite_individuals (int): The number of elite individuals selected for crossover operations.
            - individual_fix_size (bool): Flag indicating whether individuals in the population should maintain a fixed size.
            - species_parameters (list of lists): The initial parameters for the species in the population.
            - complex_parameters (list of tuples): The initial parameters for the complexes in the population.

        Returns:
            - list of numpy.ndarray: The optimized population after applying evolutionary operations, with individuals
              closer to the target state.
        """

    _, y, x = population[0].shape
    m = len(population)
    predictions = np.zeros((m, y, x))
    delta_D = []

    # Simulate each individual and collect predictions and deltas
    for i in range(m):
        predictions[i, :, :], dd = individual_simulation(individual=population[i])
        delta_D.append(dd)

    # Compute costs for the population
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
    lowest_indices = sorted_indices[:num_elite_individuals]

    # Separate individuals into low-cost and high-cost groups
    low_cost_individuals = [population[i] for i in range(len(costs)) if costs[i] < mean_cost]
    high_cost_individuals = [population[i] for i in range(len(costs)) if costs[i] >= mean_cost]
    elite_individuals = [population[i] for i in lowest_indices]

    # Apply mutations to low-cost individuals
    for i in range(len(low_cost_individuals)):
        low_cost_individuals[i] = apply_mutation(
            individual=low_cost_individuals[i],
            sim_mutation_rate=sim_mutation_rate,
            compartment_mutation_rate=compartment_mutation_rate,
            parameter_mutation_rate=parameter_mutation_rate,
            insertion_mutation_rate=insertion_mutation_rate,
            deletion_mutation_rate=deletion_mutation_rate,
            sim_means=sim_means,
            sim_std_devs=sim_std_devs,
            sim_min_vals=sim_min_vals,
            sim_max_vals=sim_max_vals,
            compartment_mean=compartment_mean,
            compartment_std=compartment_std,
            compartment_min_val=compartment_min_val,
            compartment_max_val=compartment_max_val,
            sim_distribution=sim_distribution,
            compartment_distribution=compartment_distribution,
            species_param_means=species_param_means,
            species_param_stds=species_param_stds,
            species_param_min_vals=species_param_min_vals,
            species_param_max_vals=species_param_max_vals,
            complex_param_means=complex_param_means,
            complex_param_stds=complex_param_stds,
            complex_param_min_vals=complex_param_min_vals,
            complex_param_max_vals=complex_param_max_vals,
            param_distribution=param_distribution,
            sim_mutation=sim_mutation,
            compartment_mutation=compartment_mutation,
            param_mutation=param_mutation,
            species_insertion_mutation=species_insertion_mutation,
            species_deletion_mutation=species_deletion_mutation
        )

    # Apply crossover to high-cost individuals
    for i in range(len(high_cost_individuals)):
        filtered_elite_individuals = filter_elite_individuals(
            low_cost_individuals=low_cost_individuals,
            elite_individuals=elite_individuals,
            high_cost_individual=high_cost_individuals[i]
        )

        high_cost_individuals[i] = apply_crossover(
            elite_individuals=filtered_elite_individuals,
            individual=high_cost_individuals[i],
            crossover_alpha=crossover_alpha,
            sim_crossover=sim_crossover,
            compartment_crossover=compartment_crossover,
            param_crossover=param_crossover
        )

    # Recompute costs after crossover
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

    # Filter out individuals that improved after crossover
    for i in range(len(costs1)):
        if costs1[i] < mean_cost:
            low_cost_individuals.append(high_cost_individuals[i])
            del high_cost_individuals[i]

    # Apply mutation to remaining high-cost individuals
    if len(high_cost_individuals) > 0:
        for i in range(len(high_cost_individuals)):
            high_cost_individuals[i] = apply_mutation(
                individual=high_cost_individuals[i],
                sim_mutation_rate=sim_mutation_rate,
                compartment_mutation_rate=compartment_mutation_rate,
                parameter_mutation_rate=parameter_mutation_rate,
                insertion_mutation_rate=insertion_mutation_rate,
                deletion_mutation_rate=deletion_mutation_rate,
                sim_means=sim_means,
                sim_std_devs=sim_std_devs,
                sim_min_vals=sim_min_vals,
                sim_max_vals=sim_max_vals,
                compartment_mean=compartment_mean,
                compartment_std=compartment_std,
                compartment_min_val=compartment_min_val,
                compartment_max_val=compartment_max_val,
                sim_distribution=sim_distribution,
                compartment_distribution=compartment_distribution,
                species_param_means=species_param_means,
                species_param_stds=species_param_stds,
                species_param_min_vals=species_param_min_vals,
                species_param_max_vals=species_param_max_vals,
                complex_param_means=complex_param_means,
                complex_param_stds=complex_param_stds,
                complex_param_min_vals=complex_param_min_vals,
                complex_param_max_vals=complex_param_max_vals,
                param_distribution=param_distribution,
                sim_mutation=sim_mutation,
                compartment_mutation=compartment_mutation,
                param_mutation=param_mutation,
                species_insertion_mutation=species_insertion_mutation,
                species_deletion_mutation=species_deletion_mutation
            )

    # Recompute costs after mutation
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

    # Filter out individuals that improved after mutation
    for i in range(len(costs2)):
        if costs2[i] < mean_cost:
            low_cost_individuals.append(high_cost_individuals[i])
            del high_cost_individuals[i]

    # Reinitialize remaining high-cost individuals if any
    if len(high_cost_individuals) > 0:
        high_cost_individuals = population_initialization(
            population_size=len(high_cost_individuals),
            individual_shape=high_cost_individuals[0].shape,
            species_parameters=species_parameters,
            complex_parameters=complex_parameters,
            num_species=high_cost_individuals[0][-1, -1, 0],
            num_pairs=high_cost_individuals[0][-1, -1, 1],
            max_sim_epochs=high_cost_individuals[0][-1, -1, 2],
            sim_stop_time=high_cost_individuals[0][-1, -1, 3],
            time_step=high_cost_individuals[0][-1, -1, 4],
            individual_fix_size=individual_fix_size
        )

        low_cost_individuals = low_cost_individuals + high_cost_individuals

    return low_cost_individuals






    





