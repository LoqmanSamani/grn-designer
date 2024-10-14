from cost import *
from crossover import *
from initialization import *
from mutation import *
from simulation import *
import math


def evolutionary_optimization(
        population, target, population_size, num_patterns, init_species, init_pairs, cost_alpha, cost_beta, max_val, sim_mutation_rate,
        compartment_mutation_rate, parameter_mutation_rate, insertion_mutation_rate, deletion_mutation_rate,
        sim_means, sim_std_devs, sim_min_vals, sim_max_vals, compartment_mean, compartment_std, compartment_min_val,
        compartment_max_val, sim_distribution, compartment_distribution, species_param_means, species_param_stds,
        species_param_min_vals, species_param_max_vals, complex_param_means, complex_param_stds, complex_param_min_vals,
        complex_param_max_vals, param_distribution, sim_mutation, compartment_mutation, param_mutation,
        species_insertion_mutation, species_deletion_mutation, crossover_alpha, sim_crossover, compartment_crossover,
        param_crossover, num_elite_individuals, individual_fix_size, species_parameters, complex_parameters, simulation_parameters
):
    """
    Perform an evolutionary optimization process to evolve a population of individuals toward a target state.

    This function simulates each individual in the population, computes their performance relative to a target state,
    applies mutations and crossover operations, and retains elite individuals based on performance. Underperforming
    individuals are reinitialized or optimized through mutation and crossover mechanisms.

        Parameters:
            - population (list of numpy.ndarray): The population of individuals to be evolved, each represented by a 3D numpy array.
            - target (numpy.ndarray): The target state the individuals aim to approximate, represented as a 2D numpy array.
            - population_size (int): The total size of the population.
            - num_patterns (int): The number of pattern simulations for each individual.
            - init_species (int): The number of initial species for the individuals.
            - init_pairs (int): The number of initial complex pairs for the individuals.
            - cost_alpha (float): Weighting factor for the primary component of the cost function.
            - cost_beta (float): Weighting factor for the secondary component of the cost function.
            - max_val (float): Maximum allowable value in the cost function computation.
            - sim_mutation_rate (float): Mutation rate for the simulation parameters.
            - compartment_mutation_rate (float): Mutation rate for the compartment parameters.
            - parameter_mutation_rate (float): Mutation rate for the species and complex parameters.
            - insertion_mutation_rate (float): Mutation rate for species insertion operations.
            - deletion_mutation_rate (float): Mutation rate for species deletion operations.
            - sim_means (list of float): Mean values for the simulation parameters.
            - sim_std_devs (list of float): Standard deviation values for the simulation parameters.
            - sim_min_vals (list of float): Minimum values for the simulation parameters.
            - sim_max_vals (list of float): Maximum values for the simulation parameters.
            - compartment_mean (float): Mean value for the compartment parameters.
            - compartment_std (float): Standard deviation for the compartment parameters.
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
            - sim_mutation (bool): Whether to apply mutations to the simulation parameters.
            - compartment_mutation (bool): Whether to apply mutations to the compartment parameters.
            - param_mutation (bool): Whether to apply mutations to the species and complex parameters.
            - species_insertion_mutation (bool): Whether to apply species insertion mutations.
            - species_deletion_mutation (bool): Whether to apply species deletion mutations.
            - crossover_alpha (float): Weighting factor for crossover operations.
            - sim_crossover (bool): Whether to apply crossover to simulation variables.
            - compartment_crossover (bool): Whether to apply crossover to compartment parameters.
            - param_crossover (bool): Whether to apply crossover to species and complex parameters.
            - num_elite_individuals (int): The number of elite individuals selected for crossover.
            - individual_fix_size (bool): Whether individuals should maintain a fixed size across the population.
            - species_parameters (list of lists): Initial parameters for the species in the individuals.
            - complex_parameters (list of tuples): Initial parameters for the complexes in the individuals.
            - simulation_parameters (dict): Dictionary containing simulation-specific parameters (e.g., "max_simulation_epoch",
              "simulation_stop_time", "time_step").

        Returns:
            - low_cost_individuals (list of numpy.ndarray): The population of evolved individuals with lower costs.
            - low_costs (list of float): The corresponding costs of the evolved individuals.
            - mean_cost (float): The mean cost of the evolved population.
    """

    _, y, x = population[0].shape
    m = len(population)
    predictions = np.zeros((num_patterns, m, y, x))

    # Simulate each individual and collect predictions
    for i in range(m):
        predictions[:, i, :, :] = individual_simulation(
            individual=population[i],
            num_patterns=num_patterns
        )
        # reset each individual after simulation
        population[i] = reset_individual(individual=population[i])

    # Compute costs for the population
    costs = compute_cost(
        predictions=predictions,
        target=target,
        alpha=cost_alpha,
        beta=cost_beta,
        max_val=max_val
    )
    # delete NaN cost individuals from the population
    costs = list(costs)

    filtered_data = [(ind, cost) for ind, cost in zip(population, costs) if not math.isnan(cost) and cost != float('inf')]
    if not filtered_data:
        population = []
        costs = np.array([])
    else:
        population, costs = zip(*filtered_data)
        population = list(population)
        costs = np.array(costs)

    mean_cost = np.mean(costs)

    # Sort the costs and get the indices of the elite individuals
    sorted_indices = np.argsort(costs)
    lowest_indices = sorted_indices[:num_elite_individuals]

    # Separate individuals and costs into low-cost and high-cost groups
    low_cost_individuals = [population[i] for i in range(len(costs)) if costs[i] < mean_cost]
    high_cost_individuals = [population[i] for i in range(len(costs)) if costs[i] >= mean_cost]

    # Separate the costs in the same way
    low_costs = [costs[i] for i in range(len(costs)) if costs[i] < mean_cost]
    high_costs = [costs[i] for i in range(len(costs)) if costs[i] >= mean_cost]

    # Elite individuals based on the lowest costs
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
    predictions1 = np.zeros((num_patterns, len(high_cost_individuals), y, x))

    for i in range(len(high_cost_individuals)):
        predictions1[:, i, :, :] = individual_simulation(
            individual=high_cost_individuals[i],
            num_patterns=num_patterns
        )
        high_cost_individuals[i] = reset_individual(individual=high_cost_individuals[i])


    costs1 = compute_cost(
        predictions=predictions1,
        target=target,
        alpha=cost_alpha,
        beta=cost_beta,
        max_val=max_val
    )

    costs1 = list(costs1)
    filtered_data1 = [(ind, cost) for ind, cost in zip(high_cost_individuals, costs1) if not math.isnan(cost) and cost != float('inf')]

    if not filtered_data1:

        high_cost_individuals = []
        costs1 = np.array([])
    else:
        high_cost_individuals, costs1 = zip(*filtered_data1)
        high_cost_individuals = list(high_cost_individuals)
        costs1 = np.array(costs1)

    # Filter out individuals that improved after crossover
    inxs = []
    for i in range(len(costs1)):
        if costs1[i] < mean_cost:
            low_cost_individuals.append(high_cost_individuals[i])
            low_costs.append(costs1[i])
            inxs.append(i)

    for inx in sorted(inxs, reverse=True):
        del high_cost_individuals[inx]


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
    predictions2 = np.zeros((num_patterns, len(high_cost_individuals), y, x))

    for i in range(len(high_cost_individuals)):
        predictions2[:, i, :, :] = individual_simulation(
            individual=high_cost_individuals[i],
            num_patterns=num_patterns
        )
        high_cost_individuals[i] = reset_individual(individual=high_cost_individuals[i])

    costs2 = compute_cost(
        predictions=predictions2,
        target=target,
        alpha=cost_alpha,
        beta=cost_beta,
        max_val=max_val
    )

    costs2 = list(costs2)
    filtered_data2 = [(ind, cost) for ind, cost in zip(high_cost_individuals, costs2) if not math.isnan(cost) and cost != float('inf')]
    if not filtered_data2:

        high_cost_individuals = []
        costs2 = np.array([])
    else:
        high_cost_individuals, costs2 = zip(*filtered_data2)
        high_cost_individuals = list(high_cost_individuals)
        costs2 = np.array(costs2)

    # Filter out individuals that improved after mutation
    inxs2 = []
    for i in range(len(costs2)):
        if costs2[i] < mean_cost:
            low_cost_individuals.append(high_cost_individuals[i])
            low_costs.append(costs2[i])
            inxs2.append(i)

    for inx in sorted(inxs2, reverse=True):
        del high_cost_individuals[inx]


    pop_size = population_size - len(low_cost_individuals)
    if pop_size > 0:
        initialized_individuals = population_initialization(
            population_size=pop_size,
            individual_shape=low_cost_individuals[0].shape,
            species_parameters=species_parameters,
            complex_parameters=complex_parameters,
            num_species=low_cost_individuals[0][-1, -1, 0],
            num_pairs=low_cost_individuals[0][-1, -1, 1],
            max_sim_epochs=simulation_parameters["max_simulation_epoch"],
            sim_stop_time=simulation_parameters["simulation_stop_time"],
            time_step=simulation_parameters["time_step"],
            individual_fix_size=individual_fix_size,
            init_species=init_species,
            init_pairs=init_pairs
        )

        # Recompute costs after initialization
        predictions3 = np.zeros((num_patterns, len(initialized_individuals), y, x))

        for i in range(len(initialized_individuals)):
            predictions3[:, i, :, :] = individual_simulation(
                individual=initialized_individuals[i],
                num_patterns=num_patterns
            )
            initialized_individuals[i] = reset_individual(individual=initialized_individuals[i])

        costs3 = compute_cost(
            predictions=predictions3,
            target=target,
            alpha=cost_alpha,
            beta=cost_beta,
            max_val=max_val
        )

        costs3 = [1000 if math.isnan(cost) or cost == float('inf') else cost for cost in costs3]
        low_cost_individuals = low_cost_individuals + initialized_individuals
        low_costs = low_costs + costs3

    return low_cost_individuals, low_costs, mean_cost

