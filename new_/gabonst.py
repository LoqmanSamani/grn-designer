from cost import *
from crossover import *
from initialization import *
from mutation import *
from simulation import *
import math


def evolutionary_optimization(
        population,
        target,
        population_size,
        num_patterns,
        init_species,
        init_pairs,
        cost,
        rates,
        bounds,
        mutation,
        crossover,
        elite_agents,
        agent_shape,
        parameters
):

    _, y, x = population[0].shape
    m = len(population)
    (species_parameters,
     complex_parameters,
     simulation_parameters) = parameters
    (crossover_alpha,
     simulation_crossover,
     initial_condition_crossover,
     parameter_crossover) = crossover
    (sim_mutation,
     initial_condition_mutation,
     parameter_mutation,
     insertion_mutation,
     deletion_mutation) = mutation
    (simulation_min,
     simulation_max,
     initial_condition_min,
     initial_condition_max,
     parameter_min,
     parameter_max) = bounds
    (sim_mutation_rate,
     initial_condition_mutation_rate,
     parameter_mutation_rate,
     insertion_mutation_rate,
     deletion_mutation_rate) = rates
    cost_alpha, cost_beta = cost

    predictions = np.zeros(
        shape=(num_patterns, m, y, x),
        dtype=np.float32
    )

    # Simulate each individual and collect predictions
    for i in range(m):
        predictions[:, i, :, :] = agent_simulation(
            agent=np.copy(population[i]),
            num_patterns=num_patterns
        )

    # Compute costs for the population
    costs = compute_cost(
        predictions=predictions,
        target=target,
        alpha=cost_alpha,
        beta=cost_beta
    )

    mean_cost = np.mean(costs)

    # Sort the costs and get the indices of the elite individuals
    sorted_indices = np.argsort(costs)
    lowest_indices = sorted_indices[:elite_agents]

    # Separate individuals and costs into low-cost and high-cost groups
    low_cost_agents = [population[i] for i in range(len(costs)) if costs[i] < mean_cost]
    high_cost_agents = [population[i] for i in range(len(costs)) if costs[i] >= mean_cost]

    # Separate the costs in the same way
    low_costs = [costs[i] for i in range(len(costs)) if costs[i] < mean_cost]
    high_costs = [costs[i] for i in range(len(costs)) if costs[i] >= mean_cost]

    sort_ = np.argsort(np.array(low_costs))
    sorted_ = [low_cost_agents[i] for i in sort_]

    # Elite individuals based on the lowest costs
    elite_agents_ = [population[i] for i in lowest_indices]

    # Apply mutations to low-cost individuals
    for i in range(len(low_cost_agents)):
        low_cost_agents[i] = apply_mutation(population=sorted_,
                                            agent=low_cost_agents[i], sim_mutation_rate=sim_mutation_rate,
                                            compartment_mutation_rate=initial_condition_mutation_rate,
                                            parameter_mutation_rate=parameter_mutation_rate,
                                            insertion_mutation_rate=insertion_mutation_rate,
                                            deletion_mutation_rate=deletion_mutation_rate,
                                            simulation_min=simulation_min, simulation_max=simulation_max,
                                            initial_condition_min=initial_condition_min,
                                            initial_condition_max=initial_condition_max,
                                            parameter_min=parameter_min, parameter_max=parameter_max,
                                            simulation_mutation=sim_mutation,
                                            initial_condition_mutation=initial_condition_mutation,
                                            parameter_mutation=parameter_mutation,
                                            insertion_mutation=insertion_mutation,
                                            deletion_mutation=deletion_mutation)

    # Apply crossover to high-cost individuals
    for i in range(len(high_cost_agents)):
        filtered_elite_individuals = filter_elite_individuals(
            low_cost_individuals=low_cost_agents,
            elite_individuals=elite_agents,
            high_cost_individual=high_cost_agents[i]
        )

        high_cost_agents[i] = apply_crossover(
            elite_individuals=filtered_elite_individuals,
            individual=high_cost_agents[i],
            crossover_alpha=crossover_alpha,
            sim_crossover=simulation_crossover,
            compartment_crossover=initial_condition_crossover,
            param_crossover=parameter_crossover
        )

    # Recompute costs after crossover
    predictions1 = np.zeros((num_patterns, len(high_cost_agents), y, x))

    for i in range(len(high_cost_agents)):
        predictions1[:, i, :, :] = agent_simulation(agent=high_cost_agents[i], num_patterns=num_patterns)
        high_cost_agents[i] = reset_individual(individual=high_cost_agents[i])


    costs1 = compute_cost(
        predictions=predictions1,
        target=target,
        alpha=cost_alpha,
        beta=cost_beta,
        max_val=max_val
    )

    costs1 = list(costs1)
    filtered_data1 = [(ind, cost) for ind, cost in zip(high_cost_agents, costs1) if not math.isnan(cost) and cost != float('inf')]

    if not filtered_data1:

        high_cost_agents = []
        costs1 = np.array([])
    else:
        high_cost_agents, costs1 = zip(*filtered_data1)
        high_cost_agents = list(high_cost_agents)
        costs1 = np.array(costs1)

    # Filter out individuals that improved after crossover
    inxs = []
    for i in range(len(costs1)):
        if costs1[i] < mean_cost:
            low_cost_agents.append(high_cost_agents[i])
            low_costs.append(costs1[i])
            inxs.append(i)

    for inx in sorted(inxs, reverse=True):
        del high_cost_agents[inx]


    # Apply mutation to remaining high-cost individuals
    if len(high_cost_agents) > 0:
        for i in range(len(high_cost_agents)):
            high_cost_agents[i] = apply_mutation(agent=high_cost_agents[i],
                                                 sim_mutation_rate=sim_mutation_rate,
                                                 compartment_mutation_rate=initial_condition_mutation_rate,
                                                 parameter_mutation_rate=parameter_mutation_rate,
                                                 insertion_mutation_rate=insertion_mutation_rate,
                                                 deletion_mutation_rate=deletion_mutation_rate,
                                                 simulation_min=simulation_min, simulation_max=simulation_max,
                                                 initial_condition_min=initial_condition_min,
                                                 initial_condition_max=initial_condition_max,
                                                 parameter_min=parameter_min, parameter_max=parameter_max,
                                                 simulation_mutation=sim_mutation,
                                                 initial_condition_mutation=initial_condition_mutation,
                                                 parameter_mutation=parameter_mutation,
                                                 insertion_mutation=insertion_mutation,
                                                 deletion_mutation=deletion_mutation)

    # Recompute costs after mutation
    predictions2 = np.zeros((num_patterns, len(high_cost_agents), y, x))

    for i in range(len(high_cost_agents)):
        predictions2[:, i, :, :] = agent_simulation(agent=high_cost_agents[i], num_patterns=num_patterns)
        high_cost_agents[i] = reset_individual(individual=high_cost_agents[i])

    costs2 = compute_cost(
        predictions=predictions2,
        target=target,
        alpha=cost_alpha,
        beta=cost_beta,
        max_val=max_val
    )

    costs2 = list(costs2)
    filtered_data2 = [(ind, cost) for ind, cost in zip(high_cost_agents, costs2) if not math.isnan(cost) and cost != float('inf')]
    if not filtered_data2:

        high_cost_agents = []
        costs2 = np.array([])
    else:
        high_cost_agents, costs2 = zip(*filtered_data2)
        high_cost_agents = list(high_cost_agents)
        costs2 = np.array(costs2)

    # Filter out individuals that improved after mutation
    inxs2 = []
    for i in range(len(costs2)):
        if costs2[i] < mean_cost:
            low_cost_agents.append(high_cost_agents[i])
            low_costs.append(costs2[i])
            inxs2.append(i)

    for inx in sorted(inxs2, reverse=True):
        del high_cost_agents[inx]


    pop_size = population_size - len(low_cost_agents)
    if pop_size > 0:
        initialized_individuals = population_initialization(
            population_size=pop_size,
            individual_shape=low_cost_agents[0].shape,
            species_parameters=species_parameters,
            complex_parameters=complex_parameters,
            num_species=low_cost_agents[0][-1, -1, 0],
            num_pairs=low_cost_agents[0][-1, -1, 1],
            max_sim_epochs=simulation_parameters["max_simulation_epoch"],
            sim_stop_time=simulation_parameters["simulation_stop_time"],
            time_step=simulation_parameters["time_step"],
            individual_fix_size=agent_shape,
            init_species=init_species,
            init_pairs=init_pairs
        )

        # Recompute costs after initialization
        predictions3 = np.zeros((num_patterns, len(initialized_individuals), y, x))

        for i in range(len(initialized_individuals)):
            predictions3[:, i, :, :] = agent_simulation(agent=initialized_individuals[i], num_patterns=num_patterns)
            initialized_individuals[i] = reset_individual(individual=initialized_individuals[i])

        costs3 = compute_cost(
            predictions=predictions3,
            target=target,
            alpha=cost_alpha,
            beta=cost_beta,
            max_val=max_val
        )

        costs3 = [1000 if math.isnan(cost) or cost == float('inf') else cost for cost in costs3]
        low_cost_agents = low_cost_agents + initialized_individuals
        low_costs = low_costs + costs3

    return low_cost_agents, low_costs, mean_cost

