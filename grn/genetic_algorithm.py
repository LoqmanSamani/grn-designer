from cost import *
from crossover import *
from initialization import *
from mutation import *
from simulation import *
import gc



def evolutionary_optimization(
        population,
        target,
        population_size,
        cost,
        rates,
        bounds,
        mutation,
        crossover,
        num_elite_agents,
        parameters,
        cost_constant,
        fixed_agent_shape
):
    z, y, x = population[0].shape
    m = len(population)

    (species_parameters,
     simulation_parameters) = parameters

    (crossover_alpha,
     simulation_crossover,
     initial_condition_crossover,
     parameter_crossover) = crossover

    (sim_mutation,
     initial_condition_mutation,
     parameter_mutation,
     species_insertion_mutation,
     connection_insertion_mutation,
     connection_deletion_mutation) = mutation

    (simulation_min,
     simulation_max,
     initial_condition_min,
     initial_condition_max,
     parameter_min,
     parameter_max) = bounds

    (sim_mutation_rate,
     initial_condition_mutation_rate,
     parameter_mutation_rate,
     species_insertion_mutation_rate,
     connection_insertion_mutation_rate,
     connection_deletion_mutation_rate) = rates

    cost_alpha, cost_beta, cost_pattern_proportion = cost
    predictions = []
    for i in range(m):
        predicted = agent_simulation(
            agent=population[i]
        )
        population[i] = reset_agent(agent=population[i])
        predictions.append(
            weighted_prediction(
                prediction=predicted,
                pattern_proportion=cost_pattern_proportion
            )
        )

    costs = compute_cost(
        predictions=predictions,
        target=target,
        alpha=cost_alpha,
        beta=cost_beta
    )

    costs[np.isnan(costs) | np.isinf(costs)] = cost_constant
    mean_cost = np.mean(costs)

    del predictions
    gc.collect()

    sorted_indices = np.argsort(costs)
    lowest_indices = sorted_indices[:num_elite_agents]

    low_cost_agents = [population[i] for i in range(len(costs)) if costs[i] < mean_cost]
    high_cost_agents = [population[i] for i in range(len(costs)) if costs[i] >= mean_cost]

    low_costs = [costs[i] for i in range(len(costs)) if costs[i] < mean_cost]
    sorted_low_costs = np.argsort(np.array(low_costs))
    sorted_low_cost_agents = [low_cost_agents[i] for i in sorted_low_costs]

    elite_agents = [population[i] for i in lowest_indices]

    del sorted_indices, lowest_indices, sorted_low_costs
    gc.collect()

    for i in range(len(low_cost_agents)):
        low_cost_agents[i] = apply_mutation(
            population=sorted_low_cost_agents,
            agent=low_cost_agents[i],
            sim_mutation_rate=sim_mutation_rate,
            initial_condition_mutation_rate=initial_condition_mutation_rate,
            parameter_mutation_rate=parameter_mutation_rate,
            species_insertion_mutation_rate=species_insertion_mutation_rate,
            connection_insertion_mutation_rate=connection_insertion_mutation_rate,
            connection_deletion_mutation_rate=connection_deletion_mutation_rate,
            simulation_min=simulation_min,
            simulation_max=simulation_max,
            initial_condition_min=initial_condition_min,
            initial_condition_max=initial_condition_max,
            parameter_min=parameter_min,
            parameter_max=parameter_max,
            simulation_mutation=sim_mutation,
            initial_condition_mutation=initial_condition_mutation,
            parameter_mutation=parameter_mutation,
            species_insertion_mutation=species_insertion_mutation,
            connection_insertion_mutation=connection_insertion_mutation,
            connection_deletion_mutation=connection_deletion_mutation
        )

    for i in range(len(high_cost_agents)):

        filtered_elite_agents = filter_elite_agents(
            low_cost_agents=low_cost_agents,
            elite_agents=elite_agents,
            high_cost_agent=high_cost_agents[i]
        )
        high_cost_agents[i] = apply_crossover(
            elite_agents=filtered_elite_agents,
            agent=high_cost_agents[i],
            crossover_alpha=crossover_alpha,
            simulation_crossover=simulation_crossover,
            initial_condition_crossover=initial_condition_crossover,
            parameter_crossover=parameter_crossover
        )

    predictions1 = []

    for i in range(len(high_cost_agents)):

        predicted = agent_simulation(agent=high_cost_agents[i])
        high_cost_agents[i] = reset_agent(agent=high_cost_agents[i])
        predictions1.append(weighted_prediction(prediction=predicted, pattern_proportion=cost_pattern_proportion))

    costs1 = compute_cost(
        predictions=predictions1,
        target=target,
        alpha=cost_alpha,
        beta=cost_beta
    )

    costs1[np.isnan(costs1) | np.isinf(costs1)] = cost_constant

    del predictions1
    gc.collect()

    inxs = []
    for i in range(len(costs1)):
        if costs1[i] < mean_cost:
            low_cost_agents.append(high_cost_agents[i])
            low_costs.append(costs1[i])
            inxs.append(i)

    for inx in sorted(inxs, reverse=True):
        del high_cost_agents[inx]

    del costs1, inxs
    gc.collect()

    if len(high_cost_agents) > 0:
        for i in range(len(high_cost_agents)):
            high_cost_agents[i] = apply_mutation(
                population=sorted_low_cost_agents,
                agent=high_cost_agents[i],
                sim_mutation_rate=sim_mutation_rate,
                initial_condition_mutation_rate=initial_condition_mutation_rate,
                parameter_mutation_rate=parameter_mutation_rate,
                species_insertion_mutation_rate=species_insertion_mutation_rate,
                connection_insertion_mutation_rate=connection_insertion_mutation_rate,
                connection_deletion_mutation_rate=connection_deletion_mutation_rate,
                simulation_min=simulation_min,
                simulation_max=simulation_max,
                initial_condition_min=initial_condition_min,
                initial_condition_max=initial_condition_max,
                parameter_min=parameter_min,
                parameter_max=parameter_max,
                simulation_mutation=sim_mutation,
                initial_condition_mutation=initial_condition_mutation,
                parameter_mutation=parameter_mutation,
                species_insertion_mutation=species_insertion_mutation,
                connection_insertion_mutation=connection_insertion_mutation,
                connection_deletion_mutation=connection_deletion_mutation
            )

    predictions2 = []

    for i in range(len(high_cost_agents)):
        predicted = agent_simulation(
            agent=high_cost_agents[i]
        )
        high_cost_agents[i] = reset_agent(agent=high_cost_agents[i])
        predictions2.append(weighted_prediction(prediction=predicted, pattern_proportion=cost_pattern_proportion))

    costs2 = compute_cost(
        predictions=predictions2,
        target=target,
        alpha=cost_alpha,
        beta=cost_beta
    )

    costs2[np.isnan(costs2) | np.isinf(costs2)] = cost_constant

    del predictions2
    gc.collect()

    inxs2 = []
    for i in range(len(costs2)):
        if costs2[i] < mean_cost:
            low_cost_agents.append(high_cost_agents[i])
            low_costs.append(costs2[i])
            inxs2.append(i)

    for inx in sorted(inxs2, reverse=True):
        del high_cost_agents[inx]

    del costs2, inxs2
    gc.collect()

    pop_size = int(population_size - len(low_cost_agents))

    if pop_size > 0:
        initialized_agents = population_initialization(
            population_size=pop_size,
            agent_shape=(z, y, x),
            species_parameters=species_parameters,
            max_sim_epochs=simulation_parameters["max_simulation_epoch"],
            sim_stop_time=simulation_parameters["simulation_stop_time"],
            time_step=simulation_parameters["time_step"],
            fixed_shape=fixed_agent_shape,
            low_costs=low_cost_agents
        )


        next_generation = low_cost_agents + initialized_agents
    else:
        next_generation = low_cost_agents

    return next_generation, costs, mean_cost
