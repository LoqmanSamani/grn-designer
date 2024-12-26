from cost import *
from crossover import *
from initialization import *
from mutation import *
from simulation import *
import gc



def evolutionary_optimization(
        population, agent_, target, population_size, cost, rates, bounds,
        mutation, crossover, num_elite_agents, parameters, cost_constant,
        fixed_agent_shape, num_init_genes, shape_init_condition, radius_range,
        conc_range
):
    """
    Executes the evolutionary optimization phase of the GRN-Designer algorithm.

    This function applies evolutionary principles, including mutation and crossover,
    to iteratively refine a population of agents toward minimizing the cost associated
    with deviations from a predefined target pattern. It incorporates selective pressure
    to retain elite agents and generates a next-generation population.

    Workflow:
        1. Simulation and Cost Calculation:
           - Simulates the behavior of each agent in the population.
           - Calculates the associated cost based on the target pattern and defined metrics.

        2. Selection:
           - Retains elite agents based on their costs.
           - Splits the population into low-cost and high-cost agents for tailored processing.

        3. Mutation and Crossover:
           - Applies mutation to low-cost agents, introducing genetic diversity.
           - Applies crossover to high-cost agents using filtered elite agents to generate new solutions.

        4. Re-Evaluation:
           - Simulates modified high-cost agents and integrates those meeting improvement criteria
             into the low-cost agent pool.

        5. Population Regeneration:
           - Replenishes the population by initializing new agents if the size of low-cost agents
             is insufficient to meet the required population size.

    Args:
        population (list): List of agents representing the current population.
        agent_ (np.ndarray): A representative agent used for population initialization and as a low-cost benchmark.
        target (np.ndarray): 2D array representing the target spatial pattern.
        population_size (int): Total desired size of the population after regeneration.
        cost (tuple): Weights and proportions for cost calculation, defined as (alpha, beta, pattern_proportion).
        rates (tuple): Mutation rates for different aspects of the agent, defined as:
                       (simulation, initial_condition, parameter, species_insertion,
                       connection_insertion, connection_deletion).
        bounds (tuple): Lower and upper bounds for simulation parameters, initial conditions,
                        and parameters, defined as:
                        (simulation_min, simulation_max, initial_condition_min,
                        initial_condition_max, parameter_min, parameter_max).
        mutation (tuple): Flags indicating mutation types to be applied, defined as:
                          (simulation, initial_condition, parameter, species_insertion,
                          connection_insertion, connection_deletion).
        crossover (tuple): Parameters controlling crossover behavior, defined as:
                           (crossover_alpha, simulation_crossover, initial_condition_crossover, parameter_crossover).
        num_elite_agents (int): Number of top-performing agents retained for crossover.
        parameters (tuple): Parameters for initializing new agents, defined as:
                            (species_parameters, simulation_parameters).
        cost_constant (float): Replacement value for NaN or Inf costs during computation.
        fixed_agent_shape (bool): Whether the agent shape is fixed, preventing changes in gene count.
        num_init_genes (int): Number of genes to initialize for newly generated agents.
        shape_init_condition (bool): Enables initialization of new agents with specific shapes (e.g., a circle).
        radius_range (tuple): Range of radii for shape-based initialization.
        conc_range (tuple): Range of concentrations for shape-based initialization.

    Returns:
        tuple:
            - next_generation (list): The next generation of agents after evolutionary optimization.
            - costs (np.ndarray): Array of computed costs for the current generation.
            - mean_cost (float): Average cost of the current generation.

    Key Features:
        - Supports dynamic mutation and crossover strategies to balance exploration and exploitation.
        - Retains elite agents to ensure continuity of the best-performing solutions.
        - Adapts to constraints through customizable mutation rates, bounds, and initialization parameters.
        - Efficiently handles NaN or Inf values in cost computation to maintain stability.

    Raises:
        RuntimeError: If any invalid configuration for mutation, crossover, or bounds is encountered.
    """

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
            connection_deletion_mutation=connection_deletion_mutation,
            shape_init_condition=shape_init_condition,
            radius_range=radius_range,
            conc_range=conc_range
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
                connection_deletion_mutation=connection_deletion_mutation,
                shape_init_condition=shape_init_condition,
                radius_range=radius_range,
                conc_range=conc_range
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
            low_costs=agent_,
            sim_opt=sim_mutation,
            param_opt=parameter_mutation,
            init_opt=initial_condition_mutation,
            num_genes=num_init_genes,
            sim_min=simulation_min,
            sim_max=simulation_max,
            param_min=parameter_min,
            param_max=parameter_max,
            init_min=initial_condition_min,
            init_max=initial_condition_max,
            shape_init_condition=shape_init_condition,
            radius_range=radius_range,
            conc_range=conc_range
        )


        next_generation = low_cost_agents + initialized_agents
    else:
        next_generation = low_cost_agents

    return next_generation, costs, mean_cost
