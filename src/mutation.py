import numpy as np
import random


def apply_mutation(
        population,
        agent,
        sim_mutation_rate,
        initial_condition_mutation_rate,
        parameter_mutation_rate,
        species_insertion_mutation_rate,
        connection_insertion_mutation_rate,
        connection_deletion_mutation_rate,
        simulation_min,
        simulation_max,
        initial_condition_min,
        initial_condition_max,
        parameter_min,
        parameter_max,
        simulation_mutation,
        initial_condition_mutation,
        parameter_mutation,
        species_insertion_mutation,
        connection_insertion_mutation,
        connection_deletion_mutation,
        shape_init_condition,
        radius_range,
        conc_range
):
    """
    Applies various mutations to an agent to introduce genetic diversity.

    This function applies multiple mutation types, including simulation parameter
    mutation, initial condition mutation, parameter mutation, species insertion,
    connection insertion, and connection deletion. Each mutation type is controlled
    by a corresponding mutation rate and other parameters.

    Args:
        population (list): List of agents representing the population.
        agent (np.ndarray): The agent to which mutations are applied.
        sim_mutation_rate (float): Rate for mutating simulation parameters.
        initial_condition_mutation_rate (float): Rate for mutating initial conditions.
        parameter_mutation_rate (float): Rate for mutating gene and interaction parameters.
        species_insertion_mutation_rate (float): Rate for inserting new species into the agent.
        connection_insertion_mutation_rate (float): Rate for adding new connections between species.
        connection_deletion_mutation_rate (float): Rate for removing existing connections between species.
        simulation_min (tuple): Minimum values for simulation parameters (duration and time step).
        simulation_max (tuple): Maximum values for simulation parameters (duration and time step).
        initial_condition_min (float): Minimum value for initial conditions.
        initial_condition_max (float): Maximum value for initial conditions.
        parameter_min (tuple): Minimum values for gene and interaction parameters.
        parameter_max (tuple): Maximum values for gene and interaction parameters.
        simulation_mutation (bool): Enables simulation parameter mutation.
        initial_condition_mutation (bool): Enables initial condition mutation.
        parameter_mutation (bool): Enables parameter mutation.
        species_insertion_mutation (bool): Enables species insertion mutation.
        connection_insertion_mutation (bool): Enables connection insertion mutation.
        connection_deletion_mutation (bool): Enables connection deletion mutation.
        shape_init_condition (bool): If True, initializes new species with specific shapes (e.g., circles).
        radius_range (tuple): Range of radii for circle-based initialization.
        conc_range (tuple): Range of concentration values for circle-based initialization.

    Returns:
        np.ndarray: The mutated agent.
    """

    if simulation_mutation:
        agent = apply_simulation_parameters_mutation(
            population=population,
            agent=agent,
            mutation_rate=sim_mutation_rate,
            min_vals=simulation_min,
            max_vals=simulation_max
        )

    if initial_condition_mutation:
        agent = apply_compartment_mutation(
            population=population,
            agent=agent,
            mutation_rate=initial_condition_mutation_rate,
            min_val=initial_condition_min,
            max_val=initial_condition_max
        )


    if parameter_mutation:
        agent = apply_parameters_mutation(
            population=population,
            agent=agent,
            mutation_rate=parameter_mutation_rate,
            min_val=parameter_min,
            max_val=parameter_max
        )

    if species_insertion_mutation:
        agent = apply_species_insertion_mutation(
            agent=agent,
            mutation_rate=species_insertion_mutation_rate,
            shape_init_condition=shape_init_condition,
            radius_range=radius_range,
            conc_range=conc_range
        )

    if connection_insertion_mutation:
        agent = apply_connection_insertion_mutation(
            agent=agent,
            mutation_rate=connection_insertion_mutation_rate
        )

    if connection_deletion_mutation:
        apply_connection_deletion_mutation(
            agent=agent,
            mutation_rate=connection_deletion_mutation_rate
        )

    if agent[-1, -1, 2] / agent[-1, -1, 3] > 200 or agent[-1, -1, 2] / agent[-1, -1, 3] < 70:
        agent[-1, -1, 2] = 20
        agent[-1, -1, 3] = 0.2


    return agent




def apply_simulation_parameters_mutation(
        population,
        agent,
        mutation_rate,
        min_vals,
        max_vals,
        F=0.8
):
    """
    Mutates the simulation parameters of an agent.

    This mutation modifies the simulation duration and time step of an agent by blending
    values from other randomly selected agents in the population.

    Args:
        population (list): List of agents in the population.
        agent (np.ndarray): The agent to mutate.
        mutation_rate (float): Probability of applying mutation to each parameter.
        min_vals (tuple): Minimum allowable values for simulation parameters.
        max_vals (tuple): Maximum allowable values for simulation parameters.
        F (float, optional): Scaling factor for mutation. Defaults to 0.8.

    Returns:
        np.ndarray: The agent with mutated simulation parameters.
    """

    pop = [ag for ag in population if ag.shape[0] == agent.shape[0]]
    if len(pop) >= 3:
        agent1, agent2, agent3 = random.sample(pop, k=3)

        mutation_mask = np.random.rand(2) < mutation_rate
        idx = 2
        for i in range(2):
            if mutation_mask[i]:
                agent[-1, -1, idx] = agent1[-1, -1, idx] + F*(agent2[-1, -1, idx] - agent3[-1, -1, idx])
                agent[-1, -1, idx] = max(min_vals[i], min(max_vals[i], agent[-1, -1, idx]))
            idx += 1

    return agent


def apply_compartment_mutation(
        population,
        agent,
        mutation_rate,
        min_val,
        max_val,
        F=0.8
):
    """
    Mutates the initial conditions of an agent's compartments.

    This mutation alters the compartment values of an agent using differential mutation
    by combining values from other randomly selected agents in the population.

    Args:
        population (list): List of agents in the population.
        agent (np.ndarray): The agent to mutate.
        mutation_rate (float): Probability of applying mutation to each compartment value.
        min_val (float): Minimum allowable value for compartments.
        max_val (float): Maximum allowable value for compartments.
        F (float, optional): Scaling factor for mutation. Defaults to 0.8.

    Returns:
        np.ndarray: The agent with mutated compartment values.
    """

    pop = [ag for ag in population if ag.shape[0] == agent.shape[0]]

    if len(pop) >= 3:
        agent1, agent2, agent3 = random.sample(pop, k=3)
        num_species = int(agent[-1, -1, 0])
        z, y, x = agent.shape

        for i in range(1, num_species * 2, 2):
            mutation_mask = np.random.rand(y, x) < mutation_rate
            mutant_section = agent1[i, :, :] + F * (agent2[i, :, :] - agent3[i, :, :])
            agent[i, :, :] = np.where(mutation_mask, mutant_section, agent[i, :, :])

            agent[i, :, :] = np.maximum(agent[i, :, :], min_val)
            agent[i, :, :] = np.minimum(agent[i, :, :], max_val)

    return agent



def apply_parameters_mutation(
        population,
        agent,
        mutation_rate,
        min_val,
        max_val,
        F=0.8
):
    """
    Mutates the gene and interaction parameters of an agent.

    This mutation modifies the parameter values of genes and their interactions by
    blending values from other randomly selected agents in the population.

    Args:
        population (list): List of agents in the population.
        agent (np.ndarray): The agent to mutate.
        mutation_rate (float): Probability of applying mutation to each parameter.
        min_val (tuple): Minimum allowable values for gene and interaction parameters.
        max_val (tuple): Maximum allowable values for gene and interaction parameters.
        F (float, optional): Scaling factor for mutation. Defaults to 0.8.

    Returns:
        np.ndarray: The agent with mutated gene and interaction parameters.
    """

    pop = [ag for ag in population if ag.shape[0] == agent.shape[0]]

    if len(pop) >= 3:

        agent1, agent2, agent3 = random.sample(pop, k=3)
        num_species = int(agent[-1, -1, 0])

        for i in range(0, num_species * 2, 2):
            #constant_value = np.random.rand()
            num_param = int(int(agent[-1, i, -1]*3) + 3)
            mutation_mask = np.random.rand(num_param) < mutation_rate
            mutated_values = agent1[-1, i, :num_param] + F * (agent2[-1, i, :num_param] - agent3[-1, i, :num_param])
            #mutated_values = np.where(mutated_values < 0.010, constant_value, mutated_values)
            agent[-1, i, :3] = np.where(mutation_mask[:3], mutated_values[:3], agent[-1, i, :3])
            agent[-1, i, :3] = np.clip(agent[-1, i, :3], min_val[0], max_val[0])
            agent[-1, i, 3:num_param] = np.where(mutation_mask[3:], mutated_values[3:], agent[-1, i, 3:num_param])
            agent[-1, i, 3:num_param] = np.clip(agent[-1, i, 3:num_param], min_val[-1], max_val[-1])

    return agent


def apply_species_insertion_mutation(agent, mutation_rate, shape_init_condition, radius_range, conc_range):
    """
    Inserts a new species into an agent.

    This mutation adds a new species to an agent, initializing its compartments and
    interactions with either random values or specific shapes, depending on the
    initialization settings.

    Args:
        agent (np.ndarray): The agent to mutate.
        mutation_rate (float): Probability of applying species insertion.
        shape_init_condition (bool): If True, initializes compartments with specific shapes (e.g., circles).
        radius_range (tuple): Range of radii for circle-based initialization.
        conc_range (tuple): Range of concentration values for circle-based initialization.

    Returns:
        np.ndarray: The agent with an inserted species, or the original agent if no mutation occurred.
    """

    num_species = int(agent[-1, -1, 0])
    sps_ = np.array([int(i*2) for i in range(num_species)])
    con_type = np.array([0, 1])
    z, y, x = agent.shape

    if np.random.rand() < mutation_rate:

        new_agent = np.zeros(shape=(z+2, y, x), dtype=np.float32)
        new_species = np.zeros(shape=(2, y, x), dtype=np.float32)
        if shape_init_condition:
            new_species[1, :, :] = create_circle(
                matrix=new_species[1, :, :],
                value_range=conc_range,
                radius_range=radius_range
            )
        else:
            new_species[1, :, :] = np.random.rand(y, x)

        affected = int(np.random.choice(sps_))
        rel_type = int(np.random.choice(con_type))

        nn = np.zeros(shape=(y, x), dtype=np.float32)
        for i in range(y):
            for j in range(x):
                nn[i, j] = agent[-1, i, j]

        nn[int(num_species * 2), :6] = np.random.rand(6)
        nn[int(num_species * 2), -1] = 1
        nn[int((num_species * 2) + 1), 0] = int(affected)
        nn[int((num_species * 2) + 1), -1] = int(rel_type)
        nn[-1, 0] = int(nn[-1, 0] + 1)

        new_agent[:z-1, :, :] = agent[:-1, :, :]
        new_agent[z-1:-1, :, :] = new_species
        new_agent[-1, :, :] = nn

        return new_agent
    else:
        return agent






def apply_connection_insertion_mutation(agent, mutation_rate):
    """
    Inserts a new connection between species in an agent.

    This mutation adds a new interaction between two species in the agent, initializing
    the connection parameters with random values.

    Args:
        agent (np.ndarray): The agent to mutate.
        mutation_rate (float): Probability of applying connection insertion.

    Returns:
        np.ndarray: The agent with an inserted connection, or the original agent if no mutation occurred.
    """

    num_species = int(agent[-1, -1, 0])
    con_type = np.array([0, 1])
    sps_ = [int(i*2) for i in range(1, num_species)]

    if len(sps_) > 1:
        random_sp = int(random.choice(sps_))
        sps_.remove(random_sp)
        sps_.append(0)
        sps_ = np.array(sps_)


        if np.random.rand() < mutation_rate:
            affected = int(np.random.choice(sps_))
            rel_type = int(np.random.choice(con_type))
            k = int(agent[-1, random_sp, -1])
            connections = list(agent[-1, random_sp+1, :k])
            if affected not in connections:
                hh = int(3 + (agent[-1, random_sp, -1]*3))
                agent[-1, random_sp, -1] = int(agent[-1, random_sp, -1]+1)
                agent[-1, random_sp, hh:hh+3] = np.random.rand(3)
                agent[-1, random_sp+1, int(agent[-1, random_sp, -1]-1)] = affected
                agent[-1, random_sp+1, -int(agent[-1, random_sp, -1])] = rel_type

    return agent






def apply_connection_deletion_mutation(agent, mutation_rate):
    """
    Deletes an existing connection between species in an agent.

    This mutation removes an interaction between two species in the agent by
    updating the connection and interaction matrices.

    Args:
        agent (np.ndarray): The agent to mutate.
        mutation_rate (float): Probability of applying connection deletion.

    Returns:
        np.ndarray: The agent with a deleted connection, or the original agent if no mutation occurred.
    """

    num_species = int(agent[-1, -1, 0])
    sps_ = np.array([i * 2 for i in range(1, num_species)])

    if np.random.rand() < mutation_rate:
        if len(sps_) > 2:  #
            species = int(np.random.choice(sps_))
            if agent[-1, species, -1] > 1:
                agent[-1, species, -1] = int(agent[-1, species, -1] - 1)
                rates = list(agent[-1, species, 3:3 + int((agent[-1, species, -1]*3) + 3)])
                con_ind = list(agent[-1, species + 1, :int(agent[-1, species, -1] + 1)])
                con_ = list(agent[-1, species + 1, -int(agent[-1, species, -1] + 1):])
                inx, d = random.choice(list(enumerate(con_ind)))

                del con_[-(inx + 1)]
                con_.insert(0, 0)
                hhh = inx*3
                del rates[hhh: hhh+3]

                rates.extend([0.0, 0.0, 0.0])
                con_ind.remove(d)
                con_ind.append(0)
                agent[-1, species + 1, :int(agent[-1, species, -1] + 1)] = con_ind
                agent[-1, species + 1, -int(agent[-1, species, -1] + 1):] = con_
                agent[-1, species, 3:3 + len(rates)] = rates

    return agent


def create_circle(matrix, value_range, radius_range):
    """
    Creates a circular region in a matrix with a specified value and radius.

    This function modifies a matrix by setting values within a circular region
    to a randomly chosen value from a specified range. The circle's radius and
    center are also randomly chosen within defined bounds.

    Args:
        matrix (np.ndarray): The 2D array to modify.
        value_range (tuple): Range (min, max) of values to assign within the circle.
        radius_range (tuple): Range (min, max) of circle radii.

    Returns:
        np.ndarray: The modified matrix with a circular region filled with the specified value.
    """

    value = np.random.uniform(low=value_range[0], high=value_range[1])
    radius = np.random.uniform(low=radius_range[0], high=radius_range[1])
    center = np.random.choice(a=np.arange(matrix.shape[0]), size=2)
    
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if (i - center[0])**2 + (j - center[1])**2 <= radius**2:
                matrix[i, j] = value
                
    return matrix

