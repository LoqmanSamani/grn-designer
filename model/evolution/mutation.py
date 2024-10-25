import itertools
import numpy as np
from initialization import species_initialization
import random


def apply_mutation(
        population,
        agent,
        sim_mutation_rate,
        initial_condition_mutation_rate,
        parameter_mutation_rate,
        insertion_mutation_rate,
        deletion_mutation_rate,
        simulation_min,
        simulation_max,
        initial_condition_min,
        initial_condition_max,
        parameter_min,
        parameter_max,
        simulation_mutation,
        initial_condition_mutation,
        parameter_mutation,
        insertion_mutation,
        deletion_mutation
):

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

    if insertion_mutation:
        agent = apply_species_insertion_mutation(
            agent=agent,
            mutation_rate=insertion_mutation_rate
        )

    if deletion_mutation and agent.shape[0] > 3:
        agent = apply_species_deletion_mutation(
            agent=agent,
            mutation_rate=deletion_mutation_rate
        )

    if agent[-1, -1, 3] / agent[-1, -1, 4] > 200 or agent[-1, -1, 3] / agent[-1, -1, 4] < 70:
        agent[-1, -1, 3] = 20
        agent[-1, -1, 4] = 0.2


    return agent




def apply_simulation_parameters_mutation(
        population,
        agent,
        mutation_rate,
        min_vals,
        max_vals,
        F=0.8
):
    pop = [ag for ag in population if ag.shape[0] == agent.shape[0]]
    if len(pop) >= 3:
        agent1, agent2, agent3 = random.sample(pop, k=3)
        # if population[0].shape[0] == agent.shape[0]:
        # agent1 = population[0]
        mutation_mask = np.random.rand(2) < mutation_rate
        idx = 3
        for i in range(2):
            agent[-1, -1, idx] = agent1[-1, -1, idx] + F*(agent2[-1, -1, idx] - agent3[-1, -1, idx]) * mutation_mask[i]
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


    pop = [ag for ag in population if ag.shape[0] == agent.shape[0]]
    if len(pop) >= 3:
        agent1, agent2, agent3 = random.sample(pop, k=3)
        #if population[0].shape[0] == agent.shape[0]:
            #agent1 = population[0]
            
        num_species = int(agent[-1, -1, 0])
        z, y, x = agent.shape


        for i in range(1, num_species * 2, 2):
            mutation_mask = np.random.rand(y, x) < mutation_rate
            agent[i, :, :] = agent1[i, :, :] + F * (agent2[i, :, :] - agent3[i, :, :]) * mutation_mask
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
    pop = [ag for ag in population if ag.shape[0] == agent.shape[0]]
    if len(pop) >= 3:
        agent1, agent2, agent3 = random.sample(pop, k=3)
        #if population[0].shape[0] == agent.shape[0]:
            #agent1 = population[0]

        num_species = int(agent[-1, -1, 0])
        num_pairs = int(agent[-1, -1, 1])
        pair_start = num_species * 2
        pair_stop = pair_start + (num_pairs * 2)

        for i in range(0, num_species * 2, 2):

            mutation_mask = np.random.rand(3) < mutation_rate
            agent[-1, i, :3] = agent1[-1, i, :3] + F * (agent2[-1, i, :3] - agent3[-1, i, :3]) * mutation_mask

            agent[-1, i, :3] = np.minimum(agent[-1, i, :3], min_val)
            agent[-1, i, :3] = np.maximum(agent[-1, i, :3], max_val)

        for i in range(pair_start + 1, pair_stop, 2):

            mutation_mask = np.random.rand(4) < mutation_rate
            agent[i, 1, :4] = agent1[i, 1, :4] + F * (agent2[i, 1, :4]- agent3[i, 1, :4]) * mutation_mask
            agent[i, 1, :4] = np.minimum(agent[i, 1, :4], min_val)
            agent[i, 1, :4] = np.maximum(agent[i, 1, :4], max_val)

    return agent




def apply_species_insertion_mutation(agent, mutation_rate):

    num_species = int(agent[-1, -1, 0])
    num_pairs = int(agent[-1, -1, 1])
    z, y, x = agent.shape

    if np.random.rand() < mutation_rate:
        pairs = pair_finding(
            num_species=num_species
        )
        init_matrix = species_initialization(
            compartment_size=(y, x),
            pairs=pairs
        )
        agent = species_combine(
            agent=agent,
            init_matrix=init_matrix,
            num_species=num_species,
            num_pairs=num_pairs
        )

    return agent





def apply_species_deletion_mutation(agent, mutation_rate):

    num_species = int(agent[-1, -1, 0])

    if np.random.rand() < mutation_rate and num_species > 1:
        deleted_species = int(np.random.choice(np.arange(2, num_species+1)))

        agent = species_deletion(
            agent=agent,
            deleted_species=deleted_species
        )

    return agent





def pair_finding(num_species):

    last = num_species + 1
    species = [i for i in range(1, num_species + 2, 1)]
    pairs = list(itertools.combinations(species, 2))

    related_pairs = [pair for pair in pairs if last in pair]
    pair_indices = [((pair[0] - 1) * 2, (pair[1]-1)*2) for pair in related_pairs]

    return pair_indices




def species_combine(agent, init_matrix, num_species, num_pairs):

    z, y, x = agent.shape
    z1 = z + init_matrix.shape[0]

    updated_agent = np.zeros(shape=(z1, y, x), dtype=np.float32)
    updated_agent[:num_species * 2, :, :] = agent[:num_species * 2, :, :]
    updated_agent[num_species * 2:num_species * 2 + init_matrix.shape[0], :, :] = init_matrix
    updated_agent[num_species * 2 + init_matrix.shape[0]:, :, :] = agent[num_species * 2:, :, :]
    updated_agent[-1, -1, 0] = int(num_species + 1)
    updated_agent[-1, -1, 1] = int(num_pairs + ((init_matrix.shape[0] - 2) / 2))
    updated_agent[-1, num_species * 2, :3] = np.random.rand(3)

    return updated_agent




def species_deletion(agent, deleted_species):


    num_species = int(agent[-1, -1, 0])
    num_pairs = int(agent[-1, -1, 1])
    pair_start = int((num_species * 2) + 1)
    pair_stop = int(pair_start + (num_pairs * 2))

    delete_indices = [(deleted_species-1)*2, ((deleted_species-1)*2)+1]

    for i in range(pair_start, pair_stop, 2):
        if int((agent[i, 0, 0] / 2) + 1) == deleted_species or int((agent[i, 0, 1] / 2) + 1) == deleted_species:
            delete_indices.extend([i - 1, i])

    updated_agent = np.delete(agent, delete_indices, axis=0)

    updated_agent[-1, -1, 0] = num_species - 1
    updated_agent[-1, -1, 1] = num_pairs - len(delete_indices) // 2 + 1

    return updated_agent
