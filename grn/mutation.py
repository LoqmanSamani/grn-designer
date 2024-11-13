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
        connection_deletion_mutation
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

    if species_insertion_mutation:
        agent = apply_species_insertion_mutation(
            agent=agent,
            mutation_rate=species_insertion_mutation_rate
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

    pop = [ag for ag in population if ag.shape[0] == agent.shape[0]]

    if len(pop) >= 3:

        agent1, agent2, agent3 = random.sample(pop, k=3)
        num_species = int(agent[-1, -1, 0])

        for i in range(0, num_species * 2, 2):
            constant_value = np.random.rand() * .2
            num_param = int(agent[-1, i, -1] + 3)
            mutation_mask = np.random.rand(num_param) < mutation_rate
            mutated_values = agent1[-1, i, :num_param] + F * (agent2[-1, i, :num_param] - agent3[-1, i, :num_param])
            mutated_values = np.where(mutated_values < .03, constant_value, mutated_values)
            agent[-1, i, :num_param] = np.where(mutation_mask, mutated_values, agent[-1, i, :num_param])
            agent[-1, i, :num_param] = np.clip(agent[-1, i, :num_param], min_val, max_val)

    return agent


def apply_species_insertion_mutation(agent, mutation_rate):
    num_species = int(agent[-1, -1, 0])
    sps_ = np.array([int(i*2) for i in range(num_species)])
    con_type = np.array([0, 1])
    z, y, x = agent.shape

    if np.random.rand() < mutation_rate:

        new_agent = np.zeros((z+2, y, x), dtype=np.float32)
        new_species = np.zeros((2, y, x), dtype=np.float32)
        new_species[1, :, :] = np.random.rand(y, x)

        affected = int(np.random.choice(sps_))
        rel_type = int(np.random.choice(con_type))

        nn = np.zeros((y, x), dtype=np.float32)
        for i in range(y):
            for j in range(x):
                nn[i, j] = agent[-1, i, j]


        nn[int(num_species * 2), :4] = np.random.rand(4)
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
                agent[-1, random_sp, -1] = int(agent[-1, random_sp, -1]+1)
                agent[-1, random_sp, int(agent[-1, random_sp, -1]+2)] = np.random.rand()
                agent[-1, random_sp+1, int(agent[-1, random_sp, -1]-1)] = affected
                agent[-1, random_sp+1, -int(agent[-1, random_sp, -1])] = rel_type

    return agent






def apply_connection_deletion_mutation(agent, mutation_rate):
    num_species = int(agent[-1, -1, 0])
    sps_ = np.array([i * 2 for i in range(1, num_species)])

    if np.random.rand() < mutation_rate:
        if len(sps_) > 2:  #
            species = int(np.random.choice(sps_))
            if agent[-1, species, -1] > 1:
                agent[-1, species, -1] = int(agent[-1, species, -1] - 1)
                rates = list(agent[-1, species, 3:3 + int(agent[-1, species, -1] + 1)])
                con_ind = list(agent[-1, species + 1, :int(agent[-1, species, -1] + 1)])
                con_ = list(agent[-1, species + 1, -int(agent[-1, species, -1] + 1):])
                inx, d = random.choice(list(enumerate(con_ind)))

                del con_[-(inx + 1)]
                con_.insert(0, 0)
                del rates[inx]

                rates.append(0)
                con_ind.remove(d)
                con_ind.append(0)
                agent[-1, species + 1, :int(agent[-1, species, -1] + 1)] = con_ind
                agent[-1, species + 1, -int(agent[-1, species, -1] + 1):] = con_
                agent[-1, species, 3:3 + len(rates)] = rates

    return agent




