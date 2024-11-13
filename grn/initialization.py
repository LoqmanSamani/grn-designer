import numpy as np
import random



def population_initialization(
        population_size, agent_shape,
        species_parameters, max_sim_epochs,
        sim_stop_time, time_step,
        fixed_shape, low_costs
):

    if fixed_shape:

        population = []
        for i in range(population_size):
            ag_ = random.choice(low_costs)
            sh_ = ag_.shape
            ag = np.zeros(shape=sh_, dtype=np.float32)
            ag[-1, :, :] = ag_[-1, :, :]
            for j in range(1, int(ag_[-1, -1, 0]*2), 2):
                ag[j, :, :] = np.random.rand(ag_.shape[1], ag_.shape[2])
            population.append(ag)

    else:
        population = [np.zeros(shape=(3, agent_shape[1], agent_shape[2]), dtype=np.float32) for _ in range(population_size)]

        for ag in population:

            ag[-1, -1, :4] = [1, max_sim_epochs, sim_stop_time, time_step]
            ag[-1, 0, :3] = species_parameters
            ag[-1, 0, -1] = 0
            ag[1, :, :] = np.random.rand(agent_shape[1], agent_shape[2]) 


    return population



def reset_agent(agent):

    num_species = int(agent[-1, -1, 0])

    for i in range(0, num_species * 2, 2):
        agent[i, :, :] = 0.0

    return agent