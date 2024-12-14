import numpy as np
import random



def population_initialization(population_size, agent_shape, species_parameters,
                              max_sim_epochs, sim_stop_time, time_step, fixed_shape,
                              low_costs, sim_opt, param_opt, init_opt, num_genes, sim_min, sim_max):

    if fixed_shape:

        population = []
        for i in range(population_size):
            ag_ = random.choice(low_costs)
            sh_ = ag_.shape
            ag = np.zeros(shape=sh_, dtype=np.float32)
            ag[-1, :, :] = ag_[-1, :, :]
            if init_opt:
                for j in range(1, int(ag_[-1, -1, 0]*2), 2):
                    ag[j, :, :] = np.random.rand(ag_.shape[1], ag_.shape[2])
            else:
                for j in range(1, int(ag_[-1, -1, 0] * 2), 2):
                    ag[j, :, :] = ag_[j,:, :]

            population.append(ag)

    else:
        population = [np.zeros(shape=(num_genes, agent_shape[1], agent_shape[2]), dtype=np.float32) for _ in range(population_size)]

        for ag in population:
            if sim_opt:
                sim_dur = np.random.uniform(low=sim_min[0], high=sim_max[0])
                sim_dt = np.random.uniform(low=sim_min[1], high=sim_max[1])
                ag[-1, -1, :4] = [num_genes, max_sim_epochs, sim_dur, sim_dt]
            else:
                ag[-1, -1, :4] = [num_genes, max_sim_epochs, sim_stop_time, time_step]
            if param_opt:
                if num_genes > 1:
                    for i in range(0, num_genes*2, 2):
                        ag[-1, i, :6] = np.random.rand(6)
                        ag[-1, i, -1] = 1
                        inter_type = np.random.choice([0, 1])
                        ag[-1, i+1, -1] = inter_type
                        ag[i+1, :, :] = np.random.rand(agent_shape[1], agent_shape[2])
                else:
                    ag[-1, 0, :3] = np.random.rand(3)
                    ag[-1, 0, -1] = 0
                    ag[1, :, :] = np.random.rand(agent_shape[1], agent_shape[2])
            else:
                if num_genes > 1:
                    for i in range(0, num_genes*2, 2):
                        ag[-1, i, :6] = species_parameters
                        ag[-1, i, -1] = 1
                        inter_type = np.random.choice([0, 1])
                        ag[-1, i+1, -1] = inter_type
                        ag[i+1, :, :] = np.random.rand(agent_shape[1], agent_shape[2])
                else:
                    ag[-1, 0, :3] = species_parameters
                    ag[-1, 0, -1] = 0
                    ag[1, :, :] = np.random.rand(agent_shape[1], agent_shape[2])

    return population



def reset_agent(agent):

    num_species = int(agent[-1, -1, 0])

    for i in range(0, num_species * 2, 2):
        agent[i, :, :] = 0.0

    return agent