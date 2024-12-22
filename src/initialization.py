import numpy as np
import random



def population_initialization(population_size, agent_shape, species_parameters,
                              max_sim_epochs, sim_stop_time, time_step, fixed_shape,
                              low_costs, sim_opt, param_opt, init_opt, num_genes, sim_min,
                              sim_max, param_min, param_max, init_min, init_max,
                              shape_init_condition, radius_range, conc_range):

    if fixed_shape:

        population = []
        for i in range(population_size):
            ag_ = low_costs[0]
            sh_ = ag_.shape
            ag = np.zeros(shape=sh_, dtype=np.float32)
            ag[-1, :, :] = ag_[-1, :, :]
            
            if init_opt:
                for j in range(1, int(ag_[-1, -1, 0]*2), 2):
                    ag[j, :, :] = np.random.uniform(low=init_min, high=init_max, size=(sh_[1], sh_[2])) 
                    
            elif shape_init_condition and not init_opt:
                for j in range(1, int(ag_[-1, -1, 0]*2), 2):
                    ag[j, :, :] = create_circle(
                        matrix=ag[j, :, :],
                        value_range=conc_range,
                        radius_range=radius_range
                    )
                    
            else:
                for j in range(1, int(ag_[-1, -1, 0]*2), 2):
                    ag[j, :, :] = ag_[j,:, :]
                    
            if param_opt:
                for j in range(2, int(ag_[-1, -1, 0]*2), 2):
                    num_iter = int(ag_[-1, j, -1]*3)
                    ag[-1, j, :3] = np.random.uniform(low=param_min[0], high=param_max[0], size=3)
                    ag[-1, j, 3:num_iter+3] = np.random.uniform(low=param_min[1], high=param_max[1], size=num_iter)
                    #ag[-1, j, -1] = ag_[-1, j, -1]
                    #ag[-1, j+1, :num_iter] = ag_[-1, j+1, :num_iter]
                    #ag[-1, j+1, -num_iter:] = ag_[-1, j+1, -num_iter:]
                    
            if sim_opt:
                sim_dur = np.random.uniform(low=sim_min[0], high=sim_max[0])
                sim_dt = np.random.uniform(low=sim_min[1], high=sim_max[1])
                ag[-1, -1, 2:4] = [sim_dur, sim_dt]

            population.append(ag)

    else:
        population = [np.zeros(shape=(int((num_genes*2)+1), agent_shape[1], agent_shape[2]), dtype=np.float32) for _ in range(population_size)]

        for ag in population:
            if sim_opt:
                sim_dur = np.random.uniform(low=sim_min[0], high=sim_max[0])
                sim_dt = np.random.uniform(low=sim_min[1], high=sim_max[1])
                ag[-1, -1, :4] = [num_genes, max_sim_epochs, sim_dur, sim_dt]
            else:
                ag[-1, -1, :4] = [num_genes, max_sim_epochs, sim_stop_time, time_step]
            
            if num_genes > 1:
                ag[-1, 0, :3] = np.random.uniform(low=param_min[0], high=param_max[0], size=3)
                for i in range(2, num_genes*2, 2):
                    ag[-1, i, :3] = np.random.uniform(low=param_min[0], high=param_max[0], size=3)
                    ag[-1, i, 3:6] = np.random.uniform(low=param_min[1], high=param_max[1], size=3)
                    ag[-1, i, -1] = 1
                    ag[-1, i+1, 0] = 0
                    inter_type = int(np.random.choice([0, 1]))
                    ag[-1, i+1, -1] = inter_type
                        
                
                if shape_init_condition and not init_opt:
                    for j in range(1, num_genes*2, 2):
                        ag[j, :, :] = create_circle(
                            matrix=ag[j, :, :],
                            value_range=conc_range,
                            radius_range=radius_range
                        )
                else:
                    for j in range(1, num_genes*2, 2):
                        ag[j, :, :] = np.random.uniform(low=init_min, high=init_max, size=(agent_shape[1], agent_shape[2]))
                    
            else:
                ag[-1, 0, :3] = np.random.uniform(low=param_min[0], high=param_max[0], size=3)
                ag[-1, 0, -1] = 0
                
                if shape_init_condition and not init_opt:
                    ag[1, :, :] = create_circle(
                        matrix=ag[1, :, :],
                        value_range=conc_range,
                        radius_range=radius_range
                    )
                else:
                    ag[1, :, :] = np.random.uniform(low=init_min, high=init_max, size=(agent_shape[1], agent_shape[2]))
            
    return population



def create_circle(matrix, value_range, radius_range):

    value = np.random.uniform(low=value_range[0], high=value_range[1])
    radius = np.random.uniform(low=radius_range[0], high=radius_range[1])
    center = np.random.choice(a=np.arange(matrix.shape[0]), size=2)
    
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if (i - center[0])**2 + (j - center[1])**2 <= radius**2:
                matrix[i, j] = value
                
    return matrix



def reset_agent(agent):

    num_species = int(agent[-1, -1, 0])

    for i in range(0, num_species * 2, 2):
        agent[i, :, :] = 0.0

    return agent