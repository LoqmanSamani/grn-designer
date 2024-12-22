import numpy as np
import random



def population_initialization(population_size, agent_shape, species_parameters,
                              max_sim_epochs, sim_stop_time, time_step, fixed_shape,
                              low_costs, sim_opt, param_opt, init_opt, num_genes, sim_min,
                              sim_max, param_min, param_max, init_min, init_max,
                              shape_init_condition, radius_range, conc_range):
    """
    Initializes a population of agents for evolutionary optimization.

    This function generates a list of agents, each represented as a 3D matrix, where
    the structure and parameters of the agents are defined based on the input settings.
    Agents can either be initialized with fixed shapes derived from a low-cost benchmark
    agent or created dynamically with randomized configurations.

    Args:
        population_size (int): Number of agents in the population.
        agent_shape (tuple): Shape of the agent (channels, height, width).
        species_parameters (list): List of species-specific parameters.
        max_sim_epochs (int): Maximum simulation epochs for agents.
        sim_stop_time (float): Default stopping time for simulations.
        time_step (float): Default time step for simulations.
        fixed_shape (bool): If True, agents are initialized with a fixed structure
                            derived from the low-cost benchmark.
        low_costs (list): List of low-cost agents used for benchmarking and initialization.
        sim_opt (bool): If True, simulation parameters (duration and time step) are randomized.
        param_opt (bool): If True, gene and interaction parameters are randomized.
        init_opt (bool): If True, initial conditions are randomized.
        num_genes (int): Number of genes for initializing agents.
        sim_min (tuple): Minimum values for simulation duration and time step.
        sim_max (tuple): Maximum values for simulation duration and time step.
        param_min (tuple): Minimum values for gene and interaction parameters.
        param_max (tuple): Maximum values for gene and interaction parameters.
        init_min (float): Minimum value for initializing agent conditions.
        init_max (float): Maximum value for initializing agent conditions.
        shape_init_condition (bool): If True, initial conditions are set with specific shapes (e.g., circles).
        radius_range (tuple): Range of radii for circle-based initial conditions.
        conc_range (tuple): Range of concentration values for circle-based initial conditions.

    Returns:
        list: A list of initialized agents, where each agent is a 3D numpy array.
    """

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



def reset_agent(agent):
    """
    Resets an agent by clearing the interaction matrix for all species.

    This function zeroes out the interaction values for all species within an agent,
    effectively resetting it to a baseline state while preserving its structure.

    Args:
        agent (np.ndarray): The agent to reset, represented as a 3D matrix.

    Returns:
        np.ndarray: The reset agent with cleared interaction matrices.
    """

    num_species = int(agent[-1, -1, 0])

    for i in range(0, num_species * 2, 2):
        agent[i, :, :] = 0.0

    return agent