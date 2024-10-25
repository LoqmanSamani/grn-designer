import numpy as np




def population_initialization(
        population_size, agent_shape, species_parameters, complex_parameters,
        num_species, num_complex, max_sim_epochs, sim_stop_time, time_step, fixed_agent_shape,
        init_species, init_complex
):

    init_shape = int((init_species*2) + (init_complex * 2) + 1)
    init_pair_start = int(init_species * 2)
    init_pair_stop = int(init_pair_start + (init_complex * 2))
    pair_start = int(num_species * 2)
    pair_stop = int(pair_start + (num_complex * 2))

    if fixed_agent_shape:
        population = [np.zeros(shape=agent_shape, dtype=np.float32) for _ in range(population_size)]

        for ag in population:
            ag[-1, -1, :5] = [num_species, num_complex, max_sim_epochs, sim_stop_time, time_step]
            
            for i in range(0, num_species * 2, 2):
                ag[-1, i, :3] = species_parameters[int(i // 2)]
                ag[i + 1, :, :] = np.random.rand(agent_shape[1], agent_shape[2]) * 0.5

            for i in range(pair_start + 1, pair_stop + 1, 2):
                ag[i, 0, :2] = complex_parameters[int((i - (pair_start + 1)) // 2)][0]
                ag[i, 1, :4] = complex_parameters[int((i - (pair_start + 1)) // 2)][1]

    else:
        population = [np.zeros(shape=(init_shape, agent_shape[1], agent_shape[2]), dtype=np.float32) for _ in range(population_size)]
        for ag in population:
            ag[-1, -1, :5] = [init_species, init_complex, max_sim_epochs, sim_stop_time, time_step]
            for i in range(0, init_species * 2, 2):
                ag[-1, i, :3] = species_parameters[int(i // 2)]
                ag[i + 1, :, :] = np.random.rand(agent_shape[1], agent_shape[2]) * 0.5

            for i in range(init_pair_start + 1, init_pair_stop + 1, 2):
                ag[i, 0, :2] = complex_parameters[int((i - (init_pair_start + 1)) // 2)][0]
                ag[i, 1, :4] = complex_parameters[int((i - (init_pair_start + 1)) // 2)][1]

    return population







def species_initialization(compartment_size, pairs):

    num_species = len(pairs) + 1
    num_matrices = num_species * 2
    init_matrix = np.zeros(shape=(num_matrices, compartment_size[0], compartment_size[1]), dtype=np.float32)
    init_matrix[1, :, :] = np.random.rand(compartment_size[0], compartment_size[1]) * .5

    for i in range(len(pairs)):
        m = np.zeros((2, compartment_size[0], compartment_size[1]))
        m[-1, 0, 0] = int(pairs[i][0])
        m[-1, 0, 1] = int(pairs[i][1])
        m[-1, 1, :4] = np.random.rand(4)
        init_matrix[i*2+2:i*2+4, :, :] = m

    return init_matrix


