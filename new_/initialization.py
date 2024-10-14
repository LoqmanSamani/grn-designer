import numpy as np




def population_initialization(
        population_size, individual_shape, species_parameters, complex_parameters,
        num_species, num_pairs, max_sim_epochs, sim_stop_time, time_step, individual_fix_size,
        init_species, init_pairs
):
    """
    Initializes a population of individuals for an evolutionary simulation. Each individual represents
    a simulated entity in an evolutionary algorithm with specific species and complex (pairwise interactions)
    parameters. The initialization process defines species behavior and interaction rules for the simulation.

    Parameters:
        - population_size (int): The total number of individuals to initialize.
        - individual_shape (tuple of int): The 3D shape of each individual, represented as (z, y, x),
          where 'z' is the depth, and (y, x) are the 2D matrix dimensions.
        - species_parameters (list of lists): A list of lists, where each inner list contains parameters
          for a species (e.g., production rate, degradation rate, diffusion rate).
        - complex_parameters (list of tuples): A list where each entry is a tuple representing a pairwise
          inter action (complex) between species. The tuple's first element is a list of species involved,
          and the second element is a list of interaction rates (e.g., collision and dissociation rates).
        - num_species (int): The number of different species each individual contains.
        - num_pairs (int): The number of pairwise species interactions (complexes) each individual has.
        - max_sim_epochs (int): The maximum number of epochs or iterations the simulation will run.
        - sim_stop_time (float): The simulation stop time, defining when the simulation will end.
        - time_step (float): The time step for each iteration of the simulation.
        - individual_fix_size (bool): A flag that determines whether all individuals should have a fixed
          size and structure based on `individual_shape`.
          - If True: All individuals are initialized with the same shape and structure.
          - If False: Individuals are initialized with a smaller size and random components.
        - init_species (int): The number of species in the initial population for when `individual_fix_size` is False.
        - init_pairs (int): The number of species pairs (complexes) in the initial population when `individual_fix_size` is False.

    Returns:
        - population (list of np.ndarray): A list of initialized individuals, where each individual
          is represented as a 3D numpy array. Each array holds species parameters, pairwise complex parameters,
          and simulation parameters for the evolutionary process.

    Notes:
        - If `individual_fix_size` is True, each individual is initialized with `num_species` species
          and `num_pairs` complexes.
        - If False, individuals are initialized with a smaller number of species (`init_species`)
          and complexes (`init_pairs`).
    """
    init_shape = int((init_species*2) + (init_pairs*2) + 1)
    init_pair_start = int(init_species * 2)
    init_pair_stop = int(init_pair_start + (init_pairs * 2))
    pair_start = int(num_species * 2)
    pair_stop = int(pair_start + (num_pairs * 2))

    if individual_fix_size:
        population = [np.zeros(individual_shape) for _ in range(population_size)]

        for ind in population:
            ind[-1, -1, :5] = [num_species, num_pairs, max_sim_epochs, sim_stop_time, time_step]
            
            for i in range(0, len(species_parameters) * 2, 2):
                ind[-1, i, :3] = species_parameters[int(i // 2)]
                ind[i+1, :, :] = np.random.rand(individual_shape[1], individual_shape[2]) * .01

            for i in range(pair_start + 1, pair_stop + 1, 2):
                ind[i, 0, :2] = complex_parameters[int((i-(pair_start+1))//2)][0]
                ind[i, 1, :4] = complex_parameters[int((i-(pair_start+1))//2)][1]

    else:
        population = [np.zeros((init_shape, individual_shape[1], individual_shape[2])) for _ in range(population_size)]
        for ind in population:
            ind[-1, -1, :5] = [1, 0, max_sim_epochs, sim_stop_time, time_step]
            for i in range(0, init_species * 2, 2):
                ind[-1, i, :3] = species_parameters[int(i // 2)]
                ind[i + 1, :, :] = np.random.rand(individual_shape[1], individual_shape[2]) * .01

            for i in range(init_pair_start + 1, init_pair_stop + 1, 2):
                ind[i, 0, :2] = complex_parameters[int((i-(init_pair_start+1))//2)][0]
                ind[i, 1, :4] = complex_parameters[int((i-(init_pair_start+1))//2)][1]

    return population







def species_initialization(compartment_size, pairs):
    """
    Initializes the parameters for a new species and its complexes with existing species.

    This function generates an initialization matrix for a new species and the complexes that
    form between the new species and existing species. The matrix consists of compartments
    filled with zeros, except for specific locations that store the indices of the species pairs
    and some random initial parameters for each complex.

    Parameters:
        - compartment_size (tuple of int): The size of each compartment in the initialization matrix.
        - pairs (list of tuples): A list of species pairs, where each pair represents a complex
          formed between the new species and an existing species.

    Returns:
        - numpy.ndarray: A matrix containing initialized values for the new species and its complexes.
          Each complex occupies two compartments in the matrix, with the last row storing species pair indices
          and initial random parameter values.
    """

    num_species = len(pairs) + 1
    num_matrices = num_species * 2
    init_matrix = np.zeros((num_matrices, compartment_size[0], compartment_size[1]))

    for i in range(len(pairs)):
        m = np.zeros((2, compartment_size[0], compartment_size[1]))
        m[-1, 0, 0] = int(pairs[i][0])
        m[-1, 0, 1] = int(pairs[i][1])
        m[-1, 1, :4] = np.random.rand(4)
        init_matrix[i*2+2:i*2+4, :, :] = m

    return init_matrix







def reset_individual(individual):
    """
    Resets an individual's internal state by clearing species and pairwise interaction matrices.
    This can be used to prepare an individual for a new simulation run.

    Parameters:
        - individual (np.ndarray): The individual to reset, represented as a 3D numpy array.
                                       The array contains species and pairwise interaction matrices.

    Returns:
        - individual (np.ndarray): The modified individual with cleared species and complex matrices.

    Notes:
        - The species matrices are zeroed out by iterating over the number of species.
        - The complex (pairwise interaction) matrices are also zeroed, effectively resetting all
            the dynamic properties of the individual for a new simulation cycle.
    """

    num_species = int(individual[-1, -1, 0])
    num_pairs = int(individual[-1, -1, 1])
    pair_start = int(num_species * 2)
    pair_stop = int(pair_start + (num_pairs * 2))
    _, y, x = individual.shape

    for i in range(0, num_species * 2, 2):
        individual[i, :, :] = 0.0

    for j in range(pair_start, pair_stop, 2):
        individual[j, :, :] = 0.0

    return individual