import numpy as np
from numba import jit
import itertools



def population_initialization(population_size, individual_shape, species_parameters, complex_parameters,
                              num_species, num_pairs, max_sim_epochs, sim_stop_time, time_step, individual_fix_size):
    """
    Initializes a population of individuals for simulation in an evolutionary algorithm.

    Parameters:

        - population_size(int): The number of individuals in the population.
        - individual_shape(tuple of int): The shape of each individual, represented as a 3D array (z, y, x).
        - species_parameters(list of lists): A list of parameter sets for each species in the individual.
                                             Each species is defined by a list containing its production rate, degradation rate, and diffusion rate.
        - complex_parameters(list of tuples): A list of parameter sets for each complex. Each entry contains a tuple,
                                              where the first element is a list of species involved, and the second element
                                              is a list of corresponding rates (e.g., collision rate, dissociation rate, etc.).
        - num_species(int): The number of species in each individual.
        - num_pairs(int): The number of complexes (species pairs) in each individual.
        - max_sim_epochs(int): The maximum number of simulation epochs.
        - sim_stop_time(float): The stop time for the simulation.
        - time_step(float): The time step for the simulation.
        - individual_fix_size(bool): A flag indicating whether all individuals in the population
                                     should have a fixed size and structure based on `individual_shape`.
                                     If `True`, all individuals have the same shape. If `False`,
                                     individuals are initialized with a smaller size and random components.

    Returns:

        - population(list of np.ndarray): A list of initialized individuals, where each individual is a 3D numpy array.
                                          The shape and structure of the individuals depend on the `individual_fix_size` flag.
                                           Each individual's array contains species parameters, complex parameters, and simulation parameters.
    """

    pair_start = int(num_species * 2)
    pair_stop = int(pair_start + (num_pairs * 2))

    if individual_fix_size:
        population = [np.zeros(individual_shape) for _ in range(population_size)]

        for i in range(0, len(species_parameters) * 2, 2):
            print(i//2)
            for ind in population:
                ind[-1, i, :3] = species_parameters[int(i // 2)]

        for i in range(pair_start + 1, pair_stop + 1, 2):
            print(int((i-(pair_start+1))//2))
            for ind in population:
                ind[i, 0, :2] = complex_parameters[int((i-(pair_start+1))//2)][0]
                ind[i, 1, :4] = complex_parameters[int((i-(pair_start+1))//2)][1]

        for ind in population:
            ind[-1, -1, :5] = [num_species, num_pairs, max_sim_epochs, sim_stop_time, time_step]

    else:
        population = [np.zeros((3, individual_shape[1], individual_shape[2])) for _ in range(population_size)]
        for ind in population:
            ind[1, :, :] = np.random.rand(individual_shape[1], individual_shape[2])
            ind[-1, 0, :3] = np.random.rand(3)
            ind[-1, -1, :5] = [1, 0, max_sim_epochs, sim_stop_time, time_step]


    return population






def species_initialization(compartment_size, pairs):
    """
    Initializes the parameters for the new species and its complexes.

    This function creates an initialization matrix containing the new species and the complexes
    formed between the new species and each existing species. It sets random initial values for the
    parameters of these species and complexes.

    Parameters:
        - compartment_size (tuple of int): The size of each compartment in the individual matrix.
        - pairs (list of tuples): A list of pairs representing the complexes between the new species and existing species.

    Returns:
        - numpy.ndarray: A matrix containing the initialized values for the new species and its complexes.
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