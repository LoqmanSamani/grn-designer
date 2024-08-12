import numpy as np
import itertools





def species_initialization(compartment_size, num_species):

    num_matrices = num_species * 2

    init_matrix = np.random.rand(num_matrices, compartment_size[0], compartment_size[1])

    return init_matrix


