import numpy as np
import itertools




def species_initialization(compartment_size, pairs):

    num_species = len(pairs) + 1
    num_matrices = num_species * 2
    init_matrix = np.zeros((num_matrices, compartment_size[0], compartment_size[1]))

    for i in range(len(pairs)):
        m = np.random.rand(2, compartment_size[0], compartment_size[1])
        m[-1, 0, 0] = int(pairs[i][0])
        m[-1, 0, 1] = int(pairs[i][1])
        init_matrix[i*2+2:i*2+4, :, :] = m


    return init_matrix


