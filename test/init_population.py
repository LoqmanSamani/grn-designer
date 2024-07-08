import random
import numpy as np




def initialize_population(pop_size, bit_length):

    """
    Initialize a population for a genetic algorithm.

    Args:
        - pop_size (int): The number of individuals in the population.
        - bit_length (int): The length of the binary string representing each individual.

    Returns:
        - population (list): A list of binary strings, each representing an individual in the population.
    """
    population = [''.join(random.choice('01') for _ in range(bit_length)) for _ in range(pop_size)]

    return population





def create_population(sp1, sp2, sp1_cells, sp2_cells, params):

    """
    Create a population of chromosomes from species concentrations, cell distributions, and parameters.

    Args:
        sp1 (list): List of initialized concentration matrices for species 1.
        sp2 (list): List of initialized concentration matrices for species 2.
        sp1_cells (list): nested List of cell distributions for species 1.
        sp2_cells (list): nested List of cell distributions for species 2.
        params (list): nested List of parameter sets.

    Returns:
        population (nested list): Population of chromosomes, where each chromosome is a list containing
                      [species 1 concentration matrix, species 2 concentration matrix, species 1 cell distribution matrix,
                      species 2 cell distribution matrix , parameters].
    """
    population = [[sp1[i], sp2[i], sp1_cells[i], sp2_cells[i], params[i]] for i in range(len(sp1))]

    return population




def extract_based_on_max_index(list1, list2):
    """
    Extract an object from list1 based on the index of the maximum value in list2.

    Args:
        list1 (list): The list from which to extract the object.
        list2 (list): The list used to determine the index of the maximum value.

    Returns:
        object: The object from list1 corresponding to the index of the maximum value in list2.
    """
    max_index = list2.index(max(list2))

    return list1[max_index]


def check_nan(sim_result):
    """
    Check if there are any NaN (Not a Number) values in the simulation result.

    Args:
        sim_result (numpy.ndarray): Array containing simulation results.

    Returns:
        num_nan (int): Number of NaN values in the simulation result.
    """
    nan_indices = np.isnan(sim_result)
    num_nan = np.sum(nan_indices)

    return num_nan



def check_inf(sim_result):
    """
    Check if there are any infinite values in the simulation result.

    Args:
        sim_result (numpy.ndarray): Array containing simulation results.

    Returns:
        num_inf (int): Number of infinite values in the simulation result.
    """
    inf_indices = np.isinf(sim_result)
    num_inf = np.sum(inf_indices)

    return num_inf
