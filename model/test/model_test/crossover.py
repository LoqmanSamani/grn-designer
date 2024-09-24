import random


def apply_crossover(elite_individuals, individual, crossover_alpha, sim_crossover, compartment_crossover, param_crossover):
    """
    Applies crossover operations between an individual and a randomly selected elite individual.

    This function blends various aspects (simulation variables, compartment properties, and parameters) of the
    selected elite individual with the current individual based on specified crossover operations. The blending is
    controlled by the `crossover_alpha` parameter, which determines the weighting between the elite and the current
    individual's attributes.

    Parameters:
        - elite_individuals (list of numpy.ndarray): A list of elite individuals from which one is randomly selected to
          participate in the crossover.
        - individual (numpy.ndarray): The individual that will undergo the crossover, having its attributes blended with
          those of the selected elite individual.
        - crossover_alpha (float): The weighting factor for the crossover, ranging between 0 and 1. A higher value gives
          more weight to the original individual's attributes.
        - sim_crossover (bool): If True, the crossover is applied to the simulation variables.
        - compartment_crossover (bool): If True, the crossover is applied to the compartment properties.
        - param_crossover (bool): If True, the crossover is applied to the species and complex parameters.

    Returns:
        - numpy.ndarray: The updated individual after applying the specified crossover operations.
    """
    if len(elite_individuals) > 0:
        elite_individual = random.choice(elite_individuals)
        if sim_crossover:
            individual = apply_simulation_variable_crossover(
                elite_individual=elite_individual,
                individual=individual,
                alpha=crossover_alpha
            )

        if compartment_crossover:
            individual = apply_compartment_crossover(
                elite_individual=elite_individual,
                individual=individual,
                alpha=crossover_alpha
            )

        if param_crossover:
            individual = apply_parameter_crossover(
                elite_individual=elite_individual,
                individual=individual,
                alpha=crossover_alpha
            )
    else:
        print("there is no elite-individual !!!")

    return individual



def apply_simulation_variable_crossover(elite_individual, individual, alpha):
    """
    Performs crossover on the simulation variables between an elite individual and another individual.

    This function blends the simulation variables (assumed to be located in the last row and specific columns) of the
    two individuals using a weighted average controlled by the alpha parameter.

    Parameters:
        - elite_individual (numpy.ndarray): The elite individual from which simulation variables are partially inherited.
        - individual (numpy.ndarray): The individual whose simulation variables will be updated.
        - alpha (float): The weighting factor for the crossover, ranging between 0 and 1. A higher value gives more weight
          to the original individual's variables.

    Returns:
        - numpy.ndarray: The updated individual after applying the crossover on the simulation variables.
    """

    individual[-1, -1, 3:5] = (alpha * individual[-1, -1, 3:5]) + ((1 - alpha) * elite_individual[-1, -1, 3:5])
    if individual[-1, -1, 3] / individual[-1, -1, 4] > 300 or individual[-1, -1, 3] / individual[-1, -1, 4] < 100:
        individual[-1, -1, 3] = 20
        individual[-1, -1, 4] = 0.1

    return individual


def apply_compartment_crossover(elite_individual, individual, alpha):
    """
    Performs crossover on compartment properties between an elite individual and another individual.

    This function blends the compartment properties (assumed to be in odd-indexed layers of the individual matrix)
    of the two individuals using a weighted average controlled by the alpha parameter.

    Parameters:
        - elite_individual (numpy.ndarray): The elite individual from which compartment properties are partially inherited.
        - individual (numpy.ndarray): The individual whose compartment properties will be updated.
        - alpha (float): The weighting factor for the crossover, ranging between 0 and 1. A higher value gives more weight
          to the original individual's properties.

    Returns:
        - numpy.ndarray: The updated individual after applying the crossover on compartment properties.
    """
    num_species = int(individual[-1, -1, 0])

    for i in range(1, num_species * 2 + 1, 2):
        individual[i, :, :] = (alpha * individual[i, :, :]) + ((1 - alpha) * elite_individual[i, :, :])

    return individual


def apply_parameter_crossover(elite_individual, individual, alpha):
    """
    Performs crossover on species and complex parameters between an elite individual and another individual.

    This function blends the species parameters (assumed to be located in even-indexed layers) and complex parameters
    (assumed to be located in specific layers corresponding to species pairs) of the two individuals using a weighted
    average controlled by the alpha parameter.

    Parameters:
        - elite_individual (numpy.ndarray): The elite individual from which parameters are partially inherited.
        - individual (numpy.ndarray): The individual whose parameters will be updated.
        - alpha (float): The weighting factor for the crossover, ranging between 0 and 1. A higher value gives more weight
          to the original individual's parameters.

    Returns:
        - numpy.ndarray: The updated individual after applying the crossover on species and complex parameters.
    """

    num_species = int(individual[-1, -1, 0])
    num_pairs = int(individual[-1, -1, 1])
    pair_start = int(num_species * 2)
    pair_stop = int(pair_start + (num_pairs * 2))

    for i in range(0, num_species * 2, 2):
        individual[-1, i, :3] = (alpha * individual[-1, i, :3]) + ((1 - alpha) * elite_individual[-1, i, :3])

    for i in range(pair_start + 1, pair_stop + 1, 2):
        individual[i, 1, :4] = (alpha * individual[i, 1, :4]) + ((1 - alpha) * elite_individual[i, 1, :4])

    return individual




def filter_elite_individuals(low_cost_individuals, elite_individuals, high_cost_individual):
    """
    Filters elite individuals to find those with the same size (number of layers) as the high-cost individual.

    This function ensures that the elite individuals used in further operations have the same number of layers (first
    dimension size) as the high-cost individual. If no matching elite individuals are found, it falls back to selecting
    individuals from the low-cost population that match the required size.

    Parameters:
        - low_cost_individuals (list of numpy.ndarray): A list of individuals with lower associated costs, used as a fallback
          if no elite individuals match the required size.
        - elite_individuals (list of numpy.ndarray): The primary list of elite individuals, which is filtered to match the size
          of the high-cost individual.
        - high_cost_individual (numpy.ndarray): The reference individual whose size is used to filter the elite individuals.

    Returns:
        - list of numpy.ndarray: A filtered list of individuals that have the same size as the high-cost individual. If no
          matching elite individuals are found, the list will contain low-cost individuals with the matching size.
    """

    filtered_elite_individuals = [ind for ind in elite_individuals if ind.shape[0] == high_cost_individual.shape[0]]
    if len(filtered_elite_individuals) == 0:
        filtered_elite_individuals = [ind for ind in low_cost_individuals if ind.shape[0] == high_cost_individual.shape[0]]
        filtered_elite_individuals = filtered_elite_individuals[: len(elite_individuals)]

    return filtered_elite_individuals






