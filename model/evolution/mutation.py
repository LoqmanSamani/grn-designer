import numpy as np
from initialization import *
import itertools

def apply_simulation_variable_mutation(individual, mutation_rates, min_vals, max_vals, dtypes):
    """
    Apply "Range-Bounded Mutations" to simulation variables based on mutation rates, value ranges, and data types.
    This function is designed to allpy mutation on "stop time of simulation (individual[-1, -1, 3])"
    and "time step (individual[-1, -1, 4])"

    Parameters:
    - individual (np.ndarray): an individual with shape (z, y, x).
    - mutation_rates (list or np.ndarray): The probability of mutation for each parameter.
    - min_vals (list or np.ndarray): The minimum values for each parameter.
    - max_vals (list or np.ndarray): The maximum values for each parameter.
    - dtypes (list or np.ndarray): The data type ('int' or 'float') for each parameter.

    Returns:
    - np.ndarray: The mutated `individual` array.
    """

    min_vals = np.array(min_vals)
    max_vals = np.array(max_vals)
    dtypes = np.array(dtypes)

    for i in range(len(min_vals)):
        if np.random.rand() < mutation_rates[i]:
            if dtypes[i] == "int":
                individual[-1, -1, 3 + i] = np.random.randint(min_vals[i], max_vals[i] + 1)
            elif dtypes[i] == "float":
                individual[-1, -1, 3 + i] = min_vals[i] + (max_vals[i] - min_vals[i]) * np.random.random()
            else:
                raise ValueError(f"Unsupported dtype '{dtypes[i]}' at index {i}. Must be 'int' or 'float'.")

    return individual


def apply_compartment_mutation(individual, mutation_rate, mean, std_dev, min_val, max_val, distribution):
    """
    Apply mutation to the compartments of the individual based on the specified distribution.
    This mutation function is specific for pattern matrices of the species.

    Parameters:
    - individual (np.ndarray): The array representing the individual with shape (z, y, x).
    - mutation_rate (float): The probability of mutation for each cell in the compartment.
    - mean (float): Mean of the Gaussian distribution used for mutation (ignored if uniform distribution).
    - std_dev (float): Standard deviation of the Gaussian distribution used for mutation (ignored if uniform distribution).
    - min_val (float, optional): Minimum value for uniform distribution (required if distribution is uniform).
    - max_val (float, optional): Maximum value for uniform distribution (required if distribution is uniform).
    - distribution (str): Type of distribution to use for mutation ("normal" or "uniform").

    Returns:
    - np.ndarray: The mutated individual array.
    """
    num_species = int(individual[-1, -1, 0])
    compartment_size = individual[0, :, :].shape

    for i in range(1, num_species * 2, 2):
        # Create random mask where mutation will happen (False or True)
        mut_mask = np.random.rand(compartment_size[0], compartment_size[1]) < mutation_rate

        if distribution == "normal":
            noise = np.random.normal(loc=mean, scale=std_dev, size=compartment_size)
        elif distribution == "uniform":
            noise = np.random.uniform(low=min_val, high=max_val, size=compartment_size)

        individual[i, :, :] += np.where(mut_mask, noise, 0)
        individual[i, :, :] = np.maximum(individual[i, :, :], 0)

    return individual


def apply_parameter_mutation(individual, mutation_rate, means, std_devs, min_vals, max_vals, distribution):
    """
    Apply mutation to specific regions of the individual based on the provided mutation rate and distribution.

    Parameters:
    - individual (np.ndarray): The array representing the individual with shape (z, y, x).
    - mutation_rate (float): The probability of mutation for each parameter.
    - means (list or np.ndarray): Means for the normal distribution.
    - std_devs (list or np.ndarray): Standard deviations for the normal distribution.
    - min_vals (list or np.ndarray): Minimum values for the uniform distribution.
    - max_vals (list or np.ndarray): Maximum values for the uniform distribution.
    - distribution (str): Type of distribution for mutation ("normal" or "uniform").

    Returns:
    - np.ndarray: The mutated individual array.
    """
    num_species = int(individual[-1, -1, 0])
    num_pairs = int(individual[-1, -1, 1])
    pair_start = num_species * 2
    pair_stop = pair_start + (num_pairs * 2)

    count = 0
    for i in range(0, num_species * 2, 2):
        mut_mask = np.random.rand(3) < mutation_rate
        if distribution == "normal":
            for j in range(3):
                individual[-1, i, j] += np.random.normal(loc=means[count], scale=std_devs[count]) * mut_mask[j]
        elif distribution == "uniform":
            for j in range(3):
                individual[-1, i, j] += (np.random.uniform(low=min_vals[count], high=max_vals[count]) - individual[-1, i, j]) * mut_mask[j]
        count += 1

    for i in range(pair_start, pair_stop, 2):
        mut_mask = np.random.rand(4) < mutation_rate
        if distribution == "normal":
            for j in range(4):
                individual[i, 1, j] += np.random.normal(loc=means[count], scale=std_devs[count]) * mut_mask[j]
        elif distribution == "uniform":
            for j in range(4):
                individual[i, 1, j] += (np.random.uniform(low=min_vals[count], high=max_vals[count]) - individual[i, 1, j]) * mut_mask[j]
        count += 1

    return individual


def apply_species_insertion_mutation(individual, mutation_rate):

    num_species = int(individual[-1, -1, 0])
    num_pairs = int(individual[-1, -1, 1])
    compartment_size = individual[0, :, :].shape

    if np.random.rand() < mutation_rate:
        pairs = pair_finding(num_species=num_species)
        init_matrix = species_initialization(compartment_size=compartment_size, num_species=len(pairs) + 1)
        individual = species_combine(individual=individual, init_matrix=init_matrix, num_species=num_species, num_pairs=num_pairs)

    return individual












def pair_finding(num_species):

    last = num_species + 1
    species = [i for i in range(1, num_species + 2, 1)]
    pairs = list(itertools.combinations(species, 2))

    out_pairs = [pair for pair in pairs if last in pair]

    return out_pairs



def species_combine(individual, init_matrix, num_species, num_pairs):

    z1 = individual.shape[0] + init_matrix.shape[0]
    z, y, x = individual.shape

    updated_individual = np.zeros((z1, y, x))
    updated_individual[:num_species*2, :, :] = individual[:num_species*2, :, :]
    updated_individual[num_species*2: num_species*2+2, :, :] = init_matrix[:2, :, :]
    updated_individual[num_species*2+2: num_species*2+num_pairs*2+2, :, :] = individual[num_species*2:-1, :, :]
    updated_individual[num_species*2+num_pairs*2+2: num_species*2+num_pairs*2+init_matrix.shape[0], :, :] = init_matrix[2:, :, :]
    updated_individual[-1] = individual[-1]

    return updated_individual

