import numpy as np
from initialization import *
import itertools


def apply_mutation(
        individual, sim_mutation_rates, compartment_mutation_rate, parameter_mutation_rate,
        insertion_mutation_rate, deletion_mutation_rate, sim_min_vals, sim_max_vals, sim_dtypes,
        compartment_mean, compartment_std, compartment_min_val, compartment_max_val, compartment_distribution,
        param_means, param_stds, param_min_vals, param_max_vals, param_distribution, sim_mutation,
        compartment_mutation, param_mutation, species_insertion_mutation, species_deletion_mutation
):

    if sim_mutation:
        individual = apply_simulation_variable_mutation(
            individual=individual,
            mutation_rates=sim_mutation_rates,
            min_vals=sim_min_vals,
            max_vals=sim_max_vals,
            dtypes=sim_dtypes
        )
    if compartment_mutation:
        individual = apply_compartment_mutation(
            individual=individual,
            mutation_rate=compartment_mutation_rate,
            mean=compartment_mean,
            std_dev=compartment_std,
            min_val=compartment_min_val,
            max_val=compartment_max_val,
            distribution=compartment_distribution
        )
    if param_mutation:
        individual = apply_parameter_mutation(
            individual=individual,
            mutation_rate=parameter_mutation_rate,
            means=param_means,
            std_devs=param_stds,
            min_vals=param_min_vals,
            max_vals=param_max_vals,
            distribution=param_distribution
        )
    if species_insertion_mutation:
        individual = apply_species_insertion_mutation(
            individual=individual,
            mutation_rate=insertion_mutation_rate
        )
    if species_deletion_mutation:
        individual = apply_species_deletion_mutation(
            individual=individual,
            mutation_rate=deletion_mutation_rate
        )

    return individual







def apply_simulation_variable_mutation(individual, mutation_rates, min_vals, max_vals, dtypes):
    """
    Apply "Range-Bounded Mutations" to simulation variables based on mutation rates, value ranges, and data types.
    This function is designed to apply mutation on "stop time of simulation (individual[-1, -1, 3])"
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
        init_matrix = species_initialization(compartment_size=compartment_size, pairs=pairs)
        individual = species_combine(individual=individual, init_matrix=init_matrix, num_species=num_species, num_pairs=num_pairs)

    return individual


def apply_species_deletion_mutation(individual, mutation_rate):

    num_species = int(individual[-1, -1, 0])
    if np.random.rand() < mutation_rate:
        deleted_species = int(np.random.choice(np.arange(1, num_species)))
        individual = species_deletion(individual=individual, deleted_species=deleted_species)

        return individual






def species_deletion(individual, deleted_species):

    num_species = int(individual[-1, -1, 0])
    num_pairs = int(individual[-1, -1, 1])
    pair_start = int(num_species * 2) + 1
    pair_stop = int(pair_start + (num_pairs * 2))
    count = [deleted_species*2, deleted_species*2+1]

    for i in range(pair_start, pair_stop, 2):
        if individual[i, 0, 0] == deleted_species or individual[i, 0, 1] == deleted_species:
            count.append(i-1)
            count.append(i)
    updated_individual = np.delete(individual, count, axis=0)

    return updated_individual



def pair_finding(num_species):

    last = num_species + 1
    species = [i for i in range(1, num_species + 2, 1)]
    pairs = list(itertools.combinations(species, 2))

    out_pairs = [pair for pair in pairs if last in pair]

    return out_pairs



def species_combine(individual, init_matrix, num_species, num_pairs):

    z, y, x = individual.shape
    z1 = z + init_matrix.shape[0]

    updated_individual = np.zeros((z1, y, x))
    updated_individual[:num_species*2, :, :] = individual[:num_species*2, :, :]
    updated_individual[num_species*2:num_species*2+init_matrix.shape[0], :, :] = init_matrix
    updated_individual[num_species*2+init_matrix.shape[0]:, :, :] = individual[num_species*2:, :, :]
    updated_individual[-1, 0, 0] = int(num_species+1)
    updated_individual[-1, 0, 1] = int(num_pairs+((init_matrix.shape[0]-2)/2))

    return updated_individual


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
