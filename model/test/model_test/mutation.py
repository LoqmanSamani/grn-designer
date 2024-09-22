from initialization import species_initialization
import itertools
import numpy as np




def apply_mutation(individual, sim_mutation_rate, compartment_mutation_rate, parameter_mutation_rate,
                   insertion_mutation_rate, deletion_mutation_rate, sim_means, sim_std_devs, sim_min_vals, sim_max_vals,
                   compartment_mean, compartment_std, compartment_min_val, compartment_max_val, sim_distribution,
                   compartment_distribution, species_param_means, species_param_stds, species_param_min_vals,
                   species_param_max_vals, complex_param_means, complex_param_stds, complex_param_min_vals,
                   complex_param_max_vals, param_distribution, sim_mutation, compartment_mutation, param_mutation,
                   species_insertion_mutation, species_deletion_mutation):
    """
        Applies various types of mutations to an individual matrix based on provided mutation rates and parameters.

        This function orchestrates the mutation process, allowing different aspects of the individual's configuration
        (such as simulation parameters, compartment properties, species parameters, and species presence) to be mutated.

        Parameters:
        - individual (numpy.ndarray): The multidimensional array representing the individual to be mutated.
        - sim_mutation_rate (float): The mutation rate for the simulation parameters.
        - compartment_mutation_rate (float): The mutation rate for the compartment properties.
        - parameter_mutation_rate (float): The mutation rate for species and complex parameters.
        - insertion_mutation_rate (float): The mutation rate for species insertion.
        - deletion_mutation_rate (float): The mutation rate for species deletion.
        - sim_means (list of float): Mean values for simulation parameter mutation (used if sim_distribution is "normal").
        - sim_std_devs (list of float): Standard deviations for simulation parameter mutation (used if sim_distribution is "normal").
        - sim_min_vals (list of float): Minimum values for simulation parameter mutation (used if sim_distribution is "uniform").
        - sim_max_vals (list of float): Maximum values for simulation parameter mutation (used if sim_distribution is "uniform").
        - compartment_mean (float): Mean value for compartment mutation (used if compartment_distribution is "normal").
        - compartment_std (float): Standard deviation for compartment mutation (used if compartment_distribution is "normal").
        - compartment_min_val (float): Minimum value for compartment mutation (used if compartment_distribution is "uniform").
        - compartment_max_val (float): Maximum value for compartment mutation (used if compartment_distribution is "uniform").
        - sim_distribution (str): The distribution type ("normal" or "uniform") for simulation parameter mutation.
        - compartment_distribution (str): The distribution type ("normal" or "uniform") for compartment mutation.
        - species_param_means (list of float): Mean values for species parameter mutation (used if param_distribution is "normal").
        - species_param_stds (list of float): Standard deviations for species parameter mutation (used if param_distribution is "normal").
        - species_param_min_vals (list of float): Minimum values for species parameter mutation (used if param_distribution is "uniform").
        - species_param_max_vals (list of float): Maximum values for species parameter mutation (used if param_distribution is "uniform").
        - complex_param_means (list of float): Mean values for complex parameter mutation (used if param_distribution is "normal").
        - complex_param_stds (list of float): Standard deviations for complex parameter mutation (used if param_distribution is "normal").
        - complex_param_min_vals (list of float): Minimum values for complex parameter mutation (used if param_distribution is "uniform").
        - complex_param_max_vals (list of float): Maximum values for complex parameter mutation (used if param_distribution is "uniform").
        - param_distribution (str): The distribution type ("normal" or "uniform") for species and complex parameter mutation.
        - sim_mutation (bool): Flag indicating whether to apply simulation parameter mutation.
        - compartment_mutation (bool): Flag indicating whether to apply compartment mutation.
        - param_mutation (bool): Flag indicating whether to apply species and complex parameter mutation.
        - species_insertion_mutation (bool): Flag indicating whether to apply species insertion mutation.
        - species_deletion_mutation (bool): Flag indicating whether to apply species deletion mutation.

        Returns:
        - numpy.ndarray: The mutated individual matrix after all applicable mutations have been applied.
        """

    if sim_mutation:
        individual = apply_simulation_parameters_mutation(
            individual=individual,
            mutation_rate=sim_mutation_rate,
            means=sim_means,
            std_devs=sim_std_devs,
            min_vals=sim_min_vals,
            max_vals=sim_max_vals,
            distribution=sim_distribution
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
        individual = apply_parameters_mutation(
            individual=individual,
            mutation_rate=parameter_mutation_rate,
            species_means=species_param_means,
            species_std_devs=species_param_stds,
            species_min_vals=species_param_min_vals,
            species_max_vals=species_param_max_vals,
            complex_means=complex_param_means,
            complex_std_devs=complex_param_stds,
            complex_min_vals=complex_param_min_vals,
            complex_max_vals=complex_param_max_vals,
            distribution=param_distribution
        )
    if species_insertion_mutation:
        individual = apply_species_insertion_mutation(
            individual=individual,
            mutation_rate=insertion_mutation_rate
        )
    if species_deletion_mutation and individual.shape[0] > 3:
        individual = apply_species_deletion_mutation(
            individual=individual,
            mutation_rate=deletion_mutation_rate
        )

    return individual




def apply_simulation_parameters_mutation(individual, mutation_rate, means, std_devs, min_vals, max_vals, distribution):
    """
    Applies mutation to the stop time and time step parameters of a simulation individual.

    Parameters:
    - individual: ndarray
        A multidimensional array representing an individual in the simulation.
        The stop time and time step parameters are located at positions [-1, -1, 3]
        and [-1, -1, 4] respectively.
    - mutation_rate: float
        The probability of each parameter being mutated, a value between 0 and 1.
    - means: list or ndarray
        A list or array containing the means for the normal distribution mutation.
        Only used if distribution is set to "normal".
    - std_devs: list or ndarray
        A list or array containing the standard deviations for the normal distribution mutation.
        Only used if distribution is set to "normal".
    - min_vals: list or ndarray
        A list or array containing the minimum bounds for the uniform distribution mutation.
        Only used if distribution is set to "uniform".
    - max_vals: list or ndarray
        A list or array containing the maximum bounds for the uniform distribution mutation.
        Only used if distribution is set to "uniform".
    - distribution: str
        The type of distribution to use for mutation. Must be either "uniform" or "normal".

    Returns:
    - individual: ndarray
        The individual with possibly mutated stop time and time step parameters.
    """

    mut_mask = np.random.rand(2) < mutation_rate

    for i in range(2):

        if distribution == "uniform":
            individual[-1, -1, i + 3] += (np.random.uniform(low=min_vals[i], high=max_vals[i]) - individual[
                -1, -1, i + 3]) * mut_mask[i]
        elif distribution == "normal":
            individual[-1, -1, i + 3] += np.random.normal(loc=means[i], scale=std_devs[i]) * mut_mask[i]

        individual[-1, -1, i + 3] = max(min_vals[i], min(max_vals[i], individual[-1, -1, i + 3]))

    individual[-1, -1, 3:5] = np.maximum(individual[-1, -1, 3:5], 0)
    if individual[-1, -1, 3] / individual[-1, -1, 4] > 1000 or individual[-1, -1, 3] / individual[-1, -1, 4] < 100:
        individual[-1, -1, 3] = 20
        individual[-1, -1, 4] = 0.1

    return individual




def apply_compartment_mutation(individual, mutation_rate, mean, std_dev, min_val, max_val, distribution):
    """
    Applies mutation to specific compartments (matrices) within an individual matrix.
    The compartments that undergo mutation represent simulation patterns indicating
    which cells are capable of producing specific products (initial conditions).

    Parameters:
    - individual: ndarray
        A multidimensional array representing an individual in the simulation.
        The compartments to be mutated are located at odd indices from 1 to num_species*2.
    - mutation_rate: float
        The probability of each element in the compartment being mutated, a value between 0 and 1.
    - mean: float
        The mean value used for the normal distribution mutation.
        Only used if distribution is set to "normal".
    - std_dev: float
        The standard deviation for the normal distribution mutation.
        Only used if distribution is set to "normal".
    - min_val: float
        The minimum bound for the uniform distribution mutation and the minimum value allowed
        after mutation.
    - max_val: float
        The maximum bound for the uniform distribution mutation and the maximum value allowed
        after mutation.
    - distribution: str
        The type of distribution to use for mutation. Must be either "uniform" or "normal".

    Returns:
    - individual: ndarray
        The individual with mutated compartments, ensuring all values are non-negative
        and within specified bounds.
    """

    num_species = int(individual[-1, -1, 0])
    z, y, x = individual.shape

    for i in range(1, num_species * 2, 2):
        mut_mask = np.random.rand(y, x) < mutation_rate

        if distribution == "normal":
            noise = np.random.normal(loc=mean, scale=std_dev, size=(y, x))
        elif distribution == "uniform":
            noise = np.random.uniform(low=min_val, high=max_val, size=(y, x))

        individual[i, :, :] += np.where(mut_mask, noise, 0)
        individual[i, :, :] = np.maximum(individual[i, :, :], 0)
        individual[i, :, :] = np.minimum(individual[i, :, :], 1)

    return individual




def apply_parameters_mutation(individual, mutation_rate, species_means, species_std_devs,
                             species_min_vals, species_max_vals, complex_means, complex_std_devs,
                             complex_min_vals, complex_max_vals, distribution):
    """
    Apply mutations to the parameters of an individual in an evolutionary algorithm.

    This function mutates the species and complex parameters of an individual.
    The mutations are applied based on a specified mutation rate and distribution type (normal or uniform).

    Parameters:
    - individual (numpy.ndarray): A multidimensional array representing the species and complex parameters.
                                  The last row of the individual holds metadata:
                                  [number of species, number of complexes].
    - mutation_rate (float): The probability with which each parameter is mutated. Should be between 0 and 1.
    - species_means (list of float): Mean values used for normal distribution mutation of species parameters.
    - species_std_devs (list of float): Standard deviation values for normal distribution mutation of species parameters.
    - species_min_vals (list of float): Minimum values used for uniform distribution mutation of species parameters.
    - species_max_vals (list of float): Maximum values used for uniform distribution mutation of species parameters.
    - complex_means (list of float): Mean values used for normal distribution mutation of complex parameters.
    - complex_std_devs (list of float): Standard deviation values for normal distribution mutation of complex parameters.
    - complex_min_vals (list of float): Minimum values used for uniform distribution mutation of complex parameters.
    - complex_max_vals (list of float): Maximum values used for uniform distribution mutation of complex parameters.
    - distribution (str): Type of distribution to use for mutation, either "normal" or "uniform".

    Returns:
    - numpy.ndarray: The mutated individual with the same shape as the input.
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
                individual[-1, i, j] += np.random.normal(loc=species_means[count], scale=species_std_devs[count]) * \
                                        mut_mask[j]
        elif distribution == "uniform":
            for j in range(3):
                individual[-1, i, j] += (np.random.uniform(low=species_min_vals[count], high=species_max_vals[count]) -
                                         individual[-1, i, j]) * mut_mask[j]

        individual[-1, i, :3] = np.minimum(individual[-1, i, :3], 1)
        individual[-1, i, :3] = np.maximum(individual[-1, i, :3], np.random.rand())
        count += 1

    count = 0
    for i in range(pair_start + 1, pair_stop, 2):
        mut_mask = np.random.rand(4) < mutation_rate
        if distribution == "normal":
            for j in range(4):
                individual[i, 1, j] += np.random.normal(loc=complex_means[count], scale=complex_std_devs[count]) * \
                                       mut_mask[j]
        elif distribution == "uniform":
            for j in range(4):
                individual[i, 1, j] += (np.random.uniform(low=complex_min_vals[count], high=complex_max_vals[count]) -
                                        individual[i, 1, j]) * mut_mask[j]

        individual[i, 1, :4] = np.minimum(individual[i, 1, :4], 1)
        individual[i, 1, :4] = np.maximum(individual[i, 1, :4], np.random.rand())
        count += 1

    return individual




def apply_species_insertion_mutation(individual, mutation_rate):
    """
    Applies a species insertion mutation to the individual with a given probability.

    This function adds a new species to the individual if a random value is below the specified mutation rate.
    When a new species is added, it automatically generates all possible complexes between the new species and
    the existing species. The updated individual structure is then returned.

    Parameters:
    - individual (numpy.ndarray): The multidimensional array representing the species and complexes.
    - mutation_rate (float): The probability of adding a new species.

    Returns:
    - numpy.ndarray: The updated individual with a new species and its complexes if the mutation occurred.
    """

    num_species = int(individual[-1, -1, 0])
    num_pairs = int(individual[-1, -1, 1])
    z, y, x = individual.shape

    if np.random.rand() < mutation_rate:
        pairs = pair_finding(
            num_species=num_species
        )
        init_matrix = species_initialization(
            compartment_size=(y, x),
            pairs=pairs
        )
        individual = species_combine(
            individual=individual,
            init_matrix=init_matrix,
            num_species=num_species,
            num_pairs=num_pairs
        )

    return individual





def apply_species_deletion_mutation(individual, mutation_rate):
    """
    Applies a species deletion mutation to the individual with a given probability.

    This function randomly selects a species and deletes it along with all complexes involving that species
    if a random value is below the specified mutation rate.

    Parameters:
    - individual (numpy.ndarray): The multidimensional array representing the species and complexes.
    - mutation_rate (float): The probability of deleting a species.

    Returns:
    - numpy.ndarray: The updated individual with the species and related complexes removed if the mutation occurred.
    """
    num_species = int(individual[-1, -1, 0])

    if np.random.rand() < mutation_rate and num_species > 1:
        deleted_species = int(np.random.choice(np.arange(2, num_species+1)))

        individual = species_deletion(
            individual=individual,
            deleted_species=deleted_species
        )

    return individual





def pair_finding(num_species):
    """
    Finds all possible pairs between the new species and the existing species.

    Given the current number of species, this function generates all possible pairs between
    the new species (which is one more than the current number) and the existing species.

    Parameters:
    - num_species (int): The current number of species.

    Returns:
    - list of tuples: A list of pairs (as tuples) where each pair includes the new species.
    """

    last = num_species + 1
    species = [i for i in range(1, num_species + 2, 1)]
    pairs = list(itertools.combinations(species, 2))

    related_pairs = [pair for pair in pairs if last in pair]
    pair_indices = [((pair[0] - 1) * 2, (pair[1]-1)*2) for pair in related_pairs]

    return pair_indices




def species_combine(individual, init_matrix, num_species, num_pairs):
    """
    Combines the new species and its complexes with the existing individual.

    This function takes the existing individual matrix and the initialization matrix for the new species and its
    complexes, and combines them into a single updated individual matrix. It also updates the metadata to reflect
    the addition of the new species and complexes.

    Parameters:
    - individual (numpy.ndarray): The original individual matrix.
    - init_matrix (numpy.ndarray): The initialization matrix for the new species and complexes.
    - num_species (int): The original number of species.
    - num_pairs (int): The original number of complexes.

    Returns:
    - numpy.ndarray: The updated individual matrix with the new species and complexes added.
    """

    z, y, x = individual.shape
    z1 = z + init_matrix.shape[0]

    updated_individual = np.zeros((z1, y, x))
    updated_individual[:num_species * 2, :, :] = individual[:num_species * 2, :, :]
    updated_individual[num_species * 2:num_species * 2 + init_matrix.shape[0], :, :] = init_matrix
    updated_individual[num_species * 2 + init_matrix.shape[0]:, :, :] = individual[num_species * 2:, :, :]
    updated_individual[-1, -1, 0] = int(num_species + 1)
    updated_individual[-1, -1, 1] = int(num_pairs + ((init_matrix.shape[0] - 2) / 2))
    updated_individual[-1, num_species * 2, :3] = np.random.rand(3)

    return updated_individual




def species_deletion(individual, deleted_species):
    """
    Deletes a species and all complexes involving that species from the individual matrix.

    Parameters:
    - individual (numpy.ndarray): The individual matrix before deletion.
    - deleted_species (int): The index of the species to be deleted.

    Returns:
    - numpy.ndarray: The updated individual matrix after deletion.
    """
    num_species = int(individual[-1, -1, 0])
    num_pairs = int(individual[-1, -1, 1])
    pair_start = int((num_species * 2) + 1)
    pair_stop = int(pair_start + (num_pairs * 2))

    delete_indices = [(deleted_species-1)*2, ((deleted_species-1)*2)+1]

    for i in range(pair_start, pair_stop, 2):
        if int((individual[i, 0, 0] / 2) + 1) == deleted_species or int((individual[i, 0, 1] / 2) + 1) == deleted_species:
            delete_indices.extend([i - 1, i])

    updated_individual = np.delete(individual, delete_indices, axis=0)

    updated_individual[-1, -1, 0] = num_species - 1
    updated_individual[-1, -1, 1] = num_pairs - len(delete_indices) // 2 + 1

    return updated_individual
