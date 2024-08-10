import numpy as np


def apply_simulation_variable_mutation(individual, mutation_rates, min_vals, max_vals, dtypes):
    """
    Apply mutations to simulation variables based on mutation rates, value ranges, and data types.

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



