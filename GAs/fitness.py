

import numpy as np


def compute_fitness(population, target):
    """
    Compute the fitness value for each individual in the population.

    Args:
        population (list of list of str): The population of individuals, each represented as a binary string.
        target (list of str): The target binary string to compare against.

    Returns:
        fitness_values (list of float): A list of fitness values, one for each individual in the population.
    """
    fitness_values = []

    for individual in population:
        # Initialize fitness for current individual
        individual_fitness = 0

        # Compare each element in the individual with the corresponding element in the target
        for individual_str, target_str in zip(individual, target):
            # Count matching elements
            for i, j in zip(individual_str, target_str):
                if i == j:
                    individual_fitness += 1

        # Append fitness for current individual to the list
        fitness_values.append(individual_fitness)

    return fitness_values



























"""

def compute_fitness(population, target):
    
    Compute the fitness value for each individual in the population.

    Args:
        population (list of list of str): The population of individuals, each represented as a binary string.
        target (list of str): The target binary string to compare against.

    Returns:
        fitness_values (list of float): A list of fitness values, one for each individual in the population.
    
    fitness_values = []

    for individual in population:
        fitness = sum(1 for i, j in zip(individual, target) if i == j)
        fitness_values.append(fitness)

    return fitness_values

"""



