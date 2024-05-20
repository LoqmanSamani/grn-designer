import numpy as np


def select_parents(population, fitness_scores):
    """
    Select parents for the next generation based on their fitness scores.
    using Fitness Proportionate Selection (Roulette Wheel Selection) method.

    Args:
        - population (list of str): The current population of individuals, each represented as a binary string.
        - fitness_scores (list of float): The fitness scores of the individuals in the population.

    Returns:
        selected (list) : The selected parents for the next generation.
    """

    probabilities = fitness_scores / np.sum(fitness_scores)

    selected = np.random.choice(population, size=len(population), p=probabilities)

    return selected
