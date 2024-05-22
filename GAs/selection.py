import numpy as np



def select_parents(population, fitness_scores):
    """
    Select parents for the next generation based on their fitness scores
    using the Fitness Proportionate Selection (Roulette Wheel Selection) method.

    Args:
        population (list of list of str): The current population of individuals, each represented as a binary string.
        fitness_scores (list of float): The fitness scores of the individuals in the population.

    Returns:
        selected (list): The selected parents for the next generation.
    """

    total_fitness = np.sum(fitness_scores)

    if total_fitness == 0:
        probabilities = np.ones(len(fitness_scores)) / len(fitness_scores)
    else:
        probabilities = np.array(fitness_scores) / total_fitness

    selected_indices = np.random.choice(len(population), size=len(population), p=probabilities)
    selected = [population[i] for i in selected_indices]

    return selected

