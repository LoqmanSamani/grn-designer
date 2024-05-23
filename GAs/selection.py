import numpy as np



def select_parents_roulette(population, fitness_scores, population_size):
    """
    Select parents for the next generation based on their fitness scores
    using the Fitness Proportionate Selection (Roulette Wheel Selection) method.

    Args:
        population (list of list of str): The current population of individuals, each represented as a binary string.
        fitness_scores (list of float): The fitness scores of the individuals in the population.
        population_size (int): number of chromosomes in the original population

    Returns:
        selected (list): The selected parents for the next generation.
    """

    total_fitness = np.sum(fitness_scores)

    if total_fitness == 0:
        probabilities = np.ones(len(fitness_scores)) / len(fitness_scores)
    else:
        probabilities = np.array(fitness_scores) / total_fitness

    selected_indices = np.random.choice(len(population), size=population_size, p=probabilities)
    selected = [population[i] for i in selected_indices]

    return selected





def select_parents_tournament(population, fitness_scores, population_size, tournament_size):
    """
    Select parents for the next generation based on their fitness scores
    using the Tournament Selection method.

    Args:
        population (list of list of str): The current population of individuals, each represented as a binary string.
        fitness_scores (list of float): The fitness scores of the individuals in the population.
        population_size (int): Number of chromosomes in the original population.
        tournament_size (int): The number of individuals to participate in each tournament.

    Returns:
        selected (list): The selected parents for the next generation.
    """
    selected = []

    for _ in range(population_size):

        tournament_indices = np.random.choice(len(population), size=tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]

        best_index = tournament_indices[np.argmax(tournament_fitness)]
        selected.append(population[best_index])

    return selected

