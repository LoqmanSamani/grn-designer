
def compute_fitness(population, target):

    """
    Compute the fitness value for each individual in the population.

    Args:
        - population (list of str): The population of individuals, each represented as a binary string.
        - target (str): The target binary string to compare against.

    Returns:
        fitness_values (list of int): A list of fitness values, one for each individual in the population.
    """

    fitness_values = []

    for individual in population:

        fitness = sum(1 for i, j in zip(individual, target) if i == j)
        fitness_values.append(fitness)

    return fitness_values



