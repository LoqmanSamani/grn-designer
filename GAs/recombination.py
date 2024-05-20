import random


def crossover(parent1, parent2, crossover_rate=0.8, num_crossover_points=1):
    """
    Perform multi-point crossover between two parents to generate two offspring.

    Args:
        parent1 (str): The binary string representing the first parent.
        parent2 (str): The binary string representing the second parent.
        crossover_rate (float, optional): The probability of performing a crossover. Defaults to 0.7.
        num_crossover_points (int, optional): The number of crossover points between the two parents. Defaults to 1.

    Returns:
        tuple of str: Two binary strings representing the offspring.

    Example:
        offspring1, offspring2 = crossover('11001', '10110', crossover_rate=0.7, num_crossover_points=2)
    """
    if random.random() < crossover_rate:

        crossover_points = sorted(random.sample(range(1, len(parent1)), num_crossover_points))
        offspring1 = []
        offspring2 = []

        for i in range(len(crossover_points) + 1):

            start = 0 if i == 0 else crossover_points[i - 1]
            end = len(parent1) if i == len(crossover_points) else crossover_points[i]

            if i % 2 == 0:

                offspring1.extend(parent1[start:end])
                offspring2.extend(parent2[start:end])

            else:

                offspring1.extend(parent2[start:end])
                offspring2.extend(parent1[start:end])
        return "".join(offspring1), "".join(offspring2)

    return parent1, parent2


