import random




def crossover(parents1, parents2, crossover_rates, num_crossover_points):

    """
    Perform multi-points crossover between two parents to generate two offspring.

    Args:
        parents1 (list of str): A list of binary strings representing the first parent.
        parents2 (list of str): A list of binary strings representing the second parent.
        crossover_rates (list of float): A list of probabilities of performing crossover for each sub-chromosome.
        num_crossover_points (list of int): A list containing the number of crossover points for each sub-chromosome.

    Returns:
        tuple of list: Two lists of binary strings representing the offspring.

    """

    offspring1 = []
    offspring2 = []

    for parent1, parent2, crossover_rate, num_points in zip(parents1, parents2, crossover_rates, num_crossover_points):

        if random.random() < crossover_rate:

            crossover_points = sorted(random.sample(range(1, len(parent1)), num_points))
            child1, child2 = [], []
            start = 0

            for i, point in enumerate(crossover_points + [len(parent1)]):

                if i % 2 == 0:

                    child1.extend(parent1[start:point])
                    child2.extend(parent2[start:point])

                else:

                    child1.extend(parent2[start:point])
                    child2.extend(parent1[start:point])

                start = point

            offspring1.append(''.join(child1))
            offspring2.append(''.join(child2))

        else:

            offspring1.append(parent1)
            offspring2.append(parent2)

    return offspring1, offspring2


