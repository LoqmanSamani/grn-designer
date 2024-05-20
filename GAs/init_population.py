import random


def initialize_population(pop_size, bit_length):

    """
    Initialize a population for a genetic algorithm.

    Args:
        - pop_size (int): The number of individuals in the population.
        - bit_length (int): The length of the binary string representing each individual.

    Returns:
        - population (list): A list of binary strings, each representing an individual in the population.
    """
    population = [''.join(random.choice('01') for _ in range(bit_length)) for _ in range(pop_size)]

    return population








pop_binary = initialize_population(20, 40)

pop_decimal = [binary_to_decimal(pop_binary[i], [(0, 1, 8) for _ in range(5)]) for i in range(len(pop_binary))]
print(pop_decimal)
print(len(pop_decimal))
print(len(pop_decimal[0]))
