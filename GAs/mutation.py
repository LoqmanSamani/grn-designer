import random


def mutate(chromosome, mutation_rate):
    """
    Mutate a binary chromosome based on the mutation rate.

    Args:
        - chromosome (str): The binary string representing the chromosome.
        - mutation_rate (float): The probability of mutating each bit.

    Returns:
        - str: The mutated binary string.
    """

    mutated_chromosome = []

    for bit in chromosome:

        if random.random() <= mutation_rate:

            mutated_chromosome.append('1' if bit == '0' else '0')

        else:

            mutated_chromosome.append(bit)

    return "".join(mutated_chromosome)



