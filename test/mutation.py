import random





def mutate(chromosome, mutation_rates):

    """
    Mutate a chromosome with sub-chromosomes (each sub-chromosome is a binary string) based on the mutation rates.

    Args:
        - chromosome (list of str): The list contains binary strings representing the chromosome,
                                    where each binary string is a sub-chromosome.
        - mutation_rates (list of float): The probabilities of mutating each bit in each sub-chromosome.

    Returns:
        - mutated_chromosome (list of str): The mutated chromosome, with each sub-chromosome mutated based on the corresponding mutation rate.

    Example:
        chromosome = ['11001', '10110']
        mutation_rates = [0.1, 0.2]
        mutated_chromosome = mutate(chromosome, mutation_rates)
    """

    mutated_chromosome = []

    for sub_chromosome, mutation_rate in zip(chromosome, mutation_rates):

        mutated_sub_chromosome = ''.join(
            '1' if bit == '0' and random.random() <= mutation_rate else
            '0' if bit == '1' and random.random() <= mutation_rate else
            bit
            for bit in sub_chromosome
        )

        mutated_chromosome.append(mutated_sub_chromosome)

    return mutated_chromosome




