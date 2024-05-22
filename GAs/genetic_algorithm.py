from simulation import simulation
from store_results import *
from array_binary_converter import *
from init_population import *
from fitness import *
from selection import *
from recombination import crossover
from mutation import *
import time
import numpy as np


def genetic_algorithm(population_size, specie_matrix_shape, precision_bits, num_params, generations, mutation_rates,
                      crossover_rates, num_crossover_points, target, target_precision_bits, result_path,
                      file_name="sim_result", dt=0.1, sim_start=0, sim_stop=5, epochs=500):

    """
    Execute a genetic algorithm to optimize a population of binary-encoded chromosomes.

    Args:
        population_size (int): Number of individuals in the population.
        specie_matrix_shape (tuple): Shape of the species matrices (number of rows, number of columns).
        precision_bits (dict): Dictionary specifying the number of bits for encoding each component of the chromosomes.
            - "sp1" (tuple): (min_val, max_val, bits).
            - "sp2" (tuple): (min_val, max_val, bits).
            - "sp1_cells" (tuple): (min_val, max_val, bits).
            - "sp2_cells" (tuple): (min_val, max_val, bits).
            - "params" (tuple): (min_val, max_val, bits).
        num_params (int): Number of parameters to be optimized.
        generations (int): Number of generations to run the algorithm.
        mutation_rates (list of float): Probabilities of mutating each bit in each sub-chromosome.
        crossover_rates (list of float): Probabilities of performing crossover for each sub-chromosome.
        num_crossover_points (list of int): List containing the number of crossover points for each sub-chromosome.
        target (numpy.ndarray): Target array to compare simulation results against.
        target_precision_bits (tuple): Tuple containing (min_val, max_val, bits) for encoding the target.
            - min_val (float): Minimum possible value of the target range.
            - max_val (float): Maximum possible value of the target range.
            - bits (int): Number of bits used to represent the target values in the binary string.
        result_path (str): Path to the directory where results will be saved.
        file_name (str, optional): Name of the file to save results. Defaults to "sim_result".
        dt (float): Time step.
        sim_start (float or integer): Start time of the simulation.
        sim_stop (float or integer): Stop time of the simulation.
        epochs (int): maximum number of simulation iteration

    Returns:
        population (list of list): Final population of binary-encoded chromosomes.

    Example usage:
        final_population = genetic_algorithm(
            population_size=100,
            specie_matrix_shape=(20, 20),
            precision_bits={
                "sp1": (0, 1, 8),
                "sp2": (0, 1, 8),
                "sp1_cells": (0, 1, 8),
                "sp2_cells": (0, 1, 8),
                "params": (0, 1, 8)
                },
            num_params=6,
            generations=50,
            mutation_rates=[0.01, 0.01, 0.01, 0.01, 0.01],
            crossover_rates=[0.7, 0.7, 0.7, 0.7, 0.7],
            num_crossover_points=[2, 2, 2, 2, 2],
            target=np.array([[0.5, 0.5], [0.5, 0.5]]),
            target_precision_bits=(0, 1, 8),
            result_path="path/to/save/the/results"
        )
    """

    elite_chromosomes = {}  # a dictionary to store the best chromosome of each generation
    best_fitness = []  # a list to store the best fitness of each generation
    simulation_duration = []

    print("-------------------------------------------------------------------------------------")
    print("                             *** Genetic Algorithm ***                               ")
    print("-------------------------------------------------------------------------------------")

    sp1 = initialize_population(
        pop_size=population_size,
        bit_length=specie_matrix_shape[0] * specie_matrix_shape[1] * precision_bits["sp1"][-1]
    )
    sp2 = initialize_population(
        pop_size=population_size,
        bit_length=specie_matrix_shape[0] * specie_matrix_shape[1] * precision_bits["sp2"][-1]
    )

    sp1_cells = initialize_population(
        pop_size=population_size,
        bit_length=specie_matrix_shape[0] * specie_matrix_shape[1] * precision_bits["sp1_cells"][-1]
    )
    sp2_cells = initialize_population(
        pop_size=population_size,
        bit_length=specie_matrix_shape[0] * specie_matrix_shape[1] * precision_bits["sp2_cells"][-1]
    )
    params = initialize_population(
        pop_size=population_size,
        bit_length=num_params * precision_bits["params"][-1]
    )
    precision_bits_list = [precision_bits["sp1"], precision_bits["sp2"], precision_bits["sp1_cells"], precision_bits["sp2_cells"], precision_bits["sp2_cells"]]
    population = create_population(sp1=sp1, sp2=sp2, sp1_cells=sp1_cells, sp2_cells=sp2_cells, params=params)
    binary_target = decimal_to_binary(array_list=[target], precision_bits_list=[target_precision_bits])
    max_fitness = sum([len(sub_target) for sub_target in binary_target])
    target_shape = target.shape

    for generation in range(1, generations + 1):

        tic = time.time()

        decoded_population = [binary_to_decimal(
            binary_string_list=chromosome,
            precision_bits_list=precision_bits_list,
            shapes=[specie_matrix_shape, specie_matrix_shape, specie_matrix_shape, specie_matrix_shape,
                    (1, num_params)]) for chromosome in population
        ]

        # simulate the system with each chromosome in the population
        binary_simulation_results = []
        simulation_results = []
        for chromosome in decoded_population:
            simulation_result = simulation(
                sp1=chromosome[0],
                sp2=chromosome[1],
                sp1_cells=chromosome[2],
                sp2_cells=chromosome[3],
                params=chromosome[-1],
                dt=dt,
                sim_start=sim_start,
                sim_stop=sim_stop,
                epochs=epochs,
                target_shape=target_shape
            )
            simulation_results.append(simulation_result)
            binary_simulation_results.append(
                decimal_to_binary(
                    array_list=[simulation_result],
                    precision_bits_list=[target_precision_bits]
                )
            )
        generation_fitness = compute_fitness(
            population=binary_simulation_results,
            target=binary_target
        )

        best_fitness.append(max(generation_fitness))
        elite_chromosomes[f"Generation {generation}"] = extract_based_on_max_index(
            list1=simulation_results,
            list2=generation_fitness
        )

        new_population = []
        print(binary_simulation_results[0])
        parents = select_parents(
            population=binary_simulation_results,
            fitness_scores=generation_fitness
        )
        print(parents[0])
        for _ in range(len(parents) // 2):
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)

            offspring1, offspring2 = crossover(
                parents1=parent1,
                parents2=parent2,
                crossover_rates=crossover_rates,
                num_crossover_points=num_crossover_points
            )
            new_population.extend([
                mutate(
                    chromosome=offspring1,
                    mutation_rates=mutation_rates
                ),
                mutate(
                    chromosome=offspring2,
                    mutation_rates=mutation_rates
                )
            ])

            toc = time.time()
            simulation_duration.append(toc - tic)

        print(f"Generation {generation}; Best/Max Fitness: {max(generation_fitness)}/{max_fitness}; Generation Duration: {simulation_duration[-1]}")
        print(type(new_population))
        print(len(new_population))
        print(type(new_population[0]))
        print(len(new_population[0]))
        print(type(population))
        print(len(population))
        print(type(population[0]))
        print(len(population[0]))
        population = new_population


    save_results(
        result_path=result_path,
        file_name=file_name,
        elite_chromosomes=elite_chromosomes,
        best_fitness=best_fitness,
        simulation_duration=simulation_duration,
        population=population
    )

    return population



