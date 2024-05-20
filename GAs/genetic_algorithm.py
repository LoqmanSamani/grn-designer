from sim.simulation import *
from sim.diffusion import *
from sim.simulation import *
from array_binary_converter import *
from init_population import *
from fitness import *
from selection import *
from recombination import crossover
from mutation import *


def genetic_algorithm(population_size, specie_matrix_shape, specie_precision_bits,
                      param_precision_bits, generations, mutation_rate,
                      crossover_rate, num_crossover_points, target):

    gfp = initialize_population(
        pop_size=population_size,
        bit_length=specie_matrix_shape[0]*specie_matrix_shape[1]*specie_precision_bits
    )
    gfp_mCherry = initialize_population(
        pop_size=population_size,
        bit_length=specie_matrix_shape[0] * specie_matrix_shape[1] * specie_precision_bits
    )
    inhibitor = initialize_population(
        pop_size=population_size,
        bit_length=specie_matrix_shape[0] * specie_matrix_shape[1] * specie_precision_bits
    )
    inhibitor_gfp_mCherry = initialize_population(
        pop_size=population_size,
        bit_length=specie_matrix_shape[0] * specie_matrix_shape[1] * specie_precision_bits
    )
    anchor_gfp_mCherry = initialize_population(
        pop_size=population_size,
        bit_length=specie_matrix_shape[0] * specie_matrix_shape[1] * specie_precision_bits
    )
    gfp_cells = initialize_population(
        pop_size=population_size,
        bit_length=specie_matrix_shape[0] * specie_matrix_shape[1] * specie_precision_bits
    )
    gfp_mCherry_cells = initialize_population(
        pop_size=population_size,
        bit_length=specie_matrix_shape[0] * specie_matrix_shape[1] * specie_precision_bits
    )
    inhibitor_cells = initialize_population(
        pop_size=population_size,
        bit_length=specie_matrix_shape[0] * specie_matrix_shape[1] * specie_precision_bits
    )
    anchor_cells = initialize_population(
        pop_size=population_size,
        bit_length=specie_matrix_shape[0] * specie_matrix_shape[1] * specie_precision_bits
    )

    parameters = initialize_population(
        pop_size=population_size,
        bit_length=16 * param_precision_bits
    )

    # TODO: still not finished


    for generation in range(generations):

        decoded_population = [binary_to_numpy_array(ind, precision_bits) for ind in population]
        fitness_scores = np.array([evaluate_fitness(ind, observed_data) for ind in decoded_population])

        new_population = []
        for _ in range(len(population) // 2):
            parent1, parent2 = select_parents(population, fitness_scores)
            offspring1, offspring2 = crossover(parent1, parent2)
            new_population.extend([mutate(offspring1, mutation_rate), mutate(offspring2, mutation_rate)])

        population = new_population

        # Optional: Track the best solution, convergence criteria, etc.

    return population
