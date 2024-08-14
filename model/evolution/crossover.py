import numpy as np


def apply_crossover(elite_individuals, individual, crossover_alpha, sim_crossover, compartment_crossover, param_crossover):

    index = int(np.random.choice(np.arange(len(elite_individuals))))
    elite_individual = elite_individuals[index]

    if sim_crossover:
        individual = apply_simulation_variable_crossover(
            elite_individual=elite_individual,
            individual=individual,
            alpha=crossover_alpha
        )
    if compartment_crossover:
        individual = apply_compartment_crossover(
            elite_individual=elite_individual,
            individual=individual,
            alpha=crossover_alpha
        )

    if param_crossover:
        individual = apply_parameter_crossover(
            elite_individual=elite_individual,
            individual=individual,
            alpha=crossover_alpha
        )

    return individual



def apply_simulation_variable_crossover(elite_individual, individual, alpha):

    individual[-1, -1, 3:5] = (alpha * individual[-1, -1, 3:5]) + ((1 - alpha) * elite_individual[-1, -1, 3:5])

    return individual


def apply_compartment_crossover(elite_individual, individual, alpha):
    num_species = int(individual[-1, -1, 0])

    for i in range(1, num_species*2+1, 2):
        individual[i, :, :] = (alpha * individual[i, :, :]) + ((1 - alpha) * elite_individual[i, :, :])

    return individual


def apply_parameter_crossover(elite_individual, individual, alpha):

    num_species = int(individual[-1, -1, 0])
    num_pairs = int(individual[-1, -1, 1])
    pair_start = int(num_species * 2)
    pair_stop = int(pair_start + (num_pairs * 2))

    for i in range(0, num_species*2, 2):
        individual[-1, i, :3] = (alpha * individual[-1, i, :3]) + ((1 - alpha) * elite_individual[-1, i, :3])

    for i in range(pair_start+1, pair_stop+1, 2):
        individual[i, 1, :4] = (alpha * individual[i, 1, :4]) + ((1 - alpha) * elite_individual[i, 1, :4])

    return individual









