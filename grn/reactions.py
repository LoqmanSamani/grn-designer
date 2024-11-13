import numpy as np
from numba import jit


@jit(nopython=True)
def apply_component_production(initial_concentration, production_pattern, production_rate, time_step):

    updated_concentration = np.maximum(initial_concentration + (production_pattern * production_rate * time_step), 0)

    return updated_concentration


@jit(nopython=True)
def apply_component_degradation(initial_concentration, degradation_rate, time_step):

    updated_concentration = np.maximum(initial_concentration - (initial_concentration * degradation_rate * time_step), 0)

    return updated_concentration


@jit(nopython=True)
def apply_component_inhibition(species_1, species_2, inhibition_rate, time_step):

    inhibited = species_2 * inhibition_rate * time_step
    species_1 = np.maximum(species_1 - inhibited, 0)

    return species_1


@jit(nopython=True)
def apply_component_activation(species_1, species_2, production_pattern, production_rate, activation_rate, time_step):

    activation_effect = species_2 * activation_rate * time_step
    updated_species_1 = np.maximum(species_1 + (production_pattern * production_rate * activation_effect * time_step), 0)

    return updated_species_1




