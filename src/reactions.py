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
def apply_component_inhibition(species_1, species_2, inhibition_rate, hill_coefficient, dissociation_constant, time_step):

    hill_inhibition = (species_2 ** hill_coefficient) / (dissociation_constant ** hill_coefficient + species_2 ** hill_coefficient + 1e-8)

    inhibited = inhibition_rate * hill_inhibition * time_step
    
    species_1 = np.maximum(species_1 - inhibited, 0)

    return species_1


@jit(nopython=True)
def apply_component_activation(species_1, species_2, production_pattern, activation_rate, hill_coefficient, dissociation_constant, time_step):

    hill_activation = (species_2 ** hill_coefficient) / (dissociation_constant ** hill_coefficient + species_2 ** hill_coefficient + 1e-8)

    activated = production_pattern * activation_rate * hill_activation * time_step
    
    species_1 = np.maximum(species_1 + activated, 0)

    return species_1





