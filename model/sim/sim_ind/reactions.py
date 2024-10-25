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
def apply_species_collision(species1, species2, complex_, collision_rate, time_step):

    collision_effect = collision_rate * time_step
    complex_formed = np.minimum(species1 * collision_effect, species2 * collision_effect)
    complex_formed = np.maximum(complex_formed, 0)

    updated_species1 = np.maximum(species1 - complex_formed, 0)
    updated_species2 = np.maximum(species2 - complex_formed, 0)

    updated_complex = complex_ + complex_formed

    return updated_species1, updated_species2, updated_complex



@jit(nopython=True)
def apply_complex_dissociation(species1, species2, complex_, dissociation_rate, time_step):

    dissociation_effect = dissociation_rate * time_step
    dissociated_amount = complex_ * dissociation_effect
    dissociated_amount = np.maximum(dissociated_amount, 0)

    updated_complex = np.maximum(complex_ - dissociated_amount, 0)
    updated_species1 = np.maximum(species1 + dissociated_amount, 0)
    updated_species2 = np.maximum(species2 + dissociated_amount, 0)

    return updated_species1, updated_species2, updated_complex

