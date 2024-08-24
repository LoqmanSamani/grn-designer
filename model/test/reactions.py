import numpy as np
from numba import jit



def apply_component_production(initial_concentration, production_pattern, production_rate, time_step):
    """
    Update the concentration of a species in each cell of a compartment.

    Parameters:
    - initial_concentration(1d array): Array of initial concentrations for each cell.
    - production_pattern(1d array): Array indicating which cells can produce the species.
    - production_rate(float): Rate at which the species are produced.
    - time_step(float): Discrete time step for the calculation.

    Returns:
    - Updated concentration array.
    """
    updated_concentration = np.maximum(initial_concentration + (production_pattern * production_rate * time_step), 0)

    return updated_concentration



def apply_component_degradation(initial_concentration, degradation_rate, time_step):
    """
    Apply degradation to the concentration of a species over time.

    Parameters:
    - initial_concentration (1d array): Array of initial concentrations for each cell.
    - degradation_rate (float): Rate at which the species degrades.
    - time_step (float): Discrete time step for the calculation.

    Returns:
    - Updated concentrations after applying degradation.
    """
    updated_concentration = np.maximum(initial_concentration - (initial_concentration * degradation_rate * time_step), 0)

    return updated_concentration


def apply_species_collision(species1, species2, complex_, collision_rate, time_step):
    """
    Apply the effect of species collision to form a complex and update the concentrations of the species.

    Parameters:
    - species1 (1d array): Array of concentrations of the first species.
    - species2 (1d array): Array of concentrations of the second species.
    - complex_ (1d array): Array of current concentrations of the complex.
    - collision_rate (float): Rate at which collisions occur between the two species.
    - time_step (float): Discrete time step  for the calculation.

    Returns:
    - tuple of numpy.ndarray: Updated concentrations of both species and the total amount of complex formed.
    """
    collision_effect = collision_rate * time_step
    complex_formed = np.minimum(species1 * collision_effect, species2 * collision_effect)
    complex_formed = np.maximum(complex_formed, 0)

    updated_species1 = np.maximum(species1 - complex_formed, 0)
    updated_species2 = np.maximum(species2 - complex_formed, 0)

    updated_complex = complex_ + complex_formed

    return updated_species1, updated_species2, updated_complex



def apply_complex_dissociation(species1, species2, complex_, dissociation_rate, time_step):
    """
    Apply the effect of complex dissociation to update the concentrations of the two species and the complex of them.

    Parameters:
    - species1 (1d array): Array of concentrations of the first species.
    - species2 (1d array): Array of concentrations of the second species.
    - complex_ (1d array): Array of current concentrations of the complex.
    - dissociation_rate (float): Rate at which the complex dissociates into the two species.
    - time_step (float): Discrete time step for the calculation.

    Returns:
    - tuple of numpy.ndarray: Updated concentrations of both species and the remaining amount of the complex.
    """
    dissociation_effect = dissociation_rate * time_step
    dissociated_amount = complex_ * dissociation_effect
    dissociated_amount = np.maximum(dissociated_amount, 0)

    updated_complex = np.maximum(complex_ - dissociated_amount, 0)
    updated_species1 = np.maximum(species1 + dissociated_amount, 0)
    updated_species2 = np.maximum(species2 + dissociated_amount, 0)

    return updated_species1, updated_species2, updated_complex

