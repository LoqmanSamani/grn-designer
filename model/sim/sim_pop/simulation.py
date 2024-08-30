from reactions import apply_complex_dissociation, apply_species_collision, apply_component_degradation, apply_component_production
from numba import jit
from diffusion import apply_diffusion
import numpy as np



@jit(nopython=True)
def population_simulation(population):
    """
    Simulate the dynamics of a population of individuals within a specified compartmental system.

    Parameters:
    - population (ndarray or tensor):
        - a numpy array (or pytorch tensor), it should have a shape of (m, z, y, x), where:
            - m: number of individuals
            - z: number of species in the system (species, complexes and also info matrices)
            - y: number of compartment rows
            - x: number of compartment columns

    Returns:
    - np.ndarray: An array of shape (m, y, x) containing the final results for each individual,
      where m is the number of individuals, y and x are the compartment shape.
    """
    m, z, y, x = population.shape  # m: number of individuals, z: species (including complexes), (y, x): compartment shape
    results = np.zeros((m, y, x))  # ndarray to store the simulation results, one final result for each individual
    num_iters = int(x)  # Number of iterations in each epoch (equal to x)
    num_species = int(population[0, -1, -1, 0])  # Number of species present in the system
    num_pairs = int(population[0, -1, -1, 1])  # Number of pairs of interacting species
    max_epoch = int(population[0, -1, -1, 2])  # Maximum number of epochs
    stop = int(population[0, -1, -1, 3])  # Simulation duration
    time_step = population[0, -1, -1, 4]  # Time step
    num_epochs = int(stop / time_step)  # Total number of epochs
    pair_start = int(num_species * 2)  # Starting index for species pairs
    pair_stop = int(pair_start + (num_pairs * 2))  # Ending index for species pairs

    epoch = 0
    while epoch < max_epoch or epoch < num_epochs:

        for i in range(num_iters):

            # Update species production
            for j in range(0, num_species*2, 2):
                population[:, j, :, i] = apply_component_production(
                    initial_concentration=population[:, j, :, i],
                    production_pattern=population[:, j+1, :, i],
                    production_rates=population[:, -1, j, 0],
                    time_step=time_step
                )

            # Handle species collision
            for j in range(pair_start, pair_stop, 2):
                (population[:, int(population[0, j+1, 0, 0]), :, i],
                 population[:, int(population[0, j+1, 0, 1]), :, i],
                 population[:, j, :, i]) = apply_species_collision(
                    species1=population[:, int(population[0, j+1, 0, 0]), :, i],
                    species2=population[:, int(population[0, j+1, 0, 1]), :, i],
                    complex_=population[:, j, :, i],
                    collision_rates=population[:, j+1, 1, 0],
                    time_step=time_step
                )

            # Update species degradation
            for j in range(0, num_species*2, 2):
                population[:, j, :, i] = apply_component_degradation(
                    initial_concentration=population[:, j, :, i],
                    degradation_rates=population[:, -1, j, 1],
                    time_step=time_step
                )

            # Handle complex degradation
            for j in range(pair_start, pair_stop, 2):
                population[:, j, :, i] = apply_component_degradation(
                    initial_concentration=population[:, j, :, i],
                    degradation_rates=population[:, j+1, 1, 2],
                    time_step=time_step
                )

            # Handle complex dissociation
            for j in range(pair_start, pair_stop, 2):
                (population[:, int(population[0, j+1, 0, 0]), :, i],
                 population[:, int(population[0, j+1, 0, 1]), :, i],
                 population[:, j, :, i]) = apply_complex_dissociation(
                    species1=population[:, int(population[0, j+1, 0, 0]), :, i],
                    species2=population[:, int(population[0, j+1, 0, 1]), :, i],
                    complex_=population[:, j, :, i],
                    dissociation_rates=population[:, j+1, 1, 1],
                    time_step=time_step
                )

            # Update species diffusion
            for j in range(0, num_species*2, 2):
                population[:, j, :, i] = apply_diffusion(
                    current_concentration=population[:, j, :, i],
                    compartment=population[:, j, :, :],
                    column_position=i,
                    diffusion_rates=population[:, -1, j, 2],
                    time_step=time_step
                )

            # Handle complex diffusion
            for j in range(pair_start, pair_stop, 2):
                population[:, j, :, i] = apply_diffusion(
                    current_concentration=population[:, j, :, i],
                    compartment=population[:, j, :, :],
                    column_position=i,
                    diffusion_rates=population[:, j+1, 1, 3],
                    time_step=time_step
                )

        epoch += 1

    return population[:, 0, :, :]















