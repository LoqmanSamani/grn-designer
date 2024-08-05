from reactions import *
from diffusion import *


@jit(nopython=True)
def individual_simulation(individual):
    """
    Simulate the dynamics of a single individual within a specified compartmental system.

    Parameters:
    - individual (np.ndarray):
        - A numpy array with shape (z, y, x), where:
            - z: number of species in the system (including complexes)
            - y: number of compartment rows
            - x: number of compartment columns

        - The last slice (i.e., individual[-1, :, :]) contains simulation parameters:
            - [0]: number of species present in the system
            - [1]: number of pairs of interacting species
            - [2]: maximum number of epochs
            - [3]: simulation duration
            - [4]: time step

        - The penultimate slice (i.e., individual[-2, :, :]) contains parameters for species pairs:
            - [0:2]: Indices of species pairs and their corresponding interaction parameters.

    Returns:
    - np.ndarray: The final concentrations of the first species in the compartment,
      with shape (y, x) where y and x are the compartment dimensions.
    """
    z, y, x = individual.shape  # z: species (including complexes), (y, x): compartment shape
    num_iters = int(x)  # Number of iterations in each epoch (equal to x)
    num_species = int(individual[-1, -1, 0])  # Number of species present in the system
    num_pairs = int(individual[-1, -1, 1])  # Number of pairs of interacting species
    max_epoch = int(individual[-1, -1, 2])  # Maximum number of epochs
    stop = int(individual[-1, -1, 3])  # Simulation duration
    time_step = individual[-1, -1, 4]  # Time step
    num_epochs = int(stop / time_step)  # Total number of epochs
    pair_start = int(num_species * 2)  # Starting index for species pairs
    pair_stop = int(pair_start + (num_pairs * 2))  # Ending index for species pairs

    epoch = 0
    while epoch < max_epoch or epoch < num_epochs:

        for i in range(num_iters):

            # Update species production
            for j in range(0, num_species*2, 2):
                individual[j, :, i] = apply_component_production(
                    initial_concentration=individual[j, :, i],
                    production_pattern=individual[j+1, :, i],
                    production_rate=individual[-1, j, 0],
                    time_step=time_step
                )

            # Handle species collision
            for j in range(pair_start, pair_stop, 2):
                (individual[int(individual[j+1, 0, 0]), :, i],
                 individual[int(individual[j+1, 0, 1]), :, i],
                 individual[j, :, i]) = apply_species_collision(
                    species1=individual[int(individual[j+1, 0, 0]), :, i],
                    species2=individual[int(individual[j+1, 0, 1]), :, i],
                    complex_=individual[j, :, i],
                    collision_rate=individual[j+1, 1, 0],
                    time_step=time_step
                )

            # Update species degradation
            for j in range(0, num_species*2, 2):
                individual[j, :, i] = apply_component_degradation(
                    initial_concentration=individual[j, :, i],
                    degradation_rate=individual[-1, j, 1],
                    time_step=time_step
                )

            # Handle complex degradation
            for j in range(pair_start, num_pairs, 2):
                individual[j, :, i] = apply_component_degradation(
                    initial_concentration=individual[j, :, i],
                    degradation_rate=individual[j+1, 1, 2],
                    time_step=time_step
                )

            # Handle complex dissociation
            for j in range(pair_start, pair_stop, 2):
                (individual[int(individual[j+1, 0, 0]), :, i],
                 individual[int(individual[j+1, 0, 1]), :, i],
                 individual[j, :, i]) = apply_complex_dissociation(
                    species1=individual[int(individual[j+1, 0, 0]), :, i],
                    species2=individual[int(individual[j+1, 0, 1]), :, i],
                    complex_=individual[j, :, i],
                    dissociation_rate=individual[j+1, 1, 1],
                    time_step=time_step
                )

            # Update species diffusion
            for j in range(0, num_species*2, 2):
                individual[j, :, i] = apply_diffusion(
                    current_concentration=individual[j, :, i],
                    compartment=individual[j, :, :],
                    column_position=i,
                    diffusion_rate=individual[-1, j, 2],
                    time_step=time_step
                )

            # Handle complex diffusion
            for j in range(pair_start, num_pairs, 2):
                individual[j, :, i] = apply_diffusion(
                    current_concentration=individual[j, :, i],
                    compartment=individual[j, :, :],
                    column_position=i,
                    diffusion_rate=individual[j+1, 1, 3],
                    time_step=time_step
                )

        epoch += 1

    return individual[0, :, :]
