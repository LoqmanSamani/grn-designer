from reactions import *
from diffusion import *
from numba import jit


@jit(nopython=True)
def individual_simulation(agent, num_patterns):
    z, y, x = agent.shape  # z: species (including complexes), (y, x): compartment shape
    num_iters = int(x)  # Number of iterations in each epoch (equal to x)
    num_species = int(agent[-1, -1, 0])  # Number of species present in the system
    num_pairs = int(agent[-1, -1, 1])  # Number of pairs of interacting species
    max_epoch = int(agent[-1, -1, 2])  # Maximum number of epochs
    stop = int(agent[-1, -1, 3])  # Simulation duration
    time_step = agent[-1, -1, 4]  # Time step
    num_epochs = int(stop / time_step)  # Total number of epochs
    pair_start = int(num_species * 2)  # Starting index for species pairs
    pair_stop = int(pair_start + (num_pairs * 2))  # Ending index for species pairs
    sim_results = np.zeros((num_patterns, y, x))

    epoch = 0
    while epoch <= max_epoch or epoch <= num_epochs:

        for i in range(num_iters):

            # Update species production
            for j in range(0, num_species*2, 2):
                agent[j, :, i] = apply_component_production(
                    initial_concentration=agent[j, :, i],
                    production_pattern=agent[j + 1, :, i],
                    production_rate=agent[-1, j, 0],
                    time_step=time_step
                )

            # Handle species collision
            for j in range(pair_start, pair_stop, 2):
                (agent[int(agent[j + 1, 0, 0]), :, i],
                 agent[int(agent[j + 1, 0, 1]), :, i],
                 agent[j, :, i]) = apply_species_collision(
                    species1=agent[int(agent[j + 1, 0, 0]), :, i],
                    species2=agent[int(agent[j + 1, 0, 1]), :, i],
                    complex_=agent[j, :, i],
                    collision_rate=agent[j + 1, 1, 0],
                    time_step=time_step
                )

            # Update species degradation
            for j in range(0, num_species*2, 2):
                agent[j, :, i] = apply_component_degradation(
                    initial_concentration=agent[j, :, i],
                    degradation_rate=agent[-1, j, 1],
                    time_step=time_step
                )

            # Handle complex degradation
            for j in range(pair_start, pair_stop, 2):
                agent[j, :, i] = apply_component_degradation(
                    initial_concentration=agent[j, :, i],
                    degradation_rate=agent[j + 1, 1, 2],
                    time_step=time_step
                )

            # Handle complex dissociation
            for j in range(pair_start, pair_stop, 2):
                (agent[int(agent[j + 1, 0, 0]), :, i],
                 agent[int(agent[j + 1, 0, 1]), :, i],
                 agent[j, :, i]) = apply_complex_dissociation(
                    species1=agent[int(agent[j + 1, 0, 0]), :, i],
                    species2=agent[int(agent[j + 1, 0, 1]), :, i],
                    complex_=agent[j, :, i],
                    dissociation_rate=agent[j + 1, 1, 1],
                    time_step=time_step
                )

            # Update species diffusion
            for j in range(0, num_species*2, 2):
                agent[j, :, i] = apply_diffusion(
                    current_concentration=agent[j, :, i],
                    compartment=agent[j, :, :],
                    column_position=i,
                    diffusion_rate=agent[-1, j, 2],
                    time_step=time_step
                )

            # Handle complex diffusion
            for j in range(pair_start, pair_stop, 2):
                agent[j, :, i] = apply_diffusion(
                    current_concentration=agent[j, :, i],
                    compartment=agent[j, :, :],
                    column_position=i,
                    diffusion_rate=agent[j + 1, 1, 3],
                    time_step=time_step
                )

        epoch += 1
    sp = 0
    for i in range(num_patterns):
        sim_results[i, :, :] = agent[sp, :, :]
        sp += 2

    return sim_results