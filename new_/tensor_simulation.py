from tensor_reactions import *
from tensor_diffusion import *


def tensor_simulation(individual, parameters, num_species, num_pairs, stop, time_step, max_epoch, compartment, device):

    individual = individual.to(device)

    z, y, x = individual.shape  # z: species (including complexes), (y, x): compartment shape
    num_iters = int(x)  # Number of iterations in each epoch (equal to x)
    pair_start = int(num_species * 2)  # Starting index for species pairs
    pair_stop = int(pair_start + (num_pairs * 2))  # Ending index for species pairs
    num_epochs = int(stop / time_step)  # Total number of epochs
    comp_ = int(compartment * 2)



    epoch = 0
    while epoch <= max_epoch or epoch <= num_epochs:

        for i in range(num_iters):

            # Update species production
            for j in range(0, num_species * 2, 2):
                h = int(j / 2) + 1
                individual[j, :, i] = apply_component_production(
                    initial_concentration=individual[j, :, i],
                    production_pattern=parameters[f"compartment_{h}"][:, i],
                    production_rate=parameters[f"species_{h}"][0],
                    time_step=time_step
                )

            # Handle species collision
            for j in range(pair_start, pair_stop, 2):
                (individual[int(individual[j + 1, 0, 0]), :, i],
                 individual[int(individual[j + 1, 0, 1]), :, i],
                 individual[j, :, i]) = apply_species_collision(
                    species1=individual[int(individual[j + 1, 0, 0]), :, i],
                    species2=individual[int(individual[j + 1, 0, 1]), :, i],
                    complex_=individual[j, :, i],
                    collision_rate=parameters[f"pair_{int((j / 2) - num_species + 1)}"][0],
                    time_step=time_step
                )


            # Update species degradation
            for j in range(0, num_species * 2, 2):
                individual[j, :, i] = apply_component_degradation(
                    initial_concentration=individual[j, :, i],
                    degradation_rate=parameters[f"species_{int((j / 2) + 1)}"][1],
                    time_step=time_step
                )

            # Handle complex degradation
            for j in range(pair_start, pair_stop, 2):
                individual[j, :, i] = apply_component_degradation(
                    initial_concentration=individual[j, :, i],
                    degradation_rate=parameters[f"pair_{int((j / 2) - num_species + 1)}"][2],
                    time_step=time_step
                )

            # Handle complex dissociation
            for j in range(pair_start, pair_stop, 2):
                (individual[int(individual[j + 1, 0, 0]), :, i],
                 individual[int(individual[j + 1, 0, 1]), :, i],
                 individual[j, :, i]) = apply_complex_dissociation(
                    species1=individual[int(individual[j + 1, 0, 0]), :, i],
                    species2=individual[int(individual[j + 1, 0, 1]), :, i],
                    complex_=individual[j, :, i],
                    dissociation_rate=parameters[f"pair_{int((j / 2) - num_species + 1)}"][1],
                    time_step=time_step
                )

            # Update species diffusion
            for j in range(0, num_species * 2, 2):
                individual[j, :, i] = apply_diffusion(
                    current_concentration=individual[j, :, i],
                    compartment=individual[j, :, :],
                    column_position=i,
                    diffusion_rate=parameters[f"species_{int((j / 2) + 1)}"][2],
                    time_step=time_step
                )

            # Handle complex diffusion
            for j in range(pair_start, pair_stop, 2):
                individual[j, :, i] = apply_diffusion(
                    current_concentration=individual[j, :, i],
                    compartment=individual[j, :, :],
                    column_position=i,
                    diffusion_rate=parameters[f"pair_{int((j / 2) - num_species + 1)}"][3],
                    time_step=time_step
                )

        epoch += 1

    return individual[comp_, :, :]





