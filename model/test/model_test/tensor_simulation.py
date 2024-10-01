from tensor_reactions import *
from tensor_diffusion import *





def tensor_simulation(individual, parameters, num_species, num_pairs, stop, time_step, max_epoch, compartment):
    """
    Simulates the evolution of species and complexes in a spatial compartment over time.

    This function performs a series of updates to the concentrations of species and complexes within
    a 3D tensor representation of the compartment. The updates include production, collision, degradation,
    dissociation, and diffusion processes, iteratively applied over a specified number of epochs.

        Args:
            individual (tf.Tensor): A 3D tensor representing the state of species and complexes across
                                    different compartments and time steps. Shape should be [z, y, x],
                                    where `z` is the number of species and complexes, and `(y, x)`
                                    represents the spatial dimensions of the compartment.
            parameters (dict): A dictionary of TensorFlow variables containing the parameters for the
                               simulation. Includes production rates, collision rates, degradation rates,
                               and diffusion rates for species and pairs.
            num_species (int): The number of distinct species in the simulation.
            num_pairs (int): The number of distinct pairs (complexes) in the simulation.
            stop (float): The total simulation time.
            time_step (float): The time increment for each simulation step.
            max_epoch (int): The maximum number of epochs to run the simulation.

        Returns:
            tf.Tensor (y_hat): A 2D tensor representing the final state of the first species across all compartments,
                       after the simulation has completed.

        Notes:
            - The function performs updates to the `individual` tensor using `tf.tensor_scatter_nd_update`,
              which modifies the tensor in place for specific indices.
            - The simulation updates species production, handles collisions, updates degradation,
              manages dissociation, and performs diffusion for both species and complexes.
            - The simulation runs for either `max_epoch` or `num_epochs` epochs, whichever is larger.
    """

    z, y, x = individual.shape  # z: species (including complexes), (y, x): compartment shape
    num_iters = int(x)  # Number of iterations in each epoch (equal to x)
    num_epochs = int(stop / time_step)  # Total number of epochs
    pair_start = int(num_species * 2)  # Starting index for species pairs
    pair_stop = int(pair_start + (num_pairs * 2))  # Ending index for species pairs
    comp_ = int(compartment * 2)
    epoch = 0
    while epoch <= max_epoch or epoch <= num_epochs:

        for i in range(num_iters):

            # Update species production
            for j in range(0, num_species * 2, 2):

                individual = tf.tensor_scatter_nd_update(
                    tensor=individual,
                    indices=list([j, k, i] for k in range(y)),
                    updates=apply_component_production(
                        initial_concentration=individual[j, :, i],
                        production_pattern=parameters[f"compartment_{int(compartment + 1)}"][:, i],
                        production_rate=parameters[f"species_{int(compartment + 1)}"][0],
                        time_step=time_step
                    )
                )


            # Handle species collision
            for j in range(pair_start, pair_stop, 2):
                species1_idx = int(individual[j + 1, 0, 0])
                species2_idx = int(individual[j + 1, 0, 1])

                updated_species1, updated_species2, updated_complex = apply_species_collision(
                    species1=individual[species1_idx, :, i],
                    species2=individual[species2_idx, :, i],
                    complex_=individual[j, :, i],
                    collision_rate=parameters[f"pair_{int((j / 2) - num_species + 1)}"][0],
                    time_step=time_step
                )

                individual = tf.tensor_scatter_nd_update(
                    tensor=individual,
                    indices=list([species1_idx, k, i] for k in range(y)),
                    updates=updated_species1
                )
                individual = tf.tensor_scatter_nd_update(
                    tensor=individual,
                    indices=list([species2_idx, k, i] for k in range(y)),
                    updates=updated_species2
                )
                individual = tf.tensor_scatter_nd_update(
                    tensor=individual,
                    indices=list([j, k, i] for k in range(y)),
                    updates=updated_complex
                )

            # Update species degradation
            for j in range(0, num_species * 2, 2):
                individual = tf.tensor_scatter_nd_update(
                    tensor=individual,
                    indices=list([j, k, i] for k in range(y)),
                    updates=apply_component_degradation(
                        initial_concentration=individual[j, :, i],
                        degradation_rate=parameters[f"species_{int((j / 2) + 1)}"][1],
                        time_step=time_step
                    )
                )

            # Handle complex degradation
            for j in range(pair_start, pair_stop, 2):
                individual = tf.tensor_scatter_nd_update(
                    tensor=individual,
                    indices=list([j, k, i] for k in range(y)),
                    updates=apply_component_degradation(
                        initial_concentration=individual[j, :, i],
                        degradation_rate=parameters[f"pair_{int((j / 2) - num_species + 1)}"][2],
                        time_step=time_step
                    )
                )

            # Handle complex dissociation
            for j in range(pair_start, pair_stop, 2):
                species1_idx = int(individual[j + 1, 0, 0])
                species2_idx = int(individual[j + 1, 0, 1])

                updated_species1, updated_species2, updated_complex = apply_complex_dissociation(
                    species1=individual[species1_idx, :, i],
                    species2=individual[species2_idx, :, i],
                    complex_=individual[j, :, i],
                    dissociation_rate=parameters[f"pair_{int((j / 2) - num_species + 1)}"][1],
                    time_step=time_step
                )

                individual = tf.tensor_scatter_nd_update(
                    tensor=individual,
                    indices=list([species1_idx, k, i] for k in range(y)),
                    updates=updated_species1
                )
                individual = tf.tensor_scatter_nd_update(
                    tensor=individual,
                    indices=list([species2_idx, k, i] for k in range(y)),
                    updates=updated_species2
                )
                individual = tf.tensor_scatter_nd_update(
                    tensor=individual,
                    indices=list([j, k, i] for k in range(y)),
                    updates=updated_complex
                )

            # Update species diffusion
            for j in range(0, num_species * 2, 2):
                individual = tf.tensor_scatter_nd_update(
                    tensor=individual,
                    indices=list([j, k, i] for k in range(y)),
                    updates=apply_diffusion(
                        current_concentration=individual[j, :, i],
                        compartment=individual[j, :, :],
                        column_position=i,
                        diffusion_rate=parameters[f"species_{int((j / 2) + 1)}"][2],
                        time_step=time_step
                    )
                )

            # Handle complex diffusion
            for j in range(pair_start, pair_stop, 2):
                individual = tf.tensor_scatter_nd_update(
                    tensor=individual,
                    indices=list([j, k, i] for k in range(y)),
                    updates=apply_diffusion(
                        current_concentration=individual[j, :, i],
                        compartment=individual[j, :, :],
                        column_position=i,
                        diffusion_rate=parameters[f"pair_{int((j / 2) - num_species + 1)}"][3],
                        time_step=time_step
                    )
                )

        epoch += 1

    return individual[comp_, :, :]









