def tensor_simulation(individual, parameters, num_species, num_pairs, stop, time_step, max_epoch, param_opt, compartment_opt):
    import tensorflow as tf

    z, y, x = individual.shape  # z: species (including complexes), (y, x): compartment shape
    num_iters = x  # Number of iterations in each epoch (equal to x)
    num_epochs = int(stop / time_step)  # Total number of epochs
    pair_start = num_species * 2  # Starting index for species pairs
    pair_stop = pair_start + (num_pairs * 2)  # Ending index for species pairs

    epoch = 0

    while epoch <= max_epoch or epoch <= num_epochs:
        # Precompute for this epoch, only once
        for i in range(num_iters):
            updates_indices = []
            updates_values = []

            # Update species production
            if compartment_opt and param_opt:
                s = 1
                for j in range(0, num_species * 2, 2):
                    # Collect indices and updates for batching
                    updates_indices += list([j, k, i] for k in range(y))
                    updates_values += apply_component_production(
                        initial_concentration=individual[j, :, i],
                        production_pattern=parameters[f"compartment_{s}"][:, i],
                        production_rate=parameters[f"species_{int((j / 2) + 1)}"][0],
                        time_step=time_step
                    )
                    s += 1

            elif compartment_opt and not param_opt:
                s = 1
                for j in range(0, num_species * 2, 2):
                    updates_indices += list([j, k, i] for k in range(y))
                    updates_values += apply_component_production(
                        initial_concentration=individual[j, :, i],
                        production_pattern=parameters[f"compartment_{s}"][:, i],
                        production_rate=individual[-1, j, 0],  # Use current rate from tensor
                        time_step=time_step
                    )
                    s += 1

            elif not compartment_opt and param_opt:
                for j in range(0, num_species * 2, 2):
                    updates_indices += list([j, k, i] for k in range(y))
                    updates_values += apply_component_production(
                        initial_concentration=individual[j, :, i],
                        production_pattern=individual[j + 1, :, i],  # Using current pattern
                        production_rate=parameters[f"species_{int((j / 2) + 1)}"][0],
                        time_step=time_step
                    )

            else:
                for j in range(0, num_species * 2, 2):
                    updates_indices += list([j, k, i] for k in range(y))
                    updates_values += apply_component_production(
                        initial_concentration=individual[j, :, i],
                        production_pattern=individual[j + 1, :, i],
                        production_rate=individual[-1, j, 0],
                        time_step=time_step
                    )

            # Handle species collisions and other dynamics
            for j in range(pair_start, pair_stop, 2):
                species1_idx = int(individual[j + 1, 0, 0])
                species2_idx = int(individual[j + 1, 0, 1])

                # Collect collision updates
                updated_species1, updated_species2, updated_complex = apply_species_collision(
                    species1=individual[species1_idx, :, i],
                    species2=individual[species2_idx, :, i],
                    complex_=individual[j, :, i],
                    collision_rate=parameters[f"pair_{int((j / 2) - num_species + 1)}"][0],
                    time_step=time_step
                )

                updates_indices += list([species1_idx, k, i] for k in range(y))
                updates_values += updated_species1
                updates_indices += list([species2_idx, k, i] for k in range(y))
                updates_values += updated_species2
                updates_indices += list([j, k, i] for k in range(y))
                updates_values += updated_complex

            # Similarly batch degradation and diffusion

            # Apply all collected updates at once for this iteration
            individual = tf.tensor_scatter_nd_update(
                tensor=individual,
                indices=updates_indices,
                updates=updates_values
            )

        epoch += 1

    return individual[0, :, :]
