from tensor_reactions import *
from tensor_diffusion import *

def tensor_simulation(individual, parameters, num_species, num_pairs, stop, time_step, max_epoch, compartment, device):
    individual = individual.to(device)  # Move the tensor to the device (e.g., GPU)

    z, y, x = individual.shape  # z: species (including complexes), (y, x): compartment shape
    num_iters = int(x)  # Number of iterations in each epoch (equal to x)
    pair_start = int(num_species * 2)  # Starting index for species pairs
    pair_stop = int(pair_start + (num_pairs * 2))  # Ending index for species pairs
    num_epochs = int(stop / time_step)  # Total number of epochs
    comp_ = int(compartment * 2)

    epoch = 0
    while epoch <= max_epoch and epoch <= num_epochs:  # Use "and" instead of "or"
        for i in range(num_iters):
            # Create a copy of the individual tensor for updated values
            updated_individual = individual.clone()

            # Update species production
            for j in range(0, num_species * 2, 2):
                h = int(j / 2) + 1
                updated_individual[j, :, i] = apply_component_production(
                    initial_concentration=individual[j, :, i],
                    production_pattern=parameters[f"compartment_{h}"][:, i],
                    production_rate=parameters[f"species_{h}"][0],
                    time_step=time_step
                )
            individual = updated_individual.clone()

            # Handle species collision
            for j in range(pair_start, pair_stop, 2):
                species1, species2, complex_ = apply_species_collision(
                    species1=individual[int(individual[j + 1, 0, 0]), :, i],
                    species2=individual[int(individual[j + 1, 0, 1]), :, i],
                    complex_=individual[j, :, i],
                    collision_rate=parameters[f"pair_{int((j / 2) - num_species + 1)}"][0],
                    time_step=time_step
                )
                updated_individual[int(individual[j + 1, 0, 0]), :, i] = species1
                updated_individual[int(individual[j + 1, 0, 1]), :, i] = species2
                updated_individual[j, :, i] = complex_

            individual = updated_individual.clone()

            # Update species degradation
            for j in range(0, num_species * 2, 2):
                updated_individual[j, :, i] = apply_component_degradation(
                    initial_concentration=individual[j, :, i],
                    degradation_rate=parameters[f"species_{int((j / 2) + 1)}"][1],
                    time_step=time_step
                )

            individual = updated_individual.clone()

            # Handle complex degradation
            for j in range(pair_start, pair_stop, 2):
                updated_individual[j, :, i] = apply_component_degradation(
                    initial_concentration=individual[j, :, i],
                    degradation_rate=parameters[f"pair_{int((j / 2) - num_species + 1)}"][2],
                    time_step=time_step
                )
            individual = updated_individual.clone()

            # Handle complex dissociation
            for j in range(pair_start, pair_stop, 2):
                species1, species2, complex_ = apply_complex_dissociation(
                    species1=individual[int(individual[j + 1, 0, 0]), :, i],
                    species2=individual[int(individual[j + 1, 0, 1]), :, i],
                    complex_=individual[j, :, i],
                    dissociation_rate=parameters[f"pair_{int((j / 2) - num_species + 1)}"][1],
                    time_step=time_step
                )
                updated_individual[int(individual[j + 1, 0, 0]), :, i] = species1
                updated_individual[int(individual[j + 1, 0, 1]), :, i] = species2
                updated_individual[j, :, i] = complex_

            individual = updated_individual.clone()

            # Update species diffusion
            for j in range(0, num_species * 2, 2):
                updated_individual[j, :, i] = apply_diffusion(
                    current_concentration=individual[j, :, i],
                    compartment=individual[j, :, :],
                    column_position=i,
                    diffusion_rate=parameters[f"species_{int((j / 2) + 1)}"][2],
                    time_step=time_step
                )
            individual = updated_individual.clone()

            # Handle complex diffusion
            for j in range(pair_start, pair_stop, 2):
                updated_individual[j, :, i] = apply_diffusion(
                    current_concentration=individual[j, :, i],
                    compartment=individual[j, :, :],
                    column_position=i,
                    diffusion_rate=parameters[f"pair_{int((j / 2) - num_species + 1)}"][3],
                    time_step=time_step
                )

            # Update individual tensor with the new values
            individual = updated_individual.clone()

        epoch += 1

    print("sim works!")
    return individual[comp_, :, :]


"""
def tensor_simulation(individual, parameters, num_species, num_pairs, stop, time_step, max_epoch, compartment, device):
    individual = individual.to(device)  # Move the tensor to the device (e.g., GPU)

    z, y, x = individual.shape  # z: species (including complexes), (y, x): compartment shape
    num_iters = int(x)  # Number of iterations in each epoch (equal to x)
    pair_start = int(num_species * 2)  # Starting index for species pairs
    pair_stop = int(pair_start + (num_pairs * 2))  # Ending index for species pairs
    num_epochs = int(stop / time_step)  # Total number of epochs
    comp_ = int(compartment * 2)

    epoch = 0
    while epoch <= max_epoch and epoch <= num_epochs:  # Use "and" instead of "or"
        for i in range(num_iters):
            # Create a copy of the individual tensor for updated values
            updated_individual = individual.clone()

            # Update species production
            for j in range(0, num_species * 2, 2):
                h = int(j / 2) + 1
                updated_individual[j, :, i] = apply_component_production(
                    initial_concentration=individual[j, :, i],
                    production_pattern=parameters[f"compartment_{h}"][:, i],
                    production_rate=parameters[f"species_{h}"][0],
                    time_step=time_step
                )

            # Handle species collision
            for j in range(pair_start, pair_stop, 2):
                species1, species2, complex_ = apply_species_collision(
                    species1=individual[int(individual[j + 1, 0, 0]), :, i],
                    species2=individual[int(individual[j + 1, 0, 1]), :, i],
                    complex_=individual[j, :, i],
                    collision_rate=parameters[f"pair_{int((j / 2) - num_species + 1)}"][0],
                    time_step=time_step
                )
                updated_individual[int(individual[j + 1, 0, 0]), :, i] = species1
                updated_individual[int(individual[j + 1, 0, 1]), :, i] = species2
                updated_individual[j, :, i] = complex_

            # Update species degradation
            for j in range(0, num_species * 2, 2):
                updated_individual[j, :, i] = apply_component_degradation(
                    initial_concentration=individual[j, :, i],
                    degradation_rate=parameters[f"species_{int((j / 2) + 1)}"][1],
                    time_step=time_step
                )

            # Handle complex degradation
            for j in range(pair_start, pair_stop, 2):
                updated_individual[j, :, i] = apply_component_degradation(
                    initial_concentration=individual[j, :, i],
                    degradation_rate=parameters[f"pair_{int((j / 2) - num_species + 1)}"][2],
                    time_step=time_step
                )

            # Handle complex dissociation
            for j in range(pair_start, pair_stop, 2):
                species1, species2, complex_ = apply_complex_dissociation(
                    species1=individual[int(individual[j + 1, 0, 0]), :, i],
                    species2=individual[int(individual[j + 1, 0, 1]), :, i],
                    complex_=individual[j, :, i],
                    dissociation_rate=parameters[f"pair_{int((j / 2) - num_species + 1)}"][1],
                    time_step=time_step
                )
                updated_individual[int(individual[j + 1, 0, 0]), :, i] = species1
                updated_individual[int(individual[j + 1, 0, 1]), :, i] = species2
                updated_individual[j, :, i] = complex_

            # Update species diffusion
            for j in range(0, num_species * 2, 2):
                updated_individual[j, :, i] = apply_diffusion(
                    current_concentration=individual[j, :, i],
                    compartment=individual[j, :, :],
                    column_position=i,
                    diffusion_rate=parameters[f"species_{int((j / 2) + 1)}"][2],
                    time_step=time_step
                )

            # Handle complex diffusion
            for j in range(pair_start, pair_stop, 2):
                updated_individual[j, :, i] = apply_diffusion(
                    current_concentration=individual[j, :, i],
                    compartment=individual[j, :, :],
                    column_position=i,
                    diffusion_rate=parameters[f"pair_{int((j / 2) - num_species + 1)}"][3],
                    time_step=time_step
                )

            # Update individual tensor with the new values
            individual = updated_individual.clone()

        epoch += 1

    print("sim works!")
    return individual[comp_, :, :]



def tensor_simulation(individual, parameters, num_species, num_pairs, stop, time_step, max_epoch, compartment, device):
    individual = individual.to(device)  # Move the tensor to the device (e.g., GPU)

    z, y, x = individual.shape  # z: species (including complexes), (y, x): compartment shape
    num_iters = int(x)  # Number of iterations in each epoch (equal to x)
    pair_start = int(num_species * 2)  # Starting index for species pairs
    pair_stop = int(pair_start + (num_pairs * 2))  # Ending index for species pairs
    num_epochs = int(stop / time_step)  # Total number of epochs
    comp_ = int(compartment * 2)

    epoch = 0
    while epoch <= max_epoch or epoch <= num_epochs:
        for i in range(num_iters):
            # Clone the tensor before any operation to avoid in-place modifications
            #individual = individual.clone()

            # Update species production
            for j in range(0, num_species * 2, 2):
                h = int(j / 2) + 1
                individual[j, :, i] = apply_component_production(
                    initial_concentration=individual[j, :, i],
                    production_pattern=parameters[f"compartment_{h}"][:, i],
                    production_rate=parameters[f"species_{h}"][0],
                    time_step=time_step
                )
                #individual[j, :, i] = production.detach().clone()
            #print("till here works!")
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
                # Avoid in-place modifications
                #individual[int(individual[j + 1, 0, 0]), :, i] = species1.detach().clone()
                #individual[int(individual[j + 1, 0, 1]), :, i] = species2.detach().clone()
                #individual[j, :, i] = complex_.detach().clone()
            #print("till here works!")
            # Update species degradation
            for j in range(0, num_species * 2, 2):
                individual[j, :, i] = apply_component_degradation(
                    initial_concentration=individual[j, :, i],
                    degradation_rate=parameters[f"species_{int((j / 2) + 1)}"][1],
                    time_step=time_step
                )
                #individual[j, :, i] = degradation.detach().clone()

            # Handle complex degradation
            for j in range(pair_start, pair_stop, 2):
                individual[j, :, i] = apply_component_degradation(
                    initial_concentration=individual[j, :, i],
                    degradation_rate=parameters[f"pair_{int((j / 2) - num_species + 1)}"][2],
                    time_step=time_step
                )
                #individual[j, :, i] = degradation.detach().clone()

            # Handle complex dissociation
            for j in range(pair_start, pair_stop, 2):
                individual[int(individual[j + 1, 0, 0]), :, i], individual[int(individual[j + 1, 0, 1]), :, i], individual[j, :, i] = apply_complex_dissociation(
                    species1=individual[int(individual[j + 1, 0, 0]), :, i],
                    species2=individual[int(individual[j + 1, 0, 1]), :, i],
                    complex_=individual[j, :, i],
                    dissociation_rate=parameters[f"pair_{int((j / 2) - num_species + 1)}"][1],
                    time_step=time_step
                )
                #individual[int(individual[j + 1, 0, 0]), :, i] = species1.detach().clone()
                #individual[int(individual[j + 1, 0, 1]), :, i] = species2.detach().clone()
                #individual[j, :, i] = complex_.detach().clone()
            #print("till here works!")
            # Update species diffusion
            for j in range(0, num_species * 2, 2):
                individual[j, :, i] = apply_diffusion(
                    current_concentration=individual[j, :, i],
                    compartment=individual[j, :, :],
                    column_position=i,
                    diffusion_rate=parameters[f"species_{int((j / 2) + 1)}"][2],
                    time_step=time_step
                )
                #individual[j, :, i] = diffusion.detach().clone()

            # Handle complex diffusion
            for j in range(pair_start, pair_stop, 2):
                individual[j, :, i] = apply_diffusion(
                    current_concentration=individual[j, :, i],
                    compartment=individual[j, :, :],
                    column_position=i,
                    diffusion_rate=parameters[f"pair_{int((j / 2) - num_species + 1)}"][3],
                    time_step=time_step
                )
                #individual[j, :, i] = diffusion.detach().clone()

        epoch += 1
    print("sim works!")
    return individual[comp_, :, :]










def tensor_simulation(individual, parameters, num_species, num_pairs, stop, time_step, max_epoch, compartment, device):
    individual = individual.to(device)  # Move the tensor to the device (e.g., GPU)

    z, y, x = individual.shape  # z: species (including complexes), (y, x): compartment shape
    num_iters = int(x)  # Number of iterations in each epoch (equal to x)
    pair_start = int(num_species * 2)  # Starting index for species pairs
    pair_stop = int(pair_start + (num_pairs * 2))  # Ending index for species pairs
    num_epochs = int(stop / time_step)  # Total number of epochs
    comp_ = int(compartment * 2)

    epoch = 0
    while epoch <= max_epoch or epoch <= num_epochs:
        for i in range(num_iters):
            # Clone the tensor before any operation to avoid in-place modifications
            individual = individual.clone()

            # Update species production
            for j in range(0, num_species * 2, 2):
                h = int(j / 2) + 1
                production = apply_component_production(
                    initial_concentration=individual[j, :, i],
                    production_pattern=parameters[f"compartment_{h}"][:, i],
                    production_rate=parameters[f"species_{h}"][0],
                    time_step=time_step
                )
                individual[j, :, i] = production.detach().clone()

            # Handle species collision
            for j in range(pair_start, pair_stop, 2):
                species1, species2, complex_ = apply_species_collision(
                    species1=individual[int(individual[j + 1, 0, 0]), :, i],
                    species2=individual[int(individual[j + 1, 0, 1]), :, i],
                    complex_=individual[j, :, i],
                    collision_rate=parameters[f"pair_{int((j / 2) - num_species + 1)}"][0],
                    time_step=time_step
                )
                # Avoid in-place modifications
                individual[int(individual[j + 1, 0, 0]), :, i] = species1.detach().clone()
                individual[int(individual[j + 1, 0, 1]), :, i] = species2.detach().clone()
                individual[j, :, i] = complex_.detach().clone()

            # Update species degradation
            for j in range(0, num_species * 2, 2):
                degradation = apply_component_degradation(
                    initial_concentration=individual[j, :, i],
                    degradation_rate=parameters[f"species_{int((j / 2) + 1)}"][1],
                    time_step=time_step
                )
                individual[j, :, i] = degradation.detach().clone()

            # Handle complex degradation
            for j in range(pair_start, pair_stop, 2):
                degradation = apply_component_degradation(
                    initial_concentration=individual[j, :, i],
                    degradation_rate=parameters[f"pair_{int((j / 2) - num_species + 1)}"][2],
                    time_step=time_step
                )
                individual[j, :, i] = degradation.detach().clone()

            # Handle complex dissociation
            for j in range(pair_start, pair_stop, 2):
                species1, species2, complex_ = apply_complex_dissociation(
                    species1=individual[int(individual[j + 1, 0, 0]), :, i],
                    species2=individual[int(individual[j + 1, 0, 1]), :, i],
                    complex_=individual[j, :, i],
                    dissociation_rate=parameters[f"pair_{int((j / 2) - num_species + 1)}"][1],
                    time_step=time_step
                )
                individual[int(individual[j + 1, 0, 0]), :, i] = species1.detach().clone()
                individual[int(individual[j + 1, 0, 1]), :, i] = species2.detach().clone()
                individual[j, :, i] = complex_.detach().clone()

            # Update species diffusion
            for j in range(0, num_species * 2, 2):
                diffusion = apply_diffusion(
                    current_concentration=individual[j, :, i],
                    compartment=individual[j, :, :],
                    column_position=i,
                    diffusion_rate=parameters[f"species_{int((j / 2) + 1)}"][2],
                    time_step=time_step
                )
                individual[j, :, i] = diffusion.detach().clone()

            # Handle complex diffusion
            for j in range(pair_start, pair_stop, 2):
                diffusion = apply_diffusion(
                    current_concentration=individual[j, :, i],
                    compartment=individual[j, :, :],
                    column_position=i,
                    diffusion_rate=parameters[f"pair_{int((j / 2) - num_species + 1)}"][3],
                    time_step=time_step
                )
                individual[j, :, i] = diffusion.detach().clone()

        epoch += 1

    return individual[comp_, :, :]





def tensor_simulation(individual, parameters, num_species, num_pairs, stop, time_step, max_epoch, compartment, device):
    individual = individual.to(device)  # Move the tensor to the device

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
                production = apply_component_production(
                    initial_concentration=individual[j, :, i],
                    production_pattern=parameters[f"compartment_{h}"][:, i],
                    production_rate=parameters[f"species_{h}"][0],
                    time_step=time_step
                )
                individual[j, :, i] = production  # No detach/clone

            # Handle species collision
            for j in range(pair_start, pair_stop, 2):
                species1, species2, complex_ = apply_species_collision(
                    species1=individual[int(individual[j + 1, 0, 0]), :, i],
                    species2=individual[int(individual[j + 1, 0, 1]), :, i],
                    complex_=individual[j, :, i],
                    collision_rate=parameters[f"pair_{int((j / 2) - num_species + 1)}"][0],
                    time_step=time_step
                )
                individual[int(individual[j + 1, 0, 0]), :, i] = species1  # No detach/clone
                individual[int(individual[j + 1, 0, 1]), :, i] = species2  # No detach/clone
                individual[j, :, i] = complex_  # No detach/clone

            # Update species degradation
            for j in range(0, num_species * 2, 2):
                degradation = apply_component_degradation(
                    initial_concentration=individual[j, :, i],
                    degradation_rate=parameters[f"species_{int((j / 2) + 1)}"][1],
                    time_step=time_step
                )
                individual[j, :, i] = degradation  # No detach/clone

            # Handle complex degradation
            for j in range(pair_start, pair_stop, 2):
                degradation = apply_component_degradation(
                    initial_concentration=individual[j, :, i],
                    degradation_rate=parameters[f"pair_{int((j / 2) - num_species + 1)}"][2],
                    time_step=time_step
                )
                individual[j, :, i] = degradation  # No detach/clone

            # Handle complex dissociation
            for j in range(pair_start, pair_stop, 2):
                species1, species2, complex_ = apply_complex_dissociation(
                    species1=individual[int(individual[j + 1, 0, 0]), :, i],
                    species2=individual[int(individual[j + 1, 0, 1]), :, i],
                    complex_=individual[j, :, i],
                    dissociation_rate=parameters[f"pair_{int((j / 2) - num_species + 1)}"][1],
                    time_step=time_step
                )
                individual[int(individual[j + 1, 0, 0]), :, i] = species1  # No detach/clone
                individual[int(individual[j + 1, 0, 1]), :, i] = species2  # No detach/clone
                individual[j, :, i] = complex_  # No detach/clone

            # Update species diffusion
            for j in range(0, num_species * 2, 2):
                diffusion = apply_diffusion(
                    current_concentration=individual[j, :, i],
                    compartment=individual[j, :, :],
                    column_position=i,
                    diffusion_rate=parameters[f"species_{int((j / 2) + 1)}"][2],
                    time_step=time_step
                )
                individual[j, :, i] = diffusion  # No detach/clone

            # Handle complex diffusion
            for j in range(pair_start, pair_stop, 2):
                diffusion = apply_diffusion(
                    current_concentration=individual[j, :, i],
                    compartment=individual[j, :, :],
                    column_position=i,
                    diffusion_rate=parameters[f"pair_{int((j / 2) - num_species + 1)}"][3],
                    time_step=time_step
                )
                individual[j, :, i] = diffusion  # No detach/clone

        epoch += 1

    return individual[comp_, :, :]


def tensor_simulation(individual, parameters, num_species, num_pairs, stop, time_step, max_epoch, compartment, device):
    individual = individual.to(device)  # Move the tensor to the device (e.g., GPU)

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
                production = apply_component_production(
                    initial_concentration=individual[j, :, i],
                    production_pattern=parameters[f"compartment_{h}"][:, i],
                    production_rate=parameters[f"species_{h}"][0],
                    time_step=time_step
                )
                individual[j, :, i] = production.detach().clone()  # Avoid in-place modification

            # Handle species collision
            for j in range(pair_start, pair_stop, 2):
                species1, species2, complex_ = apply_species_collision(
                    species1=individual[int(individual[j + 1, 0, 0]), :, i],
                    species2=individual[int(individual[j + 1, 0, 1]), :, i],
                    complex_=individual[j, :, i],
                    collision_rate=parameters[f"pair_{int((j / 2) - num_species + 1)}"][0],
                    time_step=time_step
                )
                individual[int(individual[j + 1, 0, 0]), :, i] = species1.detach().clone()  # Avoid in-place modification
                individual[int(individual[j + 1, 0, 1]), :, i] = species2.detach().clone()  # Avoid in-place modification
                individual[j, :, i] = complex_.detach().clone()  # Avoid in-place modification

            # Update species degradation
            for j in range(0, num_species * 2, 2):
                degradation = apply_component_degradation(
                    initial_concentration=individual[j, :, i],
                    degradation_rate=parameters[f"species_{int((j / 2) + 1)}"][1],
                    time_step=time_step
                )
                individual[j, :, i] = degradation.detach().clone()  # Avoid in-place modification

            # Handle complex degradation
            for j in range(pair_start, pair_stop, 2):
                degradation = apply_component_degradation(
                    initial_concentration=individual[j, :, i],
                    degradation_rate=parameters[f"pair_{int((j / 2) - num_species + 1)}"][2],
                    time_step=time_step
                )
                individual[j, :, i] = degradation.detach().clone()  # Avoid in-place modification

            # Handle complex dissociation
            for j in range(pair_start, pair_stop, 2):
                species1, species2, complex_ = apply_complex_dissociation(
                    species1=individual[int(individual[j + 1, 0, 0]), :, i],
                    species2=individual[int(individual[j + 1, 0, 1]), :, i],
                    complex_=individual[j, :, i],
                    dissociation_rate=parameters[f"pair_{int((j / 2) - num_species + 1)}"][1],
                    time_step=time_step
                )
                individual[int(individual[j + 1, 0, 0]), :, i] = species1.detach().clone()  # Avoid in-place modification
                individual[int(individual[j + 1, 0, 1]), :, i] = species2.detach().clone()  # Avoid in-place modification
                individual[j, :, i] = complex_.detach().clone()  # Avoid in-place modification

            # Update species diffusion
            for j in range(0, num_species * 2, 2):
                diffusion = apply_diffusion(
                    current_concentration=individual[j, :, i],
                    compartment=individual[j, :, :],
                    column_position=i,
                    diffusion_rate=parameters[f"species_{int((j / 2) + 1)}"][2],
                    time_step=time_step
                )
                individual[j, :, i] = diffusion.detach().clone()  # Avoid in-place modification

            # Handle complex diffusion
            for j in range(pair_start, pair_stop, 2):
                diffusion = apply_diffusion(
                    current_concentration=individual[j, :, i],
                    compartment=individual[j, :, :],
                    column_position=i,
                    diffusion_rate=parameters[f"pair_{int((j / 2) - num_species + 1)}"][3],
                    time_step=time_step
                )
                individual[j, :, i] = diffusion.detach().clone()  # Avoid in-place modification

        epoch += 1

    return individual[comp_, :, :]



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
"""




