from tensor_reactions import *
from tensor_diffusion import *

def tensor_simulation(agent, parameters, num_species, num_pairs, stop, time_step, max_epoch, compartment, device):
    agent = agent.to(device)

    z, y, x = agent.shape
    num_iters = int(x)
    pair_start = int(num_species * 2)
    pair_stop = int(pair_start + (num_pairs * 2))
    num_epochs = int(stop / time_step)
    comp_ = int(compartment * 2)

    epoch = 0
    while epoch <= max_epoch and epoch <= num_epochs:

        for i in range(num_iters):
            updated_individual = agent.clone()

            for j in range(0, num_species * 2, 2):
                h = int(j / 2) + 1
                updated_individual[j, :, i] = apply_component_production(
                    initial_concentration=agent[j, :, i],
                    production_pattern=parameters[f"compartment_{h}"][:, i],
                    production_rate=parameters[f"species_{h}"][0],
                    time_step=time_step
                )
            agent = updated_individual.clone()

            for j in range(pair_start, pair_stop, 2):
                species1, species2, complex_ = apply_species_collision(
                    species1=agent[int(agent[j + 1, 0, 0]), :, i],
                    species2=agent[int(agent[j + 1, 0, 1]), :, i],
                    complex_=agent[j, :, i],
                    collision_rate=parameters[f"pair_{int((j / 2) - num_species + 1)}"][0],
                    time_step=time_step
                )
                updated_individual[int(agent[j + 1, 0, 0]), :, i] = species1
                updated_individual[int(agent[j + 1, 0, 1]), :, i] = species2
                updated_individual[j, :, i] = complex_

            agent = updated_individual.clone()

            for j in range(0, num_species * 2, 2):
                updated_individual[j, :, i] = apply_component_degradation(
                    initial_concentration=agent[j, :, i],
                    degradation_rate=parameters[f"species_{int((j / 2) + 1)}"][1],
                    time_step=time_step
                )

            agent = updated_individual.clone()

            for j in range(pair_start, pair_stop, 2):
                updated_individual[j, :, i] = apply_component_degradation(
                    initial_concentration=agent[j, :, i],
                    degradation_rate=parameters[f"pair_{int((j / 2) - num_species + 1)}"][2],
                    time_step=time_step
                )
            agent = updated_individual.clone()

            for j in range(pair_start, pair_stop, 2):
                species1, species2, complex_ = apply_complex_dissociation(
                    species1=agent[int(agent[j + 1, 0, 0]), :, i],
                    species2=agent[int(agent[j + 1, 0, 1]), :, i],
                    complex_=agent[j, :, i],
                    dissociation_rate=parameters[f"pair_{int((j / 2) - num_species + 1)}"][1],
                    time_step=time_step
                )
                updated_individual[int(agent[j + 1, 0, 0]), :, i] = species1
                updated_individual[int(agent[j + 1, 0, 1]), :, i] = species2
                updated_individual[j, :, i] = complex_

            agent = updated_individual.clone()

            for j in range(0, num_species * 2, 2):
                updated_individual[j, :, i] = apply_diffusion(
                    current_concentration=agent[j, :, i],
                    compartment=agent[j, :, :],
                    column_position=i,
                    diffusion_rate=parameters[f"species_{int((j / 2) + 1)}"][2],
                    time_step=time_step
                )
            agent = updated_individual.clone()

            for j in range(pair_start, pair_stop, 2):
                updated_individual[j, :, i] = apply_diffusion(
                    current_concentration=agent[j, :, i],
                    compartment=agent[j, :, :],
                    column_position=i,
                    diffusion_rate=parameters[f"pair_{int((j / 2) - num_species + 1)}"][3],
                    time_step=time_step
                )

            agent = updated_individual.clone()

        epoch += 1

    return agent[comp_, :, :]

