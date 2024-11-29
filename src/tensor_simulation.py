from tensor_reactions import *
from tensor_diffusion import *



def agent_simulation(agent, parameters, num_species, stop, time_step, max_epoch, device):
    agent = agent.to(device)

    z, y, x = agent.shape
    num_iters = int(x)
    num_epochs = int(stop / time_step)
    patterns = [i for i in range(0, num_species*2, 2)]

    epoch = 0
    while epoch <= max_epoch or epoch <= num_epochs:
        for i in range(num_iters):
            updated_agent = agent.clone()

            # Update species production
            for j in range(0, num_species * 2, 2):
                h = int(j / 2) + 1
                updated_agent[j, :, i] = apply_component_production(
                    initial_concentration=agent[j, :, i],
                    production_pattern=parameters[f"initial_conditions_{h}"][:, i],
                    production_rate=parameters[f"species_{h}"][0],
                    time_step=time_step
                )
            del h
            agent = updated_agent.clone()
            del updated_agent

            updated_agent = agent.clone()
            for j in range(0, num_species * 2, 2):
                num_effects = int(agent[-1, j, -1])

                for k in range(num_effects):
                    effect_type = agent[-1, j + 1, -int(k + 1)]
                    effect_index = int(agent[-1, j + 1, k])
                    species_num_ = int((effect_index / 2) + 1)

                    if effect_type == 0:
                        updated_agent[effect_index, :, i] = apply_component_inhibition(
                            species_1=agent[effect_index, :, i],
                            species_2=agent[j, :, i],
                            inhibition_rate=parameters[f"species_{int(j / 2) + 1}"][k + 3],
                            time_step=time_step
                        )
                    elif effect_type == 1:
                        updated_agent[effect_index, :, i] = apply_component_activation(
                            species_1=agent[effect_index, :, i],
                            species_2=agent[j, :, i],
                            production_pattern=parameters[f"initial_conditions_{species_num_}"][:, i],
                            production_rate=parameters[f"species_{species_num_}"][0],
                            activation_rate=parameters[f"species_{int(j / 2) + 1}"][k + 3],
                            time_step=time_step
                        )

            agent = updated_agent.clone()
            del updated_agent

            updated_agent = agent.clone()
            for j in range(0, num_species * 2, 2):
                updated_agent[j, :, i] = apply_component_degradation(
                    initial_concentration=agent[j, :, i],
                    degradation_rate=parameters[f"species_{int((j / 2) + 1)}"][1],
                    time_step=time_step
                )

            agent = updated_agent.clone()
            del updated_agent

            updated_agent = agent.clone()
            for j in range(0, num_species * 2, 2):
                updated_agent[j, :, i] = apply_diffusion(
                    current_concentration=agent[j, :, i],
                    init_conditions=agent[j, :, :],
                    column_position=i,
                    diffusion_rate=parameters[f"species_{int((j / 2) + 1)}"][2],
                    time_step=time_step
                )

            agent = updated_agent.clone()
            del updated_agent

        epoch += 1

    return agent[patterns, :, :]

