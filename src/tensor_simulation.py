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

            sp = 1
            for j in range(0, num_species * 2, 2):

                updated_agent[j, :, i] = apply_component_production(
                    agent=agent,
                    parameters=parameters,
                    num_species=num_species,
                    time_step=time_step,
                    column=i,
                    species_index=j,
                    species_num=sp
                )
                sp += 1

            agent = updated_agent.clone()
            del updated_agent, sp

            updated_agent = agent.clone()
            sp = 1
            for j in range(0, num_species * 2, 2):
                updated_agent[j, :, i] = apply_component_degradation(
                    initial_concentration=agent[j, :, i],
                    degradation_rate=parameters[f"species_{sp}"][1],
                    time_step=time_step
                )
                sp += 1

            agent = updated_agent.clone()
            del updated_agent, sp

            updated_agent = agent.clone()
            sp = 1
            for j in range(0, num_species * 2, 2):
                updated_agent[j, :, i] = apply_diffusion(
                    current_concentration=agent[j, :, i],
                    init_conditions=agent[j, :, :],
                    column_position=i,
                    diffusion_rate=parameters[f"species_{sp}"][2],
                    time_step=time_step
                )
                sp += 1

            agent = updated_agent.clone()
            del updated_agent, sp

        epoch += 1

    return agent[patterns, :, :]

