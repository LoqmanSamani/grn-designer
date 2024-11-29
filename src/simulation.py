from reactions import *
from diffusion import *
from numba import jit


@jit(nopython=True)
def agent_simulation(agent):

    z, y, x = agent.shape
    num_iters = int(x)
    num_species = int(agent[-1, -1, 0])
    max_epoch = int(agent[-1, -1, 1])
    stop = int(agent[-1, -1, 2])
    time_step = agent[-1, -1, 3]
    num_epochs = int(stop / time_step)
    sim_results = np.zeros(shape=(num_species, y, x), dtype=np.float32)

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

            # update species activation & inhibition
            for j in range(0, num_species*2, 2):
                num_effects = int(agent[-1, j, -1])
                for k in range(num_effects):
                    effect_type = agent[-1, j+1, -int(k+1)]
                    effect_index = int(agent[-1, j+1, k])
                    if effect_type == 0:
                        agent[effect_index, :, i] = apply_component_inhibition(
                            species_1=agent[effect_index, :, i],
                            species_2=agent[j, :, i],
                            inhibition_rate=agent[-1, j, k+3],
                            time_step=time_step
                        )
                    elif effect_type == 1:
                        production_rate = agent[-1, effect_index, 0]
                        agent[effect_index, :, i] = apply_component_activation(
                            species_1=agent[effect_index, :, i],
                            species_2=agent[j, :, i],
                            production_pattern=agent[effect_index+1, :, i],
                            production_rate=production_rate,
                            activation_rate=agent[-1, j, k + 3],
                            time_step=time_step
                        )


            # Update species degradation
            for j in range(0, num_species*2, 2):
                agent[j, :, i] = apply_component_degradation(
                    initial_concentration=agent[j, :, i],
                    degradation_rate=agent[-1, j, 1],
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

        epoch += 1
    sp = 0
    for i in range(num_species):
        sim_results[i, :, :] = agent[sp, :, :]
        sp += 2

    return sim_results
