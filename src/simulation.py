from reactions import *
from diffusion import *
from numba import jit


@jit(nopython=True)
def agent_simulation(agent):

    agent = np.asarray(agent, dtype=np.float64)
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

            for j in range(0, num_species * 2, 2):
                agent[j, :, i] = apply_component_production(
                    agent=agent,
                    num_species=num_species,
                    time_step=time_step,
                    column=i,
                    species_index=j
                )

            for j in range(0, num_species*2, 2):
                agent[j, :, i] = apply_component_degradation(
                    initial_concentration=agent[j, :, i],
                    degradation_rate=agent[-1, j, 1],
                    time_step=time_step
                )

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
