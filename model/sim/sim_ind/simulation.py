from reactions import *
from diffusion import *
from numba import jit


@jit(nopython=True)
def agent_simulation(agent, num_patterns):

    z, y, x = agent.shape
    num_iters = int(x)
    num_species = int(agent[-1, -1, 0])
    num_pairs = int(agent[-1, -1, 1])
    max_epoch = int(agent[-1, -1, 2])
    stop = int(agent[-1, -1, 3])
    time_step = agent[-1, -1, 4]
    num_epochs = int(stop / time_step)
    pair_start = int(num_species * 2)
    pair_stop = int(pair_start + (num_pairs * 2))
    sim_results = np.zeros((num_patterns, y, x), dtype=np.float32)

    epoch = 0
    while epoch <= max_epoch or epoch <= num_epochs:

        for i in range(num_iters):

            for j in range(0, num_species*2, 2):
                agent[j, :, i] = apply_component_production(
                    initial_concentration=agent[j, :, i],
                    production_pattern=agent[j + 1, :, i],
                    production_rate=agent[-1, j, 0],
                    time_step=time_step
                )

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

            for j in range(0, num_species*2, 2):
                agent[j, :, i] = apply_component_degradation(
                    initial_concentration=agent[j, :, i],
                    degradation_rate=agent[-1, j, 1],
                    time_step=time_step
                )

            for j in range(pair_start, pair_stop, 2):
                agent[j, :, i] = apply_component_degradation(
                    initial_concentration=agent[j, :, i],
                    degradation_rate=agent[j + 1, 1, 2],
                    time_step=time_step
                )

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

            for j in range(0, num_species*2, 2):
                agent[j, :, i] = apply_diffusion(
                    current_concentration=agent[j, :, i],
                    compartment=agent[j, :, :],
                    column_position=i,
                    diffusion_rate=agent[-1, j, 2],
                    time_step=time_step
                )

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
