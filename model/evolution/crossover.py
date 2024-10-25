import random


def apply_crossover(elite_agents, agent, crossover_alpha, simulation_crossover, initial_condition_crossover, parameter_crossover):

    if len(elite_agents) > 0:
        elite_agent = random.choice(elite_agents)
        if simulation_crossover:
            agent = apply_simulation_variable_crossover(
                elite_agent=elite_agent,
                agent=agent,
                alpha=crossover_alpha
            )

        if initial_condition_crossover:
            agent = apply_compartment_crossover(
                elite_agent=elite_agent,
                agent=agent,
                alpha=crossover_alpha
            )

        if parameter_crossover:
            agent = apply_parameter_crossover(
                elite_agent=elite_agent,
                agent=agent,
                alpha=crossover_alpha
            )
   
    if agent[-1, -1, 3] / agent[-1, -1, 4] > 200 or agent[-1, -1, 3] / agent[-1, -1, 4] < 70:
        agent[-1, -1, 3] = 20
        agent[-1, -1, 4] = 0.2

    return agent



def apply_simulation_variable_crossover(elite_agent, agent, alpha):

    agent[-1, -1, 3:5] = (alpha * agent[-1, -1, 3:5]) + ((1 - alpha) * elite_agent[-1, -1, 3:5])

    return agent


def apply_compartment_crossover(elite_agent, agent, alpha):

    num_species = int(agent[-1, -1, 0])

    for i in range(1, num_species * 2 + 1, 2):
        agent[i, :, :] = (alpha * agent[i, :, :]) + ((1 - alpha) * elite_agent[i, :, :])

    return agent


def apply_parameter_crossover(elite_agent, agent, alpha):

    num_species = int(agent[-1, -1, 0])
    num_pairs = int(agent[-1, -1, 1])
    pair_start = int(num_species * 2)
    pair_stop = int(pair_start + (num_pairs * 2))

    for i in range(0, num_species * 2, 2):
        agent[-1, i, :3] = (alpha * agent[-1, i, :3]) + ((1 - alpha) * elite_agent[-1, i, :3])

    for i in range(pair_start + 1, pair_stop + 1, 2):
        agent[i, 1, :4] = (alpha * agent[i, 1, :4]) + ((1 - alpha) * elite_agent[i, 1, :4])

    return agent




def filter_elite_agents(low_cost_agents, elite_agents, high_cost_agent):

    filtered_elite_agents = [ag for ag in elite_agents if ag.shape[0] == high_cost_agent.shape[0]]
    if len(filtered_elite_agents) == 0:
        filtered_elite_agents = [ag for ag in low_cost_agents if ag.shape[0] == high_cost_agent.shape[0]]
        filtered_elite_agents = filtered_elite_agents[: len(elite_agents)]

    return filtered_elite_agents






