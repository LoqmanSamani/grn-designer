import random
import numpy as np


def apply_crossover(elite_agents, agent, crossover_alpha, simulation_crossover, initial_condition_crossover, parameter_crossover):

    crossover_beta = np.random.uniform(low=-crossover_alpha, high=1.0+crossover_alpha)

    if len(elite_agents) > 0:
        elite_agent = random.choice(elite_agents)
        if simulation_crossover:
            agent = apply_simulation_variable_crossover(
                elite_agent=elite_agent,
                agent=agent,
                beta=crossover_beta
            )

        if initial_condition_crossover:
            agent = apply_compartment_crossover(
                elite_agent=elite_agent,
                agent=agent,
                bata=crossover_beta
            )

        if parameter_crossover:
            agent = apply_parameter_crossover(
                elite_agent=elite_agent,
                agent=agent,
                beta=crossover_beta
            )
   
    if agent[-1, -1, 2] / agent[-1, -1, 3] > 200 or agent[-1, -1, 2] / agent[-1, -1, 3] < 70:
        agent[-1, -1, 2] = 20
        agent[-1, -1, 3] = 0.2

    return agent



def apply_simulation_variable_crossover(elite_agent, agent, beta):

    agent[-1, -1, 2:4] = (beta * agent[-1, -1, 2:4]) + ((1 - beta) * elite_agent[-1, -1, 2:4])

    return agent


def apply_compartment_crossover(elite_agent, agent, bata):

    num_species = int(agent[-1, -1, 0])

    for i in range(1, num_species * 2, 2):
        agent[i, :, :] = (bata * agent[i, :, :]) + ((1 - bata) * elite_agent[i, :, :])

    return agent


def apply_parameter_crossover(elite_agent, agent, beta):

    num_species = int(agent[-1, -1, 0])

    for i in range(0, num_species * 2, 2):
        num_params = int((agent[-1, i, -1]*3) + 3)

        new_params = (beta * agent[-1, i, :num_params]) + ((1 - beta) * elite_agent[-1, i, :num_params])

        for j in range(len(new_params)):
            if new_params[j] <= 0:
                new_params[j] = np.random.rand()

        new_params[:3] = np.clip(a=new_params[:3], a_min=0.010, a_max=0.999)
        new_params[3:] = np.clip(a=new_params[3:], a_min=0.010, a_max=2)

        agent[-1, i, :num_params] = new_params

    return agent


def filter_elite_agents(low_cost_agents, elite_agents, high_cost_agent):

    filtered_elite_agents = [ag for ag in elite_agents if ag.shape[0] == high_cost_agent.shape[0]]
    if len(filtered_elite_agents) == 0:
        filtered_elite_agents = [ag for ag in low_cost_agents if ag.shape[0] == high_cost_agent.shape[0]]
        filtered_elite_agents = filtered_elite_agents[: len(elite_agents)]

    return filtered_elite_agents






