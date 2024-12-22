import random
import numpy as np


def apply_crossover(elite_agents, agent, crossover_alpha, simulation_crossover, initial_condition_crossover, parameter_crossover):
    """
    Applies crossover operations to an agent using a randomly chosen elite agent.

    This function performs crossover on various components of the agent, including
    simulation parameters, initial conditions, and gene interaction parameters.
    The crossover operation combines values from the input agent and a selected elite agent
    based on a weighted factor `beta`.

    Args:
        elite_agents (list): List of elite agents available for crossover.
        agent (np.ndarray): The agent to which crossover is applied.
        crossover_alpha (float): Factor controlling the range of the crossover weight `beta`.
                                 The value of `beta` is drawn from `[-crossover_alpha, 1 + crossover_alpha]`.
        simulation_crossover (bool): If True, applies crossover to simulation parameters.
        initial_condition_crossover (bool): If True, applies crossover to initial conditions.
        parameter_crossover (bool): If True, applies crossover to gene interaction parameters.

    Returns:
        np.ndarray: The agent after crossover operations.

    Notes:
        - Ensures simulation duration to time-step ratio remains within a valid range.
        - Defaults simulation parameters to predefined values if the ratio is out of bounds.
    """

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
    """
    Applies crossover to simulation parameters of an agent.

    This function blends the simulation duration and time-step parameters of the input
    agent with those of an elite agent using a weighted factor `beta`.

    Args:
        elite_agent (np.ndarray): The elite agent used for crossover.
        agent (np.ndarray): The agent to which crossover is applied.
        beta (float): Weight factor for blending parameters.

    Returns:
        np.ndarray: The agent with updated simulation parameters.
    """

    agent[-1, -1, 2:4] = (beta * agent[-1, -1, 2:4]) + ((1 - beta) * elite_agent[-1, -1, 2:4])

    return agent


def apply_compartment_crossover(elite_agent, agent, bata):
    """
    Applies crossover to the initial conditions of an agent's compartments.

    This function combines the compartment values of the input agent with those of
    an elite agent using a weighted factor `bata`.

    Args:
        elite_agent (np.ndarray): The elite agent used for crossover.
        agent (np.ndarray): The agent to which crossover is applied.
        bata (float): Weight factor for blending compartment values.

    Returns:
        np.ndarray: The agent with updated compartment initial conditions.
    """

    num_species = int(agent[-1, -1, 0])

    for i in range(1, num_species * 2, 2):
        agent[i, :, :] = (bata * agent[i, :, :]) + ((1 - bata) * elite_agent[i, :, :])

    return agent


def apply_parameter_crossover(elite_agent, agent, beta):
    """
    Applies crossover to the gene and interaction parameters of an agent.

    This function combines the parameter values of the input agent with those of an elite
    agent using a weighted factor `beta`. It ensures parameter values are within valid
    bounds and replaces invalid values with random values.

    Args:
        elite_agent (np.ndarray): The elite agent used for crossover.
        agent (np.ndarray): The agent to which crossover is applied.
        beta (float): Weight factor for blending parameter values.

    Returns:
        np.ndarray: The agent with updated gene and interaction parameters.

    Notes:
        - Ensures the resulting parameter values are clipped to valid ranges.
        - Replaces negative or invalid parameter values with randomly generated values.
    """

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
    """
    Filters elite agents based on compatibility with a high-cost agent.

    This function identifies elite agents whose structures match the high-cost agent.
    If no compatible elite agents are found, low-cost agents are used as substitutes,
    limited to the number of elite agents.

    Args:
        low_cost_agents (list): List of low-cost agents available as substitutes.
        elite_agents (list): List of elite agents to filter.
        high_cost_agent (np.ndarray): The high-cost agent used for compatibility checks.

    Returns:
        list: A filtered list of elite agents compatible with the high-cost agent.
    """

    filtered_elite_agents = [ag for ag in elite_agents if ag.shape[0] == high_cost_agent.shape[0]]
    if len(filtered_elite_agents) == 0:
        filtered_elite_agents = [ag for ag in low_cost_agents if ag.shape[0] == high_cost_agent.shape[0]]
        filtered_elite_agents = filtered_elite_agents[: len(elite_agents)]

    return filtered_elite_agents






