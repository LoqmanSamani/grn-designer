from tensor_reactions import *
from tensor_diffusion import *



def agent_simulation(agent, parameters, num_species, stop, time_step, max_epoch, device):
    """
    Simulates the dynamics of an agent over time based on given parameters.

    This function models the behavior of an agent's components (e.g., species concentrations)
    through production, inhibition, activation, degradation, and diffusion processes.
    The simulation runs for a defined number of epochs or until a specified stopping time.

    Args:
        agent (torch.Tensor): The agent to simulate, represented as a 3D tensor
                              (components × height × width).
        parameters (dict): Dictionary containing parameter tensors for each species, including:
                           - `initial_conditions_{i}`: Initial conditions for species `i`.
                           - `species_{i}`: Parameters such as production rate, degradation rate,
                                            diffusion rate, and interaction rates for species `i`.
        num_species (int): Number of species in the agent.
        stop (float): Simulation stopping time.
        time_step (float): Time step size for the simulation.
        max_epoch (int): Maximum number of simulation epochs.
        device (str): The computation device (e.g., "cpu" or "cuda").

    Returns:
        torch.Tensor: The simulated agent's components over time, including:
                      - Updated species concentrations for each time step.
                      - Simulated patterns for the specified species.

    Workflow:
        1. Initialization:
           - Set up simulation parameters, including the number of iterations and epochs.

        2. Component Update:
           - Production: Update species concentrations based on production rates and patterns.
           - Inhibition and Activation: Apply interaction effects between species.
           - Degradation: Reduce species concentrations based on degradation rates.
           - Diffusion: Apply diffusion effects across spatial dimensions.

        3. Iteration:
           - Repeat the component updates for each time step in the simulation.

        4. Output:
           - Extract and return the simulated patterns for the specified species.

    Notes:
        - The simulation is constrained by both the maximum epoch and the stopping time.
        - Interaction effects (inhibition or activation) depend on species-specific parameters.
    """

    agent = agent.to(device)

    z, y, x = agent.shape
    num_iters = int(x)
    num_epochs = int(stop / time_step)
    patterns = [i for i in range(0, num_species*2, 2)]

    epoch = 0
    while epoch <= max_epoch or epoch <= num_epochs:
        for i in range(num_iters):
            updated_agent = agent.clone()

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
                inx = 3
                for k in range(num_effects):
                    effect_type = agent[-1, j + 1, -int(k + 1)]
                    effect_index = int(agent[-1, j + 1, k])
                    species_num_ = int((effect_index / 2) + 1)

                    if effect_type == 0:
                        updated_agent[effect_index, :, i] = apply_component_inhibition(
                            species_1=agent[effect_index, :, i],
                            species_2=agent[j, :, i],
                            inhibition_rate=parameters[f"species_{int(j / 2) + 1}"][inx],
                            hill_coefficient=parameters[f"species_{int(j / 2) + 1}"][inx+1],
                            dissociation_constant=parameters[f"species_{int(j / 2) + 1}"][inx+2],
                            time_step=time_step
                        )
                    elif effect_type == 1:
                        updated_agent[effect_index, :, i] = apply_component_activation(
                            species_1=agent[effect_index, :, i],
                            species_2=agent[j, :, i],
                            production_pattern=parameters[f"initial_conditions_{species_num_}"][:, i],
                            activation_rate=parameters[f"species_{int(j / 2) + 1}"][inx],
                            hill_coefficient=parameters[f"species_{int(j / 2) + 1}"][inx+1],
                            dissociation_constant=parameters[f"species_{int(j / 2) + 1}"][inx+2],
                            time_step=time_step
                        )
                    inx += 3

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

