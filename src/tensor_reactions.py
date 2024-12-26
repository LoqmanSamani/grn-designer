import torch

def apply_component_production(agent, parameters, num_species, time_step, column, species_index, species_num):

    initial_concentration = agent[species_index, :, column]
    production_pattern = parameters[f"initial_conditions_{species_num}"][:, column]
    production_rate = parameters[f"species_{species_num}"][0]
    basal_expression = (production_rate * time_step * production_pattern) + initial_concentration

    s = 1
    for k in range(0, num_species * 2, 2):
        t = agent[-1, k+1, :int(agent[-1, k, -1])]
        effect_mask = (t == species_index)
        if effect_mask.any():
            idx = torch.nonzero(effect_mask)[0].item()
            effect_type = agent[-1, k+1, -int(idx+1)]
            species_effect = agent[k, :, column]
            params = parameters[f"species_{s}"][3 + idx * 3 : 6 + idx * 3]

            basal_expression = apply_hill_effect(
                species_1=basal_expression,
                species_2=species_effect,
                rate=params[0],
                hill_coefficient=params[1],
                dissociation_constant=params[2],
                time_step=time_step,
                effect_type=effect_type
            )
        s += 1

    updated_concentration = torch.maximum(basal_expression, torch.tensor(0.0))
    return updated_concentration


def apply_hill_effect(species_1, species_2, rate, hill_coefficient, dissociation_constant, time_step, effect_type):

    hill_term = (species_2 ** hill_coefficient)
    denominator = dissociation_constant ** hill_coefficient + hill_term + 1e-8
    hill_effect = hill_term / denominator

    effect = rate * hill_effect * time_step
    if effect_type == 0:
        species_1 = torch.maximum(species_1 - effect, torch.tensor(0.0))
    elif effect_type == 1:
        species_1 = torch.maximum(species_1 + effect, torch.tensor(0.0))

    return species_1


def apply_component_degradation(initial_concentration, degradation_rate, time_step):

    updated_concentration = torch.maximum(initial_concentration - (initial_concentration * degradation_rate * time_step), torch.tensor(0.0))
    return updated_concentration

