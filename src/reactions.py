import numpy as np
from numba import jit



@jit(nopython=True)
def apply_component_production(agent, num_species, time_step, column, species_index):

    initial_concentration = agent[species_index, :, column].astype(np.float64)
    production_pattern = agent[species_index+1, :, column].astype(np.float64)
    production_rate = float(agent[-1, species_index, 0])
    basal_expression = (production_rate * time_step * production_pattern) + initial_concentration

    s = 1
    for k in range(0, num_species * 2, 2):
        t = agent[-1, k + 1, :int(agent[-1, k, -1])].astype(np.float64)
        if species_index in t:
            idx = np.where(t == species_index)[0][0]
            effect_type = int(agent[-1, k + 1, -int(idx + 1)])
            species_effect = agent[k, :, column].astype(np.float64)
            params = agent[-1, species_index, 3 + idx * 3: 6 + idx * 3].astype(np.float64)

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

    updated_concentration = np.maximum(basal_expression, 0.0)

    return updated_concentration


@jit(nopython=True)
def apply_hill_effect(species_1, species_2, rate, hill_coefficient, dissociation_constant, time_step, effect_type):

    species_1 = np.asarray(species_1, dtype=np.float64)
    species_2 = np.asarray(species_2, dtype=np.float64)
    rate = float(rate)
    hill_coefficient = float(hill_coefficient)
    dissociation_constant = float(dissociation_constant)
    time_step = float(time_step)

    hill_term = (species_2 ** hill_coefficient)
    denominator = dissociation_constant ** hill_coefficient + hill_term + 1e-8
    hill_effect = hill_term / denominator

    effect = rate * hill_effect * time_step
    if effect_type == 0:
        species_1 = np.maximum(species_1 - effect, 0.0)
    elif effect_type == 1:
        species_1 = np.maximum(species_1 + effect, 0.0)

    return species_1



@jit(nopython=True)
def apply_component_degradation(initial_concentration, degradation_rate, time_step):

    updated_concentration = np.maximum(initial_concentration - (initial_concentration * degradation_rate * time_step), 0.0)

    return updated_concentration






