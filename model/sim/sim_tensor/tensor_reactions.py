import torch



def apply_component_production(initial_concentration, production_pattern, production_rate, time_step):

    production_pattern_ = torch.clamp(production_pattern, min=0.0)
    updated_concentration = torch.clamp(initial_concentration + (production_pattern_ * production_rate * time_step), min=0.0)

    return updated_concentration


def apply_component_degradation(initial_concentration, degradation_rate, time_step):

    updated_concentration = torch.clamp(initial_concentration - (initial_concentration * degradation_rate * time_step), min=0.0)

    return updated_concentration


def apply_species_collision(species1, species2, complex_, collision_rate, time_step):

    collision_effect = collision_rate * time_step
    complex_formed = torch.minimum(species1 * collision_effect, species2 * collision_effect)

    updated_species1 = torch.clamp(species1 - complex_formed, min=0.0)
    updated_species2 = torch.clamp(species2 - complex_formed, min=0.0)
    updated_complex = torch.clamp(complex_ + complex_formed, min=0.0)

    return updated_species1, updated_species2, updated_complex


def apply_complex_dissociation(species1, species2, complex_, dissociation_rate, time_step):

    dissociation_effect = dissociation_rate * time_step
    dissociated_amount = complex_ * dissociation_effect

    updated_complex = torch.clamp(complex_ - dissociated_amount, min=0.0)
    updated_species1 = torch.clamp(species1 + dissociated_amount, min=0.0)
    updated_species2 = torch.clamp(species2 + dissociated_amount, min=0.0)

    return updated_species1, updated_species2, updated_complex













"""
def apply_component_production(initial_concentration, production_pattern, production_rate, time_step):

    production_pattern_ = torch.maximum(production_pattern, torch.tensor(0.0, device=initial_concentration.device))
    updated_concentration = torch.maximum(initial_concentration + (production_pattern_ * production_rate * time_step), torch.tensor(0.0, device=initial_concentration.device))

    return updated_concentration


def apply_component_degradation(initial_concentration, degradation_rate, time_step):

    updated_concentration = torch.maximum(initial_concentration -
                                          (initial_concentration * degradation_rate * time_step),
                                          torch.tensor(0.0, device=initial_concentration.device))

    return updated_concentration


def apply_species_collision(species1, species2, complex_, collision_rate, time_step):

    collision_effect = collision_rate * time_step
    complex_formed = torch.minimum(species1 * collision_effect, species2 * collision_effect)

    updated_species1 = torch.maximum(species1 - complex_formed, torch.tensor(0.0, device=species1.device))
    updated_species2 = torch.maximum(species2 - complex_formed, torch.tensor(0.0, device=species2.device))
    updated_complex = torch.maximum(complex_ + complex_formed, torch.tensor(0.0, device=complex_.device))

    return updated_species1, updated_species2, updated_complex


def apply_complex_dissociation(species1, species2, complex_, dissociation_rate, time_step):

    dissociation_effect = dissociation_rate * time_step
    dissociated_amount = complex_ * dissociation_effect

    updated_complex = torch.maximum(complex_ - dissociated_amount, torch.tensor(0.0, device=complex_.device))
    updated_species1 = torch.maximum(species1 + dissociated_amount, torch.tensor(0.0, device=species1.device))
    updated_species2 = torch.maximum(species2 + dissociated_amount, torch.tensor(0.0, device=species2.device))

    return updated_species1, updated_species2, updated_complex
"""