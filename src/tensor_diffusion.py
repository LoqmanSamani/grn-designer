import torch


def apply_diffusion(current_concentration, init_conditions, column_position, diffusion_rate, time_step):

    init_condition_size = init_conditions.shape[1]
    temporary_concentration = current_concentration.clone()

    if column_position == 0:
        temporary_concentration[0] = update_upper_left_corner_concentration(
            cell_concentration=current_concentration[0],
            lower_cell_concentration=init_conditions[1, 0],
            right_cell_concentration=init_conditions[0, 1],
            diffusion_rate=diffusion_rate,
            time_step=time_step
        )

        temporary_concentration[init_condition_size - 1] = update_lower_left_corner_concentration(
            cell_concentration=current_concentration[init_condition_size - 1],
            upper_cell_concentration=init_conditions[-2, 0],
            right_cell_concentration=init_conditions[-1, 1],
            diffusion_rate=diffusion_rate,
            time_step=time_step
        )

        temporary_concentration[1:-1] = update_left_side_concentration(
            cell_concentration=current_concentration[1:-1],
            upper_cell_concentration=init_conditions[:-2, 0],
            lower_cell_concentration=init_conditions[2:, 0],
            right_cell_concentration=init_conditions[1:-1, 1],
            diffusion_rate=diffusion_rate,
            time_step=time_step
        )

    elif column_position == init_condition_size - 1:
        temporary_concentration[0] = update_upper_right_corner_concentration(
            cell_concentration=current_concentration[0],
            lower_cell_concentration=init_conditions[1, -1],
            left_cell_concentration=init_conditions[0, -2],
            diffusion_rate=diffusion_rate,
            time_step=time_step
        )

        temporary_concentration[init_condition_size - 1] = update_lower_right_corner_concentration(
            cell_concentration=current_concentration[-1],
            upper_cell_concentration=init_conditions[-2, -1],
            left_cell_concentration=init_conditions[-1, -2],
            diffusion_rate=diffusion_rate,
            time_step=time_step
        )

        temporary_concentration[1:-1] = update_right_side_concentration(
            cell_concentration=current_concentration[1:-1],
            upper_cell_concentration=init_conditions[:-2, -1],
            lower_cell_concentration=init_conditions[2:, -1],
            left_cell_concentration=init_conditions[1:-1, -2],
            diffusion_rate=diffusion_rate,
            time_step=time_step
        )

    else:
        temporary_concentration[0] = update_central_concentration_upper(
            cell_concentration=current_concentration[0],
            lower_cell_concentration=init_conditions[1, column_position],
            right_cell_concentration=init_conditions[0, column_position + 1],
            left_cell_concentration=init_conditions[0, column_position - 1],
            diffusion_rate=diffusion_rate,
            time_step=time_step
        )

        temporary_concentration[init_condition_size - 1] = update_central_concentration_lower(
            cell_concentration=current_concentration[init_condition_size - 1],
            upper_cell_concentration=init_conditions[-2, column_position],
            right_cell_concentration=init_conditions[-1, column_position + 1],
            left_cell_concentration=init_conditions[-1, column_position - 1],
            diffusion_rate=diffusion_rate,
            time_step=time_step
        )

        temporary_concentration[1:init_condition_size - 1] = update_central_concentration_middle(
            cell_concentration=current_concentration[1:init_condition_size - 1],
            upper_cell_concentration=init_conditions[:-2, column_position],
            lower_cell_concentration=init_conditions[2:, column_position],
            right_cell_concentration=init_conditions[1:-1, column_position + 1],
            left_cell_concentration=init_conditions[1:-1, column_position - 1],
            diffusion_rate=diffusion_rate,
            time_step=time_step
        )

    updated_concentration = torch.maximum(temporary_concentration, torch.tensor(0.0))

    return updated_concentration


def update_lower_left_corner_concentration(
    cell_concentration,
    upper_cell_concentration,
    right_cell_concentration,
    diffusion_rate,
    time_step
):
    in_diffusion = (time_step * upper_cell_concentration * diffusion_rate) + \
                   (time_step * right_cell_concentration * diffusion_rate)
    out_diffusion = time_step * cell_concentration * diffusion_rate * 2

    updated_concentration = cell_concentration + in_diffusion - out_diffusion
    return updated_concentration.unsqueeze(0)


def update_lower_right_corner_concentration(
    cell_concentration,
    upper_cell_concentration,
    left_cell_concentration,
    diffusion_rate,
    time_step
):
    in_diffusion = (time_step * upper_cell_concentration * diffusion_rate) + \
                   (time_step * left_cell_concentration * diffusion_rate)
    out_diffusion = time_step * cell_concentration * diffusion_rate * 2

    updated_concentration = cell_concentration + in_diffusion - out_diffusion
    return updated_concentration.unsqueeze(0)


def update_upper_left_corner_concentration(
    cell_concentration,
    lower_cell_concentration,
    right_cell_concentration,
    diffusion_rate,
    time_step
):
    in_diffusion = (time_step * lower_cell_concentration * diffusion_rate) + \
                   (time_step * right_cell_concentration * diffusion_rate)
    out_diffusion = time_step * cell_concentration * diffusion_rate * 2

    updated_concentration = cell_concentration + in_diffusion - out_diffusion
    return updated_concentration.unsqueeze(0)


def update_upper_right_corner_concentration(
    cell_concentration,
    lower_cell_concentration,
    left_cell_concentration,
    diffusion_rate,
    time_step
):
    in_diffusion = (time_step * lower_cell_concentration * diffusion_rate) + \
                   (time_step * left_cell_concentration * diffusion_rate)
    out_diffusion = time_step * cell_concentration * diffusion_rate * 2

    updated_concentration = cell_concentration + in_diffusion - out_diffusion
    return updated_concentration.unsqueeze(0)


def update_left_side_concentration(
    cell_concentration,
    upper_cell_concentration,
    lower_cell_concentration,
    right_cell_concentration,
    diffusion_rate,
    time_step
):
    upper_cell_in = time_step * upper_cell_concentration * diffusion_rate
    lower_cell_in = time_step * lower_cell_concentration * diffusion_rate
    right_cell_in = time_step * right_cell_concentration * diffusion_rate

    in_diffusion = upper_cell_in + lower_cell_in + right_cell_in
    out_diffusion = time_step * cell_concentration * diffusion_rate * 3

    updated_concentration = cell_concentration + in_diffusion - out_diffusion

    return updated_concentration


def update_right_side_concentration(
    cell_concentration,
    upper_cell_concentration,
    lower_cell_concentration,
    left_cell_concentration,
    diffusion_rate,
    time_step
):
    upper_cell_in = time_step * upper_cell_concentration * diffusion_rate
    lower_cell_in = time_step * lower_cell_concentration * diffusion_rate
    left_cell_in = time_step * left_cell_concentration * diffusion_rate

    in_diffusion = upper_cell_in + lower_cell_in + left_cell_in
    out_diffusion = time_step * cell_concentration * diffusion_rate * 3

    updated_concentration = cell_concentration + in_diffusion - out_diffusion

    return updated_concentration


def update_central_concentration_middle(
    cell_concentration,
    upper_cell_concentration,
    lower_cell_concentration,
    right_cell_concentration,
    left_cell_concentration,
    diffusion_rate,
    time_step
):
    upper_cell_in = time_step * upper_cell_concentration * diffusion_rate
    lower_cell_in = time_step * lower_cell_concentration * diffusion_rate
    right_cell_in = time_step * right_cell_concentration * diffusion_rate
    left_cell_in = time_step * left_cell_concentration * diffusion_rate

    in_diffusion = upper_cell_in + lower_cell_in + right_cell_in + left_cell_in
    out_diffusion = time_step * cell_concentration * diffusion_rate * 4

    updated_concentration = cell_concentration + in_diffusion - out_diffusion

    return updated_concentration


def update_central_concentration_upper(
    cell_concentration,
    lower_cell_concentration,
    right_cell_concentration,
    left_cell_concentration,
    diffusion_rate,
    time_step
):
    lower_cell_in = time_step * lower_cell_concentration * diffusion_rate
    right_cell_in = time_step * right_cell_concentration * diffusion_rate
    left_cell_in = time_step * left_cell_concentration * diffusion_rate

    in_diffusion = lower_cell_in + right_cell_in + left_cell_in
    out_diffusion = time_step * cell_concentration * diffusion_rate * 3

    updated_concentration = cell_concentration + in_diffusion - out_diffusion

    return updated_concentration


def update_central_concentration_lower(
    cell_concentration,
    upper_cell_concentration,
    right_cell_concentration,
    left_cell_concentration,
    diffusion_rate,
    time_step
):
    upper_cell_in = time_step * upper_cell_concentration * diffusion_rate
    right_cell_in = time_step * right_cell_concentration * diffusion_rate
    left_cell_in = time_step * left_cell_concentration * diffusion_rate

    in_diffusion = upper_cell_in + right_cell_in + left_cell_in
    out_diffusion = time_step * cell_concentration * diffusion_rate * 3

    updated_concentration = cell_concentration + in_diffusion - out_diffusion

    return updated_concentration



