import torch



def apply_diffusion(current_concentration, compartment, column_position, diffusion_rate, time_step):
    """
    Apply diffusion to update the concentration of species in a specific column of a 2D compartment for all individuals in the population.

    Parameters:
        - current_concentration (torch.Tensor): Tensor of current concentrations for each cell in the column.
        - compartment (torch.Tensor): Tensor representing the 2D compartment where diffusion takes place.
        - column_position (int): Column position of the cells being updated (0-based index).
        - diffusion_rate (float): Rate at which the species diffuses between cells.
        - time_step (float): Discrete time step for the calculation.

    Returns:
        - torch.Tensor: Tensor with updated concentrations.
    """
    compartment_size = compartment.shape[1]
    temporary_concentration = current_concentration.clone()

    if column_position == 0:
        temporary_concentration[0] = update_upper_left_corner_concentration(
            cell_concentration=current_concentration[0],
            lower_cell_concentration=compartment[1, 0],
            right_cell_concentration=compartment[0, 1],
            diffusion_rate=diffusion_rate,
            time_step=time_step
        )

        temporary_concentration[compartment_size - 1] = update_lower_left_corner_concentration(
            cell_concentration=current_concentration[compartment_size - 1],
            upper_cell_concentration=compartment[-2, 0],
            right_cell_concentration=compartment[-1, 1],
            diffusion_rate=diffusion_rate,
            time_step=time_step
        )

        temporary_concentration[1:-1] = update_left_side_concentration(
            cell_concentration=current_concentration[1:-1],
            upper_cell_concentration=compartment[:-2, 0],
            lower_cell_concentration=compartment[2:, 0],
            right_cell_concentration=compartment[1:-1, 1],
            diffusion_rate=diffusion_rate,
            time_step=time_step
        )

    elif column_position == compartment_size - 1:
        temporary_concentration[0] = update_upper_right_corner_concentration(
            cell_concentration=current_concentration[0],
            lower_cell_concentration=compartment[1, -1],
            left_cell_concentration=compartment[0, -2],
            diffusion_rate=diffusion_rate,
            time_step=time_step
        )

        temporary_concentration[compartment_size - 1] = update_lower_right_corner_concentration(
            cell_concentration=current_concentration[-1],
            upper_cell_concentration=compartment[-2, -1],
            left_cell_concentration=compartment[-1, -2],
            diffusion_rate=diffusion_rate,
            time_step=time_step
        )

        temporary_concentration[1:-1] = update_right_side_concentration(
            cell_concentration=current_concentration[1:-1],
            upper_cell_concentration=compartment[:-2, -1],
            lower_cell_concentration=compartment[2:, -1],
            left_cell_concentration=compartment[1:-1, -2],
            diffusion_rate=diffusion_rate,
            time_step=time_step
        )

    else:
        temporary_concentration[0] = update_central_concentration_upper(
            cell_concentration=current_concentration[0],
            lower_cell_concentration=compartment[1, column_position],
            right_cell_concentration=compartment[0, column_position + 1],
            left_cell_concentration=compartment[0, column_position - 1],
            diffusion_rate=diffusion_rate,
            time_step=time_step
        )

        temporary_concentration[compartment_size - 1] = update_central_concentration_lower(
            cell_concentration=current_concentration[compartment_size - 1],
            upper_cell_concentration=compartment[-2, column_position],
            right_cell_concentration=compartment[-1, column_position + 1],
            left_cell_concentration=compartment[-1, column_position - 1],
            diffusion_rate=diffusion_rate,
            time_step=time_step
        )

        temporary_concentration[1:compartment_size - 1] = update_central_concentration_middle(
            cell_concentration=current_concentration[1:compartment_size - 1],
            upper_cell_concentration=compartment[:-2, column_position],
            lower_cell_concentration=compartment[2:, column_position],
            right_cell_concentration=compartment[1:-1, column_position + 1],
            left_cell_concentration=compartment[1:-1, column_position - 1],
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
    return updated_concentration.unsqueeze(0)  # Reshape to match expected output shape


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
    return updated_concentration.unsqueeze(0)  # Reshape to match expected output shape


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
    return updated_concentration.unsqueeze(0)  # Reshape to match expected output shape


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
    return updated_concentration.unsqueeze(0)  # Reshape to match expected output shape


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








"""
import tensorflow as tf




def apply_diffusion(current_concentration, compartment, column_position, diffusion_rate, time_step):

    compartment_size = compartment.shape[1]
    temporary_concentration = tf.identity(current_concentration)

    if column_position == 0:

        temporary_concentration = tf.tensor_scatter_nd_update(
            tensor=temporary_concentration,
            indices=[[0]],
            updates=update_upper_left_corner_concentration(
                cell_concentration=current_concentration[0],
                lower_cell_concentration=compartment[1, 0],
                right_cell_concentration=compartment[0, 1],
                diffusion_rate=diffusion_rate,
                time_step=time_step
            )
        )

        temporary_concentration = tf.tensor_scatter_nd_update(
            tensor=temporary_concentration,
            indices=[[compartment_size-1]],
            updates=update_lower_left_corner_concentration(
                cell_concentration=current_concentration[compartment_size-1],
                upper_cell_concentration=compartment[-2, 0],
                right_cell_concentration=compartment[-1, 1],
                diffusion_rate=diffusion_rate,
                time_step=time_step
            )
        )

        temporary_concentration = tf.tensor_scatter_nd_update(
            tensor=temporary_concentration,
            indices=list([i] for i in range(1, compartment_size - 1)),
            updates=update_left_side_concentration(
                cell_concentration=current_concentration[1:-1],
                upper_cell_concentration=compartment[:-2, 0],
                lower_cell_concentration=compartment[2:, 0],
                right_cell_concentration=compartment[1:-1, 1],
                diffusion_rate=diffusion_rate,
                time_step=time_step
            )
        )

    elif column_position == compartment_size - 1:

        temporary_concentration = tf.tensor_scatter_nd_update(
            tensor=temporary_concentration,
            indices=[[0]],
            updates=update_upper_right_corner_concentration(
                cell_concentration=current_concentration[0],
                lower_cell_concentration=compartment[1, -1],
                left_cell_concentration=compartment[0, -2],
                diffusion_rate=diffusion_rate,
                time_step=time_step
            )
        )

        temporary_concentration = tf.tensor_scatter_nd_update(
            tensor=temporary_concentration,
            indices=[[compartment_size-1]],
            updates=update_lower_right_corner_concentration(
                cell_concentration=current_concentration[-1],
                upper_cell_concentration=compartment[-2, -1],
                left_cell_concentration=compartment[-1, -2],
                diffusion_rate=diffusion_rate,
                time_step=time_step
            )
        )

        temporary_concentration = tf.tensor_scatter_nd_update(
            tensor=temporary_concentration,
            indices=list([i] for i in range(1, compartment_size - 1)),
            updates=update_right_side_concentration(
                cell_concentration=current_concentration[1:-1],
                upper_cell_concentration=compartment[0:-2, -1],
                lower_cell_concentration=compartment[2:, -1],
                left_cell_concentration=compartment[1:-1, -2],
                diffusion_rate=diffusion_rate,
                time_step=time_step
            )
        )

    else:
        temporary_concentration = tf.tensor_scatter_nd_update(
            tensor=temporary_concentration,
            indices=[[0]],
            updates=update_central_concentration_upper(
                cell_concentration=current_concentration[0],
                lower_cell_concentration=compartment[1, column_position],
                right_cell_concentration=compartment[0, column_position + 1],
                left_cell_concentration=compartment[0, column_position - 1],
                diffusion_rate=diffusion_rate,
                time_step=time_step
            )
        )

        temporary_concentration = tf.tensor_scatter_nd_update(
            tensor=temporary_concentration,
            indices=[[compartment_size-1]],
            updates=update_central_concentration_lower(
                cell_concentration=current_concentration[compartment_size-1],
                upper_cell_concentration=compartment[-2, column_position],
                right_cell_concentration=compartment[-1, column_position+1],
                left_cell_concentration=compartment[-1, column_position-1],
                diffusion_rate=diffusion_rate,
                time_step=time_step
            )
        )

        temporary_concentration = tf.tensor_scatter_nd_update(
            tensor=temporary_concentration,
            indices=list([i] for i in range(1, compartment_size-1)),
            updates=update_central_concentration_middle(
                cell_concentration=current_concentration[1: compartment_size-1],
                upper_cell_concentration=compartment[0:-2, column_position],
                lower_cell_concentration=compartment[2:, column_position],
                right_cell_concentration=compartment[1:-1, column_position+1],
                left_cell_concentration=compartment[1:-1, column_position-1],
                diffusion_rate=diffusion_rate,
                time_step=time_step
            )
        )

    updated_concentration = tf.maximum(temporary_concentration, 0.0)

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
    updated_concentration = tf.reshape(tensor=updated_concentration, shape=(1,))

    return updated_concentration



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
    updated_concentration = tf.reshape(tensor=updated_concentration, shape=(1,))

    return updated_concentration



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
    updated_concentration = tf.reshape(tensor=updated_concentration, shape=(1,))

    return updated_concentration



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
    updated_concentration = tf.reshape(tensor=updated_concentration, shape=(1,))

    return updated_concentration




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
    right_cell_in = time_step * left_cell_concentration * diffusion_rate

    in_diffusion = upper_cell_in + lower_cell_in + right_cell_in
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
    updated_concentration = tf.reshape(tensor=updated_concentration, shape=(1,))

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
    updated_concentration = tf.reshape(tensor=updated_concentration, shape=(1,))

    return updated_concentration
"""















