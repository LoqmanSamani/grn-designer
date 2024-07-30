from numba import jit
import numpy as np


@jit(nopython=True)
def apply_diffusion(current_concentration, compartment, column_position, diffusion_rate, time_step):
    """
    Apply diffusion to update the concentration of species in a specific column of a 2D compartment.

    Parameters:
    - current_concentration (numpy.ndarray): Array of current concentrations for each cell.
    - compartment (numpy.ndarray): Array representing the 2D compartment where diffusion takes place.
    - column_position (int): Column position of the cells being updated (0-based index).
    - diffusion_rate (float): Rate at which the species diffuses between cells.
    - time_step (float): Discrete time step for the calculation.

    Returns:
    - None: The function updates the `current_concentration` array in place.
    """
    compartment_size = compartment.shape[1]

    if column_position == 0:
        # Update concentration for the upper-left corner cell
        current_concentration[0] = update_upper_left_corner_concentration(
            cell_concentration=current_concentration[0],
            lower_cell_concentration=compartment[1, 0],
            right_cell_concentration=compartment[0, 1],
            diffusion_rate=diffusion_rate,
            time_step=time_step
        )
        # Update concentration for the lower-left corner cell
        current_concentration[-1] = update_lower_left_corner_concentration(
            cell_concentration=current_concentration[-1],
            upper_cell_concentration=compartment[-2, 0],
            right_cell_concentration=compartment[-1, 1],
            diffusion_rate=diffusion_rate,
            time_step=time_step
        )
        # Update concentrations for the left-side cells (excluding corners)
        current_concentration[1:-1] = update_left_side_concentration(
            cell_concentration=current_concentration[1:-1],
            upper_cell_concentration=compartment[0:-2, 0],
            lower_cell_concentration=compartment[2:, 0],
            right_cell_concentration=compartment[1:-1, 1],
            diffusion_rate=diffusion_rate,
            time_step=time_step
        )

    elif column_position == compartment_size - 1:
        # Update concentration for the upper-right corner cell
        current_concentration[0] = update_upper_right_corner_concentration(
            cell_concentration=current_concentration[0],
            lower_cell_concentration=compartment[1, -1],
            left_cell_concentration=compartment[0, -2],
            diffusion_rate=diffusion_rate,
            time_step=time_step
        )

        # Update concentration for the lower-right corner cell
        current_concentration[-1] = update_lower_right_corner_concentration(
            cell_concentration=current_concentration[-1],
            upper_cell_concentration=compartment[-2, -1],
            left_cell_concentration=compartment[-1, -2],
            diffusion_rate=diffusion_rate,
            time_step=time_step
        )
        # Update concentrations for the left-side cells (excluding corners)
        current_concentration[1:-1] = update_right_side_concentration(
            cell_concentration=current_concentration[1:-1],
            upper_cell_concentration=compartment[0:-2, -1],
            lower_cell_concentration=compartment[2:, -1],
            left_cell_concentration=compartment[1:-1, -2],
            diffusion_rate=diffusion_rate,
            time_step=time_step
        )


    else:

        current_concentration[0] = update_central_concentration_upper(
            cell_concentration=current_concentration[0],
            lower_cell_concentration=current_concentration[1],
            right_cell_concentration=compartment[0, column_position + 1],
            left_cell_concentration=compartment[0, column_position - 1],
            diffusion_rate=diffusion_rate,
            time_step=time_step
        )

        current_concentration[-1] = update_central_concentration_lower(
            cell_concentration=current_concentration[-1],
            upper_cell_concentration=current_concentration[-2],
            right_cell_concentration=compartment[-1, column_position + 1],
            left_cell_concentration=compartment[-1, column_position - 1],
            diffusion_rate=diffusion_rate,
            time_step=time_step
        )

        current_concentration[1:-1] = update_central_concentration_middle(
            cell_concentration=current_concentration[1:-1],
            upper_cell_concentration=compartment[0:-2, column_position],
            lower_cell_concentration=compartment[2:, column_position],
            right_cell_concentration=compartment[1:-1, column_position+1],
            left_cell_concentration=compartment[1:-1, column_position-1],
            diffusion_rate=diffusion_rate,
            time_step=time_step
        )

    current_concentration = np.maximum(current_concentration, 0)

    return current_concentration




@jit(nopython=True)
def update_lower_left_corner_concentration(
    cell_concentration,
    upper_cell_concentration,
    right_cell_concentration,
    diffusion_rate,
    time_step):
    """
    Update the concentration of a species in a cell located at the lower-left corner of a 2D compartment
    based on diffusion from neighboring cells.

    Parameters:
    - cell_concentration (float): Concentration of the species in the lower-left corner cell.
    - upper_cell_concentration (float): Concentration of the species in the cell directly above the lower-left cell.
    - right_cell_concentration (float): Concentration of the species in the cell directly to the right of the lower-left cell.
    - diffusion_rate (float): Rate at which the species diffuses between cells.
    - time_step (float): Discrete time step for the calculation.

    Returns:
    - float: Updated concentration of the species in the lower-left corner cell.
    """
    in_diffusion = (time_step * upper_cell_concentration * diffusion_rate) + \
                   (time_step * right_cell_concentration * diffusion_rate)
    out_diffusion = time_step * cell_concentration * diffusion_rate * 2

    updated_concentration = cell_concentration + in_diffusion - out_diffusion

    return updated_concentration


@jit(nopython=True)
def update_lower_right_corner_concentration(
    cell_concentration,
    upper_cell_concentration,
    left_cell_concentration,
    diffusion_rate,
    time_step):
    """
        Update the concentration of a species in a cell located at the lower-right corner of a 2D compartment
        based on diffusion from neighboring cells.

        Parameters:
        - cell_concentration (float): Concentration of the species in the lower-left corner cell.
        - upper_cell_concentration (float): Concentration of the species in the cell directly above the lower-right cell.
        - left_cell_concentration (float): Concentration of the species in the cell directly to the left of the lower-right cell.
        - diffusion_rate (float): Rate at which the species diffuses between cells.
        - time_step (float): Discrete time step for the calculation.

        Returns:
        - float: Updated concentration of the species in the lower-left corner cell.
        """
    in_diffusion = (time_step * upper_cell_concentration * diffusion_rate) + \
                   (time_step * left_cell_concentration * diffusion_rate)
    out_diffusion = time_step * cell_concentration * diffusion_rate * 2

    updated_concentration = cell_concentration + in_diffusion - out_diffusion

    return updated_concentration


@jit(nopython=True)
def update_upper_left_corner_concentration(
    cell_concentration,
    lower_cell_concentration,
    right_cell_concentration,
    diffusion_rate,
    time_step):
    """
    Update the concentration of a species in a cell located at the upper-left corner of a 2D compartment
    based on diffusion from neighboring cells.

    Parameters:
    - cell_concentration (float): Concentration of the species in the upper-left corner cell.
    - lower_cell_concentration (float): Concentration of the species in the cell directly below the upper-left cell.
    - right_cell_concentration (float): Concentration of the species in the cell directly to the right of the upper-left cell.
    - diffusion_rate (float): Rate at which the species diffuses between cells.
    - time_step (float): Discrete time step for the calculation.

    Returns:
    - float: Updated concentration of the species in the upper-left corner cell.
    """
    in_diffusion = (time_step * lower_cell_concentration * diffusion_rate) + \
                   (time_step * right_cell_concentration * diffusion_rate)
    out_diffusion = time_step * cell_concentration * diffusion_rate * 2

    updated_concentration = cell_concentration + in_diffusion - out_diffusion

    return updated_concentration


@jit(nopython=True)
def update_upper_right_corner_concentration(
    cell_concentration,
    lower_cell_concentration,
    left_cell_concentration,
    diffusion_rate,
    time_step):
    """
        Update the concentration of a species in a cell located at the upper-right corner of a 2D compartment
        based on diffusion from neighboring cells.

        Parameters:
        - cell_concentration (float): Concentration of the species in the upper-right corner cell.
        - lower_cell_concentration (float): Concentration of the species in the cell directly below the upper-left cell.
        - left_cell_concentration (float): Concentration of the species in the cell directly to the left of the upper-right cell.
        - diffusion_rate (float): Rate at which the species diffuses between cells.
        - time_step (float): Discrete time step for the calculation.

        Returns:
        - float: Updated concentration of the species in the upper-left corner cell.
        """
    in_diffusion = (time_step * lower_cell_concentration * diffusion_rate) + \
                   (time_step * left_cell_concentration * diffusion_rate)
    out_diffusion = time_step * cell_concentration * diffusion_rate * 2

    updated_concentration = cell_concentration + in_diffusion - out_diffusion

    return updated_concentration


@jit(nopython=True)
def update_left_side_concentration(
    cell_concentration,
    upper_cell_concentration,
    lower_cell_concentration,
    right_cell_concentration,
    diffusion_rate,
    time_step):
    """
    Update the concentration of a species in cells located along the left side of a 2D compartment
    (excluding the corners) based on diffusion from neighboring cells.

    Parameters:
    - cell_concentration (numpy.ndarray): Array of concentrations for cells in the leftmost column (excluding corners).
    - upper_cell_concentration (numpy.ndarray): Array of concentrations for cells directly above the current cells.
    - lower_cell_concentration (numpy.ndarray): Array of concentrations for cells directly below the current cells.
    - right_cell_concentration (numpy.ndarray): Array of concentrations for cells directly to the right of the current cells.
    - diffusion_rate (float): Rate at which the species diffuses between cells.
    - time_step (float): Discrete time step for the calculation.

    Returns:
    - numpy.ndarray: Updated concentrations of the species in the current cells.
    """
    upper_cell_in = time_step * upper_cell_concentration * diffusion_rate
    lower_cell_in = time_step * lower_cell_concentration * diffusion_rate
    right_cell_in = time_step * right_cell_concentration * diffusion_rate
    in_diffusion = upper_cell_in + lower_cell_in + right_cell_in

    out_diffusion = time_step * cell_concentration * diffusion_rate * 3

    updated_concentration = cell_concentration + in_diffusion - out_diffusion

    return updated_concentration


@jit(nopython=True)
def update_right_side_concentration(
        cell_concentration,
        upper_cell_concentration,
        lower_cell_concentration,
        left_cell_concentration,
        diffusion_rate,
        time_step):
    """
        Update the concentration of a species in cells located along the right side of a 2D compartment
        (excluding the corners) based on diffusion from neighboring cells.

        Parameters:
        - cell_concentration (numpy.ndarray): Array of concentrations for cells in the leftmost column (excluding corners).
        - upper_cell_concentration (numpy.ndarray): Array of concentrations for cells directly above the current cells.
        - lower_cell_concentration (numpy.ndarray): Array of concentrations for cells directly below the current cells.
        - left_cell_concentration (numpy.ndarray): Array of concentrations for cells directly to the left of the current cells.
        - diffusion_rate (float): Rate at which the species diffuses between cells.
        - time_step (float): Discrete time step for the calculation.

        Returns:
        - numpy.ndarray: Updated concentrations of the species in the current cells.
        """
    upper_cell_in = time_step * upper_cell_concentration * diffusion_rate
    lower_cell_in = time_step * lower_cell_concentration * diffusion_rate
    right_cell_in = time_step * left_cell_concentration * diffusion_rate
    in_diffusion = upper_cell_in + lower_cell_in + right_cell_in

    out_diffusion = time_step * cell_concentration * diffusion_rate * 3

    updated_concentration = cell_concentration + in_diffusion - out_diffusion

    return updated_concentration

@jit(nopython=True)
def update_central_concentration_middle(
        cell_concentration,
        upper_cell_concentration,
        lower_cell_concentration,
        right_cell_concentration,
        left_cell_concentration,
        diffusion_rate,
        time_step):
    """
    Update the concentration of species in multiple interior cells of a 2D compartment based on diffusion
    from neighboring cells.

    Parameters:
    - cell_concentration (ndarray): Array of concentrations for multiple interior cells.
    - upper_cell_concentration (ndarray): Array of concentrations for the cells directly above the current cells.
    - lower_cell_concentration (ndarray): Array of concentrations for the cells directly below the current cells.
    - right_cell_concentration (ndarray): Array of concentrations for the cells directly to the right of the current cells.
    - left_cell_concentration (ndarray): Array of concentrations for the cells directly to the left of the current cells.
    - diffusion_rate (float): Rate at which the species diffuses between cells.
    - time_step (float): Discrete time step for the calculation.

    Returns:
    - ndarray: Updated concentrations of the species in the current interior cells.
    """
    upper_cell_in = time_step * upper_cell_concentration * diffusion_rate
    lower_cell_in = time_step * lower_cell_concentration * diffusion_rate
    right_cell_in = time_step * right_cell_concentration * diffusion_rate
    left_cell_in = time_step * left_cell_concentration * diffusion_rate

    in_diffusion = upper_cell_in + lower_cell_in + right_cell_in + left_cell_in
    out_diffusion = time_step * cell_concentration * diffusion_rate * 4

    updated_concentration = cell_concentration + in_diffusion - out_diffusion

    return updated_concentration


@jit(nopython=True)
def update_central_concentration_upper(
        cell_concentration,
        lower_cell_concentration,
        right_cell_concentration,
        left_cell_concentration,
        diffusion_rate,
        time_step):
    """
    Update the concentration of a species in a cell located in the upper interior of a 2D compartment
    based on diffusion from neighboring cells.

    Parameters:
    - cell_concentration (float): Concentration of the species in the current upper interior cell.
    - lower_cell_concentration (float): Concentration of the species in the cell directly below the current cell.
    - right_cell_concentration (float): Concentration of the species in the cell directly to the right of the current cell.
    - left_cell_concentration (float): Concentration of the species in the cell directly to the left of the current cell.
    - diffusion_rate (float): Rate at which the species diffuses between cells.
    - time_step (float): Discrete time step for the calculation.

    Returns:
    - float: Updated concentration of the species in the current upper interior cell.
    """
    lower_cell_in = time_step * lower_cell_concentration * diffusion_rate
    right_cell_in = time_step * right_cell_concentration * diffusion_rate
    left_cell_in = time_step * left_cell_concentration * diffusion_rate

    in_diffusion = lower_cell_in + right_cell_in + left_cell_in
    out_diffusion = time_step * cell_concentration * diffusion_rate * 3

    updated_concentration = cell_concentration + in_diffusion - out_diffusion

    return updated_concentration


@jit(nopython=True)
def update_central_concentration_lower(
        cell_concentration,
        upper_cell_concentration,
        right_cell_concentration,
        left_cell_concentration,
        diffusion_rate,
        time_step):
    """
    Update the concentration of a species in a cell located in the lower interior of a 2D compartment
    based on diffusion from neighboring cells.

    Parameters:
    - cell_concentration (float): Concentration of the species in the current lower interior cell.
    - upper_cell_concentration (float): Concentration of the species in the cell directly above the current cell.
    - right_cell_concentration (float): Concentration of the species in the cell directly to the right of the current cell.
    - left_cell_concentration (float): Concentration of the species in the cell directly to the left of the current cell.
    - diffusion_rate (float): Rate at which the species diffuses between cells.
    - time_step (float): Discrete time step for the calculation.

    Returns:
    - float: Updated concentration of the species in the current lower interior cell.
    """
    upper_cell_in = time_step * upper_cell_concentration * diffusion_rate
    right_cell_in = time_step * right_cell_concentration * diffusion_rate
    left_cell_in = time_step * left_cell_concentration * diffusion_rate

    in_diffusion = upper_cell_in + right_cell_in + left_cell_in
    out_diffusion = time_step * cell_concentration * diffusion_rate * 3

    updated_concentration = cell_concentration + in_diffusion - out_diffusion

    return updated_concentration








