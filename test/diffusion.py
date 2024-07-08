from numba import jit


@jit
def diffusion(specie, length, width, k_diff, dt, compartment_length, compartment_width):
    """
    Args:
    - specie: specie concentration array
    - length: index of the ith cell in compartment(column), integer
    - depth: index of the jth cell in compartment(row), integer
    - k_diff: diffusion rate constant, float
    - dt: delta t, time step size, float
    - compartment_length: number of cells (columns) in compartment
    - compartment_depth: number of cells (rows) in compartment

    Returns:
    - specie[length, depth]: calculated concentration of the cell[length, depth] in the compartment, integer
    """

    if length == 0 and width == 0:
        specie[length, width] = lower_left_corner_diff(
            specie=specie,
            length=length,
            depth=width,
            k_diff=k_diff,
            dt=dt
        )

    elif length == compartment_length-1 and width == 0:
        specie[length, width] = lower_right_corner_diff(
            specie=specie,
            length=length,
            depth=width,
            k_diff=k_diff,
            dt=dt
        )

    elif length == 0 and width == compartment_width-1:
        specie[length, width] = upper_left_corner_diff(
            specie=specie,
            length=length,
            depth=width,
            k_diff=k_diff,
            dt=dt
        )

    elif length == compartment_length-1 and width == compartment_width-1:
        specie[length, width] = upper_right_corner_diff(
            specie=specie,
            length=length,
            depth=width,
            k_diff=k_diff,
            dt=dt
        )

    elif width == 0 and length != 0 and length != compartment_length-1:
        specie[length, width] = lower_side_diff(
            specie=specie,
            length=length,
            depth=width,
            k_diff=k_diff,
            dt=dt
        )

    elif length == 0 and width != 0 and width != compartment_width-1:
        specie[length, width] = left_side_diff(
            specie=specie,
            length=length,
            depth=width,
            k_diff=k_diff,
            dt=dt
        )

    elif length == compartment_length-1 and width != 0 and width != compartment_width-1:
        specie[length, width] = right_side_diff(
            specie=specie,
            length=length,
            depth=width,
            k_diff=k_diff,
            dt=dt
        )

    elif width == compartment_width-1 and length != 0 and length != compartment_length-1:
        specie[length, width] = upper_side_diff(
            specie=specie,
            length=length,
            depth=width,
            k_diff=k_diff,
            dt=dt
        )

    else:
        specie[length, width] = central_diffusion(
            specie=specie,
            length=length,
            depth=width,
            k_diff=k_diff,
            dt=dt
        )

    return specie[length, width]


@jit
def lower_side_diff(specie, length, depth, k_diff, dt):

    specie[length, depth] = (specie[length, depth]+
                             (dt*k_diff*(specie[length+1, depth]+specie[length-1, depth]+specie[length, depth+1]-(specie[length, depth]*3))))

    return specie[length, depth]


@jit
def upper_side_diff(specie, length, depth, k_diff, dt):

    specie[length, depth] = (specie[length, depth]+
                             (dt*k_diff*(specie[length+1, depth]+specie[length-1, depth]+specie[length, depth-1]-(specie[length, depth]*3))))

    return specie[length, depth]


@jit
def left_side_diff(specie, length, depth, k_diff, dt):

    specie[length, depth] = (specie[length, depth]+
                             (dt*k_diff*(specie[length+1, depth]+specie[length, depth+1]+specie[length, depth-1]-(specie[length, depth]*3))))

    return specie[length, depth]


@jit
def right_side_diff(specie, length, depth, k_diff, dt):

    specie[length, depth] = (specie[length, depth]+
                             (dt*k_diff*(specie[length-1, depth]+specie[length, depth+1] +specie[length, depth-1]-(specie[length, depth] * 3))))

    return specie[length, depth]


@jit
def central_diffusion(specie, length, depth, k_diff, dt):

    specie[length, depth] = (specie[length, depth]+
                             (dt*k_diff*(specie[length+1, depth]+specie[length-1, depth]+specie[length, depth+1]+specie[length, depth-1]-(specie[length, depth]*4))))

    return specie[length, depth]


@jit
def lower_left_corner_diff(specie, length, depth, k_diff, dt):

    specie[length, depth] = (specie[length, depth] +
                             (dt*k_diff*(specie[length+1, depth]+specie[length, depth+1]-(specie[length, depth]*2))))

    return specie[length, depth]


@jit
def lower_right_corner_diff(specie, length, depth, k_diff, dt):

    specie[length, depth] = (specie[length, depth] +
                             (dt*k_diff*(specie[length-1, depth]+specie[length, depth+1]-(specie[length, depth]*2))))

    return specie[length, depth]


@jit
def upper_left_corner_diff(specie, length, depth, k_diff, dt):

    specie[length, depth] = (specie[length, depth] +
                             (dt*k_diff*(specie[length+1, depth]+specie[length, depth-1]-(specie[length, depth]*2))))

    return specie[length, depth]


@jit
def upper_right_corner_diff(specie, length, depth, k_diff, dt):

    specie[length, depth] = (specie[length, depth] +
                             (dt*k_diff*(specie[length-1, depth]+specie[length, depth-1]-(specie[length, depth]*2))))

    return specie[length, depth]


