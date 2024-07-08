from numba import jit


@jit
def diffusion2d(specie, length, depth, k_diff, dt, compartment_length, compartment_depth):
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

    if length == 0 and depth == 0:
        specie[length, depth] = lower_left_corner_diff(
            specie=specie,
            length=length,
            depth=depth,
            k_diff=k_diff,
            dt=dt
        )

    elif length == compartment_length-1 and depth == 0:
        specie[length, depth] = lower_right_corner_diff(
            specie=specie,
            length=length,
            depth=depth,
            k_diff=k_diff,
            dt=dt
        )

    elif length == 0 and depth == compartment_depth-1:
        specie[length, depth] = upper_left_corner_diff(
            specie=specie,
            length=length,
            depth=depth,
            k_diff=k_diff,
            dt=dt
        )

    elif length == compartment_length-1 and depth == compartment_depth-1:
        specie[length, depth] = upper_right_corner_diff(
            specie=specie,
            length=length,
            depth=depth,
            k_diff=k_diff,
            dt=dt
        )

    elif depth == 0 and length != 0 and length != 1 and length != compartment_length-1 and length != compartment_length-2:
        specie[length, depth] = lower_side_diff1(
            specie=specie,
            length=length,
            depth=depth,
            k_diff=k_diff,
            dt=dt
        )

    elif depth == 0 and length == 1:
        specie[length, depth] = lower_side_diff2(
            specie=specie,
            length=length,
            depth=depth,
            k_diff=k_diff,
            dt=dt
        )

    elif depth == 0 and length == compartment_length-2:
        specie[length, depth] = lower_side_diff3(
            specie=specie,
            length=length,
            depth=depth,
            k_diff=k_diff,
            dt=dt
        )

    elif length == 0 and depth != 0 and depth != 1 and depth != compartment_depth-1 and depth != compartment_depth-2:
        specie[length, depth] = left_side_diff1(
            specie=specie,
            length=length,
            depth=depth,
            k_diff=k_diff,
            dt=dt
        )
    elif length == 0 and depth == 1:
        specie[length, depth] = left_side_diff2(
            specie=specie,
            length=length,
            depth=depth,
            k_diff=k_diff,
            dt=dt
        )
    elif length == 0 and depth == compartment_depth-2:
        specie[length, depth] = left_side_diff3(
            specie=specie,
            length=length,
            depth=depth,
            k_diff=k_diff,
            dt=dt
        )

    elif length == compartment_length-1 and depth != 0 and depth != 1 and depth != compartment_depth-1 and depth != compartment_depth-2:
        specie[length, depth] = right_side_diff1(
            specie=specie,
            length=length,
            depth=depth,
            k_diff=k_diff,
            dt=dt
        )
    elif length == compartment_length-1 and depth == 1:
        specie[length, depth] = right_side_diff2(
            specie=specie,
            length=length,
            depth=depth,
            k_diff=k_diff,
            dt=dt
        )
    elif length == compartment_length-1 and depth == compartment_depth-2:
        specie[length, depth] = right_side_diff3(
            specie=specie,
            length=length,
            depth=depth,
            k_diff=k_diff,
            dt=dt
        )

    elif depth == compartment_depth-1 and length != 0 and length != 1 and length != compartment_length-1 and length != compartment_length-2:
        specie[length, depth] = upper_side_diff1(
            specie=specie,
            length=length,
            depth=depth,
            k_diff=k_diff,
            dt=dt
        )
    elif depth == compartment_depth-1 and length == 1:
        specie[length, depth] = upper_side_diff2(
            specie=specie,
            length=length,
            depth=depth,
            k_diff=k_diff,
            dt=dt
        )
    elif depth == compartment_depth-1 and length == compartment_length-2:
        specie[length, depth] = upper_side_diff3(
            specie=specie,
            length=length,
            depth=depth,
            k_diff=k_diff,
            dt=dt
        )

    else:
        specie[length, depth] = central_diffusion(
            specie=specie,
            length=length,
            depth=depth,
            k_diff=k_diff,
            dt=dt
        )

    return specie[length, depth]


@jit
def lower_side_diff1(specie, length, depth, k_diff, dt):

    specie[length, depth] = (specie[length, depth]+
                             (dt*k_diff*(specie[length+1, depth]+specie[length+2, depth]+specie[length-1, depth]+specie[length-2, depth]+specie[length, depth+1]+specie[length, depth+2]-(specie[length, depth]*6))))

    return specie[length, depth]

@jit
def lower_side_diff2(specie, length, depth, k_diff, dt):

    specie[length, depth] = (specie[length, depth]+
                             (dt*k_diff*(specie[length+1, depth]+specie[length+2, depth]+specie[length-1, depth]+specie[length, depth+1]+specie[length, depth+2]-(specie[length, depth]*5))))

    return specie[length, depth]

@jit
def lower_side_diff3(specie, length, depth, k_diff, dt):

    specie[length, depth] = (specie[length, depth]+
                             (dt*k_diff*(specie[length+1, depth]+specie[length-1, depth]+specie[length-2, depth]+specie[length, depth+1]+specie[length, depth+2]-(specie[length, depth]*5))))

    return specie[length, depth]


@jit
def left_side_diff1(specie, length, depth, k_diff, dt):

    specie[length, depth] = (specie[length, depth]+
                             (dt*k_diff*(specie[length+1, depth]+specie[length+2, depth]+specie[length, depth+1]+specie[length, depth+2]+specie[length, depth-1]+specie[length, depth-2]-(specie[length, depth]*6))))

    return specie[length, depth]



@jit
def left_side_diff2(specie, length, depth, k_diff, dt):

    specie[length, depth] = (specie[length, depth]+
                             (dt*k_diff*(specie[length+1, depth]+specie[length+2, depth]+specie[length, depth+1]+specie[length, depth+2]+specie[length, depth-1]-(specie[length, depth]*5))))

    return specie[length, depth]


@jit
def left_side_diff3(specie, length, depth, k_diff, dt):

    specie[length, depth] = (specie[length, depth]+
                             (dt*k_diff*(specie[length+1, depth]+specie[length+2, depth]+specie[length, depth+1]+specie[length, depth-1]+specie[length, depth-2]-(specie[length, depth]*5))))

    return specie[length, depth]


@jit
def upper_side_diff1(specie, length, depth, k_diff, dt):

    specie[length, depth] = (specie[length, depth]+
                             (dt*k_diff*(specie[length+1, depth]+specie[length+2, depth]+specie[length-1, depth]+specie[length-2, depth]+specie[length, depth-1]+specie[length, depth-2]-(specie[length, depth]*6))))

    return specie[length, depth]

@jit
def upper_side_diff2(specie, length, depth, k_diff, dt):

    specie[length, depth] = (specie[length, depth]+
                             (dt*k_diff*(specie[length+1, depth]+specie[length+2, depth]+specie[length-1, depth]+specie[length, depth-1]+specie[length, depth-2]-(specie[length, depth]*5))))

    return specie[length, depth]

@jit
def upper_side_diff3(specie, length, depth, k_diff, dt):

    specie[length, depth] = (specie[length, depth]+
                             (dt*k_diff*(specie[length+1, depth]+specie[length-1, depth]+specie[length-2, depth]+specie[length, depth-1]+specie[length, depth-2]-(specie[length, depth]*5))))

    return specie[length, depth]



@jit
def right_side_diff1(specie, length, depth, k_diff, dt):

    specie[length, depth] = (specie[length, depth]+
                             (dt*k_diff*(specie[length-1, depth]+specie[length-2, depth]+specie[length, depth+1]+specie[length, depth+2]+specie[length, depth-1]+specie[length, depth-2]-(specie[length, depth]*6))))

    return specie[length, depth]

@jit
def right_side_diff2(specie, length, depth, k_diff, dt):

    specie[length, depth] = (specie[length, depth]+
                             (dt*k_diff*(specie[length-1, depth]+specie[length-2, depth]+specie[length, depth+1]+specie[length, depth+2]+specie[length, depth-1]-(specie[length, depth]*5))))

    return specie[length, depth]


@jit
def right_side_diff3(specie, length, depth, k_diff, dt):

    specie[length, depth] = (specie[length, depth]+
                             (dt*k_diff*(specie[length-1, depth]+specie[length-2, depth]+specie[length, depth+1]+specie[length, depth-1]+specie[length, depth-2]-(specie[length, depth]*6))))

    return specie[length, depth]



@jit
def central_diffusion(specie, length, depth, k_diff, dt):

    specie[length, depth] = (specie[length, depth]+
                             (dt*k_diff*(specie[length+1, depth]+specie[length+2, depth]+specie[length-1, depth]+specie[length-2, depth]+specie[length, depth+1]+specie[length, depth+2]+specie[length, depth-1]+specie[length, depth-2]-(specie[length, depth]*8))))

    return specie[length, depth]


@jit
def lower_left_corner_diff(specie, length, depth, k_diff, dt):

    specie[length, depth] = (specie[length, depth] +
                             (dt*k_diff*(specie[length+1, depth]+specie[length+2, depth]+specie[length, depth+1]+specie[length, depth+2]-(specie[length, depth]*4))))

    return specie[length, depth]


@jit
def lower_right_corner_diff(specie, length, depth, k_diff, dt):

    specie[length, depth] = (specie[length, depth] +
                             (dt*k_diff*(specie[length-1, depth]+specie[length-2, depth]+specie[length, depth+1]+specie[length, depth+2]-(specie[length, depth]*4))))

    return specie[length, depth]


@jit
def upper_left_corner_diff(specie, length, depth, k_diff, dt):

    specie[length, depth] = (specie[length, depth] +
                             (dt*k_diff*(specie[length+1, depth]+specie[length+2, depth]+specie[length, depth-2]+specie[length, depth-1]-(specie[length, depth]*4))))

    return specie[length, depth]


@jit
def upper_right_corner_diff(specie, length, depth, k_diff, dt):

    specie[length, depth] = (specie[length, depth] +
                             (dt*k_diff*(specie[length-1, depth]+specie[length-2, depth]+specie[length, depth-1]+specie[length, depth-2]-(specie[length, depth]*4))))

    return specie[length, depth]
