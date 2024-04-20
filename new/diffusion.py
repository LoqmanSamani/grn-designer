def diffusion2d(M, length, depth, diffusionparam, dt, compartment_length, compartment_depth):

    if length == 1 and depth == 1:
        M[length, depth] = lower_left_corner_diff(M, length, depth, diffusionparam, dt)

    elif length == compartment_length and depth == 1:
        M[length, depth] = lower_right_corner_diff(M, length, depth, diffusionparam, dt)

    elif length == 1 and depth == compartment_depth:
        M[length, depth] = upper_left_corner_diff(M, length, depth, diffusionparam, dt)

    elif length == compartment_length and depth == compartment_depth:
        M[length, depth] = upper_right_corner_diff(M, length, depth, diffusionparam, dt)

    elif depth == 1 and length != 1 and length != compartment_length:
        M[length, depth] = lower_side_diff(M, length, depth, diffusionparam, dt)

    elif length == 1 and depth != 1 and depth != compartment_depth:
        M[length, depth] = left_side_diff(M, length, depth, diffusionparam, dt)

    elif length == compartment_length and depth != 1 and depth != compartment_depth:
        M[length, depth] = right_side_diff(M, length, depth, diffusionparam, dt)

    elif depth == compartment_depth and length != 1 and length != compartment_length:
        M[length, depth] = upper_side_diff(M, length, depth, diffusionparam, dt)

    else:
        M[length, depth] = central_diffusion(M, length, depth, diffusionparam, dt)

    return M[length, depth]


def lower_side_diff(M, length, depth, diffusionparam, dt):
    M[length, depth] = (M[length, depth] +
                        dt * diffusionparam *
                        (M[length + 1, depth] +
                         M[length - 1, depth] +
                         M[length, depth + 1] -
                         M[length, depth] * 3))
    return M[length, depth]


def upper_side_diff(M, length, depth, diffusionparam, dt):
    M[length, depth] = (M[length, depth] +
                        dt * diffusionparam *
                        (M[length + 1, depth] +
                         M[length - 1, depth] +
                         M[length, depth - 1] -
                         M[length, depth] * 3))
    return M[length, depth]


def left_side_diff(M, length, depth, diffusionparam, dt):
    M[length, depth] = (M[length, depth] +
                        dt * diffusionparam *
                        (M[length + 1, depth] +
                         M[length, depth + 1] +
                         M[length, depth - 1] -
                         M[length, depth] * 3))
    return M[length, depth]


def right_side_diff(M, length, depth, diffusionparam, dt):
    M[length, depth] = (M[length, depth] +
                        dt * diffusionparam *
                        (M[length - 1, depth] +
                         M[length, depth + 1] +
                         M[length, depth - 1] -
                         M[length, depth] * 3))
    return M[length, depth]


def central_diffusion(M, length, depth, diffusionparam, dt):
    M[length, depth] = (M[length, depth] +
                        dt * diffusionparam *
                        (M[length + 1, depth] +
                         M[length - 1, depth] +
                         M[length, depth + 1] +
                         M[length, depth - 1] -
                         M[length, depth] * 4))
    return M[length, depth]


def lower_left_corner_diff(M, length, depth, diffusionparam, dt):
    M[length, depth] = (M[length, depth] +
                        dt * diffusionparam *
                        (M[length + 1, depth] +
                         M[length, depth + 1] -
                         M[length, depth] * 2))
    return M[length, depth]


def lower_right_corner_diff(M, length, depth, diffusionparam, dt):
    M[length, depth] = (M[length, depth] +
                        dt * diffusionparam *
                        (M[length - 1, depth] +
                         M[length, depth + 1] -
                         M[length, depth] * 2))
    return M[length, depth]


def upper_left_corner_diff(M, length, depth, diffusionparam, dt):
    M[length, depth] = (M[length, depth] +
                        dt * diffusionparam *
                        (M[length + 1, depth] +
                         M[length, depth - 1] -
                         M[length, depth] * 2))
    return M[length, depth]


def upper_right_corner_diff(M, length, depth, diffusionparam, dt):
    M[length, depth] = (M[length, depth] +
                        dt * diffusionparam *
                        (M[length - 1, depth] +
                         M[length, depth - 1] -
                         M[length, depth] * 2))
    return M[length, depth]

