
class Diffusion:

    def diffusion2d(self, specie, length, depth, k_diff, dt, compartment_length, compartment_depth):
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

        if length == 1 and depth == 1:
            specie[length, depth] = self.lower_left_corner_diff(
                specie=specie,
                length=length,
                depth=depth,
                k_diff=k_diff,
                dt=dt
            )

        elif length == compartment_length and depth == 1:
            specie[length, depth] = self.lower_right_corner_diff(
                specie=specie,
                length=length,
                depth=depth,
                k_diff=k_diff,
                dt=dt
            )

        elif length == 1 and depth == compartment_depth:
            specie[length, depth] = self.upper_left_corner_diff(
                specie=specie,
                length=length,
                depth=depth,
                k_diff=k_diff,
                dt=dt
            )

        elif length == compartment_length and depth == compartment_depth:
            specie[length, depth] = self.upper_right_corner_diff(
                specie=specie,
                length=length,
                depth=depth,
                k_diff=k_diff,
                dt=dt
            )

        elif depth == 1 and length != 1 and length != compartment_length:
            specie[length, depth] = self.lower_side_diff(
                specie=specie,
                length=length,
                depth=depth,
                k_diff=k_diff,
                dt=dt
            )

        elif length == 1 and depth != 1 and depth != compartment_depth:
            specie[length, depth] = self.left_side_diff(
                specie=specie,
                length=length,
                depth=depth,
                k_diff=k_diff,
                dt=dt
            )

        elif length == compartment_length and depth != 1 and depth != compartment_depth:
            specie[length, depth] = self.right_side_diff(
                specie=specie,
                length=length,
                depth=depth,
                k_diff=k_diff,
                dt=dt
            )

        elif depth == compartment_depth and length != 1 and length != compartment_length:
            specie[length, depth] = self.upper_side_diff(
                specie=specie,
                length=length,
                depth=depth,
                k_diff=k_diff,
                dt=dt
            )

        else:
            specie[length, depth] = self.central_diffusion(
                specie=specie,
                length=length,
                depth=depth,
                k_diff=k_diff,
                dt=dt
            )

        return specie[length, depth]

    def lower_side_diff(self, specie, length, depth, k_diff, dt):

        specie[length, depth] = (
                specie[length, depth] +
                dt * k_diff *
                (specie[length + 1, depth] +
                 specie[length - 1, depth] +
                 specie[length, depth + 1] -
                 specie[length, depth] * 3)
        )

        return specie[length, depth]

    def upper_side_diff(self, specie, length, depth, k_diff, dt):

        specie[length, depth] = (
                specie[length, depth] +
                dt * k_diff *
                (specie[length + 1, depth] +
                specie[length - 1, depth] +
                specie[length, depth - 1] -
                specie[length, depth] * 3)
        )

        return specie[length, depth]

    def left_side_diff(self, specie, length, depth, k_diff, dt):

        specie[length, depth] = (
                specie[length, depth] +
                dt * k_diff *
                (specie[length + 1, depth] +
                specie[length, depth + 1] +
                specie[length, depth - 1] -
                specie[length, depth] * 3)
        )
        return specie[length, depth]

    def right_side_diff(self, specie, length, depth, k_diff, dt):

        specie[length, depth] = (
                specie[length, depth] +
                dt * k_diff *
                (specie[length - 1, depth] +
                specie[length, depth + 1] +
                specie[length, depth - 1] -
                specie[length, depth] * 3)
        )

        return specie[length, depth]

    def central_diffusion(self, specie, length, depth, k_diff, dt):

        specie[length, depth] = (
                specie[length, depth] +
                dt * k_diff *
                (specie[length + 1, depth] +
                specie[length - 1, depth] +
                specie[length, depth + 1] +
                specie[length, depth - 1] -
                specie[length, depth] * 4)
        )

        return specie[length, depth]

    def lower_left_corner_diff(self, specie, length, depth, k_diff, dt):

        specie[length, depth] = (
                specie[length, depth] +
                dt * k_diff *
                (specie[length + 1, depth] +
                specie[length, depth + 1] -
                specie[length, depth] * 2)
        )

        return specie[length, depth]

    def lower_right_corner_diff(self, specie, length, depth, k_diff, dt):

        specie[length, depth] = (
                specie[length, depth] +
                dt * k_diff *
                (specie[length - 1, depth] +
                specie[length, depth + 1] -
                specie[length, depth] * 2)
        )

        return specie[length, depth]

    def upper_left_corner_diff(self, specie, length, depth, k_diff, dt):

        specie[length, depth] = (
                specie[length, depth] +
                dt * k_diff *
                (specie[length + 1, depth] +
                specie[length, depth - 1] -
                specie[length, depth] * 2)
        )

        return specie[length, depth]

    def upper_right_corner_diff(self, specie, length, depth, k_diff, dt):

        specie[length, depth] = (
                specie[length, depth] +
                dt * k_diff *
                (specie[length - 1, depth] +
                specie[length, depth - 1] -
                specie[length, depth] * 2)
        )

        return specie[length, depth]
