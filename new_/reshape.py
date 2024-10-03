import numpy as np
from scipy.ndimage import zoom



class Resize:
    """
    A class to perform zoom-in and zoom-out operations on multi-dimensional arrays,
    such as images or population datasets, using spline interpolation.

    Attributes
    ----------
        order : int
            The order of spline interpolation. The value must be between 0 and 5.
        mode : str
            The mode parameter determines how the input array's edges are handled.
            Modes can be 'constant', 'nearest', 'reflect', 'mirror', or 'wrap'.
        cval : float
            The value used for padding when mode is 'constant'. Default is 0.0.
        grid_mode : bool
            If False, pixel centers are zoomed. If True, the full pixel extent is used.

    Methods
    -------
        zoom_in(target, zoom_):
            Zooms in on the target array using the specified zoom factor.

        zoom_out(population, zoom_, x_, y_):
            Zooms out a population of individuals and adjusts the dimensions of the array
            based on the zoom factor and target dimensions (x_, y_).
    """
    def __init__(self, order, mode, cval, grid_mode):
        self.order = order
        self.mode = mode
        self.cval = cval
        self.grid_mode = grid_mode


    def zoom_in(self, target, zoom_):
        """
        Zooms in on the target array by the specified zoom factor.

        Parameters
        ----------
            target : ndarray
                The input array to be zoomed in on. This can be a 2D or 3D array representing
                an image or population dataset.
            zoom_ : float or sequence
                The zoom factor. A float applies the same zoom across all axes. A sequence
                allows different zoom factors for each axis.

        Returns
        -------
            zoomed : ndarray
                The zoomed-in version of the target array, with new dimensions depending
                on the zoom factor.
        """

        zoomed = zoom(
            input=target,
            zoom=zoom_,
            order=self.order,
            mode=self.mode,
            cval=self.cval,
            grid_mode=self.grid_mode
        )

        return zoomed

    def zoom_out(self, population, zoom_, x_, y_):
        """
        Zooms out on a population of individuals by resizing their arrays using the specified
        zoom factor and adjusts their shape to match the target dimensions.

        Parameters
        ----------
            population : list of ndarrays
                A list of individuals, each represented as a 3D array (z, y, x) containing
                features and metadata for each individual.
            zoom_ : float or sequence
                The zoom factor to apply to each individual in the population. A float applies
                the same zoom across all axes. A sequence allows different zoom factors for
                each axis.
            x_ : int
                The target number of columns (width) for the resized individuals.
            y_ : int
                The target number of rows (height) for the resized individuals.

        Returns
        -------
            up_population : list of ndarrays
                A list of zoomed-out individuals with new dimensions (z, x_, y_) after applying
                the zoom factor and adjustments.
        """

        up_population = []

        for individual in population:
            z, y, x = individual.shape
            num_species = int(individual[-1, -1, 0])
            num_pairs = int(individual[-1, -1, 1])
            pair_start = int(num_species * 2)
            pair_stop = int(pair_start + (num_pairs * 2))
            up_individual = np.zeros(shape=(z, x_, y_))

            for i in range(1, num_species * 2, 2):
                up_individual[i, :, :] = zoom(
                    input=individual[i, :, :],
                    zoom=zoom_,
                    order=self.order,
                    mode=self.mode,
                    cval=self.cval,
                    grid_mode=self.grid_mode
                )
                up_individual[-1, i-1, 0:3] = individual[-1, i-1, 0:3]

            for i in range(pair_start+1, pair_stop+1, 2):

                up_individual[i, 0, :2] = individual[i, 0, :2]
                up_individual[i, 1, :4] = individual[i, 1, :4]

            up_individual[-1, -1, :5] = individual[-1, -1, :5]
            up_population.append(up_individual)

        return up_population

