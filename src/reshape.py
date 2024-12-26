import numpy as np
from scipy.ndimage import zoom



class Resize:

    def __init__(self, order, mode, cval, grid_mode):
        """
        A utility class for resizing spatial patterns and agents.

        This class provides methods to apply zoom-in and zoom-out transformations
        to spatial patterns or populations of agents. The transformations are
        performed using specified interpolation and mode settings.

        Args:
            order (int): The interpolation order for resizing. Higher values result
                         in smoother interpolations. See `scipy.ndimage.zoom` for details.
            mode (str): The mode parameter for handling boundaries during resizing
                        (e.g., 'constant', 'nearest'). See `scipy.ndimage.zoom` for details.
            cval (float): Value to fill past edges when mode is 'constant'.
            grid_mode (bool): If True, the transformation uses grid mode for resizing.
                              See `scipy.ndimage.zoom` for details.
        """

        self.order = order
        self.mode = mode
        self.cval = cval
        self.grid_mode = grid_mode


    def zoom_in(self, target, zoom_):
        """
        Applies a zoom-in transformation to a target spatial pattern.

        This method increases the resolution of the input target pattern by applying
        a specified zoom factor.

        Args:
            target (np.ndarray): The 2D array representing the target spatial pattern.
            zoom_ (float): The zoom factor for increasing resolution. Values >1 zoom in.

        Returns:
            np.ndarray: The zoomed-in target pattern.
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
        Applies a zoom-out transformation to a population of agents.

        This method reduces the resolution of the agents' spatial components while
        preserving other properties, such as parameters and metadata.

        Args:
            population (list): A list of agents, where each agent is a 3D array.
            zoom_ (float): The zoom factor for reducing resolution. Values <1 zoom out.
            x_ (int): The target width for the resized agents.
            y_ (int): The target height for the resized agents.

        Returns:
            list: A list of agents with spatial components resized to the target resolution.
        """

        up_population = []

        for agent in population:
            z, y, x = agent.shape
            num_species = int(agent[-1, -1, 0])

            up_agent = np.zeros(
                shape=(z, x_, y_),
                dtype=np.float32
            )

            for i in range(1, num_species * 2, 2):
                up_agent[i, :, :] = zoom(
                    input=agent[i, :, :],
                    zoom=zoom_,
                    order=self.order,
                    mode=self.mode,
                    cval=self.cval,
                    grid_mode=self.grid_mode
                )
                num_params = int(agent[-1, i-1, -1] + 3)
                if num_params > 3:
                    up_agent[-1, i-1, :num_params] = agent[-1, i-1, :num_params]
                    up_agent[-1, i, :int(num_params-3)] = agent[-1, i, :int(num_params-3)]
                    up_agent[-1, i, -int(num_params-3):] = agent[-1, i, -int(num_params - 3):]
                else:
                    up_agent[-1, i - 1, :3] = agent[-1, i - 1, :3]
                    up_agent[-1, i - 1, -1] = 0
                    

            up_agent[-1, -1, :4] = agent[-1, -1, :4]
            up_population.append(up_agent)

        return up_population

