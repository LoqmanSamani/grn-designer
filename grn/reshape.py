import numpy as np
from scipy.ndimage import zoom



class Resize:

    def __init__(self, order, mode, cval, grid_mode):
        self.order = order
        self.mode = mode
        self.cval = cval
        self.grid_mode = grid_mode


    def zoom_in(self, target, zoom_):

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

