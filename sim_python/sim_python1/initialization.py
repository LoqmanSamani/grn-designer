import numpy as np


class Initialization:

    def set_initial_condition(self, theta):

        cl = theta["compartment_length"]
        cd = theta["compartment_depth"]
        t = 1.0
        t_max = theta["t_max"]
        dt = theta["dt"]
        num_time_steps = int(np.ceil(t_max / dt)) + 1
        timestep = 1

        # cellSeed might be some sort of seed value or initial condition for cell growth
        cellNumberInitial = theta["cellSeed"] / (theta["compartment_length"] * theta["compartment_depth"])  # the initial number of cells
        saveStepInterval = theta["saveStepInterval"]
        fM = np.zeros((cl, cd), dtype=np.float32)
        bM = np.zeros((cl, cd), dtype=np.float32)
        iM = np.zeros((cl, cd), dtype=np.float32)
        ibM = np.zeros((cl, cd), dtype=np.float32)
        cells_anchor = np.zeros((cl, cd), dtype=np.float32)
        cells_GFP = np.zeros((cl, cd), dtype=np.float32)
        cells_mCherry = np.zeros((cl, cd), dtype=np.float32)
        cells_iM = np.zeros((cl, cd), dtype=np.float32)
        bM_all = np.zeros((cl, cd, int(np.ceil(num_time_steps / saveStepInterval))), dtype=np.float32)
        fM_all = np.zeros((cl, cd, int(np.ceil(num_time_steps / saveStepInterval))), dtype=np.float32)
        iM_all = np.zeros((cl, cd, int(np.ceil(num_time_steps / saveStepInterval))), dtype=np.float32)
        ibM_all = np.zeros((cl, cd, int(np.ceil(num_time_steps / saveStepInterval))), dtype=np.float32)
        cells_anchor_all = np.zeros((cl, cd, int(np.ceil(num_time_steps / saveStepInterval))), dtype=np.float32)
        cells_GFP_all = np.zeros((cl, cd, int(np.ceil(num_time_steps / saveStepInterval))), dtype=np.float32)
        cells_mCherry_all = np.zeros((cl, cd, int(np.ceil(num_time_steps / saveStepInterval))), dtype=np.float32)
        cells_iM_all = np.zeros((cl, cd, int(np.ceil(num_time_steps / saveStepInterval))), dtype=np.float32)

        # Change cell numbers with initial seed
        # cells_anker, cells_GFP, cells_mCherry, cells_iM = initialcellseed(cl, cd, cellNumberInitial, cells_anker,cells_GFP, cells_mCherry, cells_iM)

        # Save in dictionary
        simulation_input = {
            "cl": cl,
            "cd": cd,
            "t": t,
            "t_max": t_max,
            "dt": dt,
            "num_time_steps": num_time_steps,
            "timestep": timestep,
            "cellNumberInitial": cellNumberInitial,
            "fM": fM,
            "bM": bM,
            "iM": iM,
            "ibM": ibM,
            "bM_all": bM_all,
            "fM_all": fM_all,
            "iM_all": iM_all,
            "ibM_all": ibM_all,
            "cells_anchor_all": cells_anchor_all,
            "cells_GFP_all": cells_GFP_all,
            "cells_mCherry_all": cells_mCherry_all,
            "cells_iM_all": cells_iM_all,
            "cells_anchor": cells_anchor,
            "cells_GFP": cells_GFP,
            "cells_mCherry": cells_mCherry,
            "cells_iM": cells_iM
        }

        return simulation_input

