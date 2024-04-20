import numpy as np

def set_initial_condition(theta):
    cl = theta["compartment_length"] + 1  # +1 since Python is zero-indexed
    cd = theta["compartment_depth"] + 1
    t = 0.0 + 1
    tmax = theta["tmax"]
    dt = theta["dt"]
    num_timesteps = int(np.ceil(tmax / dt)) + 1
    timestep = 1
    cellNumberInitial = theta["cellSeed"] / (theta["compartment_length"] * theta["compartment_depth"])

    fM = np.zeros((cl, cd))
    bM = np.zeros((cl, cd))
    iM = np.zeros((cl, cd))
    ibM = np.zeros((cl, cd))
    cells_anker = np.zeros((cl, cd))
    cells_GFP = np.zeros((cl, cd))
    cells_mCherry = np.zeros((cl, cd))
    cells_iM = np.zeros((cl, cd))
    bM_all = np.zeros((cl, cd, int(np.ceil(num_timesteps / saveStepInterval))))
    fM_all = np.zeros((cl, cd, int(np.ceil(num_timesteps / saveStepInterval))))
    iM_all = np.zeros((cl, cd, int(np.ceil(num_timesteps / saveStepInterval))))
    ibM_all = np.zeros((cl, cd, int(np.ceil(num_timesteps / saveStepInterval))))
    cells_anker_all = np.zeros((cl, cd, int(np.ceil(num_timesteps / saveStepInterval))), dtype=np.int32)
    cells_GFP_all = np.zeros((cl, cd, int(np.ceil(num_timesteps / saveStepInterval))), dtype=np.int32)
    cells_mCherry_all = np.zeros((cl, cd, int(np.ceil(num_timesteps / saveStepInterval))), dtype=np.int32)
    cells_iM_all = np.zeros((cl, cd, int(np.ceil(num_timesteps / saveStepInterval))), dtype=np.int32)

    # Change cell numbers with initial seed
    cells_anker, cells_GFP, cells_mCherry, cells_iM = initialcellseed(cl, cd, cellNumberInitial, cells_anker,
                                                                      cells_GFP, cells_mCherry, cells_iM)

    # Save in dictionary
    simulation_input = {
        "cl": cl, "cd": cd, "t": t, "tmax": tmax, "dt": dt, "num_timesteps": num_timesteps, "timestep": timestep,
        "cellNumberInitial": cellNumberInitial, "fM": fM, "bM": bM, "iM": iM, "ibM": ibM, "bM_all": bM_all,
        "fM_all": fM_all, "iM_all": iM_all, "ibM_all": ibM_all, "cells_anker_all": cells_anker_all,
        "cells_GFP_all": cells_GFP_all, "cells_mCherry_all": cells_mCherry_all, "cells_iM_all": cells_iM_all,
        "cells_anker": cells_anker, "cells_GFP": cells_GFP, "cells_mCherry": cells_mCherry, "cells_iM": cells_iM
    }

    return simulation_input
