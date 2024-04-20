from reactions import *
from diffusion import *
import h5py
import os


def simulate2DMS(theta, saveStepInterval, directory_path, file_name):

    cellNumberInitial = theta["cellSeed"] / (theta["compartment_length"] * theta["compartment_depth"])
    growthRate = theta["growthRate"]
    maxCellNumber = theta["maxCellNumber"]
    cl = theta["compartment_length"]
    cd = theta["compartment_depth"]
    t = 0.0 + 1
    tmax = theta["tmax"]
    dt = theta["dt"]
    num_timesteps = int(np.ceil(tmax / dt)) + 1
    timestep = 1

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

    # load initial cell numbers from the ic-.. file defined in myoptions
    #cells_anker, cells_GFP, cells_mCherry, cells_iM = initialcellseed(cl, cd, cellNumberInitial, cells_anker, cells_GFP, cells_mCherry, cells_iM)

    fM_all[:, :, 0] = fM
    bM_all[:, :, 0] = bM
    iM_all[:, :, 0] = iM
    ibM_all[:, :, 0] = ibM
    cells_anker_all[:, :, 0] = cells_anker
    cells_GFP_all[:, :, 0] = cells_GFP
    cells_mCherry_all[:, :, 0] = cells_mCherry
    cells_iM_all[:, :, 0] = cells_iM

    while t <= tmax:
        tmp_fM = fM.copy()
        tmp_bM = bM.copy()
        tmp_iM = iM.copy()
        tmp_ibM = ibM.copy()
        tmp_cells_anker = cells_anker.copy()
        tmp_cells_GFP = cells_GFP.copy()
        tmp_cells_mCherry = cells_mCherry.copy()
        tmp_cells_iM = cells_iM.copy()

        for length in range(cl):
            for depth in range(cd):
                # production
                fM[length, depth] = production(tmp_fM[length, depth], theta["k_fM_src"], tmp_cells_GFP[length, depth], cellNumberInitial, dt)
                iM[length, depth] = production(tmp_iM[length, depth], theta["k_iM_src"], tmp_cells_iM[length, depth], cellNumberInitial, dt)
                # anchor_binding
                fM[length, depth], bM[length, depth] = anchor_binding(tmp_fM[length, depth], theta["k_fM_bind"], theta["k_fM_off"], dt, tmp_cells_anker[length, depth], tmp_bM[length, depth])
                # inhibitor_binding
                iM[length, depth], fM[length, depth], ibM[length, depth] = inhibitor_binding(tmp_fM[length, depth], theta["k_iM_bind"], theta["k_iM_off"], dt, tmp_iM[length, depth], tmp_ibM[length, depth])
                # degradation
                fM[length, depth] = degradation(tmp_fM[length, depth], theta["k_fM_deg"], dt)
                bM[length, depth] = degradation(tmp_bM[length, depth], theta["k_bM_deg"], dt)
                iM[length, depth] = degradation(tmp_iM[length, depth], theta["k_iM_deg"], dt)
                ibM[length, depth] = degradation(tmp_ibM[length, depth], theta["k_ibM_deg"], dt)
                # diffusion
                fM[length, depth] = diffusion2d(tmp_fM, length, depth, theta["d_free"], dt, cl, cd)
                iM[length, depth] = diffusion2d(tmp_iM, length, depth, theta["d_i"], dt, cl, cd)
                ibM[length, depth] = diffusion2d(tmp_ibM, length, depth, theta["d_i"], dt, cl, cd)
                # Cell growth Verhulst
                cells_anker[length, depth] = logistic_growth(tmp_cells_anker[length, depth], tmp_bM[length, depth], growthRate, maxCellNumber, dt)

        # save every saveStepInterval
        if timestep % saveStepInterval == 0:
            idx = int(timestep / saveStepInterval)
            fM_all[:, :, idx] = fM
            bM_all[:, :, idx] = bM
            iM_all[:, :, idx] = iM
            ibM_all[:, :, idx] = ibM
            cells_anker_all[:, :, idx] = cells_anker
            cells_GFP_all[:, :, idx] = cells_GFP
            cells_mCherry_all[:, :, idx] = cells_mCherry
            cells_iM_all[:, :, idx] = cells_iM

        # update time
        t += dt
        timestep += 1

    full_path = os.path.join(directory_path, file_name)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    with h5py.File(full_path, "w") as file:

        file.create_dataset("expected_fM", data=fM_all[:, :, 1:])
        file.create_dataset("expected_bM", data=bM_all[:, :, 1:])
        file.create_dataset("expected_iM", data=iM_all[:, :, 1:])
        file.create_dataset("expected_ibM", data=ibM_all[:, :, 1:])
        file.create_dataset("expected_cells_anker_all", data=cells_anker_all[:, :, 1:])
        file.create_dataset("expected_cells_GFP_all", data=cells_GFP_all[:, :, 1:])
        file.create_dataset("expected_cells_mCherry_all", data=cells_mCherry_all[:, :, 1:])
        file.create_dataset("expected_cells_iM_all", data=cells_iM_all[:, :, 1:])

    print("Simulation results saved to:", full_path)

    sim_output = {"fM_all": fM_all, "bM_all": bM_all, "iM_all": iM_all, "ibM_all": ibM_all,
                 "cells_anker_all": cells_anker_all, "cells_GFP_all": cells_GFP_all,
                 "cells_mCherry_all": cells_mCherry_all, "cells_iM_all": cells_iM_all}
    return sim_output

