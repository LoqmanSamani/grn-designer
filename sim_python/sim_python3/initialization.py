import numpy as np


def initialization(params):

    growth_rate = params["growth rate"]
    num_max_cell = params["max cell number"]
    cell_seed = params["cell seed"]
    k_fm_sec = params["k_fm_sec"]
    k_im_sec = params["k_im_sec"]
    k_fm_bind = params["k_fm_bind"]
    k_fm_off = params["k_fm_off"]
    k_im_bind = params["k_im_bind"]
    k_im_off = params["k_im_off"]
    k_fm_deg = params["k_fm_deg"]
    k_im_deg = params["k_im_deg"]
    k_bm_deg = params["k_bm_deg"]
    k_ibm_deg = params["k_ibm_deg"]
    d_free = params["d_free"]
    d_i = params["d_i"]
    com_len = params["compartment length"]
    com_wid = params["compartment width"]
    start = params["start"]
    stop = params["stop"]
    dt = params["dt"]
    num_time_steps = int(np.ceil(stop / dt)) + 1
    epoch = 1
    num_cell_init = params["cell seed"] / (com_len * com_wid)
    interval_save = params["save step interval"]

    fM = np.zeros((com_len, com_wid), dtype=np.float32)  # free morphogen
    bM = np.zeros((com_len, com_wid), dtype=np.float32)  # bound morphogen
    iM = np.zeros((com_len, com_wid), dtype=np.float32)  # inhibitor & morpgogen
    ibM = np.zeros((com_len, com_wid), dtype=np.float32)  # inhibitor & bound morphogen
    cells_anchor = np.zeros((com_len, com_wid), dtype=np.float32)
    cells_GFP = np.zeros((com_len, com_wid), dtype=np.float32)
    cells_mCherry = np.zeros((com_len, com_wid), dtype=np.float32)
    cells_iM = np.zeros((com_len, com_wid), dtype=np.float32)
    bM_all = np.zeros((com_len, com_wid, int(np.ceil(num_time_steps / interval_save))), dtype=np.float32)
    fM_all = np.zeros((com_len, com_wid, int(np.ceil(num_time_steps / interval_save))), dtype=np.float32)
    iM_all = np.zeros((com_len, com_wid, int(np.ceil(num_time_steps / interval_save))), dtype=np.float32)
    ibM_all = np.zeros((com_len, com_wid, int(np.ceil(num_time_steps / interval_save))), dtype=np.float32)
    cells_anchor_all = np.zeros((com_len, com_wid, int(np.ceil(num_time_steps / interval_save))), dtype=np.float32)
    cells_GFP_all = np.zeros((com_len, com_wid, int(np.ceil(num_time_steps / interval_save))), dtype=np.float32)
    cells_mCherry_all = np.zeros((com_len, com_wid, int(np.ceil(num_time_steps / interval_save))), dtype=np.float32)
    cells_iM_all = np.zeros((com_len, com_wid, int(np.ceil(num_time_steps / interval_save))), dtype=np.float32)

    init_params = (
        growth_rate,
        num_max_cell,
        cell_seed,
        k_fm_sec,
        k_im_sec,
        k_fm_bind,
        k_fm_off,
        k_im_bind,
        k_im_off,
        k_fm_deg,
        k_im_deg,
        k_bm_deg,
        k_ibm_deg,
        d_free,
        d_i,
        com_len,
        com_wid,
        start,
        stop,
        dt,
        num_time_steps,
        epoch,
        num_cell_init,
        interval_save,
        fM,
        bM,
        iM,
        ibM,
        cells_anchor,
        cells_GFP,
        cells_mCherry,
        cells_iM,
        bM_all,
        fM_all,
        iM_all,
        ibM_all,
        cells_anchor_all,
        cells_GFP_all,
        cells_mCherry_all,
        cells_iM_all
    )

    return init_params

