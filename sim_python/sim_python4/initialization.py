import numpy as np


def initialization(params):
    """
    Initializes the parameters needed to simulate the model.

    Arguments:
        - params (dictionary): Contains all necessary parameters for simulating the model.
            - k_fm_sec (float): Morphogen secretion rate constant.
            - k_fi_sec (float): Inhibitor secretion rate constant.
            - k_am_on (float): Binding anchor-morphogen rate constant.
            - k_am_off (float): Unbinding anchor-morphogen rate constant.
            - k_im_on (float): Binding inhibitor-morphogen rate constant.
            - k_im_off (float): Unbinding inhibitor-morphogen rate constant.
            - k_fm_deg (float): Free morphogen degradation rate constant.
            - k_fi_deg (float): Free inhibitor degradation rate constant.
            - k_im_deg (float): Inhibitor-morphogen degradation rate constant.
            - k_am_deg (float): Anchor-morphogen degradation rate constant.
            - k_m_diff (float): Free morphogen diffusion rate.
            - k_i_diff (float): Free inhibitor diffusion rate.
            - k_im_diff (float): Free inhibitor-morphogen diffusion rate.
            - com_len (integer): Number of cells in the X-axis (length of compartment).
            - com_wid (integer): Number of cells in the Y-axis (width of compartment).
            - start (float or integer): Start time of the simulation.
            - stop (float or integer): Stop time of the simulation.
            - dt (float): Time step.
            - interval_save (integer): Interval used to save concentration matrices.

    Returns:
        - init_params (tuple): Contains necessary parameters and initialized ndarray concentration matrices.
    """
    # Extracting parameters from the dictionary
    k_fm_sec = params["k_fm_sec"]
    k_fi_sec = params["k_fi_sec"]
    k_am_on = params["k_am_on"]
    k_am_off = params["k_am_off"]
    k_im_on = params["k_im_on"]
    k_im_off = params["k_im_off"]
    k_fm_deg = params["k_fm_deg"]
    k_fi_deg = params["k_fi_deg"]
    k_im_deg = params["k_im_deg"]
    k_am_deg = params["k_am_deg"]
    k_m_diff = params["k_m_diff"]
    k_i_diff = params["k_i_diff"]
    k_im_diff = params["k_im_diff"]
    com_len = params["compartment length"]
    com_wid = params["compartment width"]
    start = params["start"]
    stop = params["stop"]
    dt = params["dt"]
    interval_save = params["save step interval"]

    # Calculating number of time steps and intermediate matrices
    num_time_steps = int(np.ceil(stop / dt)) + 1
    idx = int(np.ceil(num_time_steps / interval_save))
    epoch = 1

    fM = np.zeros((com_len, com_wid), dtype=np.float32)  # Free morphogen 2D matrix
    fM[:, 0] = 1
    fI = np.zeros((com_len, com_wid), dtype=np.float32)  # Free inhibitor 2D matrix
    fI[:, -1] = 1
    A = np.ones((com_len, com_wid), dtype=np.float32)  # Anchor 2D matrix (cells which have anchor)
    IM = np.zeros((com_len, com_wid), dtype=np.float32)  # Inhibitor-morphogen 2D matrix
    AM = np.zeros((com_len, com_wid), dtype=np.float32)  # Anchor-morphogen 2D matrix

    # Initializing concentration matrices
    fM_all = np.zeros((com_len, com_wid, idx), dtype=np.float32)  # Free morphogen 3D matrix
    fM_all[:, :, 0] = fM
    fI_all = np.zeros((com_len, com_wid, idx), dtype=np.float32)  # Free inhibitor 3D matrix
    fI_all[:, :, 0] = fI
    A_all = np.zeros((com_len, com_wid, idx), dtype=np.float32)  # Anchor 3D matrix (cells which have anchor)
    A_all[:, :, 0] = A
    IM_all = np.zeros((com_len, com_wid, idx), dtype=np.float32)  # Inhibitor-morphogen 3D matrix
    IM_all[:, :, 0] = IM
    AM_all = np.zeros((com_len, com_wid, idx), dtype=np.float32)  # Anchor-morphogen 3D matrix
    AM_all[:, :, 0] = AM

    # Packaging initialized parameters into a tuple
    init_params = (
        k_fm_sec,
        k_fi_sec,
        k_am_on,
        k_am_off,
        k_im_on,
        k_im_off,
        k_fm_deg,
        k_fi_deg,
        k_im_deg,
        k_am_deg,
        k_m_diff,
        k_i_diff,
        k_im_diff,
        com_len,
        com_wid,
        start,
        stop,
        dt,
        num_time_steps,
        epoch,
        interval_save,
        fM,
        fI,
        A,
        IM,
        AM,
        fM_all,
        fI_all,
        A_all,
        IM_all,
        AM_all
    )

    return init_params
