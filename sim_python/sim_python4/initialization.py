import numpy as np


def initialization(params, anchor=False, num_col=10, ratio=2):
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
        - anchor (boolean): if True, anchor cells will be initialized.
        num_col (int): number of the matrix columns, which are able to produce the product.
        ratio (int or float): use to define the concentration ratio.

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
    init_num_cell = params["initial cell number"]

    # Calculating number of time steps and intermediate matrices
    num_time_steps = int(np.ceil(stop / dt)) + 1
    idx = int(np.ceil(num_time_steps / interval_save))
    epoch = 1

    fM = np.zeros((com_len, com_wid), dtype=np.float32)  # Free morphogen, 2D matrix
    fI = np.zeros((com_len, com_wid), dtype=np.float32)  # Free inhibitor, 2D matrix
    IM = np.zeros((com_len, com_wid), dtype=np.float32)  # Inhibitor-morphogen, 2D matrix
    AM = np.zeros((com_len, com_wid), dtype=np.float32)  # Anchor-morphogen, 2D matrix

    M_cells = np.zeros((com_len, com_wid), dtype=np.float32)  # cells, which produce morphogen, 2D matrix
    I_cells = np.zeros((com_len, com_wid), dtype=np.float32)  # cells, which produce inhibitor, 2D matrix
    A_cells = np.zeros((com_len, com_wid), dtype=np.float32)  # cells, which have anchor, 2D matrix

    #  initialize cells
    M_cells, I_cells, A_cells = initial_cell_seed1(
        num_col=num_col,
        ratio=ratio,
        init_num_cell=init_num_cell,
        M_cells=M_cells,
        I_cells=I_cells,
        A_cells=A_cells,
        anchor=anchor
    )

    # Initializing concentration matrices
    fM_all = np.zeros((com_len, com_wid, idx), dtype=np.float32)  # Free morphogen, 3D matrix
    fM_all[:, :, 0] = fM
    fI_all = np.zeros((com_len, com_wid, idx), dtype=np.float32)  # Free inhibitor, 3D matrix
    fI_all[:, :, 0] = fI
    IM_all = np.zeros((com_len, com_wid, idx), dtype=np.float32)  # Inhibitor-morphogen, 3D matrix
    IM_all[:, :, 0] = IM
    AM_all = np.zeros((com_len, com_wid, idx), dtype=np.float32)  # Anchor-morphogen, 3D matrix
    AM_all[:, :, 0] = AM
    M_cells_all = np.zeros((com_len, com_wid, idx), dtype=np.float32)  # cells which have anchor, 3D matrix
    M_cells_all[:, :, 0] = M_cells
    I_cells_all = np.zeros((com_len, com_wid, idx), dtype=np.float32)  # cells which have anchor, 3D matrix
    I_cells_all[:, :, 0] = I_cells
    A_cells_all = np.zeros((com_len, com_wid, idx), dtype=np.float32)  # cells which have anchor, 3D matrix
    A_cells_all[:, :, 0] = A_cells

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
        init_num_cell,
        fM_all,
        fI_all,
        IM_all,
        AM_all,
        M_cells_all,
        I_cells_all,
        A_cells_all
    )

    return init_params


def initial_cell_seed1(num_col, ratio, init_num_cell, M_cells, I_cells, A_cells, anchor):
    """
    Sets initial cell configurations based on given parameters.

    Arguments:
        num_col (int): number of the matrix columns, which are able to produce the product.
        ratio (int or float): use to define the concentration ratio
        init_num_cell (float): Initial cell number (I think it is the initial concentration of product in corresponding cells !!!).
        A_cells (ndarray): Array for anchor cells.
        M_cells (ndarray): Array for Morphogen (GFP) cells.
        I_cells (ndarray): Array for inhibitor cells.
        anchor (boolean) : if True, anchor cells will be initialized.

    Returns:
        tuple: Arrays for anchor, morphogen (GFP), and inhibitor cells with updated configurations.
    """
    if anchor:
        M_cells[:, 0:num_col] = init_num_cell
        I_cells[:, - num_col:] = init_num_cell
        A_cells[:] = init_num_cell / ratio
    else:
        M_cells[:, 0:num_col] = init_num_cell
        I_cells[:, - num_col:] = init_num_cell

    return M_cells, I_cells, A_cells


