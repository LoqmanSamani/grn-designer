import numpy as np


def initialization(params, anchor=False, num_col=10, ratio=2):
    """
    Initializes the parameters needed to simulate the model.

    Arguments:
        - params (dictionary): Contains all necessary parameters for simulating the model.
            - k_fm_sec (float): GFP secretion rate constant.
            - k_mc_sec (float): mCherry secretion rate constant.
            - k_fi_sec (float): Inhibitor secretion rate constant.
            - k_amc_on (float): Binding anchor-mCherry rate constant.
            - k_amc_off (float): Unbinding anchor-morphogen rate constant.
            - k_imc_on (float): Binding inhibitor-mCherry rate constant.
            - k_imc_off (float): Unbinding inhibitor-mCherry rate constant.
            - k_fm_deg (float): Free GFP degradation rate constant.
            - k_mc_deg (float): Free mCherry degradation rate constant.
            - k_fi_deg (float): Free inhibitor degradation rate constant.
            - k_imc_deg (float): Inhibitor-mCherry degradation rate constant.
            - k_amc_deg (float): Anchor-mCherry degradation rate constant.
            - k_m_diff (float): Free GFP diffusion rate.
            - k_mc_diff (float): Free mCherry diffusion rate.
            - k_i_diff (float): Free inhibitor diffusion rate.
            - k_imc_diff (float): Free inhibitor-mCherry diffusion rate.
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
    k_mc_sec = params["k_mc_sec"]
    k_fi_sec = params["k_fi_sec"]
    k_amc_on = params["k_amc_on"]
    k_amc_off = params["k_amc_off"]
    k_imc_on = params["k_imc_on"]
    k_imc_off = params["k_imc_off"]
    k_fm_deg = params["k_fm_deg"]
    k_mc_deg = params["k_mc_deg"]
    k_fi_deg = params["k_fi_deg"]
    k_imc_deg = params["k_imc_deg"]
    k_amc_deg = params["k_amc_deg"]
    k_m_diff = params["k_m_diff"]
    k_mc_diff = params["k_mc_diff"]
    k_i_diff = params["k_i_diff"]
    k_imc_diff = params["k_imc_diff"]
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

    fM = np.zeros((com_len, com_wid), dtype=np.float32)  # Free GFP, 2D matrix
    MC = np.zeros((com_len, com_wid), dtype=np.float32)  # Free mCherry, 2D matrix
    fI = np.zeros((com_len, com_wid), dtype=np.float32)  # Free inhibitor, 2D matrix
    IMC = np.zeros((com_len, com_wid), dtype=np.float32)  # Inhibitor-mCherry, 2D matrix
    AMC = np.zeros((com_len, com_wid), dtype=np.float32)  # Anchor-mCherry, 2D matrix

    M_cells = np.zeros((com_len, com_wid), dtype=np.float32)  # cells, which produce GFP, 2D matrix
    MC_cells = np.zeros((com_len, com_wid), dtype=np.float32)  # cells, which produce mCherry, 2D matrix
    I_cells = np.zeros((com_len, com_wid), dtype=np.float32)  # cells, which produce inhibitor, 2D matrix
    A_cells = np.zeros((com_len, com_wid), dtype=np.float32)  # cells, which have anchor, 2D matrix

    #  initialize cells
    M_cells, MC_cells, I_cells, A_cells = initial_cell_seed1(
        num_col=num_col,
        ratio=ratio,
        init_num_cell=init_num_cell,
        M_cells=M_cells,
        MC_cells=MC_cells,
        I_cells=I_cells,
        A_cells=A_cells,
        anchor=anchor
    )

    # Initializing concentration matrices
    fM_all = np.zeros((com_len, com_wid, idx), dtype=np.float32)  # Free GFP, 3D matrix
    fM_all[:, :, 0] = fM
    MC_all = np.zeros((com_len, com_wid, idx), dtype=np.float32)  # Free mCherry, 3D matrix
    MC_all[:, :, 0] = MC
    fI_all = np.zeros((com_len, com_wid, idx), dtype=np.float32)  # Free inhibitor, 3D matrix
    fI_all[:, :, 0] = fI
    IMC_all = np.zeros((com_len, com_wid, idx), dtype=np.float32)  # Inhibitor-mCherry, 3D matrix
    IMC_all[:, :, 0] = IMC
    AMC_all = np.zeros((com_len, com_wid, idx), dtype=np.float32)  # Anchor-mCherry, 3D matrix
    AMC_all[:, :, 0] = AMC
    M_cells_all = np.zeros((com_len, com_wid, idx), dtype=np.float32)  # cells which produce GFP, 3D matrix
    M_cells_all[:, :, 0] = M_cells
    MC_cells_all = np.zeros((com_len, com_wid, idx), dtype=np.float32)  # cells which produce mCherry, 3D matrix
    MC_cells_all[:, :, 0] = MC_cells
    I_cells_all = np.zeros((com_len, com_wid, idx), dtype=np.float32)  # cells which produce Inhibitor, 3D matrix
    I_cells_all[:, :, 0] = I_cells
    A_cells_all = np.zeros((com_len, com_wid, idx), dtype=np.float32)  # cells which have anchor, 3D matrix
    A_cells_all[:, :, 0] = A_cells

    # Packaging initialized parameters into a tuple
    init_params = (
        k_fm_sec,
        k_mc_sec,
        k_fi_sec,
        k_amc_on,
        k_amc_off,
        k_imc_on,
        k_imc_off,
        k_fm_deg,
        k_mc_deg,
        k_fi_deg,
        k_imc_deg,
        k_amc_deg,
        k_m_diff,
        k_mc_diff,
        k_i_diff,
        k_imc_diff,
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
        MC_all,
        fI_all,
        IMC_all,
        AMC_all,
        M_cells_all,
        MC_cells_all,
        I_cells_all,
        A_cells_all
    )

    return init_params


def initial_cell_seed1(num_col, ratio, init_num_cell, M_cells, MC_cells, I_cells, A_cells, anchor):
    """
    Sets initial cell configurations based on given parameters.

    Arguments:
        num_col (int): number of the matrix columns, which are able to produce the product.
        ratio (int or float): use to define the concentration ratio
        init_num_cell (float): Initial cell number (I think it is the initial concentration of product in corresponding cells !!!).
        A_cells (ndarray): Array for anchor cells.
        M_cells (ndarray): Array for Morphogen (GFP) cells.
        MC_cells (ndarray): Array for mCherry cells
        I_cells (ndarray): Array for inhibitor cells.
        anchor (boolean) : if True, anchor cells will be initialized.

    Returns:
        tuple: Arrays for anchor, morphogen (GFP), and inhibitor cells with updated configurations.
    """
    if anchor:
        M_cells[:, 0:num_col] = init_num_cell
        MC_cells[:, 30:30+num_col] = init_num_cell/ratio
        I_cells[:, - num_col:] = init_num_cell
        A_cells[:] = init_num_cell / ratio
    else:
        M_cells[:, 0:num_col] = init_num_cell
        MC_cells[:, 30:30+num_col] = init_num_cell/ratio
        I_cells[:, - num_col:] = init_num_cell

    return M_cells, MC_cells, I_cells, A_cells
