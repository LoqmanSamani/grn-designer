from diffusion import *
from reactions import *
from numba import jit


@jit
def simulation(init_params, one_cell=True):
    """
    Simulate the system in 2D.

    Arguments:
        - init_params (tuple): Contains initialized parameters and simulation matrices.
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
            - num_time_steps (integer): Total number of time steps in the simulation.
            - epoch (integer): Current epoch or time step.
            - interval_save (integer): Interval used to save concentration matrices.
            - fM (ndarray): Initial concentration of free morphogen.
            - fI (ndarray): Initial concentration of free inhibitor.
            - A (ndarray): Initial concentration of anchor-morphogen.
            - IM (ndarray): Initial concentration of inhibitor-morphogen.
            - AM (ndarray): Initial concentration of anchor-morphogen.

        - one_cell (bool, optional): Flag to indicate whether each cell be affected (diffusion) by one neighbor cell
             or two cell Default is one cell(True). When False, the diffusion model considers diffusion across two
             neighboring cells in depth.

        Returns:
            - Tuple of ndarrays: Contains concentration matrices of free morphogen, free inhibitor,
              inhibitor-morphogen, anchor-morphogen, and anchor at each time step.
        """

    (
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
    ) = init_params

    time_ = start
    max_epoch = AM_all.shape[2] * interval_save

    fM = fM_all[:, :, 0]
    fI = fI_all[:, :, 0]
    IM = IM_all[:, :, 0]
    AM = AM_all[:, :, 0]
    M_cells = M_cells_all[:, :, 0]
    I_cells = I_cells_all[:, :, 0]
    A_cells = A_cells_all[:, :, 0]

    while time_ <= stop or epoch <= max_epoch:

        for length in range(com_len):
            for width in range(com_wid):

                # morphogen production
                fM[length, width] = production(
                    pre_con=fM[length, width],
                    num_cell=M_cells[length, width],
                    pk=k_fm_sec,
                    dt=dt
                )

                # inhibitor production
                fI[length, width] = production(
                    pre_con=fI[length, width],
                    num_cell=I_cells[length, width],
                    pk=k_fi_sec,
                    dt=dt
                )

                # anchor_binding
                AM[length, width], fM[length, width] = bound_anchor(
                    fm_pre_con=fM[length, width],
                    am_pre_con=AM[length, width],
                    k_on=k_am_on,
                    k_off=k_am_off,
                    a_cells=A_cells[length, width],
                    dt=dt
                )

                # inhibitor_binding
                fM[length, width], fI[length, width], IM[length, width] = bound_inhibitor(
                    fm_pre_con=fM[length, width],
                    fi_pre_con=fI[length, width],
                    im_pre_con=IM[length, width],
                    k_on=k_im_on,
                    k_off=k_im_off,
                    dt=dt
                )

                # degradation
                fM[length, width] = degradation(
                    pre_con=fM[length, width],
                    dk=k_fm_deg,
                    dt=dt
                )
                AM[length, width] = degradation(
                    pre_con=AM[length, width],
                    dk=k_am_deg,
                    dt=dt
                )
                IM[length, width] = degradation(
                    pre_con=IM[length, width],
                    dk=k_im_deg,
                    dt=dt
                )
                fI[length, width] = degradation(
                    pre_con=fI[length, width],
                    dk=k_fi_deg,
                    dt=dt
                )

                # diffusion
                if one_cell:

                    fM[length, width] = diffusion1(
                        specie=fM,
                        length=length,
                        width=width,
                        k_diff=k_m_diff,
                        dt=dt,
                        compartment_length=com_len,
                        compartment_width=com_wid
                    )
                    fI[length, width] = diffusion1(
                        specie=fI,
                        length=length,
                        width=width,
                        k_diff=k_i_diff,
                        dt=dt,
                        compartment_length=com_len,
                        compartment_width=com_wid
                    )
                    IM[length, width] = diffusion1(
                        specie=IM,
                        length=length,
                        width=width,
                        k_diff=k_im_diff,
                        dt=dt,
                        compartment_length=com_len,
                        compartment_width=com_wid
                    )
                else:

                    fM[length, width] = diffusion2(
                        specie=fM,
                        length=length,
                        width=width,
                        k_diff=k_m_diff,
                        dt=dt,
                        compartment_length=com_len,
                        compartment_width=com_wid
                    )
                    fI[length, width] = diffusion2(
                        specie=fI,
                        length=length,
                        width=width,
                        k_diff=k_i_diff,
                        dt=dt,
                        compartment_length=com_len,
                        compartment_width=com_wid
                    )
                    IM[length, width] = diffusion2(
                        specie=IM,
                        length=length,
                        width=width,
                        k_diff=k_im_diff,
                        dt=dt,
                        compartment_length=com_len,
                        compartment_width=com_wid
                    )

        # save every saveStepInterval
        if epoch % interval_save == 0:

            idx = int(epoch / interval_save)
            fM_all[:, :, idx] = fM
            fI_all[:, :, idx] = fI
            IM_all[:, :, idx] = IM
            AM_all[:, :, idx] = AM
            M_cells_all[:, :, idx] = M_cells
            I_cells_all[:, :, idx] = I_cells
            A_cells_all[:, :, idx] = A_cells

        # update time
        time_ += dt
        epoch += 1
    print("It's Done!")

    return (fM_all, fI_all, IM_all, AM_all, M_cells_all, I_cells_all, A_cells_all)
