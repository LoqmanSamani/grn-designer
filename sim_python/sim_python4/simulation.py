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
    ) = init_params

    time_ = start

    while time_ <= stop:

        for length in range(com_len):
            for width in range(com_wid):

                # morphogen production
                fM[length, width] = production(
                    pre_con=fM[length, width],
                    pk=k_fm_sec,
                    dt=dt
                )

                # inhibitor production
                fI[length, width] = production(
                    pre_con=fI[length, width],
                    pk=k_fi_sec,
                    dt=dt
                )

                # anchor_binding
                AM[length, width], fM[length, width] = bound_anchor(
                    fm_pre_con=fM[length, width],
                    am_pre_con=AM[length, width],
                    k_on=k_am_on,
                    k_off=k_am_off,
                    a_pre_con=A[length, width],
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
                    pre_con=fM[length, width],
                    dk=k_am_deg,
                    dt=dt
                )
                IM[length, width] = degradation(
                    pre_con=fM[length, width],
                    dk=k_im_deg,
                    dt=dt
                )
                fI[length, width] = degradation(
                    pre_con=fM[length, width],
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
                        specie=fM,
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
                        specie=fM,
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
            A_all[:, :, idx] = A

        # update time
        time_ += dt
        epoch += 1
    print("It's Done!")

    return (fM_all, fI_all, IM_all, AM_all, A_all)
