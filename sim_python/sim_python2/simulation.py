from initialization import *
from diffusion import *
from reactions import *
from numba import jit
import time
import h5py
import os


@jit
def simulation(init_params):

    (
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

    ) = init_params

    time_ = start

    while time_ <= stop:

        for length in range(com_len):
            for depth in range(com_wid):

                # production
                fM[length, depth] = production(
                    concentration=fM[length, depth],
                    production_rate=k_fm_sec,
                    num_cell=cells_GFP[length, depth],
                    dt=dt
                )

                iM[length, depth] = production(
                    concentration=iM[length, depth],
                    production_rate=k_im_sec,
                    num_cell=cells_iM[length, depth],
                    dt=dt
                )

                # anchor_binding
                fM[length, depth], bM[length, depth] = anchor_binding(
                    fm=fM[length, depth],
                    k_bind=k_fm_bind,
                    k_off=k_fm_off,
                    dt=dt,
                    cells_anchor=cells_anchor[length, depth],
                    bm=bM[length, depth]
                )

                # inhibitor_binding
                iM[length, depth], fM[length, depth], ibM[length, depth] = inhibitor_binding(
                    fm=fM[length, depth],
                    k_bind=k_im_bind,
                    k_off=k_im_off,
                    dt=dt,
                    im=iM[length, depth],
                    ibm=ibM[length, depth]
                )

                # degradation
                fM[length, depth] = degradation(
                    concentration=fM[length, depth],
                    degradation_rate=k_fm_deg,
                    dt=dt
                )

                bM[length, depth] = degradation(
                    concentration=bM[length, depth],
                    degradation_rate=k_bm_deg,
                    dt=dt
                )

                iM[length, depth] = degradation(
                    concentration=iM[length, depth],
                    degradation_rate=k_im_deg,
                    dt=dt
                )

                ibM[length, depth] = degradation(
                    concentration=ibM[length, depth],
                    degradation_rate=k_ibm_deg,
                    dt=dt
                )

                # diffusion
                fM[length, depth] = diffusion2d(
                    specie=fM,
                    length=length,
                    depth=depth,
                    k_diff=d_free,
                    dt=dt,
                    compartment_length=com_len,
                    compartment_depth=com_wid
                )

                iM[length, depth] = diffusion2d(
                    specie=iM,
                    length=length,
                    depth=depth,
                    k_diff=d_i,
                    dt=dt,
                    compartment_length=com_len,
                    compartment_depth=com_wid
                )

                ibM[length, depth] = diffusion2d(
                    specie=ibM,
                    length=length,
                    depth=depth,
                    k_diff=d_i,
                    dt=dt,
                    compartment_length=com_len,
                    compartment_depth=com_wid
                )

                # Cell growth Verhulst
                cells_anchor[length, depth] = logistic_growth(
                    cells=cells_anchor[length, depth],
                    bm=bM[length, depth],
                    growth_rate=growth_rate,
                    num_max_cell=num_max_cell,
                    dt=dt
                )

        # save every saveStepInterval
        if epoch % interval_save == 0:
            idx = int(epoch / interval_save)
            fM_all[:, :, idx] = fM
            bM_all[:, :, idx] = bM
            iM_all[:, :, idx] = iM
            ibM_all[:, :, idx] = ibM
            cells_anchor_all[:, :, idx] = cells_anchor
            cells_GFP_all[:, :, idx] = cells_GFP
            cells_mCherry_all[:, :, idx] = cells_mCherry
            cells_iM_all[:, :, idx] = cells_iM

        # update time
        time_ += dt
        epoch += 1

    return (fM_all, bM_all, iM_all, ibM_all, cells_anchor_all, cells_GFP_all, cells_mCherry_all, cells_iM_all)






