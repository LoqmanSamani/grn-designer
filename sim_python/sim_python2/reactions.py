import numpy as np
from numba import jit


@jit
def production(concentration, production_rate, num_cell_init, dt):

    concentration = concentration + production_rate * num_cell_init * dt

    return concentration


@jit
def degradation(concentration, degradation_rate, dt):

    concentration = concentration - degradation_rate * concentration * dt

    return concentration


@jit
def free_anchor_cells(cells_anchor, bm):

    cells_anchor_free = np.maximum(0, cells_anchor - bm)

    return cells_anchor_free


@jit
def anchor_binding(fm, k_bind, k_off, dt, cells_anchor, bm):

    cells_anchor_free = free_anchor_cells(
        cells_anchor=cells_anchor,
        bm=bm
    )

    fm_new = fm - k_bind * fm * cells_anchor_free * dt + k_off * bm * dt
    bm_new = bm + k_bind * fm * cells_anchor_free * dt - k_off * bm * dt

    return fm_new, bm_new


@jit
def inhibitor_binding(fm, k_bind, k_off, dt, im, ibm):

    im_new = im - k_bind * fm * im * dt + k_off * ibm * dt
    fm_new = fm - k_bind * fm * im * dt + k_off * ibm * dt
    ibm_new = ibm + k_bind * fm * im * dt - k_off * ibm * dt

    return im_new, fm_new, ibm_new


@jit
def logistic_growth(cells, bm, growth_rate, num_max_cell, dt):

    growth = cells + growth_rate * bm * (1 - cells / num_max_cell) * dt

    return growth
