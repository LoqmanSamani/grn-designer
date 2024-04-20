import numpy as np


def production(M, productionrate, cellNumber, dt):
    return M + productionrate * cellNumber * dt

def degradation(M, degradationrate, dt):
    return M - degradationrate * M * dt

def free_anker_cells(cells_anker, bM):
    cells_anker_free = np.maximum(0, cells_anker - bM)
    return cells_anker_free

def anchor_binding(fM, k_bind, k_off, dt, cells_anker, bM):
    fM_new = fM - k_bind * fM * free_anker_cells(cells_anker, bM) * dt + k_off * bM * dt
    bM_new = bM + k_bind * fM * free_anker_cells(cells_anker, bM) * dt - k_off * bM * dt
    return fM_new, bM_new

def inhibitor_binding(fM, k_bind, k_off, dt, iM, ibM):
    iM_new = iM - k_bind * fM * iM * dt + k_off * ibM * dt
    fM_new = fM - k_bind * fM * iM * dt + k_off * ibM * dt
    ibM_new = ibM + k_bind * fM * iM * dt - k_off * ibM * dt
    return iM_new, fM_new, ibM_new

def logistic_growth(Cells, bM, growthRate, maxCellNumber, dt):
    return Cells + growthRate * bM * (1 - Cells / maxCellNumber) * dt
