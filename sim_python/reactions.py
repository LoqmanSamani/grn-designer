import numpy as np


class Reactions:

    def production(self, M, production_rate, cellNumber, dt):

        M = M + production_rate * cellNumber * dt

        return M

    def degradation(self, M, degradation_rate, dt):

        M = M - degradation_rate * M * dt

        return M

    def free_anchor_cells(self, cells_anchor, bM):

        cells_anchor_free = np.maximum(0, cells_anchor - bM)

        return cells_anchor_free

    def anchor_binding(self, fM, k_bind, k_off, dt, cells_anchor, bM):

        cells_anchor_free = self.free_anchor_cells(
            cells_anchor=cells_anchor,
            bM=bM
        )

        fM_new = fM - k_bind * fM * cells_anchor_free * dt + k_off * bM * dt
        bM_new = bM + k_bind * fM * cells_anchor_free * dt - k_off * bM * dt

        return fM_new, bM_new

    def inhibitor_binding(self, fM, k_bind, k_off, dt, iM, ibM):

        iM_new = iM - k_bind * fM * iM * dt + k_off * ibM * dt
        fM_new = fM - k_bind * fM * iM * dt + k_off * ibM * dt
        ibM_new = ibM + k_bind * fM * iM * dt - k_off * ibM * dt

        return iM_new, fM_new, ibM_new

    def logistic_growth(self, Cells, bM, growthRate, maxCellNumber, dt):

        growth = Cells + growthRate * bM * (1 - Cells / maxCellNumber) * dt

        return growth
