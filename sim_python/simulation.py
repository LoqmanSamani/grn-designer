from reactions import Reactions
from diffusion import Diffusion
from initialization import Initialization
import h5py
import os


class Simulation:

    def __init__(self, saveStepInterval, directory_path, file_name):

        self.saveStepInterval = saveStepInterval
        self.directory_path = directory_path
        self.file_name = file_name
        self.sim_output = {}

    def simulate2DMS(self, theta):

        initialization = Initialization()
        reactions = Reactions()
        diffusion = Diffusion()

        growthRate = theta["growthRate"]
        maxCellNumber = theta["maxCellNumber"]

        # timestep = 1
        # load initial cell numbers from the ic-... file defined in myoptions
        # cells_anker, cells_GFP, cells_mCherry, cells_iM = initialcellseed(
        # cl, cd, cellNumberInitial, cells_anker, cells_GFP, cells_mCherry, cells_iM
        # )

        sim_input = initialization.set_initial_condition(
            theta=theta
        )
        cl = sim_input["cl"]
        cd = sim_input["cd"]
        t = sim_input["t"]
        t_max = sim_input["t_max"]
        dt = sim_input["dt"]
        num_time_steps = sim_input["num_time_steps"]
        timestep = sim_input["timestep"]
        cellNumberInitial = sim_input["cellNumberInitial"]
        fM = sim_input["fM"]
        bM = sim_input["bM"]
        iM = sim_input["iM"]
        ibM = sim_input["ibM"]
        fM_all = sim_input["fM_all"]
        bM_all = sim_input["bM_all"]
        iM_all = sim_input["iM_all"]
        ibM_all = sim_input["ibM_all"]
        cells_anchor_all = sim_input["cells_anchor_all"]
        cells_GFP_all = sim_input["cells_GFP_all"]
        cells_mCherry_all = sim_input["cells_mCherry_all"]
        cells_iM_all = sim_input["cells_iM_all"]
        cells_anchor = sim_input["cells_anchor"]
        cells_GFP = sim_input["cells_GFP"]
        cells_mCherry = sim_input["cells_mCherry"]
        cells_iM = sim_input["cells_iM"]

        while t <= t_max:
            tmp_fM = fM.copy()
            tmp_bM = bM.copy()
            tmp_iM = iM.copy()
            tmp_ibM = ibM.copy()
            tmp_cells_anchor = cells_anchor.copy()
            tmp_cells_GFP = cells_GFP.copy()
            tmp_cells_mCherry = cells_mCherry.copy()
            tmp_cells_iM = cells_iM.copy()

            for length in range(cl):
                for depth in range(cd):
                    """ production(self, M, production_rate, cellNumber, dt) """
                    # production
                    fM[length, depth] = reactions.production(
                        M=tmp_fM[length, depth],
                        production_rate=theta["k_fM_src"],
                        cellNumber=cellNumberInitial,
                        dt=dt
                    )
                    iM[length, depth] = reactions.production(
                        M=tmp_iM[length, depth],
                        production_rate=theta["k_iM_src"],
                        cellNumber=cellNumberInitial,
                        dt=dt
                    )
                    # anchor_binding
                    fM[length, depth], bM[length, depth] = reactions.anchor_binding(
                        fM=tmp_fM[length, depth],
                        k_bind=theta["k_fM_bind"],
                        k_off=theta["k_fM_off"],
                        dt=dt,
                        cells_anchor=tmp_cells_anchor[length, depth],
                        bM=tmp_bM[length, depth]
                    )
                    # inhibitor_binding
                    iM[length, depth], fM[length, depth], ibM[length, depth] = reactions.inhibitor_binding(
                        fM=tmp_fM[length, depth],
                        k_bind=theta["k_iM_bind"],
                        k_off=theta["k_iM_off"],
                        dt=dt,
                        iM=tmp_iM[length, depth],
                        ibM=tmp_ibM[length, depth]
                    )
                    # degradation
                    fM[length, depth] = reactions.degradation(
                        M=tmp_fM[length, depth],
                        degradation_rate=theta["k_fM_deg"],
                        dt=dt
                    )
                    bM[length, depth] = reactions.degradation(
                        M=tmp_bM[length, depth],
                        degradation_rate=theta["k_bM_deg"],
                        dt=dt
                    )
                    iM[length, depth] = reactions.degradation(
                        M=tmp_iM[length, depth],
                        degradation_rate=theta["k_iM_deg"],
                        dt=dt
                    )
                    ibM[length, depth] = reactions.degradation(
                        M=tmp_ibM[length, depth],
                        degradation_rate=theta["k_ibM_deg"],
                        dt=dt
                    )
                    # diffusion
                    fM[length, depth] = diffusion.diffusion2d(
                        specie=tmp_fM,
                        length=length,
                        depth=depth,
                        k_diff=theta["d_free"],
                        dt=dt,
                        compartment_length=cl,
                        compartment_depth=cd
                    )
                    iM[length, depth] = diffusion.diffusion2d(
                        specie=tmp_iM,
                        length=length,
                        depth=depth,
                        k_diff=theta["d_i"],
                        dt=dt,
                        compartment_length=cl,
                        compartment_depth=cd
                    )
                    ibM[length, depth] = diffusion.diffusion2d(
                        specie=tmp_ibM,
                        length=length,
                        depth=depth,
                        k_diff=theta["d_i"],
                        dt=dt,
                        compartment_length=cl,
                        compartment_depth=cd
                    )
                    # Cell growth Verhulst
                    cells_anchor[length, depth] = reactions.logistic_growth(
                        Cells=tmp_cells_anchor[length, depth],
                        bM=tmp_bM[length, depth],
                        growthRate=growthRate,
                        maxCellNumber=maxCellNumber,
                        dt=dt
                    )

            # save every saveStepInterval
            if timestep % self.saveStepInterval == 0:
                idx = int(timestep / self.saveStepInterval)
                fM_all[:, :, idx] = fM
                bM_all[:, :, idx] = bM
                iM_all[:, :, idx] = iM
                ibM_all[:, :, idx] = ibM
                cells_anchor_all[:, :, idx] = cells_anchor
                cells_GFP_all[:, :, idx] = cells_GFP
                cells_mCherry_all[:, :, idx] = cells_mCherry
                cells_iM_all[:, :, idx] = cells_iM

            # update time
            t += dt
            timestep += 1

        full_path = os.path.join(self.directory_path, self.file_name)
        if not os.path.exists(self.directory_path):
            os.makedirs(self.directory_path)

        with h5py.File(full_path, "w") as file:

            file.create_dataset("expected_fM", data=fM_all[:, :, 1:])
            file.create_dataset("expected_bM", data=bM_all[:, :, 1:])
            file.create_dataset("expected_iM", data=iM_all[:, :, 1:])
            file.create_dataset("expected_ibM", data=ibM_all[:, :, 1:])
            file.create_dataset("expected_cells_anker_all", data=cells_anchor_all[:, :, 1:])
            file.create_dataset("expected_cells_GFP_all", data=cells_GFP_all[:, :, 1:])
            file.create_dataset("expected_cells_mCherry_all", data=cells_mCherry_all[:, :, 1:])
            file.create_dataset("expected_cells_iM_all", data=cells_iM_all[:, :, 1:])

        print("Simulation results saved to:", full_path)

        sim_output = {
            "fM_all": fM_all,
            "bM_all": bM_all,
            "iM_all": iM_all,
            "ibM_all": ibM_all,
            "cells_anker_all": cells_anchor_all,
            "cells_GFP_all": cells_GFP_all,
            "cells_mCherry_all": cells_mCherry_all,
            "cells_iM_all": cells_iM_all
        }

        self.sim_output = sim_output



