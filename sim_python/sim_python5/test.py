from initialization import *
from simulation import *
import os
import h5py



infos = {
    "compartment length": 100,
    "compartment width": 100,
    "initial cell number": 2,
    "start": 1,
    "stop": 50,
    "dt": 0.02,
    "save step interval": 10,
    "k_fm_sec": 0.5,
    "k_mc_sec": 0.5,
    "k_fi_sec": 0.5,
    "k_amc_on": 0.3,
    "k_amc_off": 2e-6,
    "k_imc_on": 0.2,
    "k_imc_off": 2e-4,
    "k_fm_deg": 0.02,
    "k_mc_deg": 0.02,
    "k_fi_deg": 0.02,
    "k_imc_deg": 0.02,
    "k_amc_deg": 0.02,
    "k_m_diff": 4,
    "k_mc_diff": 3,
    "k_i_diff": 4,
    "k_imc_diff": 2.8
}

params = initialization(infos, anchor=True, num_col=10, ratio=5)


result = simulation(init_params=params, one_cell=True)

full_path = "/home/samani/Documents/sim"

if not os.path.exists(full_path):
    os.makedirs(full_path)

full_file_path = os.path.join(full_path, "sim12.h5")

with h5py.File(full_file_path, "w") as file:
    file.create_dataset("GFP", data=result[0])
    file.create_dataset("MC", data=result[1])
    file.create_dataset("fI", data=result[2])
    file.create_dataset("IMC", data=result[3])
    file.create_dataset("AMC", data=result[4])
    file.create_dataset("M_cells", data=result[5])
    file.create_dataset("MC_cells", data=result[6])
    file.create_dataset("I_cells", data=result[7])
    file.create_dataset("A_cells", data=result[8])
