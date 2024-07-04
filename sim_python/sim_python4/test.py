from initialization import *
from simulation import *
import os
import h5py




infos = {
    "compartment length": 30,
    "compartment width": 30,
    "initial cell number": 5,
    "start": 1,
    "stop": 5,
    "dt": 0.02,
    "save step interval": 10,
    "k_fm_sec": 0.2,
    "k_fi_sec": 0.1,
    "k_am_on": 0.04,
    "k_am_off": 2e-6,
    "k_im_on": 0.2,
    "k_im_off": 2e-4,
    "k_fm_deg": 0.02,
    "k_fi_deg": 0.02,
    "k_im_deg": 0.02,
    "k_am_deg": 0.02,
    "k_m_diff": .9,
    "k_i_diff": .7,
    "k_im_diff": .5
}

params = initialization(infos, anchor=False, num_col=3, ratio=2)


result = simulation(init_params=params, one_cell=True)

full_path = "/home/samani/Documents/sim"

if not os.path.exists(full_path):
    os.makedirs(full_path)

full_file_path = os.path.join(full_path, "sim2.h5")

with h5py.File(full_file_path, "w") as file:
    file.create_dataset("fM", data=result[0])
    file.create_dataset("fI", data=result[1])
    file.create_dataset("IM", data=result[2])
    file.create_dataset("AM", data=result[3])
    file.create_dataset("M_cells", data=result[4])
    file.create_dataset("I_cells", data=result[5])
    file.create_dataset("A_cells", data=result[6])
