from initialization import *
from simulation import *
import os
import h5py




infos = {
    "compartment length": 100,
    "compartment width": 100,
    "start": 1,
    "stop": 10,
    "dt": 0.01,
    "save step interval": 50,
    "k_fm_sec": 0.9,
    "k_fi_sec": 0.9,
    "k_am_on": 0.1,
    "k_am_off": 0.09,
    "k_im_on": 0.2,
    "k_im_off": 0.3,
    "k_fm_deg": 0.1,
    "k_fi_deg": 0.2,
    "k_im_deg": 0.3,
    "k_am_deg": 0.2,
    "k_m_diff": 0.3,
    "k_i_diff": 0.9,
    "k_im_diff": 0.9
}

params = initialization(infos)

result = simulation(init_params=params, one_cell=False)

full_path = "/home/samani/Documents/sim"

if not os.path.exists(full_path):
    os.makedirs(full_path)

full_file_path = os.path.join(full_path, "sim2.h5")

with h5py.File(full_file_path, "w") as file:
    file.create_dataset("fM", data=result[0])
    file.create_dataset("fI", data=result[1])
    file.create_dataset("IM", data=result[2])
    file.create_dataset("AM", data=result[3])
    file.create_dataset("A", data=result[4])
