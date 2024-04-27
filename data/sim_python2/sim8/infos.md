infos = {

    "growth rate": 1,
    "max cell number": 10000,
    "compartment length": 100,
    "compartment width": 100,
    "start": 1,
    "stop": 200,
    "dt": 0.01,
    "cell seed": 1000000,
    "save step interval": 50,
    "k_fm_sec": 0.9,
    "k_im_sec": 0.9,
    "k_fm_bind": 0.8,
    "k_fm_off": 0.3,
    "k_im_bind": 0.7,
    "k_im_off": 0.2,
    "k_fm_deg": 0.3,
    "k_im_deg": 0.3,
    "k_bm_deg": 0.3,
    "k_ibm_deg": 0.3,
    "d_free": 0.9,
    "d_i": 0.9
}


params = initialization(infos)

result = simulation(
    init_params=params,
)

full_path = "/home/samani/Documents/sim"

if not os.path.exists(full_path):

    os.makedirs(full_path)

full_file_path = os.path.join(full_path, "sim8.h5")

with h5py.File(full_file_path, "w") as file:

    file.create_dataset("expected_fM", data=result[0])
    file.create_dataset("expected_bM", data=result[1])
    file.create_dataset("expected_iM", data=result[2])
    file.create_dataset("expected_ibM", data=result[3])
    file.create_dataset("expected_cells_anker_all", data=result)
    file.create_dataset("expected_cells_GFP_all", data=result[5])
    file.create_dataset("expected_cells_mCherry_all", data=result[6])
    file.create_dataset("expected_cells_iM_all", data=result[7])



cells_GFP = np.zeros((com_len, com_wid), dtype=np.float32)

    cells_GFP[:, 0] = 1
    cells_GFP[:, -1] = 1

cells_iM = np.zeros((com_len, com_wid), dtype=np.float32)

    cells_iM[:, 50] = 1
    cells_iM[:, 40] = 1
    cells_iM[:, 60] = 1
    
