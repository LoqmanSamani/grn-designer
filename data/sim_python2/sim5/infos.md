infos = {

    "growth rate": 1,
    "max cell number": 1000000,
    "compartment length": 100,
    "compartment width": 100,
    "start": 1,
    "stop": 200,
    "dt": 0.05,
    "cell seed": 1000000,
    "save step interval": 10,
    "k_fm_sec": 0.6,
    "k_im_sec": 0.7,
    "k_fm_bind": 0.8,
    "k_fm_off": 0.5,
    "k_im_bind": 0.7,
    "k_im_off": 0.5,
    "k_fm_deg": 0.3,
    "k_im_deg": 0.3,
    "k_bm_deg": 0.3,
    "k_ibm_deg": 0.3,
    "d_free": 4,
    "d_i": 4

}

cells_GFP = np.zeros((com_len, com_wid), dtype=np.float32)

    cells_GFP[:, 0] = 0.1
   

cells_iM = np.zeros((com_len, com_wid), dtype=np.float32)

    cells_iM[:, -1] = 0.1
   
