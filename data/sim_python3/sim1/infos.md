infos = {

    "growth rate": 1,
    "max cell number": 1000000,
    "compartment length": 100,
    "compartment width": 100,
    "start": 1,
    "stop": 50,
    "dt": 0.01,
    "cell seed": 100000,
    "save step interval": 10,
    "k_fm_sec": 0.4,
    "k_im_sec": 0.3,
    "k_fm_bind": 0.2,
    "k_fm_off": 0.3,
    "k_im_bind": 0.2,
    "k_im_off": 0.23,
    "k_fm_deg": 0.001,
    "k_im_deg": 0.001,
    "k_bm_deg": 0.001,
    "k_ibm_deg": 0.001,
    "d_free": 0.6,
    "d_i": 0.6

}





cells_GFP = np.zeros((com_len, com_wid), dtype=np.float32)

    cells_GFP[:, 0] = 0.1
    
cells_iM = np.zeros((com_len, com_wid), dtype=np.float32)

    cells_iM[:, -1] = 0.1