infos = {

    "growth rate": 0,
    "max cell number": 10000,
    "compartment length": 1000,
    "compartment width": 1000,
    "start": 1,
    "stop": 20,
    "dt": 0.01,
    "cell seed": 100000000,
    "save step interval": 50,
    "k_fm_sec": 0.4,
    "k_im_sec": 0.3,
    "k_fm_bind": 0.17,
    "k_fm_off": 0.12,
    "k_im_bind": 0.09,
    "k_im_off": 0.01,
    "k_fm_deg": 0.04,
    "k_im_deg": 0.04,
    "k_bm_deg": 0.04,
    "k_ibm_deg": 0.04,
    "d_free": 0.2,
    "d_i": 0.15

}

params = initialization(infos)

result = simulation(

    init_params=params, 
    directory_path="/home/samani/Documents/sim", 
    file_name="sims"

)

simulation duration: about 3 mins

