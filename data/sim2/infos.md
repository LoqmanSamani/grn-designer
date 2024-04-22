infos = {

    "growthRate": 0,
    "maxCellNumber": 10000,
    "compartment_length": 100,
    "compartment_depth": 100,
    "t_max": 10,
    "dt": 0.01,
    "cellSeed": 100000000,
    "saveStepInterval": 100,
    "k_fM_src": 0.3,
    "k_iM_src": 0.25,
    "k_fM_bind": 0.3,
    "k_fM_off": 0.2,
    "k_iM_bind": 0.013,
    "k_iM_off": 0.01,
    "k_fM_deg": 0.002,
    "k_iM_deg": 0.002,
    "k_bM_deg": 0.002,
    "k_ibM_deg": 0.002,
    "d_free": 0.2,
    "d_i": 0.2
}


model = Simulation(

    saveStepInterval=100,
    directory_path="/home/samani/Documents/sim",
    file_name="simulation"

)

model.simulate2DMS(infos)

simulation duration: 138 s