from diffusion import *
from reactions import *
from simulation import *
from initialization import *
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation







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



params = initialization(infos)

result = simulation(
    init_params=params,
)


full_path = "/home/samani/Documents/sim"
if not os.path.exists(full_path):
    os.makedirs(full_path)

full_file_path = os.path.join(full_path, "sim5.h5")

with h5py.File(full_file_path, "w") as file:
    file.create_dataset("expected_fM", data=result[0])
    file.create_dataset("expected_bM", data=result[1])
    file.create_dataset("expected_iM", data=result[2])
    file.create_dataset("expected_ibM", data=result[3])
    file.create_dataset("expected_cells_anker_all", data=result)
    file.create_dataset("expected_cells_GFP_all", data=result[5])
    file.create_dataset("expected_cells_mCherry_all", data=result[6])
    file.create_dataset("expected_cells_iM_all", data=result[7])
    
 
    

def plot_animation(dataset, title):

    fig, ax = plt.subplots()

    def update(frame):
        # inferno
        # magma
        # hot
        ax.clear()
        ax.set_title(f"{title} - {frame + 1}")
        ax.imshow(dataset[:, :, frame], cmap="hot")

    anim = FuncAnimation(fig, update, frames=dataset.shape[2], interval=200)
    anim.save(f'/home/samani/Documents/sim/{title}.mp4', writer='ffmpeg')
    plt.show()






sim_result = "/home/samani/Documents/sim/sim5.h5"


with h5py.File(sim_result, 'r') as hdf_file:
    print(hdf_file.keys())

    iM = hdf_file["expected_iM"]
    # iM_norm = (iM - np.min(iM)) / (np.max(iM) - np.min(iM))
    fM = hdf_file["expected_fM"]
    # fM_norm = (fM - np.min(fM)) / (np.max(fM) - np.min(fM))
    ibM = hdf_file["expected_ibM"]
    # ibM_norm = (ibM - np.min(ibM)) / (np.max(ibM) - np.min(ibM))
    bM = hdf_file["expected_bM"]
    # bM_norm = (bM - np.min(bM)) / (np.max(bM) - np.min(bM))

    plot_animation(iM, "iM")
    # plot_animation(iM_norm, "iM_norm")
    plot_animation(fM, "fM")
    # plot_animation(fM_norm, "fM_norm")
    plot_animation(ibM, "ibM")
    # plot_animation(ibM_norm, "ibM_norm")
    plot_animation(bM, "bM")
    # plot_animation(bM_norm, "bM")
