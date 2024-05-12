from heatmap_plot import *
from scatter_plot import *
from surface_plot import *
from initialization import *
from simulation import *
import os
import h5py



"""
The following code runs a simulation based on the fifth system (/home/samani/Documents/projects/master_project/sim_python/sim_python5). 
It simulates a biological system (100 * 100 cells) in which on one side the first ten columns (from the left) of the compartment are 
capable of producing and releasing green fluorescent protein (GFP), and on the other side the first ten columns (from the right) are 
cells capable of producing and releasing anti-mCherry (mCherry inhibitor). The cells in columns 30-40 are capable of producing mCherry 
molecules, which must be activated by GFP molecules. very cell in the system can produce anchors to capture GFP molecules (anchor=True).
"""




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






# heatmap visualizations
model1 = HeatMap(
    data_path="/home/samani/Documents/sim/sim12.h5",
    video_directory="/home/samani/Documents/sim/",
    video_name="free_mcherry",
    title="Free mCherry",
    x_label="Number of Cells",
    y_label="Number of Cells",
    c_map="RedBlack",
    fps=10,
    interval=50,
    writer='ffmpeg',
    color_bar=True,
    norm=False
)

model1.heatmap_animation(key="MC")


model2 = HeatMap(
    data_path="/home/samani/Documents/sim/sim12.h5",
    video_directory="/home/samani/Documents/sim/",
    video_name="free_gfp",
    title="Free GFP",
    x_label="Number of Cells",
    y_label="Number of Cells",
    c_map="GreenBlack",
    fps=10,
    interval=50,
    writer='ffmpeg',
    color_bar=True,
    norm=False
)

model2.heatmap_animation(key="GFP")





model3 = HeatMap(
    data_path="/home/samani/Documents/sim/sim12.h5",
    video_directory="/home/samani/Documents/sim/",
    video_name="free_inhibitor",
    title="Free Inhibitor",
    x_label="Number of Cells",
    y_label="Number of Cells",
    c_map="PurpleBlack",
    fps=10,
    interval=50,
    writer='ffmpeg',
    color_bar=True,
    norm=False
)

model3.heatmap_animation(key="fI")

model4 = HeatMap(
    data_path="/home/samani/Documents/sim/sim12.h5",
    video_directory="/home/samani/Documents/sim/",
    video_name="inhibitor_mcherry",
    title="Inhibitor-mCherry",
    x_label="Number of Cells",
    y_label="Number of Cells",
    c_map="RedPurpleBlack",
    fps=10,
    interval=50,
    writer='ffmpeg',
    color_bar=True,
    norm=False
)

model4.heatmap_animation(key="IMC")


model5 = HeatMap(
    data_path="/home/samani/Documents/sim/sim12.h5",
    video_directory="/home/samani/Documents/sim/",
    video_name="anchor_mcherry",
    title="mCherry-Anchor",
    x_label="Number of Cells",
    y_label="Number of Cells",
    c_map="YellowBlack",
    fps=10,
    interval=50,
    writer='ffmpeg',
    color_bar=True,
    norm=False
)

model5.heatmap_animation(key="AMC")






# surface plots
surface_animation(
    data_path="/home/samani/Documents/sim/sim12.h5",
    key="MC",
    video_directory="/home/samani/Documents/sim/",
    video_name="free_mcherry",
    title="Free mCherry",
    x_label="Number of Cells",
    y_label="Number of Cells",
    z_label="Concentration",
    colorbar=False,
    c_map="RedBlack"
)



surface_animation(
    data_path="/home/samani/Documents/sim/sim12.h5",
    key="fI",
    video_directory="/home/samani/Documents/sim/",
    video_name="free_inhibitor",
    title="Free Inhibitor",
    x_label="Number of Cells",
    y_label="Number of Cells",
    z_label="Concentration",
    colorbar=False,
    c_map="PurpleBlack"
)



surface_animation(
    data_path="/home/samani/Documents/sim/sim12.h5",
    key="IMC",
    video_directory="/home/samani/Documents/sim/",
    video_name="inhibitor_mcherry",
    title="Inhibitor-mCherry",
    x_label="Number of Cells",
    y_label="Number of Cells",
    z_label="Concentration",
    colorbar=False,
    c_map="RedPurpleBlack"
)



surface_animation(
    data_path="/home/samani/Documents/sim/sim12.h5",
    key="GFP",
    video_directory="/home/samani/Documents/sim/",
    video_name="free_gfp",
    title="Free GFP",
    x_label="Number of Cells",
    y_label="Number of Cells",
    z_label="Concentration",
    colorbar=False,
    c_map="GreenBlack"
)

surface_animation(
    data_path="/home/samani/Documents/sim/sim12.h5",
    key="AMC",
    video_directory="/home/samani/Documents/sim/",
    video_name="anchor_mcherry",
    title="mCherry-Anchor",
    x_label="Number of Cells",
    y_label="Number of Cells",
    z_label="Concentration",
    colorbar=False,
    c_map="YellowBlack"
)







# scatter plots
scatter_animation(
    data_path="/home/samani/Documents/sim/sim12.h5",
    key="MC",
    video_directory="/home/samani/Documents/sim/",
    video_name="free_mcherry",
    title="Free mCherry",
    x_label="Number of Cells",
    y_label="Number of Cells",
    z_label="Concentration",
    colorbar=False,
    c_map="RedBlack"
)



scatter_animation(
    data_path="/home/samani/Documents/sim/sim12.h5",
    key="fI",
    video_directory="/home/samani/Documents/sim/",
    video_name="free_inhibitor",
    title="Free Inhibitor",
    x_label="Number of Cells",
    y_label="Number of Cells",
    z_label="Concentration",
    colorbar=False,
    c_map="PurpleBlack"
)

scatter_animation(
    data_path="/home/samani/Documents/sim/sim12.h5",
    key="IMC",
    video_directory="/home/samani/Documents/sim/",
    video_name="inhibitor_mcherry",
    title="Inhibitor-mCherry",
    x_label="Number of Cells",
    y_label="Number of Cells",
    z_label="Concentration",
    colorbar=False,
    c_map="RedPurpleBlack"
)

scatter_animation(
    data_path="/home/samani/Documents/sim/sim12.h5",
    key="GFP",
    video_directory="/home/samani/Documents/sim/",
    video_name="free_gfp",
    title="Free GFP",
    x_label="Number of Cells",
    y_label="Number of Cells",
    z_label="Concentration",
    colorbar=False,
    c_map="GreenBlack"
)

scatter_animation(
    data_path="/home/samani/Documents/sim/sim12.h5",
    key="AMC",
    video_directory="/home/samani/Documents/sim/",
    video_name="anchor_mcherry",
    title="mCherry-Anchor",
    x_label="Number of Cells",
    y_label="Number of Cells",
    z_label="Concentration",
    colorbar=False,
    c_map="YellowBlack"
)



