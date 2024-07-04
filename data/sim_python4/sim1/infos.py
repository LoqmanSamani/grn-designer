from initialization import *
from simulation import *
import os
import h5py



"""
The following code runs a simulation based on the fourth system (/home/samani/Documents/projects/master_project/sim_python/sim_python4). 
It simulates a biological system (100 * 100 cells) in which, on one side, the first ten columns (from the left) of the compartment are
capable of producing and releasing green fluorescent protein (GFP), and on the other side, the first ten columns (from the right) 
cells are capable of producing and releasing anti-GFP (GFP inhibitor). There is no potential to produce anchors (anchor=False).
"""



# parameter dictionary
infos = {
    "compartment length": 100,
    "compartment width": 100,
    "initial cell number": 5,
    "start": 1,
    "stop": 50,
    "dt": 0.02,
    "save step interval": 5,
    "k_fm_sec": 0.5,
    "k_fi_sec": 0.5,
    "k_am_on": 0.04,
    "k_am_off": 2e-6,
    "k_im_on": 0.2,
    "k_im_off": 2e-4,
    "k_fm_deg": 0.02,
    "k_fi_deg": 0.02,
    "k_im_deg": 0.02,
    "k_am_deg": 0.02,
    "k_m_diff": 4,
    "k_i_diff": 4,
    "k_im_diff": 3
}

# initializes conditions
params = initialization(infos, anchor=False, num_col=10, ratio=10)

# simulates the system
result = simulation(init_params=params, one_cell=True)



#  stores the simulated system
full_path = "/home/samani/Documents/sim"

if not os.path.exists(full_path):
    os.makedirs(full_path)

full_file_path = os.path.join(full_path, "sim15.h5")

with h5py.File(full_file_path, "w") as file:
    file.create_dataset("fM", data=result[0])
    file.create_dataset("fI", data=result[1])
    file.create_dataset("IM", data=result[2])
    file.create_dataset("AM", data=result[3])
    file.create_dataset("M_cells", data=result[4])
    file.create_dataset("I_cells", data=result[5])
    file.create_dataset("A_cells", data=result[6])





# heatmap visualizations
model1 = HeatMap(
    data_path="/home/samani/Documents/sim/sim15.h5",
    video_directory="/home/samani/Documents/sim/",
    video_name="free_morphogen",
    title="Free Morphogen (GFP)",
    x_label="Number of Cells",
    y_label="Number of Cells",
    c_map="GreenBlack",
    fps=10,
    interval=50,
    writer='ffmpeg',
    color_bar=True,
    norm=False
)

model1.heatmap_animation(key="fM")

model2 = HeatMap(
    data_path="/home/samani/Documents/sim/sim15.h5",
    video_directory="/home/samani/Documents/sim/",
    video_name="free_inhibitor",
    title="Free Inhibitor",
    x_label="Number of Cells",
    y_label="Number of Cells",
    c_map="BlueBlack",
    fps=10,
    interval=50,
    writer='ffmpeg',
    color_bar=True,
    norm=False
)

model2.heatmap_animation(key="fI")

model3 = HeatMap(
    data_path="/home/samani/Documents/sim/sim15.h5",
    video_directory="/home/samani/Documents/sim/",
    video_name="inhibitor_morphogen",
    title="Inhibitor-Morphogen",
    x_label="Number of Cells",
    y_label="Number of Cells",
    c_map="BlueGreenBlack",
    fps=10,
    interval=50,
    writer='ffmpeg',
    color_bar=True,
    norm=False
)

model3.heatmap_animation(key="IM")

model4 = HeatMap(
    data_path="/home/samani/Documents/sim/sim15.h5",
    video_directory="/home/samani/Documents/sim/",
    video_name="anchor_morphogen",
    title="Anchor-Morphogen",
    x_label="Number of Cells",
    y_label="Number of Cells",
    c_map="YellowBlack",
    fps=10,
    interval=50,
    writer='ffmpeg',
    color_bar=True,
    norm=False
)

model4.heatmap_animation(key="AM")







# surface visualizations
surface_animation(
    data_path="/home/samani/Documents/sim/sim15.h5",
    key="fM",
    video_directory="/home/samani/Documents/sim/",
    video_name="free_morphogen",
    title="Free Morphogen (GFP)",
    x_label="Number of Cells",
    y_label="Number of Cells",
    z_label="Concentration",
    colorbar=False,
    c_map="GreenBlack"
)

surface_animation(
    data_path="/home/samani/Documents/sim/sim15.h5",
    key="fI",
    video_directory="/home/samani/Documents/sim/",
    video_name="free_inhibitor",
    title="Free Inhibitor",
    x_label="Number of Cells",
    y_label="Number of Cells",
    z_label="Concentration",
    colorbar=False,
    c_map="BlueBlack"
)

surface_animation(
    data_path="/home/samani/Documents/sim/sim15.h5",
    key="IM",
    video_directory="/home/samani/Documents/sim/",
    video_name="inhibitor_morphogen",
    title="Inhibitor-Morphogen",
    x_label="Number of Cells",
    y_label="Number of Cells",
    z_label="Concentration",
    colorbar=False,
    c_map="BlueGreenBlack"
)

surface_animation(
    data_path="/home/samani/Documents/sim/sim15.h5",
    key="AM",
    video_directory="/home/samani/Documents/sim/",
    video_name="anchor_morphogen",
    title="Anchor-Morphogen",
    x_label="Number of Cells",
    y_label="Number of Cells",
    z_label="Concentration",
    colorbar=False,
    c_map="YellowBlack"
)








# scatter visualizations
scatter_animation(
    data_path="/home/samani/Documents/sim/sim15.h5",
    key="fM",
    video_directory="/home/samani/Documents/sim/",
    video_name="free_morphogen",
    title="Free Morphogen (GFP)",
    x_label="Number of Cells",
    y_label="Number of Cells",
    z_label="Concentration",
    colorbar=False,
    c_map="GreenBlack"
)

scatter_animation(
    data_path="/home/samani/Documents/sim/sim15.h5",
    key="fI",
    video_directory="/home/samani/Documents/sim/",
    video_name="free_inhibitor",
    title="Free Inhibitor",
    x_label="Number of Cells",
    y_label="Number of Cells",
    z_label="Concentration",
    colorbar=False,
    c_map="BlueBlack"
)

scatter_animation(
    data_path="/home/samani/Documents/sim/sim15.h5",
    key="IM",
    video_directory="/home/samani/Documents/sim/",
    video_name="inhibitor_morphogen",
    title="Inhibitor-Morphogen",
    x_label="Number of Cells",
    y_label="Number of Cells",
    z_label="Concentration",
    colorbar=False,
    c_map="BlueGreenBlack"
)

scatter_animation(
    data_path="/home/samani/Documents/sim/sim15.h5",
    key="AM",
    video_directory="/home/samani/Documents/sim/",
    video_name="anchor_morphogen",
    title="Anchor-Morphogen",
    x_label="Number of Cells",
    y_label="Number of Cells",
    z_label="Concentration",
    colorbar=False,
    c_map="YellowBlack"
)




subplot_titles = ["Free GFP", "Inhibitor-GFP", "Free Inhibitor"]  # Titles for each subplot

scatter_animation = ScatterAnimation(
    data_path="/home/samani/Documents/sim/sim/sim15.h5",
    keys=["fM", "IM", "fI"],
    video_directory='videos',
    video_name='multi_scatter',
    subplot_titles=subplot_titles,
    x_label="Number of Cells",
    y_label="Number of Cells",
    z_label="Concentration",
    subplots=(1, 3),
    cmaps=["GreenBlack", "BlueGreenBlack", "BlueBlack"],
    title_size=18,
    label_size=10,
    fps=20,
    interval=50,
    writer='ffmpeg',
    fig_size=(30, 10),  # Adjusted figure size
    colorbar=False,
    pad=0.1,
    grid=False,  # Enable grids
    subplot_size=(7, 6),  # Adjusted subplot size
    plot_margins=(0.1, 0.9, 0.9, 0.1),
    hide_axis=False,
    background_color='white',
    title_color='black',
    xlabel_color='black',
    ylabel_color='black',
    colorbar_axis=False,
    sub_dir="/home/samani/Documents/sim/"
)

scatter_animation.create_video()


subplot_titles = ["Free GFP", "Inhibitor-GFP", "Free Inhibitor"]

surface_animation = SurfaceAnimation(
    data_path="/home/samani/Documents/sim/sim/sim15.h5",
    keys=["fM", "IM", "fI"],
    video_directory='videos',
    video_name='multi_surface',
    subplot_titles=subplot_titles,
    x_label="Number of Cells",
    y_label="Number of Cells",
    z_label="Concentration",
    subplots=(1, 3),
    cmaps=["GreenBlack", "BlueGreenBlack", "BlueBlack"],
    title_size=18,
    label_size=10,
    fps=20,
    interval=50,
    writer='ffmpeg',
    fig_size=(30, 10),   # Adjusted figure size
    colorbar=False,
    pad=0.1,
    grid=True,  # Enable grids
    subplot_size=(7, 7),  # Adjusted subplot size
    plot_margins=(0.1, 0.9, 0.9, 0.1),
    hide_axis=False,
    background_color='white',
    title_color='black',
    xlabel_color='black',
    ylabel_color='black',
    colorbar_axis=False,
    sub_dir="/home/samani/Documents/sim/"
)

surface_animation.create_video()