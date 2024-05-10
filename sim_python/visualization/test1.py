from heatmap_plot import *
from heatmap_plots import *
from scatter_plot import *
from surface_plot import *


model1 = HeatMap(
    data_path="/home/samani/Documents/sim/sim.h5",
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




"""
model2 = HeatMap(
    data_path="/home/samani/Documents/sim/sim16.h5",
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
    data_path="/home/samani/Documents/sim/sim16.h5",
    video_directory="/home/samani/Documents/sim/",
    video_name="free_inhibitor",
    title="Free Inhibitor",
    x_label="Number of Cells",
    y_label="Number of Cells",
    c_map="YellowBlack",
    fps=10,
    interval=50,
    writer='ffmpeg',
    color_bar=True,
    norm=False
)

model3.heatmap_animation(key="fI")

model4 = HeatMap(
    data_path="/home/samani/Documents/sim/sim16.h5",
    video_directory="/home/samani/Documents/sim/",
    video_name="inhibitor_mCherry",
    title="Inhibitor-mCherry",
    x_label="Number of Cells",
    y_label="Number of Cells",
    c_map="PurpleBlack",
    fps=10,
    interval=50,
    writer='ffmpeg',
    color_bar=True,
    norm=False
)

model4.heatmap_animation(key="IMC")



surface_animation(
    data_path="/home/samani/Documents/sim/sim.h5",
    key="MC",
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
"""
