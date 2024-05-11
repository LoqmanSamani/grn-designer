from heatmap_plot import *
from heatmap_plots import *
from scatter_plot import *
from surface_plot import *

"""
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


model2 = HeatMap(
    data_path="/home/samani/Documents/sim/sim.h5",
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
    data_path="/home/samani/Documents/sim/sim.h5",
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
    data_path="/home/samani/Documents/sim/sim.h5",
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


surface_animation(
    data_path="/home/samani/Documents/sim/sim.h5",
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
    data_path="/home/samani/Documents/sim/sim.h5",
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
    data_path="/home/samani/Documents/sim/sim.h5",
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
    data_path="/home/samani/Documents/sim/sim.h5",
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
"""



scatter_animation(
    data_path="/home/samani/Documents/sim/sim.h5",
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
    data_path="/home/samani/Documents/sim/sim.h5",
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
    data_path="/home/samani/Documents/sim/sim.h5",
    key="IMC",
    video_directory="/home/samani/Documents/sim/",
    video_name="inhibitor_morphogen",
    title="Inhibitor-mCherry",
    x_label="Number of Cells",
    y_label="Number of Cells",
    z_label="Concentration",
    colorbar=False,
    c_map="RedPurpleBlack"
)

scatter_animation(
    data_path="/home/samani/Documents/sim/sim.h5",
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




