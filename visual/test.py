from heatmap import *
from multi_heatmap import *
from scatter import *
from surface import *
import h5py





model1 = HeatMap(
    data_path="/home/samani/Documents/sim/sim2.h5",
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

model1.heatmap_animation(key="fM")




model2 = HeatMap(
    data_path="/home/samani/Documents/sim/sim_new.h5",
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

model2.heatmap_animation(key="fI")

model3 = HeatMap(
    data_path="/home/samani/Documents/sim/sim_new.h5",
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

model3.heatmap_animation(key="MC")

model4 = HeatMap(
    data_path="/home/samani/Documents/sim/sim_new.h5",
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
    data_path="/home/samani/Documents/sim/sim_new.h5",
    video_directory="/home/samani/Documents/sim/",
    video_name="anchor_mcherry",
    title="Anchor-mCherry",
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








surface_animation(
    data_path="/home/samani/Documents/sim/sim2.h5",
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



scatter_animation(
    data_path="/home/samani/Documents/sim/sim2.h5",
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





keys = ["fM", "IM", "fI"]

model = HeatMaps(
    data_path="/home/samani/Documents/sim/sim2.h5",
    video_directory="/home/samani/Documents/sim",
    video_name="heatmaps",
    title="Heat Maps",
    x_label="Number of Cells",
    y_label="Number of Cells",
    z_labels=["Free GFP", 'Inhibitor-GFP', "Free Inhibitor"],
    subplots=(1, 3),
    cmaps=["GreenBlack", "BlueGreenBlack", "BlueBlack"],
    title_size=14,
    label_size=12,
    fps=20,
    interval=50,
    writer='ffmpeg',
    fig_size=(30, 10),
    colorbar=True,
    grid=None,
    subplot_size=(8, 8),
    plot_margins=(0.2, 0.1),
    hide_axis=False,
    colorbar_axis=True,
    background_color='white',
    title_color='black',
    xlabel_color='black',
    ylabel_color='black'
)


model.heatmap_animation(keys)


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



keys = ["GFP", "MC", "IMC", "fI", "AMC"]

model = HeatMaps(
    data_path="/home/samani/Documents/sim/sim/sim12.h5",
    video_directory="/home/samani/Documents/sim",
    video_name="h55",
    title="Heat Maps",
    x_label="Number of Cells",
    y_label="Number of Cells",
    z_labels=["Free GFP", 'Free mCherry', "mCherry-Inhibitor", "Free Inhibitor", "Anchor-mCherry"],
    subplots=(1, 5),
    cmaps=["GreenBlack", "RedBlack", "RedPurpleBlack", "PurpleBlack", "YellowBlack"],
    title_size=14,
    label_size=12,
    fps=20,
    interval=50,
    writer='ffmpeg',
    fig_size=(50, 10),
    colorbar=True,
    grid=None,
    subplot_size=(8, 8),
    plot_margins=(0.2, 0.1),
    hide_axis=True,
    colorbar_axis=False,
    background_color='white',
    title_color='black',
    xlabel_color='black',
    ylabel_color='black'
)


model.heatmap_animation(keys)
