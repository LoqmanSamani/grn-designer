from heatmap_plot import *
from heatmap_plots import *
from scatter_plot import *
from surface_plot import *



model1 = HeatMap(
    data_path="/home/samani/Documents/sim/sim1.h5",
    video_directory="/home/samani/Documents/sim/",
    video_name="test",
    title="first one",
    x_label="x",
    y_label="y",
    cmap="hot",
    fps=10,
    interval=50,
    writer='ffmpeg',
    color_bar=True
)

model1.heatmap_animation(key="fM")



model2 = HeatMaps(
    data_path="/home/samani/Documents/sim/sim1.h5",
    video_directory="/home/samani/Documents/sim/",
    video_name="test",
    title="first one",
    x_label="x",
    y_label="y",
    z_labels=["z1", "z2", "z3", "z4"],
    cmaps=["Reds", "Blues", "Greens", "hot"],
    fps=10,
    interval=50,
    writer='ffmpeg',
    fig_size=(20, 20)
)

model2.heatmap_animation(keys=["fM", "fI", "IM", "AM"])




scatter_animation(
    data_path="/home/samani/Documents/sim/sim1.h5",
    key="fI",
    video_directory="/home/samani/Documents/sim/",
    video_name="nn",
    title="Test",
    x_label="X",
    y_label="Y",
    z_label="Z",
    colorbar=False
)


surface_animation(
    data_path="/home/samani/Documents/sim/sim1.h5",
    key="fI",
    video_directory="/home/samani/Documents/sim/",
    video_name="nn1",
    title="Test",
    x_label="X",
    y_label="Y",
    z_label="Z",
    colorbar=False
)
