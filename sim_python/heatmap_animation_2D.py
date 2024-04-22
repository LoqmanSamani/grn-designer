import h5py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from copy import copy


class Heatmap2D:
    def __init__(self, path):

        self.path = path
        self.data = {}

    def open_data(self):

        data = {}
        hdf_file = h5py.File(self.path, mode='r+')
        keys = hdf_file.keys()

        for key in keys:
            data[key] = hdf_file[key]

        self.data = data

    def plot_animation(self, dataset, save_path, name, title="Simulation Hit-Map", x_label="X", y_label="Y", cmap="coolwarm"):

        self.open_data()
        fig, ax = plt.subplots()

        def update(frame):
            ax.clear()
            ax.set_title(f"{title} - {frame+1}")
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.imshow(self.data[dataset][:, :, frame], cmap=cmap)

        anim = FuncAnimation(fig, update, frames=self.data[dataset].shape[2], interval=200)
        anim.save(f"{save_path}{name}.mp4", writer='ffmpeg')


obj = Heatmap2D(path="/path/to/simulation/data")

obj.plot_animation(
    dataset="expected_iM",
    save_path="/path/to/the/save/directory",
    name="iM",
    title="iM",
    x_label="X",
    y_label="Y",
    cmap="coolwarm"
)

