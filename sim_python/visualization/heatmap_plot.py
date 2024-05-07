import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
import h5py
import numpy as np


class HeatMap:
    def __init__(self, data_path, video_directory, video_name, title, x_label, y_label, norm=None, which_norm="N",
                 c_map="GreenBlack", fps=10, interval=50, writer='ffmpeg', fig_size=(12, 12), label_size=16,
                 title_size=18, color_bar=False, grid=False, aspect=20, pad=0.1):

        self.data_path = data_path
        self.video_directory = video_directory
        self.video_name = video_name
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.norm = norm
        self.which_norm = which_norm
        self.c_map = c_map
        self.cmap = self.cmap_()
        self.fps = fps
        self.interval = interval
        self.writer = writer
        self.fig_size = fig_size
        self.label_size = label_size
        self.title_size = title_size
        self.color_bar = color_bar
        self.grid = grid
        self.aspect = aspect
        self.pad = pad
        self.data = None

    def cmap_(self):

        red_ = [(0, 0, 0), (1, 0, 0)]  # red
        green_ = [(0, 0, 0), (0, 1, 0)]    # green
        blue_ = [(0, 0, 0), (0, 0, 1)]   # blue
        yellow_ = [(0, 0, 0), (1, 1, 0)]  # yellow
        purple_ = [(0, 0, 0), (1, 0, 1)]  # purple
        blue_green_ = [(0, 0, 0), (0, 1, 1)]  # blue-green

        if self.c_map == "RedBlack":
            cmap = LinearSegmentedColormap.from_list("RedBlack", red_)
        elif self.c_map == "GreenBlack":
            cmap = LinearSegmentedColormap.from_list("GreenBlack", green_)
        elif self.c_map == "BlueBlack":
            cmap = LinearSegmentedColormap.from_list("BlueBlack", blue_)
        elif self.c_map == "YellowBlack":
            cmap = LinearSegmentedColormap.from_list("YellowBlack", yellow_)
        elif self.c_map == "PurpleBlack":
            cmap = LinearSegmentedColormap.from_list("PurpleBlack", purple_)
        elif self.c_map == "BlueGreenBlack":
            cmap = LinearSegmentedColormap.from_list("PurpleBlack", blue_green_)

        else:
            cmap = LinearSegmentedColormap.from_list("RedBlack", red_)


        return cmap

    def load_data(self, data_path, key):
        file = h5py.File(data_path, "r")
        data = np.array(file[key])
        if self.norm:
            if self.which_norm == "N":
                data = self.n_norm(data)
            else:
                data = self.z_norm(data)
        return data

    def plot_animation(self, data):

        fig, ax = plt.subplots(figsize=self.fig_size)
        ax.set_xlabel(self.x_label, fontsize=self.label_size)
        ax.set_ylabel(self.y_label, fontsize=self.label_size)
        if self.grid:
            ax.grid(True)

        img = ax.imshow(data[:, :, 0], cmap=self.cmap)
        if self.color_bar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="8%", pad=self.pad)
            plt.colorbar(img, cax=cax)

        def update(frame):
            ax.set_title(f"{self.title} - {frame + 1}/{data.shape[2]-1}", fontsize=self.title_size)
            img.set_array(data[:, :, frame])

        anim = FuncAnimation(fig, update, frames=data.shape[2], interval=self.interval)
        anim.save(f'{self.video_directory}{self.video_name}.mp4', writer=self.writer, fps=self.fps)

    def heatmap_animation(self, key):
        data = self.load_data(self.data_path, key)
        self.data = data
        self.plot_animation(data)
        print("It's Done!")

    def n_norm(self, array):
        min_ = np.mean(array, axis=2, keepdims=True)
        max_ = np.max(array, axis=2, keepdims=True)
        normalized_array = (array - min_) / ((max_ - min_) + 1e-8)
        return normalized_array

    def z_norm(self, array):
        mean = np.mean(array, axis=2, keepdims=True)
        std_dev = np.std(array, axis=2, keepdims=True)
        normalized_array = (array - mean) / (std_dev + 1e-8)
        return normalized_array
