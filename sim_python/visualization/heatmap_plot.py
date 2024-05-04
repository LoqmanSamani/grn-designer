import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import h5py
import numpy as np


class HeatMap:
    def __init__(self, data_path, video_directory, video_name, title, x_label, y_label, cmap="hot", fps=10, interval=50, writer='ffmpeg', fig_size=(12, 12), label_size=14, title_size=18, color_bar=False, grid=False, aspect=20):

        self.data_path = data_path
        self.video_directory = video_directory
        self.video_name = video_name
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.cmap = cmap
        self.fps = fps
        self.interval = interval
        self.writer = writer
        self.fig_size = fig_size
        self.label_size = label_size
        self.title_size = title_size
        self.color_bar = color_bar
        self.grid = grid
        self.aspect = aspect  # aspect ratio to control length of colorbar

    def load_data(self, data_path, key):

        file = h5py.File(data_path, "r")
        data = np.array(file[key])
        return data

    def plot_animation(self, data):
        fig, ax = plt.subplots(figsize=self.fig_size)
        ax.set_xlabel(self.x_label, fontsize=self.label_size)
        ax.set_ylabel(self.y_label, fontsize=self.label_size)
        if self.grid:
            ax.grid(True)

        img = ax.imshow(data[:, :, 0], cmap=self.cmap)
        if self.color_bar:
            cbar = fig.colorbar(img, ax=ax, aspect=self.aspect)
            cbar.ax.tick_params(labelsize=self.label_size)
            ax.set_title(self.title, fontsize=self.label_size)

        def update(frame):
            ax.set_title(f"{self.title} - {frame + 1}", fontsize=self.title_size)
            img.set_array(data[:, :, frame])

        anim = FuncAnimation(fig, update, frames=data.shape[2], interval=self.interval)
        anim.save(f'{self.video_directory}{self.video_name}.mp4', writer=self.writer, fps=self.fps)

    def heatmap_animation(self, key):

        data = self.load_data(self.data_path, key)
        self.plot_animation(data)

