import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class HeatMaps:
    def __init__(self, data_path, video_directory, video_name, title, x_label, y_label, z_labels, cmaps=None, title_size=14, label_size=12, cmap="hot", fps=10, interval=50, writer='ffmpeg', fig_size=(12, 12), colorbar=True, grid=None):

        self.data_path = data_path
        self.video_directory = video_directory
        self.video_name = video_name
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.z_labels = z_labels
        self.title_size = title_size
        self.cmaps = cmaps
        self.label_size = label_size
        self.cmap = cmap
        self.fps = fps
        self.interval = interval
        self.writer = writer
        self.fig_size = fig_size
        self.colorbar = colorbar
        self.grid = grid

    def load_data(self, data_path, keys):

        file = h5py.File(data_path, "r")
        datas = [np.array(file[key]) for key in keys]

        return datas

    def plot_animation(self, datas):
        fig, axs = plt.subplots(nrows=len(datas), figsize=self.fig_size)

        if len(datas) == 1:  # Handle the case where there is only one subplot
            axs = [axs]

        colorbars = []

        def update(frame):
            for ax, data, z_label, cmap in zip(axs, datas, self.z_labels, self.cmaps):
                ax.clear()
                ax.set_title(f"{self.title} - Frame {frame + 1}", fontsize=self.title_size)
                img = ax.imshow(data[:, :, frame], cmap=cmap)
                ax.set_xlabel(self.x_label, fontsize=self.label_size)
                ax.set_ylabel(self.y_label, fontsize=self.label_size)
                if self.grid:
                    ax.grid(True)

                if self.colorbar and ax not in colorbars:
                    cbar = fig.colorbar(img, ax=ax)
                    cbar.ax.tick_params(labelsize=self.label_size)
                    colorbars.append(ax)  # Add the subplot to the list of subplots with colorbars
                if len(datas) == 1:  # Add z label only when there is one subplot
                    ax.set_xlabel(self.x_label, fontsize=self.label_size)
                    ax.set_ylabel(self.y_label, fontsize=self.label_size)
                    cbar.set_label(z_label, fontsize=self.label_size)

        anim = FuncAnimation(fig, update, frames=datas[0].shape[2], interval=self.interval)
        anim.save(f'{self.video_directory}{self.video_name}.mp4', writer=self.writer, fps=self.fps)

    def heatmap_animation(self, keys):

        datas = self.load_data(self.data_path, keys)
        self.plot_animation(datas)
        print("It's Done!")







