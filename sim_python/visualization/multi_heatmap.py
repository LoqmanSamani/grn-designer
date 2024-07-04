import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable


class HeatMaps:
    def __init__(self, data_path, video_directory, video_name, title, x_label, y_label, z_labels, subplots, cmaps,
                 title_size=14, label_size=12, fps=20, interval=50, writer='ffmpeg', fig_size=(28, 16), colorbar=True,
                 pad=0.1, grid=None, subplot_size=None, plot_margins=None, hide_axis=False, background_color='white',
                 title_color='black', xlabel_color='black', ylabel_color='black', colorbar_axis=True):

        self.data_path = data_path
        self.video_directory = video_directory
        self.video_name = video_name
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.z_labels = z_labels
        self.subplots = subplots
        self.title_size = title_size
        self.cmaps = [self.cmap_(cmap) for cmap in cmaps]
        self.label_size = label_size
        self.fps = fps
        self.interval = interval
        self.writer = writer
        self.fig_size = fig_size
        self.colorbar = colorbar
        self.pad = pad
        self.grid = grid
        self.subplot_size = subplot_size
        self.plot_margins = plot_margins
        self.hide_axis = hide_axis
        self.background_color = background_color
        self.title_color = title_color
        self.xlabel_color = xlabel_color
        self.ylabel_color = ylabel_color
        self.colorbar_axis = colorbar_axis

    def load_data(self, data_path, keys):
        file = h5py.File(data_path, "r")
        datas = [np.array(file[key]) for key in keys]
        return datas

    def cmap_(self, c_map):
        red_ = [(0, 0, 0), (1, 0, 0)]  # red
        green_ = [(0, 0, 0), (0, 1, 0)]  # green
        blue_ = [(0, 0, 0), (0, 0, 1)]  # blue
        yellow_ = [(0, 0, 0), (1, 1, 0)]  # yellow
        purple_ = [(0, 0, 0), (1, 0, 1)]  # purple
        blue_green_ = [(0, 0, 0), (0, 1, 1)]  # blue-green
        red_purple_ = [(0, 0, 0), ((red_[1][0] + purple_[1][0]) / 2,
                                   (red_[1][1] + purple_[1][1]) / 2,
                                   (red_[1][2] + purple_[1][2]) / 2)]

        if c_map == "RedBlack":
            cmap = LinearSegmentedColormap.from_list("RedBlack", red_)
        elif c_map == "GreenBlack":
            cmap = LinearSegmentedColormap.from_list("GreenBlack", green_)
        elif c_map == "BlueBlack":
            cmap = LinearSegmentedColormap.from_list("BlueBlack", blue_)
        elif c_map == "YellowBlack":
            cmap = LinearSegmentedColormap.from_list("YellowBlack", yellow_)
        elif c_map == "PurpleBlack":
            cmap = LinearSegmentedColormap.from_list("PurpleBlack", purple_)
        elif c_map == "BlueGreenBlack":
            cmap = LinearSegmentedColormap.from_list("BlueGreenBlack", blue_green_)
        elif c_map == "RedPurpleBlack":
            cmap = LinearSegmentedColormap.from_list("RedPurpleBlack", red_purple_)
        else:
            cmap = LinearSegmentedColormap.from_list("RedBlack", red_)

        return cmap

    def plot_animation(self, datas):
        fig, axs = plt.subplots(nrows=self.subplots[0], ncols=self.subplots[1], figsize=self.fig_size,
                                gridspec_kw={'width_ratios': [1] * self.subplots[1],
                                             'height_ratios': [1] * self.subplots[0]},
                                facecolor=self.background_color)

        axs = axs.flatten()
        plt.subplots_adjust(wspace=self.plot_margins[0], hspace=self.plot_margins[1])

        if len(datas) == 1:
            axs = [axs]

        colorbars = []
        max_frames = datas[0].shape[2]
        for ax, data, z_label, cmap in zip(axs, datas, self.z_labels, self.cmaps):
            img = ax.imshow(data[:, :, 0], cmap=cmap)
            ax.set_title(f"{z_label}", fontsize=self.title_size, color=self.title_color)
            if self.hide_axis:
                ax.axis('off')
            else:
                ax.set_xlabel(self.x_label, fontsize=self.label_size, color=self.xlabel_color)
                ax.set_ylabel(self.y_label, fontsize=self.label_size, color=self.ylabel_color)
                if self.grid:
                    ax.grid(True)

            if self.colorbar:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="8%", pad=self.pad)
                plt.colorbar(img, cax=cax)
                if not self.colorbar_axis:
                    cax.axis('off')
                colorbars.append(cax)

        def update(frame):
            for i, (ax, data, z_label, cmap) in enumerate(zip(axs, datas, self.z_labels, self.cmaps)):
                ax.clear()
                img = ax.imshow(data[:, :, frame], cmap=cmap)
                ax.set_title(f"{z_label}", fontsize=self.title_size, color=self.title_color)
                if self.hide_axis:
                    ax.axis('off')
                else:
                    ax.set_xlabel(self.x_label, fontsize=self.label_size, color=self.xlabel_color)
                    ax.set_ylabel(self.y_label, fontsize=self.label_size, color=self.ylabel_color)
                    if self.grid:
                        ax.grid(True)
                if self.colorbar and not self.colorbar_axis:
                    colorbars[i].axis('off')

        anim = FuncAnimation(fig, update, frames=max_frames, interval=self.interval)
        anim.save(f'{self.video_directory}/{self.video_name}.mp4', writer=self.writer, fps=self.fps, dpi=100)
        plt.close(fig)

    def heatmap_animation(self, keys):
        datas = self.load_data(self.data_path, keys)
        self.plot_animation(datas)
        print("It's Done!")








