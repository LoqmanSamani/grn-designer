import os
import subprocess
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D


class SurfaceAnimation:
    def __init__(self, data_path, keys, video_directory, video_name, subplot_titles, x_label, y_label, z_label, subplots,
                 cmaps, title_size=14, label_size=12, fps=20, interval=50, writer='ffmpeg', fig_size=(28, 16),
                 colorbar=False, pad=0.1, grid=True, subplot_size=None, plot_margins=None, hide_axis=False,
                 background_color='white', title_color='black', xlabel_color='black', ylabel_color='black',
                 colorbar_axis=True, sub_dir=None):
        self.data_path = data_path
        self.keys = keys
        self.video_directory = video_directory
        self.video_name = video_name
        self.subplot_titles = subplot_titles  # New parameter for subplot titles
        self.x_label = x_label
        self.y_label = y_label
        self.z_label = z_label
        self.subplots = subplots
        self.cmaps = cmaps
        self.title_size = title_size
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
        self.sub_dir = sub_dir

        self.load_data()

        if self.sub_dir:
            self.video_directory = os.path.join(self.video_directory, self.sub_dir)

        self.frames_dir = os.path.join(self.video_directory, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)

    def load_data(self):
        with h5py.File(self.data_path, 'r') as data:
            self.z_datas = [data[key][:] for key in self.keys]
        self.x = np.arange(self.z_datas[0].shape[0])
        self.y = np.arange(self.z_datas[0].shape[1])
        self.X, self.Y = np.meshgrid(self.x, self.y)

    def cmap_(self, index):
        cmaps_dict = {
            "RedBlack": [(0, 0, 0), (1, 0, 0)],
            "GreenBlack": [(0, 0, 0), (0, 1, 0)],
            "BlueBlack": [(0, 0, 0), (0, 0, 1)],
            "YellowBlack": [(0, 0, 0), (1, 1, 0)],
            "PurpleBlack": [(0, 0, 0), (1, 0, 1)],
            "BlueGreenBlack": [(0, 0, 0), (0, 1, 1)],
            "RedPurpleBlack": [(0, 0, 0), (0.5, 0, 0.5)]
        }

        cmap_key = self.cmaps[index] if index < len(self.cmaps) else "RedBlack"
        colors = cmaps_dict.get(cmap_key, cmaps_dict["RedBlack"])
        return LinearSegmentedColormap.from_list(cmap_key, colors)

    def generate_frame(self, i):
        fig = plt.figure(figsize=self.fig_size)
        fig.patch.set_facecolor(self.background_color)
        n_plots = len(self.keys)

        n_rows = self.subplots[0]
        n_cols = self.subplots[1]

        for j, z_data in enumerate(self.z_datas):
            ax = fig.add_subplot(n_rows, n_cols, j + 1, projection='3d')

            # Create a surface plot
            surf = ax.plot_surface(self.X, self.Y, z_data[:, :, i], cmap=self.cmap_(j), linewidth=0, antialiased=False)

            if self.subplot_titles and len(self.subplot_titles) > j:
                ax.set_title(self.subplot_titles[j], fontsize=self.title_size, color=self.title_color)
            if self.x_label:
                ax.set_xlabel(self.x_label, fontsize=self.label_size, color=self.xlabel_color)
            if self.y_label:
                ax.set_ylabel(self.y_label, fontsize=self.label_size, color=self.ylabel_color)
            if self.z_label:
                ax.set_zlabel(self.z_label, fontsize=self.label_size, color=self.ylabel_color)

            ax.tick_params(axis='x', labelsize=self.label_size, colors=self.xlabel_color)
            ax.tick_params(axis='y', labelsize=self.label_size, colors=self.ylabel_color)
            ax.tick_params(axis='z', labelsize=self.label_size, colors=self.ylabel_color)

            if self.colorbar:
                cbar = fig.colorbar(surf, ax=ax, fraction=0.046, pad=self.pad)
                cbar.set_label('Intensity', fontsize=self.label_size, color=self.xlabel_color)
                cbar.ax.tick_params(labelsize=self.label_size, colors=self.xlabel_color)

            if self.grid:
                ax.grid(True)

            if self.hide_axis:
                ax.axis('off')

        plt.subplots_adjust(left=self.plot_margins[0], right=self.plot_margins[1],
                            top=self.plot_margins[2], bottom=self.plot_margins[3])
        frame_path = os.path.join(self.frames_dir, f"frame_{i:04d}.png")
        plt.savefig(frame_path)
        plt.close()
        return frame_path

    def create_video(self):
        for i in range(self.z_datas[0].shape[2]):
            self.generate_frame(i)
        video_path = os.path.join(self.video_directory, f"{self.video_name}.mp4")
        cmd = [
            'ffmpeg', '-y', '-framerate', str(self.fps), '-i', os.path.join(self.frames_dir, 'frame_%04d.png'),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', video_path
        ]
        subprocess.run(cmd)
        print("It's Done!")



