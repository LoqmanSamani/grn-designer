import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2
import io
from matplotlib.colors import LinearSegmentedColormap
import h5py
import numpy as np


def scatter_animation(data_path, key, video_directory, video_name, frame_width=1500, frame_height=1500, c_map="RedBlack", title=None, x_label=None, y_label=None, z_label=None, title_fontsize=20, label_fontsize=16, tick_fontsize=12, colorbar=False):

    with h5py.File(data_path, 'r') as data:
        z_data = data[key][:]

        x = np.arange(z_data.shape[0])
        y = np.arange(z_data.shape[1])
        X, Y = np.meshgrid(x, y)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = f"{video_directory}/{video_name}.mp4"
        out = cv2.VideoWriter(video_path, fourcc, 10.0, (frame_width, frame_height))

        def cmap_(c_map):

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
                cmap = LinearSegmentedColormap.from_list("PurpleBlack", blue_green_)
            elif c_map == "RedPurpleBlack":
                cmap = LinearSegmentedColormap.from_list("PurpleBlack", red_purple_)

            else:
                cmap = LinearSegmentedColormap.from_list("RedBlack", red_)

            return cmap

        for i in range(z_data.shape[2]):
            fig = plt.figure(figsize=(frame_width / 100, frame_height / 100))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(X.flatten(), Y.flatten(), z_data[:, :, i].flatten(), c=z_data[:, :, i].flatten(), cmap=cmap_(c_map=c_map))

            if title:
                ax.set_title(f"{title} - {i}/{z_data.shape[2]-1}", fontsize=title_fontsize)
            if x_label:
                ax.set_xlabel(x_label, fontsize=label_fontsize)
            if y_label:
                ax.set_ylabel(y_label, fontsize=label_fontsize)
            if z_label:
                ax.set_zlabel(z_label, fontsize=label_fontsize)

            ax.tick_params(axis='x', labelsize=tick_fontsize)
            ax.tick_params(axis='y', labelsize=tick_fontsize)
            ax.tick_params(axis='z', labelsize=tick_fontsize)

            if colorbar:

                cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Intensity', fontsize=label_fontsize)
                cbar.ax.tick_params(labelsize=tick_fontsize)

            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img = cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), 1)
            buf.close()

            out.write(img)
            plt.close()

    out.release()

    print("It's Done!")








