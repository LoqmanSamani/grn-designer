import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2
import io
import h5py


def surface_animation(data_path, key, video_directory, video_name, frame_width=1500, frame_height=1500, cmap="hot", title=None, x_label=None, y_label=None, z_label=None, title_fontsize=20, label_fontsize=16, tick_fontsize=12, colorbar=False):

    with h5py.File(data_path, 'r') as data:
        z_data = data[key][:]

        x = np.arange(z_data.shape[0])
        y = np.arange(z_data.shape[1])
        X, Y = np.meshgrid(x, y)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = f"{video_directory}/{video_name}.mp4"
        out = cv2.VideoWriter(video_path, fourcc, 10.0, (frame_width, frame_height))

        for i in range(z_data.shape[2]):
            fig = plt.figure(figsize=(frame_width / 100, frame_height / 100))
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(X, Y, z_data[:, :, i], cmap=cmap)

            if title:
                ax.set_title(title, fontsize=title_fontsize)
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
                cbar = fig.colorbar(surf, ax=ax)
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
    print("It's Done")

