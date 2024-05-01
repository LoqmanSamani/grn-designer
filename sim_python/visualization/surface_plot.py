import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2
import io
import h5py


def surface_animation(data_path, key, video_directory, video_name, frame_width=1000, frame_height=1000):

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
            ax.plot_surface(X, Y, z_data[:, :, i], cmap='hot')
            plt.title(f'Frame {i}')
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img = cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), 1)
            buf.close()

            out.write(img)
            plt.close()

    out.release()
    print("Video saved to:", video_path)

