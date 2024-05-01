import matplotlib.pyplot as plt
import numpy as np
import cv2
import io
import h5py


def scatter_animation(data_path, key, video_directory, video_name, frame_width=1000, frame_height=1000):
    # Load data from the HDF5 file
    with h5py.File(data_path, 'r') as data:
        z_data = data[key][:]  # Extract the data corresponding to the given key

        # Generate X, Y, and Z arrays matching the shape of z_data
        x = np.arange(z_data.shape[0])
        y = np.arange(z_data.shape[1])
        X, Y = np.meshgrid(x, y)

        # Create the video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = f"{video_directory}/{video_name}.mp4"
        out = cv2.VideoWriter(video_path, fourcc, 10.0, (frame_width, frame_height))

        # Plot and save each frame
        for i in range(z_data.shape[2]):
            fig = plt.figure(figsize=(frame_width / 100, frame_height / 100))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X.flatten(), Y.flatten(), z_data[:, :, i].flatten(), c=z_data[:, :, i].flatten(), cmap='hot')
            plt.title(f'Frame {i}')
            plt.tight_layout()

            # Convert plot to image
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img = cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), 1)
            buf.close()

            # Write image to video
            out.write(img)
            plt.close()

    # Release the video writer
    out.release()

    print("Video saved to:", video_path)









