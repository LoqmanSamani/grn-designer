import h5py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def heatmap_animation(data_path, key, video_directory, video_name):

    def plot_animation(dataset, key, video_name, video_directory):
        fig, ax = plt.subplots()

        def update(frame):
            # inferno
            # magma
            # hot
            ax.clear()
            ax.set_title(f"{video_name} - {frame + 1}")
            ax.imshow(dataset[:, :, frame], cmap="hot")

        anim = FuncAnimation(fig, update, frames=dataset.shape[2], interval=200)
        anim.save(f'{video_directory}{video_name}.mp4', writer='ffmpeg')
        plt.show()

    with h5py.File(data_path, 'r') as data:

        plot_animation(data[key], key, video_name, video_directory)




