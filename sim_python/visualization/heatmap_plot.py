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




















def plot_animation(dataset, title):

    fig, ax = plt.subplots()

    def update(frame):
        # inferno
        # magma
        # hot
        ax.clear()
        ax.set_title(f"{title} - {frame + 1}")
        ax.imshow(dataset[:, :, frame], cmap="hot")

    anim = FuncAnimation(fig, update, frames=dataset.shape[2], interval=200)
    anim.save(f'/home/samani/Documents/sim/{title}.mp4', writer='ffmpeg')
    plt.show()


sim_result = "/path/to/the/data"

with h5py.File(sim_result, 'r') as hdf_file:
    print(hdf_file.keys())

    iM = hdf_file["expected_iM"]
    # iM_norm = (iM - np.min(iM)) / (np.max(iM) - np.min(iM))
    fM = hdf_file["expected_fM"]
    # fM_norm = (fM - np.min(fM)) / (np.max(fM) - np.min(fM))
    ibM = hdf_file["expected_ibM"]
    # ibM_norm = (ibM - np.min(ibM)) / (np.max(ibM) - np.min(ibM))
    bM = hdf_file["expected_bM"]
    # bM_norm = (bM - np.min(bM)) / (np.max(bM) - np.min(bM))

    plot_animation(iM, "iM")
    # plot_animation(iM_norm, "iM_norm")
    plot_animation(fM, "fM")
    # plot_animation(fM_norm, "fM_norm")
    plot_animation(ibM, "ibM")
    # plot_animation(ibM_norm, "ibM_norm")
    plot_animation(bM, "bM")
    # plot_animation(bM_norm, "bM")
