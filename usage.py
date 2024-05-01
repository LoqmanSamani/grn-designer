from diffusion import *
from reactions import *
from simulation import *
from initialization import *
from heatmap_plot import *
from surface_plot import *
from scatter_plot import *

import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



# parameters
infos = {

    "growth rate": 0,
    "max cell number": 1000000,
    "compartment length": 1000,
    "compartment width": 1000,
    "start": 1,
    "stop": 50,
    "dt": 0.01,
    "cell seed": 1000000,
    "save step interval": 20,
    "k_fm_sec": 0.5,
    "k_im_sec": 0.4,
    "k_fm_bind": 0.2,
    "k_fm_off": 0.3,
    "k_im_bind": 0.2,
    "k_im_off": 0.23,
    "k_fm_deg": 0.001,
    "k_im_deg": 0.001,
    "k_bm_deg": 0.001,
    "k_ibm_deg": 0.001,
    "d_free": 0.6,
    "d_i": 0.5

}



# Simulate the system
params = initialization(infos)

result = simulation(
    init_params=params,
)




# visualize the results


# heat map animations

heatmap_animation(
        data_path="/path/to/the/results.h5",  # path to the simulation results
        key="expected_fM",  # matrix to simulate
        video_directory="/where/the/video/should/be/stored",  # store directory
        video_name="fm_surface",  # file name
)

heatmap_animation(
        data_path="/path/to/the/results.h5",  # path to the simulation results
        key="expected_iM",  # matrix to simulate
        video_directory="/where/the/video/should/be/stored",  # store directory
        video_name="im_surface",  # file name
)

heatmap_animation(
        data_path="/path/to/the/results.h5",  # path to the simulation results
        key="expected_ibM",  # matrix to simulate
        video_directory="/where/the/video/should/be/stored",  # store directory
        video_name="ibm_surface",  # file name
)

heatmap_animation(
        data_path="/path/to/the/results.h5",  # path to the simulation results
        key="expected_bM",  # matrix to simulate
        video_directory="/where/the/video/should/be/stored",  # store directory
        video_name="bm_surface",  # file name
)






# surface animations

surface_animation(
        data_path="/path/to/the/results.h5",  # path to the simulation results
        key="expected_fM",  # matrix to simulate
        video_directory="/where/the/video/should/be/stored",  # store directory
        video_name="fm_surface",  # file name
        frame_width=1000,  # quality of the video
        frame_height=1000  # quality of the video
)

surface_animation(
        data_path="/path/to/the/results.h5",
        key="expected_iM",
        video_directory="/where/the/video/should/be/stored",
        video_name="im_surface",
        frame_width=1000,
        frame_height=1000
)

surface_animation(
        data_path="/path/to/the/results.h5",
        key="expected_bM",
        video_directory="/where/the/video/should/be/stored",
        video_name="bm_surface",
        frame_width=1000,
        frame_height=1000
)

surface_animation(
        data_path="/path/to/the/results.h5",
        key="expected_ibM",
        video_directory="/where/the/video/should/be/stored",
        video_name="ibm_surface",
        frame_width=1000,
        frame_height=1000
)


# scatter animations

scatter_animation(
        data_path="/path/to/the/results.h5",  # path to the simulation results
        key="expected_fM",  # matrix to simulate
        video_directory="/where/the/video/should/be/stored",  # store directory
        video_name="fm_surface",  # file name
        frame_width=1000,  # quality of the video
        frame_height=1000  # quality of the video
)

scatter_animation(
        data_path="/path/to/the/results.h5",
        key="expected_iM",
        video_directory="/where/the/video/should/be/stored",
        video_name="im_surface",
        frame_width=1000,
        frame_height=1000
)

scatter_animation(
        data_path="/path/to/the/results.h5",
        key="expected_bM",
        video_directory="/where/the/video/should/be/stored",
        video_name="bm_surface",
        frame_width=1000,
        frame_height=1000
)

scatter_animation(
        data_path="/path/to/the/results.h5",
        key="expected_ibM",
        video_directory="/where/the/video/should/be/stored",
        video_name="ibm_surface",
        frame_width=1000,
        frame_height=1000
)







