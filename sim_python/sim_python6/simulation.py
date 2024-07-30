import numba
import numpy as np
from reactions import *
import copy
import time
import os


def simulate(pop):

    z, y, x = pop[0].shape

    results = np.zeros((len(pop), y, x))  # ndarray to store the sim results
    num_iters = x  # defines number of iterations in each epoch

    for p in pop:

        p = copy.deepcopy(p)

        # simulation info
        max_epoch = p[-1, -1, 0]
        stop = p[-1, -1, 1]
        dt = p[-1, -1, 2]
        num_epochs = int(stop / dt)

        epoch = 0
        while epoch < max_epoch or epoch < num_epochs:

            for i in range(num_iters):
                for j in range(0, z-1, 2):
                    pass








