from diffusion import *
from reactions import *
from numba import jit


@jit
def simulation(sp1, sp2, sp1_cells, sp2_cells, params, dt, sim_start, sim_stop, epochs, target_shape):

    sp1_sec = params[0, 0]
    sp2_sec = params[0, 1]
    sp1_diff = params[0, 4]
    sp2_diff = params[0, 5]
    sp1_deg = params[0, 6]
    sp2_deg = params[0, 7]

    time_ = sim_start
    epoch = 1

    while time_ <= sim_stop or epoch <= epochs:

        for i in range(target_shape[1]):
            for j in range(target_shape[0]):

                # sp1 production
                sp1[i, j] = production(
                    pre_con=sp1[i, j],
                    num_cell=sp1_cells[i, j],
                    pk=sp1_sec,
                    dt=dt
                )
                # sp2 production
                sp2[i, j] = production1(
                    pre_con=sp2[i, j],
                    num_cell=sp2_cells[i, j],
                    fm=sp1[i, j],
                    pk=sp2_sec,
                    dt=dt
                )

                # degradation
                sp1[i, j] = degradation(
                    pre_con=sp1[i, j],
                    dk=sp1_deg,
                    dt=dt
                )
                sp2[i, j] = degradation(
                    pre_con=sp2[i, j],
                    dk=sp2_deg,
                    dt=dt
                )

                # diffusion

                sp1[i, j] = diffusion(
                    specie=sp1,
                    length=i,
                    width=j,
                    k_diff=sp1_diff,
                    dt=dt,
                    compartment_length=target_shape[1],
                    compartment_width=target_shape[0]
                )
                sp2[i, j] = diffusion(
                    specie=sp2,
                    length=i,
                    width=j,
                    k_diff=sp2_diff,
                    dt=dt,
                    compartment_length=target_shape[1],
                    compartment_width=target_shape[0]
                )

        time_ += dt
        epoch += 1

    return sp2



