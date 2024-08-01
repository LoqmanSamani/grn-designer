from reactions import *
from diffusion import *


@jit(nopython=True)
def population_simulation(population):

    results = None

    if isinstance(population, np.ndarray):

        m, z, y, x = population.shape  # m: number of individual in the population
        results = np.zeros((m, y, x))  # ndarray to store the simulation results
        num_iters = x  # defines number of iterations in each epoch
        num_species = int(population[0, -1, -1, 0])  # number os species present in the system
        num_pairs = int(population[0, -1, -1, 1])  # number of pairs present in the system
        max_epoch = int(population[0, -1, -1, 2])  # maximum number of epochs
        stop = population[0, -1, -1, 3]  # simulation duration
        time_step = population[0, -1, -1, 2]  # time step
        num_epochs = int(stop / time_step)  # number of possible epochs
        pair_start = int(num_species * 2)
        pair_stop = int(pair_start + (num_pairs * 2))

        epoch = 0
        while epoch < max_epoch or epoch < num_epochs:

            for i in range(num_iters):

                # species production
                for j in range(0, num_species, 2):
                    population[:, j, :, i] = apply_component_production(
                        initial_concentration=population[:, j, :, i],
                        production_pattern=population[:, j+1, :, i],
                        production_rates=population[:, -1, j, 0],
                        time_step=time_step
                    )

                # species collision
                for j in range(pair_start, pair_stop, 2):
                    (population[:, population[:, j+1, 0, 0], :, i],
                     population[:, population[:, j+1, 0, 1], :, i],
                     population[:, j, :, i]) = apply_species_collision(
                        species1=population[0, int(population[0, j+1, 0, 0]), :, i],
                        species2=population[0, int(population[0, j+1, 0, 1]), :, i],
                        complex_=population[:, j, :, i],
                        collision_rates=population[:, int(population[0, j+1, 1, 0]), 1, 0],
                        time_step=time_step
                    )

                # complex dissociation
                for j in range(pair_start, pair_stop, 2):
                    (population[:, population[:, j + 1, 0, 0], :, i],
                     population[:, population[:, j + 1, 0, 1], :, i],
                     population[:, j, :, i]) = apply_complex_dissociation(
                        species1=population[0, int(population[:, j + 1, 0, 1]), :, i],
                        species2=population[0, int(population[:, j + 1, 0, 1]), :, i],
                        complex_=population[:, j, :, i],
                        dissociation_rates=population[:, j+1, 1, 1],
                        time_step=time_step
                    )

                # species degradation
                for j in range(0, num_species, 2):
                    population[:, j, :, i] = apply_component_degradation(
                        initial_concentration=population[:, j, :, i],
                        degradation_rates=population[:, -1, j, 1],
                        time_step=time_step
                    )

                # complex degradation
                for j in range(pair_start, num_pairs, 2):
                    population[:, j, :, i] = apply_component_degradation(
                        initial_concentration=population[:, j, :, i],
                        degradation_rates=population[:, j+1, 1, 2],
                        time_step=time_step
                    )

                # species diffusion
                for j in range(0, num_species, 2):
                    population[:, j, :, i] = apply_diffusion(
                        current_concentration=population[:, j, :, i],
                        compartment=population[:, j, :, :],
                        column_position=i,
                        diffusion_rates=population[:, -1, j, 2],
                        time_step=time_step
                    )

                # complex diffusion
                for j in range(pair_start, num_pairs, 2):
                    population[:, j, :, i] = apply_diffusion(
                        current_concentration=population[:, j, :, i],
                        compartment=population[:, j, :, :],
                        column_position=i,
                        diffusion_rates=population[:, j+1, 1, 3],
                        time_step=time_step
                    )

            epoch += 1

        results = population[:, 0, :, :]

    elif isinstance(population, list):

        # TODO: implement the simulation for the case in which populaion is a list of individuals
        pass


    else:
        raise ValueError("Unsupported type for population. Must be a numpy array or a list.")

    return results













