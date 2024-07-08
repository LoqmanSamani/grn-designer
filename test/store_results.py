import os
import h5py




def save_results(result_path, file_name, elite_chromosome, species, best_fitness, simulation_duration, population):
    """
    Save simulation results to an HDF5 file.

    Args:
        - result_path (str): The directory where the results file will be saved.
        - file_name (str): The name of the results file (without extension).
        - elite_chromosome (list or np.ndarray): The elite chromosome of the last generation.
        - species (list): A list containing the name of species corresponding to each array in the elite chromosome.
        - best_fitness (list or np.ndarray): The best fitness values over the generations.
        - simulation_duration (list): A list containing duration of each generation of the simulation.
        - population (list or np.ndarray): The final population at the end of the simulation.
    """
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    full_path = os.path.join(result_path, f"{file_name}.h5")

    with h5py.File(full_path, "w") as file:

        for specie, array in zip(species, elite_chromosome):
            file.create_dataset(name=specie, data=array)

        file.create_dataset(name="best_fitness", data=best_fitness)
        file.create_dataset(name="simulation_duration", data=simulation_duration)
        file.create_dataset(name="best_results", data=population)









