from evolution.gabonst import evolutionary_optimization
from initialization import population_initialization
from pooling import PoolingLayers
from gradient.optimization import GradientOptimization
import numpy as np
import tensorflow as tf
import json
import os
import h5py
import time





class BioEsAg:
    def __init__(self,
                 target, population_size, individual_shape, individual_parameters, simulation_parameters, store_path=None,
                 cost_alpha=0.01, cost_beta=0.1, cost_kernel_size=3, cost_method="MSE",
                 learning_rate=0.01, weight_decay=0.01, optimization_epochs=50, gradient_optimization=False, num_gradient_optimization=3,
                 num_saved_individuals=3, evolution_one_epochs=100, evolution_two_epochs=50, evolution_two_ratio=0.2,
                 pooling=False, pooling_method="average", pool_size=(3, 3), strides=(2, 2), padding="valid", zero_padding=(1, 1),
                 pool_kernel_size=(3, 3), up_padding="same", up_strides=(2, 2), individual_fix_shape=False,
                 sim_mutation=True, compartment_mutation=True, param_mutation=False, species_insertion_mutation_one=False,
                 species_deletion_mutation_one=False, species_insertion_mutation_two=False, species_deletion_mutation_two=False,
                 crossover_alpha=0.5, sim_crossover=True, compartment_crossover=True, param_crossover=False, num_elite_individuals=5,
                 sim_mutation_rate=0.05, compartment_mutation_rate=0.8, parameter_mutation_rate=0.05, insertion_mutation_rate=0.2,
                 deletion_mutation_rate=0.25, sim_means=(5.0, 0.5), sim_std_devs=(100.0, 2.0), sim_min_vals=(3.0, 0.001),
                 sim_max_vals=(200.0, 2.0), compartment_mean=0.0, compartment_std=200.0, compartment_min_val=0.0,
                 compartment_max_val=1000, sim_distribution="uniform", compartment_distribution="uniform",
                 species_param_means=(0.5, 0.1, 1.0), species_param_stds=(10.0, 5.0, 10.0), species_param_min_vals=(0.0, 0.0, 0.0),
                 species_param_max_vals=(10, 7, 20), complex_param_means=(3, 0.1, 0.1, 1.0), complex_param_stds=(100.0, 10.0, 5.0, 10.0),
                 complex_param_min_vals=(0.0, 0.0, 0.0, 0.0), complex_param_max_vals=(1000, 50, 100, 50), param_distribution="uniform"
                 ):
        """
        BioEsAg (Bio-Optimization with Evolutionary Strategies and Adaptive Gradient-based Optimization) is an
        advanced algorithm designed for optimizing biological systems. It combines evolutionary strategies,
        gradient-based optimization, and pooling techniques to optimize the initial conditions, parameters, and
        relationships between species in a biological model.

        The algorithm works through several phases:

            1. Pooling Down-sampling (Optional): This initial step reduces computational costs by down-sampling the target
                                                 compartments before initializing the population. This phase is optional;
                                                  the algorithm can proceed without down-sampling if preferred.

            2. Initialization: The algorithm starts by initializing a diverse population of individuals, each representing
                               different species and parameter settings within the model.

            3. Evolutionary Optimization (Phase 1): In this phase, the algorithm uses evolutionary methods like mutation and
                                                    crossover to explore and improve the population. This helps in finding
                                                    optimal solutions through iterative adjustments.

            4. Pooling Up-sampling (If Down-sampling Was Used): If down-sampling was performed, this step up-samples the population
                                                                back to its original size. This allows for a more detailed and refined
                                                                search in the next phase.

            5. Evolutionary Optimization (Phase 2): The algorithm then further refines the population, focusing on a subset of individuals
                                                    from the up-sampled population. This phase includes additional optimization to enhance the results.

            6. Gradient-based Optimization with Adam (Optional): Finally, if needed, the algorithm uses the Adam optimizer to fine-tune
                                                                 the best-performing individuals from Phase 2. This step is optional and only
                                                                 used if further refinement is necessary.

        This approach is particularly well-suited for complex biological systems where the interaction between species
        and environmental factors is highly non-linear and difficult to model using traditional methods.


        Initialize the BioEsAg with the given parameters.

        Parameters:
            - target (np.ndarray): The target matrix to be used for up sampling.
            - population_size (int): The number of individuals in the population.
            - individual_shape (tuple of int): The shape of each individual, represented as a 3D array (z, y, x).
            - individual_parameters (dict): Contains species_parameters and pair_parameters.
                - species_parameters (tuple): A list of parameter sets for each species in the individual, each with production rate, degradation rate, and diffusion rate.
                - pair_parameters (tuple): A list of parameter sets for each complex, each with a list of species and corresponding rates.
            - simulation_parameters (dict): Contains max_simulation_epoch (int), sim_stop_time (int/float), and time_step (float).
            - store_path (str, optional): The path where data files will be stored. If None, defaults to the user's home directory.
            - cost_alpha (float): Weighting factor for the primary component of the cost function.
            - cost_beta (float): Weighting factor for the secondary component of the cost function.
            - cost_kernel_size (int): The size of the kernel used in the cost computation.
            - cost_method (str): The method used to compute the cost function (e.g., "MSE", "MAE").
            - learning_rate (float): The learning rate for gradient optimization.
            - weight_decay (float): The weight decay rate for gradient optimization.
            - optimization_epochs (int): The number of epochs for gradient optimization.
            - gradient_optimization (bool): Flag indicating whether to apply gradient optimization.
            - num_gradient_optimization (int): Number of gradient optimization runs.
            - num_saved_individuals (int): The number of individuals to save from the evolution process.
            - evolution_one_epochs (int): Number of iterations for the first phase of evolutionary optimization.
            - evolution_two_epochs (int): Number of iterations for the second phase of evolutionary optimization.
            - evolution_two_ratio (float): Ratio of the second phase in evolutionary optimization.
            - pooling (bool): Flag indicating whether pooling should be applied.
            - pooling_method (str): Method used for pooling, e.g., "average" or "max".
            - pool_size (tuple of int): Size of the pooling window (height, width).
            - strides (tuple of int): Strides of the pooling operation (height, width).
            - padding (str): Padding mode for the pooling operation ("valid" or "same").
            - zero_padding (tuple of int): Zero padding to apply before pooling (height, width).
            - pool_kernel_size (tuple of int): Size of the kernel for the transposed convolution (height, width).
            - up_padding (str): Padding mode for the up sampling operation ("valid" or "same").
            - up_strides (tuple of int): Strides of the up sampling operation (height, width).
            - individual_fix_shape (bool): Flag indicating whether to fix the shape of individuals.
            - sim_mutation (bool): Flag indicating whether to apply simulation parameter mutations.
            - compartment_mutation (bool): Flag indicating whether to apply compartment parameter mutations.
            - param_mutation (bool): Flag indicating whether to apply species and complex parameter mutations.
            - species_insertion_mutation_one (bool): Flag for species insertion mutations in phase one.
            - species_deletion_mutation_one (bool): Flag for species deletion mutations in phase one.
            - species_insertion_mutation_two (bool): Flag for species insertion mutations in phase two.
            - species_deletion_mutation_two (bool): Flag for species deletion mutations in phase two.
            - crossover_alpha (float): Weighting factor for crossover operations.
            - sim_crossover (bool): Flag indicating whether to apply crossover to simulation variables.
            - compartment_crossover (bool): Flag indicating whether to apply crossover to compartment parameters.
            - param_crossover (bool): Flag indicating whether to apply crossover to species and complex parameters.
            - num_elite_individuals (int): Number of elite individuals selected for crossover operations.
            - sim_mutation_rate (float): Mutation rate for simulation parameters.
            - compartment_mutation_rate (float): Mutation rate for compartment parameters.
            - parameter_mutation_rate (float): Mutation rate for species and complex parameters.
            - insertion_mutation_rate (float): Mutation rate for species insertion operations.
            - deletion_mutation_rate (float): Mutation rate for species deletion operations.
            - sim_means (tuple of float): Mean values for the simulation parameters.
            - sim_std_devs (tuple of float): Standard deviation values for the simulation parameters.
            - sim_min_vals (tuple of float): Minimum values for the simulation parameters.
            - sim_max_vals (tuple of float): Maximum values for the simulation parameters.
            - compartment_mean (float): Mean value for compartment parameters.
            - compartment_std (float): Standard deviation value for compartment parameters.
            - compartment_min_val (float): Minimum value for compartment parameters.
            - compartment_max_val (float): Maximum value for compartment parameters.
            - sim_distribution (str): Distribution type for simulation mutations (e.g., "normal", "uniform").
            - compartment_distribution (str): Distribution type for compartment mutations (e.g., "normal", "uniform").
            - species_param_means (tuple of float): Mean values for species parameters.
            - species_param_stds (tuple of float): Standard deviation values for species parameters.
            - species_param_min_vals (tuple of float): Minimum values for species parameters.
            - species_param_max_vals (tuple of float): Maximum values for species parameters.
            - complex_param_means (tuple of float): Mean values for complex parameters.
            - complex_param_stds (tuple of float): Standard deviation values for complex parameters.
            - complex_param_min_vals (tuple of float): Minimum values for complex parameters.
            - complex_param_max_vals (tuple of float): Maximum values for complex parameters.
            - param_distribution (str): Distribution type for parameter mutations (e.g., "normal", "uniform").
        """
        self.target = target
        self.population_size = population_size
        self.individual_shape = individual_shape
        self.individual_parameters = individual_parameters
        self.simulation_parameters = simulation_parameters
        self.store_path = store_path

        # Cost function parameters
        self.cost_alpha = cost_alpha
        self.cost_beta = cost_beta
        self.cost_kernel_size = cost_kernel_size
        self.cost_method = cost_method

        # Optimization parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimization_epochs = optimization_epochs
        self.gradient_optimization = gradient_optimization
        self.num_gradient_optimization = num_gradient_optimization
        self.num_saved_individuals = num_saved_individuals

        # Evolutionary optimization parameters
        self.evolution_one_epochs = evolution_one_epochs
        self.evolution_two_epochs = evolution_two_epochs
        self.evolution_two_ratio = evolution_two_ratio

        # Pooling parameters
        self.pooling = pooling
        self.pooling_method = pooling_method
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.zero_padding = zero_padding
        self.pool_kernel_size = pool_kernel_size
        self.up_padding = up_padding
        self.up_strides = up_strides
        self.individual_fix_shape = individual_fix_shape

        # Mutation parameters
        self.sim_mutation = sim_mutation
        self.compartment_mutation = compartment_mutation
        self.param_mutation = param_mutation
        self.species_insertion_mutation_one = species_insertion_mutation_one
        self.species_deletion_mutation_one = species_deletion_mutation_one
        self.species_insertion_mutation_two = species_insertion_mutation_two
        self.species_deletion_mutation_two = species_deletion_mutation_two
        self.crossover_alpha = crossover_alpha
        self.sim_crossover = sim_crossover
        self.compartment_crossover = compartment_crossover
        self.param_crossover = param_crossover
        self.num_elite_individuals = num_elite_individuals

        # Mutation rates
        self.sim_mutation_rate = sim_mutation_rate
        self.compartment_mutation_rate = compartment_mutation_rate
        self.parameter_mutation_rate = parameter_mutation_rate
        self.insertion_mutation_rate = insertion_mutation_rate
        self.deletion_mutation_rate = deletion_mutation_rate

        # Simulation parameters
        self.sim_means = sim_means
        self.sim_std_devs = sim_std_devs
        self.sim_min_vals = sim_min_vals
        self.sim_max_vals = sim_max_vals
        self.compartment_mean = compartment_mean
        self.compartment_std = compartment_std
        self.compartment_min_val = compartment_min_val
        self.compartment_max_val = compartment_max_val
        self.sim_distribution = sim_distribution
        self.compartment_distribution = compartment_distribution

        # Species and complex parameters
        self.species_param_means = species_param_means
        self.species_param_stds = species_param_stds
        self.species_param_min_vals = species_param_min_vals
        self.species_param_max_vals = species_param_max_vals
        self.complex_param_means = complex_param_means
        self.complex_param_stds = complex_param_stds
        self.complex_param_min_vals = complex_param_min_vals
        self.complex_param_max_vals = complex_param_max_vals
        self.param_distribution = param_distribution

        # Initialize PoolingLayers
        self.pooling_layers = PoolingLayers(
            target=self.target,
            pooling_method=self.pooling_method,
            pool_size=self.pool_size,
            strides=self.strides,
            padding=self.padding,
            zero_padding=self.zero_padding,
            kernel_size=self.pool_kernel_size,
            up_padding=self.up_padding,
            up_strides=self.up_strides
        )

        # Initialize Gradient Optimization
        self.gradient_optimization = GradientOptimization(
            epochs=self.optimization_epochs,
            learning_rate=self.learning_rate,
            target=tf.convert_to_tensor(self.target),  # Convert target array to tf.tensor
            cost_alpha=self.cost_alpha,
            cost_beta=self.cost_beta,
            cost_kernel_size=self.cost_kernel_size,
            weight_decay=self.weight_decay
        )


    # store all input information to a json file to use it later if needed (reproduce)
    def save_to_json(self):
        """
        Save the model's input configuration to a JSON file.

        This method creates a dictionary of all the input parameters used to initialize the model.
        It then saves this dictionary as a JSON file in the specified `store_path`. If `store_path`
        is not provided, the file will be saved in the user's home directory.

        The JSON file can later be used to reproduce the exact configuration of the model.

        Parameters:
        None

        Returns:
        None
        """

        data = {
            "population_size": self.population_size,
            "individual_shape": self.individual_shape,
            "individual_parameters": self.individual_parameters,
            "simulation_parameters": self.simulation_parameters,
            "cost_alpha": self.cost_alpha,
            "cost_beta": self.cost_beta,
            "cost_kernel_size": self.cost_kernel_size,
            "cost_method": self.cost_method,
            "sim_mutation_rate": self.sim_mutation_rate,
            "compartment_mutation_rate": self.compartment_mutation_rate,
            "parameter_mutation_rate": self.parameter_mutation_rate,
            "insertion_mutation_rate": self.insertion_mutation_rate,
            "deletion_mutation_rate": self.deletion_mutation_rate,
            "sim_means": self.sim_means,
            "sim_std_devs": self.sim_std_devs,
            "sim_min_vals": self.sim_min_vals,
            "sim_max_vals": self.sim_max_vals,
            "compartment_mean": self.compartment_mean,
            "compartment_std": self.compartment_std,
            "compartment_min_val": self.compartment_min_val,
            "compartment_max_val": self.compartment_max_val,
            "sim_distribution": self.sim_distribution,
            "compartment_distribution": self.compartment_distribution,
            "species_param_means": self.species_param_means,
            "species_param_stds": self.species_param_stds,
            "species_param_min_vals": self.species_param_min_vals,
            "species_param_max_vals": self.species_param_max_vals,
            "complex_param_means": self.complex_param_means,
            "complex_param_stds": self.complex_param_stds,
            "complex_param_min_vals": self.complex_param_min_vals,
            "complex_param_max_vals": self.complex_param_max_vals,
            "param_distribution": self.param_distribution,
            "sim_mutation": self.sim_mutation,
            "compartment_mutation": self.compartment_mutation,
            "param_mutation": self.param_mutation,
            "species_insertion_mutation_one": self.species_insertion_mutation_one,
            "species_deletion_mutation_one": self.species_deletion_mutation_one,
            "species_insertion_mutation_two": self.species_insertion_mutation_two,
            "species_deletion_mutation_two": self.species_deletion_mutation_two,
            "crossover_alpha": self.crossover_alpha,
            "sim_crossover": self.sim_crossover,
            "compartment_crossover": self.compartment_crossover,
            "param_crossover": self.param_crossover,
            "num_elite_individuals": self.num_elite_individuals,
            "evolution_one_epochs": self.evolution_one_epochs,
            "evolution_two_epochs": self.evolution_two_epochs,
            "pooling": self.pooling,
            "pooling_method": self.pooling_method,
            "pool_size": self.pool_size,
            "strides": self.strides,
            "padding": self.padding,
            "zero_padding": self.zero_padding,
            "pool_kernel_size": self.pool_kernel_size,
            "up_padding": self.up_padding,
            "up_strides": self.up_strides,
            "individual_fix_shape": self.individual_fix_shape,
            "gradient_optimization": self.gradient_optimization,
            "optimization_epochs": self.optimization_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "num_saved_individuals": self.num_saved_individuals,
            "evolution_two_ratio": self.evolution_two_ratio,
            "num_gradient_optimization": self.num_gradient_optimization,
        }

        if self.store_path:
            path = os.path.join(self.store_path, "input_data.json")
            if not os.path.exists(self.store_path):
                os.makedirs(self.store_path)
        else:
            path = os.path.join(os.path.expanduser("~"), "input_data.json")

        # Save the dictionary to a JSON file
        with open(path, 'w') as json_file:
            json.dump(data, json_file, indent=4)


    def save_to_h5py(self, dataset_name, data_array):
        """
        Save a numpy array to an HDF5 file.

        This method appends or writes the provided `data_array` to an HDF5 file under the specified `dataset_name`.
        If `store_path` is specified, the file will be saved in that directory; otherwise, it will be saved in the
        user's home directory.

        The method ensures that the directory exists and creates it if necessary.

        Parameters:
            - dataset_name (str): The name of the dataset in the HDF5 file where the data will be stored.
            - data_array (np.ndarray): The numpy array containing the data to be saved.

        Returns:
        None
        """

        if self.store_path:
            path = os.path.join(self.store_path, "output_data.h5")
            if not os.path.exists(self.store_path):
                os.makedirs(self.store_path)
        else:
            path = os.path.join(os.path.expanduser("~"), "output_data.h5")

        with h5py.File(path, 'a') as h5file:
            h5file[dataset_name] = data_array



    def fit(self):
        """
        Execute the multi-phase evolutionary algorithm to optimize the population.

        This method orchestrates the entire optimization process, including:

            1. Saving the initial input parameters and target data.
            2. Performing a first phase of evolutionary optimization with potential pooling to reduce
               the computational cost by down-sampling the target.
            3. Re-scaling the population to its original size and performing a second phase of
               evolutionary optimization.
            4. Optionally applying gradient-based optimization to further refine the best individuals.

        The results are stored in HDF5 files at various stages for reproducibility and analysis.

        The optimization process is divided into three main phases:

            - **Phase 1**: Evolutionary optimization with or without pooling to reduce the target size.
            - **Phase 2**: Evolutionary optimization on the original target size.
            - **Phase 3**: Gradient-based optimization for fine-tuning the best individuals from Phase 2.

        Key Steps:

            1. Save initial inputs and targets.
            2. Pool the target if enabled and perform the first phase of evolutionary optimization.
            3. Save the elite individuals and their costs from Phase 1.
            4. Upsample the population and perform the second phase of evolutionary optimization.
            5. Optionally, perform gradient-based optimization on the top individuals from Phase 2.

        Parameters:
        None

        Returns:
        None
        """
        prepararion_time = None
        evolution_time_1 = None
        evolution_time_2 = None
        gradient_time = None

        prep_start = time.time()
        # Save the input configuration and initial target to files
        self.save_to_json()  # save the input info into a JSON file
        self.save_to_h5py(
            dataset_name="original_target",
            data_array=self.target
        )  # store the target into an HDF5 file

        num_species = len(self.individual_parameters["species_parameters"])
        num_pairs = len(self.individual_parameters["pair_parameters"])

        # Phase 1: Pooling the Target to reduce the computational cost
        if self.pooling:
            target_ = self.pooling_layers.pooling(
                compartment=self.target,
                method="target pooling"
            )
        else:
            target_ = self.target

        self.save_to_h5py(
            dataset_name="down_sampled_target",
            data_array=target_
        )  # store the down-sampled target into the already created HDF5 file


        # Initialize the population based on the reduced target shape
        population = population_initialization(
            population_size=self.population_size,
            individual_shape=(self.individual_shape[0], target_.shape[0], target_.shape[1]),
            species_parameters=self.individual_parameters["species_parameters"],
            complex_parameters=self.individual_parameters["pair_parameters"],
            num_species=num_species,
            num_pairs=num_pairs,
            max_sim_epochs=self.simulation_parameters["max_simulation_epoch"],
            sim_stop_time=self.simulation_parameters["simulation_stop_time"],
            time_step=self.simulation_parameters["time_step"],
            individual_fix_size=self.individual_fix_shape
        )
        prep_stop = time.time()
        preparation_time = prep_stop - prep_start

        evo1_start = time.time()        
        # Phase 1 of Evolutionary Optimization
        evolution_costs_one = np.zeros(shape=(self.evolution_one_epochs, self.num_saved_individuals+2)) # array to save the cost of elite chromosomes

        print("----------------------------------------------------------------")
        print("                         BioEsAg Algorithm                      ")
        print("----------------------------------------------------------------")
        print()
        print("                   Evolutionary Optimization I                  ")
        print()

        for i in range(self.evolution_one_epochs):

            population, cost, mean_cost = evolutionary_optimization(
                population=population,
                target=target_,
                cost_alpha=self.cost_alpha,
                cost_beta=self.cost_beta,
                cost_kernel_size=self.cost_kernel_size,
                cost_method=self.cost_method,
                sim_mutation_rate=self.sim_mutation_rate,
                compartment_mutation_rate=self.compartment_mutation_rate,
                parameter_mutation_rate=self.parameter_mutation_rate,
                insertion_mutation_rate=self.insertion_mutation_rate,
                deletion_mutation_rate=self.deletion_mutation_rate,
                sim_means=self.sim_means,
                sim_std_devs=self.sim_std_devs,
                sim_min_vals=self.sim_min_vals,
                sim_max_vals=self.sim_max_vals,
                compartment_mean=self.compartment_mean,
                compartment_std=self.compartment_std,
                compartment_min_val=self.compartment_min_val,
                compartment_max_val=self.compartment_max_val,
                sim_distribution=self.sim_distribution,
                compartment_distribution=self.compartment_distribution,
                species_param_means=self.species_param_means,
                species_param_stds=self.species_param_stds,
                species_param_min_vals=self.species_param_min_vals,
                species_param_max_vals=self.species_param_max_vals,
                complex_param_means=self.complex_param_means,
                complex_param_stds=self.complex_param_stds,
                complex_param_min_vals=self.complex_param_min_vals,
                complex_param_max_vals=self.complex_param_max_vals,
                param_distribution=self.param_distribution,
                sim_mutation=self.sim_mutation,
                compartment_mutation=self.compartment_mutation,
                param_mutation=self.param_mutation,
                species_insertion_mutation=self.species_insertion_mutation_one,
                species_deletion_mutation=self.species_deletion_mutation_one,
                crossover_alpha=self.crossover_alpha,
                sim_crossover=self.sim_crossover,
                compartment_crossover=self.compartment_crossover,
                param_crossover=self.param_crossover,
                num_elite_individuals=self.num_elite_individuals,
                individual_fix_size=self.individual_fix_shape,
                species_parameters=self.individual_parameters["species_parameters"],
                complex_parameters=self.individual_parameters["pair_parameters"]
            )

            print(f"Epoch {i+1}/{self.evolution_one_epochs}, Average Population Cost: {mean_cost}")

            sorted_cost = np.sort(cost)
            evolution_costs_one[i, :-2] = sorted_cost[:self.num_saved_individuals]
            evolution_costs_one[i, -2] = mean_cost
            evolution_costs_one[i, -1] = sorted_cost[-1]

            # Shrink the population size based on a ratio (self.evolution_two_ratio) at the end of Phase 1
            if i == self.evolution_one_epochs - 1:
                new_population_size = int(self.population_size * self.evolution_two_ratio)
                sorted_cost_indices = np.argsort(cost)[:new_population_size]
                population = [population[idx] for idx in sorted_cost_indices]
                elite_individuals = [population[idx] for idx in sorted_cost_indices[:self.num_saved_individuals]]
                for d, elite in enumerate(elite_individuals):
                    self.save_to_h5py(
                        dataset_name=f"elite_individual_{d + 1}_evolution_one",
                        data_array=elite
                    )

        self.save_to_h5py(
            dataset_name="evolution_costs_one",
            data_array=evolution_costs_one
        )
        evo1_stop = time.time()
        evolution_time_1 = evo1_stop - evo1_start
        # Phase 2: Up-sampling the population (resize it to the original size)
        if self.pooling:
            population = self.pooling_layers.pooling(
                compartment=population,
                method="population up sampling"
            )



            # Phase 2 of Evolutionary Optimization
            evolution_costs_two = np.zeros(shape=(self.evolution_two_epochs, self.num_saved_individuals+2))  # array to save the cost of elite chromosomes

            print()
            print("                   Evolutionary Optimization II                 ")
            print()

            for j in range(self.evolution_two_epochs):
                population, cost, mean_cost = evolutionary_optimization(
                    population=population,
                    target=self.target,
                    cost_alpha=self.cost_alpha,
                    cost_beta=self.cost_beta,
                    cost_kernel_size=self.cost_kernel_size,
                    cost_method=self.cost_method,
                    sim_mutation_rate=self.sim_mutation_rate,
                    compartment_mutation_rate=self.compartment_mutation_rate,
                    parameter_mutation_rate=self.parameter_mutation_rate,
                    insertion_mutation_rate=self.insertion_mutation_rate,
                    deletion_mutation_rate=self.deletion_mutation_rate,
                    sim_means=self.sim_means,
                    sim_std_devs=self.sim_std_devs,
                    sim_min_vals=self.sim_min_vals,
                    sim_max_vals=self.sim_max_vals,
                    compartment_mean=self.compartment_mean,
                    compartment_std=self.compartment_std,
                    compartment_min_val=self.compartment_min_val,
                    compartment_max_val=self.compartment_max_val,
                    sim_distribution=self.sim_distribution,
                    compartment_distribution=self.compartment_distribution,
                    species_param_means=self.species_param_means,
                    species_param_stds=self.species_param_stds,
                    species_param_min_vals=self.species_param_min_vals,
                    species_param_max_vals=self.species_param_max_vals,
                    complex_param_means=self.complex_param_means,
                    complex_param_stds=self.complex_param_stds,
                    complex_param_min_vals=self.complex_param_min_vals,
                    complex_param_max_vals=self.complex_param_max_vals,
                    param_distribution=self.param_distribution,
                    sim_mutation=self.sim_mutation,
                    compartment_mutation=self.compartment_mutation,
                    param_mutation=self.param_mutation,
                    species_insertion_mutation=self.species_insertion_mutation_two,
                    species_deletion_mutation=self.species_deletion_mutation_two,
                    crossover_alpha=self.crossover_alpha,
                    sim_crossover=self.sim_crossover,
                    compartment_crossover=self.compartment_crossover,
                    param_crossover=self.param_crossover,
                    num_elite_individuals=self.num_elite_individuals,
                    individual_fix_size=self.individual_fix_shape,
                    species_parameters=self.individual_parameters["species_parameters"],
                    complex_parameters=self.individual_parameters["pair_parameters"]
                )

                print(f"Epoch {j+1}/{self.evolution_two_epochs}, Average Population Cost: {mean_cost}")

                sorted_cost = np.sort(cost)
                evolution_costs_two[j, :-2] = sorted_cost[:self.num_saved_individuals]
                evolution_costs_two[j, -2] = mean_cost
                evolution_costs_two[j, -1] = sorted_cost[-1]

                # Shrink the population size at the end of Phase 2
                if j == self.evolution_two_epochs - 1:
                    sorted_cost_indices = np.argsort(cost)[:self.num_gradient_optimization]
                    population = [population[idx] for idx in sorted_cost_indices]
                    elite_individuals = [population[idx] for idx in sorted_cost_indices[:self.num_saved_individuals]]
                    for d, elite in enumerate(elite_individuals):
                        self.save_to_h5py(
                            dataset_name=f"elite_individual_{d + 1}_evolution_two",
                            data_array=elite
                        )

            self.save_to_h5py(
                dataset_name="evolution_costs_two",
                data_array=evolution_costs_two
            )


        # Phase 3: Gradient-based optimization using tf.GradientTape and the Adam algorithm
        if self.gradient_optimization:
            optimization_costs = np.zeros(shape=(self.optimization_epochs, self.num_gradient_optimization))

            print()
            print("                   Adam Optimization                 ")
            print()

            for k, individual in enumerate(population):

                print(f"                Individual {k}                  ")
                print()

                optimized_individual, costs = self.gradient_optimization.gradient_optimization(
                    individual=tf.convert_to_tensor(individual)
                )

                population[k] = optimized_individual.numpy()
                optimization_costs[k, :] = costs

                self.save_to_h5py(
                    dataset_name=f"elite_individual_{k + 1}_gradient_optimization",
                    data_array=population[k]
                )

            self.save_to_h5py(
                dataset_name="gradient_optimization_costs",
                data_array=optimization_costs
            )

