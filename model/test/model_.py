from cost import compute_cost
from crossover import apply_crossover
from simulation import individual_simulation
from tensor_simulation import tensor_simulation
from gabonst import evolutionary_optimization
from initialization import population_initialization
from mutation import apply_mutation
from pooling import PoolingLayers
from optimization import GradientOptimization


class OptModel:
    def __init__(self, target, population_size, individual_shape, individual_parameters, simulation_parameters,
                 pooling=False, pooling_method="average", pool_size=(3, 3), strides=(3, 3), padding="valid", zero_padding=(1, 1), pool_kernel_size=(3, 3),
                 up_padding="valid",  up_strides=(3, 3), individual_fix_shape=False):
        """
        Parameters:

            - target (np.ndarray): The target matrix to be used for up sampling.
            - population_size(int): The number of individuals in the population.
            - individual_shape(tuple of int): The shape of each individual, represented as a 3D array (z, y, x).
            - individual_parameters (dict):
                 - species_parameters (tuple): A list of parameter sets for each species in the individual.
                                               Each species is defined by a list containing its production rate, degradation rate, and diffusion rate.
                 - pair_parameters(tuple): A list of parameter sets for each complex. Each entry contains a tuple,
                                           where the first element is a list of species involved, and the second element
                                           is a list of corresponding rates (e.g., collision rate, dissociation rate, etc.).

            - simulation_parameters (dict): max_simulation_epoch(int), sim_stop_time(int/float), time_step(float).
            - pooling (boolean): true or false.
            - pooling_method (str): max or average
            - pool_size (tuple of int): The size of the pooling window (height, width).
            - strides (tuple of int): The strides of the pooling operation (height, width).
            - padding (str): Padding mode for the pooling operation. Either 'valid' or 'same'.
            - zero_padding (tuple of int): The amount of zero padding to apply before pooling (height, width).
            - pool_kernel_size (tuple of int): The size of the kernel for the transposed convolution (height, width).
            - up_padding (str): Padding mode for the up sampling operation. Either 'valid' or 'same'.
            - up_strides (tuple of int): The strides of the up sampling operation (height, width).

            - individual_fix_shape(boolean): true or false
        """
        self.target = target
        self.population_size = population_size
        self.individual_shape = individual_shape
        self.individual_parameters = individual_parameters
        self.pooling = pooling
        self.pooling_method = pooling_method
        self.simulation_parameters = simulation_parameters
        self.individual_fix_shape = individual_fix_shape
        self.zero_padding = zero_padding
        self.pool_size = pool_size
        self.padding = padding
        self.strides = strides
        self.zero_padding = zero_padding
        self. pooling_layers = PoolingLayers(
            target=self.target,
            pooling_method=self.pooling_method,
            pool_size=self.pool_size,
            strides=self.strides,
            padding=self.padding,
            zero_padding=self.zero_padding,
            kernel_size=pool_kernel_size,
            up_padding=up_padding,
            up_strides=up_strides
        )



        def fit():

            num_species = len(self.individual_parameters["species_parameters"])
            num_pairs = len(self.individual_parameters["pair_parameters"])

            if self.pooling:
                target_ = self.pooling_layers.down_sample_target(
                    target=self.target,
                    pooling_method=self.pooling_method,
                    zero_padding=self.zero_padding,
                    pool_size=self.pool_size,
                    strides=self.strides,
                    padding=self.padding
                )
            else:
                target_ = self.target


            population = population_initialization(
                population_size=self.population_size,
                individual_shape=(self.individual_shape[0], target_.shape[0], target_.shape[1]),
                species_parameters=self.individual_parameters["species_parameters"],
                complex_parameters=self.individual_parameters["pair_parameters"],
                num_species=num_species,
                num_pairs=num_pairs,
                max_sim_epochs=simulation_parameters["max_simulation_epoch"],
                sim_stop_time=simulation_parameters["simulation_stop_time"],
                time_step=simulation_parameters["time_step"],
                individual_fix_size=self.individual_fix_shape
            )













