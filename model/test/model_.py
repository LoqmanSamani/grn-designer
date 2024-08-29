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
    def __init__(self, population_size, pooling_method, pool_size, strides, padding, zero_padding, pooling_kernel_size, up_padding,
                 up_strides, individual_shape, species_parameters, complex_parameters,
                              num_species, num_pairs, max_sim_epochs, sim_stop_time, time_step, individual_fix_size
                 ):
        pass










