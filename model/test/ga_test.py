from gabonst import *
from simulation import *



"""
ind = np.zeros((3, 10, 10))
ind[1, :, :] = np.random.rand(10, 10)
ind[-1, 0, :3] = [.5, .4, 2]
ind[-1, -1, :5] = [1, 0, 1000, 10, 0.01]

pop = [ind for i in range(20)]
target = np.random.rand(10, 10) * 10
new_pop = evolutionary_optimization(
    population=pop,
    target=target,
    cost_alpha=.1,  # used in GRM cost method
    cost_beta=.1,  # used in GRM cost method
    cost_kernel_size=3,  # used in GRM cost method
    cost_method="MSE",  # cost method: MSE (mean squared error) in this case will be used
    sim_mutation_rate=.3,
    compartment_mutation_rate=.1,
    parameter_mutation_rate=.2,
    insertion_mutation_rate=.2,
    deletion_mutation_rate=.3,
    sim_means=[0, 0],
    sim_std_devs=[300, 0.1],
    sim_min_vals=[50, .001],
    sim_max_vals=[1000, 0.7],
    compartment_mean=0,
    compartment_std=20,
    compartment_min_val=0,
    compartment_max_val=100,
    sim_distribution="uniform",
    compartment_distribution="uniform",
    species_param_means=[0, 0, 0],
    species_param_stds=[1, 1, 10],
    species_param_min_vals=[0, 0, 0],
    species_param_max_vals=[2, 2, 20],
    complex_param_means=[0, 0, 0, 0],
    complex_param_stds=[2, 2, 2, 30],
    complex_param_min_vals=[0, 0, 0, 0],
    complex_param_max_vals=[5, 5, 5, 1000],
    param_distribution="uniform",
    sim_mutation=True,
    compartment_mutation=True,
    param_mutation=True,
    species_insertion_mutation=True,
    species_deletion_mutation=True,
    crossover_alpha=0.4,
    sim_crossover=True,
    compartment_crossover=True,
    param_crossover=True,
    num_elite_individuals=3,
    individual_fix_size=False,
    species_parameters=[.1, .2, 3],  # in this case it is not used
    complex_parameters=[.1, .01, .001, 5]  # in this case it is not used
)

print(new_pop)

for i in new_pop:
    print(i.shape)

ind = np.zeros((3, 10, 10))
ind[1, :, :] = np.random.rand(10, 10)
ind[-1, 0, :3] = [.5, .4, 2]
ind[-1, -1, :5] = [1, 0, 1000, 10, 0.01]

pop = [ind for i in range(20)]
target = np.random.rand(10, 10) * 10
new_pop = evolutionary_optimization(
    population=pop,
    target=target,
    cost_alpha=.1,  # used in GRM cost method
    cost_beta=.1,  # used in GRM cost method
    cost_kernel_size=3,  # used in GRM cost method
    cost_method="MSE",  # cost method: MSE (mean squared error) in this case will be used
    sim_mutation_rate=.3,
    compartment_mutation_rate=.1,
    parameter_mutation_rate=.2,
    insertion_mutation_rate=.2,
    deletion_mutation_rate=.3,
    sim_means=[0, 0],
    sim_std_devs=[300, 0.1],
    sim_min_vals=[50, .001],
    sim_max_vals=[1000, 0.7],
    compartment_mean=0,
    compartment_std=20,
    compartment_min_val=0,
    compartment_max_val=100,
    sim_distribution="normal",
    compartment_distribution="normal",
    species_param_means=[0, 0, 0],
    species_param_stds=[1, 1, 10],
    species_param_min_vals=[0, 0, 0],
    species_param_max_vals=[2, 2, 20],
    complex_param_means=[0, 0, 0, 0],
    complex_param_stds=[2, 2, 2, 30],
    complex_param_min_vals=[0, 0, 0, 0],
    complex_param_max_vals=[5, 5, 5, 1000],
    param_distribution="normal",
    sim_mutation=True,
    compartment_mutation=True,
    param_mutation=True,
    species_insertion_mutation=True,
    species_deletion_mutation=True,
    crossover_alpha=0.4,
    sim_crossover=True,
    compartment_crossover=True,
    param_crossover=True,
    num_elite_individuals=3,
    individual_fix_size=False,
    species_parameters=[.1, .2, 3],  # in this case it is not used
    complex_parameters=[.1, .01, .001, 5]  # in this case it is not used
)

print(new_pop)

for i in new_pop:
    print(i.shape)

ind = np.zeros((3, 10, 10))
ind[1, :, :] = np.random.rand(10, 10)
ind[-1, 0, :3] = [.5, .4, 2]
ind[-1, -1, :5] = [1, 0, 1000, 10, 0.01]

pop = [ind for i in range(20)]
target = np.random.rand(10, 10) * 10
new_pop = evolutionary_optimization(
    population=pop,
    target=target,
    cost_alpha=.1,  # used in GRM cost method
    cost_beta=.1,  # used in GRM cost method
    cost_kernel_size=3,  # used in GRM cost method
    cost_method="GRM",  # cost method: GRM fitness error
    sim_mutation_rate=.3,
    compartment_mutation_rate=.1,
    parameter_mutation_rate=.2,
    insertion_mutation_rate=.2,
    deletion_mutation_rate=.3,
    sim_means=[0, 0],
    sim_std_devs=[300, 0.1],
    sim_min_vals=[50, .001],
    sim_max_vals=[1000, 0.7],
    compartment_mean=0,
    compartment_std=20,
    compartment_min_val=0,
    compartment_max_val=100,
    sim_distribution="uniform",
    compartment_distribution="uniform",
    species_param_means=[0, 0, 0],
    species_param_stds=[1, 1, 10],
    species_param_min_vals=[0, 0, 0],
    species_param_max_vals=[2, 2, 20],
    complex_param_means=[0, 0, 0, 0],
    complex_param_stds=[2, 2, 2, 30],
    complex_param_min_vals=[0, 0, 0, 0],
    complex_param_max_vals=[5, 5, 5, 1000],
    param_distribution="uniform",
    sim_mutation=True,
    compartment_mutation=True,
    param_mutation=True,
    species_insertion_mutation=True,
    species_deletion_mutation=True,
    crossover_alpha=0.4,
    sim_crossover=True,
    compartment_crossover=True,
    param_crossover=True,
    num_elite_individuals=3,
    individual_fix_size=False,
    species_parameters=[.1, .2, 3],  # in this case it is not used
    complex_parameters=[.1, .01, .001, 5]  # in this case it is not used
)

print(new_pop)

for i in new_pop:
    print(i.shape)

ind = np.zeros((3, 10, 10))
ind[1, :, :] = np.random.rand(10, 10)
ind[-1, 0, :3] = [.5, .4, 2]
ind[-1, -1, :5] = [1, 0, 1000, 10, 0.01]

pop = [ind for i in range(20)]
target = np.random.rand(10, 10) * 10
new_pop = evolutionary_optimization(
    population=pop,
    target=target,
    cost_alpha=.1,  # used in GRM cost method
    cost_beta=.1,  # used in GRM cost method
    cost_kernel_size=3,  # used in GRM cost method
    cost_method="NCC",  # cost method: Normalized Cross-Correlation (NCC)
    sim_mutation_rate=.3,
    compartment_mutation_rate=.1,
    parameter_mutation_rate=.2,
    insertion_mutation_rate=.2,
    deletion_mutation_rate=.3,
    sim_means=[0, 0],
    sim_std_devs=[300, 0.1],
    sim_min_vals=[50, .001],
    sim_max_vals=[1000, 0.7],
    compartment_mean=0,
    compartment_std=20,
    compartment_min_val=0,
    compartment_max_val=100,
    sim_distribution="uniform",
    compartment_distribution="uniform",
    species_param_means=[0, 0, 0],
    species_param_stds=[1, 1, 10],
    species_param_min_vals=[0, 0, 0],
    species_param_max_vals=[2, 2, 20],
    complex_param_means=[0, 0, 0, 0],
    complex_param_stds=[2, 2, 2, 30],
    complex_param_min_vals=[0, 0, 0, 0],
    complex_param_max_vals=[5, 5, 5, 1000],
    param_distribution="uniform",
    sim_mutation=True,
    compartment_mutation=True,
    param_mutation=True,
    species_insertion_mutation=True,
    species_deletion_mutation=True,
    crossover_alpha=0.4,
    sim_crossover=True,
    compartment_crossover=True,
    param_crossover=True,
    num_elite_individuals=3,
    individual_fix_size=False,
    species_parameters=[.1, .2, 3],  # in this case it is not used
    complex_parameters=[.1, .01, .001, 5]  # in this case it is not used
)

print(new_pop)

for i in new_pop:
    print(i.shape)
"""



class Test:
    def __init__(self, ind):
        self.ind = ind

    def simulation_test(self):
        sim_ind = individual_simulation(self.ind)
        return sim_ind



ind = np.zeros((7, 100, 100))
ind[1, :, 0] = 1
ind[3, :, -1] = 1

ind[-1, 0, :3] = [.09, .007, 1.1]
ind[-1, 2, :3] = [0.09, 0.006, 1.2]
ind[-1, -1, :5] = [2, 1, 1000, 5, .01]
ind[-2, 0, 0:2] = [0, 2]
ind[-2, 1, 0:4] = [6, .01, 0.001, 1.3]

g = Test(ind)

sp1 = g.simulation_test()
print(sp1)
