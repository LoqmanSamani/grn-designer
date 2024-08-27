from gabonst import *
from simulation import *




# how to test evolutionary_optimization()
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











# how to test GradientOptimization:
t = np.zeros((10, 10))
t[:, 8] = 1
t[:, 7] = 0.6
t[:, 5:7] = 1.4
tt = tf.convert_to_tensor(t, dtype=tf.float32)


ind = np.zeros((7, 10, 10))
ind[1, :, 3:5] = 1
ind[3, :, -2:] = 1
ind[-1, -1, :5] = [2, 1, 50, 5, .1]
ind[-1, 0, :3] = [.9, .1, 6]
ind[-1, 2, :3] = [.9, .1, 8]
ind[-2, 0, :2] = [0, 2]
ind[-2, 1, :4] = [.6, .1, .1, 4]
t_ind = tf.convert_to_tensor(ind, dtype=tf.float32)

g = GradientOptimization(
    epochs=10,
    learning_rate=0.01,
    target=tt,
    cost_alpha=0.1,
    cost_beta=0.1,
    cost_kernel_size=3,
    weight_decay=0.01
)

inds, costs = g.gradient_optimization(t_ind)

print(costs)

"""
Epoch 1/10, Cost: 0.4855841100215912
Epoch 2/10, Cost: 0.5479841828346252
Epoch 3/10, Cost: 0.4306187331676483
Epoch 4/10, Cost: 0.41393744945526123
Epoch 5/10, Cost: 0.4987949728965759
Epoch 6/10, Cost: 0.8985116481781006
Epoch 7/10, Cost: 0.4105818569660187
Epoch 8/10, Cost: 0.40525782108306885
Epoch 9/10, Cost: 0.41774845123291016
Epoch 10/10, Cost: 0.40810641646385193
[0.4855841, 0.5479842, 0.43061873, 0.41393745, 0.49879497, 0.89851165, 0.41058186, 0.40525782, 0.41774845, 0.40810642]

"""








# How to test tensor_simulation(...)
parameters = {
    "species_1": tf.Variable([.09, .007, 1.1]),
    "species_2": tf.Variable([0.09, 0.006, 1.2]),
    "pair_1": tf.Variable([6, .01, 0.001, 1.3])
}

num_species = 2
num_pairs = 1
stop = 5
time_step = .01
max_epoch = 500

ind = np.zeros((7, 30, 30))
ind[1, :, 0] = 10
ind[3, :, -1] = 10
ind[-2, 0, 0:2] = [0, 2]
ind_tensor = tf.convert_to_tensor(ind, dtype=tf.float32)

ind1 = tensor_simulation(
    individual=ind_tensor,
    parameters=parameters,
    num_species=num_species,
    num_pairs=num_pairs,
    stop=stop,
    time_step=time_step,
    max_epoch=max_epoch
)
print(ind1)
import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(ind1)
plt.show()


