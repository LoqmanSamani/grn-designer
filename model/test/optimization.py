import tensorflow as tf
from simulation import *
from cost import *
import numpy as np

class GradientOptimization:

    def __init__(self, epochs, learning_rate, target, cost_alpha, cost_beta, cost_kernel_size, weight_decay):

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.target = target
        self.cost_alpha = cost_alpha
        self.cost_beta = cost_beta
        self.cost_kernel_size = cost_kernel_size
        self.weight_decay = weight_decay


    def parameter_extraction(self, individual):

        parameters = {}
        num_species = int(individual[-1, -1, 0])
        num_pairs = int(individual[-1, -1, 1])
        pair_start = int(num_species * 2)
        pair_stop = int(pair_start + (num_pairs * 2))

        species = 1
        for i in range(0, num_species*2, 2):
            parameters[f"species_{species}"] = tf.Variable(individual[-1, i, 0:3], trainable=True)
            species += 1

        pair = 1
        for j in range(pair_start+1, pair_stop+1, 2):
            parameters[f"pair_{pair}"] = tf.Variable(individual[j, 1, :4], trainable=True)
            pair += 1

        return parameters

    def update_parameters(self, individual, parameters):

        num_species = int(individual[-1, -1, 0])
        num_pairs = int(individual[-1, -1, 1])
        pair_start = int(num_species * 2)
        pair_stop = int(pair_start + (num_pairs * 2))

        species = 1
        for i in range(0, num_species * 2, 2):
            individual[-1, i, :3] = parameters[f"species_{species}"].numpy()
            species += 1

        pair = 1
        for j in range(pair_start + 1, pair_stop + 1, 2):
            individual[j, 1, :4] = parameters[f"pair_{pair}"].numpy()
            pair += 1

        return individual


    def simulation(self, individual):

        y_hat, delta_D = individual_simulation(individual)
        y_hat = tf.convert_to_tensor(y_hat, dtype=tf.float32)
        y_hat = tf.constant(y_hat)
        return y_hat, delta_D



    def compute_cost_(self, y_hat, target):

        cost = tf.reduce_mean(tf.square(y_hat - target))

        return cost

    def gradient_optimization(self, individual):

        costs = []
        parameters = self.parameter_extraction(individual)
        self.target = tf.convert_to_tensor(self.target, dtype=tf.float32)
        print("this is params: ", parameters)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay
        )

        y_hat, dd = self.simulation(
            individual=individual
        )
        print("this is y_hat: ", y_hat)

        for i in range(self.epochs):
            variables = list(parameters.values())
            with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
                tape.watch(variables)
                cost = self.compute_cost_(
                    y_hat=y_hat,
                    target=self.target
                )
                print("this is cost:", cost)
                costs.append(cost.numpy())  # Store the cost value for each epoch

            variables = list(parameters.values())
            print("these are variables:", variables)
            gradients = tape.gradient(cost, variables)
            print("Gradients: ", gradients)
            optimizer.apply_gradients(zip(gradients, variables))

        individual = self.update_parameters(
            individual=individual,
            parameters=parameters
        )

        return individual, costs


t = np.zeros((50, 50))
t[:, 25:30] = 10
t[:, 20:25] = 5
t[:, 30:35] = 14

ind = np.zeros((7, 50, 50))
ind[1, :, 20:25] = 1
ind[3, :, 45:] = 1
ind[-1, -1, :5] = [2, 1, 500, 20, .4]
ind[-1, 0, :3] = [.9, .1, 6]
ind[-1, 2, :3] = [.9, .1, 8]
ind[-2, 0, :2] = [0, 2]
ind[-2, 1, :4] = [.6, .1, .1, 4]

g = GradientOptimization(
    epochs=100,
    learning_rate=0.01,
    target=t,
    cost_alpha=0.1,
    cost_beta=0.1,
    cost_kernel_size=3,
    weight_decay=0.01
)

inds, costs = g.gradient_optimization(ind)

print(costs)





