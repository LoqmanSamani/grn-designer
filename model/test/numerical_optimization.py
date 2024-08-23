from simulation import *
import numpy as np

class NumericalOptimization:

    def __init__(self, target, epochs, epsilon, learning_rate):

        self.target = target
        self.epochs = epochs
        self.epsilon = epsilon
        self.learning_rate = learning_rate

    def parameter_extraction(self, individual):

        parameters = {}
        num_species = int(individual[-1, -1, 0])
        num_pairs = int(individual[-1, -1, 1])
        pair_start = int(num_species * 2)
        pair_stop = int(pair_start + (num_pairs * 2))

        species = 1
        for i in range(0, num_species*2, 2):
            parameters[f"species_{species}"] = np.array(individual[-1, i, 0:3])
            species += 1

        pair = 1
        for j in range(pair_start+1, pair_stop+1, 2):
            parameters[f"pair_{pair}"] = np.array(individual[j, 1, :4])
            pair += 1

        return parameters

    def update_parameters(self, individual, parameters):

        num_species = int(individual[-1, -1, 0])
        num_pairs = int(individual[-1, -1, 1])
        pair_start = int(num_species * 2)
        pair_stop = int(pair_start + (num_pairs * 2))

        species = 1
        for i in range(0, num_species * 2, 2):
            individual[-1, i, :3] = parameters[f"species_{species}"]
            species += 1

        pair = 1
        for j in range(pair_start + 1, pair_stop + 1, 2):
            individual[j, 1, :4] = parameters[f"pair_{pair}"]
            pair += 1

        return individual


    def simulation(self, individual):

        y_hat, delta_D = individual_simulation(individual)

        return y_hat, delta_D

    def compute_cost_(self, y_hat, target):

        y, x = y_hat.shape
        cost = np.mean((target - y_hat) ** 2) / (y * x)

        return cost


    def compute_numerical_gradient(self, individual, target, parameters, epsilon):

        gradients = {}

        for param_name, param in parameters.items():

            original_value = param
            gradient = np.zeros_like(original_value)

            # Compute gradient for each element
            for i in range(original_value.size):
                param_flat = original_value.flatten()

                # Perturb the parameter positively
                param_flat[i] += epsilon
                perturbed_value = param_flat.reshape(original_value.shape)
                param = perturbed_value
                individual = self.update_parameters(
                    individual=individual,
                    parameters=parameters
                )
                y_hat, dd = self.simulation(
                    individual=individual
                )
                cost_plus = self.compute_cost_(
                    y_hat=y_hat,
                    target=target
                )

                # Perturb the parameter negatively
                param_flat[i] -= 2 * epsilon
                perturbed_value = param_flat.reshape(original_value.shape)
                param = perturbed_value
                individual = self.update_parameters(
                    individual=individual,
                    parameters=parameters
                )
                y_hat, dd = self.simulation(
                    individual=individual
                )
                cost_minus = self.compute_cost_(
                    y_hat=y_hat,
                    target=target
                )

                # Calculate gradient
                gradient.flat[i] = (cost_plus - cost_minus) / (2 * epsilon)

                # Restore original parameter value
                param = original_value

            gradients[param_name] = np.array(gradient)

        return gradients


    def update_parameters_(self, gradients, parameters):
        updated_parameters = {}

        for param_name, param in parameters.items():

            param_value = param
            gradient = gradients[param_name]
            updated_param_value = param_value - self.learning_rate * gradient

            updated_parameters[param_name] = updated_param_value

        return updated_parameters


    def optimize(self, individual):

        costs = []
        for epoch in range(self.epochs):

            parameters = self.parameter_extraction(
                individual=individual
            )
            gradients = self.compute_numerical_gradient(
                individual=individual,
                target=self.target,
                parameters=parameters,
                epsilon=self.epsilon
            )

            updated_parameters = self.update_parameters_(
                gradients=gradients,
                parameters=parameters
            )

            individual = self.update_parameters(
                individual=individual,
                parameters=updated_parameters
            )


            y_hat, dd = self.simulation(
                individual=individual
            )
            cost = self.compute_cost_(
                y_hat=y_hat,
                target=self.target
            )
            costs.append(cost)
            print(f'Epoch {epoch+1}/{self.epochs}, Cost: {cost}')

        return individual, costs


t = np.zeros((50, 50))
t[:, 25:30] = .1
t[:, 20:25] = .01
t[:, 30:35] = .3

ind = np.zeros((7, 50, 50))
ind[1, :, 20:25] = 1
ind[3, :, 45:] = 1
ind[-1, -1, :5] = [2, 1, 500, 20, .4]
ind[-1, 0, :3] = [.9, .1, 6]
ind[-1, 2, :3] = [.9, .1, 8]
ind[-2, 0, :2] = [0, 2]
ind[-2, 1, :4] = [.6, .1, .1, 4]



g = NumericalOptimization(
    epochs=30,
    learning_rate=0.01,
    target=t,
    epsilon=1e-7
)

inds, costs = g.optimize(ind)
