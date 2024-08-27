from tensor_simulation import *




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

        num_species = int(individual[-1, -1, 0])
        num_pairs = int(individual[-1, -1, 1])
        max_epoch = int(individual[-1, -1, 2])
        stop = int(individual[-1, -1, 3])
        time_step = individual[-1, -1, 4]

        return parameters, num_species, num_pairs, max_epoch, stop, time_step


    def update_parameters(self, individual, parameters):

        num_species = int(individual[-1, -1, 0].numpy())
        num_pairs = int(individual[-1, -1, 1].numpy())
        pair_start = int(num_species * 2)

        # Update species parameters
        for species in range(1, num_species + 1):

            i = (species - 1) * 2
            individual = tf.tensor_scatter_nd_update(
                individual,
                indices=tf.constant([[individual.shape[0] - 1, i, k] for k in range(3)], dtype=tf.int32),
                updates=parameters[f"species_{species}"]
            )

        # Update pair parameters
        for pair in range(1, num_pairs + 1):
            j = pair_start + (pair - 1) * 2 + 1
            individual = tf.tensor_scatter_nd_update(
                individual,
                indices=tf.constant([[j, 1, k] for k in range(4)], dtype=tf.int32),
                updates=parameters[f"pair_{pair}"]
            )

        return individual

    def simulation(self, individual, parameters, num_species, num_pairs, stop, time_step, max_epoch):

        y_hat = tensor_simulation(
            individual=individual,
            parameters=parameters,
            num_species=num_species,
            num_pairs=num_pairs,
            stop=stop,
            time_step=time_step,
            max_epoch=max_epoch
        )

        return y_hat


    def compute_cost_(self, y_hat, target):

        cost = tf.reduce_mean(tf.square(y_hat - target))

        return cost

    def gradient_optimization(self, individual):

        costs = []
        parameters, num_species, num_pairs, max_epoch, stop, time_step = self.parameter_extraction(
            individual=individual
        )
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay
        )

        for i in range(self.epochs):
            with tf.GradientTape() as tape:

                y_hat = self.simulation(
                    individual=individual,
                    parameters=parameters,
                    num_species=num_species,
                    num_pairs=num_pairs,
                    stop=stop,
                    time_step=time_step,
                    max_epoch=max_epoch
                )

                cost = self.compute_cost_(
                    y_hat=y_hat,
                    target=self.target
                )

                costs.append(cost.numpy())

            variables = list(parameters.values())
            gradients = tape.gradient(cost, variables)
            print(gradients)
            optimizer.apply_gradients(zip(gradients, variables))

        individual = self.update_parameters(
            individual=individual,
            parameters=parameters
        )

        return individual, costs




