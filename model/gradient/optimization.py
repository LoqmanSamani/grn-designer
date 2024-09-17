from ..sim.sim_tensor import tensor_simulation
import tensorflow as tf

class GradientOptimization:
    """
    A class for performing gradient-based optimization using the Adam optimizer.
    This class is designed to optimize the parameters of species and pair interactions
    in a biological simulation model.

    Attributes:
        - epochs (int): The number of epochs for the optimization process.
        - learning_rate (float): The learning rate for the Adam optimizer.
        - target (tf.Tensor): The target tensor representing the desired diffusion pattern.
        - param_opt (bool): if True, the species and complex parameters will be optimized.
        - compartment_opt (bool): if True, the initial condition of each species will be optimized.
        - cost_alpha (float): Weighting factor for the cost function (currently unused).
        - cost_beta (float): Weighting factor for the cost function (currently unused).
        - cost_kernel_size (int): Size of the kernel used in the cost function (currently unused).
        - weight_decay (float): The weight decay (regularization) factor for the Adam optimizer.
    """

    def __init__(self, epochs, learning_rate, target, param_opt, compartment_opt, cost_alpha, cost_beta, max_val,
                 cost_kernel_size, weight_decay):

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.target = target
        self.param_opt = param_opt
        self.compartment_opt = compartment_opt
        self.cost_alpha = cost_alpha
        self.cost_beta = cost_beta
        self.max_val = max_val
        self.cost_kernel_size = cost_kernel_size
        self.weight_decay = weight_decay
        """
        Initializes the GradientOptimization class with the specified parameters.
        """

    def parameter_extraction(self, individual, compartment_opt):
        """
        Extracts the parameters of species, pairs and initial condition compartments from the given individual tensor.

        Args:
            - individual (tf.Tensor): A tensor representing an individual in the population.
            - param_opt (bool): if True, the species and complex parameters will be extracted.
            - compartment_opt (bool): if True, the initial condition of each species will be extracted.

        Returns:
            - tuple: A tuple containing:
                - parameters (dict): A dictionary of trainable parameters for species and pairs.
                - num_species (int): The number of species in the individual.
                - num_pairs (int): The number of pair interactions in the individual.
                - max_epoch (int): The maximum number of epochs for the simulation.
                - stop (int): The stop time for the simulation.
                - time_step (float): The time step for the simulation.
        """

        parameters = {}
        num_species = int(individual[-1, -1, 0])
        num_pairs = int(individual[-1, -1, 1])
        max_epoch = int(individual[-1, -1, 2])
        stop = int(individual[-1, -1, 3])
        time_step = individual[-1, -1, 4]
        pair_start = int(num_species * 2)
        pair_stop = int(pair_start + (num_pairs * 2))

        species = 1
        for i in range(0, num_species * 2, 2):
            parameters[f"species_{species}"] = tf.Variable(individual[-1, i, 0:3], trainable=True)
            species += 1

        pair = 1
        for j in range(pair_start + 1, pair_stop + 1, 2):
            parameters[f"pair_{pair}"] = tf.Variable(individual[j, 1, :4], trainable=True)
            pair += 1

        if compartment_opt:
            sp = 1
            for k in range(1, num_species * 2, 2):
                compartment = tf.Variable(individual[k, :, :], trainable=True)
                parameters[f'compartment_{sp}'] = compartment
                sp += 1

        return parameters, num_species, num_pairs, max_epoch, stop, time_step



    def update_parameters(self, individual, parameters, param_opt, compartment_opt):
        """
        Updates the parameters of species and pairs in the individual tensor after optimization.

        Args:
            - individual (tf.Tensor): The original individual tensor.
            - parameters (dict): A dictionary of updated parameters for species and pairs.
            - param_opt (bool): if True, the species and complex parameters will be extracted.
            - compartment_opt (bool): if True, the initial condition of each species will be extracted.

        Returns:
            - tf.Tensor: The updated individual tensor with optimized parameters.
        """

        num_species = int(individual[-1, -1, 0])
        num_pairs = int(individual[-1, -1, 1])
        pair_start = int(num_species * 2)

        # Update species and pairs parameters
        if param_opt:
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

        # Update initial conditions
        if compartment_opt:
            sp = 1
            for i in range(1, num_species * 2, 2):
                indices_ = []
                updates = tf.reshape(parameters[f"compartment_{sp}"], [-1])
                for row in range(individual[0, :, :].shape[0]):
                    for col in range(individual[0, :, :].shape[1]):
                        indices_.append([i, row, col])

                individual = tf.tensor_scatter_nd_update(
                    individual,
                    indices=indices_,
                    updates=updates
                )
                sp += 1

        return individual

    def simulation(self, individual, parameters, num_species, num_pairs, stop, time_step, max_epoch):
        """
        Runs a simulation using the given individual and parameters.

        Args:
            - individual (tf.Tensor): The individual tensor representing the system configuration.
            - parameters (dict): A dictionary of parameters for species and pairs.
            - num_species (int): The number of species in the simulation.
            - num_pairs (int): The number of pair interactions in the simulation.
            - stop (int): The stop time for the simulation.
            - time_step (float): The time step for the simulation.
            - max_epoch (int): The maximum number of epochs for the simulation.

        Returns:
            - tf.Tensor: The simulated output (y_hat) representing the diffusion pattern.
        """

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

    def compute_cost_(self, y_hat, target, alpha, beta, max_val):

        """
        Computes the cost (loss) between the simulated output and the target.

        Args:
            - y_hat (tf.Tensor): The simulated output tensor.
            - target (tf.Tensor): The target tensor representing the desired diffusion pattern.

        Returns:
            - tf.Tensor: The computed cost (loss) value.
        """
        mse_loss = tf.reduce_mean(tf.square(y_hat - target))
        ssim_loss_value = self.ssim_loss(y_hat, target, max_val)
        total_loss = alpha * mse_loss + beta * ssim_loss_value

        return total_loss

    def ssim_loss(self, y_hat, target, max_val):
        """
        Compute the Structural Similarity Index (SSIM) loss between two matrices.

        SSIM is used to measure the perceptual similarity between two images or matrices. A higher SSIM score indicates
        higher similarity. The SSIM loss is calculated as `1 - SSIM score`, so a lower SSIM loss indicates more perceptual
        similarity.

        Parameters:
        - y_hat (tf.Tensor): A 2D tensor representing the predicted matrix or image. Shape: (y, x).
        - target (tf.Tensor): A 2D tensor representing the target matrix or image. Shape: (y, x).
        - max_val (float, optional): The dynamic range of the input values, typically the maximum value of the pixel
          intensity. Default is 1.0.

        Returns:
        - float: The SSIM loss, computed as `1 - SSIM score`, where the SSIM score is between 0 and 1. A lower
          loss indicates more perceptual similarity between `y_hat` and `target`.
        """
        y_hat = tf.expand_dims(y_hat, axis=-1)
        target = tf.expand_dims(target, axis=-1)
        ssim_score = tf.image.ssim(y_hat, target, max_val=max_val)

        return (1 - tf.reduce_mean(ssim_score)).numpy()





    def gradient_optimization(self, individual):
        """
        Performs gradient-based optimization on the individual using the Adam optimizer.

        Args:
            - individual (tf.Tensor): The individual tensor representing the initial configuration.

        Returns:
            - tuple: A tuple containing:
                - individual (tf.Tensor): The updated individual tensor after optimization.
                - costs (list): A list of cost values recorded during the optimization process.
        """

        costs = []
        parameters, num_species, num_pairs, max_epoch, stop, time_step = self.parameter_extraction(
            individual=individual,
            compartment_opt=self.compartment_opt
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
                    target=self.target,
                    alpha=self.cost_alpha,
                    beta=self.cost_beta,
                    max_val=self.max_val
                )

                costs.append(cost.numpy())

            print(f"Epoch {i + 1}/{self.epochs}, Cost: {cost.numpy()}")
            variables = list(parameters.values())
            gradients = tape.gradient(cost, variables)
            optimizer.apply_gradients(zip(gradients, variables))

        individual = self.update_parameters(
            individual=individual,
            parameters=parameters,
            param_opt=self.param_opt,
            compartment_opt=self.compartment_opt
        )

        return individual, costs




