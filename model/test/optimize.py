from tensor_simulation import *
import os
import h5py
import time
import numpy as np





class AdamOptimization:
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

    def __init__(self,
                 target,
                 path,
                 file_name,
                 epochs=100,
                 learning_rate=None,
                 param_opt=False,
                 compartment_opt=True,
                 cost_alpha=0.6,
                 cost_beta=0.4,
                 max_val=1.0,
                 checkpoint_interval=10,
                 lr_decay=False,
                 decay_steps=40,
                 decay_rate=0.6,
                 trainable_compartment=1
                 ):

        self.epochs = epochs
        self.target = target
        self.path = path
        self.file_name = file_name
        self.param_opt = param_opt
        self.compartment_opt = compartment_opt
        self.cost_alpha = cost_alpha
        self.cost_beta = cost_beta
        self.max_val = max_val
        self.checkpoint_interval = checkpoint_interval
        self.lr_decay = lr_decay
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.trainable_compartment = trainable_compartment
        if learning_rate is None:
            learning_rate = [0.001]  # Default learning rate
        self.learning_rate = learning_rate

        """
        Initializes the GradientOptimization class with the specified parameters.
        """

    def save_to_h5py(self, dataset_name, data_array, store_path, file_name):
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

        if store_path:
            path = os.path.join(store_path, file_name)
            if not os.path.exists(store_path):
                os.makedirs(store_path)
        else:
            path = os.path.join(os.path.expanduser("~"), file_name)

        with h5py.File(path, 'a') as h5file:
            if dataset_name in h5file:
                del h5file[dataset_name]

            h5file.create_dataset(dataset_name, data=data_array)




    def parameter_extraction(self, individual, param_opt, compartment_opt, trainable_compartment):

        params = []
        num_species = int(individual[-1, -1, 0])
        num_pairs = int(individual[-1, -1, 1])
        max_epoch = int(individual[-1, -1, 2])
        stop = int(individual[-1, -1, 3])
        time_step = individual[-1, -1, 4]
        pair_start = int(num_species * 2)
        pair_stop = int(pair_start + (num_pairs * 2))

        for t in range(trainable_compartment):

            parameters = {}
            if param_opt:
                species = 1
                for i in range(0, num_species * 2, 2):
                    if int(species - 1) == t:
                        parameters[f"species_{species}"] = tf.Variable(
                            individual[0, i, 0:3],
                            name=f"species_{species}",
                            trainable=True
                        )
                        species += 1
                    else:
                        parameters[f"species_{species}"] = tf.Variable(
                            individual[0, i, 0:3],
                            name=f"species_{species}",
                            trainable=False
                        )
                        species += 1

                pair = 1
                for j in range(pair_start + 1, pair_stop + 1, 2):
                    parameters[f"pair_{pair}"] = tf.Variable(
                        individual[j, 1, :4],
                        name=f"pair_{pair}",
                        trainable=True
                    )
                    pair += 1

            else:
                species = 1
                for i in range(0, num_species * 2, 2):
                    parameters[f"species_{species}"] = tf.Variable(
                        individual[0, i, 0:3],
                        name=f"species_{species}",
                        trainable=False
                    )
                    species += 1

                pair = 1
                for j in range(pair_start + 1, pair_stop + 1, 2):
                    parameters[f"pair_{pair}"] = tf.Variable(
                        individual[j, 1, :4],
                        name=f"pair_{pair}",
                        trainable=False
                    )
                    pair += 1

            if compartment_opt:
                sp = 1
                for k in range(1, num_species * 2, 2):
                    if int(sp - 1) == t:
                        compartment = tf.Variable(
                            individual[k, :, :],
                            name=f"compartment_{sp}",
                            trainable=True
                        )
                        parameters[f'compartment_{sp}'] = compartment
                        sp += 1
                    else:
                        compartment = tf.Variable(
                            individual[k, :, :],
                            name=f"compartment_{sp}",
                            trainable=False
                        )
                        parameters[f'compartment_{sp}'] = compartment
                        sp += 1

            else:
                sp = 1
                for k in range(1, num_species * 2, 2):
                    compartment = tf.Variable(
                        individual[k, :, :],
                        name=f"compartment_{sp}",
                        trainable=False
                    )
                    parameters[f'compartment_{sp}'] = compartment
                    sp += 1

            params.append(parameters)

        if trainable_compartment < 1:
            parameters = {}
            if param_opt:
                species = 1
                for i in range(0, num_species * 2, 2):
                    parameters[f"species_{species}"] = tf.Variable(
                        individual[0, i, 0:3],
                        name=f"species_{species}",
                        trainable=True
                    )
                    species += 1

                pair = 1
                for j in range(pair_start + 1, pair_stop + 1, 2):
                    parameters[f"pair_{pair}"] = tf.Variable(
                        individual[j, 1, :4],
                        name=f"pair_{pair}",
                        trainable=True
                    )
                    pair += 1

                sp = 1
                for k in range(1, num_species * 2, 2):
                    compartment = tf.Variable(
                        individual[k, :, :],
                        name=f"compartment_{sp}",
                        trainable=False
                    )
                    parameters[f'compartment_{sp}'] = compartment
                    sp += 1

            else:
                species = 1
                for i in range(0, num_species * 2, 2):
                    parameters[f"species_{species}"] = tf.Variable(
                        individual[0, i, 0:3],
                        name=f"species_{species}",
                        trainable=False
                    )
                    species += 1

                pair = 1
                for j in range(pair_start + 1, pair_stop + 1, 2):
                    parameters[f"pair_{pair}"] = tf.Variable(
                        individual[j, 1, :4],
                        name=f"pair_{pair}",
                        trainable=False
                    )
                    pair += 1

                sp = 1
                for k in range(1, num_species * 2, 2):
                    compartment = tf.Variable(
                        individual[k, :, :],
                        name=f"compartment_{sp}",
                        trainable=False
                    )
                    parameters[f'compartment_{sp}'] = compartment
                    sp += 1

            params.append(parameters)
            print("extracted params:")
            print("--------------------------------------------")
            print(parameters)

        return params, num_species, num_pairs, max_epoch, stop, time_step




    def update_parameters(self, individual, parameters, param_opt, trainable_compartment):
        print("ind before update:")
        print("--------------------------------------------")
        print(individual[0])
        print(individual[1])
        print(individual[2])
        print(individual[3])
        print(individual[4])
        print(individual[5])
        print(individual[6])

        num_species = int(individual[-1, -1, 0])
        num_pairs = int(individual[-1, -1, 1])
        pair_start = int(num_species * 2)
        z, y, x = individual.shape

        if trainable_compartment < 1 and param_opt:

            j = 0
            for species in range(1, num_species + 1):
                individual = tf.tensor_scatter_nd_update(
                    individual,
                    indices=tf.constant([[z - 1, j, k] for k in range(3)], dtype=tf.int32),
                    updates=parameters[0][f"species_{species}"]
                )
                j += 2

            for pair in range(1, num_pairs + 1):
                j = pair_start + (pair - 1) * 2 + 1
                individual = tf.tensor_scatter_nd_update(
                    individual,
                    indices=tf.constant([[j, 1, k] for k in range(4)], dtype=tf.int32),
                    updates=parameters[0][f"pair_{pair}"]
                )

            for comp in range(1, num_species + 1):
                idx = int(((comp - 1) * 2) + 1)

                indices_ = []
                updates = tf.maximum(tf.reshape(parameters[0][f"compartment_{comp}"], [-1]), 0.0)
                for row in range(y):
                    for col in range(x):
                        indices_.append([idx, row, col])

                individual = tf.tensor_scatter_nd_update(
                    individual,
                    indices=indices_,
                    updates=updates
                )


        elif trainable_compartment >= 1:
            for i in range(len(parameters)):
                j = 0
                for species in range(1, num_species + 1):
                    if parameters[i][f"species_{species}"].trainable:
                        individual = tf.tensor_scatter_nd_update(
                            individual,
                            indices=tf.constant([[z - 1, j, k] for k in range(3)], dtype=tf.int32),
                            updates=parameters[i][f"species_{species}"]
                        )
                    j += 2

                for pair in range(1, num_pairs + 1):
                    if parameters[i][f"pair_{pair}"].trainable:
                        j = pair_start + (pair - 1) * 2 + 1
                        individual = tf.tensor_scatter_nd_update(
                            individual,
                            indices=tf.constant([[j, 1, k] for k in range(4)], dtype=tf.int32),
                            updates=parameters[i][f"pair_{pair}"]
                        )

                for comp in range(1, trainable_compartment + 1):
                    idx = int(((comp - 1) * 2) + 1)
                    if parameters[i][f"compartment_{comp}"].trainable:
                        indices_ = []
                        updates = tf.maximum(tf.reshape(parameters[i][f"compartment_{comp}"], [-1]), 0.0)
                        for row in range(y):
                            for col in range(x):
                                indices_.append([idx, row, col])

                        individual = tf.tensor_scatter_nd_update(
                            individual,
                            indices=indices_,
                            updates=updates
                        )
        print("ind after update:")
        print("--------------------------------------------")
        print(individual[0])
        print(individual[1])
        print(individual[2])
        print(individual[3])
        print(individual[4])
        print(individual[5])
        print(individual[6])

        return individual

    def simulation(self, individual, parameters, num_species, num_pairs, stop, time_step, max_epoch, compartment):
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
            max_epoch=max_epoch,
            compartment=compartment
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
        print("------------------------------------------")
        print("mse loss:")
        print(mse_loss)
        ssim_loss_value = self.ssim_loss(y_hat, target, max_val)
        print("ssim loss:")
        print(ssim_loss_value)
        total_loss = alpha * mse_loss + beta * ssim_loss_value
        print("total loss:")
        print(total_loss)
        return total_loss

    def ssim_loss(self, y_hat, target, max_val):
        """
        Compute the Structural Similarity Index (SSIM) loss between two matrices.

        SSIM is used to measure the perceptual similarity between two images or matrices.

        Parameters:
        - y_hat (tf.Tensor): A 2D tensor representing the predicted matrix or image. Shape: (y, x).
        - target (tf.Tensor): A 2D tensor representing the target matrix or image. Shape: (y, x).
        - max_val (float): The dynamic range of the input values, typically the maximum value of pixel intensity.

        Returns:
        - tf.Tensor: The SSIM loss, computed as `1 - SSIM score`.
        """
        y_hat = tf.expand_dims(y_hat, axis=-1)
        target = tf.expand_dims(target, axis=-1)
        ssim_score = tf.image.ssim(y_hat, target, max_val=max_val)

        return (1 - tf.reduce_mean(ssim_score)).numpy()



    def share_information(self, params):
        print("params before share:")
        print("_--------------------------------")
        print(params)
        for i in range(len(params)):
            current_dict = params[i]
            for j in range(len(params)):
                if i != j:
                    for key, val in current_dict.items():
                        if val.trainable:
                            if key in params[j] and not params[j][key].trainable:
                                params[j][key].assign(val)
        print("params after share:")
        print("_--------------------------------")
        print(params)

        return params


    def init_individual(self, individual):
        print("init ind:")
        print("-----------------------------")
        print("ind before init:")
        print(individual)
        num_species = int(individual[-1, -1, 0])
        num_pairs = int(individual[-1, -1, 1])
        pair_start = int(num_species * 2)
        pair_stop = int(pair_start + (num_pairs * 2))
        _, y, x = individual.shape

        for i in range(0, num_species * 2, 2):
            update = tf.zeros((y, x))
            indices = tf.constant([[i]])
            individual = tf.tensor_scatter_nd_update(individual, indices, [update])

        for j in range(pair_start, pair_stop, 2):
            update = tf.zeros((y, x))
            indices = tf.constant([[j]])
            individual = tf.tensor_scatter_nd_update(individual, indices, [update])
        print("ind after init:")
        print(individual)

        return individual



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
        time_ = []
        results = np.zeros((self.trainable_compartment, self.epochs, self.target.shape[1], self.target.shape[2]))

        self.save_to_h5py(
            dataset_name="target",
            data_array=self.target,
            store_path=self.path,
            file_name=self.file_name
        )

        parameters, num_species, num_pairs, max_epoch, stop, time_step = self.parameter_extraction(
            individual=individual,
            param_opt=self.param_opt,
            compartment_opt=self.compartment_opt,
            trainable_compartment=self.trainable_compartment
        )

        def create_optimizer(lr):
            
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=lr,
                decay_steps=self.decay_steps,
                decay_rate=self.decay_rate
            )
            if self.lr_decay:
                optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
            else:
                optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
                  
            return optimizer
            

        if len(self.learning_rate) > 1:
            optimizers = [create_optimizer(self.learning_rate[i]) for i in range(len(parameters))]
        else:
            optimizers = [create_optimizer(self.learning_rate[0]) for _ in range(len(parameters))]

        tic_ = time.time()
        tic = time.time()
        for i in range(1, self.epochs + 1):
            cost_ = []
            for j in range(len(parameters)):
                optimizer = optimizers[j]

                with tf.GradientTape() as tape:
                    y_hat = self.simulation(
                        individual=individual,
                        parameters=parameters[j],
                        num_species=num_species,
                        num_pairs=num_pairs,
                        stop=stop,
                        time_step=time_step,
                        max_epoch=max_epoch,
                        compartment=j
                    )

                    cost = self.compute_cost_(
                        y_hat=y_hat,
                        target=self.target[j, :, :],
                        alpha=self.cost_alpha,
                        beta=self.cost_beta,
                        max_val=self.max_val
                    )

                    cost_.append(cost.numpy())
                print(f"Epoch {i}/{self.epochs}, Optimizer {j + 1}, Cost: {cost.numpy()}")

                variables = list(parameters[j].values())
                gradients = tape.gradient(cost, variables)
                # gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients]
                print("grads:")
                print("------------------------------------------------------")
                print(gradients)
                optimizer.apply_gradients(zip(gradients, variables))
                results[j, i - 1, :, :] = y_hat.numpy()
                individual = self.init_individual(individual=individual)



            parameters = self.share_information(params=parameters)

            costs.append(cost_)
            if i % self.checkpoint_interval == 0:
                toc = time.time()
                time_.append(toc - tic)
                tic = time.time()
                individual = self.update_parameters(
                    individual=individual,
                    parameters=parameters,
                    param_opt=self.param_opt,
                    trainable_compartment=self.trainable_compartment

                )

                self.save_to_h5py(
                    dataset_name="ind",
                    data_array=individual,
                    store_path=self.path,
                    file_name=self.file_name
                )
                self.save_to_h5py(
                    dataset_name="cost",
                    data_array=np.array(costs),
                    store_path=self.path,
                    file_name=self.file_name
                )
                self.save_to_h5py(
                    dataset_name="time",
                    data_array=np.array(time_),
                    store_path=self.path,
                    file_name=self.file_name
                )
                self.save_to_h5py(
                    dataset_name="results",
                    data_array=results,
                    store_path=self.path,
                    file_name=self.file_name
                )

        toc_ = time.time()
        time_.append(toc_ - tic_)
        self.save_to_h5py(
            dataset_name="time",
            data_array=np.array(time_),
            store_path=self.path,
            file_name=self.file_name
        )

        individual = self.update_parameters(
            individual=individual,
            parameters=parameters,
            param_opt=self.param_opt,
            trainable_compartment=self.trainable_compartment

        )
        self.save_to_h5py(
            dataset_name="ind",
            data_array=individual,
            store_path=self.path,
            file_name=self.file_name
        )
        self.save_to_h5py(
            dataset_name="cost",
            data_array=np.array(costs),
            store_path=self.path,
            file_name=self.file_name
        )
        self.save_to_h5py(
            dataset_name="results",
            data_array=results,
            store_path=self.path,
            file_name=self.file_name
        )

        return individual, costs


