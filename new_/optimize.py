from tensor_simulation import *
import os
import h5py
import time
import numpy as np
import torch
import torch.nn.functional as F
from ignite.metrics import SSIM
#torch.autograd.set_detect_anomaly(True)






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
                 param_type="all",
                 compartment_opt=True,
                 cost_alpha=0.6,
                 cost_beta=0.4,
                 max_val=1.0,
                 checkpoint_interval=10,
                 share_info=4,
                 lr_decay=False,
                 decay_steps=40,
                 decay_rate=0.6,
                 trainable_compartment=1,
                 accumulation_steps=None,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                 ):

        self.epochs = epochs
        self.target = target
        self.path = path
        self.file_name = file_name
        self.param_opt = param_opt
        self.param_type = param_type
        self.compartment_opt = compartment_opt
        self.cost_alpha = cost_alpha
        self.cost_beta = cost_beta
        self.max_val = max_val
        self.checkpoint_interval = checkpoint_interval
        self.share_info = share_info
        self.lr_decay = lr_decay
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.trainable_compartment = trainable_compartment
        #self.accumulation_steps = accumulation_steps,
        self.device = device
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





    def parameter_extraction(self, agent, param_type, compartment_opt, trainable_compartment):

        params = []
        num_species = int(agent[-1, -1, 0])
        num_pairs = int(agent[-1, -1, 1])
        max_epoch = int(agent[-1, -1, 2])
        stop = int(agent[-1, -1, 3])
        time_step = agent[-1, -1, 4]
        pair_start = int(num_species * 2)
        pair_stop = int(pair_start + (num_pairs * 2))

        for t in range(trainable_compartment):

            parameters = {}
            if param_type == "not_all":
                species = 1
                for i in range(0, num_species * 2, 2):
                    if int(species - 1) == t:
                        parameters[f"species_{species}"] = torch.nn.Parameter(
                            agent[-1, i, 0:3].clone().float(),
                            requires_grad=True
                        )
                        species += 1
                    else:
                        parameters[f"species_{species}"] = torch.nn.Parameter(
                            agent[-1, i, 0:3].clone().detach().float(),
                            requires_grad=False
                        )
                        species += 1

                pair = 1
                for j in range(pair_start + 1, pair_stop + 1, 2):
                    parameters[f"pair_{pair}"] = torch.nn.Parameter(
                        agent[j, 1, :4].clone.float(),
                        requires_grad=True
                    )
                    pair += 1

            elif param_type == "all":
                species = 1
                for i in range(0, num_species * 2, 2):

                    parameters[f"species_{species}"] = torch.nn.Parameter(
                        agent[-1, i, 0:3].clone().float(),
                        requires_grad=True
                    )
                    species += 1

                pair = 1
                for j in range(pair_start + 1, pair_stop + 1, 2):
                    parameters[f"pair_{pair}"] = torch.nn.Parameter(
                        agent[j, 1, :4].clone().float(),
                        requires_grad=True
                    )
                    pair += 1

            else:
                species = 1
                for i in range(0, num_species * 2, 2):
                    parameters[f"species_{species}"] = torch.nn.Parameter(
                        agent[-1, i, 0:3].clone().detach().float(),
                        requires_grad=False
                    )
                    species += 1

                pair = 1
                for j in range(pair_start + 1, pair_stop + 1, 2):
                    parameters[f"pair_{pair}"] = torch.nn.Parameter(
                        agent[j, 1, :4].clone().detach().float(),
                        requires_grad=False
                    )
                    pair += 1


            if compartment_opt:
                sp = 1
                for k in range(1, num_species * 2, 2):
                    if int(sp - 1) == t:
                        parameters[f'compartment_{sp}'] = torch.nn.Parameter(
                            agent[k, :, :].clone().float(),
                            requires_grad=True
                        )
                        sp += 1
                    else:
                        parameters[f'compartment_{sp}'] = torch.nn.Parameter(
                            agent[k, :, :].clone().detach().float(),
                            requires_grad=False
                        )
                        sp += 1

            else:
                sp = 1
                for k in range(1, num_species * 2, 2):
                    parameters[f'compartment_{sp}'] = torch.nn.Parameter(
                        agent[k, :, :].clone().detach().float(),
                        requires_grad=False
                    )
                    sp += 1

            params.append(parameters)

        if trainable_compartment == 0:
            parameters = {}
            if param_type == "all":
                species = 1
                for i in range(0, num_species * 2, 2):
                    parameters[f"species_{species}"] = torch.nn.Parameter(
                        agent[-1, i, 0:3].clone().float(),
                        requires_grad=True
                    )
                    species += 1

                pair = 1
                for j in range(pair_start + 1, pair_stop + 1, 2):
                    parameters[f"pair_{pair}"] = torch.nn.Parameter(
                        agent[j, 1, :4].clone().float(),
                        requires_grad=True
                    )
                    pair += 1

                sp = 1
                for k in range(1, num_species * 2, 2):
                    parameters[f'compartment_{sp}'] = torch.nn.Parameter(
                        agent[k, :, :].clone().detach().float(),
                        requires_grad=False
                    )
                    sp += 1

            elif param_type == "not_all":
                species = 1
                for i in range(0, num_species * 2, 2):
                    if species == 1:
                        parameters[f"species_{species}"] = torch.nn.Parameter(
                            agent[-1, i, 0:3].clone().float(),
                            requires_grad=True
                        )
                        species += 1
                    else:
                        parameters[f"species_{species}"] = torch.nn.Parameter(
                            agent[-1, i, 0:3].clone().detach().float(),
                            requires_grad=False
                        )
                        species += 1


                pair = 1
                for j in range(pair_start + 1, pair_stop + 1, 2):
                    parameters[f"pair_{pair}"] = torch.nn.Parameter(
                        agent[j, 1, :4].clone().float(),
                        requires_grad=True
                    )
                    pair += 1

                sp = 1
                for k in range(1, num_species * 2, 2):
                    parameters[f'compartment_{sp}'] = torch.nn.Parameter(
                        agent[k, :, :].clone().detach().float(),
                        requires_grad=False
                    )
                    sp += 1

            else:
                species = 1
                for i in range(0, num_species * 2, 2):
                    parameters[f"species_{species}"] = torch.nn.Parameter(
                        agent[-1, i, 0:3].clone().detach().float,
                        requires_grad=False
                    )
                    species += 1

                pair = 1
                for j in range(pair_start + 1, pair_stop + 1, 2):
                    parameters[f"pair_{pair}"] = torch.nn.Parameter(
                        agent[j, 1, :4].clone().detach().float(),
                        requires_grad=False
                    )
                    pair += 1

                sp = 1
                for k in range(1, num_species * 2, 2):
                    parameters[f'compartment_{sp}'] = torch.nn.Parameter(
                        agent[k, :, :].clone().detach().float(),
                        requires_grad=False
                    )
                    sp += 1

            params.append(parameters)

        return params, num_species, num_pairs, max_epoch, stop, time_step





    def update_parameters(self, agent, parameters, param_opt, trainable_compartment):
        
        num_species = int(agent[-1, -1, 0])
        num_pairs = int(agent[-1, -1, 1])
        pair_start = int(num_species * 2)

        if trainable_compartment < 1 and param_opt:
            j = 0
            for species in range(1, num_species + 1):
                agent[-1, j, :3] = parameters[0][f"species_{species}"].detach().clone()  
                j += 2

            for pair in range(1, num_pairs + 1):
                j = pair_start + (pair - 1) * 2 + 1
                agent[j, 1, :4] = parameters[0][f"pair_{pair}"].detach().clone()  

            for comp in range(1, num_species + 1):
                idx = int(((comp - 1) * 2) + 1)
                updates = torch.max(parameters[0][f"compartment_{comp}"], torch.tensor(0.0))
                agent[idx, :, :] = updates.detach().clone() 

        elif trainable_compartment >= 1:
            for i in range(len(parameters)):
                j = 0
                for species in range(1, num_species + 1):
                    if parameters[i][f"species_{species}"].requires_grad:
                        agent[-1, j, :3] = parameters[i][f"species_{species}"].detach().clone()  
                    j += 2

                for pair in range(1, num_pairs + 1):
                    if parameters[i][f"pair_{pair}"].requires_grad:
                        j = pair_start + (pair - 1) * 2 + 1
                        agent[j, 1, :4] = parameters[i][f"pair_{pair}"].detach().clone() 

                for comp in range(1, trainable_compartment + 1):
                    idx = int(((comp - 1) * 2) + 1)
                    if parameters[i][f"compartment_{comp}"].requires_grad:
                        updates = torch.max(parameters[i][f"compartment_{comp}"], torch.tensor(0.0))
                        agent[idx, :, :] = updates.detach().clone()

        return agent




   

    def simulation(self, agent, parameters, num_species, num_pairs, stop, time_step, max_epoch, compartment, device):
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

        y_hat = tensor_simulation(agent=agent, parameters=parameters, num_species=num_species, num_pairs=num_pairs,
                                  stop=stop, time_step=time_step, max_epoch=max_epoch, compartment=compartment,
                                  device=device)

        return y_hat

    def compute_cost_(self, y_hat, target, alpha, beta, max_val):
        """
        Computes the cost (loss) between the simulated output and the target using MSE and SSIM loss.

        Args:
            - y_hat (torch.Tensor): The simulated output tensor.
            - target (torch.Tensor): The target tensor representing the desired diffusion pattern.
            - alpha (float): Weight for the MSE loss.
            - beta (float): Weight for the SSIM loss.
            - max_val (float): The dynamic range of the input values, typically the maximum pixel intensity.

        Returns:
            - torch.Tensor: The computed cost (loss) value.
        """
        mse_loss = F.mse_loss(y_hat, target)
        ssim_loss_value = self.ssim_loss(y_hat, target, max_val)
        total_loss = alpha * mse_loss + beta * ssim_loss_value

        return total_loss

    def ssim_loss(self, y_hat, target, max_val):
        """
        Compute the Structural Similarity Index (SSIM) loss between two matrices using PyTorch.

        Parameters:
        - y_hat (torch.Tensor): A 2D tensor representing the predicted matrix or image. Shape: (y, x).
        - target (torch.Tensor): A 2D tensor representing the target matrix or image. Shape: (y, x).
        - max_val (float): The dynamic range of the input values, typically the maximum value of pixel intensity.

        Returns:
        - torch.Tensor: The SSIM loss, computed as `1 - SSIM score`.
        """
        ssim_metric = SSIM(data_range=max_val)
        if y_hat.dim() == 2:
            y_hat = y_hat.unsqueeze(0).unsqueeze(0)
        if target.dim() == 2:
            target = target.unsqueeze(0).unsqueeze(0)

        ssim_metric.update((y_hat, target))
        ssim_score = ssim_metric.compute()
        ssim_metric.reset()

        return 1 - ssim_score




    def share_information(self, params):
        """
        Share the values of trainable parameters between dictionaries of parameters.

        Args:
            - params (list of dicts): A list where each element is a dictionary of parameters (tensors).
                                      Each tensor in the dictionary can either be trainable or not.

        Returns:
            - params (list of dicts): Updated list where trainable "pair_n" tensors are updated with their sums,
                                      and non-trainable tensors are updated with values from trainable ones.
        """

        pair_sums = {}

        for current_dict in params:
            for key, val in current_dict.items():
                if "pair_" in key and val.requires_grad:
                    if key not in pair_sums:
                        pair_sums[key] = val.detach().clone()
                    else:
                        pair_sums[key] += val.detach()

        for i, current_dict in enumerate(params):
            for key, val in current_dict.items():
                if val.requires_grad:
                    for j in range(len(params)):
                        if i != j and key in params[j] and not params[j][key].requires_grad:
                            with torch.no_grad():
                                params[j][key].copy_(val)

                if key in pair_sums and val.requires_grad:
                    with torch.no_grad():
                        val.copy_(pair_sums[key]/len(params))

        return params





    """
    def init_individual(self, individual):

        print("init ind:")
        print("-----------------------------")
        print("ind before init:")
        print(individual[0][:])
        print(individual[2][:])
        print(individual[4][:])


        num_species = int(individual[-1, -1, 0])
        num_pairs = int(individual[-1, -1, 1])
        pair_start = int(num_species * 2)
        pair_stop = int(pair_start + (num_pairs * 2))
        _, y, x = individual.shape

        # Set species indices to zero
        for i in range(0, num_species * 2, 2):
            individual[i] = torch.zeros((y, x), dtype=individual.dtype)

        # Set pair indices to zero
        for j in range(pair_start, pair_stop, 2):

            individual[j] = torch.zeros((y, x), dtype=individual.dtype)


        print("ind after init:")
        print(individual[0][:])
        print(individual[2][:])
        print(individual[4][:])

        return individual
    """



    def create_optimizer(self, model_parameters, lr):
        """
        Creates an Adam optimizer with optional exponential learning rate decay.

        Args:
            - model_parameters: The parameters of the model to optimize.
            - lr (float): The initial learning rate.

        Returns:
            - optimizer: The created Adam optimizer.
            - lr_scheduler: The learning rate scheduler if decay is enabled, otherwise None.
        """
        optimizer = torch.optim.Adam(params=model_parameters, lr=lr)

        lr_scheduler = None
        if self.lr_decay:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=self.decay_rate
            )

        return optimizer, lr_scheduler



    def gradient_optimization(self, agent):
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
        # create arrays to save the progress of the system
        simulation_results = np.zeros((self.trainable_compartment, self.epochs, self.target.shape[1], self.target.shape[2]))
        init_conditions = np.zeros((self.trainable_compartment, self.epochs, self.target.shape[1], self.target.shape[2]))

        self.save_to_h5py(
            dataset_name="target",
            data_array=self.target,
            store_path=self.path,
            file_name=self.file_name
        )
        # extract parameters of the model to be optimized
        parameters, num_species, num_pairs, max_epoch, stop, time_step = self.parameter_extraction(agent=agent,
                                                                                                   param_type=self.param_type,
                                                                                                   compartment_opt=self.compartment_opt,
                                                                                                   trainable_compartment=self.trainable_compartment)


        # create the needed  optimizers (number of optimizer is equal to the number of patterns in the input pattern-image)
        if len(self.learning_rate) > 1:
            optimizers = [self.create_optimizer(list(parameters[i].values()), self.learning_rate[i]) for i in range(len(parameters))]
        else:
            optimizers = [self.create_optimizer(list(parameters[i].values()), self.learning_rate[0]) for i in range(len(parameters))]

        tic_ = time.time()
        tic = time.time()
        for i in range(1, self.epochs + 1):
            cost_ = []
            for j in range(len(parameters)):
                #optimizer = optimizers[j]

                # Zero the gradient
                #optimizers[j][0].zero_grad(set_to_none=True)
                # Enable gradient tracking on the parameters for the current optimizer
                #for param in parameters[j].values():
                    #param.requires_grad = True

                # Forward pass: simulate the output
                prediction = self.simulation(agent=agent, parameters=parameters[j], num_species=num_species,
                                             num_pairs=num_pairs, stop=stop, time_step=time_step, max_epoch=max_epoch,
                                             compartment=j, device=self.device)

                # Compute cost
                cost = self.compute_cost_(
                    y_hat=prediction,
                    target=self.target[j, :, :],
                    alpha=self.cost_alpha,
                    beta=self.cost_beta,
                    max_val=self.max_val
                )
                # Backward pass: compute gradients
                cost.backward(retain_graph=True) # backward pass
                #if self.accumulation_steps:
                    #if i % self.accumulation_steps == 0:  # Wait for several backward steps
                optimizers[j][0].step()  # Now we can do an optimizer step
                if optimizers[j][1] is not None:
                    optimizers[j][1].step()  # Update learning rate according to the schedule

                optimizers[j][0].zero_grad(set_to_none=True) # Reset gradients tensors
                #else:
                    #optimizers[j][0].step() #gradient descent
                    # Optionally, step the scheduler once per epoch (or per iteration)
                    #if optimizers[j][1] is not None:
                        #optimizers[j][1].step()  # Update learning rate according to the schedule
                    #optimizers[j][0].zero_grad(set_to_none=True)  # Reset gradients tensors

                #print(parameters[j][f"species_{j + 1}"].detach().numpy())
                cost_.append(cost.item())
                simulation_results[j, i - 1, :, :] = prediction.detach()
                init_conditions[j, i - 1, :, :] = parameters[j][f"compartment_{j + 1}"].detach().numpy()
                # individual = self.init_individual(individual=individual) # initialize individual

                # print the cost
                if len(optimizers) > 1:
                    print(f"Epoch {i}/{self.epochs}, Optimizer {j + 1}, Cost: {cost.item()}")
                else:
                    print(f"Iteration {i}/{self.epochs}, Cost: {cost.item()}")

            costs.append(cost_)

            if i % self.share_info == 0 and len(optimizers) > 1:
                parameters = self.share_information(params=parameters) # share information among parameters of different optimizers, if there are more than one optimizer
                agent = self.update_parameters(agent=agent, parameters=parameters, param_opt=self.param_opt,
                                               trainable_compartment=self.trainable_compartment)  # update individual with optimized parameters
            """
            else:
                individual = self.update_parameters(
                    individual=individual,
                    parameters=parameters,
                    param_opt=self.param_opt,
                    trainable_compartment=self.trainable_compartment

                )  # update individual with optimized parameters
            """



            #print("params after share:")
            #print("_--------------------------------")
            #print(parameters)

                # Clip gradients if necessary (you can uncomment if needed)
                # for param in parameters[j].values():
                #     if param.grad is not None:
                #         param.grad = torch.clamp(param.grad, -1.0, 1.0)

                # Print gradients
                #gradients = {key: param.grad for key, param in parameters[j].items() if param.grad is not None}

            if i % self.checkpoint_interval == 0:
                toc = time.time()
                time_.append(toc - tic)
                tic = time.time()
                agent = self.update_parameters(agent=agent, parameters=parameters, param_opt=self.param_opt,
                                               trainable_compartment=self.trainable_compartment)

                self.save_to_h5py(
                    dataset_name="agent",
                    data_array=agent.detach().numpy(),
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
                    dataset_name="simulation_results",
                    data_array=simulation_results,
                    store_path=self.path,
                    file_name=self.file_name
                )
                self.save_to_h5py(
                    dataset_name="initial_conditions",
                    data_array=init_conditions,
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

        agent = self.update_parameters(agent=agent, parameters=parameters, param_opt=self.param_opt,
                                       trainable_compartment=self.trainable_compartment)
        self.save_to_h5py(
            dataset_name="agent",
            data_array=agent.detach().numpy(),
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
            dataset_name="simulation_results",
            data_array=simulation_results,
            store_path=self.path,
            file_name=self.file_name
        )
        self.save_to_h5py(
            dataset_name="initial_conditions",
            data_array=init_conditions,
            store_path=self.path,
            file_name=self.file_name
        )

        return agent, costs
