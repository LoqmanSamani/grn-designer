from ..sim.sim_tensor.tensor_simulation import *
import os
import h5py
import time
import numpy as np
import torch
import torch.nn.functional as F
from ignite.metrics import SSIM



class AdamOptimization:

    def __init__(self,
                 target,
                 path,
                 file_name,
                 epochs,
                 learning_rate,
                 param_opt,
                 param_type,
                 condition_opt,
                 cost_alpha,
                 cost_beta,
                 max_val,
                 checkpoint_interval,
                 share_info,
                 lr_decay,
                 decay_steps,
                 decay_rate,
                 trainable_compartment,
                 accumulation_steps,
                 device
                 ):

        self.epochs = epochs
        self.target = target
        self.path = path
        self.file_name = file_name
        self.param_opt = param_opt
        self.param_type = param_type
        self.compartment_opt = condition_opt
        self.cost_alpha = cost_alpha
        self.cost_beta = cost_beta
        self.max_val = max_val
        self.checkpoint_interval = checkpoint_interval
        self.share_info = share_info
        self.lr_decay = lr_decay
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.trainable_compartment = trainable_compartment
        self.accumulation_steps = accumulation_steps,
        self.device = device
        if learning_rate is None:
            learning_rate = [0.001]
        self.learning_rate = learning_rate



    def save_to_h5py(self, dataset_name, data_array, store_path, file_name):

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
            if param_type == "not_all" and trainable_compartment == num_species:
                s = 1
                for i in range(0, num_species * 2, 2):
                    if int(s - 1) == t:
                        parameters[f"species_{s}"] = torch.tensor(
                            agent[-1, i, 0:3].clone(),
                            requires_grad=True
                        )
                        s += 1
                    else:
                        parameters[f"species_{s}"] = torch.tensor(
                            agent[-1, i, 0:3].clone().detach()
                        )
                        s += 1

                p = 1
                for j in range(pair_start + 1, pair_stop + 1, 2):
                    parameters[f"pair_{p}"] = torch.tensor(
                        agent[j, 1, :4].clone(),
                        requires_grad=True
                    )
                    p += 1

                c = 1
                for com in range(1, num_species * 2, 2):
                    if int(c - 1) == t:
                        parameters[f"compartment_{c}"] = torch.tensor(
                            agent[com, :, :].clone(),
                            requires_grad=True
                        )
                        c += 1
                    else:
                        parameters[f"compartment_{c}"] = torch.tensor(
                            agent[com, :, :].clone().detach()
                        )
                        c += 1
                params.append(parameters)

            elif param_type == "not_all" and trainable_compartment < num_species:

                s = 1
                for i in range(0, num_species * 2, 2):
                    if int(s - 1) == t:
                        parameters[f"species_{s}"] = torch.tensor(
                            agent[-1, i, 0:3].clone(),
                            requires_grad=True
                        )
                        s += 1
                    elif s <= trainable_compartment:
                        parameters[f"species_{s}"] = torch.tensor(
                            agent[-1, i, 0:3].clone().detach()
                        )
                        s += 1
                    else:
                        parameters[f"species_{s}"] = torch.tensor(
                            agent[-1, i, 0:3].clone(),
                            requires_grad=True
                        )
                        s += 1

                p = 1
                for j in range(pair_start + 1, pair_stop + 1, 2):
                    parameters[f"pair_{p}"] = torch.tensor(
                        agent[j, 1, :4].clone(),
                        requires_grad=True
                    )
                    p += 1

                c = 1
                for com in range(1, num_species * 2, 2):
                    if int(c - 1) == t:
                        parameters[f"compartment_{c}"] = torch.tensor(
                            agent[com, :, :].clone(),
                            requires_grad=True
                        )
                        c += 1
                    else:
                        parameters[f"compartment_{c}"] = torch.tensor(
                            agent[com, :, :].clone(),
                            requires_grad=False
                        )
                        c += 1

                params.append(parameters)

            elif param_type == "all" and compartment_opt:
                s = 1
                for i in range(0, num_species * 2, 2):
                    parameters[f"species_{s}"] = torch.tensor(
                        agent[-1, i, 0:3].clone(),
                        requires_grad=True
                    )
                    s += 1

                p = 1
                for j in range(pair_start + 1, pair_stop + 1, 2):
                    parameters[f"pair_{p}"] = torch.tensor(
                        agent[j, 1, :4].clone(),
                        requires_grad=True
                    )
                    p += 1

                c = 1
                for com in range(1, num_species * 2, 2):
                    if int(c - 1) == t:
                        parameters[f"compartment_{c}"] = torch.tensor(
                            agent[com, :, :].clone(),
                            requires_grad=True
                        )
                        c += 1
                    else:
                        parameters[f"compartment_{c}"] = torch.tensor(
                            agent[com, :, :].clone().detach()
                        )
                        c += 1

                params.append(parameters)

            elif compartment_opt and not param_type:
                s = 1
                for i in range(0, num_species * 2, 2):
                    parameters[f"species_{s}"] = torch.tensor(
                        agent[-1, i, 0:3].clone().detach()
                    )
                    s += 1

                p = 1
                for j in range(pair_start + 1, pair_stop + 1, 2):
                    parameters[f"pair_{p}"] = torch.tensor(
                        agent[j, 1, :4].clone().detach()
                    )
                    p += 1

                c = 1
                for com in range(1, num_species * 2, 2):
                    if int(c - 1) == t:
                        parameters[f"compartment_{c}"] = torch.tensor(
                            agent[com, :, :].clone(),
                            requires_grad=True
                        )
                        c += 1
                    else:
                        parameters[f"compartment_{c}"] = torch.tensor(
                            agent[com, :, :].clone().detach()
                        )
                        c += 1

                params.append(parameters)

        parameters = {}
        if trainable_compartment == 0 and param_type:
            s = 1
            for i in range(0, num_species * 2, 2):
                parameters[f"species_{s}"] = torch.tensor(
                    agent[-1, i, 0:3].clone(),
                    requires_grad=True
                )
                s += 1

            p = 1
            for j in range(pair_start + 1, pair_stop + 1, 2):
                parameters[f"pair_{p}"] = torch.tensor(
                    agent[j, 1, :4].clone(),
                    requires_grad=True
                )
                p += 1

            c = 1
            for com in range(1, num_species * 2, 2):
                parameters[f"compartment_{c}"] = torch.tensor(
                    agent[com, :, :].clone().detach()
                )
                c += 1

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
                        agent[-1, j, :3] = parameters[i][
                            f"species_{species}"].detach().clone()
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


        y_hat = tensor_simulation(
            agent=agent,
            parameters=parameters,
            num_species=num_species,
            num_pairs=num_pairs,
            stop=stop,
            time_step=time_step,
            max_epoch=max_epoch,
            compartment=compartment,
            device=device
        )

        return y_hat

    def compute_cost_(self, y_hat, target, alpha, beta, max_val):

        mse_loss = F.mse_loss(y_hat, target)
        ssim_loss_value = self.ssim_loss(y_hat, target, max_val)
        total_loss = alpha * mse_loss + beta * ssim_loss_value

        return total_loss


    def ssim_loss(self, y_hat, target, max_val):

        ssim_metric = SSIM(data_range=max_val)
        if y_hat.dim() == 2:
            y_hat = y_hat.unsqueeze(0).unsqueeze(0)
        if target.dim() == 2:
            target = target.unsqueeze(0).unsqueeze(0)

        ssim_metric.update((y_hat, target))
        ssim_score = ssim_metric.compute()
        ssim_metric.reset()

        return 1 - ssim_score


    def share_information(self, params, num_species):

        pair_sums = {}
        species_sums = {}

        for current_dict in params:
            for key, val in current_dict.items():
                if "pair_" in key and val.requires_grad:
                    if key not in pair_sums:
                        pair_sums[key] = val.detach().clone()
                    else:
                        pair_sums[key] += val.detach()

        if len(params) > 1 and num_species > len(params):

            for current_dict in params:
                for i in range(len(params) + 1, num_species + 1):
                    for key, val in current_dict.items():
                        if key == f"species_{i}" and val.requires_grad:
                            if key not in species_sums:
                                species_sums[key] = val.detach().clone()
                            else:
                                species_sums[key] += val.detach()

        for i, current_dict in enumerate(params):
            for key, val in current_dict.items():
                if val.requires_grad:
                    for j in range(len(params)):
                        if i != j and key in params[j] and not params[j][key].requires_grad:
                            with torch.no_grad():
                                params[j][key].copy_(val)

                if key in pair_sums and val.requires_grad:
                    with torch.no_grad():
                        val.copy_(pair_sums[key] / len(params))
                if key in species_sums and val.requires_grad:
                    with torch.no_grad():
                        val.copy_(species_sums[key] / len(params))

        return params



    def create_optimizer(self, model_parameters, lr):


        optimizer = torch.optim.Adam(params=model_parameters, lr=lr)

        lr_scheduler = None
        if self.lr_decay:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=self.decay_rate
            )

        return optimizer, lr_scheduler



    def gradient_optimization(self, agent):


        costs = []
        time_ = []

        simulation_results = np.zeros(
            shape=(self.trainable_compartment, self.epochs, self.target.shape[1], self.target.shape[2]),
            dtype=np.float32
        )
        init_conditions = np.zeros(
            shape=(self.trainable_compartment, self.epochs, self.target.shape[1], self.target.shape[2]),
            dtype=np.float32
        )

        self.save_to_h5py(
            dataset_name="target",
            data_array=self.target,
            store_path=self.path,
            file_name=self.file_name
        )

        parameters, num_species, num_pairs, max_epoch, stop, time_step = self.parameter_extraction(
            agent=agent,
            param_type=self.param_type,
            compartment_opt=self.compartment_opt,
            trainable_compartment=self.trainable_compartment
        )

        if len(self.learning_rate) > 1:
            optimizers = [self.create_optimizer(list(parameters[i].values()), self.learning_rate[i]) for i in
                          range(len(parameters))]
        else:
            optimizers = [self.create_optimizer(list(parameters[i].values()), self.learning_rate[0]) for i in
                          range(len(parameters))]

        tic_ = time.time()
        tic = time.time()
        for i in range(1, self.epochs + 1):
            cost_ = []
            for j in range(len(parameters)):

                prediction = self.simulation(
                    agent=agent,
                    parameters=parameters[j],
                    num_species=num_species,
                    num_pairs=num_pairs,
                    stop=stop,
                    time_step=time_step,
                    max_epoch=max_epoch,
                    compartment=j,
                    device=self.device
                )
                cost = self.compute_cost_(
                    y_hat=prediction,
                    target=self.target[j, :, :],
                    alpha=self.cost_alpha,
                    beta=self.cost_beta,
                    max_val=self.max_val
                )
                cost.backward()
                optimizers[j][0].step()
                if optimizers[j][1] is not None:
                    optimizers[j][1].step()

                optimizers[j][0].zero_grad(set_to_none=True)
                cost_.append(cost.item())
                simulation_results[j, i - 1, :, :] = prediction.detach()
                init_conditions[j, i - 1, :, :] = parameters[j][f"compartment_{j + 1}"].detach().numpy()

                if len(optimizers) > 1:
                    print(f"Epoch {i}/{self.epochs}, Optimizer {j + 1}, Cost: {cost.item()}")
                else:
                    print(f"Iteration {i}/{self.epochs}, Cost: {cost.item()}")

            costs.append(cost_)

            if i % self.share_info == 0 and len(optimizers) > 1:
                parameters = self.share_information(
                    params=parameters,
                    num_species=num_species
                )
                agent = self.update_parameters(
                    agent=agent,
                    parameters=parameters,
                    param_opt=self.param_opt,
                    trainable_compartment=self.trainable_compartment
                )

            if i % self.checkpoint_interval == 0:
                toc = time.time()
                time_.append(toc - tic)
                tic = time.time()
                agent = self.update_parameters(
                    agent=agent,
                    parameters=parameters,
                    param_opt=self.param_opt,
                    trainable_compartment=self.trainable_compartment
                )
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

        agent = self.update_parameters(
            agent=agent,
            parameters=parameters,
            param_opt=self.param_opt,
            trainable_compartment=self.trainable_compartment
        )
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


"""
from tensor_simulation import *
import os
import h5py
import time
import numpy as np
import tensorflow as tf




class AdamOptimization:

    def __init__(self,
                 target,
                 path,
                 file_name,
                 epochs=100,
                 learning_rate=None,
                 param_opt=False,
                 initial_condition_opt=True,
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
        self.initial_condition_opt = initial_condition_opt
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


    def save_to_h5py(self, dataset_name, data_array, store_path, file_name):


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




    def parameter_extraction(self, individual, param_opt, initial_condition_opt, trainable_compartment):

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
                            individual[-1, i, 0:3],
                            name=f"species_{species}",
                            trainable=True
                        )
                        species += 1
                    else:
                        parameters[f"species_{species}"] = tf.Variable(
                            individual[-1, i, 0:3],
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
                        individual[-1, i, 0:3],
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

            if initial_condition_opt:
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
                        individual[-1, i, 0:3],
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
                        individual[-1, i, 0:3],
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
        print(params)

        return params, num_species, num_pairs, max_epoch, stop, time_step




    def update_parameters(self, individual, parameters, param_opt, trainable_compartment):
        #print("ind before update:")
        #print("--------------------------------------------")
        #print("com 0:", individual[0])
        #print("com 1:", individual[1])
        #print("com 2:", individual[2])
        #print("com 3:", individual[3])
        #print("com 4:", individual[4])
        #print("com 5:", individual[5])
        #print("com 6:", individual[6])

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
        #print("ind after update:")
        #print("--------------------------------------------")
        #print("com 0:", individual[0])
        #print("com 1:", individual[1])
        #print("com 2:", individual[2])
        #print("com 3:", individual[3])
        #print("com 4:", individual[4])
        #print("com 5:", individual[5])
        #print("com 6:", individual[6])

        return individual

    def simulation(self, individual, parameters, num_species, num_pairs, stop, time_step, max_epoch, compartment):

        y_hat = tensor_simulation(agent=individual, parameters=parameters, num_species=num_species, num_pairs=num_pairs,
                                  stop=stop, time_step=time_step, max_epoch=max_epoch, compartment=compartment, device=)

        return y_hat

    def compute_cost_(self, y_hat, target, alpha, beta, max_val):


        mse_loss = tf.reduce_mean(tf.square(y_hat - target))
        #print("------------------------------------------")
        #print("mse loss:")
        #print(mse_loss)
        ssim_loss_value = self.ssim_loss(y_hat, target, max_val)
        #print("ssim loss:")
        #print(ssim_loss_value)
        total_loss = alpha * mse_loss + beta * ssim_loss_value
        #print("total loss:")
        #print(total_loss)
        return total_loss

    def ssim_loss(self, y_hat, target, max_val):

        y_hat = tf.expand_dims(y_hat, axis=-1)
        target = tf.expand_dims(target, axis=-1)
        ssim_score = tf.image.ssim(y_hat, target, max_val=max_val)

        return (1 - tf.reduce_mean(ssim_score)).numpy()



    def share_information(self, params):

        for i in range(len(params)):
            current_dict = params[i]
            for j in range(len(params)):
                if i != j:
                    for key, val in current_dict.items():
                        if val.trainable:
                            if key in params[j] and not params[j][key].trainable:
                                params[j][key].assign(val)

        return params


    def init_individual(self, individual):
        #print("init ind:")
        #print("-----------------------------")
        #print("ind before init:")
        #print(individual)
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
        #print("ind after init:")
        #print(individual)

        return individual



    def gradient_optimization(self, individual):


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
            initial_condition_opt=self.initial_condition_opt,
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
                    individual = self.init_individual(individual=individual)
                    cost_.append(cost.numpy())
                    print(f"Epoch {i}/{self.epochs}, Optimizer {j + 1}, Cost: {cost.numpy()}")

                variables = list(parameters[j].values())
                gradients = tape.gradient(cost, variables)
                # gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients]
                #print("grads:")
                #print("------------------------------------------------------")
                #print(gradients)
                optimizer.apply_gradients(zip(gradients, variables))
                results[j, i - 1, :, :] = y_hat.numpy()


            #print("params before share:")
            #print("_--------------------------------")
            #print(parameters)

            parameters = self.share_information(params=parameters)

            #print("params after share:")
            #print("_--------------------------------")
            #print(parameters)


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
"""
