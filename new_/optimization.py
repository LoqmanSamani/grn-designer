from tensor_simulation import *
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