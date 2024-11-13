from tensor_simulation import *
import os
import h5py
import time
import numpy as np
import torch
import torch.nn.functional as F
from ignite.metrics import SSIM
import gc



class GradientOptimization:

    def __init__(self,
                 target,
                 path,
                 file_name,
                 epochs=100,
                 optimizer=None,
                 param_opt=False,
                 initial_condition_opt=True,
                 learning_rate=None,
                 cost_alpha=1.0,
                 cost_beta=1.0,
                 ssim_data_range=1.0,
                 checkpoint_interval=10,
                 interval_save=2,
                 pattern_proportion=1000,
                 lr_decay=False,
                 decay_steps=40,
                 decay_rate=0.6,
                 device=None
                 ):

        self.epochs = epochs
        if target.dtype != torch.float32:
            self.target = target.to(torch.float32)
        else:
            self.target = target
        self.path = path
        self.file_name = file_name
        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = "SGD"
        self.param_opt = param_opt
        self.initial_condition_opt = initial_condition_opt
        self.cost_alpha = cost_alpha
        self.cost_beta = cost_beta
        self.ssim_data_range = ssim_data_range
        self.checkpoint_interval = checkpoint_interval
        self.interval_save = interval_save
        self.pattern_proportion = pattern_proportion
        self.lr_decay = lr_decay
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = learning_rate or 0.001


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


    def parameter_extraction(self, agent, initial_condition_opt, parameter_opt):

        num_species = int(agent[-1, -1, 0])
        max_epoch = int(agent[-1, -1, 1])
        stop = int(agent[-1, -1, 2])
        time_step = agent[-1, -1, 3]


        parameters = {}

        if initial_condition_opt:
            c = 1
            for com in range(1, num_species * 2, 2):
                parameters[f"initial_conditions_{c}"] = torch.tensor(
                    agent[com, :, :].clone(),
                    requires_grad=True
                )
                c += 1

        elif not initial_condition_opt:
            c = 1
            for com in range(1, num_species * 2, 2):
                parameters[f"initial_conditions_{c}"] = torch.tensor(
                    agent[com, :, :].clone().detach()
                )
                c += 1

        if parameter_opt:
            s = 1
            for i in range(0, num_species * 2, 2):
                num_param = int(agent[-1, i, -1] + 3)
                parameters[f"species_{s}"] = torch.tensor(
                    agent[-1, i, :num_param].clone(),
                    requires_grad=True
                )
                s += 1

        elif not parameter_opt:
            s = 1
            for i in range(0, num_species * 2, 2):
                num_param = int(agent[-1, i, -1] + 3)
                parameters[f"species_{s}"] = torch.tensor(
                    agent[-1, i, :num_param].clone().detach()
                )
                s += 1
                
        for name, param in parameters.items():
            print(f"Parameter: {name}, Trainable: {param.requires_grad}")
            print(len(param))
            

        return parameters, num_species, max_epoch, stop, time_step



    def update_parameters(self, agent, parameters):
        
        num_species = int(agent[-1, -1, 0])

        j = 0
        for species in range(1, num_species + 1):
            num_param = int(agent[-1, int((species-1)*2), -1] + 3)
            agent[-1, j, :num_param] = parameters[f"species_{species}"].detach().clone()
            j += 2

        for comp in range(1, num_species + 1):
            idx = int(((comp - 1) * 2) + 1)
            updates = torch.max(parameters[f"initial_conditions_{comp}"], torch.tensor(0.0))
            agent[idx, :, :] = updates.detach().clone()

        return agent
    


    def simulation(self, agent, parameters, num_species, stop, time_step, max_epoch, device):

        prediction = agent_simulation(
            agent=agent,
            parameters=parameters,
            num_species=num_species,
            stop=stop,
            time_step=time_step,
            max_epoch=max_epoch,
            device=device
        )

        return prediction


    def weighted_prediction(self, prediction):

        if prediction.ndimension() == 3:
            batch_size, height, width = prediction.shape

            weights = torch.ones(batch_size, device=prediction.device)
            weights[0] = self.pattern_proportion

            weights = weights / weights.sum()
            norm_prediction = (prediction * weights[:, None, None]).sum(dim=0)
        else:
            norm_prediction = prediction

        return norm_prediction



    def compute_cost_(self, prediction, target, alpha, beta, ssim_data_range):

        if prediction.ndimension() == 3:
            norm_prediction = self.weighted_prediction(prediction=prediction)
        else:
            norm_prediction = prediction

        mse_loss = F.mse_loss(norm_prediction, target)
        ssim_loss_value = self.ssim_loss(norm_prediction, target, ssim_data_range)
        total_loss = alpha * mse_loss + beta * ssim_loss_value

        return total_loss

    def ssim_loss(self, prediction, target, ssim_data_range):

        ssim_metric = SSIM(data_range=ssim_data_range)
        if prediction.dim() == 2:
            prediction = prediction.unsqueeze(0).unsqueeze(0)
        if target.dim() == 2:
            target = target.unsqueeze(0).unsqueeze(0)

        ssim_metric.update((prediction, target))
        ssim_score = ssim_metric.compute()
        ssim_metric.reset()

        return 1 - ssim_score



    def create_optimizer(self, model_parameters, lr):

        if self.optimizer == "GSD":
            optimizer = torch.optim.SGD(params=model_parameters, lr=lr)
        else:
            optimizer = torch.optim.Adam(params=model_parameters, lr=lr)


        lr_scheduler = None
        if self.lr_decay:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=self.decay_rate
            )

        return optimizer, lr_scheduler



    def fit(self, agent):

        costs = []
        time_ = []

        self.save_to_h5py(
            dataset_name="target",
            data_array=self.target,
            store_path=self.path,
            file_name=self.file_name
        )

        parameters, num_species, max_epoch, stop, time_step = self.parameter_extraction(
            agent=agent,
            initial_condition_opt=self.initial_condition_opt,
            parameter_opt=self.param_opt
        )

        simulation_results = np.zeros(
            shape=(int(self.epochs/self.interval_save), self.target.shape[0], self.target.shape[1]),
            dtype=np.float32
        )
        init_conditions = np.zeros(
            shape=(num_species, int(self.epochs/self.interval_save), self.target.shape[0], self.target.shape[1]),
            dtype=np.float32
        )

        optimizer, lr_scheduler = self.create_optimizer(
            model_parameters=list(parameters.values()),
            lr=self.learning_rate
        )

        tic_ = time.time()
        tic = time.time()
        
        inx = 0
        for i in range(1, self.epochs + 1):

            prediction = self.simulation(
                agent=agent,
                parameters=parameters,
                num_species=num_species,
                stop=stop,
                time_step=time_step,
                max_epoch=max_epoch,
                device=self.device
            )

            cost = self.compute_cost_(
                prediction=prediction,
                target=self.target,
                alpha=self.cost_alpha,
                beta=self.cost_beta,
                ssim_data_range=self.ssim_data_range
            )
            cost.backward()
            optimizer.step()

            with torch.no_grad():
                for param, val in parameters.items():
                    val.clamp_(min=0.0, max=0.999)
                    
            costs.append(cost.item())

            if lr_scheduler is not None:
                lr_scheduler.step()

            if i % self.interval_save == 0:
                simulation_results[inx, :, :] = prediction[0, :, :].detach()
                for j in range(num_species):
                    init_conditions[j, inx, :, :] = parameters[f"initial_conditions_{j+1}"].detach().numpy()
                inx += 1

            print(f"Iteration {i}/{self.epochs}, Cost: {cost.item()}")
            optimizer.zero_grad(set_to_none=True)

            del prediction, cost
            gc.collect()

            if i % self.checkpoint_interval == 0:
                toc = time.time()
                time_.append(toc - tic)
                tic = time.time()
                agent = self.update_parameters(
                    agent=agent,
                    parameters=parameters
                )

                self.save_to_h5py(
                    dataset_name="agent",
                    data_array=agent.detach().numpy(),
                    store_path=self.path,
                    file_name=self.file_name
                )
                self.save_to_h5py(
                    dataset_name="gradient_costs",
                    data_array=np.array(costs),
                    store_path=self.path,
                    file_name=self.file_name
                )
                self.save_to_h5py(
                    dataset_name="run_time",
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
            dataset_name="run_time",
            data_array=np.array(time_),
            store_path=self.path,
            file_name=self.file_name
        )

        agent = self.update_parameters(
            agent=agent,
            parameters=parameters
        )
        self.save_to_h5py(
            dataset_name="agent",
            data_array=agent.detach().numpy(),
            store_path=self.path,
            file_name=self.file_name
        )
        self.save_to_h5py(
            dataset_name="gradient_costs",
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