from tensor_simulation import *
import os
import h5py
import time
import numpy as np
import torch
import torch.nn.functional as F
from ignite.metrics import SSIM
#torch.autograd.set_detect_anomaly(True)

def parameter_extraction(agent, param_type, compartment_opt, trainable_compartment):
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