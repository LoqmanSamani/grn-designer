from gabonst import evolutionary_optimization
from initialization import population_initialization
from reshape import Resize
from optimization import AdamOptimization
import numpy as np
import os
import h5py
import time
import torch



class BioEsAg:
    def __init__(self,
                 target, population=None, population_size=None, evolution_one_epochs=None, evolution_two_epochs=None, optimization_epochs=None,
                 agent_shape=None, agent_parameters=None, simulation_parameters=None, num_init_species=None, num_init_complex=None,
                 store_path=None, learning_rate=None, sim_mutation_rate=None, initial_condition_mutation_rate=None, accumulation_steps=None,
                 parameter_mutation_rate=None, insertion_mutation_rate=None, deletion_mutation_rate=None, crossover_alpha=None,
                 checkpoint_interval=None, lr_decay=False, decay_steps=None, decay_rate=None, trainable_compartment=None,
                 gradient_optimization=False, parameter_optimization=False, condition_optimization=False, sim_mutation=True,
                 initial_condition_mutation=True, parameter_mutation=False, species_insertion_mutation_one=False, share_info=None,
                 species_deletion_mutation_one=False, species_insertion_mutation_two=False, species_deletion_mutation_two=False,
                 initial_condition_crossover=True, parameter_crossover=False, simulation_crossover=True, fixed_agent_shape=False,
                 cost_alpha=None, cost_beta=None, cost_constant=None, evolution_two_ratio=None, zoom_=False, zoom_in_factor=None,
                 zoom_out_factor=None, num_elite_agents=None, simulation_min=None, simulation_max=None, param_type=None,
                 initial_condition_min=None, initial_condition_max=None, parameter_min=None, parameter_max=None, device=None
                 ):


        self.target = target
        self.population = population

        self.population_size = population_size or 50
        self.evolution_one_epochs = evolution_one_epochs or 50
        self.evolution_two_epochs = evolution_two_epochs or 50
        self.optimization_epochs = optimization_epochs or 50
        self.agent_shape = agent_shape or (3, 50, 50)
        self.agent_parameters = agent_parameters or {"species_parameters":[np.random.rand(3), np.random.rand(3)], "complex_parameters":[[(0, 2), np.random.rand(4)]]}
        self.simulation_parameters = simulation_parameters or {"max_simulation_epoch":100, "simulation_stop_time":20, "time_step":0.2}
        self.num_init_species = num_init_species or 1
        self.num_init_complex = num_init_complex or 0
        self.store_path = store_path

        self.learning_rate = learning_rate
        self.sim_mutation_rate = sim_mutation_rate or 0.5
        self.initial_condition_mutation_rate = initial_condition_mutation_rate or 0.1
        self.parameter_mutation_rate = parameter_mutation_rate or 0.08
        self.insertion_mutation_rate = insertion_mutation_rate or 0.07
        self.deletion_mutation_rate = deletion_mutation_rate or 0.09
        self.crossover_alpha = crossover_alpha or 0.4

        self.checkpoint_interval = checkpoint_interval or 10
        self.lr_decay = lr_decay
        self.decay_steps = decay_steps or 100
        self.decay_rate = decay_rate or 0.96
        self.trainable_compartment = trainable_compartment or 1
        self.param_type = param_type or "all"
        self.share_info = share_info or 2
        self.accumulation_steps = accumulation_steps
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gradient_optimization = gradient_optimization
        self.parameter_optimization = parameter_optimization
        self.condition_optimization = condition_optimization

        self.sim_mutation = sim_mutation
        self.initial_condition_mutation = initial_condition_mutation
        self.parameter_mutation = parameter_mutation
        self.species_insertion_mutation_one = species_insertion_mutation_one
        self.species_deletion_mutation_one = species_deletion_mutation_one
        self.species_insertion_mutation_two = species_insertion_mutation_two
        self.species_deletion_mutation_two = species_deletion_mutation_two

        self.initial_condition_crossover = initial_condition_crossover
        self.parameter_crossover = parameter_crossover
        self.simulation_crossover = simulation_crossover
        self.fixed_agent_shape = fixed_agent_shape

        self.cost_alpha = cost_alpha or 1.0
        self.cost_beta = cost_beta or 1.0
        self.cost_constant = cost_constant or 1.0

        self.evolution_two_ratio = evolution_two_ratio or 1.0
        self.zoom_ = zoom_
        self.zoom_in_factor = zoom_in_factor or 0.5
        self.zoom_out_factor = zoom_out_factor or 2
        self.num_elite_agents = num_elite_agents or 10

        self.simulation_min = simulation_min or (5, 0.05)
        self.simulation_max = simulation_max or (40, 0.3)
        self.initial_condition_min = initial_condition_min or 0.0
        self.initial_condition_max = initial_condition_max or 2.0
        self.parameter_min = parameter_min or 0.0
        self.parameter_max = parameter_max or 0.99

        self.reshape_ = Resize(
            order=1,
            mode="constant",
            cval=0.0,
            grid_mode=False
        )

        self.gradient_optimization_ = AdamOptimization(
            target=torch.from_numpy(self.target),
            path=self.store_path,
            file_name="gradient_result",
            epochs=self.optimization_epochs,
            learning_rate=self.learning_rate,
            param_opt=self.parameter_optimization,
            param_type=self.param_type,
            condition_opt=self.condition_optimization,
            cost_alpha=self.cost_alpha,
            cost_beta=self.cost_beta,
            max_val=1.0,
            checkpoint_interval=self.checkpoint_interval,
            share_info=self.share_info,
            lr_decay=self.lr_decay,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            trainable_compartment=self.trainable_compartment,
            accumulation_steps=self.accumulation_steps,
            device=self.device
        )


    def save_to_h5py(self, dataset_name, data_array):

        if self.store_path:
            path = os.path.join(self.store_path, "evolutionary_result")
            if not os.path.exists(self.store_path):
                os.makedirs(self.store_path)
        else:
            path = os.path.join(os.path.expanduser("~"), "evolutionary_result")

        with h5py.File(path, 'a') as h5file:
            if dataset_name in h5file:
                del h5file[dataset_name]

            h5file.create_dataset(dataset_name, data=data_array)




    def fit(self):

        num_patterns, _, _ = self.target.shape
        run_time = np.zeros(shape=(1, ), dtype=np.float32)
        best_agents = []
        final_agent = None
        tic = time.time()
        evolutionary_costs = np.zeros(
            shape=(self.evolution_one_epochs + self.evolution_two_epochs, 2),
            dtype=np.float32
        )

        self.save_to_h5py(
            dataset_name="original_target",
            data_array=self.target
        )

        num_species = len(self.agent_parameters["species_parameters"])
        num_complex = len(self.agent_parameters["complex_parameters"])

        parameters = (
            self.agent_parameters["species_parameters"],
            self.agent_parameters["complex_parameters"],
            self.simulation_parameters
        )
        crossover = (
            self.crossover_alpha,
            self.simulation_crossover,
            self.initial_condition_crossover,
            self.parameter_crossover
        )
        mutation = (
            self.sim_mutation,
            self.initial_condition_mutation,
            self.parameter_mutation,
            self.species_insertion_mutation_one,
            self.species_deletion_mutation_one
        )
        bounds = (
            self.simulation_min,
            self.simulation_max,
            self.initial_condition_min,
            self.initial_condition_max,
            self.parameter_min,
            self.parameter_max
        )
        rates = (
            self.sim_mutation_rate,
            self.initial_condition_mutation_rate,
            self.parameter_mutation_rate,
            self.insertion_mutation_rate,
            self.deletion_mutation_rate
        )
        cost = (
            self.cost_alpha,
            self.cost_beta
        )

        tt = []
        if self.zoom_:
            for i in range(self.target.shape[0]):
                t_ = self.reshape_.zoom_in(
                    target=self.target[i, :, :],
                    zoom_=self.zoom_in_factor
                )
                tt.append(t_)
            target_ = np.zeros(
                shape=(self.target.shape[0], tt[0].shape[0], tt[0].shape[1]),
                dtype=np.float32
            )
            for j in range(len(tt)):
                target_[j, :, :] = tt[j]
        else:
            target_ = self.target

        self.save_to_h5py(
            dataset_name="zoomed_in_target",
            data_array=target_
        )
        if self.population:
            population = self.population
        else:
            population = population_initialization(
                population_size=self.population_size,
                agent_shape=(self.agent_shape[0], target_.shape[1], target_.shape[2]),
                species_parameters=self.agent_parameters["species_parameters"],
                complex_parameters=self.agent_parameters["complex_parameters"],
                num_species=num_species,
                num_complex=num_complex,
                max_sim_epochs=self.simulation_parameters["max_simulation_epoch"],
                sim_stop_time=self.simulation_parameters["simulation_stop_time"],
                time_step=self.simulation_parameters["time_step"],
                fixed_agent_shape=self.fixed_agent_shape,
                init_species=self.num_init_species,
                init_complex=self.num_init_complex
            )



        print("___________________________________________________________________________")
        print("                            BioEsAg Algorithm                              ")
        print("___________________________________________________________________________")
        print()

        for i in range(self.evolution_one_epochs):

            population, costs, mean_cost = evolutionary_optimization(
                population=population,
                target=target_,
                population_size=self.population_size,
                num_patterns=num_patterns,
                init_species=self.num_init_species,
                init_complex=self.num_init_complex,
                cost=cost,
                rates=rates,
                bounds=bounds,
                mutation=mutation,
                crossover=crossover,
                num_elite_agents=self.num_elite_agents,
                fixed_agent_shape=self.fixed_agent_shape,
                parameters=parameters,
                cost_constant=self.cost_constant
            )
            min_cost_index = np.argmin(costs)
            best_agent = population[min_cost_index]
            best_agents.append(best_agent)

            sorted_costs = np.sort(costs)
            evolutionary_costs[i, 0] = sorted_costs[0]
            evolutionary_costs[i, 1] = mean_cost

            print(f"Epoch {i+1}/{self.evolution_one_epochs}, Avg/Min Population Cost: {mean_cost}/{sorted_costs[0]}")


            if i == self.evolution_one_epochs - 1:
                new_population_size = int(self.population_size * self.evolution_two_ratio)
                sorted_cost_indices = np.argsort(cost)[:new_population_size]
                population = [population[idx] for idx in sorted_cost_indices]

        if self.zoom_:
            population = self.reshape_.zoom_out(
                population=population,
                zoom_=self.zoom_out_factor,
                x_=self.target.shape[1],
                y_=self.target.shape[2]
            )

            print()
            print("___________________________________________________________________________")
            print()

            pop_size = len(population)
            idx = self.evolution_one_epochs - 1
            for j in range(self.evolution_two_epochs):
                population, costs, mean_cost = evolutionary_optimization(
                    population=population,
                    target=self.target,
                    population_size=pop_size,
                    num_patterns=num_patterns,
                    init_species=self.num_init_species,
                    init_complex=self.num_init_complex,
                    cost=cost,
                    rates=rates,
                    bounds=bounds,
                    mutation=mutation,
                    crossover=crossover,
                    num_elite_agents=self.num_elite_agents,
                    fixed_agent_shape=self.fixed_agent_shape,
                    parameters=parameters,
                    cost_constant=self.cost_constant
                )

                min_cost_index = np.argmin(costs)
                best_agent = population[min_cost_index]
                best_agents.append(best_agent)

                sorted_costs = np.sort(costs)
                evolutionary_costs[idx, 0] = sorted_costs[0]
                evolutionary_costs[idx, 1] = mean_cost

                print(f"Epoch {idx + 1}/{self.evolution_one_epochs + self.evolution_two_epochs}, Avg/Min Population Cost: {mean_cost}/{sorted_costs[0]}")
                if j == self.evolution_two_epochs - 1:
                    min_cost_index = np.argmin(costs)
                    final_agent = population[min_cost_index]

        self.save_to_h5py(
            dataset_name="evolutionary_costs",
            data_array=evolutionary_costs
        )
        for ag in range(len(best_agents)):
            self.save_to_h5py(
                dataset_name=f"agent_{ag}",
                data_array=best_agents[ag]
            )
        toc = time.time()
        run_time[0] = toc - tic
        self.save_to_h5py(
            dataset_name="run_time",
            data_array=run_time
        )


        if self.gradient_optimization:
            print()
            print("___________________________________________________________________________")
            print()

            _, _ = self.gradient_optimization_.gradient_optimization(
                    agent=torch.from_numpy(final_agent)
            )

        return population, evolutionary_costs



