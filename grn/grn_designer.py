from genetic_algorithm import evolutionary_optimization
from initialization import population_initialization, reset_agent
from simulation import agent_simulation
from reshape import Resize
from optimization import GradientOptimization
import numpy as np
import os
import h5py
import time
import torch
import gc



class GRNDesigner:
    def __init__(self,
                 target, agent, population=None, population_size=None, evolution_one_epochs=None, evolution_two_epochs=None, optimization_epochs=None,
                 agent_shape=None, agent_parameters=None, simulation_parameters=None,
                 store_path=None, learning_rate=None, sim_mutation_rate=None, initial_condition_mutation_rate=None,
                 parameter_mutation_rate=None, species_insertion_mutation_rate=None,
                 connection_insertion_mutation_rate=None, connection_deletion_mutation_rate=None, cost_pattern_proportion=None,
                 crossover_alpha=None, checkpoint_interval=None, lr_decay=False, decay_steps=None, decay_rate=None,
                 gradient_optimization=False, parameter_optimization=False, condition_optimization=False, sim_mutation=True,
                 initial_condition_mutation=True, parameter_mutation=False, species_insertion_mutation_one=False,
                 connection_insertion_mutation_one=False, connection_deletion_mutation_one=False, species_insertion_mutation_two=False,
                 connection_insertion_mutation_two=False, connection_deletion_mutation_two=False,
                 initial_condition_crossover=True, parameter_crossover=False, simulation_crossover=True, fixed_agent_shape=False,
                 cost_alpha=None, cost_beta=None, cost_constant=None, evolution_two_ratio=None, zoom_=False, zoom_in_factor=None,
                 zoom_out_factor=None, num_elite_agents=None, simulation_min=None, simulation_max=None,
                 initial_condition_min=None, initial_condition_max=None, parameter_min=None, parameter_max=None, device=None,
                 interval_save=None, gradient_optimizer=None, loss_threshold=None
                 ):


        if target.dtype != np.float32:
            self.target = target.astype(np.float32)
        else:
            self.target = target

        if agent.dtype != np.float32:
            self.agent = agent.astype(np.float32)
        else:
            self.agent = agent

        self.population = population

        self.population_size = population_size or 50
        self.evolution_one_epochs = evolution_one_epochs or 50
        self.evolution_two_epochs = evolution_two_epochs or 0
        self.optimization_epochs = optimization_epochs or 0
        self.agent_shape = agent_shape or (3, 100, 100)
        self.agent_parameters = agent_parameters or np.random.rand(3)
        self.simulation_parameters = simulation_parameters or {"max_simulation_epoch":100, "simulation_stop_time":20, "time_step":0.2}
        self.store_path = store_path

        self.learning_rate = learning_rate or 0.01
        self.sim_mutation_rate = sim_mutation_rate or 0.1
        self.initial_condition_mutation_rate = initial_condition_mutation_rate or 0.1
        self.parameter_mutation_rate = parameter_mutation_rate or 0.1
        self.species_insertion_mutation_rate = species_insertion_mutation_rate or 0.04
        self.connection_insertion_mutation_rate = connection_insertion_mutation_rate or .06
        self.connection_deletion_mutation_rate = connection_deletion_mutation_rate or .03

        self.crossover_alpha = crossover_alpha or 0.4
        self.checkpoint_interval = checkpoint_interval or 10
        self.lr_decay = lr_decay
        self.decay_steps = decay_steps or 100
        self.decay_rate = decay_rate or 0.96
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gradient_optimization = gradient_optimization
        self.parameter_optimization = parameter_optimization
        self.condition_optimization = condition_optimization

        self.sim_mutation = sim_mutation
        self.initial_condition_mutation = initial_condition_mutation
        self.parameter_mutation = parameter_mutation
        self.species_insertion_mutation_one = species_insertion_mutation_one
        self.connection_insertion_mutation_one = connection_insertion_mutation_one
        self.connection_deletion_mutation_one = connection_deletion_mutation_one
        self.species_insertion_mutation_two = species_insertion_mutation_two
        self.connection_insertion_mutation_two = connection_insertion_mutation_two
        self.connection_deletion_mutation_two = connection_deletion_mutation_two

        self.initial_condition_crossover = initial_condition_crossover
        self.parameter_crossover = parameter_crossover
        self.simulation_crossover = simulation_crossover
        self.fixed_agent_shape = fixed_agent_shape

        self.cost_alpha = cost_alpha or 1.0
        self.cost_beta = cost_beta or 1.0
        self.cost_constant = cost_constant or 1.0
        self.loss_threshold = loss_threshold
        self.cost_pattern_proportion = cost_pattern_proportion or 1000

        self.evolution_two_ratio = evolution_two_ratio or 1.0
        self.zoom_ = zoom_
        self.zoom_in_factor = zoom_in_factor or 0.5
        self.zoom_out_factor = zoom_out_factor or 2
        self.num_elite_agents = num_elite_agents or 5

        self.simulation_min = simulation_min or (5, 0.05)
        self.simulation_max = simulation_max or (40, 0.3)
        self.initial_condition_min = initial_condition_min or 0.0
        self.initial_condition_max = initial_condition_max or 2.0
        self.parameter_min = parameter_min or 0.0
        self.parameter_max = parameter_max or 0.99
        self.interval_save = interval_save or 5
        self.gradient_optimizer = gradient_optimizer or "SGD"
        self.reshape_ = Resize(
            order=3,
            mode="constant",
            cval=0.0,
            grid_mode=False
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

        run_time = np.zeros(shape=(1, ), dtype=np.float32)
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

        parameters = (
            self.agent_parameters,
            self.simulation_parameters
        )
        crossover = (
            self.crossover_alpha,
            self.simulation_crossover,
            self.initial_condition_crossover,
            self.parameter_crossover
        )
        mutation_one = (
            self.sim_mutation,
            self.initial_condition_mutation,
            self.parameter_mutation,
            self.species_insertion_mutation_one,
            self.connection_insertion_mutation_one,
            self.connection_deletion_mutation_one
        )
        mutation_two = (
            self.sim_mutation,
            self.initial_condition_mutation,
            self.parameter_mutation,
            self.species_insertion_mutation_two,
            self.connection_insertion_mutation_two,
            self.connection_deletion_mutation_two
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
            self.species_insertion_mutation_rate,
            self.connection_insertion_mutation_rate,
            self.connection_deletion_mutation_rate
        )
        cost = (
            self.cost_alpha,
            self.cost_beta,
            self.cost_pattern_proportion
        )


        if self.zoom_:
            target_ = self.reshape_.zoom_in(
                target=self.target,
                zoom_=self.zoom_in_factor
            )
        else:
            target_ = self.target
            

        if self.zoom_:
            self.save_to_h5py(
                dataset_name="zoomed_in_target",
                data_array=target_
            )
            
        length = max(2, (self.evolution_one_epochs + self.evolution_two_epochs) // self.interval_save)


        simulation_results = np.zeros(
            shape=(length, self.target.shape[0], self.target.shape[1]),
            dtype=np.float32
        )

        if self.population:
            population = self.population
        else:
            population = population_initialization(
                population_size=self.population_size,
                agent_shape=(self.agent_shape[0], target_.shape[0], target_.shape[1]),
                species_parameters=self.agent_parameters,
                max_sim_epochs=self.simulation_parameters["max_simulation_epoch"],
                sim_stop_time=self.simulation_parameters["simulation_stop_time"],
                time_step=self.simulation_parameters["time_step"],
                fixed_shape=self.fixed_agent_shape,
                low_costs=[self.agent]
            )
           

        print("___________________________________________________________________________")
        print("                           GRN Designer Algorithm                          ")
        print("___________________________________________________________________________")
        print()

        sim_one = 0
        total_epochs = self.evolution_one_epochs + self.evolution_two_epochs
        for i in range(self.evolution_one_epochs):

            population, costs, mean_cost = evolutionary_optimization(
                population=population,
                target=target_,
                population_size=self.population_size,
                cost=cost,
                rates=rates,
                bounds=bounds,
                mutation=mutation_one,
                crossover=crossover,
                num_elite_agents=self.num_elite_agents,
                parameters=parameters,
                cost_constant=self.cost_constant,
                fixed_agent_shape=self.fixed_agent_shape
            )

            self.cost_constant = mean_cost
            sorted_costs = np.sort(costs)
            evolutionary_costs[i, 0] = sorted_costs[0]
            evolutionary_costs[i, 1] = mean_cost

            print(f"Epoch {i+1}/{total_epochs}, Avg/Min Population Cost: {mean_cost}/{sorted_costs[0]}")

            if i % self.interval_save == 0:

                min_cost_index = np.argmin(costs)
                predicted = agent_simulation(
                    agent=population[min_cost_index]
                )
                #print(predicted.shape)
                population[min_cost_index] = reset_agent(agent=population[min_cost_index])

                if self.zoom_:
                    predicted_up = self.reshape_.zoom_in(
                        target=predicted[0],
                        zoom_=self.zoom_out_factor
                    )

                    simulation_results[sim_one] = predicted_up
                    del predicted_up
                else:
                    simulation_results[sim_one] = predicted[0, :, :]

                sim_one += 1

                del predicted, min_cost_index
                gc.collect()

            if not self.zoom_:
                if self.loss_threshold:
                    if sorted_costs[0] < self.loss_threshold:
                        min_cost_index = np.argmin(costs)
                        best_agent = population[min_cost_index]
                        self.save_to_h5py(
                            dataset_name="best_agent_one",
                            data_array=best_agent
                        )
                        self.save_to_h5py(
                            dataset_name="simulation_results",
                            data_array=simulation_results
                        )
                        toc = time.time()
                        run_time[0] = toc - tic
                        self.save_to_h5py(
                            dataset_name="run_time",
                            data_array=run_time
                        )
                        print(f"The loss condition is met! "
                              f"Threshold: {self.loss_threshold}, "
                              f"Achieved Loss: {sorted_costs[0]:.4f}")
                        break


            if i == self.evolution_one_epochs - 1:
                min_cost_index = np.argmin(costs)
                best_agent = population[min_cost_index]
                final_agent = best_agent
                new_population_size = int(self.population_size * self.evolution_two_ratio)
                sorted_cost_indices = np.argsort(costs)[:new_population_size]
                population = [population[idx] for idx in sorted_cost_indices]

                self.save_to_h5py(
                    dataset_name="best_agent_one",
                    data_array=best_agent
                )
               
                del sorted_cost_indices, best_agent, min_cost_index

            del costs, mean_cost, sorted_costs
            gc.collect()

        if self.zoom_:
            population = self.reshape_.zoom_out(
                population=population,
                zoom_=self.zoom_out_factor,
                x_=self.target.shape[0],
                y_=self.target.shape[1]
            )

        print()
        print("___________________________________________________________________________")
        print()

        pop_size = len(population)
        idx = self.evolution_one_epochs
        sim_two = self.evolution_one_epochs // self.interval_save
        for j in range(self.evolution_two_epochs):
            population, costs, mean_cost = evolutionary_optimization(
                population=population,
                target=self.target,
                population_size=pop_size,
                cost=cost,
                rates=rates,
                bounds=bounds,
                mutation=mutation_two,
                crossover=crossover,
                num_elite_agents=self.num_elite_agents,
                fixed_agent_shape=self.fixed_agent_shape,
                parameters=parameters,
                cost_constant=self.cost_constant
            )

            self.cost_constant = mean_cost
            sorted_costs = np.sort(costs)
            evolutionary_costs[idx, 0] = sorted_costs[0]
            evolutionary_costs[idx, 1] = mean_cost

            print(f"Epoch {idx + 1}/{total_epochs}, Avg/Min Population Cost: {mean_cost}/{sorted_costs[0]}")
            idx += 1

            if j % self.interval_save == 0:

                min_cost_index = np.argmin(costs)
                predicted = agent_simulation(
                    agent=population[min_cost_index]
                )
                population[min_cost_index] = reset_agent(agent=population[min_cost_index])

                simulation_results[sim_two] = predicted[0]
                sim_two += 1

                del predicted, min_cost_index
                gc.collect()

            if self.loss_threshold:
                if sorted_costs[0] < self.loss_threshold:
                    min_cost_index = np.argmin(costs)
                    best_agent = population[min_cost_index]
                    self.save_to_h5py(
                        dataset_name="best_agent_one",
                        data_array=best_agent
                    )
                    self.save_to_h5py(
                        dataset_name="simulation_results",
                        data_array=simulation_results
                    )
                    toc = time.time()
                    run_time[0] = toc - tic
                    self.save_to_h5py(
                        dataset_name="run_time",
                        data_array=run_time
                    )
                    print(f"The loss condition is met! "
                            f"Threshold: {self.loss_threshold}, "
                            f"Achieved Loss: {sorted_costs[0]:.4f}")
                    break


            if j == self.evolution_two_epochs - 1:
                min_cost_index = np.argmin(costs)
                final_agent = population[min_cost_index]

                self.save_to_h5py(
                    dataset_name="best_agent_two",
                    data_array=final_agent
                )
                self.save_to_h5py(
                    dataset_name="simulation_results",
                    data_array=simulation_results
                )
                del min_cost_index

            del costs, mean_cost, sorted_costs
            gc.collect()

        self.save_to_h5py(
            dataset_name="evolutionary_costs",
            data_array=evolutionary_costs
        )
        toc = time.time()
        run_time[0] = toc - tic
        self.save_to_h5py(
            dataset_name="run_time",
            data_array=run_time
        )


        if self.gradient_optimization:

            model = GradientOptimization(
                target=torch.from_numpy(self.target),
                path=self.store_path,
                file_name="gradient_result",
                epochs=self.optimization_epochs,
                optimizer=self.gradient_optimizer,
                learning_rate=self.learning_rate,
                param_opt=self.parameter_optimization,
                initial_condition_opt=self.condition_optimization,
                cost_alpha=self.cost_alpha,
                cost_beta=self.cost_beta,
                ssim_data_range=1.0,
                checkpoint_interval=self.checkpoint_interval,
                interval_save=self.interval_save,
                pattern_proportion=self.cost_pattern_proportion,
                lr_decay=self.lr_decay,
                decay_steps=self.decay_steps,
                decay_rate=self.decay_rate,
                device=self.device
            )

            final_agent, _ = model.fit(
                    agent=torch.from_numpy(final_agent)
            )

        return population, final_agent



