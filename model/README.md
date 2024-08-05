# Developing The Algorithm

## Simulation Algorithm


### Individual simulation

In a genetic algorithm, one approach to simulating a population is to run each individual simulation one by one. This means using a for loop to process one individual in an iteration in the population for every generation. I developed a system to handle this using the [`individual_simulation(individual)`](https://github.com/LoqmanSamani/master_project/blob/systembiology/model/sim/simulation.py) function. This function takes one individual at a time and returns the target or predicted species compartment after running the simulation based on the individual matrix.


#### individual matrix:

The input for the individual simulation function is a three-dimensional matrix known as the ***individual matrix*** (see Figure 1).


![individual matrix](https://github.com/LoqmanSamani/master_project/blob/systembiology/model/figures/ind_matrix.png)

*Figure1: Individual Matrix*


The matrix has the shape (Z, Y, X), where (Y, X) defines the compartment size that can be simulated. In biological terms, a compartment represents a tissue with a specific number of cells. 
The matrix tracks the behavior of components such as morphogens, enzymes, signal molecules, or proteins over time. For each species (component) in the system, there are two matrices of shape (Y, X).
The first matrix represents the compartment of that species (e.g., 1a and 2a in Figure 1), and the second matrix is the initial condition matrix (2a and 2b in Figure 1).
This matrix shows the secretion pattern of the species, indicating the ability of each cell in the compartment to produce that specific species and also the initial concentration of that species in the cell.
If there are three species in the system, there will be six matrices (two for each species). 


Additionally, there are matrices for complexes—combinations of species created through interactions. Each complex has two matrices: one for its compartment (e.g., 3a in Figure 1) and another for specific complex parameters (3b in Figure 1).
Figure 2 shows that the first two cells in this matrix (index[0, 0:2]) correspond to the indices of the compartment matrices for the two species involved in forming the complex.
The second row provides parameters for simulating the complex, including collision rate, dissociation rate, degradation rate, and diffusion rate.


![complex matrix](https://github.com/LoqmanSamani/master_project/blob/systembiology/model/figures/com_matrix.png)

*Figure2: Complex Information Matrix*


Finally, there is another matrix (individual[-1, :, :]) that includes simulation details in its last row, such as the number of species and complexes, maximum number of simulation epochs, stop time, and time step (dt) (see Figure 3). 
For each species (but not complexes) in the system, there is a specific row in this matrix that contains simulation parameters for that species. 
These parameters include the production rate, degradation rate, and diffusion rate.


![complex matrix](https://github.com/LoqmanSamani/master_project/blob/systembiology/model/figures/sim_infos.png)

*Figure3: Simulation and Species Information Matrix*


All these matrices are combined into the individual matrix (Z represents the total number of matrices, related to the number of species). 
The system is then simulated using this individual matrix, and the output is the target matrix, which is the first 2D matrix ([0, :, :]) of the individual matrix after simulation.

-------------------------------------------------------------------------------------------------------------------------

The simulation process begins by extracting the necessary information from the individual matrix (see Figure 4). A while loop is then used to run the simulation, continuing until either the maximum number of epochs (max_epoch) or a specified number of epochs (num_epochs) is reached:

![simulation information](https://github.com/LoqmanSamani/master_project/blob/systembiology/model/figures/sim_info.png)

*Extracted Simulation Information from Individual Matrix*

During each epoch of the simulation, the matrices for species and complexes are updated using a for loop that iterates through the compartments (for i in range(x):):

This loop updates each column of the species and complex compartments in each step. The updating process follows these steps:

       
   1)  ***Update Species Production:*** This step applies the production rates to the species matrices using the [`apply_component_production(...)`](https://github.com/LoqmanSamani/master_project/blob/systembiology/model/sim/reactions.py) function.
   2)  ***Update species collision:*** This step updates species interactions and collisions with the [`apply_species_collision(...)`](https://github.com/LoqmanSamani/master_project/blob/systembiology/model/sim/reactions.py) function.
   3)  ***Update species degradation:*** This step applies degradation rates to the species matrices using the [`apply_component_degradation(...)`](https://github.com/LoqmanSamani/master_project/blob/systembiology/model/sim/reactions.py) function.
   4)  ***Update complex degradation:*** This step updates the degradation of complexes using the [`apply_component_degradation(...)`](https://github.com/LoqmanSamani/master_project/blob/systembiology/model/sim/reactions.py) function.
   5)  ***Update complex dissociation:*** This step applies dissociation rates to complexes with the [`apply_component_degradation(...)`](https://github.com/LoqmanSamani/master_project/blob/systembiology/model/sim/reactions.py) function.
   6)  ***Update species diffusion:*** This step applies diffusion rates to the species matrices using the [`apply_diffusion(...)`](https://github.com/LoqmanSamani/master_project/blob/systembiology/model/sim/diffusion.py) function.
   7)  ***Update complex diffusion:***  This step applies diffusion rates to the complex matrices using the [`apply_diffusion(...)`](https://github.com/LoqmanSamani/master_project/blob/systembiology/model/sim/diffusion.py) function.


Each of these steps updates the system based on the input information, including the reaction rates (production, dissociation, degradation, diffusion) and any additional required details, such as the array pattern for production or the entire compartment for diffusion.

After completing the allowed number of epochs, the first matrix of the individual matrix (individual[0, :, :]) is returned for further processing in the algorithm (the specific version of the genetic algorithm that is still under development!).



### Benchmarking Results

The benchmarking results, displayed in ***Figure 5***, provide insights into how Numba optimization affects the performance of the simulation algorithm. The results are categorized based on three key factors: compartment size, simulation epochs, and population size.


![bench_ma](https://github.com/LoqmanSamani/master_project/blob/systembiology/model/figures/bench_ma.png)



#### Compartment Size

The left-side plot in Figure 5 shows the performance based on compartment size. As the compartment size increases, the time required to complete simulations without Numba grows quickly. In contrast, Numba optimization significantly reduces this time, especially for larger compartments. For small compartments, the time difference between using Numba and not using it is less noticeable, but as the compartment size gets larger, Numba's advantage becomes much clearer.


#### Simulation Epochs

The middle plot illustrates how the number of simulation epochs affects performance. Without Numba, increasing the number of epochs leads to a sharp rise in simulation time. With Numba, the increase in time is much less steep. This means that Numba is particularly effective in speeding up simulations that run for many epochs, providing substantial time savings for longer simulations.


#### Population Size

The right-side plot compares the performance based on population size. Similar to the previous cases, simulations with larger populations take much longer without Numba. Numba optimization reduces the simulation time significantly, particularly for larger populations. For smaller populations, the time difference is smaller, but Numba still offers improvements.

#### Summary

In summary, Numba optimization proves to be highly beneficial, especially when dealing with larger datasets. It consistently reduces the time required for simulations as compartment size, number of epochs, and population size increase. The most noticeable improvements are seen with larger and more complex simulations, making Numba a valuable tool for efficient computation in these scenarios.




























