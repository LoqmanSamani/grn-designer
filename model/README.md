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































