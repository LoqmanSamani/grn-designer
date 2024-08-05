# Developing The Algorithm

## Simulation Algorithm



### Individual simulation

one way to simulate a population of genetic algorithm is to simulate the population individual after individual, which means using a for loop to simulate the entire population in each generation of genetic algorithm. The simulation system, which I developed to reach this matter is [`individual_simulation(individual)`](https://github.com/LoqmanSamani/master_project/blob/systembiology/model/sim/simulation.py).
the function accepts one individual per run as input and reterns the target or predicted species compartment after simulating the system with information gained from individual matrix.

#### individual matrix:

the input matrix of individual simulation function is a three dimensional matrix contains all information which is needed to simulate the system with some perticular conditions mentioned in the matrix called "individual matrix" (figure 1).



![individual matrix](https://github.com/LoqmanSamani/master_project/blob/systembiology/model/figures/ind_matrix.png)

*Individual Matrix*


the shape of the matrix is equal to: (Z, Y, X), in which (Y, X) is equal to compartment shape, in which system can be simulated. In a biological manner, compartment can be interpreted as a tissue with a specific amount of cells, in which the behaviour of specific components like morphogen, enzymes, signal molecules or proteins involved in a cascade, can be tracked over time. for each species(component) in the system there are two matrices(each with the shape of(Y, X) defined, the first matrix corresponds for compartment of that perticular species (1a and 2a in fogure 1) and the second matrix is called ***initial condition matrix*** (2a and 2b in figure 1) which defines the secretion pattern of that species, in other words, this matrix defines the ability of each cell in the compartment to produce that specific species and also the initial concentration of that species in the cell.


if there are three species in the system, there will be also 6 matrices of shape (Y, X) for them (each species needs two matrices). beside matrix of species, there are matrices for complexes, which are species created through simulation based on the behaviour of species together. in other words there will be a new complex compound in the system for each two species (species1-species2, species1-species3, species2-species3). these complexes are actually products of collision of species. for each of these complexes there are two matrices defined in the individual matrix: one defined ad compartment matrix of that specific complex(the same as species compartment matrix) (3a in figure 1) and the second one contains some specific information about that perticular complex (3b in figure 1).

As illustrated in figure 2, the first two cells in this matrix (index[0, 0:2]) are correspond to the indices of the compartment matrices of those two specific species which collided together to create this specific complex. the second row of this matrix is corresponded to the parameters of the complex, which are needed to simulate the complex in the system, these parameters are: collision rate(rate in which two species collided together), dissociation rate(rate in which complex will collapse to the species), degradation rate(rate in which complex will be disappeared) and diffusion rate(rate of diffusion of the complex in the compartment).


![complex matrix](https://github.com/LoqmanSamani/master_project/blob/systembiology/model/figures/com_matrix.png)

*Complex Information Matrix*



beside these matrices there is another, last one, matrix which contains simulation informatrion in the last row of it (these are from left to right: number of species in the system, number of complexes in the system, maximum nuber of simulation epochs, stop time of the simulation, and the last one time step(dt)), figure 3.


![complex matrix](https://github.com/LoqmanSamani/master_project/blob/systembiology/model/figures/sim_infos.png)

*Simulation and Species Information Matrix*


all the defined matrices are collected together in the individual matrix (Z: is actually the number of these matrices which is relative to the number of species in the system). the system will be simulated with those information in the individual matrix, which is the input of the simulation function and the target matrix which should be the first 2d matrix ([0, :, :]) of the individual matrix after simulation.































