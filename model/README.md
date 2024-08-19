# Development of the algorithm

--------------------
--------------------


## Simulation Algorithm


### Individual simulation

In a genetic algorithm, one approach to simulating a population is to run each individual simulation one by one. This means using a for loop to process one individual in an iteration in the population for every generation.
I developed a system to handle this using the [`individual_simulation(individual)`](https://github.com/LoqmanSamani/master_project/blob/systembiology/model/sim/sim_ind/simulation.py) function. This function takes one individual at a time and returns the target or predicted species compartment after running the simulation based on the individual matrix.


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


![complex matrix](https://github.com/LoqmanSamani/master_project/blob/systembiology/model/figures/com_matrix.jpg)

*Figure2: Complex Information Matrix*


Finally, there is another matrix (individual[-1, :, :]) that includes simulation details in its last row, such as the number of species and complexes, maximum number of simulation epochs, stop time, and time step (dt) (see Figure 3). 
For each species (but not complexes) in the system, there is a specific row in this matrix that contains simulation parameters for that species. 
These parameters include the production rate, degradation rate, and diffusion rate.


![complex matrix](https://github.com/LoqmanSamani/master_project/blob/systembiology/model/figures/sim_infos.jpg)

*Figure3: Simulation and Species Information Matrix*


All these matrices are combined into the individual matrix (Z represents the total number of matrices, related to the number of species). 
The system is then simulated using this individual matrix, and the output is the target matrix, which is the first 2D matrix ([0, :, :]) of the individual matrix after simulation.

-------------------------------------------------------------------------------------------------------------------------

The simulation process begins by extracting the necessary information from the individual matrix (see Figure 4). A while loop is then used to run the simulation, continuing until either the maximum number of epochs (max_epoch) or a specified number of epochs (num_epochs) is reached:

![simulation information](https://github.com/LoqmanSamani/master_project/blob/systembiology/model/figures/sim_info.png)

*Figure4: Extracted Simulation Information from Individual Matrix*

During each epoch of the simulation, the matrices for species and complexes are updated using a for loop that iterates through the compartments (for i in range(x):):

This loop updates each column of the species and complex compartments in each step. The updating process follows these steps:

       
   1)  ***Update Species Production:*** This step applies the production rates to the species matrices using the [`apply_component_production(...)`](https://github.com/LoqmanSamani/master_project/blob/systembiology/model/sim/sim_ind/reactions.py) function.
   2)  ***Update species collision:*** This step updates species interactions and collisions with the [`apply_species_collision(...)`](https://github.com/LoqmanSamani/master_project/blob/systembiology/model/sim/sim_ind/reactions.py) function.
   3)  ***Update species degradation:*** This step applies degradation rates to the species matrices using the [`apply_component_degradation(...)`](https://github.com/LoqmanSamani/master_project/blob/systembiology/model/sim/sim_ind/reactions.py) function.
   4)  ***Update complex degradation:*** This step updates the degradation of complexes using the [`apply_component_degradation(...)`](https://github.com/LoqmanSamani/master_project/blob/systembiology/model/sim/sim_ind/reactions.py) function.
   5)  ***Update complex dissociation:*** This step applies dissociation rates to complexes with the [`apply_component_degradation(...)`](https://github.com/LoqmanSamani/master_project/blob/systembiology/model/sim/sim_ind/reactions.py) function.
   6)  ***Update species diffusion:*** This step applies diffusion rates to the species matrices using the [`apply_diffusion(...)`](https://github.com/LoqmanSamani/master_project/blob/systembiology/model/sim/sim_ind/diffusion.py) function.
   7)  ***Update complex diffusion:***  This step applies diffusion rates to the complex matrices using the [`apply_diffusion(...)`](https://github.com/LoqmanSamani/master_project/blob/systembiology/model/sim/sim_ind/diffusion.py) function.


Each of these steps updates the system based on the input information, including the reaction rates (production, dissociation, degradation, diffusion) and any additional required details, such as the array pattern for production or the entire compartment for diffusion.

After completing the allowed number of epochs, the first matrix of the individual matrix (individual[0, :, :]) is returned for further processing in the algorithm (the specific version of the genetic algorithm that is still under development!).

--------------------------------------------------------------------------------------------------------------------------


### Population simulation

An alternative approach to simulating a population is to perform the simulation for the entire population in parallel, rather than sequentially. This method leverages the power of vectorization to simulate multiple individuals simultaneously.

In the population simulation approach, the [`population_simulation(population)`](https://github.com/LoqmanSamani/master_project/blob/systembiology/model/sim/sim_pop/simulation.py) function processes an entire population as a four-dimensional tensor (or array). The tensor is structured as `(m, z, y, x)`.
Each individual in this tensor is essentially a three-dimensional matrix `(z, y, x)` similar to the one used in the [`individual_simulation(individual)`](https://github.com/LoqmanSamani/master_project/blob/systembiology/model/sim/sim_ind/simulation.py) function. 
The simulation in this case proceeds in a manner similar to the individual simulation, with a small difference:
In each epoch, the simulation processes the compartments column-wise. Instead of updating a specific column in a single individual, the function updates the same column across all individuals simultaneously. As illustrated in ***Figure 5***, each column from every individual is selected and updated concurrently. To facilitate this, all individuals must have the same number of species (z) and the same number of epochs, allowing them to be organized into a 4D tensor for parallel processing.



![pop-sim](https://github.com/LoqmanSamani/master_project/blob/systembiology/model/figures/pop-sim.png)


*Figure 5: Population Simulation*


After completing the allowed number of epochs, the first matrix of each individual of population matrix (`population[:, 0, :, :]`) is returned for further processing in the algorithm.



***[Here](https://github.com/LoqmanSamani/master_project/tree/systembiology/data/sim-model/sim-videos) you can view videos generated from simulations using both the individual and population algorithms.***







### Reactions

The [reactions](https://github.com/LoqmanSamani/master_project/blob/systembiology/model/sim/sim_ind/reactions.py) currently available for simulating a biological system are:

1. **Component Production**: This reaction type is responsible for producing components (such as species or proteins) in the system. These components are created by cells that have a specific production rate for each component.


2. **Component Degradation**: This is the opposite of component production. In this reaction, produced components are broken down or disappear at specific rates. This reaction is also used to degrade formed complexes in the system. The balance between production and degradation determines the quantity of each component in the system.


3. **Component Collision**: When there are multiple components in the system, interactions between them need to be considered. Two specific species (components) can collide at a defined rate, forming a new complex with its own properties. This new complex can move (diffusion) or be degraded like other components in the system.


4. **Component Dissociation**: This reaction type applies only to complexes. It is the opposite of component collision. When a complex is formed, it can also dissociate back into its simple components (A-B <=> A + B). The rate of this reaction can be very small or even zero, meaning no dissociation occurs.




***So, can a biological system be accurately simulated with just these four reaction types?***

The answer is yes! These four reaction types—component production, component degradation, component collision, and component dissociation—provide a solid foundation for modeling a wide range of biological systems. By carefully defining the rates for these reactions, we can simulate complex biological interactions and phenomena.

#### Examples and Further Details:

1. **Component Production and Degradation**:

   - **Example**: Imagine a system where a cell produces a protein that acts as a signaling molecule. We can simulate this by setting a production rate for the protein and a degradation rate to control its lifespan. If we want to simulate a situation where the signaling molecule gradually breaks down, we adjust the degradation rate accordingly.
   
   - **Complexity**: By adjusting these rates, we can explore how the concentration of signaling molecules changes over time and how this impacts other components or processes in the system.

2. **Component Collision**:

   - **Example**: In a cell signaling system, two different proteins might interact to form a complex that triggers a response. For example, Protein A and Protein B might collide to form a Protein A-B complex. By setting a collision rate between Protein A and Protein B, we can simulate how often they interact and how this complex behaves (e.g., it might move within the cell or be degraded).
   
   - **Complexity**: We can also define specific interactions, such as how Protein A interacts with Protein B but not with Protein C. This helps in understanding selective interactions and their effects on cellular processes.

3. **Component Dissociation**:

   - **Example**: Consider a situation where a complex formed by proteins A and B dissociates into its individual components. This process can be simulated by setting a dissociation rate for the A-B complex. If the rate is very low, the complex will remain stable for longer periods; if high, it will break apart quickly.
   
   - **Complexity**: This can be particularly useful in studying processes like receptor-ligand interactions where the ligand binding can be reversible.

4. **Special Cases**:

   - **Anchor Components**: Suppose we want to simulate a protein that acts as an anchor on the cell membrane, meaning it doesn’t move around much. We can achieve this by setting its diffusion rate to nearly zero, ensuring it remains fixed in place. Additionally, if we want this anchor to interact only with a specific ligand, we set the collision rate to zero with all other components, ensuring specificity.
   
   - **Signal Molecules**: For signal molecules that diffuse through the system, we can adjust their diffusion rates to study how quickly they spread and influence other components. For example, a morphogen that diffuses to form a gradient can be simulated by setting a specific diffusion rate and observing how its gradient affects cell behavior.

By manipulating these reaction rates, we can model a variety of biological systems with different types of components and interactions. The flexibility of defining these rates allows us to capture a wide range of biological behaviors and dynamics. ***Figure 6*** provides a schematic representation of an input system for the algorithm, showing how different rates define various types of relationships between species.


![signal-sys](https://github.com/LoqmanSamani/master_project/blob/systembiology/model/figures/signal-sys.png)

*Figure 6: An Individual*



### [Diffusion System](https://github.com/LoqmanSamani/master_project/blob/systembiology/model/sim/sim_ind/diffusion.py)

In simulating a biological system in 2D or 3D, one of the most crucial aspects is modeling the diffusion pattern of each molecule within the compartment (tissue). 
The way a molecule spreads from a cell can vary depending on several factors. For instance, diffusion might occur in a specific direction in response to a source of pressure or energy,
leading to a particular diffusion pattern. Alternatively, it could happen uniformly in all directions, depending on the dimensions of the compartment.

For this project, we've defined the system as a 2D compartment with specific axis lengths. Typically, we assume a square compartment (where x = y) and model each cell within this compartment
as a square (with a = 1 unit). For example, a compartment with dimensions x = 100 and y = 100 contains 100 * 100 cells. This setup means that each cell is in direct contact with four neighboring cells,
allowing material exchange with these adjacent cells. As shown in ***Figure 7***, the compartment (tissue) is illustrated as a 10 * 10 unit area (100 cells).

In our simulation, the diffusion of a molecule within the compartment is influenced primarily by the molecule's diffusion rate, its concentration within a specific cell,
and the presence of other molecules in the system. Some of these other molecules might anchor or trap the diffusing molecule, or they could react with it to produce a new species.
For simplicity, we have excluded all physical forces that could affect material movement within the system (in this case, a biological tissue).

To apply diffusion across all molecules in the system and simulate it over time, we define a small time step (e.g., dt = 0.001). During each epoch (equal to dt) of the simulation,
diffusion is applied throughout the entire compartment, column by column, from left to right (see Figure 7). Each epoch is divided into x steps/iterations (where x = compartment width),
and in each step, one column of the compartment is updated based on the system's diffusion rules.

During each iteration of an epoch, three different functions are used to update the specific column. Two of these diffusion functions are used for the first and last cells in the column,
while the third function handles the diffusion for the cells in between.


![pop-sim](https://github.com/LoqmanSamani/master_project/blob/systembiology/model/figures/diff.jpg)


*Figure 7: Diffusion Pattern*


----------------------------------------------------------------------------------------------------------------------------------

### Simulation Algorithms Benchmarking Results

The benchmarking results, displayed in ***Figure 8***, provide insights into how Numba optimization affects the performance of the simulation algorithm. The results are categorized based on three key factors: compartment size, simulation epochs, and population size.


![bench_m](https://github.com/LoqmanSamani/master_project/blob/systembiology/model/figures/bench-m.png)

*Figure 8: The benchmarking results**


#### Compartment Size

The left-side plot in ***Figure 8*** shows the performance based on compartment size. As the compartment size increases, the time required to complete simulations without Numba (both cases: population_simulation() and individual_population())grows quickly.
In contrast, Numba optimization significantly reduces this time, especially for larger compartments. For small compartments, the time difference between using Numba and not using it is less noticeable, but as the compartment size gets larger, Numba's advantage becomes much clearer.


#### Simulation Epochs

The middle plot illustrates how the number of simulation epochs affects performance. Without Numba (both cases: population_simulation() and individual_population()), increasing the number of epochs leads to a sharp rise in simulation time. With Numba (both cases: population_simulation() and individual_population()), the increase in time is much less steep. This means that Numba is particularly effective in speeding up simulations that run for many epochs, providing substantial time savings for longer simulations.


#### Population Size

The right-side plot compares the performance based on population size. Similar to the previous cases, simulations with larger populations take much longer without Numba (both cases: population_simulation() and individual_population()). Numba optimization reduces the simulation time significantly, particularly for larger populations. For smaller populations, the time difference is smaller, but Numba still offers improvements.

#### Population Simulation vs. Individual Simulation

Surprisingly, the population simulation does not outperform the individual simulation as expected. Despite the anticipation that processing multiple individuals simultaneously would yield faster results, the performance did not meet expectations.

#### Summary

In summary, Numba optimization proves to be highly beneficial, especially when dealing with larger datasets. It consistently reduces the time required for simulations as compartment size, number of epochs, and population size increase. The most noticeable improvements are seen with larger and more complex simulations, making Numba a valuable tool for efficient computation in these scenarios.



--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------


## Genetic Algorithm Based on Natural Selection Theory:



### Mutation Operations ([`apply_mutation(...)`](https://github.com/LoqmanSamani/master_project/blob/systembiology/model/evolution/mutation.py))

In this section of the algorithm, the mutation operation is applied to various parts of the system, ranging from initial conditions (at the compartment level, such as determining if a cell can produce a specific product) to parameters (including those for species and complexes, such as species production, degradation, diffusion, complex collision, and dissociation) to the simulation's hyperparameters (such as the simulation's stop time and the time step, `dt`). 

In addition to these mutations, which modify existing conditions and parameters, the system also includes two unique mutation operations: species deletion and species insertion.

Overall, there are five specific mutation operations in the system:

1. **Initial Compartment Conditions Mutation ([`apply_compartment_mutation(...)`](https://github.com/LoqmanSamani/master_project/blob/systembiology/model/evolution/mutation.py))**: As discussed in the ***Simulation Algorithm*** section of this report, each species in the system is associated with two compartments: one representing the species concentration in each cell and the other defining the species production pattern (i.e., which cells can produce the species). The mutation in this case (`apply_compartment_mutation(...)`) is applied to the production pattern compartment, altering the ability of cells to produce the species (molecule). These compartments are defined as initial condition compartments.

2. **Species and Complex Parameters Mutation ([`apply_parameters_mutation(...)`](https://github.com/LoqmanSamani/master_project/blob/systembiology/model/evolution/mutation.py))**: This mutation alters the parameters of both species and complexes based on a specified mutation rate. 

3. **Simulation Hyperparameters Mutation ([`apply_simulation_parameters_mutation`](https://github.com/LoqmanSamani/master_project/blob/systembiology/model/evolution/mutation.py))**: In this operation, two specific hyperparameters of the [simulation algorithm](https://github.com/LoqmanSamani/master_project/blob/systembiology/model/sim/sim_ind/simulation.py) undergo mutation. The first is the simulation stop time or duration, and the second is the time step for each epoch of the simulation. These hyperparameters determine the simulation steps (simulation steps = simulation duration / time step). Therefore, three hyperparameters of the system are affected by this mutation, with two being directly altered and one indirectly. The fourth hyperparameter, the maximum number of simulation epochs, is not mutated to ensure computational efficiency.

4. **Species Insertion Mutation ([`apply_species_insertion_mutation(...)`](https://github.com/LoqmanSamani/master_project/blob/systembiology/model/evolution/mutation.py))**: To allow the system to adapt and generate diverse end simulation patterns, this mutation adds new species to the system. This increases the system's flexibility and enhances its ability to evolve towards the desired patterns. When a new species is added, the system also generates additional compartments for any potential complexes involving the new species. For example, if the system initially contains two species (A and B), adding a third species (C) will result in the creation of new compartments for complexes A-C and B-C.

5. **Species Deletion Mutation ([`apply_species_deletion_mutation(...)`](https://github.com/LoqmanSamani/master_project/blob/systembiology/model/evolution/mutation.py))**: Complementing the insertion mutation, this mutation simplifies the system by randomly removing a species. This helps regulate system complexity and introduces a mechanism to prevent uncontrolled growth. When a species is deleted, all associated complexes that involve the deleted species are also removed. For instance, if the system contains species A, B, and C along with complexes A-B, B-C, and A-C, deleting species B will also remove complexes A-B and B-C, leaving only species A, C, and the A-C complex.

For the first three mutation types (initial compartment conditions mutation, species and complex parameters mutation, and simulation hyperparameters mutation), the mutation is applied based on specific parameters for each case. Two different distribution options (uniform and normal) can be used to select new values. Each distribution has its own hyperparameters (mean and standard deviation for normal distribution, and minimum and maximum values for uniform distribution) that can be adjusted to refine the mutation process.

Each of these five mutation types is optional and can be selected within the algorithm as needed, providing flexibility in how the system evolves.














