# Automated Design and Analysis of Gene Regulatory Networks (GRNs)

![Model Structure](data/grn.png)

## Abstract

Gene Regulatory Networks (GRNs) are essential frameworks for understanding
complex gene interactions and regulatory mechanisms in biological systems. Traditional GRN modeling approaches often rely on deterministic or probabilistic
frameworks to simulate gene expression dynamics, but they typically overlook the
critical role of spatially heterogeneous initial conditions. Synthetic developmental
biology has focused on constructing GRNs to replicate specific spatial patterns, yet
most models assume simplified, uniform distributions of gene products, limiting
biological realism. This work addresses this limitation by introducing an approach
that incorporates variable initial conditions, simulating diverse spatial distributions
of gene activity in a two-dimensional environment.

We developed the GRN-Designer algorithm, an end-to-end hybrid optimization
framework that combines evolutionary algorithms with gradient-based methods
to construct GRNs capable of replicating predefined spatial patterns. By treating
the initial expression levels and spatial activation potential of genes as adjustable
parameters, the algorithm designs GRN structure and optimizes regulatory parameters to achieve target diffusion patterns. Multiple distinct spatial patterns were
tested, each simulated through partial differential equations (PDEs) to model gene
expression dynamics.

The algorithm successfully designed GRNs for all target patterns, ranging from
simple gradients to intricate shapes, demonstrating flexibility in managing diverse
spatial configurations. The constructed GRNs adaptively balanced complexity,
tailoring the number of genes and regulatory interactions to match the intricacies
of each target pattern. Results indicate that our approach effectively combines
computational efficiency with biological realism, producing spatial patterns that
closely resemble their intended designs.

This work establishes a robust framework for GRN design, with promising applications in synthetic biology and tissue engineering. By integrating heterogeneous
spatial initial conditions, our model advances the realism of GRN simulations.
Future enhancements may explore stochastic simulation techniques to account
for biological variability, extend the framework to three-dimensional environments, and improve optimization performance to address more complex biological
systems.



## Key Features

1. **Dual Optimization**:
   - Uses genetic algorithms for global search.
   - Combines with gradient-based methods for fine-tuning locally.

2. **Custom Initial Conditions**:
   - Allows defining spatially varied gene expression levels.

3. **Scalable Complexity**:
   - Simulates various patterns and GRN architectures by adjusting model parameters.




## Requirements

To install the required dependencies for the project, you can use the `requirements.txt` file. 



Install these libraries using `pip`:
```bash
pip install -r requirements.txt
```



## How to Use the Code

1. **Prepare the Code**:
   - Download all `.py` files from the `src` directory and place them in the same folder as your script. 
 ```python
     from grn_designer import GRNDesigner as grnd
     model = grnd(
         target="targen-diffusion-pattern" # a 2d matrix,
         agent="agent" # a 3d matrix as an example agent used for initialization. see 'grn_designer.py' to find out more about it.
         ... # the hyperparameters of the model are set by default but should be adjust based on the model. see 'grn_designer.py' to find out more about them. 
     ) 
     results = model.fit() # run the process
```

For a full guideline on how to use it: [guide.ipynb](src/guide.ipynb)

     
     
     
## Citation

```bibtex
@MastersThesis{loghman-samani-2024-stuttgart,
    author    = {Samani, Loghman},
    title     = {Automated Design and Analysis of Gene Regulatory Networks for Simulating Complex Spatial Patterns},
    school    = {University of Stuttgart},
    year      = {2024},
    type      = {Master's Thesis},
}
```
