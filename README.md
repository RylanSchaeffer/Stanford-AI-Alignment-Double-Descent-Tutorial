# Double Descent Demystified
## Identifying, Interpreting & Ablating the Sources of a Deep Learning Puzzle

This repository contains the code and data for our preprint
["Double Descent Demystified: Identifying, Interpreting \& Ablating the Sources of a Deep Learning Puzzle"](https://arxiv.org/abs/2303.14151).

For a step-by-step explanation, see [the walkthrough](walkthrough.md).

The walkthrough will soon be available on LessWrong (and the Stanford AI Labs Blog?). The walkthrough contains 
visual intuition via polynomial regression, mathematical intuition via ordinary linear regression, and ablations in linear regression on real data.

### Polynomial Regression on Synthetic Data

<p align="middle">
  <img align="top" src="figures/double_descent_polynomial_regression_horizontal.png" width="99%" />
</p>

### Double Descent in Ordinary Linear Regression on Real Data

<p align="middle">
  <img align="top" src="results/real_data/double_descent_dataset=WHO%20Life%20Expectancy.png" width="48%" />
  <img align="top" src="results/real_data/least_informative_singular_value_dataset=WHO%20Life%20Expectancy.png" width="48%" />
</p>

   
### Ablations in Ordinary Linear Regression on Real Data

<p align="middle">
  <img align="top" src="results/real_data_ablations/double_descent_ablations_dataset=WHO%20Life%20Expectancy.png" width="90%" />
</p>


## Attribution

Authors: Rylan Schaeffer, Mikail Khona, Zachary Robertson, Akhilan Boopathy, Kateryna Pistunova, Jason W. Rocks, Ila Rani Fiete, Sanmi Koyejo.