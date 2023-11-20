# Double Descent Demystified: Identifying, Interpreting & Ablating the Sources of a Deep Learning Puzzle

This repository contains the code and data for our preprint
["Double Descent Demystified: Identifying, Interpreting \& Ablating the Sources of a Deep Learning Puzzle"](https://arxiv.org/abs/2303.14151).

For a step-by-step explanation, see [the walkthrough](walkthrough.md). The walkthrough contains mathematical intuition via ordinary linear regression, visual intuition via polynomial regression, and ablations in linear regression on real data.

- For ordinary linear regression on real and synthetic data with ablations of double descent, see [linear_regression_ablations.py](linear_regression_ablations.py) 
- For polynomial regression on synthetic data, see [polynomial_regression.py](polynomial_regression.py).


<p align="middle">
  <img align="top" src="results/real_data_ablations/WHO%20Life%20Expectancy/unablated.png" width="95%" />
</p>


## Setup

We include a conda environment file but this is honestly overkill. As long as you have `matplotlib`,
`numpy`, `pandas`, `scikit-learn` and `seaborn`, you should be good to go. If you insist on creating
a new conda environment, here are the steps:

(Optional) Update conda:

`conda update -n base -c defaults conda`

Create a conda environment with the required packages:

`conda env create --file environment.yml`

To activate the environment:

`conda activate double_descent`

## Running

### Double Descent in Linear Regression

Run [linear_regression_ablations.py](linear_regression_ablations.py).

### Adversarial Training and Test Data in Linear Regression

Run [linear_regression_adversarial.py](linear_regression_adversarial.py).

### Geometric Intuition for Smallest Non-Zero Singular Value

Run [smallest_nonzero_singular_value.py](smallest_nonzero_singular_value.py)

[](results/smallest_nonzero_singular_value/data_distribution_num_data=9.png)

### Double Descent in Polynomial Regression

Run [polynomial_regression.py](polynomial_regression.py).

## Contributing

1. Use `black` to format your code. See here for more information. To install, `pip install black`. To format the repo, run black . from the root directory. 
2. Use type hints as much as possible. 
3. Imports should proceed in two blocks: (1) general python libraries, (2) custom python code. Both blocks should be alphabetically ordered.

## Authorship

Authors: Rylan Schaeffer, Zachary Robertson, Akhilan Boopathy, Mikail Khona, Kateryna Pistunova, Jason W. Rocks, Ila Rani Fiete, Sanmi Koyejo.