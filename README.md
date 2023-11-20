# Double Descent Demystified: Identifying, Interpreting & Ablating the Sources of a Deep Learning Puzzle

This repository contains the code and data for our preprint
["Double Descent Demystified: Identifying, Interpreting \& Ablating the Sources of a Deep Learning Puzzle"](https://arxiv.org/abs/2303.14151).


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

Note: the code was written for simplicity and understandability.
Minimizing code duplication was intentionally not a priority. 

### Double Descent in Linear Regression

Run [linear_regression_ablations.py](linear_regression_ablations.py).

### Adversarial Training and Test Data in Linear Regression

Run [linear_regression_adversarial.py](linear_regression_adversarial.py).

<p align="middle">
  <img align="top" src="results/real_data_adversarial/California%20Housing/adversarial_test_datum.pdf" width="45%" />
<img align="top" src="results/real_data_adversarial/California%20Housing/adversarial_train_data.pdf" width="45%" />
</p>

### Geometric Intuition for Smallest Non-Zero Singular Value

Run [smallest_nonzero_singular_value.py](smallest_nonzero_singular_value.py)

[](results/smallest_nonzero_singular_value/data_distribution_num_data=9.png)

### Double Descent in Polynomial Regression

Run [polynomial_regression.py](polynomial_regression.py).

## Contributing

Contact Rylan Schaeffer at rylanschaeffer@gmail.com. General preferences:

1. Use `black` to format your code. See here for more information. To install, `pip install black`. To format the repo, run black . from the root directory. 
2. Use type hints as much as possible. 
3. Imports should proceed in two blocks: (1) general python libraries, (2) custom python code. Both blocks should be alphabetically ordered.

## Authorship

Authors: Rylan Schaeffer, Zachary Robertson, Akhilan Boopathy, Mikail Khona, Kateryna Pistunova, Jason W. Rocks, Ila Rani Fiete, Sanmi Koyejo.