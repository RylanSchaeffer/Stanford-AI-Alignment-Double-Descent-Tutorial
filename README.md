# Double Descent Demystified
## Identifying, Interpreting & Ablating the Sources of a Deep Learning Puzzle

This repository contains the code and data for our preprint
["Double Descent Demystified: Identifying, Interpreting \& Ablating the Sources of a Deep Learning Puzzle"](https://arxiv.org/abs/2303.14151).

For a step-by-step explanation, see [the walkthrough](walkthrough.md). The walkthrough contains mathematical intuition via ordinary linear regression, visual intuition via polynomial regression, and ablations in linear regression on real data.

- For ordinary linear regression on real and synthetic data, see [linear_regression.py](linear_regression.py).
- For ordinary linear regression on real and synthetic data with ablations of double descent, see [linear_regression_ablations.py](linear_regression_ablations.py) 
- For polynomial regression on synthetic data, see [polynomial_regression.py](polynomial_regression.py).


<p align="middle">
  <img align="top" src="results/real_data_ablations/WHO%20Life%20Expectancy/unablated.png" width="95%" />
</p>


## Authorship

Authors: Rylan Schaeffer, Mikail Khona, Zachary Robertson, Akhilan Boopathy, Kateryna Pistunova, Jason W. Rocks, Ila Rani Fiete, Sanmi Koyejo.