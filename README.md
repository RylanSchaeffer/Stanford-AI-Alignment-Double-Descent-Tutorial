# An Introductory Tutorial on Double Descent
## What Double Descent Is and Why It Occurs

Authors: Rylan Schaeffer, etc.

This repository contains the code and data for our preprint 
"An Introductory Tutorial on Double Descent: What Double Descent Is and Why It Occurs".
The tutorial will be available on arXiv (and LessWrong?).

[Overleaf Draft](https://www.overleaf.com/read/xscfwstjdqwy)

## Polynomial Regression on Synthetic Data

<p align="middle">
  <img align="top" src="figures/double_descent_polynomial_regression_horizontal.png" width="99%" />
</p>

## Ordinary Linear Regression on Real Data

<p align="middle">
  <img align="top" src="results/real_data/double_descent_dataset=WHO%20Life%20Expectancy.png" width="48%" />
  <img align="top" src="results/real_data/least_informative_singular_value_dataset=WHO%20Life%20Expectancy.png" width="48%" />
</p>

## Mathematical Explanation in Ordinary Linear Regression

Let $X \in \mathbb{R}^{N \times D}, Y \int \mathbb{R}^{N \times 1}$ denote our training data features and targets, respectively.
Let $X := U S V^T$ denote the singular value decomposition of $X$ with rank $R$, and let $E = Y - X \beta^*$ denote the
residuals of the best possible model in the model class (i.e. insurmountable "errors" from the "perspective" of the model class).
_The below equation is critical_. It reveals that our test prediction error (and thus, our test squared error!) will
depend on an interaction between 3 quantities:

$$\sum_{r=1}^R  \frac{1}{\sigma_r} (\vec{x}_{test} \cdot \vec{v}_r) (\vec{u}_r \cdot E)$$

1. How much the training features $X$ vary in each direction; more formally, the inverted (non-zero) singular values of the _training features_ $X$:

$$\frac{1}{\sigma_r}$$
    
2. How much, and in which directions, the _test features_ $\vec{x}_{test}$ vary relative to the _training features_ $X$; more formally: how $\vec{x}_{test}$ projects onto $X$'s right singular vectors $V$:
    
    $$\vec{x}_{test} \cdot \vec{v}_r$$
    
3. How well the _best possible model in the model class_ can correlate the variance in the _training features_ $X$ with the _training regression targets_ $Y$; more formally: how the residuals $E$ of the best possible model project onto $X$'s left singular vectors $U$:
    
    $$\vec{u}_r \cdot E$$
   
## Causal Ablations on Real Data

<p align="middle">
  <img align="top" src="results/real_data_ablations/double_descent_ablations_dataset=California%20Housing.png" width="99%" />
  <img align="top" src="results/real_data_ablations/double_descent_ablations_dataset=Diabetes.png" width="99%" />
  <img align="top" src="results/real_data_ablations/double_descent_ablations_dataset=WHO%20Life%20Expectancy.png" width="99%" />
</p>


**Double descent will not occur if any of the three critical quantities are absent.** 
We demonstrate this via causal ablations. Left to Right: Double descent appears in ordinary linear regression. 
Removing small singular values in the training features $$X$$ prevents double descent. 
Preventing the test features $$\vec{x}_{test}$$ from varying in the trailing singular modes of the training features $$X$$
prevents double descent. Ensuring that the optimal model in the model class has zero residual
prediction errors $$E$$ prevents double descent.