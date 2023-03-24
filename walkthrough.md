# Double Descent Demystified
## Identifying, Interpreting & Ablating the Sources of a Deep Learning Puzzle


### Notation & Terminology

Consider a supervised dataset of $N$ training data for regression:

$$ \mathcal{D} :=  \{ (\vec{x}_n, y_n) \}_{n=1}^N $$

with covariates $\vec{x}_n \in \mathbb{R}^D$ and targets $y_n \in \mathbb{R}$.
We'll sometimes use matrix-vector notation to refer to our training data, treating the
features $\vec{x}_n$ as row vectors:

$$X := \begin{bmatrix} - \vec{x}_1 - \ \vdots\ \vec{x}_N - \end{bmatrix} \in \mathbb{R}^{N \times D} \quad \quad \quad \quad Y := \begin{bmatrix} y_1\ \vdots \ y_N \end{bmatrix} \in \mathbb{R}^{N \times 1} $$

In general, our goal is to use our training dataset $\mathcal{D}$ find a function $f: \mathcal{X} \rightarrow \mathcal{Y}$ that makes:

$$f(x) \approx y $$

In the setting of ordinary linear regression, we assume that $f$ is a linear function
i.e. $f(\vec{x}) = \vec{x} \cdot \vec{\beta}$, meaning our goal is to find (estimate) linear
parameters $\hat{\vec{\beta}} \in \mathbb{R}^{D}$ that make: 

$$\begin{equation*} \vec{x} \cdot \vec{\beta} \approx y \end{equation*}$$

Of course, our real goal is to hopefully find a function that generalizes well to new data. 
As a matter of terminology, there are typically three key parameters:

1. The number of model parameters $P$ 
2. The number of training data $N$
3. The dimensionality of the data $D$

We say that a model is _overparameterized_ (a.k.a. underconstrained) if $N < P$ and _underparameterized_ (a.k.a. overconstrained) if $N > P$. 
The _interpolation threshold_ refers to where $N=P$, because when $P\geq N$, the model can perfectly interpolate the training points.