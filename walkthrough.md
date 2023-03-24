# Double Descent Demystified
## Identifying, Interpreting & Ablating the Sources of a Deep Learning Puzzle


### Notation & Terminology

Consider a supervised dataset of $N$ training data for regression:

$$ \mathcal{D} :=  \{ (\vec{x}_n, y_n) \}_{n=1}^N $$

with covariates $\vec{x}_n \in \mathbb{R}^D$ and targets $y_n \in \mathbb{R}$.
We'll sometimes use matrix-vector notation to refer to our training data, treating the
features $\vec{x}_n$ as row vectors:

$$X := \begin{bmatrix} - \vec{x}_1 - \\ \vdots\\ \vec{x}_N - \end{bmatrix} \in \mathbb{R}^{N \times D} \quad \quad \quad \quad Y := \begin{bmatrix} y_1\\ \vdots \\ y_N \end{bmatrix} \in \mathbb{R}^{N \times 1} $$

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

### Mathematical Intuition from Ordinary Linear Regression

To offer an intuitive yet quantitative understanding of double descent, we turn to ordinary linear regression.
Recall that in linear regression, the number of fit parameters $P$ must equal the dimension $D$ of the covariates; 
consequently, rather than thinking about changing the number of parameters $P$, we'll instead think about changing 
the number of data $N$. \textit{Because double descent is fundamentally about the ratio of number of parameters $P$ 
to number of data $N$}, varying the number of data is as valid an approach as varying the number of parameters is. 
To understand where and why double descent occurs in linear regression, we'll study how linear regression behaves in 
the two parameterization regimes. 

If the regression is \textit{underparameterized}, we estimate the linear relationship between the covariates $\vec{x}_n$ and the target $y_n$ by solving the classical least-squares minimization problem:
%
\begin{equation*}
    \hat{\vec{\beta}}_{under} \, \defeq \,  \arg \min_{\vec{\beta}} \frac{1}{N} \sum_n ||\vec{x}_n \cdot \vec{\beta} - y_n||_2^2 \, = \, \arg \min_{\vec{\beta}} ||X \vec{\beta} - Y ||_F^2
\end{equation*}

The solution to this underparameterized optimization problem is the well-known ordinary least squares estimator that uses the second moment matrix $X^T X$:
%
\begin{equation*}
    \hat{\vec{\beta}}_{under} = (X^T X)^{-1} X^T Y
\end{equation*}

If the model is \textit{overparameterized}, the above optimization problem is ill-posed since there are infinitely many solutions; this is because we have fewer constraints than parameters. Consequently, we need to choose a different (constrained) optimization problem:
%
\begin{equation*}
    \hat{\vec{\beta}}_{over} \, \defeq \, \arg \min_{\vec{\beta}} ||\vec{\beta}||_2^2 \quad \quad \text{s.t.} \quad \quad \forall \, n \in \{1, ..., N\} \quad \vec{x}_n \cdot \vec{\beta} = y_n
\end{equation*}

One way to see why the Gram matrix appears is via constrained optimization. Define the Lagrangian with Lagrange multipliers $\vec{\lambda} \in \mathbb{R}^N$:
%
\begin{equation*}
    \mathcal{L}(\vec{\beta}, \vec{\lambda}) \, \defeq \, ||\vec{\beta}||_2^2 + \vec{\lambda}^T (Y - X \vec{\beta})
\end{equation*}

Differentiating with respect to both the parameters and the Lagrange multipliers yields:
%
\begin{align*}
    \nabla_{\vec{\beta}}\,  \mathcal{L}(\vec{\beta}, \vec{\lambda}) = \vec{0} = 2\hat{\vec{\beta}} - X^T \vec{\lambda} &\Rightarrow \hat{\vec{\beta}}_{over} = \frac{1}{2} X^T \vec{\lambda}\\
    \nabla_{\vec{\lambda}} \,\mathcal{L}(\beta, \lambda) = \vec{0} = Y - X \hat{\vec{\beta}}_{over} &\Rightarrow Y = \frac{1}{2} X X^T \vec{\lambda}\\
    &\Rightarrow \vec{\lambda} = 2 (X X^T)^{-1} Y\\
    &\Rightarrow \hat{\vec{\beta}}_{over} = X^T (X X^T)^{-1} Y
\end{align*}

Here, we are able to invert the Gram matrix because it is full rank in the overparametrized regime.
After fitting its parameters, the model will make the following predictions for given test point $\vec{x}_{test}$:
%
\begin{align*}
    \hat{y}_{test, under} &= \vec{x}_{test} \cdot \hat{\Vec{\beta}}_{under} = \vec{x}_{test} \cdot (X^T X)^{-1} X^T Y\\
    \hat{y}_{test, over} &= \vec{x}_{test} \cdot \hat{\Vec{\beta}}_{over} 
    = \vec{x}_{test} \cdot X^T (X X^T)^{-1} Y
\end{align*}

\textit{Hidden in the above equations is an interaction between three quantities that can, when all grow extreme, create double descent.} To reveal the three quantities, we'll rewrite the regression targets by introducing a slightly more detailed notation. Unknown to us, there are some ideal linear parameters $\vec{\beta}^* \in \mathbb{R}^P = \mathbb{R}^D$ that truly minimize the test mean squared error. We can write any regression target as the inner product of the data $\vec{x}_n$ and the ideal parameters $\beta^*$, plus an additional error term $e_n$ that is an ``uncapturable" residual from the ``perspective" of the model class:
%
\begin{equation*}
    y_n = \vec{x}_n \cdot \vec{\beta}^* + e_n
\end{equation*}

In matrix-vector form, we will equivalently write:
%
\begin{equation*}
    Y = X \vec{\beta}^* + E
\end{equation*}

with $E \in \mathbb{R}^{N \times 1}$. To be clear, we are \textit{not} imposing assumptions on the model or data. Rather, we are introducing notation to express that there are (unknown) ideal linear parameters, and possibly residuals that even the ideal model might be unable to capture; these residuals could be random noise or could be fully deterministic patterns that this particular model class cannot capture. Using this new notation, we rewrite the model's predictions to show how the test datum's features $\vec{x}_{test}$, training data's features $X$ and training data's regression targets $Y$ interact. In the underparameterized regime:
%
\begin{align*}
    \hat{y}_{test,under} &= \Vec{x}_{test} \cdot (X^T X)^{-1} X^T Y\\
    &= \Vec{x}_{test} \cdot (X^T X)^{-1} X^T (X \beta^* + E)\\
    &= \Vec{x}_{test} \cdot (X^T X)^{-1} X^T X \beta^* + \vec{x}_{test} \cdot (X^T X)^{-1} X^T E\\
    &= \underbrace{\Vec{x}_{test} \cdot \beta^*}_{\defeq y_{test}^*} + \, \vec{x}_{test} \cdot (X^T X)^{-1} X^T E\\
    \hat{y}_{test,under} - y_{test}^* &= \vec{x}_{test} \cdot (X^T X)^{-1} X^T E
    % \hat{y}_{test,over} &= \underbrace{\vec{x}_{test} \cdot \vec{\beta}^*}_{\defeq y_{test}^*} \quad + \quad \Vec{x}_{test} \cdot \underbrace{X^T (X X^T)^{-1}}_{\defeq X^+} E
\end{align*}

This equation is important, but opaque. To extract the intuition, we will replace $X$ with its \href{https://en.wikipedia.org/wiki/Singular_value_decomposition}{singular value decomposition}\footnote{For those unfamiliar with the SVD, any real-valued $X$ can be decomposed into the product of three matrices $X = U \Sigma V^T$ where $U$ and $V$ are both orthonormal matrices and $\Sigma$ is diagonal; intuitively, any linear transformation can be viewed as composition of first a rotoreflection, then a scaling, then another rotoreflection. Because $\Sigma$ is diagonal and because $U, V$ are orthonormal matrices, we can equivalently write $X = U \Sigma V^T$ in vector-summation notation as a sum of rank-1 outer products $X = \sum_{r=1}^{rank(X)} \sigma_r u_r v_r^T$. Each term in the sum is referred to as a ``singular mode", akin to eigenmodes.} $X = U \Sigma V^T$ to reveal how different quantities interact. Let $R \, \defeq \, rank(X)$ and let $\sigma_1 > \sigma_2 > ... > \sigma_R > 0$ be $X$'s (non-zero) singular values. Recalling $E \in \mathbb{R}^{N \times 1}$, we can decompose the (underparameterized) prediction error $\hat{y}_{test, under} - y_{test}^*$ along the orthogonal singular modes:
%
\begin{align*}
    \hat{y}_{test, under} - y_{test}^* &= \Vec{x}_{test} \cdot V \Sigma^{+} U^T E = \sum_{r=1}^R  \frac{1}{\sigma_r} (\Vec{x}_{test} \cdot \vec{v}_r) (\vec{u}_r \cdot E)
\end{align*}

In the overparameterized regime, our calculations change slightly:
%
\begin{align*}
    \hat{y}_{test,over} &= \Vec{x}_{test} \cdot X^T (X X^T)^{-1}  Y\\
    &= \vec{x}_{test} \cdot X^T (X X^T)^{-1} (X \beta^* + E)\\
    &= \vec{x}_{test} \cdot X^T (X X^T)^{-1} X \beta^* + \vec{x}_{test} \cdot X^T (X X^T)^{-1} E\\
    \hat{y}_{test,over} - \underbrace{\vec{x}_{test} \cdot \beta^*}_{\defeq y_{test}^*} &= \vec{x}_{test} \cdot X^T (X X^T)^{-1} X \beta^*  - \vec{x}_{test} \cdot I_{D} \beta^* + \vec{x}_{test} \cdot (X^T X)^{-1} X^T E\\
    \hat{y}_{test,over} - y_{test}^* &= \vec{x}_{test} \cdot (X^T (X X^T)^{-1} X - I_D) \beta^*  + \vec{x}_{test} \cdot (X^T X)^{-1} X^T E
\end{align*}

If we again replace $X$ with its SVD $U S V^T$, we can again simplify $\vec{x}_{test} \cdot (X^T X)^{-1} X^T E$. This yields our final equations for the prediction errors.
%
\begin{align*}
\hat{y}_{test,over} - y_{test}^* &= \vec{x}_{test} \cdot (X^T (X X^T)^{-1} X - I_D) \beta^* \quad \quad \quad \quad + && \sum_{r=1}^R  \frac{1}{\sigma_r} (\Vec{x}_{test} \cdot \vec{v}_r) (\vec{u}_r \cdot E)\\
    \hat{y}_{test,under} - y_{test}^* &= &&\sum_{r=1}^R  \frac{1}{\sigma_r} (\Vec{x}_{test} \cdot \vec{v}_r) (\vec{u}_r \cdot E)
\end{align*}

What is the discrepancy between the underparameterized prediction error and the overparameterized prediction error, and from where does the discrepancy originate? The overparameterized prediction error $\hat{y}_{test,over} - y_{test}^*$ has the extra term $\vec{x}_{test} \cdot (X^T (X X^T)^{-1} X - I_D) \beta^*$. To understand where this term originates, recall that our goal is to understand how fluctuations in the features $\vec{x}$ correlate with fluctuations in the targets $y$. In the overparameterized regime, there are more parameters than there are data. Consequently, for $N$ data points in $D=P$ dimensions, the model can ``see" fluctuations in at most $N$ dimensions, but has no ``visibility" into fluctuations in the remaining $P-N$ dimensions. This causes information about the optimal linear relationship $\vec{\beta}^*$ to be lost, which in turn increases the overparameterized prediction error $\hat{y}_{test, over} - y_{test}^*$. Statisticians call this term $\vec{x}_{test} \cdot (X^T (X X^T)^{-1} X - I_D) \beta^*$ the ``bias". The other term (the ``variance") is what causes double descent:
%
\begin{equation}
    \sum_{r=1}^R  \frac{1}{\sigma_r} (\Vec{x}_{test} \cdot \vec{v}_r) (\vec{u}_r \cdot E)
    \label{eqn:variance}
\end{equation}

\textit{Eqn. \ref{eqn:variance} is critical}. It reveals that our test prediction error (and thus, our test squared error!) will depend on an interaction between 3 quantities:
%

1. How much the \textit{training features} $X$ vary in each direction; more formally, the inverse (non-zero) singular values of the \textit{training features} $X$:
%
$$\frac{1}{\sigma_r}$$

2. How much, and in which directions, the \textit{test features} $\vec{x}_{test}$ vary relative to the \textit{training features} $X$; more formally: how $\vec{x}_{test}$ projects onto $X$'s right singular vectors $V$:
%
$$\Vec{x}_{test} \cdot \Vec{v}_r$$

3. How well the \textit{best possible model} can correlate the variance in the \textit{training features} $X$ with the \textit{training regression targets} $Y$; more formally: how the residuals $E$ of the best possible model in the model class (i.e. insurmountable ``errors" from the ``perspective" of the model class) project onto $X$'s left singular vectors $U$:
%
$$\Vec{u}_r \cdot E$$
    