import os
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.special
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


# Set style
sns.set_style("whitegrid")

# Set seed for reproducibility.
np.random.seed(0)


num_data_list = [20]
num_features_list = [1, 2, 3, 4, 5, 10, 15, 20, 40, 60]
num_repeat_list = np.arange(5).tolist()

results_dir = 'results/polynomial_regression'
os.makedirs(results_dir, exist_ok=True)

# Create sklearn linear regression object
regr = linear_model.LinearRegression(fit_intercept=True)

def compute_y_from_x(X: np.ndarray):
    return np.add(2. * X, np.cos(X * 25))[:, 0]


low, high = -1., 1.
for num_data in num_data_list:

    # Sample data from between low and high.
    X_train = np.random.uniform(low=low, high=high, size=(num_data, 1))
    X_test = np.linspace(low, high, 1000).reshape(-1, 1)
    y_train = compute_y_from_x(X_train)
    y_test = compute_y_from_x(X_test)

    # Plot the data.
    plt.close()
    sns.lineplot(x=X_test[:, 0], y=y_test, label='True Function')
    sns.scatterplot(x=X_train[:, 0], y=y_train, s=30, color='k', label='Data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(os.path.join(results_dir, f'data.png'),
                bbox_inches='tight',
                dpi=300)
    plt.show()
    plt.close()

    # # Fit a linear regression model.
    # regr.fit(X_train, y_train)
    # y_test_pred = regr.predict(X_test)

    for num_features in num_features_list:

        # Fit a polynomial regression model.
        poly = PolynomialFeatures(degree=num_features, interaction_only=False)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.fit_transform(X_test)
        # regr.fit(X_train_poly, y)
        # if num_features < num_data:
        beta_hat = np.linalg.pinv(X_train_poly) @ y_train
        # y_test_pred = regr.predict(X_test_poly)
        y_test_pred = X_test_poly @ beta_hat

        # Plot the polynomial fit data.
        plt.close()
        sns.lineplot(x=X_test[:, 0], y=y_test, label='True Function')
        sns.scatterplot(x=X_train[:, 0], y=y_train, s=30, color='k', label='Data')
        sns.lineplot(x=X_test[:, 0], y=y_test_pred, label=f'Num Param={X_train_poly.shape[1]}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.ylim(-1.5, 1.5)
        plt.savefig(os.path.join(results_dir, f'fit_num_params={num_features}.png'),
                    bbox_inches='tight',
                    dpi=300)
        plt.show()
        plt.close()

        # Suppose we had a different realization of data. What would the fit look like?
        for repeat_idx in range(1):
            X_train_clone = np.random.uniform(low=low, high=high, size=(num_data, 1))
            y_clone = np.cos(X_train_clone)[:, 0]
            X_train_clone_poly = poly.fit_transform(X_train_clone)
            regr.fit(X_train_clone_poly, y_clone)
            y_test_pred_clone = regr.predict(X_test_poly)

            # Plot the polynomial fit data.
            plt.close()
            sns.lineplot(x=X_test[:, 0], y=y_test, label='True Function')
            sns.scatterplot(x=X_train_clone[:, 0], y=y_clone, s=30, color='k', label='Data')
            sns.lineplot(x=X_test[:, 0], y=y_test_pred_clone, label=f'Num Param={X_train_poly.shape[1]}')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.ylim(-1.5, 1.5)
            plt.savefig(os.path.join(results_dir, f'fit_num_params={num_features}_repeat={repeat_idx}.png'),
                        bbox_inches='tight',
                        dpi=300)
            plt.show()
            plt.close()