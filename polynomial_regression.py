import os
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


# Set style
sns.set_style("whitegrid")

# Set seed for reproducibility.
np.random.seed(0)


num_data_list = [15]
num_features_list = [1, 2, 3, 4, 5, 10, 15, 20, 40, 60]
num_repeat_list = np.arange(5).tolist()

results_dir = 'results/polynomial_regression'
os.makedirs(results_dir, exist_ok=True)

# Create sklearn linear regression object
regr = linear_model.LinearRegression(fit_intercept=False)

low, high = -4., 4.
for num_data in num_data_list:

    # Sample data from between -1 and 1.
    X_train = np.random.uniform(low=low, high=high, size=(num_data, 1))
    X_infinite = np.linspace(low, high, 1000).reshape(-1, 1)
    y = np.cos(X_train)[:, 0]
    y_infinite = np.cos(X_infinite)[:, 0]

    # Plot the data.
    plt.close()
    sns.lineplot(x=X_infinite[:, 0], y=y_infinite, label='True Function')
    sns.scatterplot(x=X_train[:, 0], y=y, s=30, color='k', label='Data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(os.path.join(results_dir, f'data.png'),
                bbox_inches='tight',
                dpi=300)
    plt.show()
    plt.close()

    # Fit a linear regression model.
    regr.fit(X_train, y)
    y_infinite_pred = regr.predict(X_infinite)

    for num_features in num_features_list:

        # Fit a polynomial regression model.
        poly = PolynomialFeatures(degree=num_features, interaction_only=False)
        X_train_poly = poly.fit_transform(X_train)
        X_infinite_poly = poly.fit_transform(X_infinite)
        regr.fit(X_train_poly, y)
        y_infinite_pred = regr.predict(X_infinite_poly)

        # Plot the polynomial fit data.
        plt.close()
        sns.lineplot(x=X_infinite[:, 0], y=y_infinite, label='True Function')
        sns.scatterplot(x=X_train[:, 0], y=y, s=30, color='k', label='Data')
        sns.lineplot(x=X_infinite[:, 0], y=y_infinite_pred, label=f'Num Param={X_train_poly.shape[1]}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.ylim(-1.5, 1.5)
        plt.savefig(os.path.join(results_dir, f'fit_num_params={num_features}.png'),
                    bbox_inches='tight',
                    dpi=300)
        plt.show()
        plt.close()

        # Suppose we had a different realization of data. What would the fit look like?
        for repeat_idx in range(3):
            X_train_clone = np.random.uniform(low=low, high=high, size=(num_data, 1))
            y_clone = np.cos(X_train_clone)[:, 0]
            X_train_clone_poly = poly.fit_transform(X_train_clone)
            regr.fit(X_train_clone_poly, y_clone)
            y_infinite_pred_clone = regr.predict(X_infinite_poly)

            # Plot the polynomial fit data.
            plt.close()
            sns.lineplot(x=X_infinite[:, 0], y=y_infinite, label='True Function')
            sns.scatterplot(x=X_train_clone[:, 0], y=y_clone, s=30, color='k', label='Data')
            sns.lineplot(x=X_infinite[:, 0], y=y_infinite_pred_clone, label=f'Num Param={X_train_poly.shape[1]}')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.ylim(-1.5, 1.5)
            plt.savefig(os.path.join(results_dir, f'fit_num_params={num_features}_repeat={repeat_idx}.png'),
                        bbox_inches='tight',
                        dpi=300)
            plt.show()
            plt.close()