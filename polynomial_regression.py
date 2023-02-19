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


num_data_list = [15]
num_features_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 30, 40, 50, 100, 200]
num_repeat_list = list(range(30))

results_dir = 'results/polynomial_regression'
os.makedirs(results_dir, exist_ok=True)

# Create sklearn linear regression object
regr = linear_model.LinearRegression(fit_intercept=True)

def compute_y_from_x(X: np.ndarray):
    return np.add(2. * X, np.cos(X * 25))[:, 0]


low, high = -1., 1.
for num_data in num_data_list:
    mse_list = []
    results_num_data_dir = os.path.join(results_dir, f'num_data={num_data}')
    os.makedirs(results_num_data_dir, exist_ok=True)

    # Generate test data.
    X_test = np.linspace(start=low, stop=high, num=1000).reshape(-1, 1)
    y_test = compute_y_from_x(X_test)

    # Plot the data.
    plt.close()
    sns.lineplot(x=X_test[:, 0], y=y_test, label='True Function')
    # sns.scatterplot(x=X_train[:, 0], y=y_train, s=30, color='k', label='Data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(os.path.join(results_num_data_dir, f'data.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()

    for num_features in num_features_list:
        results_num_features_dir = os.path.join(results_num_data_dir, f'num_features={num_features}')
        os.makedirs(results_num_features_dir, exist_ok=True)
        feature_degrees = 1 + np.arange(num_features).astype(int)
        for repeat_idx in num_repeat_list:

            # Sample training data.
            X_train = np.random.uniform(low=low, high=high, size=(num_data, 1))
            y_train = compute_y_from_x(X_train)

            # Fit a polynomial regression model.
            X_train_poly = scipy.special.eval_legendre(
                feature_degrees,
                X_train)
            X_test_poly = scipy.special.eval_legendre(
                feature_degrees,
                X_test)
            beta_hat = np.linalg.pinv(X_train_poly) @ y_train
            y_train_pred = X_train_poly @ beta_hat
            y_test_pred = X_test_poly @ beta_hat
            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            mse_list.append({
                'Num. Data': num_data,
                'Num. Parameters (Num Features)': num_features,
                'repeat_idx': repeat_idx,
                'Train MSE': train_mse,
                'Test MSE': test_mse,
            })
            print(f'num_data={num_data}, num_features={num_features}, repeat_idx={repeat_idx}, train_mse={train_mse:.4f}, test_mse={test_mse:.4f}')

            # Plot the polynomial fit data.
            plt.close()
            sns.lineplot(x=X_test[:, 0], y=y_test, label='True Function')
            sns.lineplot(x=X_test[:, 0], y=y_test_pred, label=f'Num Param={X_train_poly.shape[1]}')
            sns.scatterplot(x=X_train[:, 0], y=y_train, s=30, color='k', label='Data')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.ylim(-3, 3)
            for extension in ['pdf', 'png']:
                plt.savefig(os.path.join(results_num_features_dir, f'repeat_idx={repeat_idx}.{extension}'),
                            bbox_inches='tight',
                            dpi=300)
            # plt.show()
            plt.close()

    mse_df = pd.DataFrame(mse_list)
    mse_df.to_csv(os.path.join(results_num_data_dir, 'mse.csv'), index=False)
    plt.close()
    sns.lineplot(data=mse_df,
                 x='Num. Parameters (Num Features)',
                 y='Test MSE',
                 label='Test',
                 )
    sns.lineplot(data=mse_df,
                 x='Num. Parameters (Num Features)',
                 y='Train MSE',
                 label='Train',
                 )
    plt.ylabel('Mean Squared Error')
    plt.ylim(bottom=1e-3)
    plt.yscale('log')
    plt.xscale('log')
    plt.title('Polynomial Regression')
    plt.axvline(x=num_data,
                color='black',
                linestyle='--',
                label='Interpolation Threshold')
    plt.legend()
    for extension in ['pdf', 'png']:
        plt.savefig(os.path.join(results_num_data_dir, f'mse_num_data={num_data}.{extension}'),
                    bbox_inches='tight',
                    dpi=300)
    plt.show()
    plt.close()