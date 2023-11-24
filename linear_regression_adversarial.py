import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from typing import Tuple

import src.plot
from src.utils import ylim_by_dataset, generate_synthetic_data, load_who_life_expectancy

# Set seed for reproducibility.
np.random.seed(0)


regression_datasets = [
    ("California Housing", datasets.fetch_california_housing),
    ("Diabetes", datasets.load_diabetes),
    ("Student-Teacher", generate_synthetic_data),
    ("WHO Life Expectancy", load_who_life_expectancy),
]


results_dir = "results/real_data_adversarial"
os.makedirs(results_dir, exist_ok=True)

singular_value_cutoffs = np.logspace(-3, 0, 7)

num_repeats = 50
# Chosen for good logarithmic spacing.
adversarial_test_datum_prefactors = [0.0, 0.1, 0.316, 1.0, 3.16, 10.0, 31.6]
adversarial_train_data_prefactors = [0.0, 0.1, 0.316, 1.0, 3.16, 10.0, 31.6]

for dataset_name, dataset_fn in regression_datasets:
    print("On dataset:", dataset_name)

    dataset_results_dir = os.path.join(results_dir, dataset_name)
    os.makedirs(dataset_results_dir, exist_ok=True)

    X, y = dataset_fn(return_X_y=True)

    if len(y.shape) == 1:
        y = y[:, np.newaxis]

    # One ablation will be to make the true underlying relationship linear and noiseless.
    # To do this, we need to know the ideal linear relationship. Unfortunately, we don't have
    # any way to know this in practice, so we'll use all the data as our best guess.
    beta_ideal = np.linalg.inv(X.T @ X) @ X.T @ y

    dataset_loss_unablated_df = []
    dataset_adversarial_test_datum_df = []
    dataset_adversarial_train_data_df = []
    for repeat_idx in range(num_repeats):
        subset_sizes = np.arange(1, 40, 1)
        for subset_size in subset_sizes:
            print(
                f"Dataset: {dataset_name}, repeat_idx: {repeat_idx}, subset_size: {subset_size}"
            )

            # Split the data into training/testing sets
            (
                X_train,
                X_test,
                y_train,
                y_test,
                indices_train,
                indices_test,
            ) = train_test_split(
                X,
                y,
                np.arange(X.shape[0]),
                random_state=repeat_idx,
                test_size=X.shape[0] - subset_size,
                shuffle=True,
            )

            # BEGIN: Ordinary linear regression.
            U, S, Vt = np.linalg.svd(X_train, full_matrices=False, compute_uv=True)
            min_singular_value = np.min(S[S > 0.0])
            S_inverted = 1.0 / S
            S_inverted[S_inverted == np.inf] = 0.0
            beta_hat_unablated = Vt.T @ np.diag(S_inverted) @ U.T @ y_train
            y_train_pred = X_train @ beta_hat_unablated
            train_mse_unablated = mean_squared_error(y_train, y_train_pred)
            y_test_pred = X_test @ beta_hat_unablated
            test_mse_unablated = mean_squared_error(y_test, y_test_pred)
            dataset_loss_unablated_df.append(
                {
                    "Dataset": dataset_name,
                    "Subset Size": subset_size,
                    "Num Parameters": X.shape[1],
                    "Train MSE": train_mse_unablated,
                    "Test MSE": test_mse_unablated,
                    "Repeat Index": repeat_idx,
                }
            )
            # END: Ordinary linear regression.

            # BEGIN: Adversarial test datum.
            for adversarial_test_datum_prefactor in adversarial_test_datum_prefactors:
                adversarial_X_test = np.copy(X_test)
                adversarial_X_test += (
                    adversarial_test_datum_prefactor * Vt[-1, np.newaxis, :]
                )
                adversarial_y_test_pred = adversarial_X_test @ beta_hat_unablated
                test_mse_adversarial_test_datum = mean_squared_error(
                    y_test, adversarial_y_test_pred
                )
                dataset_adversarial_test_datum_df.append(
                    {
                        "Dataset": dataset_name,
                        "Subset Size": subset_size,
                        "Num Parameters": X.shape[1],
                        "Train MSE": train_mse_unablated,
                        "Test MSE": test_mse_adversarial_test_datum,
                        "Repeat Index": repeat_idx,
                        "Prefactor": adversarial_test_datum_prefactor,
                    }
                )
            # End: Adversarial test datum.

            # # BEGIN: Adversarial training data.
            for adversarial_train_data_prefactor in adversarial_train_data_prefactors:
                residuals_train_ideal = y_train - X_train @ beta_ideal
                residuals_train_adversarial = np.copy(residuals_train_ideal)
                residuals_train_adversarial += (
                    adversarial_train_data_prefactor * U[:, np.newaxis, -1]
                )
                y_train_adversarial = X_train @ beta_ideal + residuals_train_adversarial
                beta_hat_adversarial = np.linalg.pinv(X_train) @ y_train_adversarial
                y_train_adversarial_pred = X_train @ beta_hat_adversarial
                train_mse_adversarial_train_data = mean_squared_error(
                    y_train_adversarial, y_train_adversarial_pred
                )
                test_mse_adversarial_train_data = mean_squared_error(
                    y_test, X_test @ beta_hat_adversarial
                )

                dataset_adversarial_train_data_df.append(
                    {
                        "Dataset": dataset_name,
                        "Subset Size": subset_size,
                        "Num Parameters": X.shape[1],
                        "Train MSE": train_mse_adversarial_train_data,
                        "Test MSE": test_mse_adversarial_train_data,
                        "Repeat Index": repeat_idx,
                        "Prefactor": adversarial_train_data_prefactor,
                    }
                )
                pass
            # # END: Adversarial training data.

    dataset_loss_unablated_df = pd.DataFrame(dataset_loss_unablated_df)
    dataset_adversarial_train_data_df = pd.DataFrame(dataset_adversarial_train_data_df)
    dataset_adversarial_test_datum_df = pd.DataFrame(dataset_adversarial_test_datum_df)

    dataset_loss_unablated_df["Num Parameters / Num. Training Samples"] = (
        dataset_loss_unablated_df["Num Parameters"]
        / dataset_loss_unablated_df["Subset Size"]
    )
    dataset_adversarial_train_data_df["Num Parameters / Num. Training Samples"] = (
        dataset_adversarial_train_data_df["Num Parameters"]
        / dataset_adversarial_train_data_df["Subset Size"]
    )
    dataset_adversarial_test_datum_df["Num Parameters / Num. Training Samples"] = (
        dataset_adversarial_test_datum_df["Num Parameters"]
        / dataset_adversarial_test_datum_df["Subset Size"]
    )

    ymin, ymax = ylim_by_dataset[dataset_name]

    plt.close()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.lineplot(
        data=dataset_loss_unablated_df,
        x="Num Parameters / Num. Training Samples",
        y=f"Train MSE",
        label="Train",
        ax=ax,
    )
    sns.lineplot(
        data=dataset_loss_unablated_df,
        x="Num Parameters / Num. Training Samples",
        y=f"Test MSE",
        label="Test",
        ax=ax,
    )
    ax.set_xlabel("Num Parameters / Num. Training Samples")
    ax.set_ylabel("Mean Squared Error")
    ax.axvline(x=1.0, color="black", linestyle="--", label="Interpolation Threshold")
    ax.set_title(f"Dataset: {dataset_name}\nAdversarial Manipulation: None")
    ax.set_ylim(bottom=ymin, top=ymax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    sns.move_legend(obj=ax, loc="upper right")
    src.plot.save_plot_with_multiple_extensions(
        plot_dir=dataset_results_dir, plot_title="unablated"
    )

    plt.close()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.lineplot(
        data=dataset_adversarial_test_datum_df,
        x="Num Parameters / Num. Training Samples",
        y="Train MSE",
        hue="Prefactor",
        legend=False,
        ax=ax,
        palette="PuBu",
    )
    sns.lineplot(
        data=dataset_adversarial_test_datum_df,
        x="Num Parameters / Num. Training Samples",
        y=f"Test MSE",
        hue="Prefactor",
        ax=ax,
        palette="OrRd",
    )
    ax.set_xlabel("Num Parameters / Num. Training Samples")
    ax.set_title(f"Dataset: {dataset_name}\nAdversarial Manipulation: Test Datum")
    ax.axvline(x=1.0, color="black", linestyle="--")
    ax.set_ylim(bottom=ymin, top=ymax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    sns.move_legend(obj=ax, loc="upper right")
    src.plot.save_plot_with_multiple_extensions(
        plot_dir=dataset_results_dir, plot_title="adversarial_test_datum"
    )

    plt.close()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.lineplot(
        data=dataset_adversarial_train_data_df,
        x="Num Parameters / Num. Training Samples",
        y="Train MSE",
        hue="Prefactor",
        legend=False,
        ax=ax,
        palette="PuBu",
    )
    sns.lineplot(
        data=dataset_adversarial_train_data_df,
        x="Num Parameters / Num. Training Samples",
        y=f"Test MSE",
        hue="Prefactor",
        ax=ax,
        palette="OrRd",
    )
    ax.set_xlabel("Num Parameters / Num. Training Samples")
    ax.set_title(f"Dataset: {dataset_name}\nAdversarial Manipulation: Training Data")
    ax.axvline(x=1.0, color="black", linestyle="--")
    ax.set_ylim(bottom=ymin, top=ymax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    sns.move_legend(obj=ax, loc="upper right")
    src.plot.save_plot_with_multiple_extensions(
        plot_dir=dataset_results_dir, plot_title="adversarial_train_data"
    )

print("Finished linear_regression_adversarial.py!")
