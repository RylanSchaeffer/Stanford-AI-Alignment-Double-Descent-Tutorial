import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from src.utils import generate_synthetic_data, load_who_life_expectancy


# Set style
sns.set_style("whitegrid")

# Set seed for reproducibility.
np.random.seed(0)


# Create sklearn linear regression object
ideal_regr = linear_model.LinearRegression(fit_intercept=False)
regr = linear_model.LinearRegression(fit_intercept=False)


regression_datasets = [
    ("Student-Teacher", generate_synthetic_data),
    ("California Housing", datasets.fetch_california_housing),
    (
        "Diabetes",
        datasets.load_diabetes,
    ),
    ("WHO Life Expectancy", load_who_life_expectancy),
]


results_dir = "results/real_data"
os.makedirs(results_dir, exist_ok=True)

num_repeats = 30
for dataset_name, dataset_fn in regression_datasets:
    print("On dataset:", dataset_name)

    dataset_results_dir = os.path.join(results_dir, dataset_name)
    os.makedirs(dataset_results_dir, exist_ok=True)

    # Load the diabetes dataset
    X_all, y_all = dataset_fn(return_X_y=True)
    if len(y_all.shape) == 1:
        y_all = y_all[:, np.newaxis]
    ideal_regr.fit(X_all, y_all)

    dataset_loss_list = []
    singular_modes_data_list = []
    for repeat_idx in range(num_repeats):
        # subset_sizes = np.arange(10, X_train.shape[0], X_train.shape[0] // 20)
        subset_sizes = np.arange(1, 50, 1)
        for subset_size in subset_sizes:
            print(
                f"Dataset: {dataset_name}, repeat_idx: {repeat_idx}, subset_size: {subset_size}"
            )

            # Split the data into training/testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X_all,
                y_all,
                random_state=repeat_idx,
                test_size=X_all.shape[0] - subset_size,
                shuffle=True,
            )

            regr.fit(X_train, y_train)
            min_singular_value = np.min(
                np.linalg.svd(X_train, full_matrices=False, compute_uv=False)
            )
            y_train_pred = regr.predict(X_train)
            y_train_pred_ideal = ideal_regr.predict(X_train)
            residuals_ideal = y_train - y_train_pred_ideal
            train_mse = mean_squared_error(y_train, y_train_pred)
            y_test_pred = regr.predict(X_test)
            test_mse = mean_squared_error(y_test, y_test_pred)

            # Compute the fraction of the last training datum that lies outside the subspace
            # of the all other training data.
            last_train_datum = X_train[-1, :]
            other_train_data = X_train[:-1, :]
            if other_train_data.shape[0] == 0:
                fraction_outside = 1.0
            else:
                projection_of_last_train_datum_onto_other_train_data = (
                    np.linalg.pinv(other_train_data).T
                    @ last_train_datum
                    @ other_train_data
                )
                fraction_outside = np.linalg.norm(
                    last_train_datum
                    - projection_of_last_train_datum_onto_other_train_data
                ) / np.linalg.norm(last_train_datum)

            dataset_loss_results = {
                "Dataset": dataset_name,
                "Subset Size": subset_size,
                "Train MSE": train_mse,
                "Test MSE": test_mse,
                "Repeat Index": repeat_idx,
                "Smallest Non-Zero Singular Value": min_singular_value,
                "Fraction Outside": fraction_outside,
            }
            dataset_loss_list.append(dataset_loss_results)

            U, S, Vh = np.linalg.svd(X_train, full_matrices=False)
            # Shape: (num test data, num singular modes)
            term_two = np.matmul(X_test, Vh.T)
            term_two_average_per_mode = np.abs(np.mean(term_two, axis=0))
            # Shape: (num singular modes, num train data)
            term_three = np.matmul(U.T, residuals_ideal)
            term_three_average_per_mode = np.abs(np.mean(term_three, axis=1))

            for mode_idx in range(term_two_average_per_mode.shape[0]):
                singular_modes_data = {
                    "Dataset": dataset_name,
                    "Subset Size": subset_size,
                    "Repeat Index": repeat_idx,
                    "Singular Index": mode_idx + 1,
                    "Singular Index From Smallest": len(S) - mode_idx + 1,
                    "Term Two": term_two_average_per_mode[mode_idx],
                    "Term Three": term_three_average_per_mode[mode_idx],
                }
                singular_modes_data_list.append(singular_modes_data)

    dataset_loss_df = pd.DataFrame(dataset_loss_list)
    singular_modes_df = pd.DataFrame(singular_modes_data_list)

    plt.close()
    plt.figure(figsize=(6, 5))
    sns.lineplot(
        data=dataset_loss_df,
        x="Subset Size",
        y="Train MSE",
        label="Train",
    )
    sns.lineplot(
        data=dataset_loss_df,
        x="Subset Size",
        y="Test MSE",
        label="Test",
    )
    plt.xlabel("Num. Training Samples")
    plt.ylabel("Mean Squared Error")
    plt.axvline(
        x=X_all.shape[1], color="black", linestyle="--", label="Interpolation Threshold"
    )
    title = (
        f"Ordinary Linear Regression on\n{dataset_name} (Num Repeats: {num_repeats})"
    )
    plt.title(title)
    plt.yscale("log")
    ymax = 2 * max(
        dataset_loss_df.groupby("Subset Size")["Test MSE"].mean().max(),
        dataset_loss_df.groupby("Subset Size")["Train MSE"].mean().max(),
    )
    ymin = (
        0.5
        * dataset_loss_df.groupby("Subset Size")["Train MSE"].mean()[X_all.shape[1] + 1]
    )
    plt.ylim(bottom=ymin, top=ymax)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(dataset_results_dir, f"double_descent_dataset"),
        bbox_inches="tight",
        dpi=300,
    )
    # plt.show()

    plt.close()
    plt.figure(figsize=(6, 5))
    sns.lineplot(
        data=dataset_loss_df,
        x="Subset Size",
        y="Smallest Non-Zero Singular Value",
        color="green",
    )
    plt.xlabel("Num. Training Samples")
    plt.ylabel("Smallest Non-Zero Singular\nValue of Training Features " + r"$X$")
    plt.axvline(
        x=X_all.shape[1], color="black", linestyle="--", label="Interpolation Threshold"
    )
    plt.title(title)
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            dataset_results_dir,
            f"least_informative_singular_value_dataset",
        ),
        bbox_inches="tight",
        dpi=300,
    )
    # plt.show()

    plt.close()
    # Set figure size
    plt.figure(figsize=(6, 5))
    sns.lineplot(
        data=dataset_loss_df,
        x="Subset Size",
        y="Fraction Outside",
        label="Train",
    )
    plt.ylim(bottom=1e-5)
    plt.xlabel("Num. Training Samples")
    plt.ylabel(
        "Fraction of Newest Training Datum\nOutside Subspace of Other Training Data"
    )
    plt.axvline(
        x=X_all.shape[1], color="black", linestyle="--", label="Interpolation Threshold"
    )
    plt.title(title)
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            dataset_results_dir,
            f"fraction_outside_training_features_subspace",
        ),
        bbox_inches="tight",
        dpi=300,
    )
    # plt.show()

    plt.close()
    g = sns.lineplot(
        data=singular_modes_df,
        x="Subset Size",
        y="Term Two",
        hue="Singular Index",
    )
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    plt.xlabel("Num. Training Samples")
    plt.ylabel(r"$|\; \vec{x}_{test} \cdot \vec{v}_r \; |$")
    plt.axvline(
        x=X_all.shape[1], color="black", linestyle="--", label="Interpolation Threshold"
    )
    plt.yscale("log")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            dataset_results_dir,
            f"term_two_singular_mode_contributions_indexed_from_leading",
        ),
        bbox_inches="tight",
        dpi=300,
    )
    # plt.show()

    plt.close()
    g = sns.lineplot(
        data=singular_modes_df,
        x="Subset Size",
        y="Term Two",
        hue="Singular Index From Smallest",
    )
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    plt.xlabel("Num. Training Samples")
    plt.ylabel(r"$|\; \vec{x}_{test} \cdot \vec{v}_r \; |$")
    plt.axvline(
        x=X_all.shape[1], color="black", linestyle="--", label="Interpolation Threshold"
    )
    plt.yscale("log")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            dataset_results_dir,
            f"term_two_singular_mode_contributions_indexed_from_smallest",
        ),
        bbox_inches="tight",
        dpi=300,
    )
    # plt.show()

    plt.close()
    g = sns.lineplot(
        data=singular_modes_df,
        x="Subset Size",
        y="Term Three",
        hue="Singular Index",
    )
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    plt.xlabel("Num. Training Samples")
    plt.ylabel(r"$|\; \vec{u}_R \cdot E \; |$")
    plt.axvline(
        x=X_all.shape[1], color="black", linestyle="--", label="Interpolation Threshold"
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            dataset_results_dir,
            f"term_three_singular_mode_contributions_indexed_from_leading",
        ),
        bbox_inches="tight",
        dpi=300,
    )
    # plt.show()

    plt.close()
    g = sns.lineplot(
        data=singular_modes_df,
        x="Subset Size",
        y="Term Three",
        hue="Singular Index From Smallest",
    )
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    plt.xlabel("Num. Training Samples")
    plt.ylabel(r"$|\; \vec{u}_R \cdot E \; |$")
    plt.axvline(
        x=X_all.shape[1], color="black", linestyle="--", label="Interpolation Threshold"
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            dataset_results_dir,
            f"term_three_singular_mode_contributions_indexed_from_smallest",
        ),
        bbox_inches="tight",
        dpi=300,
    )
    # plt.show()
