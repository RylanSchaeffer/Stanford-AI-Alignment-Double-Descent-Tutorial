import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# Set style
sns.set_style("whitegrid")

# Set seed for reproducibility.
np.random.seed(0)


def load_who_life_expectancy(**kwargs):
    # https://www.kaggle.com/kumarajarshi/life-expectancy-who

    life_expectancy_df = pd.read_csv('Life Expectancy Data.csv')
    life_expectancy_df.dropna(inplace=True)

    X = life_expectancy_df[[
        'Adult Mortality',
        'infant deaths',
        'Alcohol',
        'percentage expenditure',
        'Hepatitis B',
        'Measles ',
        ' BMI ',
        'under-five deaths ',
        'Polio',
        'Total expenditure',
        'Diphtheria ',
        ' HIV/AIDS',
        'GDP',
        'Population',
        ' thinness  1-19 years',
        ' thinness 5-9 years',
        'Schooling']].values
    y = life_expectancy_df['Life expectancy '].values

    return X, y


regression_datasets = [
    ('California Housing', datasets.fetch_california_housing),
    ('Diabetes', datasets.load_diabetes,),
    ('WHO Life Expectancy', load_who_life_expectancy)
]


results_dir = 'results/real_data_ablations'
os.makedirs(results_dir, exist_ok=True)

ablation_type_strs = [
    'Unablated',
    'No Small Singular Values',
    'No Residuals in Ideal Fit',
    # 'Test Datum Features in Training Feature Subspace',
]

singular_value_cutoffs = np.logspace(-3, 0, 7)

num_repeats = 30
for dataset_name, dataset_fn in regression_datasets:
    print('On dataset:', dataset_name)
    X, y = dataset_fn(return_X_y=True)

    # One ablation will be to make the true underlying relationship linear and noiseless.
    # To do this, we need to know the ideal linear relationship. Unfortunately, we don't have
    # any way to know this in practice, so we'll use all the data as our best guess.
    beta_ideal = np.linalg.inv(X.T @ X) @ X.T @ y

    dataset_loss_unablated_df = []
    dataset_loss_no_small_singular_values_df = []
    dataset_loss_no_residuals_in_ideal_fit_df = []
    dataset_loss_test_datum_features_in_training_feature_subspace_df = []
    for repeat_idx in range(num_repeats):

        # subset_sizes = np.arange(10, X_train.shape[0], X_train.shape[0] // 20)
        subset_sizes = np.arange(1, 50, 1)
        for subset_size in subset_sizes:

            print(f'Dataset: {dataset_name}, repeat_idx: {repeat_idx}, subset_size: {subset_size}')

            # Split the data into training/testing sets
            X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
                X,
                y,
                np.arange(X.shape[0]),
                random_state=repeat_idx,
                test_size=X.shape[0] - subset_size,
                shuffle=True)

            # BEGIN: Unablated linear fit.
            U, S, Vt = np.linalg.svd(X_train, full_matrices=False, compute_uv=True)
            inverted_S = 1. / S
            inverted_S[inverted_S == np.inf] = 0.
            beta_hat = Vt.T @ np.diag(inverted_S) @ U.T @ y_train
            y_train_pred = X_train @ beta_hat
            train_mse_unablated = mean_squared_error(y_train, y_train_pred)
            y_test_pred = X_test @ beta_hat
            test_mse_unablated = mean_squared_error(y_test, y_test_pred)
            # END: Unablated linear fit.

            dataset_loss_unablated_df.append({
                'Dataset': dataset_name,
                'Subset Size': subset_size,
                'Train MSE': train_mse_unablated,
                'Test MSE': test_mse_unablated,
                'Repeat Index': repeat_idx,
            })

            # BEGIN: No small singular values.
            for cutoff in singular_value_cutoffs:
                S_cutoff = np.copy(S)
                S_cutoff[S_cutoff < cutoff] = 0.
                inverted_S_cutoff = 1. / S_cutoff
                inverted_S_cutoff[inverted_S_cutoff == np.inf] = 0.
                beta_hat_cutoff = Vt.T @ np.diag(inverted_S_cutoff) @ U.T @ y_train
                y_train_pred_cutoff = X_train @ beta_hat_cutoff
                train_mse_cutoff = mean_squared_error(y_train, y_train_pred_cutoff)
                y_test_pred_cutoff = X_test @ beta_hat_cutoff
                test_mse_cutoff = mean_squared_error(y_test, y_test_pred_cutoff)
                dataset_loss_no_small_singular_values_df.append({
                    'Dataset': dataset_name,
                    'Subset Size': subset_size,
                    'Train MSE': train_mse_cutoff,
                    'Test MSE': test_mse_cutoff,
                    'Repeat Index': repeat_idx,
                    'Singular Value Cutoff': cutoff,
                })
            # END: No small singular values.

            # BEGIN: No residuals in ideal fit.
            # Replace the true targets with the ideal possible predictions.
            y_train_no_residuals = X_train @ beta_ideal
            y_test_no_residuals = X_test @ beta_ideal
            beta_hat_no_residuals = Vt.T @ np.diag(inverted_S) @ U.T @ y_train_no_residuals
            y_train_pred_no_residuals = X_train @ beta_hat_no_residuals
            train_mse_no_residuals = mean_squared_error(y_train_no_residuals, y_train_pred_no_residuals)
            y_test_pred_no_residuals = X_test @ beta_hat_no_residuals
            test_mse_no_residuals = mean_squared_error(y_test_no_residuals, y_test_pred_no_residuals)
            dataset_loss_no_residuals_in_ideal_fit_df.append({
                'Dataset': dataset_name,
                'Subset Size': subset_size,
                'Train MSE': train_mse_no_residuals,
                'Test MSE': test_mse_no_residuals,
                'Repeat Index': repeat_idx,
            })
            # END: No residuals in ideal fit.

            # # BEGIN: Test datum features in training feature subspace.
            # y_train_pred = X_train @ beta_hat
            # train_mse_unablated = mean_squared_error(y_train, y_train_pred)
            # y_test_pred = X_test @ beta_hat
            # test_mse_unablated = mean_squared_error(y_test, y_test_pred)
            # # END: Test datum features in training feature subspace.

            # Compute the fraction of the last training datum that lies outside the subspace
            # of the all other training data.
            last_train_datum = X_train[-1, :]
            other_train_data = X_train[:-1, :]
            if other_train_data.shape[0] == 0:
                fraction_outside = 1.
            else:
                projection_of_last_train_datum_onto_other_train_data = np.linalg.pinv(other_train_data).T @ last_train_datum @ other_train_data
                fraction_outside = np.linalg.norm(last_train_datum - projection_of_last_train_datum_onto_other_train_data) / np.linalg.norm(last_train_datum)



    plt.close()
    fig, axes = plt.subplots(nrows=1,
                             ncols=4,
                             figsize=(20, 5),
                             sharex=True,
                             sharey=True)
    fig.suptitle(f'Dataset: {dataset_name}')
    ax = axes[0]
    dataset_loss_unablated_df = pd.DataFrame(dataset_loss_unablated_df)
    sns.lineplot(
        data=dataset_loss_unablated_df,
        x='Subset Size',
        y=f'Train MSE',
        label='Train',
        ax=ax,
    )
    sns.lineplot(
        data=dataset_loss_unablated_df,
        x='Subset Size',
        y=f'Test MSE',
        label='Test',
        ax=ax,
    )
    ax.set_xlabel('Num. Training Samples')
    ax.set_ylabel('Mean Squared Error')
    ax.axvline(x=X.shape[1], color='black', linestyle='--', label='Interpolation Threshold')
    ax.set_title('Unablated')
    # Set the y limits based on the first plot b/c unablated will have the largest test error.
    ymax = 2 * max(dataset_loss_unablated_df.groupby('Subset Size')[f'Test MSE'].mean().max(),
                   dataset_loss_unablated_df.groupby('Subset Size')[f'Train MSE'].mean().max())
    ymin = 0.5 * dataset_loss_unablated_df.groupby('Subset Size')[f'Train MSE'].mean()[X.shape[1] + 1]
    ax.set_ylim(bottom=ymin, top=ymax)
    ax.set_yscale('log')
    ax.legend()

    ax = axes[1]
    dataset_loss_no_small_singular_values_df = pd.DataFrame(dataset_loss_no_small_singular_values_df)
    sns.lineplot(
        data=dataset_loss_no_small_singular_values_df,
        x='Subset Size',
        y=f'Train MSE',
        hue='Singular Value Cutoff',
        legend=False,
        ax=ax,
        hue_norm=LogNorm(),
        palette='OrRd_r',
    )
    sns.lineplot(
        data=dataset_loss_no_small_singular_values_df,
        x='Subset Size',
        y=f'Test MSE',
        hue='Singular Value Cutoff',
        ax=ax,
        hue_norm=LogNorm(),
        palette='OrRd_r',
    )
    ax.set_xlabel('Num. Training Samples')
    ax.set_title('No Small Singular Values')
    ax.axvline(x=X.shape[1], color='black', linestyle='--')
    ax.set_yscale('log')

    ax = axes[2]
    dataset_loss_no_residuals_in_ideal_fit_df = pd.DataFrame(dataset_loss_no_residuals_in_ideal_fit_df)
    ax.plot([1, dataset_loss_no_residuals_in_ideal_fit_df['Subset Size'].max()],
            [ymin, ymin],
            color='tab:blue',
            linestyle='--',
            label='Train = 0')
    sns.lineplot(
        data=dataset_loss_no_residuals_in_ideal_fit_df,
        x='Subset Size',
        y=f'Test MSE',
        label='Test',
        ax=ax,
    )
    # The test error will be 0 for all subset sizes >= X.shape[1]
    # b/c the linear model exactly fits the linear data.
    ax.plot([X.shape[1], dataset_loss_no_residuals_in_ideal_fit_df['Subset Size'].max()],
            [ymin, ymin],
            color='tab:orange',
            linestyle='--',
            label='Test = 0')
    ax.set_xlabel('Num. Training Samples')
    ax.set_title('No Residuals in Ideal Fit')
    ax.axvline(x=X.shape[1], color='black', linestyle='--')
    ax.set_yscale('log')
    ax.legend()

    plt.savefig(os.path.join(results_dir,
                             f'double_descent_dataset={dataset_name}'),
                bbox_inches='tight',
                dpi=300)
    plt.show()

