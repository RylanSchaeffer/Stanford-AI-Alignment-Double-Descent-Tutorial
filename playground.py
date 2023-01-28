import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Set plot defaults.
sns.set(style='ticks', font_scale=1.2)
plt.rcParams["font.family"] = ["Times New Roman"]
plt.rcParams["font.size"] = 25  # was previously 22
sns.set_style("whitegrid")

# Set seed for reproducibility.
np.random.seed(0)

# Create sklearn linear regression object
regr = linear_model.LinearRegression(fit_intercept=False)


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


num_repeats = 100
for dataset_name, dataset_fn in regression_datasets:

    print('On dataset:', dataset_name)

    # Load the diabetes dataset
    X, y = dataset_fn(return_X_y=True)

    dataset_test_loss_df = []
    for repeat_idx in range(num_repeats):

        # Split the data into training/testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            random_state=repeat_idx,
            test_size=.2,
            shuffle=True)

        # subset_sizes = np.arange(10, X_train.shape[0], X_train.shape[0] // 20)
        subset_sizes = np.arange(1, 50, 1)
        for subset_size in subset_sizes:

            # First run the random data subset
            random_subset_indices = np.random.choice(
                np.arange(0, X_train.shape[0]),
                size=subset_size,
                replace=False,
            )
            regr.fit(X_train[random_subset_indices], y_train[random_subset_indices])
            random_y_pred = regr.predict(X_test)
            random_test_mse = mean_squared_error(y_test, random_y_pred)
            dataset_test_loss_df.append({
                'Dataset': dataset_name,
                'Subset Size': subset_size,
                'Test MSE': random_test_mse,
                'Repeat Index': repeat_idx})

    dataset_test_loss_df = pd.DataFrame(dataset_test_loss_df)

    plt.close()
    sns.lineplot(
        data=dataset_test_loss_df,
        x='Subset Size',
        y='Test MSE',
    )
    plt.xlabel('Num. Training Samples')
    plt.axvline(x=X.shape[1], color='black', linestyle='--', label='Interpolation Threshold')
    plt.title(f'{dataset_name} (Num Repeats: {num_repeats})')
    plt.yscale('log')
    plt.legend()
    # plt.show()
    plt.savefig(f'double_descent_dataset={dataset_name}',
                bbox_inches='tight',
                dpi=300)
    plt.close()

