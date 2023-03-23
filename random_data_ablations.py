# Made by Akhilan.
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Set style
sns.set_style("whitegrid")

# Set seed for reproducibility.
np.random.seed(0)

P = 30
D = 20


def generate_data(N, P, D):
    X_bar = np.random.randn(N, P)
    X = X_bar[:, :D]
    X_tilde = X_bar[:, D:]
    beta_bar = np.random.randn(P, 1)
    Y = X_bar @ beta_bar
    return X, Y, X_tilde, beta_bar


def generate_test_data(n, P, D, beta_bar):
    X_bar_test = np.random.randn(n, P)
    X_test = X_bar_test[:, :D]
    Y_test = X_bar_test @ beta_bar
    return X_test, Y_test


def train_and_evaluate(X, Y, X_test, Y_test, filter_singular_values=False, cutoff=None, project_test_data=False,
                       k=None):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    if filter_singular_values and cutoff is not None:
        S = np.where(S > cutoff, S, 0)
        S = np.where(S == 0, np.inf, S)
    X_pinv = (Vt.T * (1 / S)) @ U.T
    beta = X_pinv @ Y
    Y_train_pred = X @ beta
    train_mse = np.mean((Y - Y_train_pred) ** 2)

    if project_test_data and k is not None:
        X_test = X_test @ Vt.T[:, :k] @ Vt[:k, :]

    Y_test_pred = X_test @ beta
    test_mse = np.mean((Y_test - Y_test_pred) ** 2)
    return train_mse, test_mse


def run_experiment(experiment_num, filter_singular_values=False, cutoffs=None, project_test_data=False, ks=None):
    N_values = np.arange(1, 2 * D)
    n_test = 1000
    n_trials = 100

    results = []
    for N in N_values:
        trial_results = []
        for trial in range(n_trials):
            X, Y, X_tilde, beta_bar = generate_data(N, P, D)
            X_test, Y_test = generate_test_data(n_test, P, D, beta_bar)
            if experiment_num == 1:
                train_mse, test_mse = train_and_evaluate(X, Y, X_test, Y_test)
                trial_results.append((train_mse, test_mse))
            elif experiment_num in [2, 3]:
                trial_results_for_cutoffs_or_ks = []
                for cutoff_or_k in cutoffs if experiment_num == 2 else ks:
                    train_mse, test_mse = train_and_evaluate(X, Y, X_test, Y_test,
                                                             filter_singular_values=filter_singular_values,
                                                             cutoff=cutoff_or_k, project_test_data=project_test_data,
                                                             k=cutoff_or_k)
                    trial_results_for_cutoffs_or_ks.append((train_mse, test_mse))
                trial_results.append(trial_results_for_cutoffs_or_ks)
            elif experiment_num == 4:
                Y = X @ beta_bar[:D]
                Y_test = X_test @ beta_bar[:D]

                train_mse, test_mse = train_and_evaluate(X, Y, X_test, Y_test)
                trial_results.append((train_mse, test_mse))
        results.append(trial_results)
    return results


def shaded_errorbar(x, y, yerr, color, alpha=0.3, **kwargs):
    # Calculate upper and lower error bounds
    upper = y * yerr
    lower = y / yerr

    # Plot the shaded error region
    plt.fill_between(x, upper, lower, color=color, alpha=alpha)

    # Plot the line
    plt.plot(x, y, color=color, **kwargs)


def exp_log_std(data, axis=None):
    log_data = np.log(data)
    log_std = np.std(log_data, axis=axis)
    return np.exp(log_std)


def exp_log_mean(data, axis=None):
    log_data = np.log(data)
    log_mean = np.mean(log_data, axis=axis)
    return np.exp(log_mean)


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


if __name__ == '__main__':
    # Experiment 1
    results_exp1 = run_experiment(1)

    train_mses_exp1 = exp_log_mean([[r[0] for r in trial_results] for trial_results in results_exp1], axis=1)
    train_stddevs_exp1 = exp_log_std([[r[0] for r in trial_results] for trial_results in results_exp1], axis=1)
    test_mses_exp1 = exp_log_mean([[r[1] for r in trial_results] for trial_results in results_exp1], axis=1)
    test_stddevs_exp1 = exp_log_std([[r[1] for r in trial_results] for trial_results in results_exp1], axis=1)
    N_values = np.arange(1, 2 * D)

    plt.figure()
    shaded_errorbar(N_values, train_mses_exp1, yerr=train_stddevs_exp1, label='Train', color='tab:blue')
    shaded_errorbar(N_values, test_mses_exp1, yerr=test_stddevs_exp1, label='Test', color='tab:orange')
    plt.axvline(x=D, linestyle='--', color='black', label='Interpolation Threshold')
    plt.ylim((10 ** -1, 10 ** 3))
    plt.yscale('log')
    plt.xlabel('Num. Training Samples')
    plt.ylabel('Mean Squared Error')
    plt.legend(fontsize=10)
    plt.title('Unablated', fontsize=12)
    plt.tight_layout()
    plt.savefig('figures/student_teacher_exp1.png')

    # Experiment 2
    cutoffs = [1e-3, 1e-2, 1e-1, 1e-0]
    results_exp2 = run_experiment(2, filter_singular_values=True, cutoffs=cutoffs)

    plt.figure()
    for i, cutoff in enumerate(cutoffs):
        results_exp2_cutoff = np.asarray(results_exp2)[:, :, i]
        train_mses_exp2 = exp_log_mean([[r[0] for r in trial_results] for trial_results in results_exp2_cutoff],
                                       axis=1)
        train_stddevs_exp2 = exp_log_std([[r[0] for r in trial_results] for trial_results in results_exp2_cutoff],
                                         axis=1)
        test_mses_exp2 = exp_log_mean([[r[1] for r in trial_results] for trial_results in results_exp2_cutoff],
                                      axis=1)
        test_stddevs_exp2 = exp_log_std([[r[1] for r in trial_results] for trial_results in results_exp2_cutoff],
                                        axis=1)
        N_values = np.arange(1, 2 * D)

        # Use different color shadings for different cutoffs
        shaded_errorbar(N_values, train_mses_exp2, yerr=train_stddevs_exp2,
                        color=lighten_color('tab:blue', 0.4 * i + 0.2))
        shaded_errorbar(N_values, test_mses_exp2, yerr=test_stddevs_exp2,
                        color=lighten_color('tab:orange', 0.4 * i + 0.2),
                        label='Cutoff: {}'.format(cutoff))

    plt.axvline(x=D, linestyle='--', color='black', label='Interpolation Threshold')
    plt.ylim((10 ** -1, 10 ** 3))
    plt.yscale('log')
    plt.xlabel('Num. Training Samples')
    plt.ylabel('Mean Squared Error')
    plt.legend(fontsize=10)
    plt.title('No Small Singular Values', fontsize=12)
    plt.tight_layout()
    plt.savefig('figures/student_teacher_exp2.png')

    # Experiment 3
    ks = [14, 16, 18, 20]
    results_exp3 = run_experiment(3,  ks=ks, project_test_data=True)

    plt.figure()
    for i, k in enumerate(ks):
        results_exp3_k = np.asarray(results_exp3)[:, :, i]
        train_mses_exp3 = exp_log_mean([[r[0] for r in trial_results] for trial_results in results_exp3_k],
                                       axis=1)
        train_stddevs_exp3 = exp_log_std([[r[0] for r in trial_results] for trial_results in results_exp3_k],
                                         axis=1)
        test_mses_exp3 = exp_log_mean([[r[1] for r in trial_results] for trial_results in results_exp3_k],
                                      axis=1)
        test_stddevs_exp3 = exp_log_std([[r[1] for r in trial_results] for trial_results in results_exp3_k],
                                        axis=1)
        N_values = np.arange(1, 2 * D)

        # Use different color shadings for different cutoffs
        shaded_errorbar(N_values, train_mses_exp3, yerr=train_stddevs_exp3,
                        color=lighten_color('tab:blue', 0.4 * i + 0.2))
        shaded_errorbar(N_values, test_mses_exp3, yerr=test_stddevs_exp3,
                        color=lighten_color('tab:orange', 0.4 * i + 0.2),
                        label='k: {}'.format(k))
    plt.axvline(x=D, linestyle='--', color='black', label='Interpolation Threshold')
    plt.ylim((10 ** -1, 10 ** 3))
    plt.yscale('log')
    plt.xlabel('Num. Training Samples')
    plt.ylabel('Mean Squared Error')
    plt.legend(fontsize=10)
    plt.title('Test Features in Training Feature Subspace', fontsize=12)
    plt.tight_layout()
    plt.savefig('figures/student_teacher_exp3.png')

    # Experiment 4
    results_exp4 = run_experiment(4)

    train_mses_exp1 = exp_log_mean([[r[0] for r in trial_results] for trial_results in results_exp4], axis=1)
    train_stddevs_exp1 = exp_log_std([[r[0] for r in trial_results] for trial_results in results_exp4], axis=1)
    test_mses_exp1 = exp_log_mean([[r[1] for r in trial_results] for trial_results in results_exp4], axis=1)
    test_stddevs_exp1 = exp_log_std([[r[1] for r in trial_results] for trial_results in results_exp4], axis=1)
    N_values = np.arange(1, 2 * D)

    plt.figure()
    shaded_errorbar(N_values, train_mses_exp1, yerr=train_stddevs_exp1, label='Train', color='tab:blue')
    shaded_errorbar(N_values, test_mses_exp1, yerr=test_stddevs_exp1, label='Test', color='tab:orange')
    plt.axvline(x=D, linestyle='--', color='black')
    plt.ylim((10 ** -1, 10 ** 3))
    plt.yscale('log')
    plt.xlabel('Num. Training Samples')
    plt.ylabel('Mean Squared Error')
    plt.legend(fontsize=10)
    plt.title('No Residuals in Ideal Fit', fontsize=12)
    plt.tight_layout()
    plt.savefig('figures/student_teacher_exp4.png')