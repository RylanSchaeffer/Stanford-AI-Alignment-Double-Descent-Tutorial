import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from arrow3D import Arrow3D


# Set style
sns.set_style("whitegrid")

# Set seed for reproducibility.
np.random.seed(0)

results_dir = 'results/smallest_nonzero_singular_value'
os.makedirs(results_dir, exist_ok=True)


num_data = 1500
dim = 3
mean = np.zeros(dim)
cov = np.array([[23., 9., 4.], [9., 6., 2.0], [4., 2.0, 5.]])
assert np.all(np.linalg.eigvals(cov) > 0.)
# Shape: (1000, 3)
X = np.random.multivariate_normal(
    mean=mean,
    cov=cov,
    size=num_data,
)

eigindex_color_map = {
    0: 'r',
    1: 'g',
    2: 'b',
}

# Plot X in 3D.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=1, color='k')
# Add eigenvectors
eigvals, eigvecs = np.linalg.eigh(cov)
eigvecs = eigvecs[:, np.argsort(eigvals)]
eigvals = np.sort(eigvals)
print('True Covariance Eigenvalues: ', eigvals)
print('Empirical Covariance Eigenvalues: ', np.sort(np.linalg.eigvals(np.cov(X.T))))
for eigidx, (eigval, eigvec) in enumerate(zip(eigvals, eigvecs.T)):
    scaled_eigvec = eigval * eigvec
    prefactors = [-1., 1.]
    for prefactor in prefactors:
        drawvec = Arrow3D([0, prefactor * scaled_eigvec[0]],
                          [0, prefactor * scaled_eigvec[1]],
                          [0, prefactor * scaled_eigvec[2]],
                          mutation_scale=20, lw=2, arrowstyle="-|>", color=eigindex_color_map[2 - eigidx])
        # adding the arrow to the plot
        ax.add_artist(drawvec)

# Set axes limits.
max_val = 7.
ax.set_xlim3d([-max_val, max_val])
ax.set_ylim3d([-max_val, max_val])
ax.set_zlim3d([-max_val, max_val])
ax.set_xlabel("Dim 1")
ax.set_ylabel("Dim 2")
ax.set_zlabel("Dim 3")
# Reverse y axis so all positive directions face the "camera".
ax.invert_yaxis()

ax.set_title('True Data Distribution')

plt.savefig(os.path.join(results_dir,
                         f'data_distribution'),
            bbox_inches='tight',
            dpi=300)
plt.show()





num_data_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 25, 100]
for num_data in num_data_list:
    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X_subset = X[:num_data, :]

    ax.scatter(X_subset[:num_data, 0],
               X_subset[:num_data, 1],
               X_subset[:num_data, 2],
               s=50,
               color='k')

    # Add eigenvectors
    U, S, Vt = np.linalg.svd(X_subset / np.sqrt(num_data), full_matrices=False)
    eigvals, eigvecs = np.square(S), Vt
    eigvecs = eigvecs[np.argsort(eigvals)[::-1]]
    eigvals = np.sort(eigvals)[::-1]
    print(eigvals)
    for eigidx, (eigval, eigvec) in enumerate(zip(eigvals, eigvecs)):
        scaled_eigvec = eigval * eigvec
        prefactors = [-1., 1.]
        for prefactor in prefactors:
            drawvec = Arrow3D([0, prefactor * scaled_eigvec[0]],
                              [0, prefactor * scaled_eigvec[1]],
                              [0, prefactor * scaled_eigvec[2]],
                              mutation_scale=20, lw=2, arrowstyle="-|>", color=eigindex_color_map[2-eigidx])
            # adding the arrow to the plot
            ax.add_artist(drawvec)

    # Set axes limits.
    ax.set_xlim3d([-max_val, max_val])
    ax.set_ylim3d([-max_val, max_val])
    ax.set_zlim3d([-max_val, max_val])
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_zlabel("Dim 3")
    # Reverse y axis so all positive directions face the "camera".
    ax.invert_yaxis()

    ax.set_title('Num Data: {}'.format(num_data))

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir,
                             f'data_distribution_num_data={num_data}'),
                bbox_inches='tight',
                dpi=300)
    plt.show()
