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
cov2 = np.cov(X, rowvar=False)

# Plot X in 3D.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=1)
# Add eigenvectors
eigvals, eigvecs = np.linalg.eigh(cov)
print('Eigenvalues: ', eigvals)
for eigval, eigvec in zip(eigvals, eigvecs.T):
    scaled_eigvec = eigval * eigvec
    drawvec = Arrow3D([0, scaled_eigvec[0]], [0, scaled_eigvec[1]], [0, scaled_eigvec[2]],
                      mutation_scale=20, lw=2, arrowstyle="-|>", color='r')
    # adding the arrow to the plot
    ax.add_artist(drawvec)
# Set axes limits.
max_val = 7
ax.set_xlim3d([-max_val, max_val])
ax.set_ylim3d([-max_val, max_val])
ax.set_zlim3d([-max_val, max_val])
ax.set_xlabel("Dim 1")
ax.set_ylabel("Dim 2")
ax.set_zlabel("Dim 3")
# Reverse y axis so all positive directions face the "camera".
ax.invert_yaxis()

plt.savefig(os.path.join(results_dir,
                         f'data_distribution'),
            bbox_inches='tight',
            dpi=300)
plt.show()





num_data_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
for num_data in num_data_list:
    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X_subset = X[:num_data, :]
    # Add eigenvectors
    U, S, Vt = np.linalg.svd(X_subset, full_matrices=False)
    print(S)
    for eigval, eigvec in zip(np.square(S), Vt):
        scaled_eigvec = eigval * eigvec
        drawvec = Arrow3D([0, scaled_eigvec[0]], [0, scaled_eigvec[1]], [0, scaled_eigvec[2]],
                          mutation_scale=20, lw=2, arrowstyle="-|>", color='r')
        # adding the arrow to the plot
        ax.add_artist(drawvec)

    ax.scatter(X_subset[:num_data, 0],
               X_subset[:num_data, 1],
               X_subset[:num_data, 2],
               s=50)

    # Set axes limits.
    max_val = 7
    ax.set_xlim3d([-max_val, max_val])
    ax.set_ylim3d([-max_val, max_val])
    ax.set_zlim3d([-max_val, max_val])
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_zlabel("Dim 3")
    # Reverse y axis so all positive directions face the "camera".
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir,
                             f'data_distribution_num_data={num_data}'),
                bbox_inches='tight',
                dpi=300)
    plt.show()
