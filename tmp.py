import matplotlib.pyplot as plt
import numpy as np

from arrow3D import Arrow3D


np.random.seed(0)

arrow_prop_dict = dict(mutation_scale=20, arrowstyle='-|>', shrinkA=0, shrinkB=0)

beta_true = np.array([[1., -1]]).T



# Plot the underparameterized setting.
X_under = np.array([[2., 1.]])
y_under_mean = X_under @ beta_true
plt.close()
ax = plt.figure().add_subplot(projection='3d')
a = Arrow3D([0, beta_true[0, 0]],
            [0, beta_true[1, 0]],
            zs=[0, 0],
            # label=r'$\hat{\beta}$',
            alpha=0.9,
            color='r',
            **arrow_prop_dict)
ax.add_artist(a)
for _ in range(30):
    y_under = y_under_mean + np.random.normal(0, scale=0.1)

    a = Arrow3D([0, X_under[0, 0]],
                [0, X_under[0, 1]],
                zs=[0, y_under[0, 0]],
                # label='Data',
                color='k',
                alpha=0.10,
                **arrow_prop_dict)
    ax.add_artist(a)

    # Compute ideal beta
    beta_fit = X_under.T @ np.linalg.pinv(X_under @ X_under.T) @  y_under
    a = Arrow3D([0, beta_fit[0, 0]],
                [0, beta_fit[1, 0]],
                zs=[0, 0],
                # label=r'$\hat{\beta}$',
                alpha=0.1,
                color='b',
                **arrow_prop_dict)
    ax.add_artist(a)

ax.legend()
ax.set_xlim(-1, 2.5)
ax.set_ylim(-1, 2.5)
ax.set_zlim(0, 2)
ax.set_xlabel('X Dim 1')
ax.set_ylabel('X Dim 2')
ax.set_zlabel('Y')
plt.show()


# Plot the interpolation setting.
X_interp = np.array([[2., 1.], [2., 1.5]])
y_interp_mean = X_interp @ beta_true
plt.close()
plt.plot([0, X_interp[0, 0], [0, X_interp[1, 0]]],
         [0, X_interp[0, 1], [0, X_interp[1, 1]]], label='data', color='k')
plt.legend()
# for _ in range(10):
#     y_under = y_under_mean + np.random.normal(0, scale=0.1)
#     # Compute ideal beta
#     beta_fit = X_under.T @ np.linalg.pinv(X_under @ X_under.T) @  y_interp_mean
#     plt.plot([0, beta_fit[0, 0]], [0, beta_fit[1, 0]])

plt.xlim(-1, 3)
plt.ylim(-1, 3)
plt.show()


X_over = np.array([[2., 1.], [2., 1.], [2., 1.]])

label_noise_mag = 0.05

