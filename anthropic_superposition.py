import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import seaborn as sns


data_dim = 10000
num_training_data = np.array([200, 500, 1000, 2500, 5000, 7500, 10000, 15000, 20000, 25000, 50000])
# num_training_data = np.array([200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000])
# num_train_data = np.logspace(1, 6, 50, base=10, dtype=int)
mask = np.random.rand(data_dim, np.max(num_training_data)).astype(np.float16) < 0.999
training_data = np.multiply(
    mask,
    np.random.rand(data_dim, np.max(num_training_data)).astype(np.float16)).astype(np.float16)
# Rescale training data to have unit norm
training_data = training_data / np.linalg.norm(training_data, axis=0, keepdims=True)

ranks = []
for N in num_training_data:
    ranks.append(np.linalg.matrix_rank(training_data[:, :N]))

plt.close()
plt.plot(num_training_data, ranks)
plt.xscale('log')
plt.show()


num_hidden_dimensions = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 16])
num_parameters = (num_hidden_dimensions + 1) * 1e3
# num_parameters = np.power(num_hidden_dimensions + 1, 2) * 1e3
num_parameters_tiled = np.tile(num_parameters.reshape(-1, 1), (1, len(num_training_data)))
num_train_data_tiled = np.tile(num_training_data.reshape(1, -1), (len(num_hidden_dimensions), 1))
assert num_parameters_tiled.shape == num_train_data_tiled.shape

plt.close()
sns.heatmap(
    1 + np.abs(num_parameters_tiled - num_train_data_tiled),
    norm=matplotlib.colors.LogNorm(),
    yticklabels=num_hidden_dimensions,
    # yticklabels=np.round(num_parameters),
    xticklabels=num_training_data,
    cbar_kws={'label': 'Squared Distance to Interpolation Threshold'},
    cmap='magma'
)
# Invert y axis
plt.gca().invert_yaxis()
plt.xlabel('Training Dataset Size')
plt.ylabel('Num. Hidden Units')
plt.show()


print()