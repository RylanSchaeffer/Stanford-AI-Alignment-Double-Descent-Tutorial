from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


num_data_list = [10, 100, 1000]
num_features_list = [10, 100, 1000]
num_repeat_list = np.arange(100).tolist()

df_results = []
for num_data, num_features, repeat_idx in product(num_data_list, num_features_list, num_repeat_list):
    print('num_data:', num_data, 'num_features:', num_features, 'repeat_idx:', repeat_idx)
    X = np.random.normal(size=(num_data, num_features))
    eigenvalues = np.square(np.linalg.svd(X, full_matrices=False, compute_uv=False))
    # Manually add the missing zeros:
    eigenvalues = np.concatenate([eigenvalues, np.zeros(max(num_data, num_features) - eigenvalues.shape[0])])
    df_results.append(pd.DataFrame({
        'num_data': np.full_like(eigenvalues, fill_value=num_data),
        'num_features': np.full_like(eigenvalues, fill_value=num_features),
        'repeat_idx': np.full_like(eigenvalues, fill_value=repeat_idx),
        'Eigenvalue': eigenvalues,
    }))

df = pd.concat(df_results).reset_index(drop=True)
df['num_features'] = df['num_features'].astype(int)
df['num_data'] = df['num_data'].astype(int)

plt.close()
g = sns.FacetGrid(
    data=df,
    col='num_data',
    row='num_features',
    sharey=False,
    sharex=False,
    margin_titles=True
)
g.map_dataframe(sns.histplot,
                x='Eigenvalue',
                stat='probability',
                bins=100,
                line_kws={'linewidth': 0})
g.set_titles(col_template="Num Data: {col_name}",
             row_template="Num Features: {row_name}")
g.set(yscale = 'log')
# axes = g.axes
# for row_idx in range(axes.shape[0]):
#     for col_idx in range(axes.shape[1]):
#         axes[row_idx, col_idx].axvline(0, ls='--', c='k',
#                                        label='Chance')
plt.savefig(f'random_data_eigenvalue_distribution.png',
            bbox_inches='tight',
            dpi=300)
plt.show()
plt.close()


