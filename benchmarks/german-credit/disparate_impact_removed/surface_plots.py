from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# df = pd.read_csv("disparate_removed_german_discrimination.csv")
df = pd.read_csv("reweighted_german_discrimination.csv")

def from_setting_find_model_count(perm, h1units, h2units, batch):
    hyperparameter_setting = 0
    for h1 in [16, 24, 32]:
        for h2 in [8, 12]:
            for bat in [50, 100]:
                if h1units == h1 and h2units == h2 and batch == bat:
                    return perm, hyperparameter_setting
                else:
                    hyperparameter_setting += 1


def variation(setting_now):
    model_count = 0
    for perm in range(20):
        for h1units in [16, 24, 32]:
            for h2units in [8, 12]:
                for batch in [50, 100]:      # different batch sizes for this dataset
                    if model_count < setting_now:
                        model_count += 1
                        continue
                    # print(setting_now, "done", perm, h1units, h2units, batch)
                    return perm, h1units, h2units, batch, model_count


discrimination = {}
for i in range(df.shape[0]):
    val = df.loc[i, :].tolist()
    assert(len(val) == 7)
    per, h1, h2, b = int(val[3]), int(val[0]), int(val[1]), int(val[2])
    permutation, hyperparameter_setting = from_setting_find_model_count(per, h1, h2, b)
    assert(variation(permutation*12 + hyperparameter_setting)[:4] == (per, h1, h2, b))
    # settings.append(permutation)
    assert not ((permutation, hyperparameter_setting) in discrimination)    # only one value per 240 such settings is possible
    discrimination[(permutation, hyperparameter_setting)] = int(val[-1])/1000.0     # discrimination percentage

def z_function(x, y):
    # z = discrimination[(x, y)]
    k = np.array([discrimination[i] for i in discrimination.keys()]).reshape(20, 12)
    return k



x = np.linspace(1, 12, 12)      # hyperparameter_settings
y = np.linspace(1, 20, 20)      # permutations
# import ipdb; ipdb.set_trace()
X, Y = np.meshgrid(x, y)
Z = z_function(X, Y)
fig = plt.figure()
ax = plt.axes(projection="3d")
# ax.plot_wireframe(X, Y, Z, color='green')
ax.set_xlabel('Hyper parameter settings')
ax.set_ylabel('Data permutations')
ax.set_zlabel('Remaining Discrimination')

# plt.show()
# fig = plt.figure()
# ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='winter', edgecolor='none')
# ax.set_title('Surface plot for disparate impact removed')
ax.set_title('Surface plot for Reweighting')
# plt.show()
# plt.savefig("disp_removed.png")
plt.savefig("reweighted.png")