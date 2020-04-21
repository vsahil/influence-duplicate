from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# df = pd.read_csv("disparate_removed_german_discrimination.csv")
df = pd.read_csv("all_german_discm_data.csv")

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
last = 0
for model_count in range(240):
    if not model_count in [8, 10, 15, 17, 19, 21, 23, 24, 34, 63, 65, 67, 69, 71, 97, 105, 107, 132, 136, 138, 142, 166, 168, 180, 182, 184, 186, 188, 190, 192, 195, 196, 200, 201, 202, 228, 238]:
        x = df[df['Model-count'] == model_count]
        data_split = x['Data-Split'].unique()
        # print(model_count)
        assert(len(data_split) == 1)
        data_split = data_split[0]
        h1 = x['H1Units'].unique()
        assert(len(h1) == 1)
        h1 = h1[0]
        h2 = x['H2Units'].unique()
        assert(len(h2) == 1)
        h2 = h2[0]
        batch = x['Batch'].unique()
        assert(len(batch) == 1)
        batch = batch[0]
        _, hyperparameter_setting = from_setting_find_model_count(model_count, h1, h2, batch)
        # assert(permutation == model_count)
        # print(model_count, h1, h2, batch)
        # import ipdb; ipdb.set_trace()
        assert(variation(data_split*12 + hyperparameter_setting) == (data_split, h1, h2, batch, model_count))
        assert not ((data_split, hyperparameter_setting) in discrimination)    # only one value per 240 such settings is possible
        last = round(x['Discm-percent'].min(), 3)
        discrimination[(data_split, hyperparameter_setting)] = last     # discrimination percentage
    else:
        # print(hyperparameter_setting, "hello")
        permutation, h1, h2, batch, cnt = variation(model_count)
        assert(model_count == cnt)
        cnt2, hyperparameter_setting = from_setting_find_model_count(model_count, h1, h2, batch)
        assert(model_count == cnt == cnt2)
        # print(permutation, permutation2, h1, h2, batch)
        # assert(variation(permutation*12 + hyperparameter_setting)[:4] == (model_count, h1, h2, batch))
        assert not ((permutation, hyperparameter_setting) in discrimination)    # only one value per 240 such settings is possible
        # discrimination[(permutation, hyperparameter_setting)] = last  # round(x['Discm-percent'].min(), 3)     # discrimination percentage
        discrimination[(permutation, hyperparameter_setting)] = np.nan  # round(x['Discm-percent'].min(), 3)     # discrimination percentage


def z_function(x, y):
    # z = discrimination[(x, y)]
    # import ipdb; ipdb.set_trace()
    k = np.array([discrimination[i] for i in discrimination.keys()]).reshape(20, 12)
    return k


def find_statistics():
    import math
    vals = list(discrimination.values())
    # print(vals)
    clean_vals = [i for i in vals if not math.isnan(i)]
    assert(len(clean_vals) == 240 - 37 == 203)
    print(np.min(clean_vals), np.max(clean_vals), np.mean(clean_vals), np.median(clean_vals))
    exit(0)


# find_statistics()
x = np.linspace(1, 12, 12)      # hyperparameter_settings
y = np.linspace(1, 20, 20)      # data-splits
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
ax.set_title('Surface plot for Our method')
# plt.show()
# exit(0)
plt.savefig("our_technique.png")
# plt.savefig("disp_removed.png")
# plt.savefig("reweighted.png")