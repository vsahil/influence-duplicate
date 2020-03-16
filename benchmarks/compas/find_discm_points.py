import numpy as np
import copy
import pandas as pd


def rescale_input_numpy(inp):
    assert(inp.shape[1] == 19)  # 19 features for compas dataset
    mins_and_ranges = [(0, 1), (18, 65), (0, 1), (0, 10), (1, 9), (0, 13), (0, 17), (0, 38), (-414.0, 1471.0), (0.0, 9485.0), (-1, 2), (0, 1), (0, 1), (1, 9), (1, 9), (0, 38), (0, 937), (0, 1186), (0, 1)]
    r = np.arange(19)
    out = copy.deepcopy(inp)
    for col, (min_, range_) in zip(r, mins_and_ranges):
        out[:, col] = np.divide(np.subtract(out[:, col], min_), range_)
    return out


def rescale_input_numpy_disparateremoved_compas(inp):
    assert False
    assert(inp.shape[1] == 20)  # 20 features for german credit dataset
    # means_and_ranges = [(1.001, 3), (20.903, 68), (32.545, 4), (47.148, 370), (3271.258, 18174), (1.19, 4), (3.384, 4), (2.973, 3), (0.69, 1), (101.145, 2), (2.845, 3), (122.358, 3), (35.546, 56), (142.675, 2), (151.929, 2), (1.407, 3), (172.904, 3), (1.155, 1), (1.404, 1), (1.037, 1)]
    means_and_ranges  = [(1.001, 3), (19.172, 56), (32.545, 4), (47.148, 370), (2866.523, 15607), (1.19, 4), (3.384, 4), (2.973, 3), (0.69, 1), (101.145, 2), (2.845, 3), (122.358, 3), (34.434, 56), (142.675, 2), (151.929, 2), (1.407, 3), (172.904, 3), (1.155, 1), (1.404, 1), (1.037, 1)]
    r = np.arange(20)
    out = copy.deepcopy(inp)
    for col, (mean, range_) in zip(r, means_and_ranges):
        out[:, col] = np.divide(np.subtract(out[:, col], mean), range_)
    return out


def entire_test_suite(mini=False, disparateremoved=False):
    gender0 = "race0_compas_two_year"
    gender1 = "race1_compas_two_year"
    # if mini:
        # gender0 += "_mini"
        # gender1 += "_mini"
    
    df0 = pd.read_csv(f"../../compas-dataset/{gender0}.csv")
    df1 = pd.read_csv(f"../../compas-dataset/{gender1}.csv")
    # if mini: 
        # assert(df0.shape == df1.shape == (1000, 12))
    # else:
    assert(df0.shape == df1.shape == (1000000, 19))

    assert(not df0.equals(df1))
    assert(df0.drop('race', axis=1).equals(df1.drop('race', axis=1)))     # after dropping sex they should be equal dataframe

    class0_ = df0.to_numpy(dtype=np.float64)
    class1_ = df1.to_numpy(dtype=np.float64)

    if disparateremoved:
        class0 = rescale_input_numpy_disparateremoved_compas(class0_)
        class1 = rescale_input_numpy_disparateremoved_compas(class1_)
    else:
        class0 = rescale_input_numpy(class0_)
        class1 = rescale_input_numpy(class1_)

    return class0, class1