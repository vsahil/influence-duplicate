import numpy as np
import copy
import pandas as pd


def rescale_input_numpy(inp):
    assert(inp.shape[1] == 32)  # 32 features for student dataset
    mins_and_ranges = [(0, 1), (0, 1), (15, 7), (0, 1), (0, 1), (0, 1), (0, 4), (0, 4), (0, 4), (0, 4), (0, 3), (0, 2), (1, 3), (1, 3), (0, 3), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4), (0, 32), (0, 19), (0, 19)]
    r = np.arange(32)
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
    gender0 = "sex0_student"
    gender1 = "sex1_student"
    # if mini:
        # gender0 += "_mini"
        # gender1 += "_mini"
    
    df0 = pd.read_csv(f"../../student-dataset/{gender0}.csv")
    df1 = pd.read_csv(f"../../student-dataset/{gender1}.csv")
    # if mini: 
        # assert(df0.shape == df1.shape == (1000, 12))
    # else:
    assert(df0.shape == df1.shape == (100000, 32))

    assert(not df0.equals(df1))
    assert(df0.drop('sex', axis=1).equals(df1.drop('sex', axis=1)))     # after dropping sex they should be equal dataframe

    class0_ = df0.to_numpy(dtype=np.float64)
    class1_ = df1.to_numpy(dtype=np.float64)

    if disparateremoved:
        class0 = rescale_input_numpy_disparateremoved_compas(class0_)
        class1 = rescale_input_numpy_disparateremoved_compas(class1_)
    else:
        class0 = rescale_input_numpy(class0_)
        class1 = rescale_input_numpy(class1_)

    return class0, class1