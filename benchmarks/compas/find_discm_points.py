import numpy as np
import copy, os
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
    assert(inp.shape[1] == 19)  # 19 features for compas dataset
    mins_and_ranges  = [(0.0, 1.0), (18.0, 59.0), (0.0, 1.0), (0.0, 8.0), (1.0, 9.0), (0.0, 6.0), (0.0, 7.0), (0.0, 36.0), (-414.0, 1439.0), (0.0, 4696.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (1.0, 9.0), (1.0, 9.0), (0.0, 36.0), (0.0, 640.0), (0.0, 1186.0), (0.0, 1.0)]
    r = np.arange(19)
    out = copy.deepcopy(inp)
    for col, (min_, range_) in zip(r, mins_and_ranges):
        out[:, col] = np.divide(np.subtract(out[:, col], min_), range_)
    return out


def entire_test_suite(mini=False, disparateremoved=False):
    gender0 = "race0_compas_two_year"
    gender1 = "race1_compas_two_year"
    # if mini:
        # gender0 += "_mini"
        # gender1 += "_mini"
    
    df0 = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../compas-dataset/{gender0}.csv")
    df1 = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../compas-dataset/{gender1}.csv")
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