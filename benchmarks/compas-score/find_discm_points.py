import numpy as np
import copy, os
import pandas as pd


def rescale_input_numpy(inp):
    assert(inp.shape[1] == 10)  # 10 features for compas dataset
    mins_and_ranges = [(18, 65), (0, 1), (0, 1), (0, 6035), (0, 799), (0, 38), (0, 10), (0, 13), (0, 17), (0, 1)]
    r = np.arange(10)
    out = copy.deepcopy(inp)
    for col, (min_, range_) in zip(r, mins_and_ranges):
        out[:, col] = np.divide(np.subtract(out[:, col], min_), range_)
    return out


def rescale_input_numpy_disparateremoved_compas(inp, mins_and_ranges):
    assert(inp.shape[1] == 10)  # 10 features for compas dataset
    # mins_and_ranges  = [(18.0, 59.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1940.0), (0.0, 530.0), (0.0, 36.0), (0.0, 8.0), (0.0, 6.0), (0.0, 7.0), (0.0, 1.0)]
    r = np.arange(10)
    out = copy.deepcopy(inp)
    for col, (min_, range_) in zip(r, mins_and_ranges):
        out[:, col] = np.divide(np.subtract(out[:, col], min_), range_)
    return out


def entire_test_suite(mini=False, disparateremoved=False, mins_and_ranges=None):
    gender0 = "race0_compas_two_year"
    gender1 = "race1_compas_two_year"

    df0 = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../compas-dataset/{gender0}.csv")
    df1 = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../compas-dataset/{gender1}.csv")

    assert(df0.shape == df1.shape == (615000, 10))

    assert(not df0.equals(df1))
    assert(df0.drop('race', axis=1).equals(df1.drop('race', axis=1)))     # after dropping sex they should be equal dataframe

    class0_ = df0.to_numpy(dtype=np.float64)
    class1_ = df1.to_numpy(dtype=np.float64)

    if disparateremoved:
        class0 = rescale_input_numpy_disparateremoved_compas(class0_, mins_and_ranges)
        class1 = rescale_input_numpy_disparateremoved_compas(class1_, mins_and_ranges)
    else:
        assert mins_and_ranges == None
        class0 = rescale_input_numpy(class0_)
        class1 = rescale_input_numpy(class1_)

    return class0, class1