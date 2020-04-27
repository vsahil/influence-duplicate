import numpy as np
import copy, os
import pandas as pd


def rescale_input_numpy(inp):
    assert(inp.shape[1] == 12)  # 12 features for adult income dataset
    mins_and_ranges = [(0, 4), (0, 6), (13492, 1476908), (0, 15), (0, 6), (0, 13), (0, 4), (0, 1), (0, 4), (0, 4), (0, 4), (0, 40)]
    r = np.arange(12)
    out = copy.deepcopy(inp)
    for col, (min_, range_) in zip(r, mins_and_ranges):
        out[:, col] = np.divide(np.subtract(out[:, col], min_), range_)
    return out


def rescale_input_numpy_disparateremoved(inp):
    assert(inp.shape[1] == 12)  # 12 features for adult income dataset
    mins_and_ranges  = [(0.0, 4.0), (0.0, 6.0), (13492.0, 1441943.0), (0.0, 15.0), (0.0, 6.0), (0.0, 13.0), (0.0, 4.0), (0.0, 1.0), (0.0, 4.0), (0.0, 4.0), (0.0, 4.0), (0.0, 39.0)]
    r = np.arange(12)
    out = copy.deepcopy(inp)
    for col, (min_, range_) in zip(r, mins_and_ranges):
        out[:, col] = np.divide(np.subtract(out[:, col], min_), range_)
    return out


def entire_test_suite(mini=True, disparateremoved=False):
    gender0 = "gender0_adult"
    gender1 = "gender1_adult"
    if mini:
        gender0 += "_mini"
        gender1 += "_mini"
    
    # now I can have headers in the tests files
    df0 = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../adult-dataset/{gender0}.csv")
    df1 = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../adult-dataset/{gender1}.csv")
    if mini: 
        assert(df0.shape == df1.shape == (1000, 12))
    else:
        assert(df0.shape == df1.shape == (4522200, 12))

    assert(not df0.equals(df1))
    assert(df0.drop('sex', axis=1).equals(df1.drop('sex', axis=1)))     # after dropping sex they should be equal dataframe

    class0_ = df0.to_numpy(dtype=np.float64)
    class1_ = df1.to_numpy(dtype=np.float64)

    if disparateremoved:
        class0 = rescale_input_numpy_disparateremoved(class0_)
        class1 = rescale_input_numpy_disparateremoved(class1_)
    else:
        class0 = rescale_input_numpy(class0_)
        class1 = rescale_input_numpy(class1_)

    return class0, class1
    