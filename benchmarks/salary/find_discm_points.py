import numpy as np
import copy, os
import pandas as pd


def rescale_input_numpy(inp):
    assert(inp.shape[1] == 5)  # 12 features for adult income dataset
    mins_and_ranges = [(0, 1), (1, 2), (0, 25), (0, 1), (1, 34)]
    r = np.arange(5)
    out = copy.deepcopy(inp)
    for col, (min_, range_) in zip(r, mins_and_ranges):
        out[:, col] = np.divide(np.subtract(out[:, col], min_), range_)
    return out


def rescale_input_numpy_disparateremoved(inp, mins_and_ranges):
    assert(inp.shape[1] == 5)  # 12 features for adult income dataset
    # mins_and_ranges  = [(0.0, 4.0), (0.0, 6.0), (13492.0, 1441943.0), (0.0, 15.0), (0.0, 6.0), (0.0, 13.0), (0.0, 4.0), (0.0, 1.0), (0.0, 4.0), (0.0, 4.0), (0.0, 4.0), (0.0, 39.0)]
    r = np.arange(5)
    out = copy.deepcopy(inp)
    for col, (min_, range_) in zip(r, mins_and_ranges):
        out[:, col] = np.divide(np.subtract(out[:, col], min_), range_)
    return out


def entire_test_suite(mini=False, disparateremoved=False, mins_and_ranges=None):
    gender0 = "sex0_salary"
    gender1 = "sex1_salary"
    if mini:
        raise NotImplementedError
        gender0 += "_mini"
        gender1 += "_mini"
    
    # now I can have headers in the tests files
    df0 = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../salary-dataset/{gender0}.csv")
    df1 = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../salary-dataset/{gender1}.csv")
    
    assert(df0.shape == df1.shape == (5200, 5))

    assert(not df0.equals(df1))
    assert(df0.drop('sex', axis=1).equals(df1.drop('sex', axis=1)))     # after dropping sex they should be equal dataframe

    class0_ = df0.to_numpy(dtype=np.float64)
    class1_ = df1.to_numpy(dtype=np.float64)

    if disparateremoved:
        class0 = rescale_input_numpy_disparateremoved(class0_, mins_and_ranges)
        class1 = rescale_input_numpy_disparateremoved(class1_, mins_and_ranges)
    else:
        assert mins_and_ranges == None
        class0 = rescale_input_numpy(class0_)
        class1 = rescale_input_numpy(class1_)

    return class0, class1
    