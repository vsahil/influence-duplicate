import numpy as np
import copy, os
import pandas as pd


def rescale_input_numpy(inp):
    assert(inp.shape[1] == 20)  # 20 features for german credit dataset
    mins_and_ranges = [(0, 3), (4, 68), (0, 4), (0, 9), (250, 18174), (0, 4), (0, 4), (1, 3), (0, 1), (0, 2), (1, 3), (0, 3), (19, 56), (0, 2), (0, 2), (1, 3), (0, 3), (1, 1), (0, 1), (0, 1)]
    r = np.arange(20)
    out = copy.deepcopy(inp)
    for col, (min_, range_) in zip(r, mins_and_ranges):
        out[:, col] = np.divide(np.subtract(out[:, col], min_), range_)
    return out


def rescale_input_numpy_disparateremoved_german(inp, mins_and_ranges):
    assert(inp.shape[1] == 20)  # 20 features for german credit dataset
    # mins_and_ranges  = [(0, 3), (4, 56), (0, 4), (0, 9), (250, 15607), (0, 4), (0, 4), (1, 3), (0, 1), (0, 2), (1, 3), (0, 3), (19, 56), (0, 2), (0, 2), (1, 3), (0, 3), (1, 1), (0, 1), (0, 1)]
    r = np.arange(20)
    out = copy.deepcopy(inp)
    for col, (min_, range_) in zip(r, mins_and_ranges):
        out[:, col] = np.divide(np.subtract(out[:, col], min_), range_)
    return out


def entire_test_suite(mini=True, disparateremoved=False, mins_and_ranges=None, sensitive_removed=False):
    # gender0 = "gender0_redone_german"
    # gender1 = "gender1_redone_german"
    gender0 = "gender0_german_dist10"
    gender1 = "gender1_german_dist10"
    if mini:
        assert False
        gender0 += "_mini"
        gender1 += "_mini"
    df0 = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../german-dataset/{gender0}.csv")
    df1 = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../german-dataset/{gender1}.csv")

    if "dist10" in gender0:
        assert(df0.shape == df1.shape == (1000000, 20))
    else:
        assert(df0.shape == df1.shape == (100000, 20))

    assert(not df0.equals(df1))
    if not "dist10" in gender0:
        assert(df0.drop('Gender', axis=1).equals(df1.drop('Gender', axis=1)))     # after dropping sex they should be equal dataframe

    class0_ = df0.to_numpy(dtype=np.float64)
    class1_ = df1.to_numpy(dtype=np.float64)

    if disparateremoved:
        class1 = rescale_input_numpy_disparateremoved_german(class1_, mins_and_ranges)
        class0 = rescale_input_numpy_disparateremoved_german(class0_, mins_and_ranges)
    else:
        assert mins_and_ranges == None
        class0 = rescale_input_numpy(class0_)
        class1 = rescale_input_numpy(class1_)
        if sensitive_removed:
            assert df0.columns.get_loc("Gender") == df1.columns.get_loc("Gender")
            sensitive_col = df0.columns.get_loc("Gender")
            class0 = np.delete(class0, sensitive_col, axis=1)
            class1 = np.delete(class1, sensitive_col, axis=1)

    return class0, class1
