import numpy as np
import copy, os
import pandas as pd


def rescale_input_numpy(inp):
    assert(inp.shape[1] == 32)  # 32 features for student dataset
    mins_and_ranges = [(0, 1), (0, 1), (15, 7), (0, 1), (0, 1), (0, 1), (0, 4), (0, 4), (0, 4), (0, 4), (0, 3), (0, 2), (1, 3), (1, 3), (0, 3), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4), (0, 32), (0, 19), (0, 19)]
    r = np.arange(32)
    out = copy.deepcopy(inp)
    for col, (min_, range_) in zip(r, mins_and_ranges):
        out[:, col] = np.divide(np.subtract(out[:, col], min_), range_)
    return out


def rescale_input_numpy_disparateremoved_compas(inp, mins_and_ranges):
    assert(inp.shape[1] == 32)  # 32 features for student dataset
    # mins_and_ranges  = [(0, 1), (0, 1), (15, 6), (0, 1), (0, 1), (0, 1), (0, 4), (0, 4), (0, 4), (0, 4), (0, 3), (0, 2), (1, 3), (1, 3), (0, 3), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4), (0, 26), (0, 18), (0, 18)]     # the last one is not target, so don't change to (0, 1)
    r = np.arange(32)
    out = copy.deepcopy(inp)
    for col, (min_, range_) in zip(r, mins_and_ranges):
        out[:, col] = np.divide(np.subtract(out[:, col], min_), range_)
    return out


def entire_test_suite(mini=False, disparateremoved=False, mins_and_ranges=None):
    gender0 = "sex0_student"
    gender1 = "sex1_student"
    # if mini:
        # gender0 += "_mini"
        # gender1 += "_mini"
    df0 = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../student-dataset/{gender0}.csv")
    df1 = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../student-dataset/{gender1}.csv")
    # if mini: 
        # assert(df0.shape == df1.shape == (1000, 12))
    # else:
    assert(df0.shape == df1.shape == (64900, 32))

    assert(not df0.equals(df1))
    assert(df0.drop('sex', axis=1).equals(df1.drop('sex', axis=1)))     # after dropping sex they should be equal dataframe

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
