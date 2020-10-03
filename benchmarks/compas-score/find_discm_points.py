import numpy as np
import copy, os
import pandas as pd
from load_compas_score_as_labels import dist


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


def entire_test_suite(mini=False, disparateremoved=False, mins_and_ranges=None, sensitive_removed=False):
    gender0 = f"race0_compas_two_year_dist{dist}"
    gender1 = f"race1_compas_two_year_dist{dist}"

    df0 = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../compas-dataset/{gender0}.csv")
    df1 = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../compas-dataset/{gender1}.csv")

    if "dist" in gender0:
        assert(df0.shape == df1.shape == (6150000, 10))
    else:
        assert(df0.shape == df1.shape == (615000, 10))

    assert(not df0.equals(df1))
    if not "dist" in gender0:
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
        if sensitive_removed:
            assert df0.columns.get_loc("race") == df1.columns.get_loc("race")
            sensitive_col = df0.columns.get_loc("race")
            class0 = np.delete(class0, sensitive_col, axis=1)
            class1 = np.delete(class1, sensitive_col, axis=1)

    return class0, class1