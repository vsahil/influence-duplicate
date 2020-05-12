import numpy as np
import copy, os
import pandas as pd


def rescale_input_numpy(inp):
    assert(inp.shape[1] == 23)  # 23 features for default dataset
    mins_and_ranges = [(10000, 990000), (0, 1), (0, 6), (0, 3), (21, 58), (-2, 10), (-2, 10), (-2, 10), (-2, 10), (-2, 10), (-2, 10), (-165580, 1130091), (-69777, 1053708), (-157264, 1821353), (-170000, 1061586), (-81334, 1008505), (-339603, 1301267), (0, 873552), (0, 1684259), (0, 896040), (0, 621000), (0, 426529), (0, 528666)]
    r = np.arange(23)
    out = copy.deepcopy(inp)
    for col, (min_, range_) in zip(r, mins_and_ranges):
        out[:, col] = np.divide(np.subtract(out[:, col], min_), range_)
    return out


def rescale_input_numpy_disparateremoved_compas(inp, mins_and_ranges):
    assert(inp.shape[1] == 23)  # 23 features for default prediction dataset
    # mins_and_ranges  = [(10000, 790000), (0, 1), (0, 6), (0, 3), (21, 54), (-2, 10), (-2, 9), (-2, 9), (-2, 9), (-2, 9), (-2, 9), (-165580, 912394), (-69777, 716547), (-157264, 850395), (-170000, 798699), (-81334, 904874), (-339603, 1039547), (0, 505000), (0, 388126), (0, 508229), (0, 528897), (0, 332000), (0, 527143)]
    r = np.arange(23)
    out = copy.deepcopy(inp)
    for col, (min_, range_) in zip(r, mins_and_ranges):
        out[:, col] = np.divide(np.subtract(out[:, col], min_), range_)
    return out


def entire_test_suite(mini=False, disparateremoved=False, mins_and_ranges=None):
    gender0 = "sex0_default"
    gender1 = "sex1_default"
    # if mini:
        # gender0 += "_mini"
        # gender1 += "_mini"
    
    df0 = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../default-dataset/{gender0}.csv")
    df1 = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../default-dataset/{gender1}.csv")
    # if mini: 
        # assert(df0.shape == df1.shape == (1000, 12))
    # else:
    assert(df0.shape == df1.shape == (3000000, 23))

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