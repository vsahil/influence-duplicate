import numpy as np
import copy, os
import pandas as pd


def rescale_input_numpy(inp):
    assert(inp.shape[1] == 12)  # 12 features for adult income dataset
    means_and_ranges = [(1.7695369510415284, 4), (2.1092388660386536, 6), (189734.7343107337, 1476908), (3.4133165273539428, 15), (1.0784795011277697, 6), (4.628012914068374, 13), (0.21120251205165627, 4), (0.6750475432311707, 1), (0.20505506169563487, 4), (0.1165583123258591, 4), (1.98805890938039, 4), (1.1686347353058246, 40)]
    r = np.arange(12)
    out = copy.deepcopy(inp)
    for col, (mean, range_) in zip(r, means_and_ranges):
        out[:, col] = np.divide(np.subtract(out[:, col], mean), range_)
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
    df0 = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../adult-income-dataset/{gender0}.csv")
    df1 = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../adult-income-dataset/{gender1}.csv")
    if mini: 
        assert(df0.shape == df1.shape == (1000, 12))
    else:
        assert(df0.shape == df1.shape == (4522200, 12))

    assert(not df0.equals(df1))
    assert(df0.drop('sex', axis=1).equals(df1.drop('sex', axis=1)))     # after dropping sex they should be equal dataframe

    class0_ = df0.to_numpy(dtype=np.float64)
    class1_ = df1.to_numpy(dtype=np.float64)

    # class0_ = np.genfromtxt(f"/Users/sahilverma/research/influence-duplicate/adult-income-dataset/{gender0}.csv", delimiter=",")
    # class1_ = np.genfromtxt(f"/Users/sahilverma/research/influence-duplicate/adult-income-dataset/{gender1}.csv", delimiter=",")
    
    if disparateremoved:
        class0 = rescale_input_numpy_disparateremoved(class0_)
        class1 = rescale_input_numpy_disparateremoved(class1_)
    else:
        class0 = rescale_input_numpy(class0_)
        class1 = rescale_input_numpy(class1_)

    return class0, class1
    