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


def rescale_input_numpy_disparateremoved_german(inp):
    assert(inp.shape[1] == 20)  # 20 features for german credit dataset
    mins_and_ranges  = [(0, 3), (4, 56), (0, 4), (0, 9), (250, 15607), (0, 4), (0, 4), (1, 3), (0, 1), (0, 2), (1, 3), (0, 3), (19, 56), (0, 2), (0, 2), (1, 3), (0, 3), (1, 1), (0, 1), (0, 1)]
    r = np.arange(20)
    out = copy.deepcopy(inp)
    for col, (min_, range_) in zip(r, mins_and_ranges):
        out[:, col] = np.divide(np.subtract(out[:, col], min_), range_)
    return out


def entire_test_suite(mini=True, disparateremoved=False):
    gender0 = "gender0_redone_german"
    gender1 = "gender1_redone_german"
    if mini:
        assert False
        gender0 += "_mini"
        gender1 += "_mini"
    class0_ = np.genfromtxt(f"{os.path.dirname(os.path.realpath(__file__))}/../../german-dataset/{gender0}.csv", delimiter=",")
    class1_ = np.genfromtxt(f"{os.path.dirname(os.path.realpath(__file__))}/../../german-dataset/{gender1}.csv", delimiter=",")

    if disparateremoved:
        class0 = rescale_input_numpy_disparateremoved_german(class0_)
        class1 = rescale_input_numpy_disparateremoved_german(class1_)
    else:
        class0 = rescale_input_numpy(class0_)
        class1 = rescale_input_numpy(class1_)

    return class0, class1