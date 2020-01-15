import numpy as np
import copy


def rescale_input_numpy(inp):
    assert(inp.shape[1] == 20)  # 20 features for german credit dataset
    means_and_ranges = [(1.001, 3), (20.903, 68), (32.545, 4), (47.148, 370), (3271.258, 18174), (1.19, 4), (3.384, 4), (2.973, 3), (0.69, 1), (101.145, 2), (2.845, 3), (122.358, 3), (35.546, 56), (142.675, 2), (151.929, 2), (1.407, 3), (172.904, 3), (1.155, 1), (1.404, 1), (1.037, 1)]
    r = np.arange(20)
    out = copy.deepcopy(inp)
    for col, (mean, range_) in zip(r, means_and_ranges):
        out[:, col] = np.divide(np.subtract(out[:, col], mean), range_)     
    return out


def entire_test_suite(mini=True):
    gender0 = "gender0"
    gender1 = "gender1"
    if mini:
        gender0 += "_mini"
        gender1 += "_mini"
    class0_ = np.genfromtxt("../german-credit-dataset/{}.csv".format(gender0), delimiter=",")
    class1_ = np.genfromtxt("../german-credit-dataset/{}.csv".format(gender1), delimiter=",")
    class0 = rescale_input_numpy(class0_)
    class1 = rescale_input_numpy(class1_)

    return class0, class1
    