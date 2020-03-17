import pandas as pd
import numpy as np

def raw_to_no_missing():
    df = pd.read_csv("raw_default.csv")
    # There is no missing rows, so all good
    # import ipdb; ipdb.set_trace()
    assert(df.dropna().shape == df.shape)
    # Change female from 2 to 0
    # df.SEX = df.SEX.replace({2:0})
    # df.to_csv("raw_default.csv", index=False)
    # There are no non-numerical columns


def missing_to_normalized():
    df = pd.read_csv("raw_default.csv")
    # df['sex'] = df['sex'].replace({"M":1, "F":0})
    # import ipdb; ipdb.set_trace()
    for i in df.columns:
        if df[i].dtype == "O":
            # print(i)
            assert False        # There are no such columns
            df[i], mapping_index = pd.Series(df[i]).factorize()

    target = df['target']
    df_new = df.drop(columns=['target'])
    df_new = df_new.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))          # normlization: scales the data between 0 and 1
    df_new.to_csv("normalized_default_features.csv", index=False, header=True)
    target.to_csv("default_labels.csv", index=False, header=True)


def print_mins_and_ranges():
    df = pd.read_csv("raw_default.csv")
    for i in df.columns:
        if df[i].dtype == "O":
            assert False
            df[i], mapping_index = pd.Series(df[i]).factorize()

    target = df['target']
    df_new = df.drop(columns=['target'])
    means_and_ranges = []
    for j in list(df_new):
        i = df[j]
        means_and_ranges.append((np.min(i), np.max(i) - np.min(i)))
    print(len(means_and_ranges), "\n", means_and_ranges)


import sys
if __name__ == "__main__":
    cleaning_level = int(sys.argv[1])
    if cleaning_level == 1:
        raw_to_no_missing()
    elif cleaning_level == 2:
       missing_to_normalized() 
    else:
        print_mins_and_ranges()
        
