import pandas as pd
import numpy as np

def raw_to_no_missing():
    df = pd.read_csv("student-por.csv")
    # There is no missing rows, so all good
    # import ipdb; ipdb.set_trace()
    assert(df.dropna().shape == df.shape)
    # all the non-numerical columns are categorical
    # [len(df[i].unique()) for i in df.columns if df[i].dtype == 'O']
    # [2, 2, 2, 2, 2, 5, 5, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2]
    # df_new.to_csv("missing_compas_removed.csv", index=False)


def missing_to_normalized():
    df = pd.read_csv("student-por.csv")
    df['sex'] = df['sex'].replace({"M":1, "F":0})
    # import ipdb; ipdb.set_trace()
    for i in df.columns:
        if df[i].dtype == "O":
            # print(i)
            df[i], mapping_index = pd.Series(df[i]).factorize()

    target = df['G3']
    df_new = df.drop(columns=['G3'])
    df_new = df_new.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))          # normlization: scales the data between 0 and 1
    df_new.to_csv("normalized_student_features.csv", index=False, header=True)
    target = target.apply(lambda x: 0 if x <= 11 else 1)        # I put 11 instead of 10 to have a balanced dataset; both 10 and 12 lead to a higher imbalance
    target.to_csv("student_labels.csv", index=False, header=True)


def print_mins_and_ranges():
    df = pd.read_csv("student-por.csv")
    df['sex'] = df['sex'].replace({"M":1, "F":0})
    for i in df.columns:
        if df[i].dtype == "O":
            df[i], mapping_index = pd.Series(df[i]).factorize()

    target = df['G3']
    df_new = df.drop(columns=['G3'])
    mins_and_ranges = []
    for j in list(df_new):
        i = df[j]
        mins_and_ranges.append((np.min(i), np.max(i) - np.min(i)))
    print(len(mins_and_ranges), "\n", mins_and_ranges)


def convert_to_nosensitive():
    df = pd.read_csv("normalized_student_features.csv")
    df['sex'] = 1
    df.to_csv("normalized_student_nosensitive_features.csv", index=False, header=True)



import sys
if __name__ == "__main__":
    cleaning_level = int(sys.argv[1])
    if cleaning_level == 1:
        raw_to_no_missing()
    elif cleaning_level == 2:
       missing_to_normalized() 
    elif cleaning_level == 3:
        print_mins_and_ranges()
    elif cleaning_level == 4:
        convert_to_nosensitive()
    else:
        assert False
        
