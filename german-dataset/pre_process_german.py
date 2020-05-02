import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

def raw_to_no_missing():
    o_df = pd.read_csv("original_german.csv")
    # remove rows with missing values
    df = o_df.dropna()
    assert df.shape[0] == o_df.shape[0]

    # Change all categorical features into numeric
    df['Checking-ccount'], mapping_index = pd.Series(df['Checking-ccount']).factorize()
    df['Credit-history'], mapping_index = pd.Series(df['Credit-history']).factorize()
    df['Purpose'], mapping_index = pd.Series(df['Purpose']).factorize()
    df['Svings-ccount'], mapping_index = pd.Series(df['Svings-ccount']).factorize()
    df['Present-employment-since'], mapping_index = pd.Series(df['Present-employment-since']).factorize()
    df['Other-debtors'], mapping_index = pd.Series(df['Other-debtors']).factorize()
    df['Property'], mapping_index = pd.Series(df['Property']).factorize()
    df['Other-instllment-plns'], mapping_index = pd.Series(df['Other-instllment-plns']).factorize()
    df['Housing'], mapping_index = pd.Series(df['Housing']).factorize()
    df['Job'], mapping_index = pd.Series(df['Job']).factorize()
    df['Telephone'], mapping_index = pd.Series(df['Telephone']).factorize()
    df['Foreign-worker'], mapping_index = pd.Series(df['Foreign-worker']).factorize()

    outcome_map = {1:1, 2:0}
    df['target'] = df['target'].replace(outcome_map)
    for i in df.columns:
        # assert(isinstance(df[i].dtype, np.int64))
        # print(i, df[i].dtype)
        assert is_numeric_dtype(df[i])

    df_normalized = df.drop('target', axis=1)
    df_normalized = df_normalized.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))        # corrected to min, instead of mean
    df.to_csv("german_redone.csv", index=False)
    df_normalized.to_csv("german_redone_normalized_withheader.csv", index=False)


def missing_to_normalized():
    df = pd.read_csv("german_redone.csv")
    target = df['target']
    df = df.drop('target', axis=1)
    df_normalized = df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))        # corrected to min, instead of mean
    df_normalized.to_csv("german_redone_normalized_withheader.csv", index=False)
    target.to_csv("german_labels_withheader.csv", index=False)      # same as labels.csv


def print_mins_and_ranges():
    df = pd.read_csv("german_redone.csv")
    df_new = df.drop(columns=['target'])
    mins_and_ranges = []
    for j in list(df_new):
        i = df[j]
        mins_and_ranges.append((np.min(i), np.max(i) - np.min(i)))
    print(mins_and_ranges)
    print(len(mins_and_ranges))


def convert_to_nosensitive():
    df = pd.read_csv("german_redone_normalized_withheader.csv")
    df['Gender'] = 1
    df.to_csv("normalized_german_nosensitive_features.csv", index=False, header=True)



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
        
