import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

def raw_to_no_missing():
    df = pd.read_csv("adult_csv.csv")
    # remove rows with missing values
    df = df.dropna()

    # makes no sense to have, also this can lead to generate unrelistic data, like unmarried but married person, haha
    df = df.drop('relationship', axis=1)      

    for i in range(1, 17):
        assert(len(df.loc[df['education-num'] == i, 'education'].value_counts().index.tolist()) == 1)        # this means education-num and education have one-to one mapping, can drop one of them
    df = df.drop('education-num', axis=1)

    # Change all categorical features into numeric
    df.workclass, mapping_index = pd.Series(df.workclass).factorize()
    df.education, mapping_index = pd.Series(df.education).factorize()
    df['marital-status'], mapping_index = pd.Series(df['marital-status']).factorize()
    df.occupation, mapping_index = pd.Series(df.occupation).factorize()
    df.race, mapping_index = pd.Series(df.race).factorize()
    df['native-country'], mapping_index = pd.Series(df['native-country']).factorize()
    sex_map = {'Male':1, 'Female':0}
    df['sex'] = df['sex'].replace(sex_map)

    outcome_map = {'>50K':1, '<=50K':0}
    df['class'] = df['class'].replace(outcome_map)
    df = df.rename(columns={"class": "target"})
    for i in df.columns:
        # assert(isinstance(df[i].dtype, np.int64))
        # print(i, df[i].dtype)
        assert is_numeric_dtype(df[i])
    df.to_csv("adult_no_missing.csv", index=False)


def missing_to_normalized():
    df = pd.read_csv("adult_no_missing.csv")
    df.drop(df.loc[df['race']==2].index, inplace=True)
    df.drop(df.loc[df['race']==3].index, inplace=True)
    df.drop(df.loc[df['race']==4].index, inplace=True)
    assert len(df['race'].unique()) == 2
    target = df['target']
    df_new = df.drop(columns=['target'])
    df_ = df_new.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    df_.to_csv("normalized_adult_features_race.csv", index=False, header=True)
    target.to_csv("adult_labels_race.csv", index=False, header=True)


def print_mins_and_ranges():
    df = pd.read_csv("adult_no_missing.csv")
    df.drop(df.loc[df['race']==2].index, inplace=True)
    df.drop(df.loc[df['race']==3].index, inplace=True)
    df.drop(df.loc[df['race']==4].index, inplace=True)
    assert len(df['race'].unique()) == 2
    target = df['target']
    df_new = df.drop(columns=['target'])
    mins_and_ranges = []
    for j in list(df_new):
        i = df[j]
        mins_and_ranges.append((np.min(i), np.max(i) - np.min(i)))
    print(len(mins_and_ranges), mins_and_ranges)


# def convert_to_nosensitive():
#     df = pd.read_csv("normalized_adult_features.csv")
#     df['sex'] = 1
#     df.to_csv("normalized_adult_nosensitive_features.csv", index=False, header=True)


import sys
if __name__ == "__main__":
    cleaning_level = int(sys.argv[1])
    if cleaning_level == 1:
        pass
        # raw_to_no_missing()
    elif cleaning_level == 2:
       missing_to_normalized() 
    elif cleaning_level == 3:
        print_mins_and_ranges()
    elif cleaning_level == 4:
        pass        
        # convert_to_nosensitive()
    else:
        assert False
        
