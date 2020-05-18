import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

def raw_to_no_missing():
    df = pd.read_csv("salary.csv")
    # remove rows with missing values
    df1 = df.dropna()
    assert(df1.shape == df.shape)
    target = df['salary']
    target = (target > target.median()).astype(int)
    df['salary'] = target
    df['sex'] = df['sex'].replace({'male':1, 'female':0})
    df['rank'] = df['rank'].replace({'full':3, 'associate':2, 'assistant':1})
    df['degree'] = df['degree'].replace({'doctorate':1, 'masters':0})
    df.to_csv("numeric_salary.csv", index=False)
    

def missing_to_normalized():
    df = pd.read_csv("salary.csv")
    target = df['salary']
    # import ipdb; ipdb.set_trace()
    target = (target > target.median()).astype(int)
    df = df.drop(columns=['salary'])
    df['sex'] = df['sex'].replace({'male':1, 'female':0})
    df['rank'] = df['rank'].replace({'full':3, 'associate':2, 'assistant':1})
    df['degree'] = df['degree'].replace({'doctorate':1, 'masters':0})
    df_ = df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    df_.to_csv("normalized_salary_features.csv", index=False, header=True)
    target.to_csv("salary_labels.csv", index=False, header=True)


def print_mins_and_ranges():
    df = pd.read_csv("salary.csv")
    df = df.drop(columns=['salary'])
    df['sex'] = df['sex'].replace({'male':1, 'female':0})
    df['rank'] = df['rank'].replace({'full':3, 'associate':2, 'assistant':1})
    df['degree'] = df['degree'].replace({'doctorate':1, 'masters':0})
    mins_and_ranges = []
    for j in list(df):
        i = df[j]
        mins_and_ranges.append((np.min(i), np.max(i) - np.min(i)))
    print(len(mins_and_ranges), mins_and_ranges)



import sys
if __name__ == "__main__":
    cleaning_level = int(sys.argv[1])
    if cleaning_level == 1:
        raw_to_no_missing()
    elif cleaning_level == 2:
       missing_to_normalized() 
    elif cleaning_level == 3:
        print_mins_and_ranges()
    else:
        assert False
        
