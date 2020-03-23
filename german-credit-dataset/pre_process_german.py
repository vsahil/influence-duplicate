import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
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
# df = df.rename(columns={"class": "target"})
for i in df.columns:
    # assert(isinstance(df[i].dtype, np.int64))
    # print(i, df[i].dtype)
    assert is_numeric_dtype(df[i])

df_normalized = df.drop('target', axis=1)
# df_normalized = df_normalized.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
df_normalized = df_normalized.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))        # corrected to min, instead of mean
df.to_csv("german_redone.csv", index=False)
df_normalized.to_csv("german_redone_normalized.csv", index=False)

# target = df['target']
# target.to_csv("check_lables.csv", index=False)      # same as labels.csv