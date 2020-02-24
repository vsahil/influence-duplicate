import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
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
