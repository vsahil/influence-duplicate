import pandas as pd
import numpy as np
df = pd.read_csv("adult_no_missing.csv")
# normalizes the features (not labels) and prints them to a file
target = df['target']
df_new = df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
# df_new.to_csv("normalized_adult_features.csv", index=False, header=True)

mins_and_ranges = []
df = df.drop(columns=['target'])
for j in list(df):
    i = df[j]
    mins_and_ranges.append((np.min(i), np.max(i) - np.min(i)))
print(mins_and_ranges)
