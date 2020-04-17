import pandas as pd
import numpy as np
df = pd.read_csv("adult_no_missing.csv")
# import ipdb; ipdb.set_trace()

# features.to_csv("german_gender_reversed.csv", index=False)
# normalizes the features (not labels) and prints them to a file
target = df['target']
# _map = {1: 1, 2: 0}
# target = target.replace(_map)
df_new = df.drop(columns=['target'])
df_ = df_new.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
# df_.to_csv("normalized_adult_features.csv", index=False, header=True)
# target.to_csv("adult_labels.csv", index=False, header=True)

mins_and_ranges = []
for j in list(df_new):
    i = df[j]
    mins_and_ranges.append((np.min(i), np.max(i) - np.min(i)))
print(len(mins_and_ranges), mins_and_ranges)
