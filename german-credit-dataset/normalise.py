import pandas as pd
import numpy as np
df = pd.read_csv("german_redone.csv")
df_new = df.drop(columns=['target'])
means_and_ranges = []
for j in list(df_new):
    i = df[j]
    means_and_ranges.append((np.mean(i), np.max(i) - np.min(i)))
print(means_and_ranges)
print(len(means_and_ranges))