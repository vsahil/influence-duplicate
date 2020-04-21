import pandas as pd
import numpy as np
df = pd.read_csv("german_redone.csv")
df_new = df.drop(columns=['target'])
mins_and_ranges = []
for j in list(df_new):
    i = df[j]
    mins_and_ranges.append((np.min(i), np.max(i) - np.min(i)))
print(mins_and_ranges)
print(len(mins_and_ranges))