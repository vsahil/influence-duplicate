import pandas as pd
import numpy as np
df = pd.read_csv("disparate_impact_removed_student.csv")
# normalizes the features (not labels) and prints them to a file
target = df['G3']
df_new = df.drop(columns=['G3'])
df_ = df_new.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
df_.to_csv("normalized_disparateremoved_features-student.csv", index=False, header=False)
target.to_csv("normalized_disparateremoved_labels-student.csv", index=False, header=False)

mins_and_ranges = []
for j in list(df_new):
    i = df[j]
    mins_and_ranges.append((np.min(i), np.max(i) - np.min(i)))
print(mins_and_ranges)
