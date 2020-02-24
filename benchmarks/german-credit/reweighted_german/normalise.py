import pandas as pd
import numpy as np
df = pd.read_csv("reweighted_german.csv")
# import ipdb; ipdb.set_trace()

# features.to_csv("german_gender_reversed.csv", index=False)
# normalizes the features (not labels) and prints them to a file
target = df['credit']
_map = {1: 1, 2: 0}
target = target.replace(_map)
df_new = df.drop(columns=['credit'])
# import ipdb; ipdb.set_trace()
df_new = df_new.reindex(['status','month','credit_history','purpose','credit_amount','savings','employment','investment_as_income_percentage','sex','other_debtors','residence_since','property','age','installment_plans','housing','number_of_credits','skill_level','people_liable_for','telephone','foreign_worker'], axis=1)
# df_new = df_new.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
# df_new.to_csv("normalized_reweighted_features-german.csv", index=False, header=False)
# target.to_csv("normalized_reweighted_labels-german.csv", index=False, header=False)
means_and_ranges = []
for j in list(df_new):
    i = df[j]
    means_and_ranges.append((np.mean(i), np.max(i) - np.min(i)))
print(means_and_ranges)
