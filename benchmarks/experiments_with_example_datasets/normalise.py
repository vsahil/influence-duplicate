import pandas as pd
import numpy as np
# df = pd.read_csv("race0_biased.csv")
# import ipdb; ipdb.set_trace()

# features.to_csv("german_gender_reversed.csv", index=False)
# normalizes the features (not labels) and prints them to a file
# target = df['target']
# _map = {1: 1, 2: 0}
# target = target.replace(_map)
# df_new = df.drop(columns=['target'])
# import ipdb; ipdb.set_trace()
# df_new = df_new.reindex(['status','month','credit_history','purpose','credit_amount','savings','employment','investment_as_income_percentage','sex','other_debtors','residence_since','property','age','installment_plans','housing','number_of_credits','skill_level','people_liable_for','telephone','foreign_worker'], axis=1)
df = pd.read_csv("race0_biased.csv")
# modDfObj = df.apply(lambda x: np.square(x) if x.name == 'Income' else x)
# df_new = df.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)) if x.name == 'Income' else x )
# Note that makes data between 0 and 1
df['Neighbor-income'] = df['Neighbor-income'].replace({0:0, 1:0.5, 2:1})    # integers were used for generating test cases, floating numbers were used in training set
df_new = df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)) if x.name == 'Income' else x )
df_new.to_csv("normalized_race0.csv", index=False, header=False)
del df
df = pd.read_csv("race1_biased.csv")
df['Neighbor-income'] = df['Neighbor-income'].replace({0:0, 1:0.5, 2:1})    # integers were used for generating test cases, floating numbers were used in training set
df_new = df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)) if x.name == 'Income' else x )
df_new.to_csv("normalized_race1.csv", index=False, header=False)
# target.to_csv("normalized_adult_labels.csv", index=False, header=False)

# means_and_ranges = []
# for j in list(df_new):
#     i = df[j]
#     means_and_ranges.append((np.mean(i), np.max(i) - np.min(i)))
# print(means_and_ranges)
