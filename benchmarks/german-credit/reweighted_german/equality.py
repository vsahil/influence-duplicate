import pandas as pd
df = pd.read_csv("reweighted_german.csv")
new_cols = ['status', 'month', 'credit_history', 'purpose', 'credit_amount', 'savings', 'employment', 'investment_as_income_percentage', 'sex', 'other_debtors', 'residence_since', 'property', 'age', 'installment_plans', 'housing', 'number_of_credits', 'skill_level', 'people_liable_for', 'telephone', 'foreign_worker', 'credit']
df = df[new_cols]
# df.to_csv("rearranged_reweighted_german.csv", index=False)
df0 = pd.read_csv("/Users/sahilverma/research/influence-duplicate/german-credit-dataset/original_german.csv")
df0.equals(df)      # False
(df0.values == df.values).all()     # True, proving they are equal

# Experiment useless because their is no reweighing