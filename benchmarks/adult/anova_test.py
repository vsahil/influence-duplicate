import pandas as pd
import researchpy as rp
import seaborn as sns

import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp


def find_min_discm_each_model(df):
    x = df.sort_values("Discm_percent").groupby("Model-count", as_index=False).first()    
    assert len(x) == 240
    return x

# df = pandas.read_csv("min_discm.csv")
# df = pandas.read_csv("removal.csv")
# df1 = pd.read_csv("adult_results_first120.csv")
# df2 = pd.read_csv("adult_results_last120.csv")
# df3 = pd.read_csv("adult_results_lastest80.csv")
# df = pd.concat([df1, df2, df3])
# if len(df) > 240:
#     df = find_min_discm_each_model(df)
# df.to_csv("min_discm.csv", index=False)
# exit(0)
df = pd.read_csv("min_discm.csv")
# import ipdb; ipdb.set_trace()
# print(rp.summary_cont(df['Discm']))
# print(rp.summary_cont(df.groupby(['Permutation']))['Discm'])
model = ols('Discm_percent ~ C(H1Units) + C(H2Units) + C(Batch) + C(DataSplit)', df).fit()
# model = ols('Removal ~ C(H1units) + C(H2units) + C(Batch) + C(Permutation)', df).fit()
# model = ols('Error ~ C(H1units) + C(H2units) + C(Batch) + C(Permutation)', df).fit()
print(f"Overall model F({model.df_model: .0f},{model.df_resid: .0f}) = {model.fvalue: .3f}, p = {model.f_pvalue: .4f}")
print(model.summary())
# print(rp.summary_cont(df.groupby(['H2units']))['Error'])
res = sm.stats.anova_lm(model, typ= 2)
print(res)


#
# from sklearn.linear_model import LinearRegression
# target_col = 'Discm'
# X = df.loc[:, df.columns != target_col]
# Y = df.loc[:, target_col]
# lr = LinearRegression()
# lr.fit(X, Y)
# print(lr.coef_)
# print(lr.intercept_)