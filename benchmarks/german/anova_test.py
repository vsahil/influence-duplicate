import pandas
import researchpy as rp
import seaborn as sns

import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp



# df = pandas.read_csv("min_discm.csv")
# df = pandas.read_csv("removal.csv")
df = pandas.read_csv("find_correlation_hyperparams.csv")
# import ipdb; ipdb.set_trace()
# print(rp.summary_cont(df['Discm']))
# print(rp.summary_cont(df.groupby(['Permutation']))['Discm'])
# model = ols('Discm ~ C(H1units) + C(H2units) + C(Batch) + C(Permutation)', df).fit()
# model = ols('Removal ~ C(H1units) + C(H2units) + C(Batch) + C(Permutation)', df).fit()
model = ols('Error ~ C(H1units) + C(H2units) + C(Batch) + C(Permutation)', df).fit()
print(f"Overall model F({model.df_model: .0f},{model.df_resid: .0f}) = {model.fvalue: .3f}, p = {model.f_pvalue: .4f}")
# print(model.summary())
print(rp.summary_cont(df.groupby(['H2units']))['Error'])
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