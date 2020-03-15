import pandas as pd
import numpy as np
# There is no sense in using compas_two_year.csv data because that shows the ground truth 2 year recidivism, so it makes no sense to remove rows over there.
# It makes only sense to remove rows in the compas.csv data. 

def raw_to_no_missing():
    # df_two_year = pd.read_csv("compas-scores-two-years.csv")
    df2 = pd.read_csv("compas-scores.csv")
    # import ipdb; ipdb.set_trace()
    # df_new = df2.dropna()       # each row has some column missing, so you can't choose this, try to drop columns
    # we should first normalise the columns and then put the missing values to be 0, then we are all good.
    # df2.columns[df2.isnull().any()] - Gives the list of column that have any missing values - len(22) it includes score_text, I will remove the row with missing score test first
    # 17 rows have missing score text - delete them. - Code to find them
    # df2[df2['score_text'].isnull()] - done
    # Remaining dataset size = 11742
    # find the number of times a column is missing 
    # [1174, 1174, 1174, 738, 2595, 9885, 738, 745, 11742, 8041, 9283, 8041, 8101, 9283, 9283, 11742, 10860, 10860, 10860, 10860]
    # I will drop columns which have missing value over 50% of the dataset size = 5871, there is anyway a clear divide between the no of rows missing for these columns
    import copy
    df_new = copy.deepcopy(df2)
    for i in df2.columns[df2.isnull().any()]:
        if df2[i].isnull().sum() > 5871:
            # DROP: c_arrest_date, num_r_cases, r_case_number, r_days_from_arrest, r_offense_date, r_charge_desc, r_jail_in, r_jail_out, num_vr_cases, vr_case_number, vr_charge_degree, vr_offense_date, vr_charge_desc
            # drop 13 out of 20 columns with missing values
            df_new = df_new.drop(i, axis=1)
    # print(df_new.shape, df2.shape)
    # drop columns with english text/dates as they are not processed by the system. Remove columns with only 1 value as well
    df_new = df_new.drop("id", axis=1)
    df_new = df_new.drop("name", axis=1)
    df_new = df_new.drop("first", axis=1)
    df_new = df_new.drop("last", axis=1)
    df_new = df_new.drop("compas_screening_date", axis=1)
    df_new = df_new.drop("dob", axis=1)
    df_new = df_new.drop("age_cat", axis=1)
    df_new = df_new.drop("c_jail_in", axis=1)
    df_new = df_new.drop("c_jail_out", axis=1)     # this is already dropped
    df_new = df_new.drop("c_case_number", axis=1)
    df_new = df_new.drop("c_offense_date", axis=1)
    # df_new = df_new.drop("c_arrest_date", axis=1)     # this is already dropped
    df_new = df_new.drop("c_charge_desc", axis=1)
    # df_new = df_new.drop("r_case_number", axis=1)     # this is already dropped
    # df_new = df_new.drop("r_offense_date", axis=1)     # this is already dropped
    # df_new = df_new.drop("r_charge_desc", axis=1)     # this is already dropped
    # df_new = df_new.drop("r_jail_in", axis=1)     # this is already dropped
    # df_new = df_new.drop("r_jail_out", axis=1)     # this is already dropped
    # df_new = df_new.drop("vr_case_number", axis=1)     # this is already dropped
    df_new = df_new.drop("v_type_of_assessment", axis=1)    # it has only one unique value: "Risk of violence"
    df_new = df_new.drop("v_screening_date", axis=1)
    df_new = df_new.drop("type_of_assessment", axis=1)    # it has only one unique value: "Risk of recidivism"
    df_new = df_new.drop("screening_date", axis=1)
    # You can also drop the score texts as you have the exact value of the score in range 0-10, it is informatically lossy
    df_new = df_new.drop("v_score_text", axis=1)
    df_new = df_new.drop("score_text", axis=1)
    # (df['decile_score'] == df['decile_score_again']).all() - True, drop 
    df_new = df_new.drop("decile_score", axis=1)
    # print(df_new.shape, df2.shape)
    # print(df_new.columns[df_new.isnull().any()])
    # import ipdb; ipdb.set_trace()
    df_new = df_new.fillna(df_new.mean().map(int))    # this fills the misisng values with averages of respective columns, after converting them to int
    assert len(df_new.columns[df_new.isnull().any()]) == 0
    # print("see")
    # Since the sensitive feature can only be binary in our setup, we only keep "African-American" and "Caucasian" people
    indexNames = df_new[(df_new['race'] == "Other") | (df_new['race'] == "Hispanic") | (df_new['race'] == "Asian") | (df_new['race'] == "Native American") ].index
    df_new.drop(indexNames , inplace=True)
    print(df_new.shape, df2.shape)      # (9884, 15) (11742, 47)
    # print(df_new.columns)
    
    # import ipdb; ipdb.set_trace()
    df_new.to_csv("missing_compas_removed.csv", index=False)


def missing_to_normalized():
    df = pd.read_csv("missing_compas_removed.csv")
    df['sex'] = df['sex'].replace({"Male":1, "Female":0})
    df['race'] = df['race'].replace({"Caucasian":1, "African-American":0})
    df['c_charge_degree'] = df['c_charge_degree'].replace({"O":0, "F":1, "M":-1})    # O : Ordinary crime, F: Felony, M: Misconduct
    df['r_charge_degree'] = df['r_charge_degree'].replace({"O":0, "F":1, "M":-1})    # O : Ordinary crime, F: Felony, M: Misconduct
    # now decide which columns to use as features and which column to use as label
    # decile_score is the label
    # Remove the features related to columns about recividism: is_recid,r_charge_degree,is_violent_recid,v_decile_score
    df = df.drop(columns=["is_recid","r_charge_degree","is_violent_recid","v_decile_score"])
    # print(df.shape)
    target = df['decile_score_again']
    df_new = df.drop(columns=['decile_score_again'])
    # df_new = df_new.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))       # This means nothing
    df_new = df_new.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))          # normlization: scales the data between 0 and 1
    df_new.to_csv("normalized_compas_features.csv", index=False, header=True)
    # We will leave target as it is
    # target = target.apply(lambda x: x/10)       # dividing by 9 results in weird numbers, np.min(x) = 1, so no huge difference by not dividing it
    # target = target.apply(lambda x: x - 1)      # making labels from [0 to 9], required by genericNeuralNet.py
    # target = target.apply(lambda x: 0 if x <= 4 else (1 if x <= 7 else 2))
    target = target.apply(lambda x: 0 if x <= 5 else 1)
    target.to_csv("compas_labels_binary.csv", index=False, header=True)
    # import ipdb; ipdb.set_trace()
    # print("see")


def print_mins_and_ranges():
    df = pd.read_csv("missing_compas_removed.csv")
    df['sex'] = df['sex'].replace({"Male":1, "Female":0})
    df['race'] = df['race'].replace({"Caucasian":1, "African-American":0})
    df['c_charge_degree'] = df['c_charge_degree'].replace({"O":0, "F":1, "M":-1})    # O : Ordinary crime, F: Felony, M: Misconduct
    df['r_charge_degree'] = df['r_charge_degree'].replace({"O":0, "F":1, "M":-1})    # O : Ordinary crime, F: Felony, M: Misconduct
    df = df.drop(columns=["is_recid","r_charge_degree","is_violent_recid","v_decile_score"])
    df_new = df.drop(columns=['decile_score_again'])
    means_and_ranges = []
    for j in list(df_new):
        i = df[j]
        means_and_ranges.append((np.min(i), np.max(i) - np.min(i)))
    print(means_and_ranges)


import sys
if __name__ == "__main__":
    cleaning_level = int(sys.argv[1])
    if cleaning_level == 0:
        raw_to_no_missing()
    elif cleaning_level == 1:
       missing_to_normalized() 
    else:
        print_mins_and_ranges()
