import pandas as pd
import numpy as np



def raw_to_no_missing():
    df = pd.read_csv("compas-scores-two-years.csv")
    
    # We will add two new features
    df['in_custody'] = pd.to_datetime(df['in_custody'])
    df['out_custody'] = pd.to_datetime(df['out_custody'])
    df['diff_custody'] = (df['out_custody'] - df['in_custody']).dt.total_seconds()
    df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
    df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])
    df['diff_jail'] = (df['c_jail_out'] - df['c_jail_in']).dt.total_seconds()
    assert(all(df['decile_score'] == df['decile_score_1']))
    assert(all(df['priors_count'] == df['priors_count_1']))
    # df.drop(
    #     [
    #         'id', 'name', 'first', 'last', 'v_screening_date', 'compas_screening_date', 'dob', 'c_case_number',
    #         'screening_date', 'in_custody', 'out_custody', 'c_jail_in', 'c_jail_out', 'priors_count_1', 'decile_score_1'
    #     ], axis=1, inplace=True
    # )

    df = df[df['race'].isin(['African-American', 'Caucasian'])]     # size reduces to 6150
    
    # We need to drop 'is_recid'. It is problematic: See https://arxiv.org/pdf/1906.04711.pdf
    # need to do separate analysis for c and v columns.
    # Score_text and v_score_text are derived from compas score - decile_score, can't be used a feature
    dataset = df[[
            'age', 'sex', 'race', 'diff_custody', 'diff_jail', 'priors_count', 'juv_fel_count', 
            'juv_misd_count', 'juv_other_count', 
            'c_charge_degree', 'decile_score'
        ]]
    
    dataset['decile_score'] = dataset['decile_score'].apply(lambda x: 0 if x <= 4 else 1)      # balanced dataset with 4, also consistent with Propublica
    dataset = dataset.rename(columns={"decile_score": "compas_score"})
    # print(dataset['decile_score'].value_counts())
    # print(dataset.columns[dataset.isnull().any()])
    dataset = dataset.fillna(dataset.mean())    # this fills the misisng values with averages of respective columns, after converting them to int
    assert len(dataset.columns[dataset.isnull().any()]) == 0
    dataset['diff_custody'] = dataset['diff_custody'].apply(lambda x: x/86400)       # divide by number of seconds in a day
    dataset['diff_jail'] = dataset['diff_jail'].apply(lambda x: x/86400)       # divide by number of seconds in a day
    dataset['diff_custody'] = dataset['diff_custody'].map(int)
    dataset['diff_jail']  = dataset['diff_jail'].map(int)
    
    assert list(dataset.race.unique()) == ['African-American', 'Caucasian']
    print(dataset.shape)
    dataset.to_csv("compas_score_as_label.csv", index=False)


def missing_to_normalized():
    df = pd.read_csv("compas_score_as_label.csv")
    assert len(df.columns[df.isnull().any()]) == 0
    df['sex'] = df['sex'].replace({"Male":1, "Female":0})
    df['race'] = df['race'].replace({"Caucasian":1, "African-American":0})
    df['c_charge_degree'] = df['c_charge_degree'].replace({"F":1, "M":0})    # O : Ordinary crime, F: Felony, M: Misconduct
    target = df['compas_score']
    df_new = df.drop(columns=['compas_score'])
    df_new = df_new.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))          # normlization: scales the data between 0 and 1
    df_new.to_csv("normalized_scores_as_labels_features.csv", index=False, header=True)
    # We will leave target as it is
    target.to_csv("target_compas_score_as_label.csv", index=False, header=True)


def print_mins_and_ranges():
    df = pd.read_csv("compas_score_as_label.csv")
    assert len(df.columns[df.isnull().any()]) == 0
    df['sex'] = df['sex'].replace({"Male":1, "Female":0})
    df['race'] = df['race'].replace({"Caucasian":1, "African-American":0})
    df['c_charge_degree'] = df['c_charge_degree'].replace({"F":1, "M":0})    # O : Ordinary crime, F: Felony, M: Misconduct
    df_new = df.drop(columns=['compas_score'])
    mins_and_ranges = []
    for j in list(df_new):
        i = df[j]
        mins_and_ranges.append((np.min(i), np.max(i) - np.min(i)))
    print(mins_and_ranges)




import sys
if __name__ == "__main__":
    cleaning_level = int(sys.argv[1])
    if cleaning_level == 1:
        raw_to_no_missing()
    elif cleaning_level == 2:
       missing_to_normalized() 
    elif cleaning_level == 3:
        print_mins_and_ranges()
        
