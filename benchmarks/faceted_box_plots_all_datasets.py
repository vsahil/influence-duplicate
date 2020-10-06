from mpl_toolkits import mplot3d
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from plotnine import *
import copy, math
dist = 10


def find_min_discm_each_hyperparam(df):
    x = df.sort_values(by=['Discm_percent', 'Points-Removed']).groupby("Model-count", as_index=False).first()
    assert len(x) == 240
    return x


def process_rows(row, batches):
    # global batches
    model_count = 0
    for perm in range(20):
        for h1units in [16, 24, 32]:
            for h2units in [8, 12]:
                for batch in batches:      # different batch sizes for this dataset
                    if perm == row['Dataperm'] and h1units == row['H1Units'] and h2units == row['H2Units'] and batch == row['Batch']:
                        return model_count
                    else:
                        model_count += 1


def process_dfs(name, batches, df):
    # import ipdb; ipdb.set_trace()
    if 'Model-count' in df.columns:
        df['Model-count2'] = df.apply(process_rows, axis=1, args=((batches,)))
        assert (df['Model-count'] == df['Model-count2']).all()
        df.drop(columns=['Model-count2'], inplace=True)
    else:
        df['Model-count'] = df.apply(process_rows, axis=1, args=((batches,)))
    assert len(df['Model-count'].unique()) == 240 and df['Model-count'].max() == 239 and df['Model-count'].min() == 0
    df = df.sort_values("Discm_percent").groupby("Model-count", as_index=False).first()     # must be sorted in order of model count for comparison across baselines
    # df = df[['Model-count','Discm_percent','Test_acc']]
    df = df[['Model-count','Discm_percent','Test_acc', 'Class0_Pos', 'Class1_Pos']]
    df['diff'] = abs(df['Class0_Pos'] - df['Class1_Pos']) * 100
    df['Test_acc'] = df['Test_acc'].apply(lambda x: x * 100)
    df['Techniques'] = name
    if len(name.split()) > 1:
        words = name.split()
        letters = [word[0] for word in words]
        x = "".join(letters)
        df['Baseline'] = x
    else:    
        df['Baseline'] = name[:2]
    return df


def boxplots_datasets(dataset, plot):
    df1 = pd.read_csv(f"{dataset}/results_{dataset}_method1.csv")
    batches = sorted(list(df1.Batch.unique()))      # sorting is important
    assert(len(batches) == 2)
    df_our = find_min_discm_each_hyperparam(df1)
    df_our = df_our[['Model-count','Discm_percent', 'Test_acc', 'Class0_Pos', 'Class1_Pos']]
    df_our['diff'] = abs(df_our['Class0_Pos'] - df_our['Class1_Pos'])*100
    df_our['Test_acc'] = df_our['Test_acc'].apply(lambda x: x*100)
    df_our['Techniques'] = "Our Technique"
    df_our['Baseline'] = "Our"

    # Massaging
    df_massaging = process_dfs("MAssaging", batches, pd.read_csv(f"{dataset}/massaging/results_massaged_{dataset}.csv"))
    # Preferential Sampling
    df_ps = process_dfs("Prefer. Sampling", batches, pd.read_csv(f"{dataset}/preferential_sampling/results_resampling_{dataset}.csv"))
    # Learning Fair representations
    df_lfr = process_dfs("Learning Fair Repr.", batches, pd.read_csv(f"{dataset}/learning_fair_representations/results_lfr_{dataset}.csv"))
    # Disparate Impact Removed
    df_DIR = process_dfs("Disp. Impact Rem", batches, pd.read_csv(f"{dataset}/disparate_impact_removed/results_disparate_removed_{dataset}.csv"))
    # Adversarial Sampling
    df_adver = pd.read_csv(f"{dataset}/adversarial_debiasing/results_adversarial_debiased_{dataset}.csv")
    df_adver['Model-count'] = df_adver['Dataperm']*12
    df_adver = df_adver.sort_values("Discm_percent").groupby("Model-count", as_index=False).first()     # must be sorted in order of model count for comparison across baselines
    df_adver = df_adver[['Model-count','Discm_percent','Test_acc','diff']]
    df_adver['diff'] = df_adver['diff']*100
    df_adver['Test_acc'] = df_adver['Test_acc'].apply(lambda x: x*100)
    df_adver['Techniques'] = "Adversa. debias"
    df_adver['Baseline'] = "AD"

    # # Sensitive Attribute removed, therefore no discrimination
    df_nosensitive = pd.read_csv(f"{dataset}/results_{dataset}_nosensitive.csv")
    df_nosensitive = df_nosensitive[['Model-count','Test_acc', 'Class0_Pos', 'Class1_Pos']]
    df_nosensitive['diff'] = abs(df_nosensitive['Class0_Pos'] - df_nosensitive['Class1_Pos'])*100
    df_nosensitive['Discm_percent'] = 0.0
    df_nosensitive['Test_acc'] = df_nosensitive['Test_acc'].apply(lambda x: x*100)
    df_nosensitive['Techniques'] = "Sens. Removed"
    df_nosensitive['Baseline'] = "SR"
    # df_nosensitive = process_dfs("Sensitive Removed", batches, pd.read_csv(f"{dataset}/results_{dataset}_nosensitive.csv"))

    # No technique used
    df_noremoval = process_dfs("FULL", batches, pd.read_csv(f"{dataset}/results_{dataset}_noremoval.csv"))
    
    df_main = pd.concat([df_noremoval, df_nosensitive, df_massaging, df_ps, df_lfr, df_DIR, df_adver, df_our])
    try:
        assert(len(df_main) == 7*240 + 20)
    except:
        import ipdb; ipdb.set_trace()
    

    if dataset == "compas-score":
        dataset = "Recidivism-score"
    elif dataset == "compas-ground":
        dataset = "Recidivism-ground"
    # df_main['Dataset'] = dataset.capitalize()

    if dataset == "adult":
        sizeofPSI = 4522200
        id_ = "D1"
    elif dataset == "adult_race":
        sizeofPSI = 4313100
        id_ = "D2"
    elif dataset == "german":
        sizeofPSI = 100000
        id_ = "D3"
    elif dataset == "student":
        sizeofPSI = 64900
        id_ = "D4"
    elif dataset == "Recidivism-ground":
        sizeofPSI = 615000
        id_ = "D5"
    elif dataset == "Recidivism-score":
        sizeofPSI = 615000
        id_ = "D6"
    elif dataset == "default":
        sizeofPSI = 3000000
        id_ = "D7"
    elif dataset == "salary":
        sizeofPSI = 5200
        id_ = "D8"
    else:
        raise NotImplementedError
    
    df_main['Dataset'] = id_
    precision = 1

    if plot == 0:
        min_discm = True
        test_accuracy_for_min_discm = True

        max_accuracy = True
        discm_for_max_accuracy = True

        median_discm = False
        mean_accuracy = False
        median_accuracy = False

        if min_discm:
            x = ' & '.join([f"{id_}", f"{df_noremoval['Discm_percent'].min():.{precision}e}", '0.0' ,f"{df_DIR['Discm_percent'].min():.{precision}e}", f"{df_ps['Discm_percent'].min():.{precision}e}", f"{df_massaging['Discm_percent'].min():.{precision}e}", f"{df_lfr['Discm_percent'].min():.{precision}e}", f"{df_adver['Discm_percent'].min():.{precision}e}", f"{df_our['Discm_percent'].min():.{precision}e}"])
            print_to_tex(x, 'min-discm.tex', dataset)

        if max_accuracy:
            y = ' & '.join([f"{id_}", f"{df_noremoval['Test_acc'].max():.{precision}e}", f"{df_nosensitive['Test_acc'].max():.{precision}e}", f"{df_DIR['Test_acc'].max():.{precision}e}", f"{df_ps['Test_acc'].max():.{precision}e}", f"{df_massaging['Test_acc'].max():.{precision}e}", f"{df_lfr['Test_acc'].max():.{precision}e}", f"{df_adver['Test_acc'].max():.{precision}e}", f"{df_our['Test_acc'].max():.{precision}e}"])
            print_to_tex(y, 'max-test-accuracy.tex', dataset)

        if test_accuracy_for_min_discm:
            # for sensitive there is always 0 discrimination. 
            z = ' & '.join([f"{id_}", f"{df_noremoval.loc[df_noremoval['Discm_percent'] == df_noremoval['Discm_percent'].min()]['Test_acc'].max():.{precision}e}", 
                    f"{df_nosensitive['Test_acc'].max():.{precision}e}", 
                    f"{df_DIR.loc[df_DIR['Discm_percent'] == df_DIR['Discm_percent'].min()]['Test_acc'].max():.{precision}e}", 
                    f"{df_ps.loc[df_ps['Discm_percent'] == df_ps['Discm_percent'].min()]['Test_acc'].max():.{precision}e}", 
                    f"{df_massaging.loc[df_massaging['Discm_percent'] == df_massaging['Discm_percent'].min()]['Test_acc'].max():.{precision}e}", 
                    f"{df_lfr.loc[df_lfr['Discm_percent'] == df_lfr['Discm_percent'].min()]['Test_acc'].max():.{precision}e}", 
                    f"{df_adver.loc[df_adver['Discm_percent'] == df_adver['Discm_percent'].min()]['Test_acc'].max():.{precision}e}", 
                    f"{df_our.loc[df_our['Discm_percent'] == df_our['Discm_percent'].min()]['Test_acc'].max():.{precision}e}"])
            
            print_to_tex(z, 'test_accuracy_for_min_discm.tex', dataset)

        if median_discm:
            x = ' & '.join([f"{id_}", f"{df_noremoval['Discm_percent'].median():.{precision}e}", "\\textbf{%s}"%(0.0) ,f"{df_DIR['Discm_percent'].median():.{precision}e}", f"{df_ps['Discm_percent'].median():.{precision}e}", f"{df_massaging['Discm_percent'].median():.{precision}e}", f"{df_lfr['Discm_percent'].median():.{precision}e}", f"{df_adver['Discm_percent'].median():.{precision}e}", "\\textbf{%s}"%(f"{df_our['Discm_percent'].median():.{precision}e}")])
            print_to_tex(x, 'median-discm.tex', dataset)

        if mean_accuracy:
            a = ' & '.join([f"{id_}", f"{df_noremoval['Test_acc'].mean():.{precision}e}", f"{df_nosensitive['Test_acc'].mean():.{precision}e}", f"{df_DIR['Test_acc'].mean():.{precision}e}", f"{df_ps['Test_acc'].mean():.{precision}e}", f"{df_massaging['Test_acc'].mean():.{precision}e}", f"{df_lfr['Test_acc'].mean():.{precision}e}", f"{df_adver['Test_acc'].mean():.{precision}e}", "\\textbf{%s}"%(f"{df_our['Test_acc'].mean():.{precision}e}")])
            print_to_tex(a, 'mean-test-accuracy.tex', dataset)

        if median_accuracy:
            b = ' & '.join([f"{id_}", f"{df_noremoval['Test_acc'].median():.{precision}e}", f"{df_nosensitive['Test_acc'].median():.{precision}e}", f"{df_DIR['Test_acc'].median():.{precision}e}", f"{df_ps['Test_acc'].median():.{precision}e}", f"{df_massaging['Test_acc'].median():.{precision}e}", f"{df_lfr['Test_acc'].median():.{precision}e}", f"{df_adver['Test_acc'].median():.{precision}e}", "\\textbf{%s}"%(f"{df_our['Test_acc'].median():.{precision}e}")])
            print_to_tex(b, 'median-test-accuracy.tex', dataset)

        if discm_for_max_accuracy:
            k = ' & '.join([f"{id_}", f"{df_noremoval.loc[df_noremoval['Test_acc'] == df_noremoval['Test_acc'].max()]['Discm_percent'].min():.{precision}e}", 
                    "0.0", 
                    f"{df_DIR.loc[df_DIR['Test_acc'] == df_DIR['Test_acc'].max()]['Discm_percent'].min():.{precision}e}", 
                    f"{df_ps.loc[df_ps['Test_acc'] == df_ps['Test_acc'].max()]['Discm_percent'].min():.{precision}e}", 
                    f"{df_massaging.loc[df_massaging['Test_acc'] == df_massaging['Test_acc'].max()]['Discm_percent'].min():.{precision}e}", 
                    f"{df_lfr.loc[df_lfr['Test_acc'] == df_lfr['Test_acc'].max()]['Discm_percent'].min():.{precision}e}", 
                    f"{df_adver.loc[df_adver['Test_acc'] == df_adver['Test_acc'].max()]['Discm_percent'].min():.{precision}e}", 
                    f"{df_our.loc[df_our['Test_acc'] == df_our['Test_acc'].max()]['Discm_percent'].min():.{precision}e}"])
            
            print_to_tex(k, 'discm_for_max_accuracy.tex', dataset)


    return df_main


def boxplots_datasets_dist(dataset, plot):
    df1 = pd.read_csv(f"{dataset}/results_{dataset}_method1_dist{dist}.csv")
    batches = sorted(list(df1.Batch.unique()))      # sorting is important
    assert(len(batches) == 2)
    df_our = find_min_discm_each_hyperparam(df1)
    df_our = df_our[['Model-count', 'Discm_percent', 'Test_acc', 'Class0_Pos', 'Class1_Pos']]
    df_our['diff'] = abs(df_our['Class0_Pos'] - df_our['Class1_Pos']) * 100     # Statistical parity diff
    df_our['Test_acc'] = df_our['Test_acc'].apply(lambda x: x * 100)
    df_our['Techniques'] = "Our Technique"
    df_our['Baseline'] = "Our"

    # Massaging
    df_massaging = process_dfs("MAssaging", batches, pd.read_csv(f"{dataset}/massaging/results_massaged_{dataset}_dist{dist}.csv"))
    # Preferential Sampling
    df_ps = process_dfs("Prefer. Sampling", batches, pd.read_csv(f"{dataset}/preferential_sampling/results_resampling_{dataset}_dist{dist}.csv"))
    # Learning Fair representations
    df_lfr = process_dfs("Learning Fair Repr.", batches, pd.read_csv(f"{dataset}/learning_fair_representations/results_lfr_{dataset}_dist{dist}.csv"))
    # Disparate Impact Removed
    df_DIR = process_dfs("Disp. Impact Rem", batches, pd.read_csv(f"{dataset}/disparate_impact_removed/results_disparate_removed_{dataset}_dist{dist}.csv"))
    # Adversarial Sampling
    df_adver = pd.read_csv(f"{dataset}/adversarial_debiasing/results_adversarial_debiased_{dataset}_dist{dist}.csv")
    df_adver['Model-count'] = df_adver['Dataperm'] * 12
    df_adver = df_adver.sort_values("Discm_percent").groupby("Model-count", as_index=False).first()     # must be sorted in order of model count for comparison across baselines
    df_adver = df_adver[['Model-count','Discm_percent','Test_acc','diff']]
    df_adver['diff'] = df_adver['diff'] * 100
    df_adver['Test_acc'] = df_adver['Test_acc'].apply(lambda x: x*100)
    df_adver['Techniques'] = "Adversa. debias"
    df_adver['Baseline'] = "AD"

    # # Sensitive Attribute removed, therefore no discrimination
    # df_nosensitive = pd.read_csv(f"{dataset}/results_{dataset}_nosensitive.csv")
    df_nosensitive = process_dfs("Sens. Removed", batches, pd.read_csv(f"{dataset}/results_{dataset}_nosensitive_dist{dist}.csv"))
    # df_nosensitive = df_nosensitive[['Model-count','Test_acc', 'Class0_Pos', 'Class1_Pos']]
    # df_nosensitive['diff'] = abs(df_nosensitive['Class0_Pos'] - df_nosensitive['Class1_Pos'])*100
    # df_nosensitive['Discm_percent'] = 0.0
    # df_nosensitive['Test_acc'] = df_nosensitive['Test_acc'].apply(lambda x: x*100)
    # df_nosensitive['Techniques'] = "Sens. Removed"
    # df_nosensitive['Baseline'] = "SR"

    # No technique used
    df_noremoval = process_dfs("FULL", batches, pd.read_csv(f"{dataset}/results_{dataset}_noremoval_dist{dist}.csv"))
    
    df_main = pd.concat([df_noremoval, df_nosensitive, df_massaging, df_ps, df_lfr, df_DIR, df_adver, df_our], sort=True)
    try:
        assert(len(df_main) == 7*240 + 20)
    except:
        import ipdb; ipdb.set_trace()

    if dataset == "compas-score":
        dataset = "Recidivism-score"
    elif dataset == "compas-ground":
        dataset = "Recidivism-ground"
    # df_main['Dataset'] = dataset.capitalize()

    if dataset == "adult":
        sizeofPSI = 4522200
        id_ = "D1"
    elif dataset == "adult_race":
        sizeofPSI = 4313100
        id_ = "D2"
    elif dataset == "german":
        sizeofPSI = 100000
        id_ = "D3"
    elif dataset == "student":
        sizeofPSI = 64900
        id_ = "D4"
    elif dataset == "Recidivism-ground":
        sizeofPSI = 615000
        id_ = "D5"
    elif dataset == "Recidivism-score":
        sizeofPSI = 615000
        id_ = "D6"
    elif dataset == "default":
        sizeofPSI = 3000000
        id_ = "D7"
    elif dataset == "salary":
        sizeofPSI = 5200
        id_ = "D8"
    else:
        raise NotImplementedError
    
    df_main['Dataset'] = id_
    precision = 1

    if plot == 0:
        min_discm = True
        test_accuracy_for_min_discm = True

        max_accuracy = True
        discm_for_max_accuracy = True

        median_discm = False
        mean_accuracy = False
        median_accuracy = False

        if min_discm:
            x = ' & '.join([f"{id_}", f"{df_noremoval['Discm_percent'].min():.{precision}e}", f"{df_nosensitive['Discm_percent'].min():.{precision}e}" ,f"{df_DIR['Discm_percent'].min():.{precision}e}", f"{df_ps['Discm_percent'].min():.{precision}e}", f"{df_massaging['Discm_percent'].min():.{precision}e}", f"{df_lfr['Discm_percent'].min():.{precision}e}", f"{df_adver['Discm_percent'].min():.{precision}e}", f"{df_our['Discm_percent'].min():.{precision}e}"])
            print_to_tex(x, f'min-discm_dist{dist}.tex', dataset)

        if max_accuracy:
            y = ' & '.join([f"{id_}", f"{df_noremoval['Test_acc'].max():.{precision}e}", f"{df_nosensitive['Test_acc'].max():.{precision}e}", f"{df_DIR['Test_acc'].max():.{precision}e}", f"{df_ps['Test_acc'].max():.{precision}e}", f"{df_massaging['Test_acc'].max():.{precision}e}", f"{df_lfr['Test_acc'].max():.{precision}e}", f"{df_adver['Test_acc'].max():.{precision}e}", f"{df_our['Test_acc'].max():.{precision}e}"])
            print_to_tex(y, f'max-test-accuracy_dist{dist}.tex', dataset)

        if test_accuracy_for_min_discm:
            z = ' & '.join([f"{id_}", f"{df_noremoval.loc[df_noremoval['Discm_percent'] == df_noremoval['Discm_percent'].min()]['Test_acc'].max():.{precision}e}", 
                    f"{df_nosensitive.loc[df_nosensitive['Discm_percent'] == df_nosensitive['Discm_percent'].min()]['Test_acc'].max():.{precision}e}",
                    f"{df_DIR.loc[df_DIR['Discm_percent'] == df_DIR['Discm_percent'].min()]['Test_acc'].max():.{precision}e}", 
                    f"{df_ps.loc[df_ps['Discm_percent'] == df_ps['Discm_percent'].min()]['Test_acc'].max():.{precision}e}", 
                    f"{df_massaging.loc[df_massaging['Discm_percent'] == df_massaging['Discm_percent'].min()]['Test_acc'].max():.{precision}e}", 
                    f"{df_lfr.loc[df_lfr['Discm_percent'] == df_lfr['Discm_percent'].min()]['Test_acc'].max():.{precision}e}", 
                    f"{df_adver.loc[df_adver['Discm_percent'] == df_adver['Discm_percent'].min()]['Test_acc'].max():.{precision}e}", 
                    f"{df_our.loc[df_our['Discm_percent'] == df_our['Discm_percent'].min()]['Test_acc'].max():.{precision}e}"])
            
            print_to_tex(z, f'test_accuracy_for_min_discm_dist{dist}.tex', dataset)

        if median_discm:
            raise NotImplementedError
            x = ' & '.join([f"{id_}", f"{df_noremoval['Discm_percent'].median():.{precision}e}", "\\textbf{%s}"%(0.0) ,f"{df_DIR['Discm_percent'].median():.{precision}e}", f"{df_ps['Discm_percent'].median():.{precision}e}", f"{df_massaging['Discm_percent'].median():.{precision}e}", f"{df_lfr['Discm_percent'].median():.{precision}e}", f"{df_adver['Discm_percent'].median():.{precision}e}", "\\textbf{%s}"%(f"{df_our['Discm_percent'].median():.{precision}e}")])
            print_to_tex(x, 'median-discm.tex', dataset)

        if mean_accuracy:
            raise NotImplementedError
            a = ' & '.join([f"{id_}", f"{df_noremoval['Test_acc'].mean():.{precision}e}", f"{df_nosensitive['Test_acc'].mean():.{precision}e}", f"{df_DIR['Test_acc'].mean():.{precision}e}", f"{df_ps['Test_acc'].mean():.{precision}e}", f"{df_massaging['Test_acc'].mean():.{precision}e}", f"{df_lfr['Test_acc'].mean():.{precision}e}", f"{df_adver['Test_acc'].mean():.{precision}e}", "\\textbf{%s}"%(f"{df_our['Test_acc'].mean():.{precision}e}")])
            print_to_tex(a, 'mean-test-accuracy.tex', dataset)

        if median_accuracy:
            raise NotImplementedError
            b = ' & '.join([f"{id_}", f"{df_noremoval['Test_acc'].median():.{precision}e}", f"{df_nosensitive['Test_acc'].median():.{precision}e}", f"{df_DIR['Test_acc'].median():.{precision}e}", f"{df_ps['Test_acc'].median():.{precision}e}", f"{df_massaging['Test_acc'].median():.{precision}e}", f"{df_lfr['Test_acc'].median():.{precision}e}", f"{df_adver['Test_acc'].median():.{precision}e}", "\\textbf{%s}"%(f"{df_our['Test_acc'].median():.{precision}e}")])
            print_to_tex(b, 'median-test-accuracy.tex', dataset)

        if discm_for_max_accuracy:
            k = ' & '.join([f"{id_}", f"{df_noremoval.loc[df_noremoval['Test_acc'] == df_noremoval['Test_acc'].max()]['Discm_percent'].min():.{precision}e}", 
                    f"{df_nosensitive.loc[df_nosensitive['Test_acc'] == df_nosensitive['Test_acc'].max()]['Discm_percent'].min():.{precision}e}", 
                    f"{df_DIR.loc[df_DIR['Test_acc'] == df_DIR['Test_acc'].max()]['Discm_percent'].min():.{precision}e}", 
                    f"{df_ps.loc[df_ps['Test_acc'] == df_ps['Test_acc'].max()]['Discm_percent'].min():.{precision}e}", 
                    f"{df_massaging.loc[df_massaging['Test_acc'] == df_massaging['Test_acc'].max()]['Discm_percent'].min():.{precision}e}", 
                    f"{df_lfr.loc[df_lfr['Test_acc'] == df_lfr['Test_acc'].max()]['Discm_percent'].min():.{precision}e}", 
                    f"{df_adver.loc[df_adver['Test_acc'] == df_adver['Test_acc'].max()]['Discm_percent'].min():.{precision}e}", 
                    f"{df_our.loc[df_our['Test_acc'] == df_our['Test_acc'].max()]['Discm_percent'].min():.{precision}e}"])
            
            print_to_tex(k, f'discm_for_max_accuracy_dist{dist}.tex', dataset)


    return df_main


def print_to_tex(string, file, dataset, mode=None):
    if mode == None:
        if dataset == "adult":
            mode = "w"
        else:
            mode = "a"
    # with open(f"../../neurips_fairness_paper/tables/{file}", mode) as f:
    with open(f"tables/{file}", mode) as f:    
        if dataset == "salary":
            string += "  \\\  \midrule"
        else:
            string += "  \\\\  "
        print(string, file=f)
    
    # print(dataset)
    # print("Min discm: ", df_DIR['Discm_percent'].min())
    # print("Min discm: ", df_ps['Discm_percent'].min())
    # print("Min discm: ", df_massaging['Discm_percent'].min())
    # print("Min discm: ", df_lfr['Discm_percent'].min())
    # print("Min discm: ", df_adver['Discm_percent'].min())
    # print("Min discm: ", df_our['Discm_percent'].min())


def main(plot):
    df_main = None
    benchmarks = ["adult", "adult_race", "german", "student", "compas-ground", "compas-score", "default", "salary"]

    for dataset in benchmarks:
        # df_onedataset = boxplots_datasets(dataset, plot)
        df_onedataset = boxplots_datasets_dist(dataset, plot)
        if not df_main is None:
            df_main = pd.concat([df_main, df_onedataset])
        else:
            df_main = copy.deepcopy(df_onedataset)
        print(f"Done {dataset}")

    if plot == 0:
        return 

    labels = ['FU', 'SR', 'DIR', 'PS', 'MA', 'LFR', 'AD', 'Our']

    tech_cat = pd.Categorical(df_main['Baseline'], categories=labels)    
    df_main = df_main.assign(Technique_x = tech_cat)
    
    dataset_order = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"]
    data_cat = pd.Categorical(df_main['Dataset'], categories=dataset_order)    
    df_main = df_main.assign(Dataset_x = data_cat)

    # x = (ggplot(aes(x='Technique_x', y='Discm_percent', color='Techniques'), data=df_main) +\
    #         geom_boxplot() +\
    #         facet_wrap(['Dataset'], scales = 'free', nrow=2, labeller='label_both', shrink=False) + \
    #         ylab("Remaining Individual Discrimination") + \
    #         xlab("Discrimination reducing techniques") + \
    #         # ylim(0, 20) + \
    #         # ggtitle("Box plot showing remaining discrimination for each technique in each dataset") +\
    #         theme(axis_text_x = element_text(size=6), dpi=151) + \
    #         theme_seaborn()
    #         )

    # This is responsible for the legend - remove color='Techniques'

    x = (ggplot(aes(x='Technique_x', y='Discm_percent'), data=df_main) +\
            geom_boxplot() +\
            facet_wrap(['Dataset_x'], scales = 'free', nrow=2, labeller='label_value', shrink=True) + \
            ylab("Remaining Individual Discrimination") + \
            xlab("Discrimination reducing techniques") + \
            # ylim(0, 20) + \
            # ggtitle("Box plot showing remaining discrimination for each technique in each dataset") +\
            theme(axis_text_x = element_text(size=6), dpi=151) + \
            theme_seaborn()
            )
    x = x.draw()

    x.set_figwidth(20)
    x.set_figheight(12)
    for ax in range(len(benchmarks)):
        low_limit = -0.05
        top_limit = df_main[df_main['Dataset'] == f'D{ax+1}']['Discm_percent'].max()
        if df_main[df_main['Dataset'] == f'D{ax+1}']['Discm_percent'].max() > 20:
            top_limit = 20
        if top_limit > 13:      # These hacks are for aligning the 0 at the bottom of the plots. 
            low_limit = -0.3
        x.axes[ax].set_ylim(low_limit, top_limit)
    # x.tight_layout()      # This didn't work
    x.savefig(f"boxplots/boxplot_discm_freeaxis_matplotlib_dist{dist}.eps", format='eps', bbox_inches='tight')
    x.savefig(f"boxplots/boxplot_discm_freeaxis_matplotlib_dist{dist}.png", bbox_inches='tight')
    
    # x.save(f"boxplot_discm_freeaxis_matplotlib.png", height=8, width=18)
    # x.save(f"boxplot_discm_freeaxis_withoutfull.png", height=12, width=15)
    # x.save(f"boxplot_discm_fixedaxis.png", height=5, width=12)

    y = (ggplot(aes(x='Technique_x', y='Test_acc'), data=df_main) +\
            geom_boxplot() +\
            facet_wrap(['Dataset_x'], scales = 'free', nrow=2, labeller='label_value', shrink=True) + \
            ylab("Test Accuracy") + \
            xlab("Discrimination reducing techniques") + \
            # ylim(0, 100) + \
            # ggtitle("Box plot showing remaining discrimination for each technique in each dataset") +\
            theme(axis_text_x = element_text(size=6), dpi=151) + \
            theme_seaborn()
            )

    # y.save(f"boxplot_accuracy_freeaxis.png", height=8, width=18)
    y = y.draw()
    y.set_figwidth(20)
    y.set_figheight(12)
    for ax in range(len(benchmarks)):
        bot_limit = df_main[df_main['Dataset'] == f'D{ax+1}']['Test_acc'].min()
        top_limit = df_main[df_main['Dataset'] == f'D{ax+1}']['Test_acc'].max()
        y.axes[ax].set_ylim(bot_limit - 1, top_limit + 2)
    # y.tight_layout()
    y.savefig(f"boxplots/boxplot_accuracy_freeaxis_matplotlib_dist{dist}.eps", format='eps', bbox_inches='tight')
    y.savefig(f"boxplots/boxplot_accuracy_freeaxis_matplotlib_dist{dist}.png", bbox_inches='tight')


def real_accuracy_tables(debiased):
    dataset = "compas-score"
    if debiased:
        deb = "debiased"
    else:
        deb = "full"

    df1 = pd.read_csv(f"{dataset}/results_{dataset}_method1.csv")
    batches = sorted(list(df1.Batch.unique()))
    assert(len(batches) == 2)
    df_our = find_min_discm_each_hyperparam(df1)
    df_our = df_our[['Model-count','Discm_percent']]
    df_our_2 = pd.read_csv(f"{dataset}/results_our_real_accuracy_{deb}.csv")
    df_our_final = pd.merge(df_our, df_our_2, on=['Model-count'])
    df_our_final['Test_acc'] = df_our_final['Test_acc'].apply(lambda x: x*100)
    df_our_final['Techniques'] = "Our Technique"
    df_our_final['Baseline'] = "Our"
    # import ipdb; ipdb.set_trace()
    
    # Massaging
    df_massaging = process_dfs("MAssaging", batches, pd.read_csv(f"{dataset}/massaging/results_massaged_{dataset}.csv"))
    df_massaging.drop(columns=['Test_acc'], inplace=True)
    df_massaging_2 = pd.read_csv(f"{dataset}/massaging/results_massaged_{dataset}_real_accuracy_{deb}.csv")
    df_massaging_final = pd.merge(df_massaging, df_massaging_2, on=['Model-count'])
    df_massaging_final['Test_acc'] = df_massaging_final['Test_acc'].apply(lambda x: x*100)
    
    # Preferential Sampling
    df_ps = process_dfs("Prefer. Sampling", batches, pd.read_csv(f"{dataset}/preferential_sampling/results_resampling_{dataset}.csv"))
    df_ps.drop(columns=['Test_acc'], inplace=True)
    df_ps_2 = pd.read_csv(f"{dataset}/preferential_sampling/results_resampling_{dataset}_real_accuracy_{deb}.csv")
    df_ps_final = pd.merge(df_ps, df_ps_2, on=['Model-count'])
    df_ps_final['Test_acc'] = df_ps_final['Test_acc'].apply(lambda x: x*100)

    # Learning Fair representations
    df_lfr = process_dfs("Learning Fair Repr.", batches, pd.read_csv(f"{dataset}/learning_fair_representations/results_lfr_{dataset}.csv"))
    df_lfr.drop(columns=['Test_acc'], inplace=True)
    df_lfr_2 = pd.read_csv(f"{dataset}/learning_fair_representations/results_lfr_{dataset}_real_accuracy_{deb}.csv")
    df_lfr_final = pd.merge(df_lfr, df_lfr_2, on=['Model-count'])
    df_lfr_final['Test_acc'] = df_lfr_final['Test_acc'].apply(lambda x: x*100)

    # Disparate Impact Removed
    df_DIR = process_dfs("Disp. Impact Rem", batches, pd.read_csv(f"{dataset}/disparate_impact_removed/results_disparate_removed_{dataset}.csv"))
    df_DIR.drop(columns=['Test_acc'], inplace=True)
    df_DIR_2 = pd.read_csv(f"{dataset}/disparate_impact_removed/results_disparate_removed_{dataset}_real_accuracy_{deb}.csv")
    df_DIR_final = pd.merge(df_DIR, df_DIR_2, on=['Model-count'])
    df_DIR_final['Test_acc'] = df_DIR_final['Test_acc'].apply(lambda x: x*100)

    # Adversarial Sampling
    df_adver = pd.read_csv(f"{dataset}/adversarial_debiasing/results_adversarial_debiased_{dataset}.csv")
    df_adver['Model-count'] = df_adver['Dataperm']*12
    df_adver = df_adver.sort_values("Discm_percent").groupby("Model-count", as_index=False).first()     # must be sorted in order of model count for comparison across baselines
    df_adver = df_adver[['Model-count','Discm_percent']]
    df_adver_2 = pd.read_csv(f"{dataset}/adversarial_debiasing/results_adversarial_debiased_{dataset}_real_accuracy_{deb}.csv")
    df_adver_2['Model-count'] = df_adver_2['Dataperm']*12
    df_adver_final = pd.merge(df_adver, df_adver_2, on=['Model-count'])
    df_adver_final['Test_acc'] = df_adver_final['Test_acc'].apply(lambda x: x*100)
    df_adver_final['Techniques'] = "Adversa. debias"
    df_adver_final['Baseline'] = "AD"

    # # Sensitive Attribute removed, therefore no discrimination
    df_nosensitive = pd.read_csv(f"{dataset}/results_nosensitive_real_accuracy_{deb}.csv")
    df_nosensitive = df_nosensitive[['Model-count','Test_acc']]
    df_nosensitive['Test_acc'] = df_nosensitive['Test_acc'].apply(lambda x: x*100)
    df_nosensitive['Techniques'] = "Sens. Removed"
    df_nosensitive['Baseline'] = "SR"
    # df_nosensitive = process_dfs("Sensitive Removed", batches, pd.read_csv(f"{dataset}/results_{dataset}_nosensitive.csv"))

    # No technique used
    df_noremoval = process_dfs("FULL", batches, pd.read_csv(f"{dataset}/results_{dataset}_noremoval.csv"))
    df_noremoval.drop(columns=['Test_acc'], inplace=True)
    df_noremoval_2 = pd.read_csv(f"{dataset}/results_noremoval_real_accuracy_{deb}.csv")
    df_noremoval_final = pd.merge(df_noremoval, df_noremoval_2, on=['Model-count'])
    df_noremoval_final['Test_acc'] = df_noremoval_final['Test_acc'].apply(lambda x: x*100)

    max_accuracy = True
    corresponding_max_accuracy = True
    mean_accuracy = False
    median_accuracy = False

    id_ = "D5"
    precision = 1
    if corresponding_max_accuracy:
        # for sensitive there is always 0 discrimination. 
        z = ' & '.join([f"{id_}", f"{df_noremoval_final.loc[df_noremoval_final['Discm_percent'] == df_noremoval_final['Discm_percent'].min()]['Test_acc'].max():.{precision}e}", 
                f"{df_nosensitive['Test_acc'].max():.{precision}e}", 
                f"{df_DIR_final.loc[df_DIR_final['Discm_percent'] == df_DIR_final['Discm_percent'].min()]['Test_acc'].max():.{precision}e}", 
                f"{df_ps_final.loc[df_ps_final['Discm_percent'] == df_ps_final['Discm_percent'].min()]['Test_acc'].max():.{precision}e}", 
                f"{df_massaging_final.loc[df_massaging_final['Discm_percent'] == df_massaging_final['Discm_percent'].min()]['Test_acc'].max():.{precision}e}", 
                f"{df_lfr_final.loc[df_lfr_final['Discm_percent'] == df_lfr_final['Discm_percent'].min()]['Test_acc'].max():.{precision}e}", 
                f"{df_adver_final.loc[df_adver_final['Discm_percent'] == df_adver_final['Discm_percent'].min()]['Test_acc'].max():.{precision}e}", 
                f"{df_our_final.loc[df_our_final['Discm_percent'] == df_our_final['Discm_percent'].min()]['Test_acc'].max():.{precision}e}"])

        a = ' & '.join([f"{id_}", f"{df_noremoval_final['Discm_percent'].min():.{precision}e}", 
                "0.0", 
                f"{df_DIR_final['Discm_percent'].min():.{precision}e}", 
                f"{df_ps_final['Discm_percent'].min():.{precision}e}", 
                f"{df_massaging_final['Discm_percent'].min():.{precision}e}", 
                f"{df_lfr_final['Discm_percent'].min():.{precision}e}", 
                f"{df_adver_final['Discm_percent'].min():.{precision}e}", 
                f"{df_our_final['Discm_percent'].min():.{precision}e}"])
        
        print_to_tex(z, f'correspond-real-accuracy_{deb}.tex', dataset, "w")
        print_to_tex(a, f'correspond-real-accuracy_{deb}.tex', dataset, "a")
        
    if max_accuracy:
            y = ' & '.join([f"{id_}", f"{df_noremoval_final['Test_acc'].max():.{precision}e}", f"{df_nosensitive['Test_acc'].max():.{precision}e}", f"{df_DIR_final['Test_acc'].max():.{precision}e}", f"{df_ps_final['Test_acc'].max():.{precision}e}", f"{df_massaging_final['Test_acc'].max():.{precision}e}", f"{df_lfr_final['Test_acc'].max():.{precision}e}", f"{df_adver_final['Test_acc'].max():.{precision}e}", f"{df_our_final['Test_acc'].max():.{precision}e}"])
            print_to_tex(y, f'max-real-accuracy_{deb}.tex', dataset, "w")

    print("Done real accuracy")


def fpr_fnr_process_dfs(name, batches, df):
    if 'Model-count'in df.columns:
        df['Model-count2'] = df.apply(process_rows, axis=1, args=((batches,)))
        assert (df['Model-count'] == df['Model-count2']).all()
        df.drop(columns=['Model-count2'], inplace=True)
    else:
        df['Model-count'] = df.apply(process_rows, axis=1, args=((batches,)))
    assert len(df['Model-count'].unique()) == 240 and df['Model-count'].max() == 239 and df['Model-count'].min() == 0
    df = df.sort_values("Discm_percent").groupby("Model-count", as_index=False).first()     # must be sorted in order of model count for comparison across baselines
    df = df[['Model-count','Discm_percent', 'Test_acc', 'Class0_FPR', 'Class0_FNR', 'Class1_FPR', 'Class1_FNR']]
    df['FPR_diff'] = abs(df['Class0_FPR'] - df['Class1_FPR'])*100
    # df_our['FPR_sum'] = df_our['Class0_FPR'] + df_our['Class1_FPR']
    df['FPR_ratio'] = df['Class0_FPR'] / df['Class1_FPR']
    df['FNR_diff'] = abs(df['Class0_FNR'] - df['Class1_FNR'])*100
    # df_our['FNR_sum'] = df_our['Class0_FNR'] + df_our['Class1_FNR']
    df['FNR_ratio'] = df['Class0_FNR'] / df['Class1_FNR']
    # df['diff'] = abs(df['Class0_Pos'] - df['Class1_Pos'])*100
    df['Test_acc'] = df['Test_acc'].apply(lambda x: x*100)
    df['Techniques'] = name
    return df


def fpr_fnr_rates():
    
    def fpr_fnr_print(dataset, id_, kind):
        if kind:
            df1 = pd.read_csv(f"{dataset}/results_{dataset}_method1.csv")
        else:
            df1 = pd.read_csv(f"{dataset}/results_{dataset}_method1_fulltest.csv")
        batches = sorted(list(df1.Batch.unique()))
        assert(len(batches) == 2)
        # import ipdb; ipdb.set_trace()
        df_our = find_min_discm_each_hyperparam(df1)
        df_our = df_our[['Model-count','Discm_percent', 'Test_acc', 'Class0_FPR', 'Class0_FNR', 'Class1_FPR', 'Class1_FNR']]
        df_our['FPR_diff'] = abs(df_our['Class0_FPR'] - df_our['Class1_FPR'])*100
        # df_our['FPR_sum'] = df_our['Class0_FPR'] + df_our['Class1_FPR']
        df_our['FPR_ratio'] = df_our['Class0_FPR'] / df_our['Class1_FPR']
        df_our['FNR_diff'] = abs(df_our['Class0_FNR'] - df_our['Class1_FNR'])*100
        # df_our['FNR_sum'] = df_our['Class0_FNR'] + df_our['Class1_FNR']
        df_our['FNR_ratio'] = df_our['Class0_FNR'] / df_our['Class1_FNR']
        df_our['Techniques'] = "Our Technique"
        df_our['Baseline'] = "Our"

        if kind:
            df_massaging = fpr_fnr_process_dfs("MAssaging", batches, pd.read_csv(f"{dataset}/massaging/results_massaged_{dataset}_fulltest.csv"))
        else:
            df_massaging = fpr_fnr_process_dfs("MAssaging", batches, pd.read_csv(f"{dataset}/massaging/results_massaged_{dataset}.csv"))
        
        # Preferential Sampling
        if kind:
            df_ps = fpr_fnr_process_dfs("Prefer. Sampling", batches, pd.read_csv(f"{dataset}/preferential_sampling/results_resampling_{dataset}_fulltest.csv"))
        else:
            df_ps = fpr_fnr_process_dfs("Prefer. Sampling", batches, pd.read_csv(f"{dataset}/preferential_sampling/results_resampling_{dataset}.csv"))
        
        # Learning Fair representations
        if kind:
            df_lfr = fpr_fnr_process_dfs("Learning Fair Repr.", batches, pd.read_csv(f"{dataset}/learning_fair_representations/results_lfr_{dataset}_fulltest.csv"))
        else:
            df_lfr = fpr_fnr_process_dfs("Learning Fair Repr.", batches, pd.read_csv(f"{dataset}/learning_fair_representations/results_lfr_{dataset}.csv"))
        
        # Disparate Impact Removed
        if kind:
            df_DIR = fpr_fnr_process_dfs("Disp. Impact Rem", batches, pd.read_csv(f"{dataset}/disparate_impact_removed/results_disparate_removed_{dataset}_fulltest.csv"))
        else:
            df_DIR = fpr_fnr_process_dfs("Disp. Impact Rem", batches, pd.read_csv(f"{dataset}/disparate_impact_removed/results_disparate_removed_{dataset}.csv"))
        
        # Adversarial Sampling
        if kind:
            df_adver = pd.read_csv(f"{dataset}/adversarial_debiasing/results_adversarial_debiased_{dataset}_fulltest.csv")
        else:
            df_adver = pd.read_csv(f"{dataset}/adversarial_debiasing/results_adversarial_debiased_{dataset}.csv")
        df_adver['Model-count'] = df_adver['Dataperm']*12
        df_adver = df_adver.sort_values("Discm_percent").groupby("Model-count", as_index=False).first()     # must be sorted in order of model count for comparison across baselines
        # df_adver = df_adver[['Model-count','Discm_percent', 'Test_acc', 'Class0_FPR', 'Class0_FNR', 'Class1_FPR', 'Class1_FNR']]
        # df_adver['FPR_diff'] = abs(df_adver['Class0_FPR'] - df_adver['Class1_FPR'])*100
        # # df_our['FPR_sum'] = df_our['Class0_FPR'] + df_our['Class1_FPR']
        # df_adver['FPR_ratio'] = df_adver['Class0_FPR'] / df_adver['Class1_FPR']
        # df_adver['FNR_diff'] = abs(df_adver['Class0_FNR'] - df_adver['Class1_FNR'])*100
        # # df_our['FNR_sum'] = df_our['Class0_FNR'] + df_our['Class1_FNR']
        # df_adver['FNR_ratio'] = df_adver['Class0_FNR'] / df_adver['Class1_FNR']
        
        df_adver['FPR_diff'] = df_adver['FPR_ratio'] = df_adver['FNR_diff'] = df_adver['FNR_ratio'] = 1000.0 
        # df_adver['diff'] = df_adver['diff']*100
        df_adver['Test_acc'] = df_adver['Test_acc'].apply(lambda x: x*100)
        df_adver['Techniques'] = "Adversa. debias"
        df_adver['Baseline'] = "AD"


        df_nosensitive = pd.read_csv(f"{dataset}/results_{dataset}_nosensitive.csv")
        df_nosensitive = df_nosensitive[['Model-count', 'Test_acc', 'Class0_FPR', 'Class0_FNR', 'Class1_FPR', 'Class1_FNR']]
        df_nosensitive['FPR_diff'] = abs(df_nosensitive['Class0_FPR'] - df_nosensitive['Class1_FPR'])*100
        # df_nosensitive['FPR_sum'] = df_nosensitive['Class0_FPR'] + df_nosensitive['Class1_FPR']
        df_nosensitive['FPR_ratio'] = df_nosensitive['Class0_FPR'] / df_nosensitive['Class1_FPR']
        df_nosensitive['FNR_diff'] = abs(df_nosensitive['Class0_FNR'] - df_nosensitive['Class1_FNR'])*100
        # df_nosensitive['FNR_sum'] = df_nosensitive['Class0_FNR'] + df_nosensitive['Class1_FNR']
        df_nosensitive['FNR_ratio'] = df_nosensitive['Class0_FNR'] / df_nosensitive['Class1_FNR']
        df_nosensitive['Techniques'] = "Sens. Removed"
        df_nosensitive['Baseline'] = "SR"

        if kind:
            df_noremoval = fpr_fnr_process_dfs("FULL", batches, pd.read_csv(f"{dataset}/results_{dataset}_noremoval_fulltest.csv"))
        else:
            df_noremoval = fpr_fnr_process_dfs("FULL", batches, pd.read_csv(f"{dataset}/results_{dataset}_noremoval.csv"))

        min_rate_difference = True
        rate_difference_for_min_discm = True
        rate_difference_for_max_accuracy = True
        precision = 1

        if min_rate_difference:
            a = ' & '.join([f"{id_}",
            str(float(f"{df_noremoval['FPR_diff'].min():.{precision}e}")),
            str(float(f"{df_nosensitive['FPR_diff'].min():.{precision}e}")),
            str(float(f"{df_DIR['FPR_diff'].min():.{precision}e}")),
            str(float(f"{df_ps['FPR_diff'].min():.{precision}e}")),
            str(float(f"{df_massaging['FPR_diff'].min():.{precision}e}")),
            str(float(f"{df_lfr['FPR_diff'].min():.{precision}e}")),
            str(float(f"{df_adver['FPR_diff'].min():.{precision}e}")),
            str(float(f"{df_our['FPR_diff'].min():.{precision}e}"))])

            b = ' & '.join([f"{id_}",
            str(float(f"{df_noremoval['FNR_diff'].min():.{precision}e}")),
            str(float(f"{df_nosensitive['FNR_diff'].min():.{precision}e}")),
            str(float(f"{df_DIR['FNR_diff'].min():.{precision}e}")),
            str(float(f"{df_ps['FNR_diff'].min():.{precision}e}")),
            str(float(f"{df_massaging['FNR_diff'].min():.{precision}e}")),
            str(float(f"{df_lfr['FNR_diff'].min():.{precision}e}")),
            str(float(f"{df_adver['FNR_diff'].min():.{precision}e}")),
            str(float(f"{df_our['FNR_diff'].min():.{precision}e}"))])
            
            # b = ' & '.join([f"{id_}", f"{df_nosensitive['FNR_diff'].min():.{precision}e}", "\\textbf{%s}"%(f"{df_our['FNR_diff'].min():.{precision}e}")])
            
            # c = ' & '.join([f"{id_}", f"{df_nosensitive['FPR_ratio'].min():.{precision}e}", "\\textbf{%s}"%(f"{df_our['FPR_ratio'].min():.{precision}e}")])
            # d = ' & '.join([f"{id_}", f"{df_nosensitive['FNR_ratio'].min():.{precision}e}", "\\textbf{%s}"%(f"{df_our['FNR_ratio'].min():.{precision}e}")])

            # e = ' & '.join([f"{id_}", f"{df_nosensitive['Class0_FPR'].min():.{precision}e}", f"{df_nosensitive['Class1_FPR'].min():.{precision}e}", f"{df_our['Class0_FNR'].min():.{precision}e}", f"{df_our['Class1_FNR'].min():.{precision}e}"])
            if kind:
                print_to_tex(a, 'min-fpr_rate_fulltest.tex', dataset)
                print_to_tex(b, 'min-fnr_rate_fulltest.tex', dataset)
            else:
                print_to_tex(a, 'min-fpr_rate_debiasedtest.tex', dataset)
                print_to_tex(b, 'min-fnr_rate_debiasedtest.tex', dataset)


        if rate_difference_for_min_discm:
            x = ' & '.join([f"{id_}", 
                    str(float(f"{df_noremoval.loc[df_noremoval['Discm_percent'] == df_noremoval['Discm_percent'].min()]['FPR_diff'].min():.{precision}e}")),
                    str(float(f"{df_nosensitive.loc[df_nosensitive['Test_acc'] == df_nosensitive['Test_acc'].max()]['FPR_diff'].min():.{precision}e}")),
                    str(float(f"{df_DIR.loc[df_DIR['Discm_percent'] == df_DIR['Discm_percent'].min()]['FPR_diff'].min():.{precision}e}")),
                    str(float(f"{df_ps.loc[df_ps['Discm_percent'] == df_ps['Discm_percent'].min()]['FPR_diff'].min():.{precision}e}")),
                    str(float(f"{df_massaging.loc[df_massaging['Discm_percent'] == df_massaging['Discm_percent'].min()]['FPR_diff'].min():.{precision}e}")),
                    str(float(f"{df_lfr.loc[df_lfr['Discm_percent'] == df_lfr['Discm_percent'].min()]['FPR_diff'].min():.{precision}e}")),
                    str(float(f"{df_adver.loc[df_adver['Discm_percent'] == df_adver['Discm_percent'].min()]['FPR_diff'].min():.{precision}e}")),
                    str(float(f"{df_our.loc[df_our['Discm_percent'] == df_our['Discm_percent'].min()]['FPR_diff'].min():.{precision}e}"))])
            
            y = ' & '.join([f"{id_}", 
                    str(float(f"{df_noremoval.loc[df_noremoval['Discm_percent'] == df_noremoval['Discm_percent'].min()]['FNR_diff'].min():.{precision}e}")),
                    str(float(f"{df_nosensitive.loc[df_nosensitive['Test_acc'] == df_nosensitive['Test_acc'].max()]['FNR_diff'].min():.{precision}e}")),
                    str(float(f"{df_DIR.loc[df_DIR['Discm_percent'] == df_DIR['Discm_percent'].min()]['FNR_diff'].min():.{precision}e}")),
                    str(float(f"{df_ps.loc[df_ps['Discm_percent'] == df_ps['Discm_percent'].min()]['FNR_diff'].min():.{precision}e}")),
                    str(float(f"{df_massaging.loc[df_massaging['Discm_percent'] == df_massaging['Discm_percent'].min()]['FNR_diff'].min():.{precision}e}")),
                    str(float(f"{df_lfr.loc[df_lfr['Discm_percent'] == df_lfr['Discm_percent'].min()]['FNR_diff'].min():.{precision}e}")),
                    str(float(f"{df_adver.loc[df_adver['Discm_percent'] == df_adver['Discm_percent'].min()]['FNR_diff'].min():.{precision}e}")),
                    str(float(f"{df_our.loc[df_our['Discm_percent'] == df_our['Discm_percent'].min()]['FNR_diff'].min():.{precision}e}"))])

            # l = ' & '.join([f"{id_}", 
            #         f"{df_nosensitive.loc[df_nosensitive['Test_acc'] == df_nosensitive['Test_acc'].max()]['FPR_sum'].min():.{precision}e}", 
            #         f"{df_our.loc[df_our['Discm_percent'] == df_our['Discm_percent'].min()]['FPR_sum'].min():.{precision}e}"])
            
            # m = ' & '.join([f"{id_}", 
            #         f"{df_nosensitive.loc[df_nosensitive['Test_acc'] == df_nosensitive['Test_acc'].max()]['FNR_sum'].min():.{precision}e}", 
            #         f"{df_our.loc[df_our['Discm_percent'] == df_our['Discm_percent'].min()]['FNR_sum'].min():.{precision}e}"])

            # q = ' & '.join([f"{id_}", 
            #         f"{df_nosensitive.loc[df_nosensitive['Test_acc'] == df_nosensitive['Test_acc'].max()]['FPR_ratio'].min():.{precision}e}", 
            #         f"{df_our.loc[df_our['Discm_percent'] == df_our['Discm_percent'].min()]['FPR_ratio'].min():.{precision}e}"])
            
            # r = ' & '.join([f"{id_}", 
                    # f"{df_nosensitive.loc[df_nosensitive['Test_acc'] == df_nosensitive['Test_acc'].max()]['FNR_ratio'].min():.{precision}e}", 
                    # f"{df_our.loc[df_our['Discm_percent'] == df_our['Discm_percent'].min()]['FNR_ratio'].min():.{precision}e}"])


            # z = ' & '.join([f"{id_}", 
            #         f"{df_nosensitive.loc[df_nosensitive['Test_acc'] == df_nosensitive['Test_acc'].max()]['FPR_ratio'].min():.{precision}e}", 
            #         f"{df_our.loc[df_our['Discm_percent'] == df_our['Discm_percent'].min()]['FPR_ratio'].min():.{precision}e}"])
            
            # z1 = ' & '.join([f"{id_}", 
            #         f"{df_nosensitive.loc[df_nosensitive['Test_acc'] == df_nosensitive['Test_acc'].max()]['FNR_ratio'].min():.{precision}e}", 
            #         f"{df_our.loc[df_our['Discm_percent'] == df_our['Discm_percent'].min()]['FNR_ratio'].min():.{precision}e}"])
            

            # z2 = ' & '.join([f"{id_}", 
            #         f"{df_nosensitive.loc[df_nosensitive['Test_acc'] == df_nosensitive['Test_acc'].max()]['Class0_FPR'].min():.{precision}e}", 
            #         f"{df_nosensitive.loc[df_nosensitive['Test_acc'] == df_nosensitive['Test_acc'].max()]['Class1_FPR'].min():.{precision}e}", 
            #         f"{df_our.loc[df_our['Discm_percent'] == df_our['Discm_percent'].min()]['Class0_FNR'].min():.{precision}e}",
            #         f"{df_our.loc[df_our['Discm_percent'] == df_our['Discm_percent'].min()]['Class1_FNR'].min():.{precision}e}"])
            
            if kind:
                print_to_tex(x, 'fpr_rate-min-discm_fulltest.tex', dataset)
                print_to_tex(y, 'fnr_rate-min-discm_fulltest.tex', dataset)
            else:
                print_to_tex(x, 'fpr_rate-min-discm_debiasedtest.tex', dataset)
                print_to_tex(y, 'fnr_rate-min-discm_debiasedtest.tex', dataset)


        if rate_difference_for_max_accuracy:
            l = ' & '.join([f"{id_}", 
                    str(float(f"{df_noremoval.loc[df_noremoval['Test_acc'] == df_noremoval['Test_acc'].max()]['FPR_diff'].min():.{precision}e}")),
                    str(float(f"{df_nosensitive.loc[df_nosensitive['Test_acc'] == df_nosensitive['Test_acc'].max()]['FPR_diff'].min():.{precision}e}")),
                    str(float(f"{df_DIR.loc[df_DIR['Test_acc'] == df_DIR['Test_acc'].max()]['FPR_diff'].min():.{precision}e}")),
                    str(float(f"{df_ps.loc[df_ps['Test_acc'] == df_ps['Test_acc'].max()]['FPR_diff'].min():.{precision}e}")),
                    str(float(f"{df_massaging.loc[df_massaging['Test_acc'] == df_massaging['Test_acc'].max()]['FPR_diff'].min():.{precision}e}")),
                    str(float(f"{df_lfr.loc[df_lfr['Test_acc'] == df_lfr['Test_acc'].max()]['FPR_diff'].min():.{precision}e}")),
                    str(float(f"{df_adver.loc[df_adver['Test_acc'] == df_adver['Test_acc'].max()]['FPR_diff'].min():.{precision}e}")),
                    str(float(f"{df_our.loc[df_our['Test_acc'] == df_our['Test_acc'].max()]['FPR_diff'].min():.{precision}e}"))])

            m = ' & '.join([f"{id_}", 
                    str(float(f"{df_noremoval.loc[df_noremoval['Test_acc'] == df_noremoval['Test_acc'].max()]['FNR_diff'].min():.{precision}e}")),
                    str(float(f"{df_nosensitive.loc[df_nosensitive['Test_acc'] == df_nosensitive['Test_acc'].max()]['FNR_diff'].min():.{precision}e}")),
                    str(float(f"{df_DIR.loc[df_DIR['Test_acc'] == df_DIR['Test_acc'].max()]['FNR_diff'].min():.{precision}e}")),
                    str(float(f"{df_ps.loc[df_ps['Test_acc'] == df_ps['Test_acc'].max()]['FNR_diff'].min():.{precision}e}")),
                    str(float(f"{df_massaging.loc[df_massaging['Test_acc'] == df_massaging['Test_acc'].max()]['FNR_diff'].min():.{precision}e}")),
                    str(float(f"{df_lfr.loc[df_lfr['Test_acc'] == df_lfr['Test_acc'].max()]['FNR_diff'].min():.{precision}e}")),
                    str(float(f"{df_adver.loc[df_adver['Test_acc'] == df_adver['Test_acc'].max()]['FNR_diff'].min():.{precision}e}")),
                    str(float(f"{df_our.loc[df_our['Test_acc'] == df_our['Test_acc'].max()]['FNR_diff'].min():.{precision}e}"))])

            if kind:
                print_to_tex(l, 'fpr_rate-max-accuracy_fulltest.tex', dataset)
                print_to_tex(m, 'fnr_rate-max-accuracy_fulltest.tex', dataset)
            else:
                print_to_tex(l, 'fpr_rate-max-accuracy_debiasedtest.tex', dataset)
                print_to_tex(m, 'fnr_rate-max-accuracy_debiasedtest.tex', dataset)
                

    # df_main = None
    benchmarks = ["adult", "adult_race", "german", "student", "compas-ground", "compas-score", "default", "salary"]
    # benchmarks = ["adult", "german", "student", "compas-ground", "compas-score", "default"]
    kind = "debiased"
    # kind = "full"
    for dataset in benchmarks:
        if dataset == "adult":
            id_ = "D1"
        elif dataset == "adult_race":
            id_ = "D2"
        elif dataset == "german":
            id_ = "D3"
        elif dataset == "student":
            id_ = "D4"
        elif dataset == "compas-ground":
            id_ = "D5"
        elif dataset == "compas-score":
            id_ = "D6"
        elif dataset == "default":
            id_ = "D7"
        elif dataset == "salary":
            id_ = "D8"
        else:
            raise NotImplementedError   
        if kind == "full":
            fpr_fnr_print(dataset, id_, kind=True)
        elif kind == "debiased":
            fpr_fnr_print(dataset, id_, kind=False)
        print(f"Done {dataset}")
    

def parity_process_dfs(name, batches, df):
    if 'Model-count'in df.columns:
        df['Model-count2'] = df.apply(process_rows, axis=1, args=((batches,)))
        assert (df['Model-count'] == df['Model-count2']).all()
        df.drop(columns=['Model-count2'], inplace=True)
    else:
        df['Model-count'] = df.apply(process_rows, axis=1, args=((batches,)))
    assert len(df['Model-count'].unique()) == 240 and df['Model-count'].max() == 239 and df['Model-count'].min() == 0
    df = df.sort_values("Discm_percent").groupby("Model-count", as_index=False).first()     # must be sorted in order of model count for comparison across baselines
    df = df[['Model-count','Discm_percent','Test_acc', 'Class0_Pos', 'Class1_Pos']]
    df['diff'] = abs(df['Class0_Pos'] - df['Class1_Pos'])*100
    df['Test_acc'] = df['Test_acc'].apply(lambda x: x*100)
    df['Techniques'] = name
    if len(name.split()) > 1:
        words = name.split()
        letters = [word[0] for word in words]
        x = "".join(letters)
        df['Baseline'] = x
    else:    
        df['Baseline'] = name[:2]
    return df


def statistical_parity(dist_metric):

    def parity_print(dataset, id_, kind, plot=False):
        if kind:
            if dist_metric:
                df1 = pd.read_csv(f"{dataset}/results_{dataset}_method1_fulltest_dist{dist}.csv")
                df2 = pd.read_csv(f"{dataset}/results_{dataset}_method1_dist{dist}.csv")
            else:
                df1 = pd.read_csv(f"{dataset}/results_{dataset}_method1_fulltest.csv")
                df2 = pd.read_csv(f"{dataset}/results_{dataset}_method1.csv")
        else:
            if dist_metric:
                df1 = pd.read_csv(f"{dataset}/results_{dataset}_method1_dist{dist}.csv")
            else:
                df1 = pd.read_csv(f"{dataset}/results_{dataset}_method1.csv")
        batches = sorted(list(df1.Batch.unique()))
        assert(len(batches) == 2)

        df_our = find_min_discm_each_hyperparam(df1)
        df_our = df_our[['Model-count','Discm_percent', 'Test_acc', 'Class0_Pos', 'Class1_Pos']]
        df_our['diff'] = abs(df_our['Class0_Pos'] - df_our['Class1_Pos']) * 100
        if kind:
            df_our2 = find_min_discm_each_hyperparam(df2)       # since the sorting is on the basis of discm, it remains same
            df_our2['Test_acc'] = df_our2['Test_acc'].apply(lambda x: x * 100)
        df_our['Techniques'] = "Our Technique"
        df_our['Baseline'] = "Our"

        # import ipdb; ipdb.set_trace()
        if kind:
            if dist_metric:
                df_massaging = parity_process_dfs("MAssaging", batches, pd.read_csv(f"{dataset}/massaging/results_massaged_{dataset}_fulltest_dist{dist}.csv"))
                df_massaging2 = parity_process_dfs("MAssaging", batches, pd.read_csv(f"{dataset}/massaging/results_massaged_{dataset}_dist{dist}.csv"))
            else:
                df_massaging = parity_process_dfs("MAssaging", batches, pd.read_csv(f"{dataset}/massaging/results_massaged_{dataset}_fulltest.csv"))
                df_massaging2 = parity_process_dfs("MAssaging", batches, pd.read_csv(f"{dataset}/massaging/results_massaged_{dataset}.csv"))
        else:
            if dist_metric:
                df_massaging = parity_process_dfs("MAssaging", batches, pd.read_csv(f"{dataset}/massaging/results_massaged_{dataset}_dist{dist}.csv"))
            else:
                df_massaging = parity_process_dfs("MAssaging", batches, pd.read_csv(f"{dataset}/massaging/results_massaged_{dataset}.csv"))
        
        # Preferential Sampling
        if kind:
            if dist_metric:
                df_ps = parity_process_dfs("Prefer. Sampling", batches, pd.read_csv(f"{dataset}/preferential_sampling/results_resampling_{dataset}_fulltest_dist{dist}.csv"))
                df_ps2 = parity_process_dfs("Prefer. Sampling", batches, pd.read_csv(f"{dataset}/preferential_sampling/results_resampling_{dataset}_dist{dist}.csv"))
            else:
                df_ps = parity_process_dfs("Prefer. Sampling", batches, pd.read_csv(f"{dataset}/preferential_sampling/results_resampling_{dataset}_fulltest.csv"))
                df_ps2 = parity_process_dfs("Prefer. Sampling", batches, pd.read_csv(f"{dataset}/preferential_sampling/results_resampling_{dataset}.csv"))
        else:
            if dist_metric:
                df_ps = parity_process_dfs("Prefer. Sampling", batches, pd.read_csv(f"{dataset}/preferential_sampling/results_resampling_{dataset}_dist{dist}.csv"))
            else:
                df_ps = parity_process_dfs("Prefer. Sampling", batches, pd.read_csv(f"{dataset}/preferential_sampling/results_resampling_{dataset}.csv"))
        
        # Learning Fair representations
        if kind:
            if dist_metric:
                df_lfr = parity_process_dfs("Learning Fair Repr.", batches, pd.read_csv(f"{dataset}/learning_fair_representations/results_lfr_{dataset}_fulltest_dist{dist}.csv"))
                df_lfr2 = parity_process_dfs("Learning Fair Repr.", batches, pd.read_csv(f"{dataset}/learning_fair_representations/results_lfr_{dataset}_dist{dist}.csv"))
            else:
                df_lfr = parity_process_dfs("Learning Fair Repr.", batches, pd.read_csv(f"{dataset}/learning_fair_representations/results_lfr_{dataset}_fulltest.csv"))
                df_lfr2 = parity_process_dfs("Learning Fair Repr.", batches, pd.read_csv(f"{dataset}/learning_fair_representations/results_lfr_{dataset}.csv"))
        else:
            if dist_metric:
                df_lfr = parity_process_dfs("Learning Fair Repr.", batches, pd.read_csv(f"{dataset}/learning_fair_representations/results_lfr_{dataset}_dist{dist}.csv"))
            else:
                df_lfr = parity_process_dfs("Learning Fair Repr.", batches, pd.read_csv(f"{dataset}/learning_fair_representations/results_lfr_{dataset}.csv"))
        
        # Disparate Impact Removed
        if kind:
            if dist_metric:
                df_DIR = parity_process_dfs("Disp. Impact Rem", batches, pd.read_csv(f"{dataset}/disparate_impact_removed/results_disparate_removed_{dataset}_fulltest_dist{dist}.csv"))
                df_DIR2 = parity_process_dfs("Disp. Impact Rem", batches, pd.read_csv(f"{dataset}/disparate_impact_removed/results_disparate_removed_{dataset}_dist{dist}.csv"))
            else:
                df_DIR = parity_process_dfs("Disp. Impact Rem", batches, pd.read_csv(f"{dataset}/disparate_impact_removed/results_disparate_removed_{dataset}_fulltest.csv"))
                df_DIR2 = parity_process_dfs("Disp. Impact Rem", batches, pd.read_csv(f"{dataset}/disparate_impact_removed/results_disparate_removed_{dataset}.csv"))
        else:
            if dist_metric:
                df_DIR = parity_process_dfs("Disp. Impact Rem", batches, pd.read_csv(f"{dataset}/disparate_impact_removed/results_disparate_removed_{dataset}_dist{dist}.csv"))
            else:
                df_DIR = parity_process_dfs("Disp. Impact Rem", batches, pd.read_csv(f"{dataset}/disparate_impact_removed/results_disparate_removed_{dataset}.csv"))
        
        # Adversarial Sampling
        if kind:
            if dist_metric:
                df_adver = pd.read_csv(f"{dataset}/adversarial_debiasing/results_adversarial_debiased_{dataset}_fulltest_dist{dist}.csv")
                df_adver2 = pd.read_csv(f"{dataset}/adversarial_debiasing/results_adversarial_debiased_{dataset}_dist{dist}.csv")
            else:
                df_adver = pd.read_csv(f"{dataset}/adversarial_debiasing/results_adversarial_debiased_{dataset}_fulltest.csv")
                df_adver2 = pd.read_csv(f"{dataset}/adversarial_debiasing/results_adversarial_debiased_{dataset}.csv")
        else:
            if dist_metric:
                df_adver = pd.read_csv(f"{dataset}/adversarial_debiasing/results_adversarial_debiased_{dataset}_dist{dist}.csv")
            else:
                df_adver = pd.read_csv(f"{dataset}/adversarial_debiasing/results_adversarial_debiased_{dataset}.csv")
        
        df_adver['Model-count'] = df_adver['Dataperm'] * 12
        df_adver = df_adver.sort_values("Discm_percent").groupby("Model-count", as_index=False).first()     # must be sorted in order of model count for comparison across baselines
        df_adver['diff'] = df_adver['diff'] * 100
        df_adver['Test_acc'] = df_adver['Test_acc'].apply(lambda x: x * 100)
        if kind:
            df_adver2['Model-count'] = df_adver2['Dataperm'] * 12
            df_adver2 = df_adver2.sort_values("Discm_percent").groupby("Model-count", as_index=False).first()
            df_adver2['Test_acc'] = df_adver2['Test_acc'].apply(lambda x: x * 100)
        df_adver['Techniques'] = "Adversa. debias"
        df_adver['Baseline'] = "AD"

        
        if kind:
            if dist_metric:
                df_nosensitive = pd.read_csv(f"{dataset}/results_{dataset}_nosensitive_fulltest_dist{dist}.csv")
                df_nosensitive2 = pd.read_csv(f"{dataset}/results_{dataset}_nosensitive_dist{dist}.csv")
            else:
                df_nosensitive = pd.read_csv(f"{dataset}/results_{dataset}_nosensitive_fulltest.csv")
                df_nosensitive2 = pd.read_csv(f"{dataset}/results_{dataset}_nosensitive.csv")
        else:
            if dist_metric:
                df_nosensitive = pd.read_csv(f"{dataset}/results_{dataset}_nosensitive_dist{dist}.csv")
            else:
                df_nosensitive = pd.read_csv(f"{dataset}/results_{dataset}_nosensitive.csv")
        
        # import ipdb; ipdb.set_trace()
        if dist_metric:
            df_nosensitive = df_nosensitive[['Model-count', 'Test_acc', 'Class0_Pos', 'Class1_Pos', 'Discm_percent']]
        else:
            df_nosensitive = df_nosensitive[['Model-count', 'Test_acc', 'Class0_Pos', 'Class1_Pos']]
        df_nosensitive['diff'] = abs(df_nosensitive['Class0_Pos'] - df_nosensitive['Class1_Pos']) * 100
        if kind:
            df_nosensitive2['Test_acc'] = df_nosensitive2['Test_acc'].apply(lambda x: x * 100)
        df_nosensitive['Techniques'] = "Sens. Removed"
        df_nosensitive['Baseline'] = "SR"

        if kind:
            if dist_metric:
                df_noremoval = parity_process_dfs("FULL", batches, pd.read_csv(f"{dataset}/results_{dataset}_noremoval_fulltest_dist{dist}.csv"))
                df_noremoval2 = parity_process_dfs("FULL", batches, pd.read_csv(f"{dataset}/results_{dataset}_noremoval_dist{dist}.csv"))
            else:
                df_noremoval = parity_process_dfs("FULL", batches, pd.read_csv(f"{dataset}/results_{dataset}_noremoval_fulltest.csv"))
                df_noremoval2 = parity_process_dfs("FULL", batches, pd.read_csv(f"{dataset}/results_{dataset}_noremoval.csv"))
        else:
            if dist_metric:
                df_noremoval = parity_process_dfs("FULL", batches, pd.read_csv(f"{dataset}/results_{dataset}_noremoval_dist{dist}.csv"))
            else:
                df_noremoval = parity_process_dfs("FULL", batches, pd.read_csv(f"{dataset}/results_{dataset}_noremoval.csv"))

        df_main = pd.concat([df_noremoval, df_nosensitive, df_massaging, df_ps, df_lfr, df_DIR, df_adver, df_our], sort=True)
        try:
            assert(len(df_main) == 7*240 + 20)
        except:
            import ipdb; ipdb.set_trace()
        
        if plot:
            return df_main
        min_difference = True
        parity_difference_for_max_accuracy = True
        parity_difference_for_min_discm = True
        discm_for_min_parity = True
        accuracy_for_min_parity = True
        precision = 1

        if min_difference:
            a = ' & '.join([f"{id_}", 
                    str(float(f"{df_noremoval['diff'].min():.{precision}e}")),
                    str(float(f"{df_nosensitive['diff'].min():.{precision}e}")),
                    str(float(f"{df_DIR['diff'].min():.{precision}e}")), 
                    str(float(f"{df_ps['diff'].min():.{precision}e}")), 
                    str(float(f"{df_massaging['diff'].min():.{precision}e}")), 
                    str(float(f"{df_lfr['diff'].min():.{precision}e}")), 
                    str(float(f"{df_adver['diff'].min():.{precision}e}")), 
                    str(float(f"{df_our['diff'].min():.{precision}e}"))])

            if kind:
                if dist_metric:
                    print_to_tex(a, f'min-parity-diff_fulltest_dist{dist}.tex', dataset)
                else:
                    print_to_tex(a, 'min-parity-diff_fulltest.tex', dataset)
            else:
                if dist_metric:
                    print_to_tex(a, f'min-parity-diff_debiasedtest_dist{dist}.tex', dataset)
                else:
                    print_to_tex(a, 'min-parity-diff_debiasedtest.tex', dataset)

            
        if parity_difference_for_min_discm:
            if dist_metric:
                x = ' & '.join([f"{id_}",
                str(float(f"{df_noremoval.loc[df_noremoval['Discm_percent'] == df_noremoval['Discm_percent'].min()]['diff'].min():.{precision}e}")),
                str(float(f"{df_nosensitive.loc[df_nosensitive['Discm_percent'] == df_nosensitive['Discm_percent'].min()]['diff'].min():.{precision}e}")),
                str(float(f"{df_DIR.loc[df_DIR['Discm_percent'] == df_DIR['Discm_percent'].min()]['diff'].min():.{precision}e}")),
                str(float(f"{df_ps.loc[df_ps['Discm_percent'] == df_ps['Discm_percent'].min()]['diff'].min():.{precision}e}")), 
                str(float(f"{df_massaging.loc[df_massaging['Discm_percent'] == df_massaging['Discm_percent'].min()]['diff'].min():.{precision}e}")),
                str(float(f"{df_lfr.loc[df_lfr['Discm_percent'] == df_lfr['Discm_percent'].min()]['diff'].min():.{precision}e}")),
                str(float(f"{df_adver.loc[df_adver['Discm_percent'] == df_adver['Discm_percent'].min()]['diff'].min():.{precision}e}")),
                str(float(f"{df_our.loc[df_our['Discm_percent'] == df_our['Discm_percent'].min()]['diff'].min():.{precision}e}"))]
                )
            else:
                x = ' & '.join([f"{id_}",
                str(float(f"{df_noremoval.loc[df_noremoval['Discm_percent'] == df_noremoval['Discm_percent'].min()]['diff'].min():.{precision}e}")),
                str(float(f"{df_nosensitive.loc[df_nosensitive['Test_acc'] == df_nosensitive['Test_acc'].max()]['diff'].min():.{precision}e}")),
                str(float(f"{df_DIR.loc[df_DIR['Discm_percent'] == df_DIR['Discm_percent'].min()]['diff'].min():.{precision}e}")),
                str(float(f"{df_ps.loc[df_ps['Discm_percent'] == df_ps['Discm_percent'].min()]['diff'].min():.{precision}e}")), 
                str(float(f"{df_massaging.loc[df_massaging['Discm_percent'] == df_massaging['Discm_percent'].min()]['diff'].min():.{precision}e}")),
                str(float(f"{df_lfr.loc[df_lfr['Discm_percent'] == df_lfr['Discm_percent'].min()]['diff'].min():.{precision}e}")),
                str(float(f"{df_adver.loc[df_adver['Discm_percent'] == df_adver['Discm_percent'].min()]['diff'].min():.{precision}e}")),
                str(float(f"{df_our.loc[df_our['Discm_percent'] == df_our['Discm_percent'].min()]['diff'].min():.{precision}e}"))]
                )
            
            if kind:
                if dist_metric:
                    print_to_tex(x, f'parity-diff-min-discm_fulltest_dist{dist}.tex', dataset)
                else:
                    print_to_tex(x, 'parity-diff-min-discm_fulltest.tex', dataset)
            else:
                if dist_metric:
                    print_to_tex(x, f'parity-diff-min-discm_debiasedtest_dist{dist}.tex', dataset)
                else:
                    print_to_tex(x, 'parity-diff-min-discm_debiasedtest.tex', dataset)

        
        if parity_difference_for_max_accuracy:
            x = ' & '.join([f"{id_}",
            str(float(f"{df_noremoval.loc[df_noremoval['Test_acc'] == df_noremoval['Test_acc'].max()]['diff'].min():.{precision}e}")),
            str(float(f"{df_nosensitive.loc[df_nosensitive['Test_acc'] == df_nosensitive['Test_acc'].max()]['diff'].min():.{precision}e}")), 
            str(float(f"{df_DIR.loc[df_DIR['Test_acc'] == df_DIR['Test_acc'].max()]['diff'].min():.{precision}e}")),
            str(float(f"{df_ps.loc[df_ps['Test_acc'] == df_ps['Test_acc'].max()]['diff'].min():.{precision}e}")),
            str(float(f"{df_massaging.loc[df_massaging['Test_acc'] == df_massaging['Test_acc'].max()]['diff'].min():.{precision}e}")),
            str(float(f"{df_lfr.loc[df_lfr['Test_acc'] == df_lfr['Test_acc'].max()]['diff'].min():.{precision}e}")),
            str(float(f"{df_adver.loc[df_adver['Test_acc'] == df_adver['Test_acc'].max()]['diff'].min():.{precision}e}")),
            str(float(f"{df_our.loc[df_our['Test_acc'] == df_our['Test_acc'].max()]['diff'].min():.{precision}e}"))]
            )
            
            if kind:
                if dist_metric:
                    print_to_tex(x, f'parity-diff-max-accuracy_fulltest_dist{dist}.tex', dataset)
                else:
                    print_to_tex(x, 'parity-diff-max-accuracy_fulltest.tex', dataset)
            else:
                if dist_metric:
                    print_to_tex(x, 'parity-diff-max-accuracy_debiasedtest_dist{dist}.tex', dataset)
                else:
                    print_to_tex(x, 'parity-diff-max-accuracy_debiasedtest.tex', dataset)
    

        if discm_for_min_parity:    # Discm will be same across full test and debiased test
            if dist_metric:
                k = ' & '.join([f"{id_}",
                f"{df_noremoval.loc[df_noremoval['diff'] == df_noremoval['diff'].min()]['Discm_percent'].min():.{precision}e}",
                f"{df_nosensitive.loc[df_nosensitive['diff'] == df_nosensitive['diff'].min()]['Discm_percent'].min():.{precision}e}",
                f"{df_DIR.loc[df_DIR['diff'] == df_DIR['diff'].min()]['Discm_percent'].min():.{precision}e}",
                f"{df_ps.loc[df_ps['diff'] == df_ps['diff'].min()]['Discm_percent'].min():.{precision}e}",
                f"{df_massaging.loc[df_massaging['diff'] == df_massaging['diff'].min()]['Discm_percent'].min():.{precision}e}",
                f"{df_lfr.loc[df_lfr['diff'] == df_lfr['diff'].min()]['Discm_percent'].min():.{precision}e}",
                f"{df_adver.loc[df_adver['diff'] == df_adver['diff'].min()]['Discm_percent'].min():.{precision}e}",
                f"{df_our.loc[df_our['diff'] == df_our['diff'].min()]['Discm_percent'].min():.{precision}e}"])

                print_to_tex(k, f'discm_for_min_parity_dist{dist}.tex', dataset)
            
            else:
                k = ' & '.join([f"{id_}",
                f"{df_noremoval.loc[df_noremoval['diff'] == df_noremoval['diff'].min()]['Discm_percent'].min():.{precision}e}",
                "0.0",
                f"{df_DIR.loc[df_DIR['diff'] == df_DIR['diff'].min()]['Discm_percent'].min():.{precision}e}",
                f"{df_ps.loc[df_ps['diff'] == df_ps['diff'].min()]['Discm_percent'].min():.{precision}e}",
                f"{df_massaging.loc[df_massaging['diff'] == df_massaging['diff'].min()]['Discm_percent'].min():.{precision}e}",
                f"{df_lfr.loc[df_lfr['diff'] == df_lfr['diff'].min()]['Discm_percent'].min():.{precision}e}",
                f"{df_adver.loc[df_adver['diff'] == df_adver['diff'].min()]['Discm_percent'].min():.{precision}e}",
                f"{df_our.loc[df_our['diff'] == df_our['diff'].min()]['Discm_percent'].min():.{precision}e}"])
                
                print_to_tex(k, 'discm_for_min_parity.tex', dataset)


        if accuracy_for_min_parity:
            z = ' & '.join([f"{id_}", 
            f"{df_noremoval2.loc[df_noremoval['diff'] == df_noremoval['diff'].min()]['Test_acc'].max():.{precision}e}", 
            f"{df_nosensitive2.loc[df_nosensitive['diff'] == df_nosensitive['diff'].min()]['Test_acc'].max():.{precision}e}", 
            f"{df_DIR2.loc[df_DIR['diff'] == df_DIR['diff'].min()]['Test_acc'].max():.{precision}e}", 
            f"{df_ps2.loc[df_ps['diff'] == df_ps['diff'].min()]['Test_acc'].max():.{precision}e}", 
            f"{df_massaging2.loc[df_massaging['diff'] == df_massaging['diff'].min()]['Test_acc'].max():.{precision}e}", 
            f"{df_lfr2.loc[df_lfr['diff'] == df_lfr['diff'].min()]['Test_acc'].max():.{precision}e}", 
            f"{df_adver2.loc[df_adver['diff'] == df_adver['diff'].min()]['Test_acc'].max():.{precision}e}", 
            f"{df_our2.loc[df_our['diff'] == df_our['diff'].min()]['Test_acc'].max():.{precision}e}"])

            if dist_metric:
                print_to_tex(z, f'test_accuracy_for_min_parity_dist{dist}.tex', dataset)
            else:
                print_to_tex(z, 'test_accuracy_for_min_parity.tex', dataset)


    benchmarks = ["adult", "adult_race", "german", "student", "compas-ground", "compas-score", "default", "salary"]

    kind = "full"
    # kind = "debiased"
    df_main = None
    plot = False
    for dataset in benchmarks:
        if dataset == "adult":
            id_ = "D1"
        elif dataset == "adult_race":
            id_ = "D2"
        elif dataset == "german":
            id_ = "D3"
        elif dataset == "student":
            id_ = "D4"
        elif dataset == "compas-ground":
            id_ = "D5"
        elif dataset == "compas-score":
            id_ = "D6"
        elif dataset == "default":
            id_ = "D7"
        elif dataset == "salary":
            id_ = "D8"
        else:
            raise NotImplementedError   
        if kind == "full":
            df_onedataset = parity_print(dataset, id_, kind=True, plot=plot)
            if plot:
                df_onedataset['Dataset'] = id_
                if not df_main is None:
                    df_main = pd.concat([df_main, df_onedataset])
                else:
                    df_main = copy.deepcopy(df_onedataset)
        elif kind == "debiased":
            parity_print(dataset, id_, kind=False, plot=plot)
        print(f"Done {dataset}")
        
    if plot:
        labels = ['FU', 'SR', 'DIR', 'PS', 'MA', 'LFR', 'AD', 'Our']
        tech_cat = pd.Categorical(df_main['Baseline'], categories=labels)    
        df_main = df_main.assign(Technique_x = tech_cat)
        dataset_order = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"]
        data_cat = pd.Categorical(df_main['Dataset'], categories=dataset_order)    
        df_main = df_main.assign(Dataset_x = data_cat)
        x = (ggplot(aes(x='Technique_x', y='diff'), data=df_main) +\
        geom_boxplot() +\
        facet_wrap(['Dataset_x'], scales = 'free', nrow=2, labeller='label_value', shrink=True) + \
        ylab("Statistical Parity Difference") + \
        xlab("Discrimination reducing techniques") + \
        # ylim(0, 20) + \
        # ggtitle("Box plot showing remaining discrimination for each technique in each dataset") +\
        theme(axis_text_x = element_text(size=6), dpi=151) + \
        theme_seaborn()
        )
        x = x.draw()
        x.set_figwidth(20)
        x.set_figheight(12)
        x.savefig(f"boxplots/boxplot_parity_freeaxis_matplotlib.png", bbox_inches='tight')



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--discm_and_accuracy_plot", type=int, default=1,
                        help="Want to plot or generate tables for accuracy and discrimination")
    parser.add_argument("--parity", type=int, default=0,
                        help="Want generate tables for statistical parity")
    parser.add_argument("--dist", type=int, default=10,
                        help="What dist metric to use - only 10 now")
    args = parser.parse_args()
    
    if args.discm_and_accuracy_plot == 0 or args.discm_and_accuracy_plot == 1:
        main(args.discm_and_accuracy_plot)
    
    # if bool(args.parity):
    #     # statistical_parity()
    #     statistical_parity(args.dist)

    # real_accuracy_tables(True)
    # real_accuracy_tables(False)
    # fpr_fnr_rates()

