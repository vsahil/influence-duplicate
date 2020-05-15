from mpl_toolkits import mplot3d
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from plotnine import *
import copy


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
    df['Model-count'] = df.apply(process_rows, axis=1, args=((batches,)))
    assert len(df['Model-count'].unique()) == 240 and df['Model-count'].max() == 239 and df['Model-count'].min() == 0
    df = df.sort_values("Discm_percent").groupby("Model-count", as_index=False).first()     # must be sorted in order of model count for comparison across baselines
    df = df[['Model-count','Discm_percent','Test_acc']]
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


def boxplots_datasets(dataset, plot):
    # df1 = pd.read_csv(f"{dataset}/results_{dataset}_final.csv")
    df1 = pd.read_csv(f"{dataset}/results_{dataset}_method1.csv")
    batches = list(df1.Batch.unique())
    assert(len(batches) == 2)
    # import ipdb; ipdb.set_trace()
    df_our = find_min_discm_each_hyperparam(df1)
    df_our = df_our[['Model-count','Discm_percent','Test_acc']]
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
    df_adver = df_adver[['Model-count','Discm_percent','Test_acc']]
    df_adver['Test_acc'] = df_adver['Test_acc'].apply(lambda x: x*100)
    df_adver['Techniques'] = "Adversa. debias"
    df_adver['Baseline'] = "AD"

    # # Sensitive Attribute removed, therefore no discrimination
    df_nosensitive = pd.read_csv(f"{dataset}/results_{dataset}_nosensitive.csv")
    df_nosensitive = df_nosensitive[['Model-count','Test_acc']]
    df_nosensitive['Test_acc'] = df_nosensitive['Test_acc'].apply(lambda x: x*100)
    df_nosensitive['Techniques'] = "Sens. Removed"
    df_nosensitive['Baseline'] = "SR"
    # df_nosensitive = process_dfs("Sensitive Removed", batches, pd.read_csv(f"{dataset}/results_{dataset}_nosensitive.csv"))

    # No technique used
    df_noremoval = process_dfs("FULL", batches, pd.read_csv(f"{dataset}/results_{dataset}_noremoval.csv"))
    
    # Not appending sensitive removed here
    # df_main = pd.concat([df_noremoval, df_massaging, df_ps, df_lfr, df_DIR, df_adver, df_our])
    # assert(len(df_main) == 6*240 + 20)
    df_main = pd.concat([df_noremoval, df_nosensitive, df_massaging, df_ps, df_lfr, df_DIR, df_adver, df_our])
    assert(len(df_main) == 7*240 + 20)
    
    # df_main = pd.concat([df_our, df_massaging, df_ps, df_lfr, df_DIR, df_adver, df_nosensitive])
    # assert(len(df_main) == 6*240 + 20)
    if dataset == "compas-score":
        dataset = "Recidivism-score"
    elif dataset == "compas-ground":
        dataset = "Recidivism-ground"
    # df_main['Dataset'] = dataset.capitalize()

    if dataset == "adult":
        sizeofPSI = 4522200
        id_ = "D1"
    elif dataset == "german":
        sizeofPSI = 100000
        id_ = "D2"
    elif dataset == "student":
        sizeofPSI = 64900
        id_ = "D3"
    elif dataset == "Recidivism-ground":
        sizeofPSI = 615000
        id_ = "D4"
    elif dataset == "Recidivism-score":
        sizeofPSI = 615000
        id_ = "D5"
    elif dataset == "default":
        sizeofPSI = 3000000
        id_ = "D6"
    else:
        raise NotImplementedError
    
    df_main['Dataset'] = id_
    precision = 1

    if plot == 0:
        min_discm = False
        median_discm = False
        max_accuracy = False
        corresponding_max_accuracy = True
        mean_accuracy = False
        median_accuracy = False
        # import ipdb; ipdb.set_trace()
        if min_discm:
            # x = ' & '.join([f"{dataset.capitalize()}", str(float(f"{df_DIR['Discm_percent'].min():.{precision}e}")), str(float(f"{df_ps['Discm_percent'].min():.{precision}e}")), str(float(f"{df_massaging['Discm_percent'].min():.{precision}e}")), str(float(f"{df_lfr['Discm_percent'].min():.{precision}e}")), str(float(f"{df_adver['Discm_percent'].min():.{precision}e}")), str(float(f"{df_nosensitive['Discm_percent'].min():.{precision}e}")), str(float(f"{df_noremoval['Discm_percent'].min():.{precision}e}")), "\\textbf{%s}"%(str(float(f"{df_our['Discm_percent'].min():.{precision}e}")))])
            # x = ' & '.join([f"{df_noremoval['Discm_percent'].min():.{precision}e}", f"{df_nosensitive['Discm_percent'].min():.{precision}e}", f"{dataset.capitalize()}", f"{df_DIR['Discm_percent'].min():.{precision}e}", f"{df_ps['Discm_percent'].min():.{precision}e}", f"{df_massaging['Discm_percent'].min():.{precision}e}", f"{df_lfr['Discm_percent'].min():.{precision}e}", f"{df_adver['Discm_percent'].min():.{precision}e}", "\\textbf{%s}"%(f"{df_our['Discm_percent'].min():.{precision}e}")])
            x = ' & '.join([f"{id_}", f"{df_noremoval['Discm_percent'].min():.{precision}e}", f"{df_DIR['Discm_percent'].min():.{precision}e}", f"{df_ps['Discm_percent'].min():.{precision}e}", f"{df_massaging['Discm_percent'].min():.{precision}e}", f"{df_lfr['Discm_percent'].min():.{precision}e}", f"{df_adver['Discm_percent'].min():.{precision}e}", "\\textbf{%s}"%(f"{df_our['Discm_percent'].min():.{precision}e}")])
            print_to_tex(x, 'min-discm.tex', dataset)

        if median_discm:
            x = ' & '.join([f"{id_}", f"{df_noremoval['Discm_percent'].median():.{precision}e}", f"{df_DIR['Discm_percent'].median():.{precision}e}", f"{df_ps['Discm_percent'].median():.{precision}e}", f"{df_massaging['Discm_percent'].median():.{precision}e}", f"{df_lfr['Discm_percent'].median():.{precision}e}", f"{df_adver['Discm_percent'].median():.{precision}e}", "\\textbf{%s}"%(f"{df_our['Discm_percent'].median():.{precision}e}")])
            print_to_tex(x, 'median-discm.tex', dataset)


        if max_accuracy:
            y = ' & '.join([f"{id_}", f"{df_noremoval['Test_acc'].max():.{precision}e}", f"{df_nosensitive['Test_acc'].max():.{precision}e}", f"{df_DIR['Test_acc'].max():.{precision}e}", f"{df_ps['Test_acc'].max():.{precision}e}", f"{df_massaging['Test_acc'].max():.{precision}e}", f"{df_lfr['Test_acc'].max():.{precision}e}", f"{df_adver['Test_acc'].max():.{precision}e}", "\\textbf{%s}"%(f"{df_our['Test_acc'].max():.{precision}e}")])
            print_to_tex(y, 'max-test-accuracy.tex', dataset)

        if corresponding_max_accuracy:
            # for sensitive there is always 0 discrimination. 
            z = ' & '.join([f"{id_}", f"{df_noremoval.loc[df_noremoval['Discm_percent'] == df_noremoval['Discm_percent'].min()]['Test_acc'].max():.{precision}e}", 
                    f"{df_nosensitive['Test_acc'].max():.{precision}e}", 
                    f"{df_DIR.loc[df_DIR['Discm_percent'] == df_DIR['Discm_percent'].min()]['Test_acc'].max():.{precision}e}", 
                    f"{df_ps.loc[df_ps['Discm_percent'] == df_ps['Discm_percent'].min()]['Test_acc'].max():.{precision}e}", 
                    f"{df_massaging.loc[df_massaging['Discm_percent'] == df_massaging['Discm_percent'].min()]['Test_acc'].max():.{precision}e}", 
                    f"{df_lfr.loc[df_lfr['Discm_percent'] == df_lfr['Discm_percent'].min()]['Test_acc'].max():.{precision}e}", 
                    f"{df_adver.loc[df_adver['Discm_percent'] == df_adver['Discm_percent'].min()]['Test_acc'].max():.{precision}e}", 
                    "\\textbf{%s}"%(f"{df_our.loc[df_our['Discm_percent'] == df_our['Discm_percent'].min()]['Test_acc'].max():.{precision}e}")])
            
            print_to_tex(z, 'correspond-test-accuracy.tex', dataset)
        
        if mean_accuracy:
            a = ' & '.join([f"{id_}", f"{df_noremoval['Test_acc'].mean():.{precision}e}", f"{df_nosensitive['Test_acc'].mean():.{precision}e}", f"{df_DIR['Test_acc'].mean():.{precision}e}", f"{df_ps['Test_acc'].mean():.{precision}e}", f"{df_massaging['Test_acc'].mean():.{precision}e}", f"{df_lfr['Test_acc'].mean():.{precision}e}", f"{df_adver['Test_acc'].mean():.{precision}e}", "\\textbf{%s}"%(f"{df_our['Test_acc'].mean():.{precision}e}")])
            print_to_tex(a, 'mean-test-accuracy.tex', dataset)

        if median_accuracy:
            b = ' & '.join([f"{id_}", f"{df_noremoval['Test_acc'].median():.{precision}e}", f"{df_nosensitive['Test_acc'].median():.{precision}e}", f"{df_DIR['Test_acc'].median():.{precision}e}", f"{df_ps['Test_acc'].median():.{precision}e}", f"{df_massaging['Test_acc'].median():.{precision}e}", f"{df_lfr['Test_acc'].median():.{precision}e}", f"{df_adver['Test_acc'].median():.{precision}e}", "\\textbf{%s}"%(f"{df_our['Test_acc'].median():.{precision}e}")])
            print_to_tex(b, 'median-test-accuracy.tex', dataset)

    return df_main


def print_to_tex(string, file, dataset, mode=None):
    if mode == None:
        if dataset == "adult":
            mode = "w"
        else:
            mode = "a"
    with open(f"../../neurips_fairness_paper/{file}", mode) as f:
        if not dataset == "default":
            string += "  \\\  \hline"
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
    benchmarks = ["adult", "german", "student", "compas-ground", "compas-score", "default"]
    # benchmarks = ["default"] #,"german", "student", "compas-ground", "compas-score", "default"]
    for dataset in benchmarks:
        df_onedataset = boxplots_datasets(dataset, plot)
        if not df_main is None:
            df_main = pd.concat([df_main, df_onedataset])
        else:
            df_main = copy.deepcopy(df_onedataset)
        print(f"Done {dataset}")

    if plot == 0:
        return 

    # labels = ['FU', 'SR', 'DIR', 'PS', 'MA', 'LFR', 'AD', 'Our']
    # labels = ['FU', 'DIR', 'PS', 'MA', 'LFR', 'AD', 'Our']
    labels = ['FU', 'SR', 'DIR', 'PS', 'MA', 'LFR', 'AD', 'Our']
    # indexNames = df_main[df_main['Baseline'] == 'FU' ].index
    # df_main.drop(indexNames , inplace=True)

    tech_cat = pd.Categorical(df_main['Baseline'], categories=labels)    
    df_main = df_main.assign(Technique_x = tech_cat)
    
    # dataset_order = ["Adult", "German", "Student", "Recidivism-ground", "Recidivism-score", "Default"]
    dataset_order = ["D1", "D2", "D3", "D4", "D5", "D6"]
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

    # x = (ggplot(aes(x='Technique_x', y='Discm_percent'), data=df_main) +\
    #         geom_boxplot() +\
    #         facet_wrap(['Dataset_x'], scales = 'free', nrow=1, labeller='label_value', shrink=True) + \
    #         ylab("Remaining Individual Discrimination") + \
    #         xlab("Discrimination reducing techniques") + \
    #         ylim(0, 20) + \
    #         # ggtitle("Box plot showing remaining discrimination for each technique in each dataset") +\
    #         theme(axis_text_x = element_text(size=6), dpi=151) + \
    #         theme_seaborn()
    #         )

    # x.save(f"boxplot_discm_freeaxis.png", height=8, width=18)
    # x.save(f"boxplot_discm_freeaxis_withoutfull.png", height=12, width=15)
    # x.save(f"boxplot_discm_fixedaxis.png", height=5, width=12)

    y = (ggplot(aes(x='Technique_x', y='Test_acc'), data=df_main) +\
            geom_boxplot() +\
            facet_wrap(['Dataset_x'], scales = 'free', nrow=1, labeller='label_value', shrink=True) + \
            ylab("Test Accuracy") + \
            xlab("Discrimination reducing techniques") + \
            ylim(0, 100) + \
            # ggtitle("Box plot showing remaining discrimination for each technique in each dataset") +\
            theme(axis_text_x = element_text(size=6), dpi=151) + \
            theme_seaborn()
            )

    y.save(f"boxplot_accuracy_freeaxis.png", height=8, width=18)


def real_accuracy_tables(debiased):
    dataset = "compas-score"
    if debiased:
        deb = "debiased"
    else:
        deb = "full"

    df1 = pd.read_csv(f"{dataset}/results_{dataset}_method1.csv")
    batches = list(df1.Batch.unique())
    assert(len(batches) == 2)
    df_our = find_min_discm_each_hyperparam(df1)
    df_our = df_our[['Model-count','Discm_percent']]
    df_our_2 = pd.read_csv(f"{dataset}/results_our_real_accuracy_{deb}.csv")
    df_our_final = pd.merge(df_our, df_our_2, on=['Model-count'])
    df_our_final['Test_acc'] = df_our_final['Test_acc'].apply(lambda x: x*100)
    df_our_final['Techniques'] = "Our Technique"
    df_our_final['Baseline'] = "Our"
    import ipdb; ipdb.set_trace()
    
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

if __name__ == "__main__":
    plot = int(sys.argv[1])
    # main(plot)
    real_accuracy_tables(True)
    # real_accuracy_tables(False)

