from mpl_toolkits import mplot3d
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from plotnine import *
import copy


def find_min_discm_each_hyperparam(df):
    x = df.sort_values("Discm_percent").groupby("Model-count", as_index=False).first()    
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
    df['Model-count'] = df.apply(process_rows, axis=1, args=((batches,)))
    assert len(df['Model-count'].unique()) == 240 and df['Model-count'].max() == 239 and df['Model-count'].min() == 0
    df = df.sort_values("Discm_percent").groupby("Model-count", as_index=False).first()     # must be sorted in order of model count for comparison across baselines
    df = df[['Model-count','Discm_percent']]
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
    df1 = pd.read_csv(f"{dataset}/results_{dataset}_final.csv")
    batches = list(df1.Batch.unique())
    assert(len(batches) == 2)
    df_our = find_min_discm_each_hyperparam(df1)
    df_our = df_our[['Model-count','Discm_percent']]
    df_our['Techniques'] = "Our Technique"
    df_our['Baseline'] = "Our"

    # Massaging
    df_massaging = process_dfs("Massaging", batches, pd.read_csv(f"{dataset}/massaging/results_massaged_{dataset}.csv"))
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
    df_adver = df_adver[['Model-count','Discm_percent']]
    df_adver['Techniques'] = "Adversa. debias"
    df_adver['Baseline'] = "AD"

    df_main = pd.concat([df_our, df_massaging, df_ps, df_lfr, df_DIR, df_adver])
    assert(len(df_main) == 5*240 + 20)
    df_main['Dataset'] = dataset.capitalize()

    if dataset == "german":
        sizeofPSI = 100000
    elif dataset == "adult":
        sizeofPSI = 4522200
    elif dataset == "student":
        sizeofPSI = 100000
    elif dataset == "default":
        sizeofPSI = 3000000
    else:
        raise NotImplementedError
    
    rounding = 3

    if plot == 0:
        x = ' & '.join([f"{dataset.capitalize()}", f"{sizeofPSI}", f"{df_DIR['Discm_percent'].min().round(rounding)}", f"{df_ps['Discm_percent'].min().round(rounding)}", f"{df_massaging['Discm_percent'].min().round(rounding)}", f"{df_lfr['Discm_percent'].min().round(rounding)}", f"{df_adver['Discm_percent'].min().round(rounding)}", "\\textbf{%s}"%(f"{df_our['Discm_percent'].min().round(rounding)}")])
        if dataset == "adult":
            mode = "w"
        else:
            mode = "a"
        with open("../../neurips_fairness_paper/min-discm.tex", mode) as f:
            if not dataset == "default":
                x += "  \\\  \hline"
            else:
                x += "  \\\\  "
            print(x, file=f)
        
        # print(dataset)
        # print("Min discm: ", df_DIR['Discm_percent'].min())
        # print("Min discm: ", df_ps['Discm_percent'].min())
        # print("Min discm: ", df_massaging['Discm_percent'].min())
        # print("Min discm: ", df_lfr['Discm_percent'].min())
        # print("Min discm: ", df_adver['Discm_percent'].min())
        # print("Min discm: ", df_our['Discm_percent'].min())
    
    return df_main


def main(plot):
    df_main = None
    benchmarks = ["adult", "german", "student", "default"]
    for dataset in benchmarks:
    # for dataset in ["adult", "default"]:
        df_onedataset = boxplots_datasets(dataset, plot)
        if not df_main is None:
            df_main = pd.concat([df_main, df_onedataset])
        else:
            df_main = copy.deepcopy(df_onedataset)
        print(f"Done {dataset}")

    if plot == 0:
        return 
    # import ipdb; ipdb.set_trace()
    labels = ['DIR', 'PS', 'Ma', 'LFR', 'AD', 'Our']
    tech_cat = pd.Categorical(df_main['Baseline'], categories=labels)
    df_main = df_main.assign(Technique_x = tech_cat)
    
    x = (ggplot(aes(x='Technique_x', y='Discm_percent', color='Techniques'), data=df_main) +\
            geom_boxplot() +\
            facet_wrap(['Dataset'], scales = 'free', nrow=1, ncol=len(benchmarks), labeller='label_both', shrink=False) + \
            ylab("Remaining Individual Discrimination") + \
            xlab("Discrimination reducing techniques") + \
            # ggtitle("Box plot showing remaining discrimination for each technique in each dataset") +\
            theme(axis_text_x = element_text(size=6), dpi=151) +\
            theme_seaborn()
            )

    x.save(f"boxplot_discm_freeaxis.png", height=5, width=12)
    # x.save(f"boxplot_discm_fixedaxis.png", height=5, width=12)

if __name__ == "__main__":
    plot = int(sys.argv[1])
    main(plot)