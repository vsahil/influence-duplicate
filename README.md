# Removing biased data to improve fairness and accuracy

Dependencies:
- Numpy/Scipy/Scikit-learn/Pandas
- Tensorflow (tested on v1.13.1)
- Keras (tested on v2.0.4)
- Matplotlib/Seaborn (for visualizations)
- Plotnine (tested on v0.6.0)

## Replication of tables and plots in the paper
### Tables
```bash
cd benchmarks
python faceted_box_plots_all_datasets.py --plot=0 --parity=1
cd tables
bash post-process.sh
```
### Plots
```bash
cd benchmarks
python faceted_box_plots_all_datasets.py --plot=1
```


## Generating pairs of similar individuals for a dataset (for example adult)
For generating pairs of similar individuals:
```bash
cd random_fairness_test_generator
python adult_main.py
```
For a new dataset, one can uncomment first function in adult_main.py to generate settings file for the dataset.

## Running the approach for a dataset (for example adult)
If you want to train the model with model sequene "model_number", you should:

```bash
cd benchmarks/adult
python train_all_permutations.py model_number
```

After training, flip the train flag from "True" to "False", and if you want to remove a certain percentage of data, you should add the desired percentage after the model_number.

`python train_all_permutations.py model_number percentage_removal`

Results will be accumutated in the file results_adult_debiasedtrain_80percentof_total.csv file. This will include all the model architecture setting and the remaining individual discrimination.

Running for all 240 hyperparameter choices can be accomplished by

`bash check_parallel.py`

This parallelizes the process of training the models and removing biased datapoints. 

Next, run the following command to get the datapoints removed for minimum discrimination for each of the 240 settings. 

`python methodology1.py`

## For other baselines
### Full
Flip the "full_test" flag to True and run. 
### Sensitive Removal
Run

`python train_all_permutations_nosensitive.py model_number` 
### Disparate Impact Removed
```bash
cd disparate_impact_removed
python run_hyperparams_disparate_impact_remover.py model_number
```
### Preferential sampling
```bash
cd preferential_sampling
python train_preferential_sampling.py model_number
```
### Massaging
```bash
cd massaging
python train_massage.py model_number
```
Results will be accumulated in the same directory
### Learning Fair Representations
```bash
cd learning_fair_representations
python learning_fr.py model_number
```
### Adversarial Debiasing
```bash
cd adversarial_debiasing
python adversarial_debiasing.py model_number
```

Each of the above approaches can be run for either "debiased test set" or "full test set", which can be passed as "debiased_test" flag in the files. 