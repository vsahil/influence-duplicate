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

## Repository Layout

- random_fairness_test_generator directory contains the code for generation of pairs of similar individuals. We have not added that in the code because the files were huge.
- For each dataset (six) we used, the raw input data along with pre-processing scripts are present in their respective dataset directories, e.g. : `adult-dataset`, `default-dataset` etc.
- The code for using our approach in any experiment is inside the `benchmarks` directory. In the `benchmarks` directory, there are separate directories for each experiment, for eg. `adult`, `default` etc.
- Each of the baselines are present as separate sub-directories inside respective experimental directories. For example, `adult/disparate_impact_removed`, `adult/preferential_sampling`, `adult/massaging`, `adult/learning_fair_representations`, and `adult/adversarial_debiasing`. Sensitive removal and full don't have a separate sub-directory. 

## Replicating Experiments

### Generating pairs of similar individuals for a dataset (for example adult)

For generating pairs of similar individuals:

```bash
cd random_fairness_test_generator
python adult_main.py
```

For a new dataset, one can uncomment first function in adult_main.py to generate settings file for the dataset.

### Running the approach for a dataset (for example adult)

If you want to train the model with model sequence "model_number", you should run:

```bash
cd benchmarks/adult
python train_all_permutations.py --train=1 --model_number=model_number
```

Since measuring of test accuracy needs to removed biased points, it can only be performed after all 240 models have been trained, by switching "Train" to False and adding percentage removal.

```bash
cd benchmarks/adult
python train_all_permutations.py --train=0 --model_number=model_number --percentage_removal=x
```

Results will be accumutated in the file results_adult_debiasedtrain_80percentof_total.csv file in the same directory. This will include details about the model architecture and the remaining individual discrimination. 

Parallelization for the 240 hyperparameter choices can be accomplished by running

```bash
cd benchmarks/adult
bash check_parallel.py
```

Next, run the following command to get the datapoints removed for minimum discrimination for each of the 240 settings.

```bash
cd benchmarks/adult
python methodology1.py
```

### For other baselines

#### Full

```bash
cd benchmarks/adult
python train_all_permutations.py --train=1 --full_baseline=1 --model_number=model_number
```

#### Sensitive Removal

Run

```bash
cd benchmarks/adult
python train_all_permutations_nosensitive.py --model_number=model_number
```

#### Disparate Impact Removed

```bash
cd benchmarks/adult/disparate_impact_removed
python run_hyperparams_disparate_impact_remover.py --model_number=model_number
```

#### Preferential sampling

```bash
cd benchmarks/adult/preferential_sampling
python train_preferential_sampling.py --model_number=model_number
```

#### Massaging

```bash
cd benchmarks/adult/massaging
python train_massage.py --model_number=model_number
```

Results will be accumulated in the same directory

#### Learning Fair Representations

```bash
cd benchmarks/adult/learning_fair_representations
python learning_fr.py --model_number=model_number
```

#### Adversarial Debiasing

For adversarial debiasing, we pass the data-permutation, not the model number

```bash
cd benchmarks/adult/adversarial_debiasing
python adversarial_debiasing.py --permutation=permutation
```

Each of the above approaches can be run for either "debiased test set" or "full test set", which can be passed as "debiased_test" flag to the files, e.g.:

```bash
python train_all_permutations.py --train=0 --model_number=model_number --percentage_removal=x --debiased_test=0
```