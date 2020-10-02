# Removing biased data to improve fairness and accuracy

This is a modified clone of [Understanding Black-box Predictions via Influence Functions](https://github.com/kohpangwei/influence-release) repository

## Dependencies

Installation of dependencies:
```bash
pip3 install -r requirement.txt
```

## Replication of tables and plots in the paper

### Tables

```bash
cd benchmarks
python3 faceted_box_plots_all_datasets.py --plot=0 --parity=1
cd tables
bash post-process.sh
```

### Plots

```bash
cd benchmarks
python3 faceted_box_plots_all_datasets.py --plot=1
```

## Repository Layout

- random_fairness_test_generator directory contains the code for generation of pairs of similar individuals. We have not added that in the code because the files were huge.
- For each dataset (six) we used, the raw input data along with pre-processing scripts are present in their respective dataset directories, e.g. : `adult-dataset`, `default-dataset` etc.
- The baselines that have been taken from [AIF360](https://github.com/IBM/AIF360) are in the `competitors` directory.
- The code for using our approach in any experiment is inside the `benchmarks` directory. In the `benchmarks` directory, there are separate directories for each experiment, for eg. `adult`, `default` etc.
- Each of the baselines are present as separate sub-directories inside respective experimental directories. For example, `adult/disparate_impact_removed`, `adult/preferential_sampling`, `adult/massaging`, `adult/learning_fair_representations`, and `adult/adversarial_debiasing`. Sensitive removal (SR) and Full don't have a separate sub-directory.
- The results for our approach, SR and Full are generated in the experimental directory itself. The results for all other baselines are generated in their respective sub-directories.

## Replicating Experiments

### Installing AIF360 for baselines

Please clone AIF360 in the `competitors` directory, and git reset to a particular commit. After that, replace the `dataset` sub-directory with `aif360datasets` directory. This will copy the changes we made to AIF360 to support all the datasets we used.

```bash
mkdir competitors
cd competitors
git clone https://github.com/IBM/AIF360
cd AIF360
git reset --hard 10026100884ddb8620d88112c36619d6f65a4666
pip3 install -r requirements.txt
cp -r ../../../aif360datasets datasets
```

### Generating pairs of similar individuals for a dataset (for example adult)

For generating pairs of similar individuals:

```bash
cd random_fairness_test_generator
python3 adult_main.py
```

For a new dataset, one can uncomment the first function in adult_main.py to generate settings file for the dataset.

### Running the approach for a dataset (for example adult)

If you want to train the model with model sequence "model_number", you should run:

```bash
cd benchmarks/adult
python3 train_all_permutations.py --train=1 --model_number=model_number
```

Since measuring of test accuracy needs to removed biased points, it can only be performed after all 240 models have been trained, by switching "Train" to False and adding percentage removal.

```bash
cd benchmarks/adult
python3 train_all_permutations.py --train=0 --model_number=model_number --percentage_removal=x
```

Results will be accumutated in the file results_adult_debiasedtrain_80percentof_total.csv file in the same directory. This will include details about the model architecture and the remaining individual discrimination. 

Parallelization for the 240 hyperparameter choices can be accomplished by running

```bash
cd benchmarks/adult
bash check_parallel.py
```

The "unfair" points in the dataset which is generated using `common_biased_points.py` which generates `dataset_biased_point.npy` file using the points removed for minimum discrimination. This is used to remove "unfair" points from the test set of a model, when measuring test accuracy.

```bash
cd benchmarks/adult
python3 common_biased_points.py
```

Next, run the following command to get the datapoints removed for minimum discrimination for each of the 240 settings and the test accuracy of the model with "unfair" points removed.

```bash
cd benchmarks/adult
python3 methodology1.py
```


### For other baselines

#### Full

```bash
cd benchmarks/adult
python3 train_all_permutations.py --train=1 --full_baseline=1 --model_number=model_number
```

#### Sensitive Removal

```bash
cd benchmarks/adult
python3 train_all_permutations_nosensitive.py --model_number=model_number
```

#### Disparate Impact Removed

```bash
cd benchmarks/adult/disparate_impact_removed
python3 run_hyperparams_disparate_impact_remover.py --model_number=model_number
```

#### Preferential sampling

```bash
cd benchmarks/adult/preferential_sampling
python3 train_preferential_sampling.py --model_number=model_number
```

#### Massaging

```bash
cd benchmarks/adult/massaging
python3 train_massage.py --model_number=model_number
```

Results will be accumulated in the same directory

#### Learning Fair Representations

```bash
cd benchmarks/adult/learning_fair_representations
python3 learning_fr.py --model_number=model_number
```

#### Adversarial Debiasing

For adversarial debiasing, we pass the data-permutation, not the model number

```bash
cd benchmarks/adult/adversarial_debiasing
python3 adversarial_debiasing.py --permutation=permutation
```

Each of the above approaches can be run for either "debiased test set" or "full test set", which can be passed as "debiased_test" flag to the files, e.g.:

```bash
python3 train_all_permutations.py --train=0 --model_number=model_number --percentage_removal=x --debiased_test=0
```
