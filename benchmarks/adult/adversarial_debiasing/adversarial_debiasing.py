import sys, os
sys.path.insert(1, "../")  
sys.path.append("../../../")
sys.path.append("../../../competitors/AIF360/")

import numpy as np
np.random.seed(0)

from aif360.datasets import MyAdultDataset, BinaryLabelDataset, StructuredDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric

from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
import load_adult_income as load_file
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("--debiased_test", type=int, default=1,
#                     help="Use debiased test for test accuracy")
# parser.add_argument("--permutation", type=int, default=0,
#                     help="Which model number to run (out of 240)")
# args = parser.parse_args()

dist = load_file.dist
perm = int(sys.argv[1])
# perm = args.permutation
ordering = load_file.permutations(perm)
biased_test_points = np.load(f"{os.path.dirname(os.path.realpath(__file__))}/../../adult/adult_biased_points.npy")
debiased_test = bool(int(sys.argv[2]))
# debiased_test = bool(args.debiased_test)

dataset_orig = MyAdultDataset(
    protected_attribute_names=['sex'],                   
    privileged_classes=[[1]], 
    normalized = False,
    permute = perm   
)

train_examples = 36000
dataset_orig_train, dataset_orig_test = dataset_orig.split([train_examples], shuffle=False)
assert(len(dataset_orig_train.convert_to_dataframe()[0]) == train_examples)

if debiased_test:
    test_points = np.array(ordering[train_examples:])
    mask = np.in1d(test_points, biased_test_points)		# True if the point is biased
    mask_new = ~mask
    x = mask_new.astype(int).nonzero()[0]
    dataset_orig_test = dataset_orig_test.subset(x)
    assert(len(dataset_orig_test.convert_to_dataframe()[0]) < 45222 - train_examples)
else:
    assert(len(dataset_orig_test.convert_to_dataframe()[0]) == 45222 - train_examples)

privileged_groups = [{'sex': 1}]
unprivileged_groups = [{'sex': 0}]

min_max_scaler = MaxAbsScaler()
dataset_orig_train.features = min_max_scaler.fit_transform(dataset_orig_train.features)
dataset_orig_test.features = min_max_scaler.transform(dataset_orig_test.features)
sess = tf.Session()

debiased_model = AdversarialDebiasing(privileged_groups = privileged_groups,
                          unprivileged_groups = unprivileged_groups,
                          scope_name='debiased_classifier',
                          debias=True,
                          sess=sess, num_epochs=200)

debiased_model.fit(dataset_orig_train)

dataset_debiasing_train = debiased_model.predict(dataset_orig_train)
dataset_debiasing_test = debiased_model.predict(dataset_orig_test)
classified_metric_debiasing_test = ClassificationMetric(dataset_orig_test, 
                                                 dataset_debiasing_test,
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)

classified_metric_debiasing_train = ClassificationMetric(dataset_orig_train, 
                                                 dataset_debiasing_train,
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)

train_acc = classified_metric_debiasing_train.accuracy()
test_acc = classified_metric_debiasing_test.accuracy()
diff = abs(classified_metric_debiasing_test.statistical_parity_difference())

def find_discm_examples(class0_data, class1_data, print_file, scheme):
        import pandas as pd
        assert class0_data.shape[0] == class1_data.shape[0]

        cols = ['age','workclass','fnlwgt','education','marital-status','occupation',
        'race','sex','capitalgain','capitalloss','hoursperweek','native-country']
        df0 = pd.DataFrame(data=class0_data, columns=cols, dtype='float')
        df0['target'] = 0
        df0_binary = BinaryLabelDataset(df=df0, label_names=['target'], protected_attribute_names=['sex'])
        df0_pred = debiased_model.predict(df0_binary)

        df1 = pd.DataFrame(data=class1_data, columns=cols, dtype='float')
        df1['target'] = 0
        df1_binary = BinaryLabelDataset(df=df1, label_names=['target'], protected_attribute_names=['sex'])
        df1_pred = debiased_model.predict(df1_binary)

        assert(not np.all(df0_binary.labels))       # all of them should be 0
        assert(not np.all(df1_binary.labels))

        predictions_class0 = df0_pred.labels
        predictions_class1 = df1_pred.labels
        
        return sum(predictions_class0 != predictions_class1)[0]    # Gives the number of discriminating examples

sys.path.append("../../../scripts/")

from find_discm_points import entire_test_suite
class0_data, class1_data = entire_test_suite(mini=False, disparateremoved=False)     # False means loads entire data
num_dicsm = find_discm_examples(class0_data, class1_data, print_file=False, scheme=8)
size = class0_data.shape[0]/100
print("Discrimination:", num_dicsm)
dataset = "adult"
if debiased_test:
    with open(f"results_adversarial_debiased_{dataset}_dist{dist}.csv", "a") as f:
        f.write(f'{train_acc},{test_acc},{perm},{diff},{num_dicsm},{num_dicsm/size}\n')
else:
    with open(f"results_adversarial_debiased_{dataset}_fulltest_dist{dist}.csv", "a") as f:
        f.write(f'{train_acc},{test_acc},{perm},{diff},{num_dicsm},{num_dicsm/size}\n')

