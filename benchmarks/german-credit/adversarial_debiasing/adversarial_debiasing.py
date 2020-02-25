import sys
sys.path.insert(1, "../")  

import numpy as np
np.random.seed(0)

from aif360.datasets import GermanDataset, BinaryLabelDataset, StructuredDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
# from aif360.algorithms.preprocessing import Reweighing

from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf

dataset_orig = GermanDataset(
    protected_attribute_names=['sex'],                   
    privileged_classes=[['male']],      
    features_to_drop=['personal_status'] 
)


dataset_orig_train, dataset_orig_test = dataset_orig.split([0.8], shuffle=False)

privileged_groups = [{'sex': 1}]
unprivileged_groups = [{'sex': 0}]

metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)

print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())
metric_orig_test = BinaryLabelDatasetMetric(dataset_orig_test, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_test.mean_difference())



min_max_scaler = MaxAbsScaler()
dataset_orig_train.features = min_max_scaler.fit_transform(dataset_orig_train.features)
dataset_orig_test.features = min_max_scaler.transform(dataset_orig_test.features)
metric_scaled_train = BinaryLabelDatasetMetric(dataset_orig_train, 
                             unprivileged_groups=unprivileged_groups,
                             privileged_groups=privileged_groups)

print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_scaled_train.mean_difference())
metric_scaled_test = BinaryLabelDatasetMetric(dataset_orig_test, 
                             unprivileged_groups=unprivileged_groups,
                             privileged_groups=privileged_groups)
print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_scaled_test.mean_difference())

# sess = tf.Session()
# plain_model = AdversarialDebiasing(privileged_groups = privileged_groups,
#                           unprivileged_groups = unprivileged_groups,
#                           scope_name='plain_classifier',
#                           debias=False,
#                           sess=sess)
# plain_model.fit(dataset_orig_train)
# dataset_nodebiasing_train = plain_model.predict(dataset_orig_train)
# dataset_nodebiasing_test = plain_model.predict(dataset_orig_test)

# metric_dataset_nodebiasing_train = BinaryLabelDatasetMetric(dataset_nodebiasing_train, 
#                                              unprivileged_groups=unprivileged_groups,
#                                              privileged_groups=privileged_groups)

# print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_nodebiasing_train.mean_difference())

# metric_dataset_nodebiasing_test = BinaryLabelDatasetMetric(dataset_nodebiasing_test, 
#                                              unprivileged_groups=unprivileged_groups,
#                                              privileged_groups=privileged_groups)

# print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_nodebiasing_test.mean_difference())


# classified_metric_nodebiasing_test = ClassificationMetric(dataset_orig_test, 
#                                                  dataset_nodebiasing_test,
#                                                  unprivileged_groups=unprivileged_groups,
#                                                  privileged_groups=privileged_groups)
# classified_metric_nodebiasing_train = ClassificationMetric(dataset_orig_train, 
#                                                  dataset_nodebiasing_train,
#                                                  unprivileged_groups=unprivileged_groups,
#                                                  privileged_groups=privileged_groups)
# print("Test set: Classification accuracy = %f" % classified_metric_nodebiasing_test.accuracy())
# TPR = classified_metric_nodebiasing_test.true_positive_rate()
# TNR = classified_metric_nodebiasing_test.true_negative_rate()
# bal_acc_nodebiasing_test = 0.5*(TPR+TNR)
# print("Test set: Balanced classification accuracy = %f" % bal_acc_nodebiasing_test)
# print("Test set: Disparate impact = %f" % classified_metric_nodebiasing_test.disparate_impact())
# print("Test set: Equal opportunity difference = %f" % classified_metric_nodebiasing_test.equal_opportunity_difference())
# print("Test set: Average odds difference = %f" % classified_metric_nodebiasing_test.average_odds_difference())
# print("Test set: Theil_index = %f" % classified_metric_nodebiasing_test.theil_index())
# sess.close()
# train_acc = classified_metric_nodebiasing_train.accuracy()
# test_acc = classified_metric_nodebiasing_test.accuracy()
# print(train_acc, test_acc, "hello")

# exit(0)
# tf.reset_default_graph()

sess = tf.Session()

debiased_model = AdversarialDebiasing(privileged_groups = privileged_groups,
                          unprivileged_groups = unprivileged_groups,
                          scope_name='debiased_classifier',
                          debias=True,
                          sess=sess)

debiased_model.fit(dataset_orig_train)

dataset_debiasing_train = debiased_model.predict(dataset_orig_train)
dataset_debiasing_test = debiased_model.predict(dataset_orig_test)

# print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_nodebiasing_train.mean_difference())
# print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_nodebiasing_test.mean_difference())

# Metrics for the dataset from model with debiasing
print("#### Model - with debiasing - dataset metrics")
metric_dataset_debiasing_train = BinaryLabelDatasetMetric(dataset_debiasing_train, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)

print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_debiasing_train.mean_difference())

metric_dataset_debiasing_test = BinaryLabelDatasetMetric(dataset_debiasing_test, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)

print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_debiasing_test.mean_difference())



# print("#### Plain model - without debiasing - classification metrics")
# print("Test set: Classification accuracy = %f" % classified_metric_nodebiasing_test.accuracy())
# TPR = classified_metric_nodebiasing_test.true_positive_rate()
# TNR = classified_metric_nodebiasing_test.true_negative_rate()
# bal_acc_nodebiasing_test = 0.5*(TPR+TNR)
# print("Test set: Balanced classification accuracy = %f" % bal_acc_nodebiasing_test)
# print("Test set: Disparate impact = %f" % classified_metric_nodebiasing_test.disparate_impact())
# print("Test set: Equal opportunity difference = %f" % classified_metric_nodebiasing_test.equal_opportunity_difference())
# print("Test set: Average odds difference = %f" % classified_metric_nodebiasing_test.average_odds_difference())
# print("Test set: Theil_index = %f" % classified_metric_nodebiasing_test.theil_index())


print("#### Model - with debiasing - classification metrics")
classified_metric_debiasing_test = ClassificationMetric(dataset_orig_test, 
                                                 dataset_debiasing_test,
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
print("Test set: Classification accuracy = %f" % classified_metric_debiasing_test.accuracy())
classified_metric_debiasing_train = ClassificationMetric(dataset_orig_train, 
                                                 dataset_debiasing_train,
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)

TPR = classified_metric_debiasing_test.true_positive_rate()
TNR = classified_metric_debiasing_test.true_negative_rate()
bal_acc_debiasing_test = 0.5*(TPR+TNR)
print("Test set: Balanced classification accuracy = %f" % bal_acc_debiasing_test)
print("Test set: Disparate impact = %f" % classified_metric_debiasing_test.disparate_impact())
print("Test set: Equal opportunity difference = %f" % classified_metric_debiasing_test.equal_opportunity_difference())
print("Test set: Average odds difference = %f" % classified_metric_debiasing_test.average_odds_difference())
print("Test set: Theil_index = %f" % classified_metric_debiasing_test.theil_index())
train_acc = classified_metric_debiasing_train.accuracy()
test_acc = classified_metric_debiasing_test.accuracy()
print(train_acc, test_acc, "hello")

def find_discm_examples(class0_data, class1_data, print_file, scheme):
        # assert False
        # import ipdb; ipdb.set_trace()
        import pandas as pd
        length = class0_data.shape[0]
        assert length == class1_data.shape[0]

        cols = ['status', 'month', 'credit_history', 'purpose', 'credit_amount', 'savings', 'employment', 'investment_as_income_percentage', 'sex', 'other_debtors', 'residence_since', 'property', 'age', 'installment_plans', 'housing', 'number_of_credits', 'skill_level', 'people_liable_for', 'telephone', 'foreign_worker']
        df0 = pd.DataFrame(data=class0_data, columns=cols, dtype='float')
        df0['credit'] = 0
        df0_binary = BinaryLabelDataset(df=df0, label_names=['credit'], protected_attribute_names=['sex'])
        df0_pred = debiased_model.predict(df0_binary)

        df1 = pd.DataFrame(data=class1_data, columns=cols, dtype='float')
        df1['credit'] = 0
        df1_binary = BinaryLabelDataset(df=df1, label_names=['credit'], protected_attribute_names=['sex'])
        df1_pred = debiased_model.predict(df1_binary)

        assert(not np.all(df0_binary.labels))       # all of them should be 0
        assert(not np.all(df1_binary.labels))

        # l_zero = np.zeros(length, dtype=np.int)
        # l_one = np.ones(length, dtype=np.int)
        
        # feed_dict_class0_label0 = fill_feed_dict_manual(class0_data, l_zero)
        # feed_dict_class0_label1 = fill_feed_dict_manual(class0_data, l_one)
        # feed_dict_class1_label0 = fill_feed_dict_manual(class1_data, l_zero)
        # feed_dict_class1_label1 = fill_feed_dict_manual(class1_data, l_one)
        
        # ops = [preds, indiv_loss_no_reg]
        
        # predictions_class0_, loss_class0_label_0 = sess.run(ops, feed_dict=feed_dict_class0_label0)
        # predictions_class0, loss_class0_label_1 = sess.run(ops, feed_dict=feed_dict_class0_label1)
        # assert (predictions_class0_ == predictions_class0).all()    #"""This is my belief"""
        # predictions_class1_, loss_class1_label_0 = sess.run(ops, feed_dict=feed_dict_class1_label0)
        # predictions_class1, loss_class1_label_1 = sess.run(ops, feed_dict=feed_dict_class1_label1)
        # assert (predictions_class1_ == predictions_class1).all()    #"""This is my belief"""

        predictions_class0 = df0_pred.labels
        predictions_class1 = df1_pred.labels
        
        num_discriminating = sum(predictions_class0 != predictions_class1)    # Gives the number of discriminating examples
        print("Number of discriminating examples: ", num_discriminating)
        return num_discriminating

sys.path.append("../../../scripts/")

from find_discm_points import entire_test_suite
class0_data, class1_data = entire_test_suite(mini=False, reweighted_german=False, disparateremoved=False)     # False means loads entire data
num_dicsm = find_discm_examples(class0_data, class1_data, print_file=False, scheme=8)
# train_acc, test_acc = model.print_model_eval()

print("Discrimination:", num_dicsm)

with open("adversarial_debiased_german_discrimination.csv", "a") as f:
    f.write(f'{train_acc},{test_acc},{num_dicsm}\n')