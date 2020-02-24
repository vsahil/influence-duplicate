import sys
sys.path.insert(1, "../")  

import numpy as np
np.random.seed(0)

from tqdm import tqdm
from aif360.datasets import GermanDataset
from aif360.metrics import BinaryLabelDatasetMetric
# from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.preprocessing import DisparateImpactRemover
from sklearn.preprocessing import MinMaxScaler

dataset_orig = GermanDataset(
    protected_attribute_names=['sex'],                   
    privileged_classes=[['male']],      
    features_to_drop=['personal_status'] 
)

scaler = MinMaxScaler(copy=False)
dataset_orig_train, dataset_orig_test = dataset_orig.split([1.0], shuffle=False)

# dataset_orig_train.features = scaler.fit_transform(dataset_orig_train.features)
# test.features = scaler.fit_transform(dataset_orig_test.features)

# index = dataset_orig_train.feature_names.index('sex')

# privileged_groups = [{'sex': 1}]
# unprivileged_groups = [{'sex': 0}]
# import ipdb; ipdb.set_trace()
# print("ouv")
# metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train, 
#                                              unprivileged_groups=unprivileged_groups,
#                                              privileged_groups=privileged_groups)

# print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())

DIs = []
for level in tqdm(np.linspace(1., 1., 1)):
    di = DisparateImpactRemover(repair_level=level)
    train_repd = di.fit_transform(dataset_orig_train)
    # import ipdb; ipdb.set_trace()
    print("hello", level)
    # test_repd = di.fit_transform(dataset_orig_test)
    
    # X_tr = np.delete(train_repd.features, index, axis=1)
    # X_te = np.delete(test_repd.features, index, axis=1)
    # y_tr = train_repd.labels.ravel()
    
    # lmod = LogisticRegression(class_weight='balanced', solver='liblinear')
    # lmod.fit(X_tr, y_tr)
    
    # test_repd_pred = test_repd.copy()
    # test_repd_pred.labels = lmod.predict(X_te)

    # p = [{protected: 1}]
    # u = [{protected: 0}]
    # cm = BinaryLabelDatasetMetric(test_repd_pred, privileged_groups=p, unprivileged_groups=u)
    # DIs.append(cm.disparate_impact())

with open("see.csv", "w") as f:
    new_df = train_repd.convert_to_dataframe()[0].to_csv(f, index=False)

# RW = Reweighing(unprivileged_groups=unprivileged_groups,
#                 privileged_groups=privileged_groups)
# dataset_transf_train = RW.fit_transform(dataset_orig_train)

# metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train, 
#                                                unprivileged_groups=unprivileged_groups,
#                                                privileged_groups=privileged_groups)
# display(Markdown("#### Transformed training dataset"))
# import ipdb; ipdb.set_trace()
# print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_transf_train.mean_difference())

# with open("see.csv", "w") as f:
    # new_df = dataset_transf_train.convert_to_dataframe()[0].to_csv(f, index=False)
