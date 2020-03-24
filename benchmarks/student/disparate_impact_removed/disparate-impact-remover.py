import sys
sys.path.insert(1, "../")
sys.path.append("../../../competitors/AIF360/") 

import numpy as np
np.random.seed(0)

from tqdm import tqdm
from aif360.datasets import StudentDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import DisparateImpactRemover
from sklearn.preprocessing import MinMaxScaler

dataset_orig = StudentDataset(
    protected_attribute_names=['sex'],                   
    privileged_classes=[[1]], 
    normalized = False    
)

scaler = MinMaxScaler(copy=False)
dataset_orig_train, dataset_orig_test = dataset_orig.split([1.0], shuffle=False)

di = DisparateImpactRemover(repair_level=1.0)
train_repd = di.fit_transform(dataset_orig_train)
new_df = train_repd.convert_to_dataframe()[0]

write = True
if write:
    with open("disparate_impact_removed_student.csv", "w") as f:
        new_df.to_csv(f, index=False)

new_df = new_df.drop(columns=['G3'])
mins_and_ranges = []
for j in list(new_df):
    i = new_df[j]
    mins_and_ranges.append((np.min(i), np.max(i) - np.min(i)))
print(mins_and_ranges, len(mins_and_ranges))
