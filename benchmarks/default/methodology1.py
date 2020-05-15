from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import numpy as np
import IPython, sys, os
sys.path.append(".")
sys.path.append("../../")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import influence.experiments as experiments
from influence.fully_connected import Fully_Connected

from load_default import load_default_partial_method1
from find_discm_points import entire_test_suite

input_dim = 23
weight_decay = 0.001

initial_learning_rate = 1e-4 
decay_epochs = [40000, 50000]
num_steps = 10000
num_classes = 2
keep_probs = [1.0, 1.0]

scheme = 8
assert(scheme == 8)     # now always

setting_now = int(sys.argv[1])

def variation(setting_now):
    model_count = 0
    for perm in range(20):
        for h1units in [16, 24, 32]:
            for h2units in [8, 12]:
                for batch in [2048, 4096]:      # different batch sizes for this dataset
                    if model_count < setting_now:
                        model_count += 1
                        continue
                    # print(setting_now, "done", perm, h1units, h2units, batch)
                    return perm, h1units, h2units, batch, model_count


perm, h1units, h2units, batch, model_count = variation(setting_now)
assert(model_count == setting_now)
hidden1_units = h1units
hidden2_units = h2units
hidden3_units = 0
batch_size = batch

name = f"default_{model_count}"

import pandas as pd
dataset = "default"
df = pd.read_csv(f"results_{dataset}_debiasedtrain_80percentof_total.csv")
removal_df = df.sort_values(by=['Discm_percent', 'Points-removed']).groupby("Model-count", as_index=False).first()
assert len(removal_df) == 240
train_pts_removed = removal_df.loc[removal_df['Model-count'] == model_count, 'Points-removed'].values
assert len(train_pts_removed) == 1
data_sets = load_default_partial_method1(perm=perm, model_count=model_count, train_pts_removed=train_pts_removed[0], name=name)
print("Training points removed:", train_pts_removed[0])
training_size = 24000
percentage = train_pts_removed[0]*100.0/training_size
print("Start: ", model_count, " Setting: ", perm, hidden1_units, hidden2_units, batch_size)

model = Fully_Connected(
    input_dim=input_dim, 
    hidden1_units=hidden1_units, 
    hidden2_units=hidden2_units,
    hidden3_units=hidden3_units,
    weight_decay=weight_decay,
    num_classes=num_classes, 
    batch_size=batch_size,
    data_sets=data_sets,
    initial_learning_rate=initial_learning_rate,
    damping=3e-2,
    decay_epochs=decay_epochs,
    mini_batch=True,
    train_dir=f'trained_models_try', 
    log_dir=f'throw',
    hvp_files = f'inverse_HVP_student_try',
    model_name=name,
    scheme = f"{scheme}")


model.train(num_steps=num_steps, iter_to_switch_to_batch=10000000, 
    iter_to_switch_to_sgd=20000, save_checkpoints=False, verbose=False)

class0_data, class1_data = entire_test_suite(mini=False)     # False means loads entire data
size = class0_data.shape[0]/100.0
num = model.find_discm_examples(class0_data, class1_data, print_file=False, scheme=scheme)

sensitive_attr = 1
assert len(np.unique(data_sets.test.x[:, sensitive_attr])) == 2
class0_index = (data_sets.test.x[:, sensitive_attr] == 0).astype(int).nonzero()[0]
class1_index = (data_sets.test.x[:, sensitive_attr] == 1).astype(int).nonzero()[0]
train_acc, test_acc, test_predictions = model.print_model_eval()
test_predictions = np.argmax(test_predictions, axis=1)
class0_pred = test_predictions[class0_index]
class1_pred = test_predictions[class1_index]
class0_truth = data_sets.test.labels[class0_index]
class1_truth = data_sets.test.labels[class1_index]
assert(len(class0_pred) + len(class1_pred) == len(test_predictions))
assert(len(class0_truth) + len(class1_truth) == len(data_sets.test.labels))
import sklearn
class0_cm = sklearn.metrics.confusion_matrix(class0_truth, class0_pred)
class1_cm = sklearn.metrics.confusion_matrix(class1_truth, class1_pred)
tn, fp, fn, tp = class0_cm.ravel()
class0_fpr = fp / (fp + tn)
class0_fnr = fn / (fn + tp)
del tn, fp, fn, tp
tn, fp, fn, tp = class1_cm.ravel()
class1_fpr = fp / (fp + tn)
class1_fnr = fn / (fn + tp)

with open(f"results_{dataset}_method1.csv".format(scheme), "a") as f:
    print(f"{model_count},{perm},{h1units},{h2units},{batch},{train_acc},{test_acc},{class0_fpr},{class0_fnr},{class1_fpr},{class1_fnr},{percentage},{train_pts_removed[0]},{num},{num/size}", file=f)

