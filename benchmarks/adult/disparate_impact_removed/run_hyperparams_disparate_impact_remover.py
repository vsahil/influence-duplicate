import numpy as np
import IPython, sys, os
sys.path.append("../")
sys.path.append("../../../")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import influence.experiments as experiments
from influence.fully_connected import Fully_Connected

from load_adult_income import disparate_removed_load_adult_income, dist
from find_discm_points import entire_test_suite
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("--debiased_test", type=int, default=1,
#                     help="Use debiased test for test accuracy")
# parser.add_argument("--model_number", type=int, default=0,
#                     help="Which model number to run (out of 240)")
# args = parser.parse_args()

input_dim = 12
weight_decay = 0.001

initial_learning_rate = 1e-5 
decay_epochs = [30000, 40000]
num_classes = 2
keep_probs = [1.0, 1.0]
num_steps = 20000

scheme = 8
assert(scheme == 8)     # now always

setting_now = int(sys.argv[1])
# setting_now = args.model_number

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
damping = 3e-2
debiased_test = bool(int(sys.argv[2]))
# debiased_test = bool(args.debiased_test)

data_sets, mins_and_ranges = disparate_removed_load_adult_income(perm = perm, debiased_test=debiased_test)

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
    damping=damping,
    decay_epochs=decay_epochs,
    mini_batch=True,
    train_dir=f'throw/output_dont_save{model_count}', 
    log_dir=f'throw/log{model_count}',
    hvp_files = f"inverse_HVP_schm{scheme}_count{model_count}",
    model_name=f"adult_income_count{model_count}",
    scheme = f"{scheme}"
    )

model.train(num_steps=num_steps, iter_to_switch_to_batch=10000000, iter_to_switch_to_sgd=20000, save_checkpoints=False, verbose=False)

class0_data, class1_data = entire_test_suite(mini=False, 
                disparateremoved=True, mins_and_ranges=mins_and_ranges)     # False means loads entire data
num_dicsm = model.find_discm_examples(class0_data, class1_data, print_file=False, scheme=scheme)
# train_acc, test_acc = model.print_model_eval()

sensitive_attr = 7
train_acc, test_acc, test_predictions = model.print_model_eval()
import sklearn, math
assert len(np.unique(data_sets.test.x[:, sensitive_attr])) == 2
class0_index = (data_sets.test.x[:, sensitive_attr] == 0).astype(int).nonzero()[0]
class1_index = (data_sets.test.x[:, sensitive_attr] == 1).astype(int).nonzero()[0]
test_predictions = np.argmax(test_predictions, axis=1)
class0_pred = test_predictions[class0_index]
class1_pred = test_predictions[class1_index]
class0_truth = data_sets.test.labels[class0_index]
class1_truth = data_sets.test.labels[class1_index]
assert(len(class0_pred) + len(class1_pred) == len(test_predictions))
assert(len(class0_truth) + len(class1_truth) == len(data_sets.test.labels))

class0_cm = sklearn.metrics.confusion_matrix(class0_truth, class0_pred, labels=[0,1])
class1_cm = sklearn.metrics.confusion_matrix(class1_truth, class1_pred, labels=[0,1])
tn, fp, fn, tp = class0_cm.ravel()
class0_fpr = fp / (fp + tn)
class0_fnr = fn / (fn + tp)
class0_pos = (tp + fp) / len(class0_index)        # proportion that got positive outcome
del tn, fp, fn, tp
tn, fp, fn, tp = class1_cm.ravel()
class1_fpr = fp / (fp + tn)
class1_fnr = fn / (fn + tp)
class1_pos = (tp + fp) / len(class1_index)        # proportion that got positive outcome

print("Discrimination:", num_dicsm)
size = class0_data.shape[0]/100
dataset = "adult"

if debiased_test:
    with open(f"results_disparate_removed_{dataset}_dist{dist}.csv", "a") as f:
        print(f"{model_count},{h1units},{h2units},{batch},{perm},{train_acc},{test_acc},{class0_fpr},{class0_fnr},{class0_pos},{class1_fpr},{class1_fnr},{class1_pos},{num_dicsm},{num_dicsm/size}", file=f)
else:
    with open(f"results_disparate_removed_{dataset}_fulltest_dist{dist}.csv", "a") as f:
        print(f"{model_count},{h1units},{h2units},{batch},{perm},{train_acc},{test_acc},{class0_fpr},{class0_fnr},{class0_pos},{class1_fpr},{class1_fnr},{class1_pos},{num_dicsm},{num_dicsm/size}", file=f)
