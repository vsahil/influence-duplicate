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

from load_german_credit import load_german_credit_nosensitive
from find_discm_points import entire_test_suite

train = True

input_dim = 20
weight_decay = 0.001
# batch_size = 3000

initial_learning_rate = 1e-4
decay_epochs = [60000, 70000]
# hidden1_units = 24
# hidden2_units = 12
# hidden3_units = 0
num_classes = 2
keep_probs = [1.0, 1.0]

scheme = 8      # ask the model which points are responsible for the current set of predictions
# assert(scheme == 8)     # now always

setting_now = int(sys.argv[1])

def variation(setting_now):
    model_count = 0
    for perm in range(20):
        for h1units in [16, 24, 32]:
            for h2units in [8, 12]:
                for batch in [50, 100]:     # different batch sizes for this dataset
                    if model_count < setting_now:
                        model_count += 1
                        continue
                    # print(setting_now, "done", perm, h1units, h2units, batch)
                    return perm, h1units, h2units, batch, model_count


perm, h1units, h2units, batch, model_count = variation(setting_now)
assert(model_count == setting_now)
data_sets = load_german_credit_nosensitive(perm)

hidden1_units = h1units
hidden2_units = h2units
hidden3_units = 0
batch_size = batch

print("Start: ", model_count, " Setting: ", perm, hidden1_units, hidden2_units, batch_size)

name = f"german_credit_count{model_count}"
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
    damping=5e-2,
    decay_epochs=decay_epochs,
    mini_batch=True,
    train_dir=f'trained_models_scheme{scheme}_nosensitive/output_count{model_count}', 
    log_dir=f'throw/log{model_count}',
    hvp_files = f"HVP_files_scheme{scheme}_nosensitive/inverse_HVP_schm{scheme}_count{model_count}",
    model_name=name,
    scheme = f"{scheme}")


num_steps = 10000
if train:
    model.train(num_steps=num_steps, iter_to_switch_to_batch=10000000, iter_to_switch_to_sgd=20000, save_checkpoints=False, verbose=False)

class0_data, class1_data = entire_test_suite(mini=False)     # False means loads entire data
initial_num = model.find_discm_examples(class0_data, class1_data, print_file=False, scheme=scheme)
print(initial_num, "See")
size = class0_data.shape[0]/100
with open("results_german_nosensitive.csv".format(scheme), "a") as f:
     f.write(f"{model_count},{perm},{h1units},{h2units},{batch},{initial_num},{initial_num/size}\n")

exit(0)

p = int(sys.argv[2])
percentage = p/8.0
size = class0_data.shape[0]/100
# for percentage in np.linspace(removal-1, removal-0.2, 5):
tf.reset_default_graph()
# p = int(36000 * percentage / 100)
remaining_indexes = np.array(sorted_training_points[p:])
data_sets_partial = load_german_credit_partial(perm=perm, index=remaining_indexes)
try:
    assert(len(remaining_indexes) == 800 - p)
    assert(data_sets_partial.train.num_examples == 800 - p)
except:
    print(p, percentage, data_sets_partial.train.num_examples, "hello")
    assert False
model_partial_data = Fully_Connected(
    input_dim=input_dim, 
    hidden1_units=hidden1_units, 
    hidden2_units=hidden2_units,
    hidden3_units=hidden3_units,
    weight_decay=weight_decay,
    num_classes=num_classes, 
    batch_size=batch_size,
    data_sets=data_sets_partial,
    initial_learning_rate=initial_learning_rate,
    damping=1e-2,
    decay_epochs=decay_epochs,
    mini_batch=False,
    train_dir='logs/output_partial', 
    log_dir='logs/log_partial',
    hvp_files = "inverse_HVP_scheme1_",
    model_name='german_credit_partial',
    scheme = "scheme9_par")
print("Training")
print("Percentage: ", percentage, " Points removed: ", p) 
model_partial_data.train(num_steps=num_steps, iter_to_switch_to_batch=10000000, iter_to_switch_to_sgd=20000, save_checkpoints=False, verbose=False)
train_acc, test_acc = model_partial_data.print_model_eval()
print("Percentage: ", percentage, " Points removed: ", p)
num = model_partial_data.find_discm_examples(class0_data, class1_data, print_file=False, scheme=scheme)
with open("results_german_scheme{}.csv".format(scheme), "a") as f:
    f.write(f"{model_count},{perm},{h1units},{h2units},{batch},{train_acc},{test_acc},{percentage},{p},{num},{num/size}\n")     # the last ones gives percentage of discrimination
os.system("rm -rf logs/log_partial")
del model_partial_data          # to remove any chance of reusing variables and reduce memory


