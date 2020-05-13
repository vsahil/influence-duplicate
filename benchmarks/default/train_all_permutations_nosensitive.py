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

from load_default import load_default_nosensitive
from find_discm_points import entire_test_suite

input_dim = 23 - 1
weight_decay = 0.001
# batch_size = 3000

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

data_sets = load_default_nosensitive(perm)


print("Start: ", model_count, " Setting: ", perm, hidden1_units, hidden2_units, batch_size)

name = f"default_{model_count}"
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
    train_dir=f'trained_models_nosensitive/output_count{model_count}', 
    log_dir=f'throw/log{model_count}',
    hvp_files = f"inverse_HVP_default_nosensitive/inverse_HVP_schm{scheme}_count{model_count}",
    model_name=name,
    scheme = f"{scheme}")


model.train(num_steps=num_steps, iter_to_switch_to_batch=10000000, 
iter_to_switch_to_sgd=20000, save_checkpoints=False, verbose=False, plot_loss=False)
train_acc, test_acc = model.print_model_eval()

# class0_data, class1_data = entire_test_suite(mini=False)     # False means loads entire data
# initial_num = model.find_discm_examples(class0_data, class1_data, print_file=False, scheme=scheme)
# size = class0_data.shape[0]/100
with open("results_default_nosensitive.csv".format(scheme), "a") as f:
     f.write(f"{model_count},{perm},{h1units},{h2units},{batch},{train_acc},{test_acc}\n")

