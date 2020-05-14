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

from load_compas_score_as_labels import load_recidivism_groundtruth_as_test_our_approach, load_recidivism_groundtruth_as_test_full, load_recidivism_groundtruth_as_test_sensitive_removed
from find_discm_points import entire_test_suite

input_dim = 10
weight_decay = 0.002
# batch_size = 3000

initial_learning_rate = 1e-3 
decay_epochs = [20000, 30000]
num_steps = 15000
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
                for batch in [512, 1048]:      # different batch sizes for this dataset
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

our_approach = False
noremoval = False
sensitive_removed = True
debiased_real_accuracy = True

if our_approach:
    data_sets = load_recidivism_groundtruth_as_test_our_approach(perm, debiased_real_accuracy=debiased_real_accuracy)
    print("Start: ", model_count, " Setting: ", perm, hidden1_units, hidden2_units, batch_size)
    tf.reset_default_graph()
    name = f"compas_two_year{model_count}"
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
        train_dir=f'trained_models_method1/output_count{model_count}', 
        log_dir=f'throw/log{model_count}',
        hvp_files = f"inverse_HVP_compas_method1/inverse_HVP_schm{scheme}_count{model_count}",
        model_name=name,
        scheme = f"{scheme}")

    # We are not training the models, directly evaluating them. 
    train_acc, test_acc = model.print_model_eval()
    if debiased_real_accuracy:
        with open(f"results_our_real_accuracy_debiased.csv", "a") as f:
            f.write(f"{model_count},{perm},{h1units},{h2units},{batch},{test_acc}\n")     # the last ones gives percentage of discrimination
    else:
        with open(f"results_our_real_accuracy_full.csv", "a") as f:
            f.write(f"{model_count},{perm},{h1units},{h2units},{batch},{test_acc}\n")     # the last ones gives percentage of discrimination

    del model


if noremoval:
    data_sets = load_recidivism_groundtruth_as_test_full(perm, debiased_real_accuracy=debiased_real_accuracy)
    print("Start: ", model_count, " Setting: ", perm, hidden1_units, hidden2_units, batch_size)
    tf.reset_default_graph()
    name = f"compas_two_year{model_count}"
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
        train_dir=f'trained_models_try/output_count{model_count}', 
        log_dir=f'throw/log{model_count}',
        hvp_files = f"inverse_HVP_compas_try/inverse_HVP_schm{scheme}_count{model_count}",
        model_name=name,
        scheme = f"{scheme}")

    model.train(num_steps=num_steps, iter_to_switch_to_batch=10000000, iter_to_switch_to_sgd=20000, save_checkpoints=False, verbose=False)
    train_acc, test_acc = model.print_model_eval()
    if debiased_real_accuracy:
        with open(f"results_noremoval_real_accuracy_debiased.csv", "a") as f:
            f.write(f"{model_count},{perm},{h1units},{h2units},{batch},{test_acc}\n")     # the last ones gives percentage of discrimination
    else:
        with open(f"results_noremoval_real_accuracy_full.csv", "a") as f:
            f.write(f"{model_count},{perm},{h1units},{h2units},{batch},{test_acc}\n")     # the last ones gives percentage of discrimination
        
    del model


if sensitive_removed:
    data_sets = load_recidivism_groundtruth_as_test_sensitive_removed(perm, debiased_real_accuracy=debiased_real_accuracy)
    print("Start: ", model_count, " Setting: ", perm, hidden1_units, hidden2_units, batch_size)
    input_dim = input_dim - 1
    tf.reset_default_graph()
    name = f"compas_two_year{model_count}"
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
        train_dir=f'trained_models_try/output_count{model_count}', 
        log_dir=f'throw/log{model_count}',
        hvp_files = f"inverse_HVP_compas_try/inverse_HVP_schm{scheme}_count{model_count}",
        model_name=name,
        scheme = f"{scheme}")

    model.train(num_steps=num_steps, iter_to_switch_to_batch=10000000, iter_to_switch_to_sgd=20000, save_checkpoints=False, verbose=False)
    train_acc, test_acc = model.print_model_eval()
    if debiased_real_accuracy:
        with open(f"results_nosensitive_real_accuracy_debiased.csv", "a") as f:
            f.write(f"{model_count},{perm},{h1units},{h2units},{batch},{test_acc}\n")     # the last ones gives percentage of discrimination
    else:
        with open(f"results_nosensitive_real_accuracy_full.csv", "a") as f:
            f.write(f"{model_count},{perm},{h1units},{h2units},{batch},{test_acc}\n")     # the last ones gives percentage of discrimination

    del model
    
