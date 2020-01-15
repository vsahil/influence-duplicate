from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import numpy as np
import IPython, sys, os

sys.path.append(".")
sys.path.append("../")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import influence.experiments as experiments
from influence.fully_connected import Fully_Connected

from load_german_credit import load_german_credit, load_german_credit_partial
from find_discm_points import entire_test_suite

data_sets = load_german_credit()

input_dim = 20 #* input_side * input_channels 
weight_decay = 0.001
batch_size = 50

initial_learning_rate = 0.005 
decay_epochs = [40000]
hidden1_units = 16
hidden2_units = 8
num_classes = 2
keep_probs = [1.0, 1.0]


model = Fully_Connected(
    input_dim=input_dim, 
    hidden1_units=hidden1_units, 
    hidden2_units=hidden2_units,
    weight_decay=weight_decay,
    num_classes=num_classes, 
    batch_size=batch_size,
    data_sets=data_sets,
    initial_learning_rate=initial_learning_rate,
    damping=1e-2,
    decay_epochs=decay_epochs,
    mini_batch=True,
    train_dir='output', 
    log_dir='log',
    hvp_files = "inverse_HVP_scheme1",
    model_name='german_credit_try1',
    scheme = "scheme1")

num_steps = batch_size * 1000
# model.train(num_steps=num_steps, iter_to_switch_to_batch=10000000, iter_to_switch_to_sgd=20000)
iter_to_load = num_steps - 1
iter_to_load = 46999
model.load_checkpoint(iter_to_load=iter_to_load, do_checks=False)
class0_data, class1_data = entire_test_suite(False)     # False means loads entire data
model.find_discm_examples(class0_data, class1_data, print_file=True)

# test_idx = 0

predicted_loss_diffs = model.get_influence_on_test_loss(
            [i for i in range(model.discm_data_set.num_examples)], 
            np.arange(len(model.data_sets.train.labels)),
            force_refresh=False)

sorted_training_points = list(np.argsort(predicted_loss_diffs)[::-1])     # decreasing order of influence among training points
# print(sorted_training_points)
assert(len(sorted_training_points) == 750)

del model   # so that weights of the original model are not used. This will not help

for percentage in range(5, 51, 5):
    tf.reset_default_graph()
    p = int(750 * percentage / 100)
    remaining_indexes = np.array(sorted_training_points[p:])
    data_sets_partial = load_german_credit_partial(remaining_indexes)
    assert(data_sets_partial.train.num_examples == 750 - p)
    model_partial_data = Fully_Connected(
        input_dim=input_dim, 
        hidden1_units=hidden1_units, 
        hidden2_units=hidden2_units,
        weight_decay=weight_decay,
        num_classes=num_classes, 
        batch_size=batch_size,
        data_sets=data_sets_partial,
        initial_learning_rate=initial_learning_rate,
        damping=1e-2,
        decay_epochs=decay_epochs,
        mini_batch=False,
        train_dir='output_partial', 
        log_dir='log_partial',
        hvp_files = "inverse_HVP_scheme1_",
        model_name='german_credit_partial',
        scheme = "scheme1_")
    print("Training")
    model_partial_data.train(num_steps=num_steps, iter_to_switch_to_batch=10000000, iter_to_switch_to_sgd=20000, verbose=False)
    print("Percentage: ", percentage)
    num = model_partial_data.find_discm_examples(class0_data, class1_data, print_file=False)
    with open("scheme1_results.txt", "a") as f:
        f.write("Percentage: " + str(percentage) + ", Discriminating Tests: " + str(num) + "\n")
    del model_partial_data          # to remove any chance of reusing variables and reduce memory


