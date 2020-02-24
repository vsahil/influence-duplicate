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

from load_adult_income import load_adult_income, load_adult_income_partial
from find_discm_points import entire_test_suite

data_sets = load_adult_income()
# data_sets = reweighted_load_german_credit()

load_from_numpy = True

input_dim = 12
weight_decay = 0.001
batch_size = 3000

initial_learning_rate = 1e-5 
decay_epochs = [30000, 40000]
hidden1_units = 24
hidden2_units = 12
hidden3_units = 0
num_classes = 2
keep_probs = [1.0, 1.0]

scheme = 8
assert(scheme == 8)     # now always
name = 'adult_income_try1'
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
    damping=1e-2,
    decay_epochs=decay_epochs,
    mini_batch=True,
    train_dir='output', 
    log_dir='log1',
    hvp_files = "inverse_HVP_scheme{}".format(scheme),
    model_name=name,
    scheme = "scheme{}".format(scheme))

num_steps = 30000
# model.train(num_steps=num_steps, iter_to_switch_to_batch=10000000, iter_to_switch_to_sgd=20000, save_checkpoints=True)
class0_data, class1_data = entire_test_suite(mini=False)     # False means loads entire data
if not load_from_numpy:
    iter_to_load = num_steps - 1
    model.load_checkpoint(iter_to_load=iter_to_load, do_checks=False)
    # print(model.data_sets.train.num_examples)
    # exit(0)

    model.find_discm_examples(class0_data, class1_data, print_file=True, scheme=scheme)

    # exit(0)
    # test_idx = 0

    predicted_loss_diffs = model.get_influence_on_test_loss(
                [i for i in range(model.discm_data_set.num_examples)], 
                np.arange(len(model.data_sets.train.labels)),
                force_refresh=False)
    # import ipdb; ipdb.set_trace()
    sorted_training_points = np.argsort(predicted_loss_diffs)[::-1]    # decreasing order of influence among training points
    # print(sorted_training_points)
    with open("sorted_adult_training_points.csv", "a") as f:
        print(list(sorted_training_points), file=f)

    np.save(f"{name}_influential_training_points", sorted_training_points)
    del model   # so that weights of the original model are not used. This will not help

else:
   sorted_training_points = list(np.load(f"{name}_influential_training_points.npy"))

training_size = model.data_sets.train.num_examples
assert(len(sorted_training_points) == training_size)


# for percentage in range(5, 4, 0.5):
# for percentage in np.arange(0, 5.0, 0.2):
for p in range(1, 40):
    tf.reset_default_graph()
    # p = int(training_size * percentage / 100)
    # p = 19
    remaining_indexes = np.array(sorted_training_points[p:])
    data_sets_partial = load_adult_income_partial(remaining_indexes)
    assert(data_sets_partial.train.num_examples == 36000 - p)
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
        train_dir='output_partial', 
        log_dir='log_partial',
        hvp_files = "inverse_HVP_scheme1_",
        model_name='adult_income_partial',
        scheme = "scheme8_par")
    print("Training")
    model_partial_data.train(num_steps=num_steps, iter_to_switch_to_batch=10000000, iter_to_switch_to_sgd=20000, save_checkpoints=False, verbose=False)
    # print("Percentage: ", percentage, " Points removed: ", p)
    print("Points removed: ", p)
    num = model_partial_data.find_discm_examples(class0_data, class1_data, print_file=False, scheme=scheme)
    with open("scheme{}_results.txt".format(scheme), "a") as f:
        # f.write("Percentage: " + str(percentage) + ", Discriminating Tests: " + str(num) + "\n")
        f.write("Points: " + str(p) + ", Discriminating Tests: " + str(num) + "\n")
    
    del model_partial_data          # to remove any chance of reusing variables and reduce memory


