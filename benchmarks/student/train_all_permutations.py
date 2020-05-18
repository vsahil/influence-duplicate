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

from load_student import load_student, load_student_partial
from find_discm_points import entire_test_suite

train = False
modify_test = True

if not train:
    x = len(os.listdir('ranking_points_ordered_method1'))
    assert x == 240


input_dim = 32
weight_decay = 0.01
# batch_size = 3000

initial_learning_rate = 1e-4 
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
                for batch in [64, 128]:      # different batch sizes for this dataset
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
damping = 8e-2

data_sets = load_student(perm, modify_test=modify_test)

print("Start: ", model_count, " Setting: ", perm, hidden1_units, hidden2_units, batch_size)

name = f"student_{model_count}"
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


if train:
    model.train(num_steps=num_steps, iter_to_switch_to_batch=10000000, 
    iter_to_switch_to_sgd=20000, save_checkpoints=True, verbose=False, plot_loss=False)
    # train_acc, test_acc = model.print_model_eval()
    # print(train_acc, test_acc, "see accuracies", model_count)
    # exit(0)

ranked_influential_training_points = f"ranking_points_ordered_method1/{name}.npy"
# if not train and ranking of influential training points is stored in numpy file, then True
load_from_numpy = False if train else (True if os.path.exists(ranked_influential_training_points) else False)       
if not os.path.exists("ranking_points_ordered_method1"):
    os.mkdir("ranking_points_ordered_method1")

class0_data, class1_data = entire_test_suite(mini=False)     # False means loads entire data
if not load_from_numpy:
    if not train:
        iter_to_load = num_steps - 1
        model.load_checkpoint(iter_to_load=iter_to_load, do_checks=False)

    model.find_discm_examples(class0_data, class1_data, print_file=True, scheme=scheme)

    predicted_loss_diffs = model.get_influence_on_test_loss(
                [i for i in range(model.discm_data_set.num_examples)], 
                np.arange(len(model.data_sets.train.labels)),
                force_refresh=False)

    sorted_training_points = np.argsort(predicted_loss_diffs)[::-1]    # decreasing order of influence among training points
    
    # print(sorted_training_points)
    # with open("sorted_student_training_points.csv", "a") as f:
    #     print(list(sorted_training_points), file=f)
    training_size = model.data_sets.train.num_examples
    assert(len(sorted_training_points) == training_size)
    np.save(ranked_influential_training_points, sorted_training_points)
    del model   # so that weights of the original model are not used. This will not help

else:
    print("Loading from numpy")
    if modify_test:
        iter_to_load = num_steps - 1
        model.load_checkpoint(iter_to_load=iter_to_load, do_checks=False)
        initial_num = model.find_discm_examples(class0_data, class1_data, print_file=False, scheme=scheme)
        train_acc, test_acc, _ = model.print_model_eval()
        size = class0_data.shape[0]/100
        with open("results_student_noremoval.csv".format(scheme), "a") as f:
                f.write(f"{model_count},{perm},{h1units},{h2units},{batch},{train_acc},{test_acc},{initial_num},{initial_num/size}\n")
        exit(0)
    sorted_training_points = list(np.load(ranked_influential_training_points))

if train:
    exit(0)


training_size = 520
dataset = "student"
p = int(sys.argv[2])
size = class0_data.shape[0]/100
remaining_indexes = np.array(sorted_training_points[p:])
tf.reset_default_graph()
percentage = (p * 100.0) / training_size

data_sets_partial = load_student_partial(perm=perm, index=remaining_indexes)
try:
    assert(len(remaining_indexes) == training_size - p)
    assert(data_sets_partial.train.num_examples == training_size - p)
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
    damping=damping,
    decay_epochs=decay_epochs,
    mini_batch=True,
    train_dir='output_partial', 
    log_dir='log_partial',
    hvp_files = "inverse_HVP_scheme1_",
    model_name='student_partial',
    scheme = "scheme8_par")
print("Training")
print("Percentage: ", percentage, " Points removed: ", p)
model_partial_data.train(num_steps=num_steps, iter_to_switch_to_batch=10000000, 
        iter_to_switch_to_sgd=20000, save_checkpoints=False, verbose=False)
# train_acc, test_acc = model_partial_data.print_model_eval()
print("Percentage: ", percentage, " Points removed: ", p)
num = model_partial_data.find_discm_examples(class0_data, class1_data, print_file=False, scheme=scheme)
with open(f"results_{dataset}_debiasedtrain_80percentof_total.csv".format(scheme), "a") as f:
        f.write(f"{model_count},{perm},{h1units},{h2units},{batch},{percentage},{p},{num},{num/size}\n")     # the last ones gives percentage of discrimination

