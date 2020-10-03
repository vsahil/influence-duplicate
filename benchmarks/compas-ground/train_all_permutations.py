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

from load_compas_score_as_labels import load_compas_two_year, load_compas_two_year_partial, dist
from find_discm_points import entire_test_suite

train = False
full_test = True
debiased_test = True

if not train:
    x = len(os.listdir(f'ranking_points_ordered_method1_dist{dist}'))
    assert x == 240

input_dim = 10
weight_decay = 0.002
# batch_size = 3000

initial_learning_rate = 1e-3
decay_epochs = [20000, 30000]
num_steps = 16000
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
data_sets = load_compas_two_year(perm, debiased_test=debiased_test)

print("Start: ", model_count, " Setting: ", perm, hidden1_units, hidden2_units, batch_size)

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
    train_dir=f'trained_models_method1_dist{dist}/output_count{model_count}', 
    log_dir=f'throw/log{model_count}',
    hvp_files = f"inverse_HVP_compas_method1_dist{dist}/inverse_HVP_schm{scheme}_count{model_count}",
    model_name=name,
    scheme = f"{scheme}")


if train:
    model.train(num_steps=num_steps, iter_to_switch_to_batch=10000000, 
    iter_to_switch_to_sgd=20000, save_checkpoints=True, verbose=False, plot_loss=False)
    # train_acc, test_acc = model.print_model_eval()
    # print(train_acc, test_acc, "see accuracies", model_count)

ranked_influential_training_points = f"ranking_points_ordered_method1_dist{dist}/{name}.npy"
# if not train and ranking of influential training points is stored in numpy file, then True
load_from_numpy = False if train else (True if os.path.exists(ranked_influential_training_points) else False)       
if not os.path.exists(f"ranking_points_ordered_method1_dist{dist}"):
    os.mkdir(f"ranking_points_ordered_method1_dist{dist}")

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
    # with open("sorted_compas_training_points.csv", "a") as f:
    #     print(list(sorted_training_points), file=f)
    training_size = model.data_sets.train.num_examples
    assert(len(sorted_training_points) == training_size)
    np.save(ranked_influential_training_points, sorted_training_points)
    del model   # so that weights of the original model are not used. This will not help

else:
   print("Loading from numpy")
   if full_test:
        dataset = "compas-ground"
        iter_to_load = num_steps - 1
        model.load_checkpoint(iter_to_load=iter_to_load, do_checks=False)
        initial_num = model.find_discm_examples(class0_data, class1_data, print_file=False, scheme=scheme)
        if True:        # added this just so that the code folds
            sensitive_attr = 2
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
            class0_pos = (tp + fp) / len(class0_index)        # proportion that got positive outcome
            del tn, fp, fn, tp
            tn, fp, fn, tp = class1_cm.ravel()
            class1_fpr = fp / (fp + tn)
            class1_fnr = fn / (fn + tp)
            class1_pos = (tp + fp) / len(class1_index)        # proportion that got positive outcome

        size = class0_data.shape[0]/100
        if debiased_test:
            with open(f"results_{dataset}_noremoval_dist{dist}.csv", "a") as f:
                print(f"{model_count},{perm},{h1units},{h2units},{batch},{train_acc},{test_acc},{class0_fpr},{class0_fnr},{class0_pos},{class1_fpr},{class1_fnr},{class1_pos},{initial_num},{initial_num/size}", file=f)
        else:
            with open(f"results_{dataset}_noremoval_fulltest_dist{dist}.csv", "a") as f:
                print(f"{model_count},{perm},{h1units},{h2units},{batch},{train_acc},{test_acc},{class0_fpr},{class0_fnr},{class0_pos},{class1_fpr},{class1_fnr},{class1_pos},{initial_num},{initial_num/size}", file=f)

        exit(0)
   sorted_training_points = list(np.load(ranked_influential_training_points))

if train:
    exit(0)

removal = int(sys.argv[2])
# percentage = removal/5.0
percentage = removal
training_size = 5000
dataset = "compas-ground"
size = class0_data.shape[0]/100
# for percentage in np.linspace(removal-1, removal-0.2, 5):
tf.reset_default_graph()
p = int(training_size * percentage / 100)
remaining_indexes = np.array(sorted_training_points[p:])
data_sets_partial = load_compas_two_year_partial(perm=perm, index=remaining_indexes)
try:
    assert(len(remaining_indexes) == training_size - p)
    assert(data_sets_partial.train.num_examples == training_size - p)
except:
    print(p, percentage, removal, data_sets_partial.train.num_examples, "hello")
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
        model_name='compas_two_year_partial',
        scheme = "scheme8_par")
print("Training")
print("Percentage: ", percentage, " Points removed: ", p) 
model_partial_data.train(num_steps=num_steps, iter_to_switch_to_batch=10000000, iter_to_switch_to_sgd=20000, save_checkpoints=False, verbose=False)
# train_acc, test_acc = model_partial_data.print_model_eval()
print("Percentage: ", percentage, " Points removed: ", p)
num = model_partial_data.find_discm_examples(class0_data, class1_data, print_file=False, scheme=scheme)
with open(f"results_{dataset}_debiasedtrain_80percentof_total_dist{dist}.csv".format(scheme), "a") as f:
    f.write(f"{model_count},{perm},{h1units},{h2units},{batch},{percentage},{p},{num},{num/size}\n")     # the last ones gives percentage of discrimination

del model_partial_data          # to remove any chance of reusing variables and reduce memory
