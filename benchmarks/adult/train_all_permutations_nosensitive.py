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

from load_adult_income import load_adult_income_nosensitive
from find_discm_points import entire_test_suite

train = True

input_dim = 12
weight_decay = 0.001
# batch_size = 3000

initial_learning_rate = 1e-5 
decay_epochs = [30000, 40000]
# hidden1_units = 24
# hidden2_units = 12
# hidden3_units = 0
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
data_sets = load_adult_income_nosensitive(perm)

hidden1_units = h1units
hidden2_units = h2units
hidden3_units = 0
batch_size = batch

print("Start: ", model_count, " Setting: ", perm, hidden1_units, hidden2_units, batch_size)

name = f"adult_income_count{model_count}"
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
    train_dir=f'trained_models_nosensitive/output_count{model_count}', 
    log_dir=f'throw/log{model_count}',
    hvp_files = f"inverse_HVP_adult_nosensitive/inverse_HVP_schm{scheme}_count{model_count}",
    model_name=name,
    scheme = f"{scheme}")


num_steps = 30000
if train:
    model.train(num_steps=num_steps, iter_to_switch_to_batch=10000000, iter_to_switch_to_sgd=20000, save_checkpoints=False, verbose=False)

# ranked_influential_training_points = f"ranking_points_ordered_nosensitive/{name}.npy"
# # if not train and ranking of influential training points is stored in numpy file, then True
# load_from_numpy = False if train else (True if os.path.exists(ranked_influential_training_points) else False)       
# assert(load_from_numpy)
class0_data, class1_data = entire_test_suite(mini=False)     # False means loads entire data

initial_num = model.find_discm_examples(class0_data, class1_data, print_file=False, scheme=scheme)
print(initial_num, "See")
size = class0_data.shape[0]/100
with open("results_adult_nosensitive.csv".format(scheme), "a") as f:
    f.write(f"{model_count},{perm},{h1units},{h2units},{batch},{initial_num},{initial_num/size}\n")
#    sorted_training_points = list(np.load(ranked_influential_training_points))

# if train:
exit(0)

# exit(0)
# for percentage in range(5, 4, 0.5):
# for percentage in np.arange(0, 5.0, 0.2):
removal = int(sys.argv[2])
# import ipdb; ipdb.set_trace()
# for p in removal:
size = class0_data.shape[0]/100
for percentage in np.linspace(removal-1, removal-0.2, 5):
    tf.reset_default_graph()
    p = int(36000 * percentage / 100)
    remaining_indexes = np.array(sorted_training_points[p:])
    data_sets_partial = load_adult_income_partial(perm=perm, index=remaining_indexes)
    try:
        assert(len(remaining_indexes) == 36000 - p)
        assert(data_sets_partial.train.num_examples == 36000 - p)
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
        damping=1e-2,
        decay_epochs=decay_epochs,
        mini_batch=False,
        train_dir='output_partial', 
        log_dir='log_partial',
        hvp_files = "inverse_HVP_scheme1_",
        model_name='adult_income_partial',
        scheme = "scheme8_par")
    print("Training")
    print("Percentage: ", percentage, " Points removed: ", p) 
    model_partial_data.train(num_steps=num_steps, iter_to_switch_to_batch=10000000, iter_to_switch_to_sgd=20000, save_checkpoints=False, verbose=False)
    train_acc, test_acc = model_partial_data.print_model_eval()
    print("Percentage: ", percentage, " Points removed: ", p)
    num = model_partial_data.find_discm_examples(class0_data, class1_data, print_file=False, scheme=scheme)
    with open("results_redone_adult.csv".format(scheme), "a") as f:
        f.write(f"{model_count},{perm},{h1units},{h2units},{batch},{train_acc},{test_acc},{percentage},{p},{num},{num/size}\n")     # the last ones gives percentage of discrimination
    
    del model_partial_data          # to remove any chance of reusing variables and reduce memory


