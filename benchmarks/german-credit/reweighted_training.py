import numpy as np
import IPython, sys, os
sys.path.append(".")
sys.path.append("../../")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import influence.experiments as experiments
from influence.fully_connected import Fully_Connected

from load_german_credit import exclude_some_examples, reweighted_load_german_credit, disparate_removed_load_german_credit    #, load_german_credit, load_german_credit_partial
from find_discm_points import entire_test_suite


input_dim = 20
weight_decay = 0.001
#batch_size = 50
#data_sets = None

initial_learning_rate = 0.05 
# decay_epochs = [20000]
#hidden1_units = 16
#hidden2_units = 8
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
                for batch in [50, 100]:
                    if model_count < setting_now:
                        model_count += 1
                        continue
                    # print(setting_now, "done", perm, h1units, h2units, batch)
                    return perm, h1units, h2units, batch, model_count

remove_biased_test_points = False

perm, h1units, h2units, batch, model_count = variation(setting_now)
# print(h1units, h2units, batch)
# exit(0)
# data_sets = load_german_credit(perm)
assert(model_count == setting_now)
# exclude = int(sys.argv[2])
# data_sets = exclude_some_examples(exclude, remove_biased_test=remove_biased_test_points)
# data_sets = disparate_removed_load_german_credit()
data_sets = reweighted_load_german_credit(perm = perm)
hidden1_units = h1units
hidden2_units = h2units
hidden3_units = 0
batch_size = batch
print("Start: ", model_count, " Setting: ", perm, hidden1_units, hidden2_units, batch_size)
# num_steps = batch_size * 1000
# import ipdb; ipdb.set_trace()
num_steps = 50000
decay_epochs = [60000, 70000]   # no work


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
    mini_batch=False,
    train_dir=f'throw/output_dont_save{model_count}', 
    log_dir=f'log{model_count}',
    hvp_files = f"inverse_HVP_schm{scheme}_count{model_count}",
    model_name=f"german_credit_count{model_count}",
    scheme = f"{scheme}"
    )

model.train(num_steps=num_steps, iter_to_switch_to_batch=10000000, iter_to_switch_to_sgd=20000, save_checkpoints=False, verbose=False)
iter_to_load = num_steps - 1
# train_acc, test_acc = model.load_checkpoint(iter_to_load=iter_to_load)
class0_data, class1_data = entire_test_suite(mini=False, reweighted_german=True, disparateremoved=False)     # False means loads entire data
num_dicsm = model.find_discm_examples(class0_data, class1_data, print_file=False, scheme=scheme)
train_acc, test_acc = model.print_model_eval()

print("Discrimination:", num_dicsm)
with open("reweighted_german_discrimination.csv", "a") as f:
    f.write(f'{h1units},{h2units},{batch},{perm},{train_acc*100},{test_acc*100},{num_dicsm}\n')

exit(0)
if remove_biased_test_points:
    with open("really_biased_removed_biased_test_points_removed.csv", "a") as f:
        f.write(f'{h1units},{h2units},{batch},{exclude},{train_acc},{test_acc},{num_dicsm}\n')
else:
    with open("really_biased_removed_without_test_points_removed.csv", "a") as f:
        f.write(f'{h1units},{h2units},{batch},{exclude},{train_acc},{test_acc},{num_dicsm}\n')

# print("hello", train_acc, test_acc)
# if not (train_acc > 0.7 and test_acc > 0.7):
#     print("BAD: ", setting_now)

del model
tf.reset_default_graph()
print("DONE: ", model_count, " Setting: ", hidden1_units, hidden2_units, batch_size)
                
            
