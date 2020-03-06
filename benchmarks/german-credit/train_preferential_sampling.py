import numpy as np
import IPython, sys, os
# sys.path.append(".")
sys.path.append("../../")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import influence.experiments as experiments
from influence.fully_connected import Fully_Connected

from load_german_credit import before_preferential_sampling, resampled_german_credit
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
assert(model_count == setting_now)
data_sets_init, x_both = before_preferential_sampling(perm = perm)

male_good_credit_indices = x_both[(1, 1)]
male_bad_credit_indices = x_both[(1, 0)]
female_good_credit_indices = x_both[(0, 1)]
female_bad_credit_indices = x_both[(0, 0)]

hidden1_units = h1units
hidden2_units = h2units
hidden3_units = 0
batch_size = batch
print("Start: ", model_count, " Setting: ", perm, hidden1_units, hidden2_units, batch_size)
num_steps = 500
decay_epochs = [60000, 70000]   # no work


model = Fully_Connected(
    input_dim=input_dim, 
    hidden1_units=hidden1_units, 
    hidden2_units=hidden2_units,
    hidden3_units=hidden3_units,
    weight_decay=weight_decay,
    num_classes=num_classes, 
    batch_size=batch_size,
    data_sets=data_sets_init,
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

losses = model.loss_per_instance()
del data_sets_init, model
tf.reset_default_graph()
dep_neg_candidates = female_bad_credit_indices[np.argsort(losses[female_bad_credit_indices])[::-1]]   # sorted in decreasing order of loss
dep_pos_candidates = female_good_credit_indices[np.argsort(losses[female_good_credit_indices])[::-1]]  # sorted in decreasing order of loss
fav_neg_candidates = male_bad_credit_indices[np.argsort(losses[male_bad_credit_indices])[::-1]]     # sorted in decreasing order of loss
fav_pos_candidates = male_good_credit_indices[np.argsort(losses[male_good_credit_indices])[::-1]]    # sorted in decreasing order of loss

data_sets_final = resampled_german_credit(perm, dep_neg_candidates, dep_pos_candidates, fav_neg_candidates, fav_pos_candidates)

model = Fully_Connected(
    input_dim=input_dim, 
    hidden1_units=hidden1_units, 
    hidden2_units=hidden2_units,
    hidden3_units=hidden3_units,
    weight_decay=weight_decay,
    num_classes=num_classes, 
    batch_size=batch_size,
    data_sets=data_sets_final,
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

# iter_to_load = num_steps - 1
# train_acc, test_acc = model.load_checkpoint(iter_to_load=iter_to_load)
class0_data, class1_data = entire_test_suite(mini=False, disparateremoved=False)     # False means loads entire data
num_dicsm = model.find_discm_examples(class0_data, class1_data, print_file=False, scheme=scheme)
train_acc, test_acc = model.print_model_eval()

print("Discrimination:", num_dicsm)

with open("resampled_german_discrimination_results.csv", "a") as f:
    f.write(f'{h1units},{h2units},{batch},{perm},{train_acc*100},{test_acc*100},{num_dicsm}\n')
