import numpy as np
import IPython, sys, os
sys.path.append("../")
sys.path.append("../../../")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import influence.experiments as experiments
from influence.fully_connected import Fully_Connected

from load_compas_score_as_labels import before_preferential_sampling, resampled_dataset
from find_discm_points import entire_test_suite

real_accuracy = True
debiased_real_accuracy = False

input_dim = 10
weight_decay = 0.002

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
    damping=3e-2,
    decay_epochs=decay_epochs,
    mini_batch=True,
    train_dir=f'throw/output_dont_save{model_count}', 
    log_dir=f'throw/log{model_count}',
    hvp_files = f"inverse_HVP_schm{scheme}_count{model_count}",
    model_name=f"compas_two_year{model_count}",
    scheme = f"{scheme}"
    )

model.train(num_steps=num_steps, iter_to_switch_to_batch=10000000, iter_to_switch_to_sgd=20000, save_checkpoints=False, verbose=False)
# train_acc, test_acc = model.print_model_eval()
# import ipdb; ipdb.set_trace()
losses = model.loss_per_instance()          # these losses will be different for all the 240 models so need to re-evaluate each time
del data_sets_init, model
tf.reset_default_graph()
# dep_neg_candidates = female_bad_credit_indices[np.argsort(losses[female_bad_credit_indices])[::-1]]   # sorted in decreasing order of loss
# dep_pos_candidates = female_good_credit_indices[np.argsort(losses[female_good_credit_indices])[::-1]]  # sorted in decreasing order of loss
# fav_neg_candidates = male_bad_credit_indices[np.argsort(losses[male_bad_credit_indices])[::-1]]     # sorted in decreasing order of loss
# fav_pos_candidates = male_good_credit_indices[np.argsort(losses[male_good_credit_indices])[::-1]]    # sorted in decreasing order of loss

dep_neg_candidates = male_bad_credit_indices[np.argsort(losses[male_bad_credit_indices])[::-1]]   # sorted in decreasing order of loss
dep_pos_candidates = male_good_credit_indices[np.argsort(losses[male_good_credit_indices])[::-1]]  # sorted in decreasing order of loss
fav_neg_candidates = female_bad_credit_indices[np.argsort(losses[female_bad_credit_indices])[::-1]]     # sorted in decreasing order of loss
fav_pos_candidates = female_good_credit_indices[np.argsort(losses[female_good_credit_indices])[::-1]]    # sorted in decreasing order of loss

data_sets_final = resampled_dataset(perm, dep_neg_candidates, dep_pos_candidates, fav_neg_candidates, fav_pos_candidates, real_accuracy=real_accuracy, debiased_real_accuracy=debiased_real_accuracy)

# import ipdb; ipdb.set_trace()
model_ = Fully_Connected(
    input_dim=input_dim, 
    hidden1_units=hidden1_units, 
    hidden2_units=hidden2_units,
    hidden3_units=hidden3_units,
    weight_decay=weight_decay,
    num_classes=num_classes, 
    batch_size=batch_size,
    data_sets=data_sets_final,
    initial_learning_rate=initial_learning_rate,
    damping=3e-2,
    decay_epochs=decay_epochs,
    mini_batch=True,
    train_dir=f'throw/output_dont_save{model_count}', 
    log_dir=f'throw/log{model_count}',
    hvp_files = f"inverse_HVP_schm{scheme}_count{model_count}",
    model_name=f"compas_two_year{model_count}",
    scheme = f"{scheme}"
    )

model_.train(num_steps=num_steps, iter_to_switch_to_batch=10000000, iter_to_switch_to_sgd=20000, save_checkpoints=False, verbose=False)

# iter_to_load = num_steps - 1
# train_acc, test_acc = model.load_checkpoint(iter_to_load=iter_to_load)
class0_data, class1_data = entire_test_suite(mini=False, disparateremoved=False)     # False means loads entire data
num_dicsm = model_.find_discm_examples(class0_data, class1_data, print_file=False, scheme=scheme)
train_acc, test_acc = model_.print_model_eval()

print("Discrimination:", num_dicsm)
size = class0_data.shape[0]/100
dataset = "compas-score"

if not real_accuracy:
    with open(f"results_resampling_{dataset}.csv", "a") as f:
        print(f'{h1units},{h2units},{batch},{perm},{train_acc},{test_acc},{num_dicsm},{num_dicsm/size}', file=f)

if real_accuracy:
    if debiased_real_accuracy:
        with open(f"results_resampling_{dataset}_real_accuracy_debiased.csv", "a") as f:
            print(f'{model_count},{h1units},{h2units},{batch},{perm},{test_acc}', file=f)
    else:
        with open(f"results_resampling_{dataset}_real_accuracy_full.csv", "a") as f:
            print(f'{model_count},{h1units},{h2units},{batch},{perm},{test_acc}', file=f)

