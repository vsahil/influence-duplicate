import numpy as np
import IPython, sys, os
sys.path.append("../")
sys.path.append("../../../")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import influence.experiments as experiments
from influence.fully_connected import Fully_Connected

from load_student import before_massaging_dataset, massaged_dataset
from find_discm_points import entire_test_suite

input_dim = 32
weight_decay = 0.002

initial_learning_rate = 1e-3 
decay_epochs = [20000, 30000]
num_steps = 100
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

perm, h1units, h2units, batch, model_count = variation(setting_now)
assert(model_count == setting_now)

data_sets_init, male_good_credit_indices, male_bad_credit_indices, female_good_credit_indices, female_bad_credit_indices, pairs_to_flip = before_massaging_dataset(perm = perm)
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
    damping=1e-2,
    decay_epochs=decay_epochs,
    mini_batch=True,
    train_dir=f'throw/output_dont_save{model_count}', 
    log_dir=f'throw/log{model_count}',
    hvp_files = f"inverse_HVP_schm{scheme}_count{model_count}",
    model_name=f"student_count{model_count}",
    scheme = f"{scheme}"
    )


model.train(num_steps=num_steps, iter_to_switch_to_batch=10000000, iter_to_switch_to_sgd=20000, save_checkpoints=False, verbose=False)
train_acc, test_acc = model.print_model_eval()
losses = model.loss_per_instance()
del data_sets_init, model
tf.reset_default_graph()

promotion_candidates_ = losses[female_bad_credit_indices]
promotion_candidates = female_bad_credit_indices[np.argsort(promotion_candidates_)[-1:-pairs_to_flip-1:-1]]  # the highest loss members, closest to decision boundary, gives the last x items from the list x = pairs_to_flip
demotion_candidates_ = losses[male_good_credit_indices]
demotion_candidates = male_good_credit_indices[np.argsort(demotion_candidates_)[-1:-pairs_to_flip-1:-1]]  # the highest loss members, closest to decision boundary, gives the last x items from the list x = pairs_to_flip

assert len(promotion_candidates) == len(demotion_candidates)
data_sets_final = massaged_dataset(perm, promotion_candidates, demotion_candidates)

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
    damping=1e-2,
    decay_epochs=decay_epochs,
    mini_batch=True,
    train_dir=f'throw/output_dont_save{model_count}', 
    log_dir=f'throw/log{model_count}',
    hvp_files = f"inverse_HVP_schm{scheme}_count{model_count}",
    model_name=f"student_count2{model_count}",
    scheme = f"{scheme}"
    )

model_.train(num_steps=num_steps, iter_to_switch_to_batch=10000000, iter_to_switch_to_sgd=20000, save_checkpoints=False, verbose=False)
class0_data, class1_data = entire_test_suite(mini=False, disparateremoved=False)     # False means loads entire data
num_dicsm = model_.find_discm_examples(class0_data, class1_data, print_file=False, scheme=scheme)
train_acc, test_acc = model_.print_model_eval()

print("Discrimination:", num_dicsm, "pairs_to_flip", pairs_to_flip)
size = class0_data.shape[0]/100
with open("results_massaged_student.csv", "a") as f:
    f.write(f'{h1units},{h2units},{batch},{perm},{pairs_to_flip},{train_acc*100},{test_acc*100},{num_dicsm},{num_dicsm/size}\n')
