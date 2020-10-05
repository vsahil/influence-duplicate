import numpy as np
import IPython, sys, os
sys.path.append("../")
sys.path.append("../../../")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import influence.experiments as experiments
from influence.fully_connected import Fully_Connected

from load_default import before_massaging_dataset, massaged_dataset, dist
from find_discm_points import entire_test_suite

input_dim = 23
weight_decay = 0.001

initial_learning_rate = 1e-4 
decay_epochs = [40000, 50000]
num_steps = 10000
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

data_sets_init, male_good_credit_indices, male_bad_credit_indices, female_good_credit_indices, female_bad_credit_indices, pairs_to_flip = before_massaging_dataset(perm = perm)
hidden1_units = h1units
hidden2_units = h2units
hidden3_units = 0
batch_size = batch
damping = 3e-2
debiased_test = bool(int(sys.argv[2]))

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
    damping=damping,
    decay_epochs=decay_epochs,
    mini_batch=True,
    train_dir=f'throw/output_dont_save{model_count}', 
    log_dir=f'throw/log{model_count}',
    hvp_files = f"inverse_HVP_schm{scheme}_count{model_count}",
    model_name=f"default_count{model_count}",
    scheme = f"{scheme}"
    )

model.train(num_steps=num_steps, iter_to_switch_to_batch=10000000, iter_to_switch_to_sgd=20000, save_checkpoints=False, verbose=False)
# train_acc, test_acc = model.print_model_eval()
losses = model.loss_per_instance()
del data_sets_init, model
tf.reset_default_graph()

promotion_candidates_ = losses[female_bad_credit_indices]
promotion_candidates = female_bad_credit_indices[np.argsort(promotion_candidates_)[-1:-pairs_to_flip-1:-1]]  # the highest loss members, closest to decision boundary, gives the last x items from the list x = pairs_to_flip
demotion_candidates_ = losses[male_good_credit_indices]
demotion_candidates = male_good_credit_indices[np.argsort(demotion_candidates_)[-1:-pairs_to_flip-1:-1]]  # the highest loss members, closest to decision boundary, gives the last x items from the list x = pairs_to_flip

assert len(promotion_candidates) == len(demotion_candidates)
data_sets_final = massaged_dataset(perm, promotion_candidates, demotion_candidates, debiased_test=debiased_test)

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
    damping=damping,
    decay_epochs=decay_epochs,
    mini_batch=True,
    train_dir=f'throw/output_dont_save{model_count}', 
    log_dir=f'throw/log{model_count}',
    hvp_files = f"inverse_HVP_schm{scheme}_count{model_count}",
    model_name=f"default_count2{model_count}",
    scheme = f"{scheme}"
    )

model_.train(num_steps=num_steps, iter_to_switch_to_batch=10000000, 
        iter_to_switch_to_sgd=20000, save_checkpoints=False, verbose=False)
class0_data, class1_data = entire_test_suite(mini=False, disparateremoved=False)     # False means loads entire data
num_dicsm = model_.find_discm_examples(class0_data, class1_data, print_file=False, scheme=scheme)

sensitive_attr = 1
train_acc, test_acc, test_predictions = model_.print_model_eval()
import sklearn, math
assert len(np.unique(data_sets_final.test.x[:, sensitive_attr])) == 2
# import ipdb; ipdb.set_trace()
class0_index = (data_sets_final.test.x[:, sensitive_attr] == 0).astype(int).nonzero()[0]
class1_index = (data_sets_final.test.x[:, sensitive_attr] == 1).astype(int).nonzero()[0]
test_predictions = np.argmax(test_predictions, axis=1)
class0_pred = test_predictions[class0_index]
class1_pred = test_predictions[class1_index]
class0_truth = data_sets_final.test.labels[class0_index]
class1_truth = data_sets_final.test.labels[class1_index]
assert(len(class0_pred) + len(class1_pred) == len(test_predictions))
assert(len(class0_truth) + len(class1_truth) == len(data_sets_final.test.labels))
class0_cm = sklearn.metrics.confusion_matrix(class0_truth, class0_pred, labels=[0,1])
class1_cm = sklearn.metrics.confusion_matrix(class1_truth, class1_pred, labels=[0,1])
tn, fp, fn, tp = class0_cm.ravel()
class0_fpr = fp / (fp + tn)
class0_fnr = fn / (fn + tp)
class0_pos = (tp + fp) / len(class0_index)        # proportion that got positive outcome
del tn, fp, fn, tp
tn, fp, fn, tp = class1_cm.ravel()
class1_fpr = fp / (fp + tn)
class1_fnr = fn / (fn + tp)
class1_pos = (tp + fp) / len(class1_index)        # proportion that got positive outcome

print("Discrimination:", num_dicsm, "pairs_to_flip", pairs_to_flip)
size = class0_data.shape[0]/100
dataset = "default"
if debiased_test:
    with open(f"results_massaged_{dataset}_dist{dist}.csv", "a") as f:
        print(f"{model_count},{h1units},{h2units},{batch},{perm},{pairs_to_flip},{train_acc},{test_acc},{class0_fpr},{class0_fnr},{class0_pos},{class1_fpr},{class1_fnr},{class1_pos},{num_dicsm},{num_dicsm/size}", file=f)
else:
    with open(f"results_massaged_{dataset}_fulltest_dist{dist}.csv", "a") as f:
        print(f"{model_count},{h1units},{h2units},{batch},{perm},{pairs_to_flip},{train_acc},{test_acc},{class0_fpr},{class0_fnr},{class0_pos},{class1_fpr},{class1_fnr},{class1_pos},{num_dicsm},{num_dicsm/size}", file=f)    
