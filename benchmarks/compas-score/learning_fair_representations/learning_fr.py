import sys, os
sys.path.append("../")
sys.path.append("../../../")
sys.path.append("../../../competitors/AIF360/")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import influence.experiments as experiments
from influence.fully_connected import Fully_Connected

from aif360.datasets import MyCompasScoreDataset
from aif360.algorithms.preprocessing.lfr import LFR

real_accuracy = True

input_dim = 10
weight_decay = 0.002
# batch_size = 3000

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

# dataset is permuted
dataset_orig = MyCompasScoreDataset(
    protected_attribute_names=['race'],                   
    privileged_classes=[[1]], 
    normalized = True,
    permute = perm   
)

# There is a test set leak here, you need to correct that
# Correction - Transform only the training set, not the test data. 
train_examples = 5000
dataset_orig_train, dataset_orig_test = dataset_orig.split([train_examples], shuffle=False)
assert(len(dataset_orig_train.convert_to_dataframe()[0]) == train_examples)

privileged_groups = [{'race': 1}]
unprivileged_groups = [{'race': 0}]
TR = LFR(unprivileged_groups = unprivileged_groups, privileged_groups = privileged_groups)
TR = TR.fit(dataset_orig_train)
dataset_transf_train = TR.transform(dataset_orig_train)
new_df = dataset_transf_train.convert_to_dataframe()[0]

train_labels = new_df['compas_score'].to_numpy()
train_features = new_df.drop(columns=['compas_score']).to_numpy()

write = False
if write:
    with open("see.csv", "w") as f:
        new_df.to_csv(f, index=False)

from load_compas_score_as_labels import load_fair_representations
from find_discm_points import entire_test_suite

data_sets = load_fair_representations(perm, train_features, train_labels, real_accuracy = real_accuracy)

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
    data_sets=data_sets,
    initial_learning_rate=initial_learning_rate,
    damping=3e-2,
    decay_epochs=decay_epochs,
    mini_batch=True,
    train_dir=f'throw/output_dont_save{model_count}', 
    log_dir=f'throw/log{model_count}',
    hvp_files = f"inverse_HVP_schm{scheme}_count{model_count}",
    model_name=f"compas_count_two_year{model_count}",
    scheme = f"{scheme}"
    )

model.train(num_steps=num_steps, iter_to_switch_to_batch=10000000, 
    iter_to_switch_to_sgd=20000, save_checkpoints=False, verbose=False)
train_acc, test_acc = model.print_model_eval()
class0_data, class1_data = entire_test_suite(mini=False, disparateremoved=False)     # False means loads entire data
num_dicsm = model.find_discm_examples(class0_data, class1_data, print_file=False, scheme=scheme)

print("Discrimination:", num_dicsm)
size = class0_data.shape[0]/100
dataset = "compas-score"
if not real_accuracy:
    with open(f"results_lfr_{dataset}.csv", "a") as f:
        print(f'{h1units},{h2units},{batch},{perm},{train_acc},{test_acc},{num_dicsm},{num_dicsm/size}', file=f)

if real_accuracy:
    with open(f"results_lfr_{dataset}_real_accuracy.csv", "a") as f:
        print(f'{model_count},{h1units},{h2units},{batch},{perm},{test_acc}', file=f)

