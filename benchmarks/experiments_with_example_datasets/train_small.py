from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import numpy as np
import IPython, sys, os
import pandas as pd
from tensorflow.contrib.learn.python.learn.datasets import base
sys.path.append("../../")
sys.path.append("../../../")
from influence.dataset import DataSet

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import influence.experiments as experiments
from influence.fully_connected import Fully_Connected

def load_dataset(file, validation_size=0):
    df = pd.read_csv(file)
    total_labels = df['Decision'].to_numpy()
    total_dataset = df.drop('Decision', axis=1).to_numpy()
    train_examples = total_dataset.shape[0]		# entire dataset
    X_train = total_dataset[:train_examples]
    X_validation = total_dataset[train_examples:train_examples + validation_size]
    X_test  = total_dataset[train_examples + validation_size:]
    Y_train = total_labels[:train_examples]
    Y_validation = total_labels[train_examples:train_examples + validation_size]
    Y_test  = total_labels[train_examples + validation_size:]
    train = DataSet(X_train, Y_train)
    validation = DataSet(X_validation, Y_validation)
    test = DataSet(X_test, Y_test)
    return base.Datasets(train=train, validation=validation, test=test)


input_dim = 3
weight_decay = 0.001
initial_learning_rate = 0.001
num_classes = 2
keep_probs = [1.0, 1.0]

scheme = 8
# data_sets = load_dataset("biased.csv")
model_count = 5
data_sets = load_dataset(f"correlation{model_count}.csv")
hidden1_units = 2
hidden2_units = 2
hidden3_units = 0
batch_size = 3
num_steps = 5000
decay_epochs = [5000, 6000]
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
    damping=1e-1,
    decay_epochs=decay_epochs,
    mini_batch=True,
    train_dir=f'trained_models/output_count{model_count}', 
    log_dir=f'log{model_count}',
    hvp_files = f"HVP_files/inverse_HVP_schm{scheme}_count{model_count}",
    model_name=f"paper_example{model_count}",
    scheme = f"{scheme}"
    )

model.train(num_steps=num_steps, iter_to_switch_to_batch=10000000, iter_to_switch_to_sgd=20000, save_checkpoints=False, verbose=False)
train_acc, test_acc = model.print_model_eval()
# exit(0)

def entire_test_suite():
    class0 = np.genfromtxt(f"normalized_race0.csv", delimiter=",")
    class1 = np.genfromtxt(f"normalized_race1.csv", delimiter=",")
    return class0, class1


class0_data, class1_data = entire_test_suite()     # False means loads entire data
model.find_discm_examples(class0_data, class1_data, print_file=True, scheme=scheme)

# exit(0)
predicted_loss_diffs = model.get_influence_on_test_loss(
            [i for i in range(model.discm_data_set.num_examples)], 
            np.arange(len(model.data_sets.train.labels)),
            force_refresh=False)

sorted_training_points = list(np.argsort(predicted_loss_diffs)[::-1])     # decreasing order of influence among training points
# print(sorted_training_points)
training_size = model.data_sets.train.num_examples
assert(len(sorted_training_points) == training_size)

with open("sorted_training_points_biased.csv", "a") as f:
    f.write(str(model_count) + ": " + str(sorted_training_points))
    f.write("\n")
