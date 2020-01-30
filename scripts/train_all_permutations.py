import numpy as np
import IPython, sys, os
sys.path.append(".")
sys.path.append("../")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import influence.experiments as experiments
from influence.fully_connected import Fully_Connected

from load_german_credit import load_german_credit#, load_german_credit_partial
# from find_discm_points import entire_test_suite


input_dim = 20
weight_decay = 0.001
batch_size = 50
data_sets = None

initial_learning_rate = 0.005 
decay_epochs = [40000]
hidden1_units = 16
hidden2_units = 8
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
data_sets = load_german_credit(perm)
hidden1_units = h1units
hidden2_units = h2units
batch_size = batch

print("DONE: ", model_count, " Setting: ", perm, hidden1_units, hidden2_units, batch_size)
# num_steps = batch_size * 1000
num_steps = 50000
decay_epochs = [int(0.7 * num_steps)]


model = Fully_Connected(
    input_dim=input_dim, 
    hidden1_units=hidden1_units, 
    hidden2_units=hidden2_units,
    weight_decay=weight_decay,
    num_classes=num_classes, 
    batch_size=batch_size,
    data_sets=data_sets,
    initial_learning_rate=initial_learning_rate,
    damping=1e-2,
    decay_epochs=decay_epochs,
    mini_batch=True,
    train_dir=f'output_count{model_count}', 
    log_dir=f'log{model_count}',
    hvp_files = f"inverse_HVP_schm{scheme}_count{model_count}",
    model_name=f"german_credit_count{model_count}",
    scheme = f"{scheme}"
    )

model.train(num_steps=num_steps, iter_to_switch_to_batch=10000000, iter_to_switch_to_sgd=20000, verbose=False)
del model
tf.reset_default_graph()
print("DONE: ", model_count, " Setting: ", perm, hidden1_units, hidden2_units, batch_size)
                
            
