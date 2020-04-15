import sys, os
import multiprocessing, subprocess
from itertools import product

def run_command(setting):
    os.system(f"python run_hyperparams_disparate_impact_remover.py {setting}")

pool = multiprocessing.Pool(96)
l = [i for i in range(240)]
mr = pool.map_async(run_command, l)
while not mr.ready():
    sys.stdout.flush()
    mr.wait(0.1)
print("DONE!")

