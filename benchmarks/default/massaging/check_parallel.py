import sys, os
import multiprocessing, subprocess
from itertools import product

def run_command(setting, debiased):
    os.system(f"python train_massage.py {setting} {debiased}")

pool = multiprocessing.Pool(80)
l = [i for i in range(240)]
debiased = [0, 1]
mr = pool.starmap_async(run_command, product(l, debiased))
# mr = pool.map_async(run_command, l)
while not mr.ready():
    sys.stdout.flush()
    mr.wait(0.1)
print("DONE!")

