import sys, os
import multiprocessing, subprocess
from itertools import product

def run_command(setting, debiased_test):
    os.system(f"python learning_fr.py {setting} {debiased_test}")

pool = multiprocessing.Pool(240)
settings = [i for i in range(240)]
debiased_test = [i for i in range(0, 2)]      # debiased test 0, 1
# mr = pool.map_async(run_command, l)
mr = pool.starmap_async(run_command, product(settings, debiased_test))
while not mr.ready():
    sys.stdout.flush()
    mr.wait(0.1)
print("DONE!")

