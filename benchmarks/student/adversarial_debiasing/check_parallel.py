import sys, os
import multiprocessing
from itertools import product


def run_command(setting, debiased_test):
    os.system(f"python adversarial_debiasing.py {setting} {debiased_test}")

pool = multiprocessing.Pool(40)
settings = [i for i in range(20)]
debiased_test = [i for i in range(0, 2)]      # debiased test 0, 1
# mr = pool.map_async(run_command, settings)
mr = pool.starmap_async(run_command, product(settings, debiased_test))
while not mr.ready():
    sys.stdout.flush()
    mr.wait(0.1)
print("DONE!")

