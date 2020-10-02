import sys, os
import multiprocessing, subprocess
from itertools import product


def experiment_command(setting, removal_percent):
    os.system(f"python -W ignore train_all_permutations.py {setting} {removal_percent}")

# pool = multiprocessing.Pool(240)
# l = [i for i in range(0, 20)]      # upto 20 points removed
# settings = [i for i in range(240)]       # for all settings
# mr = pool.starmap_async(experiment_command, product(settings, l))
# while not mr.ready():
#     sys.stdout.flush()
#     mr.wait(0.1)
# print("DONE!")


def run_command(setting):
    os.system(f"python train_all_permutations.py {setting}")
    # os.system(f"python methodology1.py {setting}")
    # os.system(f"python train_all_permutations_nosensitive.py {setting}")


pool = multiprocessing.Pool(240)
l = [i for i in range(240)]
mr = pool.map_async(run_command, l)
while not mr.ready():
    sys.stdout.flush()
    mr.wait(0.1)
print("DONE!")
