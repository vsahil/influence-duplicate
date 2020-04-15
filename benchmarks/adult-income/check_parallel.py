import sys, os
import multiprocessing, subprocess
from itertools import product


def experiment_command(setting, removal_percent):
    os.system(f"python -W ignore train_all_permutations.py {setting} {removal_percent}")
    # process = subprocess.check_output(['python', 'train_all_permutations.py', str(index)], stdout=subprocess.PIPE)

pool = multiprocessing.Pool(72)
l = 20 * 3 * 2 * 2 
l = [i for i in range(0, 51)]      # upto 25% removal in steps of 0.2%
# mr = pool.map_async(run_command, l)
settings = [i for i in range(240)]       # for first 120 settings in total
mr = pool.starmap_async(experiment_command, product(settings, l))
while not mr.ready():
    sys.stdout.flush()
    mr.wait(0.1)
print("DONE!")


def run_command(setting):
    os.system(f"python train_all_permutations.py {setting}")

# pool = multiprocessing.Pool(120)
# l = [i for i in range(240)]
# mr = pool.map_async(run_command, l)
# while not mr.ready():
#     sys.stdout.flush()
#     mr.wait(0.1)
# print("DONE!")
