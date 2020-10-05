import sys, os
import multiprocessing, subprocess
from itertools import product


# def experiment_command(setting, removal_percent):
    # os.system(f"python train_all_permutations.py --model_number={setting} --train=0 --debiased_test=0 --full_baseline=0 --percentage_removal={removal_percent}")
    # os.system(f"python train_all_permutations_nosensitive.py --model_number={setting} --debiased_test={debiased_test}")

def experiment_command(setting, debiased_test):
    # os.system(f"python train_all_permutations.py --model_number={setting} --train=0 --debiased_test={debiased_test} --full_baseline=1")
    os.system(f"python train_all_permutations_nosensitive.py --model_number={setting} --debiased_test={debiased_test}")
    os.system(f"python methodology1.py {setting} {debiased_test}")


pool = multiprocessing.Pool(60)
# l = [i for i in range(26*5, 51*5)]      # upto 25% removal in steps of 0.2%, from 27 to 40.6 minimum discrimination 
# l = [i for i in range(1, 40)]      # upto 40% removal in steps of 1%
debiased_test = [i for i in range(0, 2)]      # debiased test 0, 1
# mr = pool.map_async(run_command, l)
settings = [i for i in range(240)]       # for first 120 settings in total
# mr = pool.starmap_async(experiment_command, product(settings, l))
mr = pool.starmap_async(experiment_command, product(settings, debiased_test))
while not mr.ready():
    sys.stdout.flush()
    mr.wait(0.1)
print("DONE!")


def run_command(setting):
    # os.system(f"python train_all_permutations.py {setting}")
    os.system(f"python train_all_permutations.py --model_number={setting} --train=1 --debiased_test=0 --full_baseline=0")
    # os.system(f"python methodology1.py {setting}")
    # os.system(f"python train_all_permutations_nosensitive.py {setting}")


# pool = multiprocessing.Pool(60)
# l = [i for i in range(240)]
# mr = pool.map_async(run_command, l)
# while not mr.ready():
#     sys.stdout.flush()
#     mr.wait(0.1)
# print("DONE!")
