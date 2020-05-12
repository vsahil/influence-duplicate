import sys, os
import multiprocessing

def run_command(setting):
    os.system(f"python adversarial_debiasing.py {setting}")

pool = multiprocessing.Pool(20)
l = [i for i in range(20)]
mr = pool.map_async(run_command, l)
while not mr.ready():
    sys.stdout.flush()
    mr.wait(0.1)
print("DONE!")

