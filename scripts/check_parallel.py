# from math import sqrt
# from joblib import Parallel, delayed
# # pip install joblib
# import multiprocessing
# import sys

# num_cores = multiprocessing.cpu_count()

# def returnfunc(x, n):

#   sys.stdout.flush()

#   x = x*5
#   output = sqrt(x)

#   print('Produced %s' % x)
#   print('Output sqrt %s' % output)  

#   return [x, output]

# # with parallel_backend('multiprocessing'):
# num_cores = multiprocessing.cpu_count()
# returnedLists = Parallel(n_jobs=num_cores,prefer="threads")(delayed(returnfunc)(x, 10) for x in range(0, 10))
# sys.stdout.flush() # May help with print ordering??  

# print([i[0] for i in returnedLists])
# print([i[1] for i in returnedLists])


import sys, os
import multiprocessing, subprocess

def bar(x):
    for i in range(x*10000000):
        pass
    print('%i done\n' % x, end='')
    return 0

def run_command(index):
    # print(index)
    os.system(f"python train_all_permutations.py {index}")
    # process = subprocess.check_output(['python', 'train_all_permutations.py', str(index)], stdout=subprocess.PIPE)

pool = multiprocessing.Pool(10)
l = 20 * 3 * 2 * 2
mr = pool.map_async(run_command, range(l))
while not mr.ready():
    sys.stdout.flush()
    mr.wait(0.1)