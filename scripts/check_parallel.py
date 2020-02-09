import sys, os
import multiprocessing, subprocess

# def bar(x):
#     for i in range(x*10000000):
#         pass
#     print('%i done\n' % x, end='')
#     return 0
from itertools import product

# setting = int(sys.argv[1])
def run_command(setting, exclude):
    # print(index)
    # os.system(f"python train_all_permutations.py {index}")
    os.system(f"python -W ignore train_all_permutations.py {setting} {exclude}")
    # process = subprocess.check_output(['python', 'train_all_permutations.py', str(index)], stdout=subprocess.PIPE)

pool = multiprocessing.Pool(11)
# l = 20 * 3 * 2 * 2 
l = [i for i in range(40)]
# mr = pool.map_async(run_command, l)
settings = [i for i in range(12)]
mr = pool.starmap_async(run_command, product(settings, l))
while not mr.ready():
    sys.stdout.flush()
    mr.wait(0.1)

# from functools import partial

# def f(a, b, c):
#     print("{} {} {}".format(a, b, c))

# def main():
#     iterable = [1, 2, 3, 4, 5]
#     pool = multiprocessing.Pool()
#     a = "hi"
#     b = ["there", "theib"]
#     func = partial(f, a, b)
#     pool.map(func, iterable)
#     pool.close()
#     pool.join()

# if __name__ == "__main__":
#     main()



# def merge_names(a, b):
    # print('{} & {}'.format(a, b))


# def main():
#     names = ['Brown', 'Wilson', 'Bartlett', 'Rivera', 'Molloy', 'Opie']
#     # with multiprocessing.Pool(processes=3) as pool:
#     #     results = pool.starmap(merge_names, product(names, repeat=2))
#     # print(results)
#     pool = multiprocessing.Pool(11)
#     # l = 20 * 3 * 2 * 2 
#     # l = [i for i in range(5)]
#     mr = pool.starmap_async(merge_names, product(names, [i for i in range(7, 10)]))
#     # mr = pool.map_async(merge_names, l)
#     while not mr.ready():
#         sys.stdout.flush()
#         mr.wait(0.1)

# if __name__ == '__main__':
#     main()
    