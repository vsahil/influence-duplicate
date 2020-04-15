import sys, os
import multiprocessing, subprocess

# def bar(x):
#     for i in range(x*10000000):
#         pass
#     print('%i done\n' % x, end='')
#     return 0
from itertools import product
# def exclude_experiments():

    # setting = int(sys.argv[1])
def experiment_command(setting, removal_percent):
    # os.system(f"python train_all_permutations.py {index}")
    os.system(f"python -W ignore train_all_permutations.py {setting} {removal_percent}")
    # process = subprocess.check_output(['python', 'train_all_permutations.py', str(index)], stdout=subprocess.PIPE)

pool = multiprocessing.Pool(11)
# l = 20 * 3 * 2 * 2 
l = [i for i in range(26)]      # upto 25% removal in steps of 0.2%
# mr = pool.map_async(run_command, l)
settings = [i for i in range(240)]       # for 240 settings in total
mr = pool.starmap_async(experiment_command, product(settings, l))
while not mr.ready():
    sys.stdout.flush()
    mr.wait(0.1)
print("DONE!")

# exclude_experiments()

# def reweighted_german_experiment():
    


    # pool = multiprocessing.Pool(4)
    # l = 20 * 3 * 2 * 2 
    # # remaining = [4, 5, 9, 10, 11, 16, 17, 21, 22, 23, 28, 29, 33, 34, 35, 40, 41, 45, 46, 47, 52, 53, 57, 58, 59, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239]
    # # right_now = [4, 16, 28, 40, 52, 64, 69, 74, 79, 84, 89]

    # # remaining = [5, 9, 10, 11, 17, 21, 22, 23, 29, 33, 34, 35, 41, 45, 46, 47, 53, 57, 58, 59, 65, 66, 67, 68, 70, 71, 72, 73, 75, 76, 77, 78, 80, 81, 82, 83, 85, 86, 87, 88, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144]
    # # bad_remaining = [8, 10, 15, 17, 19, 21, 23, 24, 25, 26, 27, 28, 30, 32, 34, 46, 63, 65, 67, 69, 71, 75, 79, 86, 88, 90, 92, 94, 97, 105, 107, 108, 110, 112, 114, 118, 128, 132, 136, 138, 142, 146, 148, 150, 158, 163, 166, 167, 168, 169, 175, 180, 182, 184, 186, 188, 190, 192, 195, 196, 199, 200, 201, 202, 203, 208, 214, 228, 238, 239]
    # # even_bad_remaining = [8, 10, 15, 17, 19, 21, 23, 24, 34, 63, 65, 67, 69, 71, 97, 105, 107, 132, 136, 138, 142, 166, 168, 180, 182, 184, 186, 188, 190, 192, 195, 196, 200, 201, 202, 228, 238]
    # # even_bad_remaining = [27, 46, 75, 86, 88, 90, 92, 94, 108, 110, 112, 114, 118, 146, 148, 150, 163, 167, 208, 214, 239]

    # # bad_pts = [8, 10, 15, 17, 19, 21, 23, 24, 34, 63, 65, 67, 69, 71, 97, 105, 107, 132, 136, 138, 142, 166, 168, 180, 182, 184, 186, 188, 190, 192, 195, 196, 200, 201, 202, 228, 238]
    # run_for_sorted_points = [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 12, 13, 14, 16, 18, 20, 22, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 64, 66, 68, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 98, 99, 100, 101, 102, 103, 104, 106, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 133, 134, 135, 137, 139, 140, 141, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 167, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 181, 183, 185, 187, 189, 191, 193, 194, 197, 198, 199, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 229, 230, 231, 232, 233, 234, 235, 236, 237, 239]
    # assert(len(run_for_sorted_points) == 203)
    # print(len(bad_remaining), len(even_bad_remaining))
    # exit(0)
    # remaining = remaining[:20]
    # for tot in remaining:
    #     filename = f"discm_points_results/model{tot}_results.txt"
    #     if os.path.exists(filename):
    #         os.system(f'rm {filename}')
        
    # num_cores = 11
    # Parallel(n_jobs=num_cores)(delayed(run_command)(ind) for ind in remaining)
def run_command(setting):
    os.system(f"python train_all_permutations.py {setting}")
    
    # os.system(f"python train_german_credit.py {index}")
    # process = subprocess.check_output(['python', 'train_all_permutations.py', str(index)], stdout=subprocess.PIPE)

# pool = multiprocessing.Pool(11)
# l = [i for i in range(240)]
# mr = pool.map_async(run_command, l)
# # # mr = pool.map_async(run_command, range(60, 72))
# while not mr.ready():
#     sys.stdout.flush()
#     mr.wait(0.1)
# print("DONE!")

# for i in remaining:
#     p = multiprocessing.Process(target=run_command, args=(i,))
#     p.start()
#     p.join()

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
#     reweighted_german_experiment()
    
