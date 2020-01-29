from joblib import Parallel, delayed
import multiprocessing, subprocess
# import pandas as pd



# def run_command(index):
#     process = subprocess.Popen(['python', 'get-influence.py', str(index)], stdout=subprocess.PIPE)
#     while process.poll() is None:
#         l_ = process.stdout.readline().decode("utf-8").rstrip("\n\r") # This blocks until it receives a newline.
#         if len(l_) > 0:
#             print(l_)


# def run_see(index):
#     subprocess.check_output(['python', 'get-influence.py', str(index)])


def driver_parallel():
    num_cores = multiprocessing.cpu_count()
    print("CPU COUNT: ", num_cores)
    # num_cores = 1
    # df_numpy = pd.read_csv("scheme8_labelled_generated_tests.csv").to_numpy()
    # l = len(df_numpy)
    l = 20 * 3 * 2 * 2      # 240 combinations
    # Parallel(n_jobs=num_cores)(delayed(subprocess.check_output)(['python', 'get-influence.py', str(ind)]) for ind in range(l))
    Parallel(n_jobs=num_cores)(delayed(subprocess.check_output)(['python', 'train_all_permutations.py', str(ind)]) for ind in range(l))
    # Parallel(n_jobs=num_cores)(delayed(run_command)(ind) for ind in range(l))


driver_parallel()