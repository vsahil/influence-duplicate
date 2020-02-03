import os, subprocess, sys
# import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# tips = sns.load_dataset("tips")
# ax = sns.scatterplot(x="total_bill", y="tip", data=tips)

def variation(setting_now):
    model_count = 0
    for perm in range(20):
        for h1units in [16, 24, 32]:
            for h2units in [8, 12]:
                for batch in [50, 100]:
                    if model_count < setting_now:
                        model_count += 1
                        continue
                    # print(setting_now, "done", perm, h1units, h2units, batch)
                    return perm, h1units, h2units, batch, model_count


# perm, h1units, h2units, batch, model_count = variation(setting_now)
values_start = []
values_lowest = []
values_end = []
def plot_with_one_line_for_each_algo():
    global values_start, values_lowest, values_end
    assert(len(values_start) == len(values_lowest) == len(values_end) == 12)    # we have 12 algorithms
    means = np.array([np.mean(values_start), np.mean(values_lowest), np.mean(values_end)])
    stddev = np.array([np.std(values_start), np.std(values_lowest), np.std(values_end)])
    upper_bound = means + stddev
    lower_bound = means - stddev
    plt.figure()
    plt.plot([0, 1, 2], means, marker='o')
    plt.annotate(f'{round(means[1],1), round(stddev[1], 1)}', xy=(1, means[1]), xytext=(0.8, means[1] + 5000),
            arrowprops=dict(facecolor='black', shrink=0.05, width=0.3, headwidth=6),
            )
    plt.plot([0, 1, 2], upper_bound)
    plt.plot([0, 1, 2], lower_bound)
    plt.fill_between([0, 1, 2], upper_bound, lower_bound, color='c')
    plt.title(f"Means and Std. Deviations of all Algos", fontsize=10, fontweight='bold')
    plt.xlabel("Biased data-points removed", fontstyle='italic', color='red')
    plt.ylabel("No. of discrminating examples", fontstyle='italic', color='red')
    plt.xticks((0, 1, 2), ('0', 'Point with lowest discm', 'End point (40 or 80)'))
    # plt.subplots_adjust(top=0.9, bottom=0.08, left=0.125, right=0.9, hspace=0.6, wspace=0.2)
    # plt.text(1, ymax-1500, f"No. of data configs: {len(range_here)}", horizontalalignment='center', verticalalignment='center')

    plt.savefig(f"algo_plots/mean_overall.png")    


def one_plot_for_each_algo(algo, data, tolerance, bad_pts):
    global values_start, values_lowest, values_end
    range_here = [data[x] for x in range(20*algo, 20*(algo + 1)) if not x in bad_pts]
    assert(len(range_here) <= 20)
    # data_here = [data[x]  not x in bad_pts]
    print(algo, len(range_here))
    vals_start = []
    vals_lowest = []
    vals_end = []
    # import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(2, 1, 1)
    for vals in range_here:
        assert(len(vals) == 3)
        plt.plot([0, 1, 2], vals)
        vals_start.append(vals[0])
        vals_lowest.append(vals[1])
        vals_end.append(vals[2])

    means = np.array([np.mean(vals_start), np.mean(vals_lowest), np.mean(vals_end)])
    variance = np.array([np.std(vals_start), np.std(vals_lowest), np.std(vals_end)])
    upper_bound = means + variance
    lower_bound = means - variance
    values_start.append(means[0])
    values_lowest.append(means[1])
    values_end.append(means[2]) 

    do_plot = sys.argv[2]
    if do_plot == '0':
        return 
    # fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
    xmin, xmax, ymin, ymax = plt.axis()
    plt.title(f"Decrease in discriminating points for Setting {algo + 1}", fontsize=10, fontweight='bold')
    plt.xlabel("Biased data-points removed", fontstyle='italic', color='red')
    plt.ylabel("No. of discrminating examples", fontstyle='italic', color='red')
    plt.xticks((0, 1, 2), ('0', 'Point with lowest discm', 'End point (40 or 80)'))
    plt.text(1, ymax-1500, f"No. of data configs: {len(range_here)}", horizontalalignment='center', verticalalignment='center')

    plt.subplot(2, 1, 2)
    
    plt.plot([0, 1, 2], means, marker='o')
    # plt.text(1, means[1] + 2000, f"{means[1]}", horizontalalignment='center', verticalalignment='center', fontsize=7)
    plt.annotate(f'{round(means[1],1), round(variance[1], 1)}', xy=(1, means[1]), xytext=(0.8, means[1] + 5000),
            arrowprops=dict(facecolor='black', shrink=0.05, width=0.3, headwidth=6),
            )
    plt.plot([0, 1, 2], upper_bound)
    plt.plot([0, 1, 2], lower_bound)
    plt.fill_between([0, 1, 2], upper_bound, lower_bound, color='c')
    # import ipdb; ipdb.set_trace()
    # plt.ylim(20000)
    # plt.yticks((means[1], 5000, 10000, 15000))
    plt.title(f"Means and Std. Deviations", fontsize=10, fontweight='bold')
    # plt.text(1, max(means), f"Means and Std. Deviations", fontsize=10, horizontalalignment='center', verticalalignment='center')
    # plt.sub
    # plt.xlabel("Biased data-points removed", fontstyle='italic', color='red')
    plt.ylabel("No. of discrminating examples", fontstyle='italic', color='red')
    plt.xticks((0, 1, 2), ('0', 'Point with lowest discm', 'End point (40 or 80)'))
    plt.subplots_adjust(top=0.9, bottom=0.08, left=0.125, right=0.9, hspace=0.6, wspace=0.2)
    # plt.text(1, ymax-1500, f"No. of data configs: {len(range_here)}", horizontalalignment='center', verticalalignment='center')

    plt.savefig(f"algo_plots/Algo{algo + 1}_conc_{tolerance}.png")    


def plot_variance_points_removed(num_points_removed, bad_pts):
    removes_mean = []
    removes_min = []
    removes_max = []
    for algo in range(12):
        range_here = [num_points_removed[x] for x in range(20*algo, 20*(algo + 1)) if not x in bad_pts]
        assert(len(range_here) <= 20)
        # print(algo, len(range_here))
        mn = np.mean(range_here)
        removes_mean.append(mn)
        # removes_stddev.append(np.std(range_here))
        removes_min.append(mn - min(range_here))
        removes_max.append(max(range_here) - mn)
        # print(algo, min(range_here), max(range_here))
    # print(removes_extremes)
    # import ipdb; ipdb.set_trace()
    # for j in num_points_removed:
    #     if num_points_removed[j] > 65 and not j in bad_pts:
    #         print("hello", j, num_points_removed[j])

    removes_min = np.array(removes_min)
    removes_max = np.array(removes_max)
    removes_extremes = np.stack((removes_min, removes_max))
    # import ipdb; ipdb.set_trace()
    assert(removes_extremes.shape == (2, 12))
    assert(len(removes_mean) == len(removes_extremes[0]) == 12)

    # print(removes_extremes[1])
    plt.figure()
    plt.errorbar([i for i in range(12)], removes_mean, yerr=removes_extremes, capsize=5, ecolor='red', marker='s')
    plt.title("Mean and Extremes of the removed points", fontsize=12, fontweight='bold')
    plt.xlabel("Algorithm Settings", fontstyle='italic', color='blue')
    plt.ylabel("No. of training points removed", fontstyle='italic', color='blue')
    plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80])
    plt.xticks([i for i in range(12)], [f'{(i+1)}' for i in range(12)])
    plt.savefig(f"removed_points.png")


def plot_on_middle_line(data):
    bad_pts = []
    tolerance = int(sys.argv[1])
    discm_values = []
    all_values_including_bad_pts = []
    num_points_removed = {}
    okay_bad_pts = []
    for count in range(240):
        x = data[count][0].strip().split()
        x0 = int(x[-1])
        # values_start.append(int(x[-1]))

        x = data[count][39].strip().split()
        x2 = int(x[-1])
        # values_end.append(int(x[-1]))

        value_min = 100000        # total dataset size
        for rem in range(len(data[count])):
            x = data[count][rem].strip().split()
            if int(x[-1]) < value_min:
                value_min  = int(x[-1])
                num_points_removed[count] = rem
        # values_middle_min.append(value_min)
        if value_min > tolerance:
            # if not len(data[count]) >= 80:
            bad_pts.append(count)
            all_values_including_bad_pts.append((x0, value_min, x2))
            continue
        else:
            if value_min > 500:
                # print("BAD", count, value_min, len(data[count]))
                okay_bad_pts.append(count)
            x1 = value_min
        all_values_including_bad_pts.append((x0, x1, x2))
        discm_values.append((x0, x1, x2))
    print(len(discm_values), len(all_values_including_bad_pts))
    print(len(bad_pts), bad_pts)        # exclude these 37 points for which the hessian didn't converge or iterations maxed out. So the ranking is vacuous. 
    print(len(okay_bad_pts), okay_bad_pts)      # no improvement if after removing 120 pts. 
    assert(len(bad_pts) == 37)      # 840 
    assert(len(num_points_removed) == 240)
    
    plot_variance_points_removed(num_points_removed, bad_pts)
    
    do_plot = sys.argv[2]
    if do_plot == '0':
        return 
    
    for algo in range(12):
        one_plot_for_each_algo(algo, all_values_including_bad_pts, tolerance, bad_pts)
    print("Overall plotting")
    plot_with_one_line_for_each_algo()

    plt.figure()
    for vals in discm_values:
        assert(len(vals) == 3)
        plt.plot([0, 1, 2], vals)
    
    plt.title("Decrease in discriminating points", fontsize=16, fontweight='bold')
    plt.xlabel("Biased data-points removed", fontstyle='italic', color='red')
    plt.ylabel("No. of discrminating examples", fontstyle='italic', color='red')
    plt.xticks((0, 1, 2), ('0', 'Point with lowest discm', '40'))
    plt.savefig(f"middle_conc_{tolerance}.png")


def read_data():
    # s = 0
    content = {}        # this is the dictionary with all the data
    for count in range(240):
        filename = f"discm_points_results/model{count}_results.txt"
        assert os.path.exists(filename)
        # process = subprocess.Popen(['wc', f'{filename}'], stdout=subprocess.PIPE)
        # num_lines = process.stdout.read().decode("utf-8").rstrip("\n\r").split()
        # if int(num_lines[0]) > 40 and int(num_lines[0]) < 80:
            # s += 1
            # print(filename)
        with open(filename, "r") as f:
            content[count] = f.readlines()
        
        # test data expectations
        for i in range(len(content[count])):
            x = content[count][i].strip().split()
            removed_pts = int(x[2][:-1])        # this must be convertible to integer
            discm_points = int(x[-1])
            # print(i, count, len(content[count]), filename, removed_pts)
            if not count in [8, 10, 15, 17, 19, 21, 23, 24, 34, 63, 65, 67, 69, 71, 97, 105, 107, 132, 136, 138, 142, 166, 168, 180, 182, 184, 186, 188, 190, 192, 195, 196, 200, 201, 202, 228, 238]:
                assert(removed_pts == i)
        
        assert(len(content[count]) >= 40)
        # if len(content[count]) > 40 and len(content[count]) < 80:   # It works for them as well, only 4 such models {3, 39, 51, 128}
            # print(filename)

        #     print(removed_pts, i)
        # print(count)

        # import ipdb; ipdb.set_trace()
        # print(content)
    plot_on_middle_line(content)


if __name__ == "__main__":
    read_data()