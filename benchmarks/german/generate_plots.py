import os, subprocess, sys
# import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from uncertainties import ufloat
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
    plt.annotate(f'{ufloat(round(means[1],1), round(stddev[1], 1))}', xy=(1, means[1]), xytext=(0.8, means[1] + 5),
            arrowprops=dict(facecolor='black', shrink=0.05, width=0.3, headwidth=6),
            )
    plt.plot([0, 1, 2], upper_bound)
    plt.plot([0, 1, 2], lower_bound)
    plt.fill_between([0, 1, 2], upper_bound, lower_bound, color='c')
    plt.title(f"Means and Std. Deviations of all settings", fontsize=10, fontweight='bold')
    plt.xlabel("Biased data-points removed", fontstyle='italic', color='red')
    plt.ylabel("% discrimination", fontstyle='italic', color='red')
    plt.xticks((0, 1, 2), ('0', 'Point with lowest discm', 'End point'))
    # plt.subplots_adjust(top=0.9, bottom=0.08, left=0.125, right=0.9, hspace=0.6, wspace=0.2)
    # plt.text(1, ymax-1500, f"No. of data configs: {len(range_here)}", horizontalalignment='center', verticalalignment='center')

    plt.savefig(f"algo_plots/mean_overall.png")    


def one_plot_for_each_algo(algo, data, tolerance, bad_pts):
    global values_start, values_lowest, values_end
    # import ipdb; ipdb.set_trace()
    range_here = [data[x] for x in range(20*algo, 20*(algo + 1)) if not x in bad_pts]
    assert(len(range_here) <= 20)
    # data_here = [data[x]  not x in bad_pts]
    # for j in range(20*algo, 20*(algo + 1)):
    #     _, h1units, h2units, batch, model_count = variation(j)
    #     print(len(range_here), h1units, h2units, batch)
    # exit(0)
    _, h1units, h2units, batch, model_count = variation(algo*20)
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
    
    plt.title(f"Decrease in discriminating points for Setting {h1units,h2units,batch}", fontsize=10, fontweight='bold')
    plt.xlabel("Biased data-points removed", fontstyle='italic', color='red')
    plt.ylabel("% discrimination", fontstyle='italic', color='red')
    plt.xticks((0, 1, 2), ('0', 'Point with lowest discm', 'End point'))
    plt.text(1, ymax-1.500, f"No. of data configs: {len(range_here)}", horizontalalignment='center', verticalalignment='center')

    plt.subplot(2, 1, 2)
    
    plt.plot([0, 1, 2], means, marker='o')
    # plt.text(1, means[1] + 2000, f"{means[1]}", horizontalalignment='center', verticalalignment='center', fontsize=7)
    # import ipdb; ipdb.set_trace()
    plt.annotate(f'{ufloat(round(means[1],1), round(variance[1],1))}', xy=(1, means[1]), xytext=(0.8, means[1] + 5),
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
    plt.ylabel("% discrimination", fontstyle='italic', color='red')
    plt.xticks((0, 1, 2), ('0', 'Point with lowest discm', 'End point'))
    plt.subplots_adjust(top=0.9, bottom=0.08, left=0.125, right=0.9, hspace=0.6, wspace=0.2)
    # plt.text(1, ymax-1500, f"No. of data configs: {len(range_here)}", horizontalalignment='center', verticalalignment='center')

    plt.savefig(f"algo_plots/Algo{algo + 1}_conc_{tolerance}.png")    


import pandas as pd
from plotnine import *
def plot_variance_points_removed(num_points_removed, bad_pts):
    removes_mean = []
    removes_min = []
    removes_max = []

    setting_id = {}
    # with open("all_german_discm_data.csv", "w") as f:
        # print("Model-count,Permutation,H1Units,H2units,Batch,Removal-point,Discm-percent", file=f)
    for model_count, valu in num_points_removed.items():
        if model_count in bad_pts:
            continue
        perm, h1units, h2units, batch, model_count_ = variation(model_count)
        assert(model_count == model_count_)
        if (h1units, h2units, batch) in setting_id:
            setting_id[h1units, h2units, batch].append(valu)
        else:
            setting_id[h1units, h2units, batch] = [valu]
        assert(model_count == model_count_)
        # for removal_points in data[model_count]:
        #     x = removal_points.strip().split()
        #     removed_pts = int(x[2][:-1])
        #     if removed_pts < 85:        # we don't care for points more than that because they occur only in cases where points were removed in percentages, not pointwise
        #         discm_percent = int(x[-1])/1000
        #         print(f"{model_count},{perm},{h1units},{h2units},{batch},{removed_pts},{discm_percent}", file = f)

    feed = {'Setting':[], 'Removed_points':[]}
    for i, j in setting_id.items():
        for z in range(len(j)):
            feed['Setting'].append(i)
            feed['Removed_points'].append(j[z]/800)         # dividing by 800 for relative number of points removed

    # feed = {'Setting':[str(i) for i in setting_id.keys()], 'Removed_points':[i for i in setting_id.values()]}
    
    # import ipdb; ipdb.set_trace()
    df = pd.DataFrame.from_dict(feed)
    
    # import ipdb; ipdb.set_trace()
    x = (ggplot(aes(x='Setting', y='Removed_points'), data=df) +\
        geom_point(size=2) +\
        # stat_smooth(colour='blue', span=0.2) +\
        # stat_summary() +\
        # ylim(0, 100) +\
        # facet_wrap(['H1Units','H2units','Batch'], nrow=3, ncol=4,scales = 'free', labeller='label_both', shrink=False) + \
        xlab("12 settings of hyper-parameters (H1units, H2units, batch)") + \
        ylab("Points removed for min discrimination") + \
        ggtitle("Plot showing the removed points for attaining minimum discrimination") +\
        theme(axis_text_x = element_text(rotation=90, hjust =1), dpi=500)  
        # theme_seaborn() 
        # theme_xkcd()
        # scale_y_discrete()
        )

    x.save("removed_points.png")    #, height=12, width=12)
    return 

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


def print_to_file(data, num_points_removed, bad_pts):
    count_discm = 0
    with open("min_discm.csv", "w") as f:
        f.write("Permutation,H1units,H2units,Batch,Discm\n")
        for i in range(len(data)):
            if not i in bad_pts:
                perm, h1units, h2units, batch, model_count = variation(i)
                assert(model_count == i)
                f.write(f'{perm},{h1units},{h2units},{batch},{data[model_count][1]}\n')     # [1] is the point with minimum discrimination
                count_discm += 1
        assert(count_discm == len(data) - len(bad_pts))

    count_removal = 0
    with open("removal.csv", "w") as f:
        f.write("Permutation,H1units,H2units,Batch,Removal\n")
        for i in range(len(data)):
            if not i in bad_pts:
                perm, h1units, h2units, batch, model_count = variation(i)
                assert(model_count == i)
                f.write(f'{perm},{h1units},{h2units},{batch},{num_points_removed[model_count]}\n')
                count_removal += 1
        assert(count_removal == len(data) - len(bad_pts))


def plot_on_middle_line(data):
    bad_pts = []
    tolerance = int(sys.argv[1])
    discm_values = []
    all_values_including_bad_pts = []
    num_points_removed = {}
    okay_bad_pts = []
    for count in range(240):
        x = data[count][0].strip().split()
        x0 = int(x[-1])/1000
        # values_start.append(int(x[-1]))

        x = data[count][39].strip().split()
        x2 = int(x[-1])/1000
        # values_end.append(int(x[-1]))

        value_min = 100000        # total dataset size
        for rem in range(len(data[count])):
            x = data[count][rem].strip().split()
            if int(x[-1]) < value_min:
                value_min  = int(x[-1])
                num_points_removed[count] = rem     # this might be incorrect because removed_pts != i for all model_counts
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
            x1 = round(value_min/1000, 1)

        all_values_including_bad_pts.append((x0, x1, x2))
        discm_values.append((x0, x1, x2))
        # all_data_points[count] = [i for i in]
    
    print(len(discm_values), len(all_values_including_bad_pts))
    print(len(bad_pts), bad_pts)        # exclude these 37 points for which the hessian didn't converge or iterations maxed out. So the ranking is vacuous. 
    # print(len(okay_bad_pts), okay_bad_pts)      # no improvement if after removing 120 pts. 
    assert(len(bad_pts) == 37)      # 840 
    assert(len(num_points_removed) == 240)
    
    plot_variance_points_removed(num_points_removed, bad_pts)

    do_plot = sys.argv[2]
    if do_plot == '0':
        return num_points_removed, bad_pts
    
    with open("find_correlation_hyperparams.csv", "w") as f:
        f.write("Permutation,H1units,H2units,Batch,Error\n")
        for i in range(240):
            perm, h1units, h2units, batch, model_count = variation(i)
            if not i in bad_pts:
                f.write(f'{perm},{h1units},{h2units},{batch},1\n')
            else:
                f.write(f'{perm},{h1units},{h2units},{batch},0\n')
    
    print_to_file(all_values_including_bad_pts, num_points_removed, bad_pts)
    # plot_variance_points_removed(num_points_removed, bad_pts)

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
    plt.ylabel("% discrmination", fontstyle='italic', color='red')
    plt.xticks((0, 1, 2), ('0', 'Point with lowest discm', '40'))
    plt.savefig(f"middle_conc_{tolerance}.png")


def print_csv(data, bad_pts):
    # import ipdb; ipdb.set_trace()
    #Let's assign ID's to the hyper-params settings (16, 8, 50) --> 0 and so on
    setting_id = {}
    setting_id_count = 0
    with open("all_german_discm_data.csv", "w") as f:
        print("Model-count,Permutation,H1Units,H2units,Batch,Removal-point,Discm-percent", file=f)
        for model_count in data.keys():
            if model_count in bad_pts:
                continue
            perm, h1units, h2units, batch, model_count_ = variation(model_count)
            if (h1units, h2units, batch) in setting_id:
                id_here = setting_id[h1units, h2units, batch]
            else:
                setting_id[h1units, h2units, batch] = setting_id_count
                setting_id_count += 1
            assert(model_count == model_count_)
            for removal_points in data[model_count]:
                x = removal_points.strip().split()
                removed_pts = int(x[2][:-1])
                if removed_pts < 85:        # we don't care for points more than that because they occur only in cases where points were removed in percentages, not pointwise
                    discm_percent = int(x[-1])/1000
                    print(f"{model_count},{perm},{h1units},{h2units},{batch},{removed_pts},{discm_percent}", file = f)


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
        here = [8, 10, 15, 17, 19, 21, 23, 24, 34, 63, 65, 67, 69, 71, 97, 105, 107, 132, 136, 138, 142, 166, 168, 180, 182, 184, 186, 188, 190, 192, 195, 196, 200, 201, 202, 228, 238]
        bad_pts = [8, 10, 15, 17, 19, 21, 23, 24, 34, 63, 65, 67, 69, 71, 97, 105, 107, 132, 136, 138, 142, 166, 168, 180, 182, 184, 186, 188, 190, 192, 195, 196, 200, 201, 202, 228, 238]
        assert(here == bad_pts)
        for i in range(len(content[count])):
            x = content[count][i].strip().split()
            removed_pts = int(x[2][:-1])        # this must be convertible to integer
            discm_points = int(x[-1])
            # print(i, count, len(content[count]), filename, removed_pts)
            if not count in here:
                assert(removed_pts == i)
        
        assert(len(content[count]) >= 40)
        # if len(content[count]) > 40 and len(content[count]) < 80:   # It works for them as well, only 4 such models {3, 39, 51, 128}
            # print(filename)

        #     print(removed_pts, i)
        # print(count)

        # import ipdb; ipdb.set_trace()
        # print(content)
    # print_csv(content, bad_pts)
    return plot_on_middle_line(content)


if __name__ == "__main__":
    read_data()