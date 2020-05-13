# import generate_plots as gp
import numpy as np
import pandas as pd
import sys
sys.path.append("./")
# command : python common_biased_points.py 840 0
# num_points_removed_dict, bad_points = gp.read_data()
# import ipdb; ipdb.set_trace()
# print(num_points_removed_dict, len(bad_points))
import load_student as load_file
# from numpy import genfromtxt
# ranked_training_points_in_original_permutation = genfromtxt("sorted_training_points.csv")
dataset = "student"

ranked_training_points_in_original_permutation = {}
for model_count in range(240):
    ranked_training_points_in_original_permutation[model_count] = list(np.load(f"ranking_points_ordered_method1/student_{model_count}.npy"))
    print(model_count, "done")

assert len(ranked_training_points_in_original_permutation[1]) == len(ranked_training_points_in_original_permutation[24])

def variation(setting_now):
    model_count = 0
    for perm in range(20):
        for h1units in [16, 24, 32]:
            for h2units in [8, 12]:
                for batch in [64, 128]:      # different batch sizes for this dataset
                    if model_count < setting_now:
                        model_count += 1
                        continue
                    # print(setting_now, "done", perm, h1units, h2units, batch)
                    return perm, h1units, h2units, batch, model_count

ranked_points_after_removing_permutation_effect = {}
biasness_of_each_point = {}

df = pd.read_csv(f"results_{dataset}_final.csv")
removal_df = df.sort_values("Discm_percent").groupby("Model-count", as_index=False).first()    
assert len(removal_df) == 240
# import ipdb; ipdb.set_trace()
training_size = len(ranked_training_points_in_original_permutation[0])
assert(training_size == 500)

total_dataset = 649
for i in range(total_dataset):  # total dataset
    biasness_of_each_point[i] = 0

for settings in range(240):
    perm, h1units, h2units, batch, model_count = variation(settings)
    assert(perm == settings//12)
    ordering = load_file.permutations(perm)
    # import ipdb; ipdb.set_trace()
    min_discm_removal_point = removal_df.loc[removal_df['Model-count'] == settings, 'Points-removed'].values
    assert len(min_discm_removal_point) == 1
    original_biased_points = [ordering[i] for i in ranked_training_points_in_original_permutation[settings]][:min_discm_removal_point[0]]     # no of training points removed is 8928
    s = set(original_biased_points)
    assert(len(s) == len(original_biased_points))       # each point is unique 
    for obp in original_biased_points:
        biasness_of_each_point[obp] += 1

sorted_biased_points = {k: v for k, v in sorted(biasness_of_each_point.items(), key=lambda item: item[1]) if v > 0}
print(sum([1 for k, v in sorted_biased_points.items() if v == 240]), "hello")
count = 0
biased_points = []
write = False
if write:
    with open("intersections.csv", "a") as f:
        for key, obj in sorted_biased_points.items():
            if obj > 0:
                f.write(f'{key}:{obj}\n')
                count += 1
        f.write(f"Total count: {count}")
# print(biased_points)
print("printed")

biased_points_global = np.array(list(sorted_biased_points.keys()))
np.save(f"{dataset}_biased_points.npy", biased_points_global)

exit(0)

points = []
occurence = []
for key, item in sorted_biased_points.items():
    points.append(key)
    occurence.append(item * 100/240)        # percentage occurence

# import pandas as pd
no_order = True
def plot_no_order_data_points():
    global no_order
    feed = {'Points':points, "Occurence":occurence}
    feed['Points'] = [i for i in range(len(feed['Points']))]
    return feed

def plot_order_data_points():
    feed = {'Points':points, "Occurence":occurence}
    return feed

if no_order:
    feed = plot_no_order_data_points()
else:
    feed = plot_order_data_points()

df = pd.DataFrame.from_dict(feed)
from plotnine import *
# import ipdb; ipdb.set_trace()
x = (ggplot(aes(x='Points', y='Occurence'), data=df) +\
    geom_point(size=1.5) +\
    # stat_smooth(colour='blue', span=0.2) +\
    # stat_summary() +\
    ylim(0, 100) +\
    # facet_wrap(['H1Units','H2units','Batch'], nrow=3, ncol=4,scales = 'free', labeller='label_both', shrink=False) + \
    # xlab("Data-Points in order of occurence (not showing order in dataset)") + \
    xlab("Data-Points in order of original datasets") + \
    ylab("Percentage Occurence") + \
    ggtitle("Plot showing percentage labelling of a point as biased among all settings") +\
    theme(axis_text_x = element_text(size=6), dpi=100) +\
    theme_seaborn()
    )

if no_order:
    x.save("biased_points_no_order.png")#, height=12, width=12)
else:
    x.save("biased_points_order.png")

