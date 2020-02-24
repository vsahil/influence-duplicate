import generate_plots as gp

# command : python common_biased_points.py 840 0
num_points_removed_dict, bad_points = gp.read_data()
# import ipdb; ipdb.set_trace()
# print(num_points_removed_dict, len(bad_points))
import load_german_credit as lg
from numpy import genfromtxt
# ranked_training_points_in_original_permutation = genfromtxt("sorted_training_points.csv")
# import ipdb; ipdb.set_trace()

with open("sorted_training_points.csv", "r") as f:
    content = f.readlines()
# import ipdb; ipdb.set_trace()
ranked_training_points_in_original_permutation = {}
for i in content:
    i = i.rstrip()
    model_count = int(i.split(":")[0])
    ranked_training_points_in_original_permutation[model_count] = [int(items) for items in i.split(":")[1].strip()[1:-1].split(", ")]

# import ipdb; ipdb.set_trace()
ranked_points_after_reversing_permutation = {}
biasness_of_each_point = {}

for i in range(1000):
    biasness_of_each_point[i] = 0

for settings in range(240):
    # print(num_points_removed_dict[settings])
    if settings in bad_points:
        continue
    perm, h1units, h2units, batch, model_count = gp.variation(settings)
    assert(perm == settings//12)
    ordering = lg.permutations(perm)
    original_biased_points = [ordering[i] for i in ranked_training_points_in_original_permutation[settings]][:num_points_removed_dict[settings]]
    s = set(original_biased_points)
    assert(len(s) == len(original_biased_points))       # each point is unique 
    for obp in original_biased_points:
        biasness_of_each_point[obp] += 1

    # if not (perm, batch) in ranked_points_after_reversing_permutation:
    #     ranked_points_after_reversing_permutation[(perm, batch)] = set([ordering[i] for i in ranked_training_points_in_original_permutation[settings]][:num_points_removed_dict[settings]])       # we only want upto the point which achieved the minimum discrimination
    # else:
    #     ranked_points_after_reversing_permutation[(perm, batch)] = ranked_points_after_reversing_permutation[(perm, batch)].intersection([ordering[i] for i in ranked_training_points_in_original_permutation[settings]][:num_points_removed_dict[settings]])
    # print(len(x))

# with open("intersections.csv", "a") as f:
#     for key, obj in ranked_points_after_reversing_permutation.items():
#         f.write(f'{key}:{obj}\n')
sorted_biased_points = {k: v for k, v in sorted(biasness_of_each_point.items(), key=lambda item: item[1]) if v > 0}
count = 0
biased_points = []
write = False

# with open("intersections.csv", "a") as f:
#     allowed = 240 - len(bad_points)
#     for key, obj in sorted_biased_points.items():
#         if obj > 0:
#             assert(obj <= allowed)
#             count += 1
#             biased_points.append(key)
#     #         f.write(f'{key}:{obj}\n')
#     # f.write(f"Total count: {count}")
# print(biased_points)
points = []
occurence = []
for key, item in sorted_biased_points.items():
    points.append(key)
    occurence.append(item * 100/203)

feed = {'Points':points, "Occurence":occurence}
feed['Points'] = [i for i in range(len(feed['Points']))]
import pandas as pd
# import ipdb; ipdb.set_trace()
df = pd.DataFrame.from_dict(feed)
from plotnine import *
# import ipdb; ipdb.set_trace()
x = (ggplot(aes(x='Points', y='Occurence'), data=df) +\
    geom_point(size=1.5) +\
    # stat_smooth(colour='blue', span=0.2) +\
    # stat_summary() +\
    ylim(0, 100) +\
    # facet_wrap(['H1Units','H2units','Batch'], nrow=3, ncol=4,scales = 'free', labeller='label_both', shrink=False) + \
    xlab("Data-Points in order of occurence (not showing order in dataset") + \
    ylab("Percentage Occurence") + \
    ggtitle("Plot showing percentage labelling of a point as biased among all settings") +\
    theme(axis_text_x = element_text(size=6), dpi=10000) +\
    theme_seaborn()
    )

x.save("biased_points.png")#, height=12, width=12)

