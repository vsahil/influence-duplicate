import numpy as np
import pandas as pd
import sys
sys.path.append("./")
import load_adult_income as load_file

dist = load_file.dist
dataset = "adult"

ranked_training_points_in_original_permutation = {}
for model_count in range(240):
    name = f"adult_income_count{model_count}"
    ranked_training_points_in_original_permutation[model_count] = list(np.load(f"ranking_points_ordered_method1_dist{dist}/{name}.npy"))
    print(model_count, "done")

assert len(ranked_training_points_in_original_permutation[1]) == len(ranked_training_points_in_original_permutation[24])

def variation(setting_now):
    model_count = 0
    for perm in range(20):
        for h1units in [16, 24, 32]:
            for h2units in [8, 12]:
                for batch in [2048, 4096]:      # different batch sizes for this dataset
                    if model_count < setting_now:
                        model_count += 1
                        continue
                    # print(setting_now, "done", perm, h1units, h2units, batch)
                    return perm, h1units, h2units, batch, model_count

ranked_points_after_removing_permutation_effect = {}
biasness_of_each_point = {}

df = pd.read_csv(f"results_{dataset}_debiasedtrain_80percentof_total_dist{dist}.csv")
removal_df = df.sort_values(by=['Discm_percent', 'Points-removed']).groupby("Model-count", as_index=False).first()
assert len(removal_df) == 240
removal_df.to_csv(f"removal_df_{dataset}_dist{dist}.csv", index=False)

training_size = len(ranked_training_points_in_original_permutation[0])
assert(training_size == 36000)
total_dataset = len(load_file.permutations(0))

for i in range(total_dataset):  # total dataset
    biasness_of_each_point[i] = 0

for settings in range(240):
    perm, h1units, h2units, batch, model_count = variation(settings)
    assert(perm == settings//12)
    ordering = load_file.permutations(perm)
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
np.save(f"{dataset}_biased_points_dist{dist}.npy", biased_points_global)
