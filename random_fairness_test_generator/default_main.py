import sys
import random
import math
import Themis

# def create_setting():
#     import pandas as pd
#     df = pd.read_csv("../default-dataset/raw_default.csv")
#     # import ipdb; ipdb.set_trace()
#     for i in df.columns:
#         if df[i].dtype == "O":
#             assert False
#             df[i], mapping_index = pd.Series(df[i]).factorize()

#     target = df['target']
#     df = df.drop(columns=['target'])
#     cols = df.columns.tolist()
#     print(f"number of input characteristics: {len(cols)}")
#     for k, i in enumerate(cols):
#         mn = df[i].min()
#         mx = df[i].max()
#         uni = df[i].unique()
#         if not len(uni) == mx - mn + 1:
#             tp = "continuousInt"
#             l = [mn, mx]
#             print(k + 1, i, tp, str(l)[1:-1].replace(" ", ""))
#         else:
#             tp = "categorical"
#             print(k + 1, i, tp, str(sorted(list(uni)))[1:-1].replace(" ", ""))
#     print("command: python .py")
#     exit(0)
# create_setting()


f = open("default_settings.txt",'r')
software_name = ""
command = ""
num_attributes = -1
names = {}
type_discm = {}
values = {}
magnt_similar_range = {}      # magnitude of the value of the similar values for each feature
percentage_similardist = 10
num_values = {}
count = 0
nums = -1

# read the settings file
# import ipdb; ipdb.set_trace()
for line in f:
    count = count + 1	
    line = line.strip()
    if count == 1:
        line = line.split(':')      # finds the number of features for this dataset
        nums = int(line[-1])
        # print(line, "see", nums)
    if "command" in line and count == nums + 2:
        line = line.split(" ")
        command  = " ".join(line[1:])       # finds the command in this line
        continue
    
    if num_attributes == -1:        # just the number of features, I don't why it is used twice
        # line = line.split(':')
        num_attributes = int(line[-1])
    else:
        line = line.split(' ')
        # print(line, count, nums)
        attr_no = int(line[0]) - 1      # -1 is used because they have written count += 1 in the first line
        names[attr_no] = line[1]
        type_discm[attr_no] = line[2]
        
        if line[2] == "categorical":
            values[attr_no] = line[3].split(",")
            num_values[attr_no] = len(values[attr_no])
            magnt_similar_range[attr_no] = 0        # no change allowed for categorical features
        
        elif line[2] == "continuousInt":
            start = int(line[3].split(",")[0])
            end   = int(line[3].split(",")[1])
            num_values[attr_no] = end - start + 1       # range of values
            value_lst = []
            value_lst = [i for i in range(start, end+1)]
            values[attr_no] = value_lst
            magnt_similar_range[attr_no] = (end - start) * percentage_similardist / 100
            assert magnt_similar_range[attr_no] > 0

print(names, num_values, command, type_discm, magnt_similar_range)
# exit(0)
soft = Themis.soft(names, values, num_values, command, type_discm, magnt_similar_range)
# soft.single_feature_discm_default(1, 0.3, 0.99, 0.01, "causal")     # 2nd feature is sex
# soft.single_feature_discm_default_dist(1, 0.3, 0.99, 0.01, "causal")     # 2nd feature is sex
