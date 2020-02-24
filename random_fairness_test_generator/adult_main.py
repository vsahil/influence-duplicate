import sys
import random
import math
import Themis

f = open("adult_settings.txt",'r')
software_name = ""
command = ""
num_attributes = -1
names = {}
type_discm = {}
values = {}
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
        
        elif line[2] == "continuousInt":
            start = int(line[3].split(",")[0])
            end   = int(line[3].split(",")[1])
            num_values[attr_no] = end - start + 1       # range of values
            value_lst = []
            value_lst = [i for i in range(start, end+1)]
            values[attr_no] = value_lst

# print(names, values, num_values, command, type_discm)
# exit(0)
# print(names)
soft = Themis.soft(names, values, num_values, command, type_discm)
soft.single_feature_discm(7, 0.3, 0.99, 0.01, "causal")     # 9th feature is gender