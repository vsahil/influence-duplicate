import sys
import random
import math
import Themis

f = open("settings.txt",'r')
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
for line in f:
    count = count + 1	
    line = line.strip()
    if count == 1:
        line = line.strip(':')      # finds the number of features for this dataset
        nums = int(line[-1]) 
    if "command" in line and count == nums + 2:
        line = line.split(" ")
        command  = " ".join(line[1:])       # finds the command in this line
        continue
    
    if num_attributes == -1:        # just the number of features, I don't why it is used twice
        line = line.strip(':')
        num_attributes = int(line[-1])
    else:
        line = line.split(' ')
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
            # while start <= end:     # it just enumerates all the values in that range 
            #     value_lst.append(start)
            #     start += 1
            value_lst = [i for i in range(start, end+1)]
            values[attr_no] = value_lst


print(names, values, num_values, command, type_discm)
# exit(0)
soft = Themis.soft(names, values, num_values, command, type_discm)


# D = soft.discriminationSearch(0.3, 0.99, 0.1, "groupandcausal")
# D = soft.discriminationSearch(0.8, 0.99, 0.1, "causal")
soft.single_feature_discm(0, 0.3, 0.99, 0.01, "causal")
# soft.printSoftwareDetails()
# suite = soft.getTestSuite()
# print(suite)
# print("\n\n\nThemis has completed \n")
# print("Software discriminates against ",D,"\n")
# X=[0,2]
# print soft.groupDiscrimination(X,99,0.1)
# print soft.causalDiscrimination(X,99,0.1)
# a = soft.getTestSuite()