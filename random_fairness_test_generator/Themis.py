# Target 1: generate test cases just for 1 feature, in this case gender which is feature no 0 - Done
# Target 2: Increase the number of test cases generated by:     - DONE
#           a. Increasing the sampling threshold    - removed out of picture
#           b. Increasing the max sampling -- increasing this increases the no of tests by same factor - DONE
#           c. Do not break out of the loop when generating values of all inputs, in the causal.py file - DONE
# Target 3: Port this code to Python3 - DONE
# Target 4: integrate this with the trained model for german credit dataset.


import sys, copy
import itertools
# import commands
import subprocess
import random
import math
import pandas as pd


class soft:

    def __init__(self, names, values, num, command, type_discm, magnt_similar_range):
        self.attr_names = names
        self.values = values
        self.num = num
        self.type = type_discm
        self.command = command
        self.magnt_similar_range = magnt_similar_range
        self.dist_similarity_test_times = 2


    def SoftwareTest(self, inp, num, values):
        i=0
        actual_inp = []
        running_command = self.command
        # import ipdb; ipdb.set_trace()
        arguments = ''
        while i < len(inp):
            actual_inp.append(values[i][inp[i]])
            arguments += str(values[i][inp[i]])
            if i < len(inp) - 1:
                arguments += ", "
            i += 1
        
        rc = running_command.split()        # split into a list and send it
        rc.append(arguments)
        result = subprocess.check_output(rc)
        # import ipdb; ipdb.set_trace()
        result = result.decode("utf-8").rstrip()       # convert binary to str
        print(type(result), result, result[-1] == "0", result[-1] == "1")
        # print(result)
        return result, result[-1] == "1"


    def randomInput_class0(self, discm_feature):
        inp = []
        for i in range(len(self.attr_names)):
            if not i == discm_feature:
                inp.append(int(random.choice(self.values[i]))) 
            else:
                inp.append(0)       # this is for gender == 0
        return inp


    def find_val_within_range(self, inp1, discm_feature, discm_feature_value, times):
        inputs = []
        # import ipdb; ipdb.set_trace()
        min_xs = []
        max_xs = []
        for sim in range(times):
            inp2 = []
            for i in range(len(self.attr_names)):
                if not i == discm_feature:
                    lower_vl = inp1[i] - self.magnt_similar_range[i]
                    if sim == 0:
                        x = [int(some) for some in self.values[i]]
                        min_xs.append(min(x))
                        max_xs.append(max(x))
                    if lower_vl < min_xs[i]:
                        lower_vl = min_xs[i]
                    upper_vl = inp1[i] + self.magnt_similar_range[i]
                    if upper_vl > max_xs[i]:
                        upper_vl = max_xs[i]
                    inp2.append(round(random.uniform(lower_vl, upper_vl)))
                else:
                    if sim == 0:
                        x = [int(some) for some in self.values[i]]
                        min_xs.append(min(x))
                        max_xs.append(max(x))
                    inp2.append(discm_feature_value)       # this is for gender == 0
            inputs.append(inp2)
        return inputs


    def find_val_within_range_adult(self, inp1, discm_feature, discm_feature_value, times):
        inputs = []
        # import ipdb; ipdb.set_trace()
        numerical = 2
        lower_vl = inp1[numerical] - self.magnt_similar_range[numerical]
        if lower_vl < int(self.values[numerical][0]):
            lower_vl = int(self.values[numerical][0])
        upper_vl = inp1[numerical] + self.magnt_similar_range[numerical]
        if upper_vl > int(self.values[numerical][-1]):
            upper_vl = int(self.values[numerical][-1])

        for sim in range(times):
            inp2 = copy.deepcopy(inp1)
            inp2[discm_feature] = discm_feature_value
            inp2[numerical] = round(random.uniform(lower_vl, upper_vl))

            # for i in range(len(self.attr_names)):
            #     if not i == discm_feature:
            #         lower_vl = inp1[i] - self.magnt_similar_range[i]
            #         if sim == 0:
            #             x = [int(some) for some in self.values[i]]
            #             min_xs.append(min(x))
            #             max_xs.append(max(x))
            #         if lower_vl < min_xs[i]:
            #             lower_vl = min_xs[i]
            #         upper_vl = inp1[i] + self.magnt_similar_range[i]
            #         if upper_vl > max_xs[i]:
            #             upper_vl = max_xs[i]
            #         inp2.append(round(random.uniform(lower_vl, upper_vl)))
            #     else:
            #         if sim == 0:
            #             x = [int(some) for some in self.values[i]]
            #             min_xs.append(min(x))
            #             max_xs.append(max(x))
            #         inp2.append(discm_feature_value)       # this is for gender == 0
            inputs.append(inp2)
        return inputs


    def single_feature_discm_adult(self, feature, theta, confidence, epsilon, type_discm):
        assert(isinstance(feature, int))
        assert(feature <= len(self.attr_names))
        # score = self.causalDiscrimination([feature], confidence, epsilon)
        # print("No. of discriminating tests: ", len(self.causal_tests), "Score: ", score)
        discm_tests_gender0 = []
        total = 0
        with open("gender0_adult.csv", "a") as f:
            f.write("age,workclass,fnlwgt,education,marital-status,occupation,race,sex,capitalgain,capitalloss,hoursperweek,native-country\n")
        
        while True:
            new = self.randomInput_class0(feature)
            # if not new in discm_tests_gender0:      # its fine for 2 or more tests to be identical, we generate it randomly
            discm_tests_gender0.append(new)
            total += 1
            x = len(discm_tests_gender0) 
            if x == 1000000:
            # if x == self.MaxSamples:
                print(total, "hello")
                with open("gender0_adult.csv", "a") as f:
                    for i in discm_tests_gender0:
                        f.write(str(i)[1:-1].replace(" ", "") + "\n")
                discm_tests_gender0 = []
            
            if total == 45222*100:
            # if total == self.MaxSamples:                
                with open("gender0_adult.csv", "a") as f:
                    for i in discm_tests_gender0:
                        f.write(str(i)[1:-1].replace(" ", "") + "\n")       # remove space
                break

        # check if any tests are duplicated:
        df = pd.read_csv("gender0_adult.csv")
        x = df.duplicated()
        print(x.any(), "see duplication")     # if this is False, we are all good.
        # np.where(x) # if this is an empty list we are good, For adult we are good.
        
        # This generates same examples for the other gender
        
        # df = pd.read_csv("gender0_adult.csv")
        df['sex'] = 1
        df.to_csv("gender1_adult.csv", index=False)
        
        # with open("gender0_adult.csv", "a") as f:
        #     for i in discm_tests_gender0:
        #         f.write(str(i)[1:-1] + "\n")

        # if score > theta:
            # print("Discriminates against: ", self.attr_names[feature])


    def single_feature_discm_adult_dist(self, feature, theta, confidence, epsilon, type_discm):
        assert(isinstance(feature, int))
        assert(feature <= len(self.attr_names))
        # score = self.causalDiscrimination([feature], confidence, epsilon)
        # print("No. of discriminating tests: ", len(self.causal_tests), "Score: ", score)
        discm_tests_gender0 = []
        total = 0
        file0 = "gender0_adult_dist10.csv"
        file1 = "gender1_adult_dist10.csv"
        with open(file0, "w") as f:
            f.write("age,workclass,fnlwgt,education,marital-status,occupation,race,sex,capitalgain,capitalloss,hoursperweek,native-country\n")
        with open(file1, "w") as f:
            f.write("age,workclass,fnlwgt,education,marital-status,occupation,race,sex,capitalgain,capitalloss,hoursperweek,native-country\n")            
        
        while True:
            new = self.randomInput_class0(feature)
            # if not new in discm_tests_gender0:      # its fine for 2 or more tests to be identical, we generate it randomly
            discm_tests_gender0.append(new)
            total += 1
            x = len(discm_tests_gender0) 
            
            if total == 45222 * 100:
                similars = []
                for cnt, i in enumerate(discm_tests_gender0):
                    # for _ in range(10):     # each datapoint get printed 10 times
                    similar_inputs = self.find_val_within_range_adult(i, feature, 1, 10)
                    for sims in similar_inputs:    
                        similars.append(sims)
                    if cnt % 100 == 0:
                        print(cnt, "done")
                
                assert len(similars) == 10 * len(discm_tests_gender0)
                with open(file1, "a") as f2:
                    for prt in similars:
                        f2.write(str(prt)[1:-1].replace(" ", "") + "\n")       # remove space
                
                with open(file0, "a") as f1:
                    for cnt, i in enumerate(discm_tests_gender0):
                        for _ in range(10):     # each datapoint get printed 10 times
                            f1.write(str(i)[1:-1].replace(" ", "") + "\n")       # remove space

                break

            
            if total % 100 == 0:
                print(total, "done")

        df = pd.read_csv(file0)
        assert df['sex'].unique() == 0
        df = pd.read_csv(file1)
        assert df['sex'].unique() == 1


    def single_feature_discm_adult_race(self, feature, theta, confidence, epsilon, type_discm):
        assert(isinstance(feature, int))
        assert(feature <= len(self.attr_names))
        # score = self.causalDiscrimination([feature], confidence, epsilon)
        # print("No. of discriminating tests: ", len(self.causal_tests), "Score: ", score)
        discm_tests_gender0 = []
        total = 0
        with open("race0_adult.csv", "a") as f:
            f.write("age,workclass,fnlwgt,education,marital-status,occupation,race,sex,capitalgain,capitalloss,hoursperweek,native-country\n")
        
        while True:
            new = self.randomInput_class0(feature)
            # if not new in discm_tests_gender0:      # its fine for 2 or more tests to be identical, we generate it randomly
            discm_tests_gender0.append(new)
            total += 1
            x = len(discm_tests_gender0) 
            if x == 100000:
            # if x == self.MaxSamples:
                print(total, "hello")
                with open("race0_adult.csv", "a") as f:
                    for i in discm_tests_gender0:
                        f.write(str(i)[1:-1].replace(" ", "") + "\n")
                discm_tests_gender0 = []
            
            if total == 43131*100:
            # if total == self.MaxSamples:                
                with open("race0_adult.csv", "a") as f:
                    for i in discm_tests_gender0:
                        f.write(str(i)[1:-1].replace(" ", "") + "\n")       # remove space
                break

        # check if any tests are duplicated:
        df = pd.read_csv("race0_adult.csv")
        x = df.duplicated()
        print(x.any(), "see duplication")     # if this is False, we are all good.
        # np.where(x) # if this is an empty list we are good, For adult we are good.
        
        # This generates same examples for the other gender
        
        # df = pd.read_csv("gender0_adult.csv")
        df['race'] = 1
        df.to_csv("race1_adult.csv", index=False)

    
    def single_feature_discm_adult_race_dist(self, feature, theta, confidence, epsilon, type_discm):
        assert(isinstance(feature, int))
        assert(feature <= len(self.attr_names))
        # score = self.causalDiscrimination([feature], confidence, epsilon)
        # print("No. of discriminating tests: ", len(self.causal_tests), "Score: ", score)
        discm_tests_gender0 = []
        total = 0
        file0 = "race0_adult_dist10.csv"
        file1 = "race1_adult_dist10.csv"
        with open(file0, "w") as f:
            f.write("age,workclass,fnlwgt,education,marital-status,occupation,race,sex,capitalgain,capitalloss,hoursperweek,native-country\n")
        with open(file1, "w") as f:
            f.write("age,workclass,fnlwgt,education,marital-status,occupation,race,sex,capitalgain,capitalloss,hoursperweek,native-country\n")            
        
        while True:
            new = self.randomInput_class0(feature)
            # if not new in discm_tests_gender0:      # its fine for 2 or more tests to be identical, we generate it randomly
            discm_tests_gender0.append(new)
            total += 1
            x = len(discm_tests_gender0) 
            
            if total == 43131 * 100:
                similars = []
                for cnt, i in enumerate(discm_tests_gender0):
                    similar_inputs = self.find_val_within_range_adult(i, feature, 1, self.dist_similarity_test_times)
                    for sims in similar_inputs:    
                        similars.append(sims)
                    if cnt % 100 == 0:
                        print(cnt, "done")
                
                assert len(similars) == self.dist_similarity_test_times * len(discm_tests_gender0)
                with open(file1, "a") as f2:
                    for prt in similars:
                        f2.write(str(prt)[1:-1].replace(" ", "") + "\n")       # remove space
                
                with open(file0, "a") as f1:
                    for cnt, i in enumerate(discm_tests_gender0):
                        for _ in range(10):     # each datapoint get printed 10 times
                            f1.write(str(i)[1:-1].replace(" ", "") + "\n")       # remove space

                break
            
            if total % 100 == 0:
                print(total, "doing")

        df = pd.read_csv(file0)
        assert df['race'].unique() == 0
        df = pd.read_csv(file1)
        assert df['race'].unique() == 1


    def single_feature_discm_salary(self, feature, theta, confidence, epsilon, type_discm):
        assert(isinstance(feature, int))
        assert(feature <= len(self.attr_names))
        discm_tests_gender0 = []
        total = 0
        with open("sex0_salary.csv", "a") as f:
            f.write("sex,rank,year,degree,Experience\n")
        
        while True:
            new = self.randomInput_class0(feature)
            discm_tests_gender0.append(new)
            total += 1
            # x = len(discm_tests_gender0)    
            if total == 52*100:             
                with open("sex0_salary.csv", "a") as f:
                    for i in discm_tests_gender0:
                        f.write(str(i)[1:-1].replace(" ", "") + "\n")       # remove space
                break

        # check if any tests are duplicated:
        df = pd.read_csv("sex0_salary.csv")
        x = df.duplicated()
        print(x.any(), "see duplication")     # if this is False, we are all good.
        # np.where(x) # if this is an empty list we are good, For adult we are good.

        # This generates same examples for the other class

        # df = pd.read_csv("gender0_adult.csv")
        df['sex'] = 1
        df.to_csv("sex1_salary.csv", index=False)    


    def single_feature_discm_salary_dist(self, feature, theta, confidence, epsilon, type_discm):
        assert(isinstance(feature, int))
        assert(feature <= len(self.attr_names))
        discm_tests_gender0 = []
        total = 0
        file0 = "sex0_salary_dist10.csv"
        file1 = "sex1_salary_dist10.csv"
        with open(file0, "w") as f:
            f.write("sex,rank,year,degree,Experience\n")
        with open(file1, "w") as f:
            f.write("sex,rank,year,degree,Experience\n")

        while True:
            new = self.randomInput_class0(feature)     # random sample one datapoint
            discm_tests_gender0.append(new)
            total += 1
            if total == 52 * 100:
                for i in discm_tests_gender0:
                    for prt in range(10):     # each datapoint get printed 10 times
                        similar_inp = self.find_val_within_range(i, feature, 1)
                        with open(file1, "a") as f2:
                            f2.write(str(similar_inp)[1:-1].replace(" ", "") + "\n")       # remove space
                    for prt in range(10):     # each datapoint get printed 10 times
                        with open(file0, "a") as f1:
                            f1.write(str(i)[1:-1].replace(" ", "") + "\n")       # remove space
                break

        df = pd.read_csv(file0)
        assert df['sex'].unique() == 0
        df = pd.read_csv(file1)
        assert df['sex'].unique() == 1


    def single_feature_discm_german(self, feature, theta, confidence, epsilon, type_discm):
        assert(isinstance(feature, int))
        assert(feature <= len(self.attr_names))     # feature is the sensitive feature
        # score = self.causalDiscrimination([feature], confidence, epsilon)
        # print("No. of discriminating tests: ", len(self.causal_tests), "Score: ", score)
        discm_tests_gender0 = []
        total = 0
        with open("gender0_german.csv", "a") as f:
            f.write("Checking-ccount,Months,Credit-history,Purpose,Credit-mount,Svings-ccount,Present-employment-since,Instllment-rte,Gender,Other-debtors,Present-residence-since,Property,ge,Other-instllment-plns,Housing,Number-of-existing-credits,Job,Number-of-people-being-lible,Telephone,Foreign-worker\n")
        
        while True:
            new = self.randomInput_class0(feature)
            # if not new in discm_tests_gender0:      # its fine for 2 or more tests to be identical, we generate it randomly
            discm_tests_gender0.append(new)
            total += 1
            x = len(discm_tests_gender0) 
            if x == 10000:
            # if x == self.MaxSamples:
                print(total, "hello")
                with open("gender0_german.csv", "a") as f:
                    for i in discm_tests_gender0:
                        f.write(str(i)[1:-1].replace(" ", "") + "\n")
                discm_tests_gender0 = []
            
            if total == 100000:
            # if total == self.MaxSamples:                
                with open("gender0_german.csv", "a") as f:
                    for i in discm_tests_gender0:
                        f.write(str(i)[1:-1].replace(" ", "") + "\n")       # remove space
                break

        # check if any tests are duplicated:
        df = pd.read_csv("gender0_german.csv")
        x = df.duplicated()
        print(x.any(), "see duplication")     # if this is False, we are all good.
        # np.where(x) # if this is an empty list we are good, For adult we are good.
        
        # This generates same examples for the other gender
        
        # df = pd.read_csv("gender0_german.csv")
        df['Gender'] = 1
        df.to_csv("gender1_german.csv", index=False)
        
        # with open("gender0_german.csv", "a") as f:
        #     for i in discm_tests_gender0:
        #         f.write(str(i)[1:-1] + "\n")

        # if score > theta:
            # print("Discriminates against: ", self.attr_names[feature])


    def single_feature_discm_german_dist(self, feature, theta, confidence, epsilon, type_discm):
        assert(isinstance(feature, int))
        assert(feature <= len(self.attr_names))     # feature is the sensitive feature
        discm_tests_gender0 = []
        total = 0
        file0 = "gender0_german_dist10.csv"
        file1 = "gender1_german_dist10.csv"
        with open(file0, "w") as f:
            f.write("Checking-ccount,Months,Credit-history,Purpose,Credit-mount,Svings-ccount,Present-employment-since,Instllment-rte,Gender,Other-debtors,Present-residence-since,Property,ge,Other-instllment-plns,Housing,Number-of-existing-credits,Job,Number-of-people-being-lible,Telephone,Foreign-worker\n")
        with open(file1, "w") as f:
            f.write("Checking-ccount,Months,Credit-history,Purpose,Credit-mount,Svings-ccount,Present-employment-since,Instllment-rte,Gender,Other-debtors,Present-residence-since,Property,ge,Other-instllment-plns,Housing,Number-of-existing-credits,Job,Number-of-people-being-lible,Telephone,Foreign-worker\n")
            
        while True:
            new = self.randomInput_class0(feature)
            discm_tests_gender0.append(new)
            total += 1
            x = len(discm_tests_gender0) 
            
            if total == 100000:
                similars = []
                for cnt, i in enumerate(discm_tests_gender0):
                    # for _ in range(10):     # each datapoint get printed 10 times
                    similar_inputs = self.find_val_within_range(i, feature, 1, 10)
                    for sims in similar_inputs:    
                        similars.append(sims)
                    if cnt % 100 == 0:
                        print(cnt, "done")
                
                assert len(similars) == 10 * len(discm_tests_gender0)
                with open(file1, "a") as f2:
                    for prt in similars:
                        f2.write(str(prt)[1:-1].replace(" ", "") + "\n")       # remove space
                
                with open(file0, "a") as f1:
                    for cnt, i in enumerate(discm_tests_gender0):
                        for _ in range(10):     # each datapoint get printed 10 times
                            f1.write(str(i)[1:-1].replace(" ", "") + "\n")       # remove space

                break

        df = pd.read_csv(file0)
        assert df['Gender'].unique() == 0
        df = pd.read_csv(file1)
        assert df['Gender'].unique() == 1


    def single_feature_discm_small_dataset(self, feature, theta, confidence, epsilon, type_discm):
        assert(isinstance(feature, int))
        assert(feature <= len(self.attr_names))
        # score = self.causalDiscrimination([feature], confidence, epsilon)
        # print("No. of discriminating tests: ", len(self.causal_tests), "Score: ", score)
        discm_tests_gender0 = []
        total = 0
        with open("race0_biased_smalldataset.csv", "a") as f:
            f.write("Income,Neighbor-income,Race\n")
        
        while True:
            new = self.randomInput_class0(feature)
            discm_tests_gender0.append(new)
            total += 1

            if total == 700:              
                with open("race0_biased_smalldataset.csv", "a") as f:
                    for i in discm_tests_gender0:
                        f.write(str(i)[1:-1].replace(" ", "") + "\n")       # remove space
                break

        # check if any tests are duplicated:
        df = pd.read_csv("race0_biased_smalldataset.csv")
        x = df.duplicated()
        print(x.any(), "see duplication")     # if this is False, we are all good.
        
        # This generates same examples for the other gender
        df['Race'] = 1
        df.to_csv("race1_biased_smalldataset.csv", index=False)
        

    def single_feature_discm_compas(self, feature, theta, confidence, epsilon, type_discm):
        assert(isinstance(feature, int))
        assert(feature <= len(self.attr_names))
        # score = self.causalDiscrimination([feature], confidence, epsilon)
        # print("No. of discriminating tests: ", len(self.causal_tests), "Score: ", score)
        discm_tests_gender0 = []
        total = 0
        with open("race0_compas.csv", "a") as f:
            f.write("sex,age,race,juv_fel_count,juv_misd_count,juv_other_count,priors_count,days_b_screening_arrest,c_days_from_compas,c_charge_degree\n")
        
        while True:
            new = self.randomInput_class0(feature)
            # if not new in discm_tests_gender0:      # its fine for 2 or more tests to be identical, we generate it randomly
            discm_tests_gender0.append(new)
            total += 1
            x = len(discm_tests_gender0) 
            
            if total == 1000000:            
                with open("race0_compas.csv", "a") as f:
                    for i in discm_tests_gender0:
                        f.write(str(i)[1:-1].replace(" ", "") + "\n")       # remove space
                break

        # check if any tests are duplicated:
        df = pd.read_csv("race0_compas.csv")
        x = df.duplicated()
        print(x.any(), "see duplication")     # if this is False, we are all good.
        # np.where(x) # if this is an empty list we are good, For adult we are good.
        
        # This generates same examples for the other demographic group
        df['race'] = 1
        df.to_csv("race1_compas.csv", index=False)
        
        # with open("gender0_adult.csv", "a") as f:
        #     for i in discm_tests_gender0:
        #         f.write(str(i)[1:-1] + "\n")

        # if score > theta:
            # print("Discriminates against: ", self.attr_names[feature])

    
    def single_feature_discm_compas_two_year(self, feature, theta, confidence, epsilon, type_discm):
        assert(isinstance(feature, int))
        assert(feature <= len(self.attr_names))
        discm_tests_gender0 = []
        total = 0
        with open("race0_compas_two_year.csv", "a") as f:
            f.write("age,sex,race,diff_custody,diff_jail,priors_count,juv_fel_count,juv_misd_count,juv_other_count,c_charge_degree\n")
            # f.write("sex,age,race,juv_fel_count,decile_score,juv_misd_count,juv_other_count,priors_count,days_b_screening_arrest,c_days_from_compas,c_charge_degree,is_recid,is_violent_recid,decile_score.1,v_decile_score,priors_count.1,start,end,event\n")
        
        while True:
            new = self.randomInput_class0(feature)
            discm_tests_gender0.append(new)
            total += 1
            x = len(discm_tests_gender0) 
            
            if total == 615000:
                with open("race0_compas_two_year.csv", "a") as f:
                    for i in discm_tests_gender0:
                        f.write(str(i)[1:-1].replace(" ", "") + "\n")       # remove space
                # discm_tests_gender0 = []
                print(total, "done")
                break
            
            # if total == 615000:
            #     assert(x % 10000 == 0)
            #     break

        # check if any tests are duplicated:
        df = pd.read_csv("race0_compas_two_year.csv")
        x = df.duplicated()
        print(x.any(), "see duplication")     # if this is False, we are all good.
        
        # This generates same examples for the other demographic group
        df['race'] = 1
        df.to_csv("race1_compas_two_year.csv", index=False)
        
    
    def single_feature_discm_compas_two_year_dist(self, feature, theta, confidence, epsilon, type_discm):
        assert(isinstance(feature, int))
        assert(feature <= len(self.attr_names))
        discm_tests_gender0 = []
        total = 0
        file0 = "race0_compas_two_year_dist10.csv"
        file1 = "race1_compas_two_year_dist10.csv"
        with open(file0, "w") as f:
            f.write("age,sex,race,diff_custody,diff_jail,priors_count,juv_fel_count,juv_misd_count,juv_other_count,c_charge_degree\n")
        with open(file1, "w") as f:
            f.write("age,sex,race,diff_custody,diff_jail,priors_count,juv_fel_count,juv_misd_count,juv_other_count,c_charge_degree\n")
        
        while True:
            new = self.randomInput_class0(feature)
            discm_tests_gender0.append(new)
            total += 1
            x = len(discm_tests_gender0) 
            
            if total == 615000:
                for cnt, i in enumerate(discm_tests_gender0):
                    for prt in range(10):     # each datapoint get printed 10 times
                        similar_inp = self.find_val_within_range(i, feature, 1)
                        with open(file1, "a") as f2:
                            f2.write(str(similar_inp)[1:-1].replace(" ", "") + "\n")       # remove space
                    for prt in range(10):     # each datapoint get printed 10 times
                        with open(file0, "a") as f1:
                            f1.write(str(i)[1:-1].replace(" ", "") + "\n")       # remove space
                    if cnt % 100 == 0:
                        print(cnt, "done")
                break

        df = pd.read_csv(file0)
        assert df['race'].unique() == 0
        df = pd.read_csv(file1)
        assert df['race'].unique() == 1


    def single_feature_discm_student(self, feature, theta, confidence, epsilon, type_discm):
        assert(isinstance(feature, int))
        assert(feature <= len(self.attr_names))
        discm_tests_gender0 = []
        total = 0
        with open("sex0_student.csv", "a") as f:
            f.write("school,sex,age,address,famsize,Pstatus,Medu,Fedu,Mjob,Fjob,reason,guardian,traveltime,studytime,failures,schoolsup,famsup,paid,activities,nursery,higher,internet,romantic,famrel,freetime,goout,Dalc,Walc,health,absences,G1,G2\n")
        
        while True:
            new = self.randomInput_class0(feature)
            discm_tests_gender0.append(new)
            total += 1
            x = len(discm_tests_gender0) 
            
            if total == 64900: 
                with open("sex0_student.csv", "a") as f:
                    for i in discm_tests_gender0:
                        f.write(str(i)[1:-1].replace(" ", "") + "\n")       # remove space
                break

        # check if any tests are duplicated:
        df = pd.read_csv("sex0_student.csv")
        x = df.duplicated()
        print(x.any(), "see duplication")     # if this is False, we are all good.
        if not x.any():
            # This generates same examples for the other demographic group, after flipping that attribute
            df['sex'] = 1
            df.to_csv("sex1_student.csv", index=False)
        

    def single_feature_discm_student_dist(self, feature, theta, confidence, epsilon, type_discm):
        assert(isinstance(feature, int))
        assert(feature <= len(self.attr_names))
        discm_tests_gender0 = []
        total = 0
        file0 = "sex0_student_dist10.csv"
        file1 = "sex1_student_dist10.csv"
        with open(file0, "w") as f:
            f.write("school,sex,age,address,famsize,Pstatus,Medu,Fedu,Mjob,Fjob,reason,guardian,traveltime,studytime,failures,schoolsup,famsup,paid,activities,nursery,higher,internet,romantic,famrel,freetime,goout,Dalc,Walc,health,absences,G1,G2\n")
        with open(file1, "w") as f:
            f.write("school,sex,age,address,famsize,Pstatus,Medu,Fedu,Mjob,Fjob,reason,guardian,traveltime,studytime,failures,schoolsup,famsup,paid,activities,nursery,higher,internet,romantic,famrel,freetime,goout,Dalc,Walc,health,absences,G1,G2\n")

        while True:
            new = self.randomInput_class0(feature)
            discm_tests_gender0.append(new)
            total += 1
            x = len(discm_tests_gender0) 
            
            if total == 64900: 
                for i in discm_tests_gender0:
                    for prt in range(10):     # each datapoint get printed 10 times
                        similar_inp = self.find_val_within_range(i, feature, 1)
                        with open(file1, "a") as f2:
                            f2.write(str(similar_inp)[1:-1].replace(" ", "") + "\n")       # remove space
                    for prt in range(10):     # each datapoint get printed 10 times
                        with open(file0, "a") as f1:
                            f1.write(str(i)[1:-1].replace(" ", "") + "\n")       # remove space
                break
            
            if total % 100 == 0:
                print(total, "done")

        df = pd.read_csv(file0)
        assert df['sex'].unique() == 0
        df = pd.read_csv(file1)
        assert df['sex'].unique() == 1


    def single_feature_discm_default(self, feature, theta, confidence, epsilon, type_discm):
        assert(isinstance(feature, int))
        assert(feature <= len(self.attr_names))
        # score = self.causalDiscrimination([feature], confidence, epsilon)
        # print("No. of discriminating tests: ", len(self.causal_tests), "Score: ", score)
        discm_tests_gender0 = []
        total = 0
        with open("sex0_default.csv", "a") as f:
            f.write("LIMIT_BAL,sex,EDUCATION,MARRIAGE,AGE,PAY_0,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6,BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6,PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6\n")
        
        while True:
            new = self.randomInput_class0(feature)
            discm_tests_gender0.append(new)
            total += 1
            x = len(discm_tests_gender0) 
            if x == 1000000:
                print(total, "hello")
                with open("sex0_default.csv", "a") as f:
                    for i in discm_tests_gender0:
                        f.write(str(i)[1:-1].replace(" ", "") + "\n")
                discm_tests_gender0 = []
            
            if total == 3000000:
                if not x == 1000000:    # only if there are cases not pushed into random test suite 
                    with open("sex0_default.csv", "a") as f:
                        for i in discm_tests_gender0:
                            f.write(str(i)[1:-1].replace(" ", "") + "\n")       # remove space
                break

        # check if any tests are duplicated:
        df = pd.read_csv("sex0_default.csv")
        x = df.duplicated()
        print(x.any(), "see duplication", df.shape)     # if this is False, we are all good.
        if not x.any():
            # This generates same examples for the other gender
            df['sex'] = 1
            df.to_csv("sex1_default.csv", index=False)
        

    def single_feature_discm_default_dist(self, feature, theta, confidence, epsilon, type_discm):
        assert(isinstance(feature, int))
        assert(feature <= len(self.attr_names))
        # score = self.causalDiscrimination([feature], confidence, epsilon)
        # print("No. of discriminating tests: ", len(self.causal_tests), "Score: ", score)
        discm_tests_gender0 = []
        total = 0
        file0 = "sex0_default_dist10.csv"
        file1 = "sex1_default_dist10.csv"
        with open(file0, "w") as f:
            f.write("LIMIT_BAL,sex,EDUCATION,MARRIAGE,AGE,PAY_0,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6,BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6,PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6\n")
        with open(file1, "w") as f:
            f.write("LIMIT_BAL,sex,EDUCATION,MARRIAGE,AGE,PAY_0,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6,BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6,PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6\n")
        
        while True:
            new = self.randomInput_class0(feature)
            discm_tests_gender0.append(new)
            total += 1
            x = len(discm_tests_gender0) 
            
            if total == 3000000:
                for cnt, i in enumerate(discm_tests_gender0):
                    for prt in range(10):     # each datapoint get printed 10 times
                        similar_inp = self.find_val_within_range(i, feature, 1)
                        with open(file1, "a") as f2:
                            f2.write(str(similar_inp)[1:-1].replace(" ", "") + "\n")       # remove space
                    for prt in range(10):     # each datapoint get printed 10 times
                        with open(file0, "a") as f1:
                            f1.write(str(i)[1:-1].replace(" ", "") + "\n")       # remove space
                    if cnt % 100 == 0:
                        print(cnt, "done")
                break

        df = pd.read_csv(file0)
        assert df['sex'].unique() == 0
        df = pd.read_csv(file1)
        assert df['sex'].unique() == 1
