import math
import copy

def ProcessCacheCausal(self, inp, X):
    # For the inputs in cache, it checks if there exists one with the same values of features other than
    # in the discriminating feature X, and has a different output from "inp". If such a input is found, it is returned. This is the discriminating test
    # Else it returns False, in which case we generate random values and test
    for cache_tuple in self.cache.items():
        i = 0
        match = True
        j = 0
        while j < len(inp):
            if j in X:
                j += 1
                continue
            if(not (cache_tuple[0][j] == inp[j])):
                match = False
                break
            i += 1
            j += 1
        if match == True and (not (cache_tuple[1] == self.cache[tuple(inp)])):
            return (True, cache_tuple)
    
    return (False, None)




def merge(inp, attr, X):
    i = 0
    while i < len(inp):
        if i in X:
            inp[i] = attr[X.index(i)]
        i += 1
    return inp



def causalDiscrimination (self, X, confidence, epsilon):
    count, score, r = 0, 0, 0

    numValues = 1
    for x in X:
        numValues *= self.num[x]        # 2 for gender
    assert(numValues == 2)      # as we have gender for now
    # assert(self.num = )
    # import ipdb; ipdb.set_trace()
    while r < self.MaxSamples:
        inp = self.randomInput(self.num, [], [])
        x = tuple(inp)
        if x in self.cache:         # if the result is already in cache, simply print it
            out = self.cache[x]
        else:
            res, out = self.SoftwareTest(inp, self.num, self.values)     # else compute the result and then append it to the list
            self.cache[x] = out        # out is a boolean value 
        # old_inp = copy.deepcopy(x)
        found_in, matching_value = ProcessCacheCausal(self, inp, X)     # this is very inefficient code
        # Process cache to find atleast one with not out as output
        
        if r > self.SamplingThreshold:
            score = count * 1.0 / r
            compare_value = self.conf_zValue[int(100 * confidence)] * math.sqrt(score * (1-score) * 1.0 / r)
            # print(compare_value)
            if (compare_value < epsilon):    # this is the formula directly
                print("epsilon: ", epsilon, "compare: ", compare_value)
                break
        
        if found_in:       # if such a tuple was found, for this random input then generate a new input
            count += 1       # count is the number of such inputs found to be discriminating
            r += 1      # r is total inputs generated
            self.causal_tests.append((x, matching_value[0])) # matching_value[0] is the test input whose output is different from inp
            continue
        
        # the above was for results present in cache else do it for all possible combinations of inputs 
        i = 0
        # found = False
        while i < numValues:
            attr = self.decodeValues(i, self.num, X)
            new_inp = merge(inp, attr, X)
            y = tuple(new_inp)
            if(y in self.cache.keys()):
                tmpout = self.cache[y]
            else:
                res2, tmpout = self.SoftwareTest(new_inp, self.num, self.values)
                self.cache[y] = tmpout
            
            if (not (tmpout == out)):
                # found = True
                i, x1, y1 = 0, '', ''       # you need to convert them to actual values
                while i < len(inp):
                    x1 += str(self.values[i][x[i]])
                    y1 += str(self.values[i][y[i]])
                    if i < len(inp) - 1:
                        x1 += ", "
                        y1 += ", "
                    i += 1
                self.causal_tests.append((x1, y1))
                # break
                count += 1
            
            i += 1
        
        # if found:
            # count += 1

            r += 1
    
    # print("no of samples: ", r)
    return score

