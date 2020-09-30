import numpy as np 


def process_row(row_split, starter_string, max_value):
    for j, i in enumerate(row_split):
        if float(i) != max_value:
            starter_string = starter_string + str(i)
        elif float(i) == max_value:
            starter_string = starter_string + " \\textbf{" + str(i) + "} "
        if j == len(row_split) - 1:
            starter_string += " \\\\"
        else:
            starter_string += " & "
    return starter_string


def test_accuracy_for_min_discm():
    file = "test_accuracy_for_min_discm.tex"
    with open(file, "r") as f:
        content = f.readlines()

    discm_techniques = {'T1':[], 'T2':[], 'T3':[] ,'T4':[] ,'T5':[] ,'T6':[] ,'T7':[] ,'T8':[]}
    rows = []
    for length, i in enumerate(content):
        j = i.split("&")
        experiment = j[0].strip()
        row = []
        j = j[1:]       # remove experiment ID
        max_value = 0
        for x, discm in enumerate(j):
            if "textbf" in discm and "\\" in discm:
                discm = discm.split("\\")[1].split("{")[1].split("}")[0]
            elif "\\" in discm:
                discm = discm.split("\\")[0]
            discm_techniques['T'+str(x+1)].append(float(discm))
            row.append(int(float(discm)))
            if float(discm) > max_value:
                max_value = int(float(discm))

        try:
            assert(len(row) == len(discm_techniques))
        except:
            if "midrule" in i and "Avg" in content[length+1]:
                return      # this file has been done
            else:
                assert False

        this_row = experiment + " & "
        this_row = process_row(row, this_row, max_value)
        rows.append(this_row)

    averages = []
    max_avg = 0
    for key in discm_techniques:
        val = int(float(np.mean(discm_techniques[key])))
        averages.append(val)
        if val > max_avg:
            max_avg = val
    avg_row = "Avg. & "
    print_avg = process_row(averages, avg_row, max_avg)

    # precision = 1
    with open(file, "w") as f:
        for i in rows:
            print(i, file=f)    
        print("\\midrule", file=f)
        print(print_avg, file=f)

    # for i in averages:
        # if 
        # with open(file, "a") as f:
            # print(int(float(f"{np.mean(discm_techniques[key]):.{precision}e}")) , end=END, file=f)


def test_accuracy_for_min_parity():
    file = "test_accuracy_for_min_parity.tex"
    with open(file, "r") as f:
        content = f.readlines()

    discm_techniques = {'T1':[], 'T2':[], 'T3':[] ,'T4':[] ,'T5':[] ,'T6':[] ,'T7':[] ,'T8':[]}
    rows = []
    for length, i in enumerate(content):
        j = i.split("&")
        experiment = j[0].strip()
        row = []
        j = j[1:]       # remove experiment ID
        max_value = 0
        for x, discm in enumerate(j):
            if "textbf" in discm and "\\" in discm:
                discm = discm.split("\\")[1].split("{")[1].split("}")[0]
            elif "\\" in discm:
                discm = discm.split("\\")[0]
            discm_techniques['T'+str(x+1)].append(float(discm))
            row.append(int(float(discm)))
            if float(discm) > max_value:
                max_value = int(float(discm))

        try:
            assert(len(row) == len(discm_techniques))
        except:
            if "midrule" in i and "Avg" in content[length+1]:
                return      # this file has been done
            else:
                assert False

        this_row = experiment + " & "
        this_row = process_row(row, this_row, max_value)
        rows.append(this_row)

    averages = []
    max_avg = 0
    for key in discm_techniques:
        val = int(float(np.mean(discm_techniques[key])))
        averages.append(val)
        if val > max_avg:
            max_avg = val
    avg_row = "Avg. & "
    print_avg = process_row(averages, avg_row, max_avg)

    # precision = 1
    with open(file, "w") as f:
        for i in rows:
            print(i, file=f)    
        print("\\midrule", file=f)
        print(print_avg, file=f)


def max_accuracy():
    file = "max-test-accuracy.tex"
    with open(file, "r") as f:
        content = f.readlines()

    discm_techniques = {'T1':[], 'T2':[], 'T3':[] ,'T4':[] ,'T5':[] ,'T6':[] ,'T7':[] ,'T8':[]}
    rows = []
    for length, i in enumerate(content):
        j = i.split("&")
        experiment = j[0].strip()
        row = []
        j = j[1:]       # remove experiment ID
        max_value = 0
        for x, discm in enumerate(j):
            if "textbf" in discm and "\\" in discm:
                discm = discm.split("\\")[1].split("{")[1].split("}")[0]
            elif "\\" in discm:
                discm = discm.split("\\")[0]
            discm_techniques['T'+str(x+1)].append(float(discm))
            row.append(int(float(discm)))
            if float(discm) > max_value:
                max_value = int(float(discm))

        try:
            assert(len(row) == len(discm_techniques))
        except:
            if "midrule" in i and "Avg" in content[length+1]:
                return      # this file has been done
            else:
                assert False

        this_row = experiment + " & "
        this_row = process_row(row, this_row, max_value)
        rows.append(this_row)

    averages = []
    max_avg = 0
    for key in discm_techniques:
        val = int(float(np.mean(discm_techniques[key])))
        averages.append(val)
        if val > max_avg:
            max_avg = val
    avg_row = "Avg. & "
    print_avg = process_row(averages, avg_row, max_avg)

    # precision = 1
    with open(file, "w") as f:
        for i in rows:
            print(i, file=f)    
        print("\\midrule", file=f)
        print(print_avg, file=f)


def parity_diff_min_discm():
    file = "parity-diff-min-discm_fulltest.tex"
    with open(file, "r") as f:
        content = f.readlines()

    discm_techniques = {'T1':[], 'T2':[], 'T3':[] ,'T4':[] ,'T5':[] ,'T6':[] ,'T7':[] ,'T8':[]}
    rows = []
    for length, i in enumerate(content):
        j = i.split("&")
        experiment = j[0].strip()
        row = []
        j = j[1:]       # remove experiment ID
        min_value = 100
        for x, discm in enumerate(j):
            if "textbf" in discm and "\\" in discm:
                discm = discm.split("\\")[1].split("{")[1].split("}")[0]
            elif "\\" in discm:
                discm = discm.split("\\")[0]
            discm_techniques['T'+str(x+1)].append(float(discm))
            row.append(float(discm))
            if float(discm) < min_value:
                min_value = float(discm)

        try:
            assert(len(row) == len(discm_techniques))
        except:
            if "midrule" in i and "Avg" in content[length+1]:
                return      # this file has been done
            else:
                assert False

        this_row = experiment + " & "
        this_row = process_row(row, this_row, min_value)
        rows.append(this_row)

    averages = []
    precision = 1
    min_avg = 100
    for key in discm_techniques:
        val = float(f"{np.mean(discm_techniques[key]):.{precision}e}")
        averages.append(val)
        if val < min_avg:
            min_avg = val
    avg_row = "Avg. & "
    print_avg = process_row(averages, avg_row, min_avg)

    # precision = 1
    with open(file, "w") as f:
        for i in rows:
            print(i, file=f)    
        print("\\midrule", file=f)
        print(print_avg, file=f)


def parity_diff_max_accuracy():
    file = "parity-diff-max-accuracy_fulltest.tex"
    with open(file, "r") as f:
        content = f.readlines()

    discm_techniques = {'T1':[], 'T2':[], 'T3':[] ,'T4':[] ,'T5':[] ,'T6':[] ,'T7':[] ,'T8':[]}
    rows = []
    for length, i in enumerate(content):
        j = i.split("&")
        experiment = j[0].strip()
        row = []
        j = j[1:]       # remove experiment ID
        min_value = 100
        for x, discm in enumerate(j):
            if "textbf" in discm and "\\" in discm:
                discm = discm.split("\\")[1].split("{")[1].split("}")[0]
            elif "\\" in discm:
                discm = discm.split("\\")[0]
            discm_techniques['T'+str(x+1)].append(float(discm))
            row.append(float(discm))
            if float(discm) < min_value:
                min_value = float(discm)

        try:
            assert(len(row) == len(discm_techniques))
        except:
            if "midrule" in i and "Avg" in content[length+1]:
                return      # this file has been done
            else:
                assert False

        this_row = experiment + " & "
        this_row = process_row(row, this_row, min_value)
        rows.append(this_row)

    averages = []
    precision = 1
    min_avg = 100
    for key in discm_techniques:
        val = float(f"{np.mean(discm_techniques[key]):.{precision}e}")
        averages.append(val)
        if val < min_avg:
            min_avg = val
    avg_row = "Avg. & "
    print_avg = process_row(averages, avg_row, min_avg)

    # precision = 1
    with open(file, "w") as f:
        for i in rows:
            print(i, file=f)    
        print("\\midrule", file=f)
        print(print_avg, file=f)


def parity_diff_min_parity():
    file = "min-parity-diff_fulltest.tex"
    with open(file, "r") as f:
        content = f.readlines()

    discm_techniques = {'T1':[], 'T2':[], 'T3':[] ,'T4':[] ,'T5':[] ,'T6':[] ,'T7':[] ,'T8':[]}
    rows = []
    for length, i in enumerate(content):
        j = i.split("&")
        experiment = j[0].strip()
        row = []
        j = j[1:]       # remove experiment ID
        min_value = 100
        for x, discm in enumerate(j):
            if "textbf" in discm and "\\" in discm:
                discm = discm.split("\\")[1].split("{")[1].split("}")[0]
            elif "\\" in discm:
                discm = discm.split("\\")[0]
            discm_techniques['T'+str(x+1)].append(float(discm))
            row.append(float(discm))
            if float(discm) < min_value:
                min_value = float(discm)

        try:
            assert(len(row) == len(discm_techniques))
        except:
            if "midrule" in i and "Avg" in content[length+1]:
                return      # this file has been done
            else:
                assert False

        this_row = experiment + " & "
        this_row = process_row(row, this_row, min_value)
        rows.append(this_row)

    averages = []
    precision = 1
    min_avg = 100
    for key in discm_techniques:
        val = float(f"{np.mean(discm_techniques[key]):.{precision}e}")
        averages.append(val)
        if val < min_avg:
            min_avg = val
    avg_row = "Avg. & "
    print_avg = process_row(averages, avg_row, min_avg)

    # precision = 1
    with open(file, "w") as f:
        for i in rows:
            print(i, file=f)    
        print("\\midrule", file=f)
        print(print_avg, file=f)


def min_discm(scientific_notation):
    file = "min-discm.tex"
    precision = 1
    with open(file, "r") as f:
        content = f.readlines()
    discm_techniques = {'T1':[], 'T2':[], 'T3':[] ,'T4':[] ,'T5':[] ,'T6':[] ,'T7':[] ,'T8':[]}
    rows = []
    for length, i in enumerate(content):
        j = i.split("&")
        experiment = j[0].strip()
        row = []
        j = j[1:]       # remove experiment ID
        min_value = 100
        for x, discm in enumerate(j):
            if "$" in discm:
                discm = discm.replace("$", "")
            if "textbf" in discm and "\\" in discm:
                discm = discm.split("\\")[1].split("{")[1].split("}")[0]
            elif "\\" in discm:
                discm = discm.split("\\")[0]
            discm_techniques['T'+str(x+1)].append(float(discm))
            discm_ = f"{float(discm):.{precision}e}"
            if scientific_notation:
                row.append(discm_)
            else:
                row.append(float(discm_))
            if float(discm_) < min_value:
                min_value = float(discm_)

        try:
            assert(len(row) == len(discm_techniques))
        except:
            if "midrule" in i and "Avg" in content[length+1]:
                return      # this file has been done
            else:
                assert False

        this_row = experiment + " & "
        this_row = process_row(row, this_row, min_value)
        rows.append(this_row)

    averages = []
    min_avg = 100
    for key in discm_techniques:
        val = f"{np.mean(discm_techniques[key]):.{precision}e}"
        if not scientific_notation:
            val = float(val)
        averages.append(val)
        if float(val) < min_avg:
            min_avg = float(val)
    avg_row = "Avg. & "
    print_avg = process_row(averages, avg_row, min_avg)

    if scientific_notation:
        file = file[:-4] + "_scientific" + ".tex"
    else:
        file = file[:-4] + "_floating" + ".tex"
    
    with open(file, "w") as f:
        for i in rows:
            print(i, file=f)    
        print("\\midrule", file=f)
        print(print_avg, file=f)
    
    # with open(file, "a") as f:
    #     print("Avg.", end=" & ", file=f)
    # for key in discm_techniques:
    #     if key != 'T8':
    #         END = " & "
    #     else:
    #         END = " \\\\"
    #     # print(key, discm_techniques[key], f"{np.mean(discm_techniques[key]):.{precision}e}")
    #     with open(file, "a") as f:
    #         print(f"{np.mean(discm_techniques[key]):.{precision}e}" , end=END, file=f)


def discm_for_max_accuracy(scientific_notation):
    file = "discm_for_max_accuracy.tex"
    precision = 1
    with open(file, "r") as f:
        content = f.readlines()
    discm_techniques = {'T1':[], 'T2':[], 'T3':[] ,'T4':[] ,'T5':[] ,'T6':[] ,'T7':[] ,'T8':[]}
    rows = []
    for length, i in enumerate(content):
        j = i.split("&")
        experiment = j[0].strip()
        row = []
        j = j[1:]       # remove experiment ID
        min_value = 100
        for x, discm in enumerate(j):
            if "$" in discm:
                discm = discm.replace("$", "")
            if "textbf" in discm and "\\" in discm:
                discm = discm.split("\\")[1].split("{")[1].split("}")[0]
            elif "\\" in discm:
                discm = discm.split("\\")[0]
            discm_techniques['T'+str(x+1)].append(float(discm))
            discm_ = f"{float(discm):.{precision}e}"
            if scientific_notation:
                row.append(discm_)
            else:
                row.append(float(discm_))
            if float(discm_) < min_value:
                min_value = float(discm_)

        try:
            assert(len(row) == len(discm_techniques))
        except:
            if "midrule" in i and "Avg" in content[length+1]:
                return      # this file has been done
            else:
                assert False

        this_row = experiment + " & "
        this_row = process_row(row, this_row, min_value)
        rows.append(this_row)

    averages = []
    min_avg = 100
    for key in discm_techniques:
        val = f"{np.mean(discm_techniques[key]):.{precision}e}"
        if not scientific_notation:
            val = float(val)
        averages.append(val)
        if float(val) < min_avg:
            min_avg = float(val)
    avg_row = "Avg. & "
    print_avg = process_row(averages, avg_row, min_avg)

    if scientific_notation:
        file = file[:-4] + "_scientific" + ".tex"
    else:
        file = file[:-4] + "_floating" + ".tex"
    
    with open(file, "w") as f:
        for i in rows:
            print(i, file=f)    
        print("\\midrule", file=f)
        print(print_avg, file=f)
    

def discm_for_min_parity(scientific_notation):
    file = "discm_for_min_parity.tex"
    precision = 1
    with open(file, "r") as f:
        content = f.readlines()
    discm_techniques = {'T1':[], 'T2':[], 'T3':[] ,'T4':[] ,'T5':[] ,'T6':[] ,'T7':[] ,'T8':[]}
    rows = []
    for length, i in enumerate(content):
        j = i.split("&")
        experiment = j[0].strip()
        row = []
        j = j[1:]       # remove experiment ID
        min_value = 100
        for x, discm in enumerate(j):
            if "$" in discm:
                discm = discm.replace("$", "")
            if "textbf" in discm and "\\" in discm:
                discm = discm.split("\\")[1].split("{")[1].split("}")[0]
            elif "\\" in discm:
                discm = discm.split("\\")[0]
            discm_techniques['T'+str(x+1)].append(float(discm))
            discm_ = f"{float(discm):.{precision}e}"
            if scientific_notation:
                row.append(discm_)
            else:
                row.append(float(discm_))
            if float(discm_) < min_value:
                min_value = float(discm_)

        try:
            assert(len(row) == len(discm_techniques))
        except:
            if "midrule" in i and "Avg" in content[length+1]:
                return      # this file has been done
            else:
                assert False

        this_row = experiment + " & "
        this_row = process_row(row, this_row, min_value)
        rows.append(this_row)

    averages = []
    min_avg = 100
    for key in discm_techniques:
        val = f"{np.mean(discm_techniques[key]):.{precision}e}"
        if not scientific_notation:
            val = float(val)
        averages.append(val)
        if float(val) < min_avg:
            min_avg = float(val)
    avg_row = "Avg. & "
    print_avg = process_row(averages, avg_row, min_avg)

    if scientific_notation:
        file = file[:-4] + "_scientific" + ".tex"
    else:
        file = file[:-4] + "_floating" + ".tex"
    
    with open(file, "w") as f:
        for i in rows:
            print(i, file=f)    
        print("\\midrule", file=f)
        print(print_avg, file=f)
    


if __name__ == "__main__":
    test_accuracy_for_min_discm()
    test_accuracy_for_min_parity()
    max_accuracy()
    for i in [True, False]:
        min_discm(i)
    for i in [True, False]:
        discm_for_max_accuracy(i)
    for i in [True, False]:
        discm_for_min_parity(i)
    parity_diff_min_discm()
    parity_diff_max_accuracy()
    parity_diff_min_parity()
