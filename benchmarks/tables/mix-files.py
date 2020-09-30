import sys, os 


def min_discm_mix(scientific_notation):
    print(scientific_notation)
    if scientific_notation:
        file1 = "min-discm_scientific.tex"
    else:
        file1 = "min-discm_floating.tex"
    with open(file1, "r") as f1:
        content_min_discm = f1.readlines()

    with open("test_accuracy_for_min_discm.tex", "r") as f2:
        content_max_accuracy = f2.readlines()

    with open("parity-diff-min-discm_fulltest.tex", "r") as f3:
        content_parity = f3.readlines()

    if scientific_notation:
        write_file = "min-discm-model_scientific.tex"
    else:
        write_file = "min-discm-model_floating.tex"

    if os.path.exists(f"{write_file}"):
        os.system(f"rm {write_file}")
    for id_, (l1, l2, l3) in enumerate(zip(content_min_discm, content_max_accuracy, content_parity)):
        l1 = l1.replace(" \\\\  \n", "")      # remove the breaklines 
        l1 = l1.replace(" \\\\  \\midrule\n", "")
        l1 = l1.replace("\\\\", "")      # remove the breaklines, for the last average line

        l2 = l2.replace(" \\\\  \n", "")      # new
        l2 = l2.replace(" \\\\  \\midrule\n", "")
        l2 = l2.replace("\\\\", "")

        if "midrule" in l1:
            assert "midrule" in l2 and "midrule" in l3
            mix = '\\midrule\n'

        else:
            # l1 = l1.split("&")
            experiment1 = l1.split("&")[0].strip()
            try:
                l1 = l1[l1.index("&"):]       # remove experiment ID
            except:
                import ipdb; ipdb.set_trace()

            # l2 = l2.split("&")
            experiment2 = l2.split("&")[0].strip()
            l2 = l2[l2.index("&"):]       # remove experiment ID

            experiment3 = l3.split("&")[0].strip()  # new
            l3 = l3[l3.index("&"):]

            try:
                assert(experiment1 == experiment2 == experiment3)
            except:
                import ipdb; ipdb.set_trace()
            mix = experiment1 + " " + l1 + l2 + l3    # this already has the breaklines

        # print(mix)
        with open(write_file, "a") as f3:
            print(mix, end="", file=f3)


def max_accuracy_mix(scientific_notation):
    print(scientific_notation)
    if scientific_notation:
        file1 = "discm_for_max_accuracy_scientific.tex"
    else:
        file1 = "discm_for_max_accuracy_floating.tex"
    with open(file1, "r") as f1:
        content_min_discm = f1.readlines()

    with open("max-test-accuracy.tex", "r") as f2:
        content_max_accuracy = f2.readlines()

    with open("parity-diff-max-accuracy_fulltest.tex", "r") as f3:
        content_parity = f3.readlines()

    if scientific_notation:
        write_file = "max-accuracy-model_scientific.tex"
    else:
        write_file = "max-accuracy-model_floating.tex"

    if os.path.exists(f"{write_file}"):
        os.system(f"rm {write_file}")
    for id_, (l1, l2, l3) in enumerate(zip(content_min_discm, content_max_accuracy, content_parity)):
        l1 = l1.replace(" \\\\  \n", "")      # remove the breaklines 
        l1 = l1.replace(" \\\\  \\midrule\n", "")
        l1 = l1.replace("\\\\", "")      # remove the breaklines, for the last average line

        l2 = l2.replace(" \\\\  \n", "")      # new
        l2 = l2.replace(" \\\\  \\midrule\n", "")
        l2 = l2.replace("\\\\", "")

        if "midrule" in l1:
            assert "midrule" in l2 and "midrule" in l3
            mix = '\\midrule\n'

        else:
            # l1 = l1.split("&")
            experiment1 = l1.split("&")[0].strip()
            try:
                l1 = l1[l1.index("&"):]       # remove experiment ID
            except:
                import ipdb; ipdb.set_trace()

            # l2 = l2.split("&")
            experiment2 = l2.split("&")[0].strip()
            l2 = l2[l2.index("&"):]       # remove experiment ID

            experiment3 = l3.split("&")[0].strip()  # new
            l3 = l3[l3.index("&"):]

            try:
                assert(experiment1 == experiment2 == experiment3)
            except:
                import ipdb; ipdb.set_trace()
            mix = experiment1 + " " + l1 + l2 + l3    # this already has the breaklines

        # print(mix)
        with open(write_file, "a") as f3:
            print(mix, end="", file=f3)


def min_parity_mix(scientific_notation):
    print(scientific_notation)
    if scientific_notation:
        file1 = "discm_for_min_parity_scientific.tex"
    else:
        file1 = "discm_for_min_parity_floating.tex"
    with open(file1, "r") as f1:
        content_min_discm = f1.readlines()

    with open("test_accuracy_for_min_parity.tex", "r") as f2:
        content_max_accuracy = f2.readlines()

    with open("min-parity-diff_fulltest.tex", "r") as f3:
        content_parity = f3.readlines()

    if scientific_notation:
        write_file = "min-parity-model_scientific.tex"
    else:
        write_file = "min-parity-model_floating.tex"

    if os.path.exists(f"{write_file}"):
        os.system(f"rm {write_file}")
    for id_, (l1, l2, l3) in enumerate(zip(content_min_discm, content_max_accuracy, content_parity)):
        l1 = l1.replace(" \\\\  \n", "")      # remove the breaklines 
        l1 = l1.replace(" \\\\  \\midrule\n", "")
        l1 = l1.replace("\\\\", "")      # remove the breaklines, for the last average line

        l2 = l2.replace(" \\\\  \n", "")      # new
        l2 = l2.replace(" \\\\  \\midrule\n", "")
        l2 = l2.replace("\\\\", "")

        if "midrule" in l1:
            assert "midrule" in l2 and "midrule" in l3
            mix = '\\midrule\n'

        else:
            # l1 = l1.split("&")
            experiment1 = l1.split("&")[0].strip()
            try:
                l1 = l1[l1.index("&"):]       # remove experiment ID
            except:
                import ipdb; ipdb.set_trace()

            # l2 = l2.split("&")
            experiment2 = l2.split("&")[0].strip()
            l2 = l2[l2.index("&"):]       # remove experiment ID

            experiment3 = l3.split("&")[0].strip()  # new
            l3 = l3[l3.index("&"):]

            try:
                assert(experiment1 == experiment2 == experiment3)
            except:
                import ipdb; ipdb.set_trace()
            mix = experiment1 + " " + l1 + l2 + l3    # this already has the breaklines

        # print(mix)
        with open(write_file, "a") as f3:
            print(mix, end="", file=f3)


if __name__ == "__main__":
    for i in [True, False]:
        min_discm_mix(i)
    for i in [True, False]:
        max_accuracy_mix(i)
    for i in [True, False]:
        min_parity_mix(i)