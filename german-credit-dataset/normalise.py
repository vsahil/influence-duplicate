import pandas as pd
import numpy as np
features = pd.read_csv("german_gender_reversed.csv")
# import ipdb; ipdb.set_trace()
features = features.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
# features.to_csv("german_gender_reversed.csv", index=False)
