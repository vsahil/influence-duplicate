import os

import pandas as pd
import sys
sys.path.append("../../../benchmarks/default/")
import load_default
from aif360.datasets import StandardDataset


default_mappings = {
    'protected_attribute_maps': [{1.0: 'Male', 0.0: 'Female'}]
}

class DefaultDataset(StandardDataset):
    """Default Prediction Dataset.
    """
    def __init__(self, label_name='target',
                 favorable_classes=[1],
                 protected_attribute_names=['sex'],
                 privileged_classes=[[1]],       
                 instance_weights_name=None,
                 categorical_features=[],
                 features_to_keep=[], features_to_drop=[],
                 na_values=[], custom_preprocessing=None,
                 metadata=default_mappings, normalized=False, permute=-1):
        """See :obj:`StandardDataset` for a description of the arguments.

        """

        if normalized:
            raise NotImplementedError
            features_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'default-dataset', 'normalized_default_features.csv')
            labels_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'default-dataset', 'default_labels.csv')
            df = pd.read_csv(features_path)
            df2 = pd.read_csv(labels_path)
            df['target'] = df2
        else:
            train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'default-dataset', 'raw_default.csv')
            df = pd.read_csv(train_path)
            assert len(df.columns[df.isnull().any()]) == 0
            if permute == -1:
                df_ordered = df
            else:
                assert(permute < 20)
                ordering = load_default.permutations(permute)
                x = df.to_numpy()
                x = x[ordering]
                df_ordered = pd.DataFrame(x, columns=df.columns.tolist())
                new1 = df_ordered.sort_values(by=['BILL_AMT3']).reset_index(drop=True)
                new2 = df.sort_values(by=['BILL_AMT3']).reset_index(drop=True)
                z = new1 == new2
                assert(sum([z[i].unique()[0] for i in z.columns.tolist()]) == len(z.columns.tolist()))      # just a sanity check
                                  
        column_names = ['LIMIT_BAL','sex','EDUCATION','MARRIAGE','AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6','target']

        super(DefaultDataset, self).__init__(df=df_ordered, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop, na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)
