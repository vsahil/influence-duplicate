import os, sys

import pandas as pd
sys.path.append("../../../benchmarks/german/")
import load_german_credit

from aif360.datasets import StandardDataset


default_mappings = {
    'protected_attribute_maps': [{1:"Male", 0:"Female"}]
}

class MyGermanDataset(StandardDataset):
    """Default Prediction Dataset.
    """
    def __init__(self, label_name='target',
                 favorable_classes=[1],
                 protected_attribute_names=['Gender'],
                 privileged_classes=[[1]],       
                 instance_weights_name=None,
                 categorical_features=[],
                 features_to_keep=[], features_to_drop=[],
                 na_values=[], custom_preprocessing=None,
                 metadata=default_mappings, normalized=False, permute=-1):
        """See :obj:`StandardDataset` for a description of the arguments.

        """

        if normalized:
            features_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'german-credit-dataset', 'german_redone_normalized_withheader.csv')
            labels_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'german-credit-dataset', 'german_labels_withheader.csv')
            df = pd.read_csv(features_path)
            df2 = pd.read_csv(labels_path)
            df['target'] = df2
        else:
            train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'german-credit-dataset', 'german_redone.csv')
            df = pd.read_csv(train_path)
            assert len(df.columns[df.isnull().any()]) == 0
        if permute == -1:
            df_ordered = df
        else:
            assert(permute < 20)
            ordering = load_german_credit.permutations(permute)
            x = df.to_numpy()
            x = x[ordering]
            df_ordered = pd.DataFrame(x, columns=df.columns.tolist())
            if not normalized:
                new1 = df_ordered.sort_values(by=['Credit-mount']).reset_index(drop=True)
                new2 = df.sort_values(by=['Credit-mount']).reset_index(drop=True)
                z = new1 == new2
                assert(sum([z[i].unique()[0] for i in z.columns.tolist()]) == len(z.columns.tolist()))      # just a sanity check
                # assert(df_ordered.sort_values(by=['credit_amount']).equals(df.sort_values(by=['credit_amount'])))       # credit-amount is probably unique
                # total_dataset, total_labels = total_dataset[ordering], total_labels[ordering]
                                    
        column_names = ['Checking-ccount','Months','Credit-history','Purpose','Credit-mount','Svings-ccount','Present-employment-since','Instllment-rte','Gender','Other-debtors','Present-residence-since','Property','ge','Other-instllment-plns','Housing','Number-of-existing-credits','Job','Number-of-people-being-lible','Telephone','Foreign-worker','target']

        super(MyGermanDataset, self).__init__(df=df_ordered, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop, na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)
