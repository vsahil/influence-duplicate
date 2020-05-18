import os

import pandas as pd
import sys
sys.path.append("../../../benchmarks/compas-score/")
import load_compas_score_as_labels
from aif360.datasets import StandardDataset


default_mappings = {
    'protected_attribute_maps': [{1.0:"Caucasian", 0.0:"African-American"}]
}

class MyCompasScoreDataset(StandardDataset):
    """Default Prediction Dataset.
    """
    def __init__(self, label_name='compas_score',
                 favorable_classes=[1],
                 protected_attribute_names=['race'],
                 privileged_classes=[[1]],       
                 instance_weights_name=None,
                 categorical_features=[],
                 features_to_keep=[], features_to_drop=[],
                 na_values=[], custom_preprocessing=None,
                 metadata=default_mappings, normalized=False, permute=-1):
        """See :obj:`StandardDataset` for a description of the arguments.

        """

        if normalized:
            features_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'compas-dataset', 'normalized_scores_as_labels_features.csv')
            labels_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'compas-dataset', 'target_compas_score_as_label.csv')
            df = pd.read_csv(features_path)
            df2 = pd.read_csv(labels_path)
            df['compas_score'] = df2
        else:
            train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'compas-dataset', 'compas_score_as_label.csv')
            df = pd.read_csv(train_path)
            assert len(df.columns[df.isnull().any()]) == 0
            df['sex'] = df['sex'].replace({"Male":1, "Female":0})
            df['race'] = df['race'].replace({"Caucasian":1, "African-American":0})
            df['c_charge_degree'] = df['c_charge_degree'].replace({"F":1, "M":0})    # O : Ordinary crime, F: Felony, M: Misconduct
        if permute == -1:
            df_ordered = df
        else:
            assert(permute < 20)
            ordering = load_compas_score_as_labels.permutations(permute)
            x = df.to_numpy()
            x = x[ordering]
            df_ordered = pd.DataFrame(x, columns=df.columns.tolist())
            # import ipdb; ipdb.set_trace()
            if not normalized:
                new1 = df_ordered.sort_values(by=['diff_custody', 'age', 'juv_misd_count', 'juv_other_count']).reset_index(drop=True)
                new2 = df.sort_values(by=['diff_custody', 'age', 'juv_misd_count', 'juv_other_count']).reset_index(drop=True)
                z = new1 == new2
                assert(sum([z[i].unique()[0] for i in z.columns.tolist()]) == len(z.columns.tolist()))      # just a sanity check

        column_names = ['age','sex','race','diff_custody','diff_jail','priors_count','juv_fel_count','juv_misd_count','juv_other_count','c_charge_degree','compas_score']

        super(MyCompasScoreDataset, self).__init__(df=df_ordered, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop, na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)
