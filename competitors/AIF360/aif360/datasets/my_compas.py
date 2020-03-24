import os

import pandas as pd

import sys
sys.path.append("../../../benchmarks/compas/")
import load_compas_two_year
from aif360.datasets import StandardDataset


default_mappings = {
    'protected_attribute_maps': [{1.0:"Caucasian", 0.0:"African-American"}]
}

class MyCompasDataset(StandardDataset):
    """Default Prediction Dataset.
    """
    def __init__(self, label_name='two_year_recid',
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
            raise NotImplementedError
            # features_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'default-dataset', 'normalized_default_features.csv')
            # labels_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'default-dataset', 'default_labels.csv')
            df = pd.read_csv(features_path)
            df2 = pd.read_csv(labels_path)
            df['target'] = df2
        else:
            train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'compas-dataset', 'missing_compas_two_year_removed.csv')
            df = pd.read_csv(train_path)
            assert len(df.columns[df.isnull().any()]) == 0
            df['sex'] = df['sex'].replace({"Male":1, "Female":0})
            df['race'] = df['race'].replace({"Caucasian":1, "African-American":0})
            df['c_charge_degree'] = df['c_charge_degree'].replace({"F":1, "M":0})    # O : Ordinary crime, F: Felony, M: Misconduct
            if permute == -1:
                df_ordered = df
            else:
                assert(permute < 20)
                ordering = load_compas_two_year.permutations(permute)
                x = df.to_numpy()
                x = x[ordering]
                df_ordered = pd.DataFrame(x, columns=df.columns.tolist())
                new1 = df_ordered.sort_values(by=['end']).reset_index(drop=True)
                new2 = df.sort_values(by=['end']).reset_index(drop=True)
                z = new1 == new2
                assert(sum([z[i].unique()[0] for i in z.columns.tolist()]) == len(z.columns.tolist()))      # just a sanity check

                                  
        column_names = ['sex','age','race','juv_fel_count','decile_score','juv_misd_count','juv_other_count','priors_count','days_b_screening_arrest','c_days_from_compas','c_charge_degree','is_recid','is_violent_recid','decile_score.1','v_decile_score','priors_count.1','start','end','event','two_year_recid']

        super(MyCompasDataset, self).__init__(df=df_ordered, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop, na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)
