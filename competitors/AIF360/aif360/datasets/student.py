import os

import pandas as pd
import sys
sys.path.append("../../../benchmarks/student/")
import load_student
from aif360.datasets import StandardDataset


default_mappings = {
    'protected_attribute_maps': [{1:"Male", 0:"Female"}]
}

class StudentDataset(StandardDataset):
    """Default Prediction Dataset.
    """
    def __init__(self, label_name='G3',
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
            # features_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'default-dataset', 'normalized_default_features.csv')
            # labels_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'default-dataset', 'default_labels.csv')
            df = pd.read_csv(features_path)
            df2 = pd.read_csv(labels_path)
            df['target'] = df2
        else:
            train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'student-dataset', 'student-por.csv')
            df = pd.read_csv(train_path)
            assert len(df.columns[df.isnull().any()]) == 0
            df['sex'] = df['sex'].replace({"M":1, "F":0})
            for i in df.columns:
                if df[i].dtype == "O":
                    df[i], mapping_index = pd.Series(df[i]).factorize()
            df['G3'] = df['G3'].apply(lambda x: 0 if x <= 11 else 1)
            
            if permute == -1:
                df_ordered = df
            else:
                assert(permute < 20)
                ordering = load_student.permutations(permute)
                x = df.to_numpy()
                x = x[ordering]
                df_ordered = pd.DataFrame(x, columns=df.columns.tolist())
                new1 = df_ordered.sort_values(by=['G2']).reset_index(drop=True)
                new2 = df.sort_values(by=['G2']).reset_index(drop=True)
                z = new1 == new2
                assert(sum([z[i].unique()[0] for i in z.columns.tolist()]) == len(z.columns.tolist()))      # just a sanity check
                                          
        column_names = ['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian','traveltime','studytime','failures','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','absences','G1','G2','G3']

        super(StudentDataset, self).__init__(df=df_ordered, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop, na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)
