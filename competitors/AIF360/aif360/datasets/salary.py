import os

import pandas as pd
import sys
sys.path.append("../../../benchmarks/salary/")
import load_salary
from aif360.datasets import StandardDataset


default_mappings = {
    'label_maps': [{1.0: '>23K', 0.0: '<=23K'}],
    'protected_attribute_maps': [{1:"male", 0:"female"}]
}

class SalaryDataset(StandardDataset):
    """Faculty Salary Dataset.

    """
    # features_to_drop=['fnlwgt']
    # categorical_features=['workclass', 'education',
                    #  'marital-status', 'occupation', 'relationship',
                    #  'native-country']
    # privileged_classes=[['White'], ['Male']]      # Male is 1 
    # favorable_classes=['>50K', '>50K.'],
    def __init__(self, label_name='salary',
                 favorable_classes=[1],
                 protected_attribute_names=['sex'],
                 privileged_classes=[[1]],       
                 instance_weights_name=None,
                 categorical_features=[],
                 features_to_keep=[], features_to_drop=[],
                 na_values=[], custom_preprocessing=None,
                 metadata=default_mappings, normalized=False, permute=-1):
        """See :obj:`StandardDataset` for a description of the arguments.

        Examples:
            The following will instantiate a dataset which uses the `fnlwgt`
            feature:

            >>> from aif360.datasets import SalaryDataset
            >>> ad = SalaryDataset(instance_weights_name='fnlwgt',
            ... features_to_drop=[])
            WARNING:root:Missing Data: 3620 rows removed from dataset.
            >>> not np.all(ad.instance_weights == 1.)
            True

            To instantiate a dataset which utilizes only numerical features and
            a single protected attribute, run:

            Now this information will stay attached to the dataset and can be
            used for more descriptive visualizations.
        """
        if normalized:
            features_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'salary-dataset', 'normalized_salary_features.csv')
            labels_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'salary-dataset', 'salary_labels.csv')
            df = pd.read_csv(features_path)
            df2 = pd.read_csv(labels_path)
            df['salary'] = df2
        else:
            train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'salary-dataset', 'numeric_salary.csv')
            df = pd.read_csv(train_path)
            assert len(df.columns[df.isnull().any()]) == 0
                                  
        if permute == -1:
            df_ordered = df
        else:
            assert(permute < 20)
            ordering = load_salary.permutations(permute)
            x = df.to_numpy()
            x = x[ordering]
            df_ordered = pd.DataFrame(x, columns=df.columns.tolist())
            if not normalized:
                # import ipdb; ipdb.set_trace()
                new1 = df_ordered.sort_values(by=['Experience','year','degree','rank','sex']).reset_index(drop=True)
                new2 = df.sort_values(by=['Experience','year','degree','rank','sex']).reset_index(drop=True)
                z = new1 == new2
                assert(sum([z[i].unique()[0] for i in z.columns.tolist()]) == len(z.columns.tolist()))      # just a sanity check

        column_names = ['sex','rank','year','degree','Experience','salary']

        # except IOError as err:
        #     print("IOError: {}".format(err))
        #     print("To use this class, please download the following files:")
        #     print("\nand place them, as-is, in the folder:")
        #     print("\n\t{}\n".format(os.path.abspath(os.path.join(
        #     import sys
        #     sys.exit(1)

        # df = pd.concat([test, train], ignore_index=True)
        # df = train

        super(SalaryDataset, self).__init__(df=df_ordered, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop, na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)
