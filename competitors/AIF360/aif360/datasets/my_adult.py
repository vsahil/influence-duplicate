import os

import pandas as pd
import sys
sys.path.append("../../../benchmarks/adult-income/")
import load_adult_income

from aif360.datasets import StandardDataset


default_mappings = {
    'label_maps': [{1.0: '>50K', 0.0: '<=50K'}],
    'protected_attribute_maps': [{1.0: 'White', 0.0: 'Non-white'},
                                 {1.0: 'Male', 0.0: 'Female'}]
}

class MyAdultDataset(StandardDataset):
    """Adult Census Income Dataset.

    See :file:`aif360/data/raw/adult/README.md`.
    """
    # features_to_drop=['fnlwgt']
    # categorical_features=['workclass', 'education',
                    #  'marital-status', 'occupation', 'relationship',
                    #  'native-country']
    # privileged_classes=[['White'], ['Male']]      # Male is 1 
    # favorable_classes=['>50K', '>50K.'],
    def __init__(self, label_name='target',
                 favorable_classes=[1],
                 protected_attribute_names=['race', 'sex'],
                 privileged_classes=[['White'], [1]],       
                 instance_weights_name=None,
                 categorical_features=[],
                 features_to_keep=[], features_to_drop=[],
                 na_values=[], custom_preprocessing=None,
                 metadata=default_mappings, normalized=False, permute=-1):
        """See :obj:`StandardDataset` for a description of the arguments.

        Examples:
            The following will instantiate a dataset which uses the `fnlwgt`
            feature:

            >>> from aif360.datasets import AdultDataset
            >>> ad = AdultDataset(instance_weights_name='fnlwgt',
            ... features_to_drop=[])
            WARNING:root:Missing Data: 3620 rows removed from dataset.
            >>> not np.all(ad.instance_weights == 1.)
            True

            To instantiate a dataset which utilizes only numerical features and
            a single protected attribute, run:

            >>> single_protected = ['sex']
            >>> single_privileged = [['Male']]
            >>> ad = AdultDataset(protected_attribute_names=single_protected,
            ... privileged_classes=single_privileged,
            ... categorical_features=[],
            ... features_to_keep=['age', 'education-num'])
            >>> print(ad.feature_names)
            ['education-num', 'age', 'sex']
            >>> print(ad.label_names)
            ['income-per-year']

            Note: the `protected_attribute_names` and `label_name` are kept even
            if they are not explicitly given in `features_to_keep`.

            In some cases, it may be useful to keep track of a mapping from
            `float -> str` for protected attributes and/or labels. If our use
            case differs from the default, we can modify the mapping stored in
            `metadata`:

            >>> label_map = {1.0: '>50K', 0.0: '<=50K'}
            >>> protected_attribute_maps = [{1.0: 'Male', 0.0: 'Female'}]
            >>> ad = AdultDataset(protected_attribute_names=['sex'],
            ... privileged_classes=[['Male']], metadata={'label_map': label_map,
            ... 'protected_attribute_maps': protected_attribute_maps})

            Now this information will stay attached to the dataset and can be
            used for more descriptive visualizations.
        """

        if normalized:
            raise NotImplementedError
            train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'raw', 'adult', 'normalized_adult_features.csv')
        else:
           train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'raw', 'adult', 'adult_no_missing.csv') 

                                  
        # train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
        #                           '..', 'data', 'raw', 'adult', 'adult.data')
        # test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
        #                           '..', 'data', 'raw', 'adult', 'adult.test')
        # as given by adult.names
        # column_names = ['age', 'workclass', 'fnlwgt', 'education',
        #     'education-num', 'marital-status', 'occupation', 'relationship',
        #     'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
        #     'native-country', 'income-per-year']
        column_names = ['age','workclass','fnlwgt','education','marital-status','occupation',
        'race','sex','capitalgain','capitalloss','hoursperweek','native-country','target']

        try:
            # train = pd.read_csv(train_path, header=None, names=column_names,
                # skipinitialspace=True, na_values=na_values)
            df = pd.read_csv(train_path)
            # test = pd.read_csv(test_path, header=0, names=column_names,
            #     skipinitialspace=True, na_values=na_values)
            assert len(df.columns[df.isnull().any()]) == 0
            if permute == -1:
                df_ordered = df
            else:
                assert(permute < 20)
                ordering = load_adult_income.permutations(permute)
                x = df.to_numpy()
                x = x[ordering]
                df_ordered = pd.DataFrame(x, columns=df.columns.tolist())
                new1 = df_ordered.sort_values(by=['fnlwgt']).reset_index(drop=True)
                new2 = df.sort_values(by=['fnlwgt']).reset_index(drop=True)
                z = new1 == new2
                assert(sum([z[i].unique()[0] for i in z.columns.tolist()]) == len(z.columns.tolist()))      # just a sanity check

        except IOError as err:
            print("IOError: {}".format(err))
            print("To use this class, please download the following files:")
            print("\n\thttps://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data")
            print("\thttps://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test")
            print("\thttps://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names")
            print("\nand place them, as-is, in the folder:")
            print("\n\t{}\n".format(os.path.abspath(os.path.join(
               os.path.abspath(__file__), '..', '..', 'data', 'raw', 'adult'))))
            import sys
            sys.exit(1)

        # df = pd.concat([test, train], ignore_index=True)
        # df = train

        super(MyAdultDataset, self).__init__(df=df_ordered, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop, na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)
