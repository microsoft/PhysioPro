
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import random
import string
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def generate_fake_df_data(type, data_path, feature_num, sequence_length, frequency, sample_num):
    """
    Generate fake df dataset
    
    Parameters:
    -----------
    type: str
        'classification', 'multiclassification', 'regression'
    feature_num: int
        number of features
    sample_num: int
        number of samples
    """
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    feature_path = os.path.join(data_path, 'feature_line')
    label_path = os.path.join(data_path, 'label_line')
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)
    if not os.path.exists(label_path):
        os.makedirs(label_path)

    identifiers = [''.join(random.choices(string.ascii_letters, k=10)) for _ in range(sample_num)]
    train, test = train_test_split(identifiers, test_size=0.3, random_state=42)

    train_df = []
    test_df = []

    for identifier in identifiers:
        sample_num_of_id = np.random.randint(1, 10)
        feature = np.random.rand(sample_num_of_id, sequence_length * frequency * feature_num)
        if type == 'regression':
            label = np.random.rand(sample_num_of_id)
        elif type == 'classification':
            label = np.random.randint(0, 2, sample_num_of_id)
        elif type == 'multiclassification':
            label = np.random.choice(['c1', 'c2', 'c3', 'c4'], sample_num_of_id)
        feature = pd.DataFrame(feature, index=[identifier] * sample_num_of_id)
        label = pd.DataFrame(label, index=[identifier] * sample_num_of_id, columns=['label'])
        feature_file_path = os.path.join(feature_path, f'{identifier}.pkl')
        label_file_path = os.path.join(label_path, f'{identifier}.pkl')
        pd.to_pickle(feature, feature_file_path)
        pd.to_pickle(label, label_file_path)

        if identifier in train:
            train_df.append(feature)
        elif identifier in test:
            test_df.append(feature)

    train_df = pd.concat(train_df, axis='index')
    test_df = pd.concat(test_df, axis='index')

    meta = {
        'train': train,
        'test': test,
        'mean@train': train_df.mean(),
        'std@train': train_df.std()
    }
    if type == 'multiclassification':
        meta['classes'] = ['c1', 'c2', 'c3', 'c4']

    meta_file_path = os.path.join(data_path, 'meta.pkl')
    pd.to_pickle(meta, meta_file_path)


prefix = './data/fake_df_data/'
generate_fake_df_data('classification', f'{prefix}classification', feature_num=3, sequence_length=5, frequency=7, sample_num=10)
generate_fake_df_data('multiclassification', f'{prefix}multiclassification', feature_num=3, sequence_length=5, frequency=7, sample_num=10)
generate_fake_df_data('regression', f'{prefix}regression', feature_num=3, sequence_length=5, frequency=7, sample_num=10)
