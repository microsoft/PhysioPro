# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import glob
import json
import argparse

import numpy as np
import pandas as pd

import scipy.io as sio

label_trails = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
label_trails = [i + 1 for i in label_trails]

subject_mapping = {}
subject_timeline_mapping = {}

def slide_window(data, label, window_size, stride):
    """
    Slide a window across the data with the given stride.
    """
    new_data = []
    new_label = []
    for i in range(0, len(data) - window_size + 1, stride):
        if label[i] != label[i + window_size - 1]:
            continue
        new_data.append(data[i:i + window_size, :, :])
        new_label.append(label[i])
    new_data = np.array(new_data)
    new_label = np.array(new_label)
    return new_data, new_label


def generate_data(dir, output_dir, duration, stride=1):
    """
    extract de features from RAW/Extractedfeatures in dataframe format
    """
    train_ids = {}
    test_ids = {}
    filenames = glob.glob(dir + '/*.mat')
    for file in filenames:
        if file.split('/')[-1].split('.')[0] == 'label':
            continue
        subject = file.split('/')[-1].split('_')[0]
        timeline = file.split('/')[-1].split('_')[1].split('.')[0]
        if subject not in subject_mapping:
            subject_mapping[subject] = str(len(subject_mapping)).zfill(2)
        if subject not in subject_timeline_mapping:
            subject_timeline_mapping[subject] = {}
        if timeline not in subject_timeline_mapping[subject]:
            subject_timeline_mapping[subject][timeline] = str(len(subject_timeline_mapping[subject])).zfill(2)
        subject, timeline = subject_mapping[subject], subject_timeline_mapping[subject][timeline]
        matdata = sio.loadmat(file)
        data = [np.array(matdata[f'de_LDS{i}']) for i in range(1, 16)]
        label = [np.array([label_trails[i - 1]] * data[i - 1].shape[1])
                for i in range(1, 16)]
        train_data, test_data = data[:9], data[9:]
        train_label, test_label = label[:9], label[9:]

        def process_data(data, label, duration, stride):
            data, label = np.concatenate(data, axis=1).transpose((1, 2, 0)), np.concatenate(label)
            data, label = slide_window(data, label, duration, stride)
            data, label = data.reshape((data.shape[0], -1)), np.array(label)
            return pd.DataFrame(data), pd.DataFrame(label)

        train_data, train_label = process_data(
            train_data, train_label, duration, stride)
        test_data, test_label = process_data(
            test_data, test_label, duration, stride)
        file_index = file.split('/')[-1].split('.')[0]
        dest_feature = f'{output_dir}/{subject}/{timeline}/feature_line'
        dest_label = f'{output_dir}/{subject}/{timeline}/label_line'
        if not os.path.exists(dest_feature):
            os.makedirs(dest_feature)
            os.makedirs(dest_label)
        train_data.to_pickle(f'{dest_feature}/{file_index}_train.pkl')
        train_label.to_pickle(f'{dest_label}/{file_index}_train.pkl')
        test_data.to_pickle(f'{dest_feature}/{file_index}_test.pkl')
        test_label.to_pickle(f'{dest_label}/{file_index}_test.pkl')
        if (subject, timeline) not in train_ids:
            train_ids[subject, timeline] = []
            test_ids[subject, timeline] = []
        train_ids[subject, timeline].append(f'{file_index}_train')
        test_ids[subject, timeline].append(f'{file_index}_test')
    return train_ids, test_ids


def get_meta(train_ids, test_ids, feature_dir, meta_dir):
    """
    Calculate and serialize meta information to pickle, in format:
        {
            'classes': list of all classes available,
            'train': list of train edf file names,
            'test': list of test edf file names,
            'mean@train': channel-wise mean of train,
            'std@train': channel-wise std of train,
            'mean@test': channel-wise mean of test,
            'std@test': channel-wise std of test,
        }
    """
    meta = {}
    meta['classes'] = [0, 1, 2]
    meta['train'] = train_ids
    meta['test'] = test_ids

    def concate_edfs(ids):
        data = [pd.read_pickle(feature_dir + f)
                for f in os.listdir(feature_dir)]
        return pd.concat(data, axis='index')

    print("Processing test meta...")
    test_df = concate_edfs(test_ids)
    meta['mean@test'] = test_df.mean(axis='index')
    meta['std@test'] = test_df.std(axis='index')
    del test_df

    print("Processing train meta...")
    train_df = concate_edfs(train_ids)
    meta['mean@train'] = train_df.mean(axis='index')
    meta['std@train'] = train_df.std(axis='index')

    print(meta)
    pd.to_pickle(meta, f"{meta_dir}/meta.pkl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=int, default=10,
                        help='window size, in seconds')
    parser.add_argument(
        '--raw-data-dir',
        type=str,
        default="data/SEED/RAW/ExtractedFeatures/",
        help='path to raw data')
    parser.add_argument('--output-dir', type=str, default="data/SEED/DE",
                        help='root path for output data.')
    args = parser.parse_args()

    print("Generating data...")
    train_ids, test_ids = generate_data(
        args.raw_data_dir, args.output_dir, args.duration)

    print("Calculating and serializing meta...")
    for k, train_id in train_ids.items():
        test_id = test_ids[k]
        get_meta(
            train_id,
            test_id,
            args.output_dir +
            f'{k[0]}/{k[1]}/feature_line/',
            args.output_dir +
            f'{k[0]}/{k[1]}')
    mapping = {
        'subject_mapping': subject_mapping,
        'subject_timeline_mapping': subject_timeline_mapping
    }
    with open(f'{args.output_dir}/mapping.meta', 'w') as f:
        json.dump(mapping, f)

if __name__ == '__main__':
    main()
