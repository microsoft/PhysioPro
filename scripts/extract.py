import os
import glob
import numpy as np
import scipy.io as sio


data_path = '/home/yansenwang/data/SEED-IV/SEED-IV/eeg_feature_smooth/' # path to the raw SEED dataset
save_path = '/home/yansenwang/data/New_SEED_IV/DE/'
os.makedirs(save_path, exist_ok=True)
labels = [
    [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
    [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
    [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
]
final_labels = {}

for exp in range(3):
    filenames = glob.glob(os.path.join(data_path, f'{exp+1}/*.mat'))
    filenames.sort()
    for sub in range(15):
        session_label = []
        data = []
        mat_path = filenames[sub]
        print(mat_path)
        T = sio.loadmat(mat_path)

        for trial in range(24):
            temp = T['de_LDS' + str(trial + 1)]
            data.append(temp)

            if sub == 0:
                temp_label = np.tile(labels[exp][trial], temp.shape[1])
                session_label.extend(temp_label)
        if sub == 0:
            final_labels[exp] = session_label
        data = np.concatenate(data, axis=1)
        sio.savemat(os.path.join(save_path, 'DE_' + str(sub * 3 + exp+1) + '.mat'), {'DE_feature': np.array(data)}) # save the features
for exp in range(3):
    sio.savemat(os.path.join(save_path, 'DE_' + str(exp+1) + '_labels.mat'), {'de_labels': np.array(final_labels[exp])})
