import os
import glob
import numpy as np
import scipy.io as sio

data_path = '/dataset/SEED/RAW/ExtractedFeatures/'

filenames = glob.glob(os.path.join(data_path, '*.mat'))
filenames.sort()

labels = []
label_mat = sio.loadmat(os.path.join(data_path, 'label.mat'))
label = label_mat['label'].flatten()  

for sub in range(15):
    for exp in range(3):
        data = []
        mat_path = os.path.join(data_path, filenames[sub * 3 + exp] + '.mat')
        T = sio.loadmat(mat_path)

        for trial in range(15):
            temp = T['de_LDS' + str(trial + 1)]
            data.append(temp)

            if sub == 0 and exp == 0:
                temp_label = np.tile(label[trial], temp.shape[1])
                labels.extend(temp_label)
        data = np.concatenate(data, axis=1)
        sio.savemat('/dataset/SEED/DE/DE_' + str(sub * 3 + exp+1) + '.mat', {'DE_feature': np.array(data)})
sio.savemat('/dataset/SEED/DE/DE_labels.mat', {'de_labels': np.array(labels)})
