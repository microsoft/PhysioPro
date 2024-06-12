import os
import glob
import numpy as np
import scipy.io as sio


if __name__ == "__main__":
    # The following codes will convert the DE features from raw SEED dataset to the compatible format for physiopro to load
    data_path = './data/SEED/RAW/ExtractedFeatures' # path to the raw SEED dataset
    save_path = './data/SEED/DE/'
    filenames = glob.glob(os.path.join(data_path, '*.mat'))
    filenames.sort()
    filenames.remove(os.path.join(data_path, 'label.mat'))

    labels = []
    label_mat = sio.loadmat(os.path.join(data_path, 'label.mat'))
    label = label_mat['label'].flatten()

    os.makedirs(save_path, exist_ok=True)

    for sub in range(15):
        for exp in range(3):
            data = []
            mat_path = filenames[sub * 3 + exp]
            print(mat_path)
            T = sio.loadmat(mat_path)

            for trial in range(15):
                temp = T['de_LDS' + str(trial + 1)]
                data.append(temp)

                if sub == 0 and exp == 0:
                    temp_label = np.tile(label[trial] + 1, temp.shape[1]) # -1, 0, 1 -> 0, 1, 2
                    labels.extend(temp_label)
            data = np.concatenate(data, axis=1)
            sio.savemat(os.path.join(save_path, 'DE_' + str(sub * 3 + exp+1) + '.mat'), {'DE_feature': np.array(data)}) # save the features
    sio.savemat(os.path.join(save_path, 'DE_labels.mat'), {'de_labels': np.array(labels)}) # save the labels
