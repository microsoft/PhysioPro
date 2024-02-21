from typing import Optional

import torch
import numpy as np
import scipy.io as sio
from scipy.stats import rankdata
from torch.utils.data import Dataset
from .base import DATASETS


@DATASETS.register_module()
class SEED(Dataset):
    def __init__(
        self,
        prefix: str = "/dataset",
        name: str = "DE", # folder name where DE features are stored
        window_size: int = 1,
        tempo: bool = False,
        subject_index: Optional[int] = -1,
        dataset_name: Optional[str] = None,
        normalize: str = 'gaussian', # minmax or gaussian
        region:int = 17,
    ):
        super().__init__()
        self.window_size = window_size
        self.tempo = tempo
        self.dataset_name = dataset_name
        self.out_size = 3
        self.region = region
        # prepare DE data
        file = prefix + "/SEED/" + name + "/"
        data_file_path = file + "DE_{}.mat"
        d_labels_path = file + "DE_labels.mat"

        # if subject_index == -1, we use all the data, otherwise, we use the data according to the subject_index
        candidate_list = (
            [subject_index] if subject_index != -1 else list(range(45))
        )

        self.data = np.array(
            [
                sio.loadmat(data_file_path.format(i + 1))["DE_feature"]
                for i in candidate_list
            ]
        )
        # num_subject , num_channel, sample, num_freq -> num_subject, sample, num_channel, num_freq
        self.data = self.data.transpose([0, 2, 1, 3])
        self.label = np.array(
            [sio.loadmat(d_labels_path)["de_labels"] for _ in candidate_list]
        )

        self.label = self.label.flatten()

        self._normalize(normalize)

        self._split(self.dataset_name)

        if tempo:
            self._addtimewindow(window_size)
            # N, T, C, F -> N, C, T, F
            self.data=self.data.transpose([0,2,1,3])
        else:
            self.data = self.data.reshape(self.data.shape[0] * self.data.shape[1], self.data.shape[2] , self.data.shape[3])

        # reorder the channel by functional area
        # the order is recorded in the original SEED dataset. You can find it in "SEED/RAW/channel_order.csv"
        idx = [0,1,2,3,4,5,6,7,8,9,
                10,17,18,19,11,12,13,
                14,15,16,20,21,22,23,
                24,25,26,27,28,29,30,
                31,32,33,34,35,36,37,
                44,45,46,38,39,40,41,
                42,43,47,48,49,50,51,
                57,52,53,54,58,59,60,
                55,56,61]
        idx=torch.tensor(idx)
        self.data = torch.tensor(self.data)
        self.data = torch.index_select(self.data,dim=1,index=idx)
        self.data = self.data.numpy()
        self.get_coordination()

    def _normalize(self,method='minmax'):
        # min-max normalization
        if method == 'minmax':
            for i in range(self.data.shape[0]):
                for j in range(5):
                    # 0~2010 is the training set, 2010:: is the valid set
                    minn = np.min(self.data[i, :2010, :, j])
                    maxx = np.max(self.data[i, :2010, :, j])
                    self.data[i,:,:,j] = (self.data[i,:,:,j] - minn) / (maxx-minn)

        # gaussian standardization
        if method == 'gaussian':
            for i in range (self.data.shape[0]):
                for j in range(5):
                    # 0~2010 is the training set, 2010:: is the valid set
                    mean = np.mean(self.data[i, :2010, :, j])
                    std = np.std(self.data[i, :2010, :, j])
                    self.data[i, :, :, j] = (self.data[i, :, :, j] - mean) / std

    def _addtimewindow(self, window):
        S = self.data.shape[0]
        data_results = []
        label_results = []
        for i in range(S):
            # padding from the last sample, to make sure the sample number is the same after addtimewindow operation
            data = self.data[i]
            label = self.label[i]
            N, C, F = data.shape
            data = np.concatenate([data, data[-(window):, :, :]], 0)
            label = np.concatenate([label, label[-(window):]], 0)
            data_res = np.zeros(shape=(N, window, C, F))
            label_res = np.zeros(shape=(N,))
            for j in range(N):
                # met the corner case
                if (
                    label[j] == label[j + window - 1]
                    and label[j] != label[j + window]
                ):
                    nearest = j + window
                #
                elif label[j] == label[j + window - 1]:
                    nearest = -1
                if nearest != -1:
                    data_res[j, :, :, :] = np.concatenate(
                        [
                            data[j:nearest, :, :],
                            np.zeros(shape=(window - nearest + j, C, F)),
                        ],
                        0,
                    )
                else:
                    data_res[j, :, :, :] = data[j : j + window, :, :]
                label_res[j] = label[j]
            data_results.append(data_res)
            label_results.extend(label_res)

        self.data = np.concatenate(data_results, 0)
        self.label = np.array(label_results)

    def _split(self, dataset_name):
        total_size = 3394
        train_size = 2010
        test_size = total_size - train_size

        if dataset_name == "train":
            self.data = self.data[:,:train_size]
            self.label = self.label[:train_size]
            self.length = train_size
        elif dataset_name == "valid":
            self.data = self.data[:,train_size:]
            self.label = self.label[train_size:]
            self.length = test_size
        else:
            raise ValueError("dataset_name should be train or valid")

    def get_coordination(self):
        # we support the best-performed region setting. If you want to use other region setting, you can modify the code here
        if self.region == 17:
            func_areas = [[0,1,2,3,4],[5,6,7],[8,9,10,11,12,13],[14,15,16],[17,18,19],[20,21,22],[23,24,25],
            [26,27,28],[29,30,31],[32,33,34],[35,36,37,38,39,40],[41,42,43],[44,45,46],[47,48,49],
            [50,51,52],[53,54,55,56,57,58],[59,60,61]]
        else:
            raise ValueError("region should be 17")
        self.func_areas = func_areas
        coordination = np.array([[
            -27, 0, 27, -36, 36, -71, -64, -48, -25, 0, 25,
            -33, 0, 33, 48, 64, 71, -83, -78, -59, 59, 78, 83,
            -87, -82, -63, -34, 0, 34, 63, 82, 87, -83, -78,
            -59, -33, 0, 33, -25, 0, 25, 59, 78, 83, -71,
            -64, -48, 48, 64, 71, -51, -40, -36, -36, 0,
            36, -27, 0, 27, 40, 51, 36
        ],
        [
            83, 87, 83, 76, 76, 51, 55, 59, 62, 63, 62, 33, 34, 33, 59, 55, 51, 27,
            30, 31, 31, 30, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, -27, -30, -31, -33, -34,
            -33, -62, -63, -62, -31, -30, -27, -51, -55, -59, -59, -55, -51,
            -71, -76, -83, -76, -82, -76, -83, -87, -83, -76, -71, -83
        ],
        [
            -3, -3, -3, 24, 24, -3, 23, 44, 56, 61, 56, 74, 81, 74, 44, 23, -3,
            -3, 27, 56, 56, 27, -3, -3, 31, 61, 81, 88, 81, 61, 31, -3, -3, 27,
            56, 74, 81, 74, 56, 61, 56, 56, 27, -3, -3, 23, 44, 44, 23, -3, -3, 24,
            -3, 24, 31, 24, -3, -3, -3, 24, -3, -3
        ]]) # coordinations of the 62 channels (x, y, z)
        for i in range(coordination.shape[0]):
            arr = coordination[i]
            rank = rankdata(arr, method="dense") - 1
            coordination[i] = rank

        # attn mask is True for the position is not allowed to attend
        attn_mask = torch.full((62+len(func_areas)+1,62+len(func_areas)+1),True,dtype=torch.bool)

        # channels can attend to each other
        for i in range(62):
            for j in range(62):
                attn_mask[i,j] = False

        # supernodes can attend to each other
        for i in range(len(func_areas)):
            for j in range(len(func_areas)):
                attn_mask[62+i,62+j] = False

        # supernodes can attend to their channels
        for i, func_area in enumerate(func_areas):
            for j in func_area:
                attn_mask[62+i,j] = False
                attn_mask[j,62+i] = False

        # supernodes can attend to the final class token
        for i in range(62+len(func_areas)+1):
            attn_mask[62+len(func_areas), i] = False
            attn_mask[i, 62+len(func_areas)] = False
        self.attn_mask = attn_mask

        # assign coordination to supernodes
        self.coordination = area_gather(coordination, func_areas)

    def get_index(self):
        return self.label.index

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, idx):
        return (
            torch.tensor(
                self.data[idx],
                dtype=torch.float,
            ),
            torch.tensor(
                self.label[idx],
                dtype=torch.long,
            ).squeeze(),
        )

    def freeup(self):
        pass

    def load(self):
        pass

def area_gather(coordination, areas):
    supernode_coordination = np.zeros([coordination.shape[0], len(areas)])
    for idx,area in enumerate(areas):
        for i in area:
            for j in range(coordination.shape[0]):
                supernode_coordination[j][idx] += coordination[j][i]/len(area)

    res = np.concatenate((coordination,supernode_coordination), axis=1)
    return res

if __name__ == "__main__":
    dataset = SEED(dataset_name="train",window_size=1,subject_index=0)
    cnt = 0
    dl = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    for data,lebl in dl:
        print(data.shape)
