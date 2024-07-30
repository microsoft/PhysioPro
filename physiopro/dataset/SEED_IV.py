from typing import Optional

import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io as sio
from scipy.stats import rankdata

from .base import DATASETS


@DATASETS.register_module()
class SEED_IV(Dataset):
    def __init__(
        self,
        prefix: str = "./data/",
        name: str = "DE",
        window_size: int = 1,
        addtime: bool = False,
        subject_index: Optional[int] = -1,
        dataset_name: Optional[str] = 'train',
        channel: int = 62,
        local: bool = False,
        normalize: str = 'gaussian',
    ):
        super().__init__()
        self.window_size = window_size
        self.addtime = addtime
        self.dataset_name = dataset_name
        self.channel = channel
        self.local = local
        self.out_size = 4
        file = prefix + "/SEED-IV/" + name + "/"
        data_file_path = file + "DE_{}.mat"
        de_label={}
        for i in range(3):
            de_label[i] = np.array(sio.loadmat(file + f"DE_{i+1}_labels.mat")['de_labels']).squeeze(0)
        self.candidate_list = (
            [subject_index] if subject_index != -1 else list(range(45))
        )
        self.label = [de_label[i%3] for i in self.candidate_list]
        self.data =[
                np.array(sio.loadmat(data_file_path.format(i + 1))["DE_feature"]).transpose(1,0,2)
                for i in self.candidate_list
            ]
        self._normalize(normalize)
        self._split(self.dataset_name)

        if addtime:
            self._addtimewindow(window_size)
            # N,T,C,F -> N,C,T,F
            self.data = self.data.transpose(0,2,1,3)
        else:
            self.data =  np.concatenate(self.data, axis=0)
            self.label = np.concatenate(self.label, axis=0)
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
        train_size = [610, 558, 567]
        # min-max normalization
        if method == 'minmax':
            for i, candidate in enumerate(self.candidate_list):
                for j in range(5):
                    # 0~train_size[candidate%3] is the training set, train_size[candidate%3]:: is the valid set
                    minn = np.min(self.data[i][ :train_size[candidate%3], :, j])
                    maxx = np.max(self.data[i][ :train_size[candidate%3], :, j])
                    self.data[i][:,:,j] = (self.data[i][:,:,j] - minn) / (maxx-minn)

        # gaussian standardization
        if method == 'gaussian':
            for i, candidate in enumerate(self.candidate_list):
                for j in range(5):
                    # 0~train_size[candidate%3] is the training set, train_size[candidate%3]:: is the valid set
                    mean = np.mean(self.data[i][ :train_size[candidate%3], :, j])
                    std = np.std(self.data[i][ :train_size[candidate%3], :, j])
                    self.data[i][:, :, j] = (self.data[i][:, :, j] - mean) / std

    def _addtimewindow(self, window):
        S = len(self.data)
        data_results = []
        label_results = []
        for i in range(S):
            # padding from the last sample, to make sure the sample number is the same after addtimewindow operation
            data = self.data[i]
            label = self.label[i]
            print(data.shape)
            print(label.shape)
            N, C, F = data.shape
            data = np.concatenate([data, data[-(window):, :, :]], 0)
            label = np.concatenate([label, label[-(window):]], 0)
            data_res = np.zeros(shape=(N, window, C, F))
            label_res = np.zeros(shape=N)
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
        train_size = [610, 558, 567]

        if dataset_name == "train":
            for idx, candidate in enumerate(self.candidate_list):
                print(self.data[idx].shape)
                self.data[idx] = self.data[idx][:train_size[candidate%3]]
                self.label[idx] = self.label[idx][:train_size[candidate%3]]
        elif dataset_name == "valid":
            for idx, candidate in enumerate(self.candidate_list):
                self.data[idx] = self.data[idx][train_size[candidate%3]:]
                self.label[idx] = self.label[idx][train_size[candidate%3]:]
        else:
            raise ValueError("dataset_name should be train or valid")

    def get_coordination(self):
        func_areas = [[0,1,2,3,4],[5,6,7],[8,9,10,11,12,13],[14,15,16],[17,18,19],[20,21,22],[23,24,25],
            [26,27,28],[29,30,31],[32,33,34],[35,36,37,38,39,40],[41,42,43],[44,45,46],[47,48,49],
            [50,51,52],[53,54,55,56,57,58],[59,60,61]]
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
        ]])
        for i in range(coordination.shape[0]):
            arr = coordination[i]
            rank = rankdata(arr, method="dense") - 1
            coordination[i] = rank
        sph_coordination = np.array([[
            18,0,-18,25,-25,54,49,39,22,0,-22,45,0,-45,-39,-49,-54,72,69,62,-62,
            -69,-72,90,90,90,90,-90,-90,-90,-90,-90,108,111,118,135,-180,-135,158,
            -180,-158,-118,-111,-108,126,131,141,-141,-131,-126,144,155,162,155,-180,
            -155,162,-180,-162,-155,-144,-162
        ],
        [
            -2,-2,-2,16,16,-2,15,30,40,44,40,58,67,58,30,15,-2,-2,
            18,40,40,18,-2,-2,21,44,67,90,67,44,21,-2,-2,18,40,58,
            67,58,40,44,40,40,18,-2,-2,15,30,30,15,-2,-2,-2,-2,16,
            21,16,-2,-2,-2,-2,-2,-2,
        ]])

        # process attention mask
        attn_mask = torch.zeros((80,80),dtype=torch.int)
        if self.local:
            for func_area in func_areas:
                for i in func_area:
                    for j in func_area:
                        attn_mask[i,j] = 1
        else:
            for i in range(62):
                for j in range(62):
                    attn_mask[i,j] = 1
        for i, _ in enumerate(func_areas):
            for j in func_areas[i]:
                attn_mask[62+i,j] = 1
                attn_mask[j,62+i] = 1
        for i in range(17):
            for j in range(17):
                attn_mask[62+i,62+j] = 1
        for i in range(62+17+1):
            attn_mask[62+17, i] = 1
            attn_mask[i, 62+17] = 1
        self.attn_mask = (1-attn_mask).bool()
        #process supernode coordination
        self.coordination = area_gather(coordination, func_areas)
        self.sph_coordination = area_gather(sph_coordination, func_areas)

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

if __name__=='__main__':
    dataset = SEED_IV(subject_index=0,window_size=10,addtime=True)
    dl = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    cnt=0
    for data,label in dl:
        print(data.shape)
        cnt+=data.shape[0]
    print(cnt)
