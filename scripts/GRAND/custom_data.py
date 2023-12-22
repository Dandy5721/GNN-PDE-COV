import os, random
import os.path as osp
import sys
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset, Batch
from torch_geometric.utils import index_to_mask, to_dense_batch 

random.seed(5432)

class custom_BOLD(InMemoryDataset):

    def __init__(self, root: str, name: str):
        root = root.replace('custom', '')
        self.name = name
        super().__init__(root)
        x, y, edges = read_BOLD_signal(root)
        self.N = len(x)
        self.index = [i for i in range(self.N)]
        random.shuffle(self.index)
        self.index = torch.LongTensor(self.index)
        self.node_nums = [len(x[i]) for i in range(len(x))]
        self.data = collate(x, y, edges)
        self.cross_val_fold_n = 5
        self.current_fold = 0
        # self.num_classes = 1
        self.next_fold()

    @property
    def num_classes(self) -> int:
        return 1

    def next_fold(self):
        split_pt1 = int(self.current_fold * self.N * (1/self.cross_val_fold_n))
        split_pt2 = int((self.current_fold+1) * self.N * (1/self.cross_val_fold_n))
        train_index = torch.cat([self.index[:split_pt1], self.index[split_pt2:]])
        val_index = self.index[split_pt1:split_pt2]
        train_mask = index_to_mask(train_index, size=self.N)
        train_mask = torch.cat([train_mask[i].repeat(self.node_nums[i]) for i in range(self.N)])
        val_mask = index_to_mask(val_index, size=self.N)
        val_mask = torch.cat([val_mask[i].repeat(self.node_nums[i]) for i in range(self.N)])
        self.data.train_mask = train_mask
        self.data.val_mask = val_mask
        self.current_fold += 1


def adj_to_edge_index(adj_mat):
    edge_index = np.where(adj_mat > 0)
    edge_attr = adj_mat[edge_index]
    return np.array(edge_index), edge_attr

def collate(x, y, edges):
    all_edge = edges[0][0]
    all_edge_attr = edges[0][1]
    for i in range(1, len(x)):
        edge = edges[i][0]
        edge_attr = edges[i][1]
        all_edge = np.concatenate([all_edge, edge+(x[i].shape[0])*i], axis=1)
        all_edge_attr = np.concatenate([all_edge_attr, edge_attr])
    x = np.concatenate(x)[..., np.newaxis]
    y = np.concatenate(y)[..., np.newaxis]
    data = Data(x=torch.from_numpy(x), edge_index=torch.from_numpy(all_edge), edge_attr=torch.from_numpy(all_edge_attr), y=torch.from_numpy(y))
    data.num_features = 1
    data.num_nodes = x.shape[0]
    return data

def read_BOLD_signal(folder, signal_folder='', label_folder='label_prediction', struct_folder='structure_prediction'):
    r = ''
    for i in folder.split('/')[:-1]: r += i+'/'
    x = [read_file(os.path.join(folder, signal_folder), name) for name in os.listdir(os.path.join(folder, signal_folder))]
    node_num = len(x[0])
    y = [read_file(os.path.join(r, label_folder), name).astype(np.int32) for name in os.listdir(os.path.join(r, label_folder))]
    edges = [adj_to_edge_index(read_file(os.path.join(r, struct_folder), name)[:node_num, :node_num]) for name in os.listdir(os.path.join(r, struct_folder))]
    return x, y, edges

def read_file(csv_r, csv_n):
    return np.loadtxt(osp.join(csv_r, csv_n), delimiter=',').astype(np.float32)


class custom_Classification(InMemoryDataset):

    def __init__(self, root: str, name: str):
        # /BAND/USERS/jiaqid/ADNI/data/Amyloid_classification
        root = root.replace('cus_cls-2', '').replace('cus_cls-_CN_AD', '')
        self.name = name.split('cus_cls-')[-1]
        self.biomarker = root.split('/')[-1].replace('_classification', '')
        super().__init__(root)
        x, y, edges = read_BOLD_signal(root, signal_folder=self.biomarker+self.name, label_folder=root.split('/')[-1]+'/'+'label'+self.name, struct_folder=root.split('/')[-1]+'/'+'structure'+self.name)
        self.N = len(x)
        self.index = [i for i in range(self.N)]
        random.shuffle(self.index)
        self.index = torch.LongTensor(self.index)
        self.node_nums = [len(x[i]) for i in range(len(x))]
        assert np.std(self.node_nums) == 0
        self.data = self.collate(x, y, edges)
        self.cross_val_fold_n = 5
        self.current_fold = 0
        # self.num_classes = 1
        self.next_fold()
        
    # @property
    # def num_classes(self) -> int:
    #     return 1

    def next_fold(self):
        split_pt1 = int(self.current_fold * self.N * (1/self.cross_val_fold_n))
        split_pt2 = int((self.current_fold+1) * self.N * (1/self.cross_val_fold_n))
        train_index = torch.cat([self.index[:split_pt1], self.index[split_pt2:]])
        val_index = self.index[split_pt1:split_pt2]
        train_mask = index_to_mask(train_index, size=self.N)
        val_mask = index_to_mask(val_index, size=self.N)
        self.data.train_mask = train_mask
        self.data.val_mask = val_mask
        self.current_fold += 1
        
    def collate(self, x, y, edges):
        all_edge = edges[0][0]
        all_edge_attr = edges[0][1]
        for i in range(1, len(x)):
            edge = edges[i][0]
            edge_attr = edges[i][1]
            all_edge = np.concatenate([all_edge, edge+(x[i].shape[0])*i], axis=1)
            all_edge_attr = np.concatenate([all_edge_attr, edge_attr])
        assert len(x) == len(y)
        x = np.concatenate(x)[..., np.newaxis]
        y = np.stack(y)
        data = Data(x=torch.from_numpy(x), edge_index=torch.from_numpy(all_edge), edge_attr=torch.from_numpy(all_edge_attr), y=torch.from_numpy(y).long())
        data.num_features = 1
        data.num_nodes = x.shape[0]
        return data