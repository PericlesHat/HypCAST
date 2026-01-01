# ---- coding: utf-8 ----
# @author: Ziyang Zhang et al.
# This work partly uses the code from CACHE.

import torch
import pickle
import os
import os.path as osp
import numpy as np
import pandas as pd

from collections import Counter
from torch_scatter import scatter_add, scatter
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_sparse import coalesce


def ExtractV2E(data):
    edge_index = data.edge_index
    _, sorted_idx = torch.sort(edge_index[0])
    edge_index = edge_index[:, sorted_idx].type(torch.LongTensor)

    num_nodes = data.n_x
    num_hyperedges = data.num_hyperedges
    if not ((data.n_x+data.num_hyperedges-1) == data.edge_index[0].max().item()):
        print('num_hyperedges does not match! 1')
        return
    cidx = torch.where(edge_index[0] == num_nodes)[0].min()
    data.edge_index = edge_index[:, :cidx].type(torch.LongTensor)
    return data


def Add_Self_Loops(data):
    edge_index = data.edge_index
    num_nodes = data.n_x
    num_hyperedges = data.num_hyperedges

    if not ((data.n_x + data.num_hyperedges - 1) == data.edge_index[1].max().item()):
        print('num_hyperedges does not match! 2')
        return

    hyperedge_appear_fre = Counter(edge_index[1].numpy())
    skip_node_lst = []
    for edge in hyperedge_appear_fre:
        if hyperedge_appear_fre[edge] == 1:
            skip_node = edge_index[0][torch.where(
                edge_index[1] == edge)[0].item()]
            skip_node_lst.append(skip_node.item())

    new_edge_idx = edge_index[1].max() + 1
    
    num_nodes_need_self_loop = 0
    for i in range(num_nodes):
        if i not in skip_node_lst:
            num_nodes_need_self_loop += 1
    
    new_edges = torch.zeros(
        (2, num_nodes_need_self_loop), dtype=edge_index.dtype)
    
    tmp_count = 0
    for i in range(num_nodes):
        if i not in skip_node_lst:
            new_edges[0][tmp_count] = i
            new_edges[1][tmp_count] = new_edge_idx
            new_edge_idx += 1
            tmp_count += 1

    data.totedges = num_hyperedges + num_nodes_need_self_loop
    edge_index = torch.cat((edge_index, new_edges), dim=1)
    _, sorted_idx = torch.sort(edge_index[0])
    data.edge_index = edge_index[:, sorted_idx].type(torch.LongTensor)
    return data


def norm_contruction(data, option='all_one', TYPE='V2E'):
    if TYPE == 'V2E':
        if option == 'all_one':
            data.norm = torch.ones_like(data.edge_index[0])

        elif option == 'deg_half_sym':
            edge_weight = torch.ones_like(data.edge_index[0])
            cidx = data.edge_index[1].min()
            Vdeg = scatter_add(edge_weight, data.edge_index[0], dim=0)
            HEdeg = scatter_add(edge_weight, data.edge_index[1]-cidx, dim=0)
            V_norm = Vdeg**(-1/2)
            E_norm = HEdeg**(-1/2)
            data.norm = V_norm[data.edge_index[0]] * \
                E_norm[data.edge_index[1]-cidx]

    elif TYPE == 'V2V':
        data.edge_index, data.norm = gcn_norm(
            data.edge_index, data.norm, add_self_loops=True)
    return data


def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=False, balance=False):
    """ Adapted from https://github.com/CUAI/Non-Homophily-Benchmarks"""
    """ randomly splits label into train/valid/test splits """
    if not balance:
        if ignore_negative:
            labeled_nodes = torch.where(label != -1)[0]
        else:
            labeled_nodes = label

        n = labeled_nodes.shape[0]
        train_num = int(n * train_prop)
        valid_num = int(n * valid_prop)

        perm = torch.as_tensor(np.random.permutation(n))

        train_indices = perm[:train_num]
        val_indices = perm[train_num:train_num + valid_num]
        test_indices = perm[train_num + valid_num:]

        if not ignore_negative:
            split_idx = {'train': train_indices,
                         'valid': val_indices,
                         'test': test_indices}
            return split_idx

        train_idx = labeled_nodes[train_indices]
        valid_idx = labeled_nodes[val_indices]
        test_idx = labeled_nodes[test_indices]

        split_idx = {'train': train_idx,
                     'valid': valid_idx,
                     'test': test_idx}
    else:
        indices = []
        for i in range(label.max()+1):
            index = torch.where((label == i))[0].view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        percls_trn = int(train_prop/(label.max()+1)*len(label))
        val_lb = int(valid_prop*len(label))
        train_idx = torch.cat([i[:percls_trn] for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]
        valid_idx = rest_index[:val_lb]
        test_idx = rest_index[val_lb:]
        split_idx = {'train': train_idx,
                     'valid': valid_idx,
                     'test': test_idx}
    return split_idx


def load_dataset(path='../data/raw_data/', dataset='demo',
                       node_feature_path="../data/raw_data/demo/node-embeddings-demo",
                       num_node=None, text=True):
    print(f'Loading hypergraph dataset from: {dataset}')

    df_labels = pd.read_csv(osp.join(path, dataset, f'edge-labels-{dataset}.txt'), sep=',', header=None)
    num_edges = df_labels.shape[0]
    labels = df_labels.values

    with open(node_feature_path, 'r') as f:
        line = f.readline()
        n_node_from_file, embedding_dim = line.split(" ")
        n_node_from_file = int(n_node_from_file)
        embedding_dim = int(embedding_dim)
        if num_node is None:
            num_node = n_node_from_file
        features = np.random.rand(num_node, embedding_dim)
        for lines in f.readlines():
            values = list(map(float, lines.split(" ")))
            features[int(values[0]) - 1] = np.array(values[1:])

    num_nodes = features.shape[0]
    print(f'number of nodes:{num_nodes}, feature dimension: {features.shape[1]}')

    features = torch.FloatTensor(features)
    labels = torch.FloatTensor(labels)

    p2hyperedge_list = osp.join(path, dataset, f'hyperedges-{dataset}.txt')
    node_list = []
    he_list = []
    he_id = num_nodes

    with open(p2hyperedge_list, 'r') as f:
        for line in f:
            if line[-1] == '\n':
                line = line[:-1]
            cur_set = line.split(',')
            cur_set = [int(x) for x in cur_set]
            node_list += cur_set
            he_list += [he_id] * len(cur_set)
            he_id += 1

    node_idx_min = np.min(node_list)
    node_list = [x - node_idx_min for x in node_list]

    edge_index = [node_list + he_list,
                  he_list + node_list]
    edge_index = torch.LongTensor(edge_index)

    data = Data(x=features,
                edge_index=edge_index,
                y=labels)
    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_attr = coalesce(data.edge_index,
                                               None,
                                               total_num_node_id_he_id,
                                               total_num_node_id_he_id)

    n_x = num_nodes
    data.n_x = n_x
    data.num_hyperedges = he_id - num_nodes

    hyperedges_dict = {}
    with open(p2hyperedge_list, "r") as file:
        lines = file.readlines()
        for idx, line in enumerate(lines):
            if line.strip() == "":
                continue
            nodes = list(map(int, line.strip().split(',')))
            hyperedges_dict[idx] = nodes

    data.hyperedges_dict = hyperedges_dict
    
    ids = None
    if dataset == 'aout2d':
        with open(osp.join(path, dataset, 'ehr_person_ids.txt'), 'r', encoding='utf-8') as file:
            ids = [line.strip() for line in file]
    elif dataset == 'gt':
        with open(osp.join(path, dataset, 'gt_person_ids.txt'), 'r', encoding='utf-8') as file:
            ids = [line.strip() for line in file]
    data.id = ids

    return data


def save_data_to_pickle(data, p2root = '../data/', file_name = None):
    surfix = 'star_expansion_dataset'
    if file_name is None:
        tmp_data_name = '_'.join(['Hypergraph', surfix])
    else:
        tmp_data_name = file_name
    p2he_StarExpan = osp.join(p2root, tmp_data_name)
    if not osp.isdir(p2root):
        os.makedirs(p2root)
    with open(p2he_StarExpan, 'bw') as f:
        pickle.dump(data, f)
    return p2he_StarExpan


class dataset_Hypergraph(InMemoryDataset):
    def __init__(self, root='../data/pyg_data/hypergraph_dataset/', name=None,
                 p2raw=None, transform=None, pre_transform=None, num_nodes=None, text=False):
        self.name = name
        
        if (p2raw is not None) and osp.isdir(p2raw):
            self.p2raw = p2raw
        elif p2raw is None:
            self.p2raw = None
        elif not osp.isdir(p2raw):
            raise ValueError(f'path to raw hypergraph dataset "{p2raw}" does not exist!')
        
        if not osp.isdir(root):
            os.makedirs(root)
            
        self.root = root
        self.myraw_dir = osp.join(root, self.name, 'raw')
        self.myprocessed_dir = osp.join(root, self.name, 'processed')
        self.num_nodes = num_nodes
        self.text = text
        super(dataset_Hypergraph, self).__init__(osp.join(root, name), transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        file_names = [self.name]
        return file_names

    @property
    def processed_file_names(self):
        file_names = ['data.pt']
        return file_names

    @property
    def num_features(self):
        return self.data.num_node_features

    def download(self):
        for name in self.raw_file_names:
            p2f = osp.join(self.myraw_dir, name)
            if not osp.isfile(p2f):
                if self.name in ['demo']:
                    tmp_data = load_dataset(path=self.p2raw,
                        dataset=self.name,
                        node_feature_path="../data/raw_data/demo/node-embeddings-demo", num_node=None, text=self.text)
                elif self.name in ['aout2d']:
                    tmp_data = load_dataset(path=self.p2raw,
                        dataset=self.name,
                        node_feature_path="../data/raw_data/aout2d/node-embeddings-aout2d", num_node=None, text=self.text)
                elif self.name in ['gt']:
                    tmp_data = load_dataset(path=self.p2raw,
                        dataset=self.name,
                        node_feature_path="../data/raw_data/gt/node-embeddings-gt", num_node=None, text=self.text)
                elif self.name in ['comp']:
                    tmp_data = load_dataset(path=self.p2raw,
                        dataset=self.name,
                        node_feature_path="../data/raw_data/comp/node-embeddings-comp", num_node=None, text=self.text)
                elif self.name in ['cgt']:
                    tmp_data = load_dataset(path=self.p2raw,
                        dataset=self.name,
                        node_feature_path="../data/raw_data/cgt/node-embeddings-cgt", num_node=None, text=self.text)
                    
                _ = save_data_to_pickle(tmp_data, 
                                          p2root = self.myraw_dir,
                                          file_name = self.raw_file_names[0])

    def process(self):
        p2f = osp.join(self.myraw_dir, self.raw_file_names[0])
        with open(p2f, 'rb') as f:
            data = pickle.load(f)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)
