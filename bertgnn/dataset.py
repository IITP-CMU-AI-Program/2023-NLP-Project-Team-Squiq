import torch.utils.data
from torch_geometric.data import Data, Batch
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import InMemoryDataset
import torch
import random
import numpy as np
import os.path as osp
import pickle
from collections import defaultdict
import os

class MaskEdge:
    def __init__(self, mask_rate):
        """
        Assume edge_attr is of the form:
        [w1, w2, w3, w4, w5, w6, w7, self_loop, mask]
        :param mask_rate: % of edges to be masked
        """
        self.mask_rate = mask_rate

    def __call__(self, data, masked_edge_indices=None):
        """

        :param data: pytorch geometric data object. Assume that the edge
        ordering is the default pytorch geometric ordering, where the two
        directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                     [1, 0, 2, 1, 3, 2]])
        :param masked_edge_indices: If None, then randomly sample num_edges * mask_rate + 1
        number of edge indices. Otherwise should correspond to the 1st
        direction of an edge pair. ie all indices should be an even number
        :return: None, creates new attributes in the original data object:
        data.mask_edge_idx: indices of masked edges
        data.mask_edge_labels: corresponding ground truth edge feature for
        each masked edge
        data.edge_attr: modified in place: the edge features (
        both directions) that correspond to the masked edges have the masked
        edge feature
        """
        
        if masked_edge_indices == None:
            # sample x distinct edges to be masked, based on mask rate. But
            # will sample at least 1 edge
            num_edges = data.edge_index.size()[1]
            #int(data.edge_index.size()[1] / 2)  # num unique edges
            sample_size = int(num_edges * self.mask_rate + 1)
            # during sampling, we only pick the 1st direction of a particular
            # edge pair
            masked_edge_indices = [i for i in random.sample(range(num_edges), sample_size)]
            #[2 * i for i in random.sample(range(num_edges), sample_size)]
            
        data.masked_edge_idx = torch.tensor(np.array(masked_edge_indices))
        
        # create ground truth edge features for the edges that correspond to
        # the masked indices
        mask_edge_labels_list = []
        for idx in masked_edge_indices:
            mask_edge_labels_list.append(data.edge_attr[idx].view(1, -1))
        data.mask_edge_label = torch.cat(mask_edge_labels_list, dim=0)

        # created new masked edge_attr, where both directions of the masked
        # edges have masked edge type. For message passing in gcn

        # append the 2nd direction of the masked edges
        all_masked_edge_indices = masked_edge_indices # + [i + 1 for i in masked_edge_indices]
                                   
        for idx in all_masked_edge_indices:
            data.edge_attr[idx] = torch.tensor(np.array([0 for _ in range(7)]),
                                                      dtype=torch.float)
        
        return data
    
class Dataset(InMemoryDataset):
    def __init__(self, root = './', name = None, 
                 p2raw = None,
                 train_percent = 0.01,
                 feature_noise = None,
                 transform=None, pre_transform=None):
        
        self.name = "our"
        self.feature_noise = feature_noise
        self.train_percent = train_percent
        self._train_percent = train_percent

        self.root = root
        self.myraw_dir = osp.join(root, 'rawKG')
        self.myprocessed_dir = osp.join(root, 'processedKG')
        
        if os.path.isfile(self.myraw_dir + "/dataKG"):
#             print("Remove!")
            os.remove(self.myraw_dir + "/dataKG")
        if os.path.isfile(self.myprocessed_dir + "/data.pt"):
#             print("Remoe!")
            os.remove(self.myprocessed_dir + "/data.pt")
            
        if os.path.isdir(self.myraw_dir) is False:
            os.makedirs(self.myraw_dir)
        if os.path.isdir(self.myprocessed_dir) is False:
            os.makedirs(self.myprocessed_dir)
        
        super(Dataset, self).__init__(osp.join(root), transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])
#         self.train_percent = self.data.train_percent.item()

    @property
    def raw_file_names(self):
#         if self.feature_noise is not None:
#             file_names = [f'{self.name}_noise_{self.feature_noise}']
#         else:
        file_names = ['data'] #[self.name]
        return file_names

    @property
    def processed_file_names(self):
#         if self.feature_noise is not None:
#             file_names = [f'data_noise_{self.feature_noise}.pt']
#         else:
        file_names = ['data.pt']
        return file_names

    @property
    def num_features(self):
        return self.data.num_node_features


    def download(self):
        nodeset = set()
        edge_index = [[], []]
        with open("./rawKG/edges.txt", "r") as f:
            for line in f.readlines():
                vi, vj = line.rstrip().split("\t")
                vi, vj = int(vi), int(vj)
                nodeset.add(vi)
                nodeset.add(vj)
                edge_index[0].append(vi)
                edge_index[1].append(vj)
        edge_index = torch.LongTensor(edge_index)
        edgefeat = torch.from_numpy(np.load("./rawKG/edge_feat.npy"))
        edgelabel = torch.FloatTensor(np.load("./rawKG/edge_label.npy"))
        dist = defaultdict(int)
        for li in range(edgelabel.shape[0]):
            labellist = edgelabel[li]
            for label in range(edgelabel.shape[-1]):
                if labellist[label] == 1:
                    dist[label] += 1
#         print(dist)
        x = torch.FloatTensor(np.load("./rawKG/node_feat.npy")).squeeze(1)

        data = Data(x = x,
                    edge_index = edge_index,
                    edge_attr = edgelabel,
                    transform = MaskEdge(mask_rate = 0.4))
        
#         print(data.edge_attr.shape)
#         total_num_node_id_he_id = len(np.unique(edge_index))
#         data.edge_index, data.edge_attr = coalesce(data.edge_index, 
#                                                     None, 
#                                                     total_num_node_id_he_id, 
#                                                     total_num_node_id_he_id)

        data.train_percent = self._train_percent

        with open(self.myraw_dir + "/data" , 'bw') as f:
            pickle.dump(data, f)

    def process(self):
        p2f = osp.join(self.myraw_dir, self.raw_file_names[0])
        with open(p2f, 'rb') as f:
            data = pickle.load(f)
        data = data if self.pre_transform is None else self.pre_transform(data)
#         print(data.edge_attr.shape)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)

