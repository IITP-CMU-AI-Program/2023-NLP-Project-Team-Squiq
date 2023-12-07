import random
import torch
import numpy as np
import networkx as nx
from loader import BioDataset, graph_data_obj_to_nx, nx_to_graph_data_obj


def save_data_to_pickle(data, p2root="../../data/", file_name=None):
    """
    if file name not specified, use time stamp.
    """
    #     now = datetime.now()
    #     surfix = now.strftime('%b_%d_%Y-%H:%M')
    surfix = "star_expansion_dataset"
    if file_name is None:
        tmp_data_name = "_".join(["Hypergraph", surfix])
    else:
        tmp_data_name = file_name
    p2he_StarExpan = osp.join(p2root, tmp_data_name)
    if not osp.isdir(p2root):
        os.makedirs(p2root)
    with open(p2he_StarExpan, "bw") as f:
        pickle.dump(data, f)
    return p2he_StarExpan
