import argparse

from bertgnn.utils.data import *

# from loader import BioDataset
# from dataloader import DataLoaderMasking

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from bertgnn.utils.model import GNN  # , GNN_graphpred

import pandas as pd

# from util import MaskEdge
from tqdm import tqdm
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

# criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()


def compute_accuracy(pred, target):
    # return float(torch.sum((pred.detach() > 0) == target.to(torch.uint8)).cpu().item())/(pred.shape[0]*pred.shape[1])
    return float(
        torch.sum(torch.max(pred.detach(), dim=1)[1] == target).cpu().item()
    ) / len(pred)


def train(epoch, args, data_list, model_list, optimizer_list, device):
    model, linear_pred_edges = model_list
    optimizer_model, optimizer_linear_pred_edges = optimizer_list

    model.train()
    linear_pred_edges.train()

    loss_accum = 0
    acc_accum = 0

    for step, batch_data in enumerate(tqdm(data_list, desc="Iteration")):
        #         batch_data = batch_data.to(device)
        batch_x = batch_data.data.x.to(device)
        batch_edgeindex = batch_data.data.edge_index.to(device)
        batch_edgeattr = batch_data.data.edge_attr.to(device)
        batch_masked_edgeidx = batch_data.masked_edge_idx.to(device)

        node_rep = model(batch_x, batch_edgeindex, batch_edgeattr)

        ### predict the edge types.
        masked_edge_index = batch_edgeindex[:, batch_masked_edgeidx]
        edge_rep = (
            node_rep[batch_edgeindex[0][batch_masked_edgeidx]]
            + node_rep[batch_edgeindex[1][batch_masked_edgeidx]]
        )
        pred_edge = linear_pred_edges(edge_rep)

        # converting the binary classification to multiclass classification
        edge_label = torch.argmax(batch_data.mask_edge_label.to(device), dim=1)

        print(batch_masked_edgeidx)
        print(edge_label)
        print()

        acc_edge = compute_accuracy(pred_edge, edge_label)
        acc_accum += acc_edge

        optimizer_model.zero_grad()
        optimizer_linear_pred_edges.zero_grad()

        loss = criterion(pred_edge, edge_label)
        loss.backward()

        optimizer_model.step()
        optimizer_linear_pred_edges.step()

        loss_accum += float(loss.cpu().item())

    return loss_accum / (step + 1), acc_accum / (step + 1)


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description="PyTorch implementation of pre-training of graph neural networks"
    )
    parser.add_argument(
        "--device", type=int, default=0, help="which gpu to use if any (default: 0)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="input batch size for training (default: 256)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--decay", type=float, default=0, help="weight decay (default: 0)"
    )
    parser.add_argument(
        "--num_layer",
        type=int,
        default=5,
        help="number of GNN message passing layers (default: 5).",
    )
    parser.add_argument(
        "--emb_dim", type=int, default=384, help="embedding dimensions (default: 300)"
    )
    parser.add_argument(
        "--dropout_ratio", type=float, default=0, help="dropout ratio (default: 0)"
    )
    parser.add_argument(
        "--mask_rate", type=float, default=0.15, help="dropout ratio (default: 0.15)"
    )
    parser.add_argument(
        "--JK",
        type=str,
        default="last",
        help="how the node features are combined across layers. last, sum, max or concat",
    )
    parser.add_argument("--gnn_type", type=str, default="gin")
    parser.add_argument(
        "--model_file",
        type=str,
        default="../model/",
        help="filename to output the model",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for splitting dataset."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="number of workers for dataset loading",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    print("num layer: %d mask rate: %f" % (args.num_layer, args.mask_rate))

    if os.path.isfile("logs/train_loss_and_acc.txt"):
        os.remove("logs/train_loss_and_acc.txt")

    # set up dataset
    #     data = Dataset()
    data_list = []
    #     np.random.seed(epoch)
    masking = MaskEdge(0.15)
    #     for di in tqdm(range(2982)):
    for _ in tqdm(range(1000)):
        data = Dataset()
        new_data = masking(data)
        data_list.append(new_data)
    #         print(new_data.masked_edge_idx)

    # set up models, one for pre-training and one for context embeddings
    model = GNN(
        args.num_layer,
        args.emb_dim,
        JK=args.JK,
        drop_ratio=args.dropout_ratio,
        gnn_type=args.gnn_type,
    ).to(device)
    # Linear layer for classifying different edge types
    linear_pred_edges = torch.nn.Linear(args.emb_dim, 7).to(device)

    model_list = [model, linear_pred_edges]

    # set up optimizers
    optimizer_model = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.decay
    )
    optimizer_linear_pred_edges = optim.Adam(
        linear_pred_edges.parameters(), lr=args.lr, weight_decay=args.decay
    )

    optimizer_list = [optimizer_model, optimizer_linear_pred_edges]

    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))

        train_loss, train_acc = train(
            epoch, args, data_list, model_list, optimizer_list, device
        )
        print(train_loss, train_acc)
        with open("logs/train_loss_and_acc.txt", "+a") as f:
            f.write(str(train_loss) + "\t" + str(train_acc) + "\n")

    if not args.model_file == "":
        torch.save(
            model.state_dict(), args.model_file + "embedder_{}.pth".format(args.emb_dim)
        )

    # make embeddings
    data = Dataset()
    entire_x = data.data.x.to(device)
    entire_edgeindex = data.data.edge_index.to(device)
    entire_edgeattr = data.data.edge_attr.to(device)
    model.eval()
    with torch.no_grad():
        node_rep = model(entire_x, entire_edgeindex, entire_edgeattr)
        node_rep = node_rep.detach().cpu().numpy()
    np.save("emb/node_embedding_{}".format(args.emb_dim), node_rep)


if __name__ == "__main__":
    main()
