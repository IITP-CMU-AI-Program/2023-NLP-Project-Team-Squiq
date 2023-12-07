import torch
import torch.nn as nn

import math

from src.TransformerEncoder import TransformerEncoder

class SegmentEmbedding(nn.Embedding):   # referenced from https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/model/embedding/segment.py

    def __init__(self, embedding_dim):
        super(SegmentEmbedding, self).__init__(3, embedding_dim)


class BERTEmbeddingBlock(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, max_seq_len=512):
        super(BERTEmbeddingBlock, self).__init__()

        self.embedding_dim = embedding_dim
        self.pos_units = [10000 ** (2 * i / self.embedding_dim) for i in range(self.embedding_dim // 2)]

        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.segment_embedding = SegmentEmbedding(self.embedding_dim)

        # Positional Embedding 미리 계산
        pos = torch.zeros((max_seq_len, self.embedding_dim))
        for p in range(max_seq_len):
            for i in range(0, self.embedding_dim, 2):
                pos[p, i] = torch.sin(torch.tensor(p) / self.pos_units[i // 2])
                pos[p, i + 1] = torch.cos(torch.tensor(p) / self.pos_units[i // 2])
        self.register_buffer('pos', pos)

    def forward(self, x, segment_info):
        out = self.embedding(x)

        seq_len = x.size(1)
        pos = self.pos[:seq_len, :]

        out += pos.unsqueeze(0).expand_as(out)
        out += self.segment_embedding(segment_info)

        return out

class BERT(nn.Module):

    """
    Arguments:
        num_embeddings : the number of word types
        num_transformer_block : the dimension of embedding vector
        num_enc_layers : the number of encoder stack
        embedding_dim : the dimension of embedding vector
        num_attention : the number of attention heads
        hidden_channel : the number of hidden channels in Position-wise Feed-Forward Networks

    Variables:
        out_channel : d_model
    """
    def __init__(self, num_embeddings=30000, num_transformer_block=6, num_enc_layers=1, embedding_dim=384, num_attention=12, hidden_channel=384):
        super(BERT, self).__init__()

        self.num_embeddings = num_embeddings
        self.num_transformer_block = num_transformer_block
        self.num_enc_layers = num_enc_layers
        self.embedding_dim = embedding_dim
        self.num_attention = num_attention
        self.hidden_channel = hidden_channel

        self.embedding = BERTEmbeddingBlock(self.num_embeddings, self.embedding_dim)
        self.lin  = nn.Linear(self.embedding_dim,self.num_embeddings)
        self.transformer_blocks = nn.ModuleList([TransformerEncoder(num_embeddings=self.embedding_dim,
                                                                  num_enc_layers=self.num_enc_layers,
                                                                  embedding_dim=self.embedding_dim,
                                                                  num_attention=self.num_attention,
                                                                  hidden_channel=self.hidden_channel,
                                                                  use_embedding=False,)
                                                                        for _ in range(self.num_transformer_block)])

    def forward(self, x, segment_info):         # referenced from https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/model/embedding/segment.py
        x = self.embedding(x, segment_info)

        for transformer in self.transformer_blocks:
            x = transformer.forward(x)
        x = self.lin(x)
        return x