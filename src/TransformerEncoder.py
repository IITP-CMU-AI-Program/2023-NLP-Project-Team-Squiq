import torch
import torch.nn as nn

import math

class EmbeddingBlock(nn.Module):
    """
    Arguments:
        num_embeddings : the number of word types
        embedding_dim : the dimension of embedding vector
    """

    def __init__(self, num_embeddings, embedding_dim):
        super(EmbeddingBlock, self).__init__()
        max_seq_len =512
        self.embedding_dim = embedding_dim
        self.pos_units = [10000**(2*i/self.embedding_dim) for i in range(self.embedding_dim//2)]

        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

        pos = torch.zeros((max_seq_len, self.embedding_dim))
        for p in range(max_seq_len):
            for i in range(0, self.embedding_dim, 2):
                pos[p, i] = torch.sin(torch.tensor(p) / self.pos_units[i // 2])
                pos[p, i + 1] = torch.cos(torch.tensor(p) / self.pos_units[i // 2])
        self.register_buffer('pos', pos)

    def forward(self, x):
        """
        input : indexed words (batch_size, num_words)
        output : word embeddings (batch_size, num_words, embedding_dim)
        """

        out = self.embedding(x)
        seq_len = x.size(1)
        pos = self.pos[:seq_len, :]
        out += pos.unsqueeze(0).expand_as(out)  # Batch size에 맞게 확장
        # pos = torch.zeros(out.shape)

        # for p in range(pos.shape[1]):
        #     for i in range(0, pos.shape[2], 2):
        #         pos[:, p, i] = torch.sin(torch.Tensor([p/self.pos_units[i//2]]))
        #         pos[:, p, i+1] = torch.cos(torch.Tensor([p/self.pos_units[i//2]]))
        # out += pos

        return out


class AttentionBlock(nn.Module):
    """
    Arguments:
        in_channel : the dimension of embedding vector
        out_channel : the dimension of query/key/value vector


    Variables:
        in_channel : d_model
        out_channel : d_k
    """

    def __init__(self, in_channel, out_channel):
        super(AttentionBlock, self).__init__()

        self.in_channel = in_channel

        self.fc_q = nn.Linear(in_channel, out_channel)  # W^Q
        self.fc_k = nn.Linear(in_channel, out_channel)  # W^K
        self.fc_v = nn.Linear(in_channel, out_channel)  # W^V

        self.softmax = nn.Softmax(dim=1)

    def forward(self, Q, K, V):
        """
        input : embedded words (batch_size, query_dim, key_dim, value_dim)
        output : attention score (batch_size, query_dim)
        """
        out_q = self.fc_q(Q)
        out_k = self.fc_k(K)
        out_v = self.fc_v(V)

        out = self.softmax(out_q @ out_k.transpose(1, 2) / math.sqrt(self.in_channel))

        out = out @ out_v

        return out


class MultiHeadAttentionBlock(nn.Module):
    """
    Arguments:
        in_channel : the dimension of embedding vector
        num_attention : the number of attention heads
        hidden_channel : the number of hidden channels in Position-wise Feed-Forward Networks

    Variables:
        in_channel : d_model
        inner_channel : d_ff
        num_attention : h
    """

    def __init__(self, in_channel, num_attention, hidden_channel):
        super(MultiHeadAttentionBlock, self).__init__()

        self.num_attention = num_attention

        self.heads = nn.ModuleList([AttentionBlock(in_channel, in_channel // self.num_attention) for _ in range(num_attention)])
        self.flatten = nn.Flatten()

        self.fc = nn.Linear(in_channel, in_channel)   # W^O

        self.ln1 = nn.LayerNorm((in_channel))


        self.ffc = nn.Sequential(nn.Linear(in_channel, hidden_channel),        # Position-wise Feed-Forward Networks
                                    nn.ReLU(),
                                    nn.Linear(hidden_channel, in_channel)
                                )

        self.ln2 = nn.LayerNorm((in_channel))


    def forward(self, x):
        """
        input : indexed words (batch_size, num_words)
        output : processed attention scores (batch_size, embedding_dim)
        """
        outs = [self.heads[i](x, x, x) for i in range(self.num_attention)]
        out = torch.cat(outs, dim=2)
        out = self.fc(out)

        out = self.ln1(out + x)

        out = self.ln2(out + self.ffc(out))

        return out


class TransformerEncoder(nn.Module):
    """
    Arguments:
        num_embeddings : the number of word types
        num_enc_layers : the number of encoder stack
        embedding_dim : the dimension of embedding vector
        num_attention : the number of attention heads
        hidden_channel : the number of hidden channels in Position-wise Feed-Forward Networks
        use_embedding : Transformer embedding enabled or not
    """

    def __init__(self, num_embeddings, num_enc_layers=6, embedding_dim=512, num_attention=8, hidden_channel=2048, use_embedding=True):
        super(TransformerEncoder, self).__init__()

        self.num_enc_layers = num_enc_layers
        self.embedding_dim = embedding_dim
        self.num_attention = num_attention
        self.hidden_channel = hidden_channel
        self.use_embedding = use_embedding

        if use_embedding:
            self.embedding = EmbeddingBlock(num_embeddings, embedding_dim)


        self.multihead_attention_blocks = nn.ModuleList([MultiHeadAttentionBlock(in_channel=self.embedding_dim,
                                                                       num_attention=self.num_attention,
                                                                       hidden_channel=self.hidden_channel)
                                                                            for _ in range(self.num_enc_layers)])

    def forward(self, x):
        """
        input : indexed words (batch_size, num_words)
        output : features (batch_size, embedding_dim)
        """

        out = x

        if self.use_embedding:
            out = self.embedding(x)

        for multihead_attention in self.multihead_attention_blocks:
            out = multihead_attention(out)

        return out