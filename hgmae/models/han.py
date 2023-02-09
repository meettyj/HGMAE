from typing import List
import torch
import torch.nn as nn

import dgl
from hgmae.models.gat import GATConv
from hgmae.utils import create_activation


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)
        out_emb = (beta * z).sum(1)  # (N, D * K)
        att_mp = beta.mean(0).squeeze()

        return out_emb, att_mp


class HANLayer(nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    meta_paths : list of metapaths, each as a list of edge types
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    g : DGLHeteroGraph
        The heterogeneous graph
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """

    def __init__(self, num_metapath, in_dim, out_dim, nhead,
                 feat_drop, attn_drop, negative_slope, residual, activation, norm, concat_out):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(num_metapath):
            self.gat_layers.append(GATConv(
                in_dim, out_dim, nhead,
                feat_drop, attn_drop, negative_slope, residual, activation, norm=norm, concat_out=concat_out))
        self.semantic_attention = SemanticAttention(in_size=out_dim * nhead)

    def forward(self, gs, h):
        semantic_embeddings = []

        for i, new_g in enumerate(gs):
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))  # flatten because of att heads
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)
        out, att_mp = self.semantic_attention(semantic_embeddings)  # (N, D * K)

        return out, att_mp


class HAN(nn.Module):
    def __init__(self,
                 num_metapath,
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 nhead,
                 nhead_out,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 norm,
                 concat_out=False,
                 encoding=False
                 ):
        super(HAN, self).__init__()
        self.out_dim = out_dim
        self.num_heads = nhead
        self.num_layers = num_layers
        self.han_layers = nn.ModuleList()
        self.activation = create_activation(activation)
        self.concat_out = concat_out

        last_activation = create_activation(activation) if encoding else create_activation(None)
        last_residual = (encoding and residual)
        last_norm = norm if encoding else None

        if num_layers == 1:
            self.han_layers.append(HANLayer(num_metapath,
                                            in_dim, out_dim, nhead_out,
                                            feat_drop, attn_drop, negative_slope, last_residual, last_activation,
                                            norm=last_norm, concat_out=concat_out))
        else:
            # input projection (no residual)
            self.han_layers.append(HANLayer(num_metapath,
                                            in_dim, num_hidden, nhead,
                                            feat_drop, attn_drop, negative_slope, residual, self.activation, norm=norm,
                                            concat_out=concat_out))
            # hidden layers
            for l in range(1, num_layers - 1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.han_layers.append(HANLayer(num_metapath,
                                                num_hidden * nhead, num_hidden, nhead,
                                                feat_drop, attn_drop, negative_slope, residual, self.activation,
                                                norm=norm, concat_out=concat_out))
            # output projection
            self.han_layers.append(HANLayer(num_metapath,
                                            num_hidden * nhead, out_dim, nhead_out,
                                            feat_drop, attn_drop, negative_slope, last_residual,
                                            activation=last_activation, norm=last_norm, concat_out=concat_out))

    def forward(self, gs: List[dgl.DGLGraph], h, return_hidden=False):
        for gnn in self.han_layers:
            h, att_mp = gnn(gs, h)
        return h, att_mp
