import math

import torch
import torch.nn as nn
import pytorch3d
from pytorch3d.utils import ico_sphere

import torch.nn.functional as F

# ref code from https://github.com/noahcao/Pixel2Mesh/tree/master

def batch_mm(matrix, batch):
    """
    https://github.com/pytorch/pytorch/issues/14489
    """
    # TODO: accelerate this with batch operations
    return torch.stack([matrix.mm(b) for b in batch], dim=0)


def dot(x, y, sparse=False):
    """Wrapper for torch.matmul (sparse vs dense)."""
    if sparse:
        return batch_mm(x, y)
    else:
        return torch.matmul(x, y)

class GConv(nn.Module):
    """Simple GCN layer

    Similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, adj_mat, bias=True):
        super(GConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.adj_mat = nn.Parameter(adj_mat, requires_grad=False)
        self.weight = nn.Parameter(torch.zeros((in_features, out_features), dtype=torch.float))
        # Following https://github.com/Tong-ZHAO/Pixel2Mesh-Pytorch/blob/a0ae88c4a42eef6f8f253417b97df978db842708/model/gcn_layers.py#L45
        # This seems to be different from the original implementation of P2M
        self.loop_weight = nn.Parameter(torch.zeros((in_features, out_features), dtype=torch.float))
        if bias:
            self.bias = nn.Parameter(torch.zeros((out_features,), dtype=torch.float))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data)
        nn.init.xavier_uniform_(self.loop_weight.data)

    def forward(self, inputs):
        support = torch.matmul(inputs, self.weight)
        support_loop = torch.matmul(inputs, self.loop_weight)
        output = dot(self.adj_mat, support, True) + support_loop
        if self.bias is not None:
            ret = output + self.bias
        else:
            ret = output
        return ret

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GResBlock(nn.Module):

    def __init__(self, in_dim, hidden_dim, adj_mat, activation=None):
        super(GResBlock, self).__init__()

        self.conv1 = GConv(in_features=in_dim, out_features=hidden_dim, adj_mat=adj_mat)
        self.conv2 = GConv(in_features=hidden_dim, out_features=in_dim, adj_mat=adj_mat)
        self.activation = F.relu if activation else None

    def forward(self, inputs):
        x = self.conv1(inputs)
        if self.activation:
            x = self.activation(x)
        x = self.conv2(x)
        if self.activation:
            x = self.activation(x)

        return (inputs + x) * 0.5

class GBottleneck(nn.Module): # from Pixel2Mesh: https://github.com/noahcao/Pixel2Mesh/blob/master/models/layers/gbottleneck.py

    def __init__(self, block_num, in_dim, hidden_dim, out_dim, adj_mat, activation=None):
        super(GBottleneck, self).__init__()

        resblock_layers = [GResBlock(in_dim=hidden_dim, hidden_dim=hidden_dim, adj_mat=adj_mat, activation=activation)
                           for _ in range(block_num)]
        self.blocks = nn.Sequential(*resblock_layers)
        self.conv1 = GConv(in_features=in_dim, out_features=hidden_dim, adj_mat=adj_mat)
        self.conv2 = GConv(in_features=hidden_dim, out_features=out_dim, adj_mat=adj_mat)
        self.activation = F.relu if activation else None

    def forward(self, inputs):
        x = self.conv1(inputs)
        if self.activation:
            x = self.activation(x)
        x_hidden = self.blocks(x)
        x_out = self.conv2(x_hidden)

        return x_out, x_hidden

import torch

def adjacency_from_faces(faces, num_verts):
    """
    faces: (F, 3) LongTensor
    num_verts: number of vertices
    returns: (num_verts, num_verts) adjacency matrix
    """
    # Collect edges (undirected)
    edges = torch.cat([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]]
    ], dim=0)

    # Make it symmetric
    edges_rev = edges[:, [1, 0]]
    edges = torch.cat([edges, edges_rev], dim=0)

    # Create adjacency matrix
    adj = torch.zeros((num_verts, num_verts), dtype=torch.float32)
    adj[edges[:, 0], edges[:, 1]] = 1.0

    return adj

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    batch_size = 32
    
    mesh_pred = ico_sphere(4, device)
    mesh_pred = pytorch3d.structures.Meshes(mesh_pred.verts_list()*batch_size, mesh_pred.faces_list()*batch_size)

    faces = mesh_pred.faces_list()[0]
    num_verts = mesh_pred.verts_list()[0].shape[0]

    adj = adjacency_from_faces(faces, num_verts)
    adj = adj + torch.eye(num_verts, device=adj.device)
    deg = adj.sum(1)
    deg_inv_sqrt = torch.diag(1.0 / torch.sqrt(deg))
    adj_norm = deg_inv_sqrt @ adj @ deg_inv_sqrt


    decoder = GBottleneck(6, 512, 128, 3, adj_norm, activation=True).to(device=device)
    
    verts = mesh_pred.verts_padded()  # (B, num_vertex, 3)
    features = torch.randn((batch_size, num_verts, 509), device=device)

    x = torch.cat([verts, features], dim=2)
    print(x.shape)
    print(features)

    # print(features)
    x_out, x_hidden = decoder(x)
    print(x_out)

