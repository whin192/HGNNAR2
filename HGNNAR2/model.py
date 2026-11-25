import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pyro
from torch.nn.parameter import Parameter

def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)


class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, num_views, activation):
        super(GraphConvSparse, self).__init__()
        self.activation = activation
        self.num_views = num_views
        self.weight = Parameter(torch.FloatTensor(input_dim, output_dim))
        self.weight = glorot_init(input_dim, output_dim)

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0.0)

    def forward(self, inputs, flt, fea_sp=False):
        x = inputs
        self.ortho_weight = self.weight
        if fea_sp:
            x = torch.spmm(x, self.ortho_weight)
        else:
            x = torch.mm(x, self.ortho_weight)
        x = torch.spmm(flt, x)
        if self.activation is None:
            return x
        else:
            return self.activation(x)

class HGNNAR2(nn.Module):
    def __init__(self, input_dims, hidden_dims, emb_dims, num_views, num_classes, dropout, alpha, tau, device):
        super(HGNNAR2, self).__init__()
        self.device = device
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.emb_dims = emb_dims
        self.num_views = num_views
        self.alpha = alpha
        self.tau = tau
        self.ep_module = nn.ModuleList()
        self.nc_module = nn.ModuleList()
        self.w = nn.Parameter(torch.randn(num_views))
        for i in range(self.num_views):
            self.ep_module.append(MvGAE(input_dims, hidden_dims, emb_dims, self.num_views))
            self.nc_module.append(MvGNN(input_dims, hidden_dims, num_classes, self.num_views, dropout))

    def sample_adj_add_bernoulli(self, adj_recon, adj_hat):
        if torch.max(adj_recon)-torch.min(adj_recon) != 0:
            edge_probs = (adj_recon - torch.min(adj_recon))/(torch.max(adj_recon)-torch.min(adj_recon))
            edge_probs = self.alpha * edge_probs + (1 - self.alpha) * adj_hat
        else:
            edge_probs = self.alpha * adj_recon + (1 - self.alpha) * adj_hat

        adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=self.tau,
                                                                         probs=edge_probs).rsample()
        adj_sampled = adj_sampled.triu(1)
        adj_sampled = adj_sampled + adj_sampled.T
        adj_sampled.fill_diagonal_(1)

        D_norm = torch.diag(torch.pow(adj_sampled.sum(1), -0.5))
        adj_new = D_norm @ adj_sampled @ D_norm
        return adj_new, edge_probs

    def forward(self, features, adj_norm_list, adj_hat_list):
        weight = F.softmax(self.w)
        adjs_recon = []
        nodes_classify = []
        edge_probs_list = []
        for i in range(self.num_views):
            adj_recon = self.ep_module[i](features, adj_norm_list[i])
            adj_new, edge_probs = self.sample_adj_add_bernoulli(adj_recon, adj_hat_list[i].to_dense())
            node_classify = weight[i]*self.nc_module[i](features, adj_new)
            adjs_recon.append(adj_recon)
            nodes_classify.append(node_classify)
            edge_probs_list.append(edge_probs)
        return adjs_recon, nodes_classify, edge_probs_list


class MvGAE(nn.Module):
    def __init__(self, input_dims, hidden_dims, emb_dims, num_views):
        super(MvGAE, self).__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.emb_dims = emb_dims
        self.num_views = num_views
        self.ep1 = GraphConvSparse(self.input_dims, self.hidden_dims, self.num_views, activation=None)
        self.ep2 = GraphConvSparse(self.hidden_dims, self.emb_dims, self.num_views, activation=F.relu)

    def forward(self, features, adj_norm):
        hidden = self.ep1(features, adj_norm)
        z = self.ep2(hidden, adj_norm)
        adj_recon = z @ z.T
        return adj_recon


class MvGNN(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, num_views, dropout):
        super(MvGNN, self).__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.dropout = dropout
        self.num_views = num_views
        self.nc1 = GraphConvSparse(self.input_dims, self.hidden_dims, self.num_views, activation=F.relu)
        self.nc2 = GraphConvSparse(self.hidden_dims, self.output_dims, self.num_views, activation=None)

    def forward(self, features, adj_norm):
        h1 = self.nc1(features, adj_norm)
        h1 = F.dropout(h1, self.dropout, training=True)
        z = self.nc2(h1, adj_norm)
        return z