import torch
import numpy as np
import scipy.io as sio
import scipy.sparse as ss
from sklearn.preprocessing import normalize
import scipy.sparse as sp
import random


def load_data(args, device):
    data = sio.loadmat(args.path + args.dataset + '.mat')
    features = normalize(data['X'])
    if ss.isspmatrix(features):
        features = features.todense()
    features = torch.from_numpy(features).float().to(device)

    adj_hat_list = []
    adj_norm_list = []
    norm_weights = []
    pose_weights = []

    labels = data['Y'].flatten()
    labels = labels - min(set(labels))
    class_idx, class_num = count_idx(labels)
    idx_labeled, idx_unlabeled, idx_val = divide(class_idx, class_num, args.ratio)
    labels = torch.from_numpy(labels).long()

    adjs = data['adj']
    for i in range(adjs.shape[1]):
        adj = ss.coo_matrix(adjs[0][i])

        adj_hat = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj_hat.setdiag(1)
        rowsum = np.array(adj_hat.sum(1))
        degree_mat_inv_sqrt = ss.diags(np.power(rowsum, -0.5).flatten())
        adj_norm = adj_hat.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)

        adj_hat = torch.from_numpy(adj_hat.todense()).float()

        norm_weight = float(adj_hat.shape[0] ** 2) / (adj_hat.shape[0] ** 2 - adj_hat.sum()) * 2
        pose_weight = float(adj_hat.shape[0] ** 2 - adj_hat.sum()) / adj_hat.sum()

        adj_norm = torch.from_numpy(adj_norm.todense()).float()

        adj_hat = construct_sparse_float_tensor(adj_hat).to(device)
        adj_norm = construct_sparse_float_tensor(adj_norm).to(device)
        adj_hat_list.append(adj_hat)
        adj_norm_list.append(adj_norm)
        norm_weights.append(norm_weight)
        pose_weights.append(pose_weight)
        del adj_hat, adj_norm, norm_weight, pose_weight
        torch.cuda.empty_cache()
    torch.cuda.empty_cache()

    return features, adj_hat_list, adj_norm_list, norm_weights, pose_weights, labels, idx_labeled, idx_unlabeled, idx_val


def construct_sparse_float_tensor(np_matrix):
    sp_matrix = ss.csc_matrix(np_matrix)
    three_tuple = sparse_to_tuple(sp_matrix)
    sparse_tensor = torch.sparse.FloatTensor(torch.LongTensor(three_tuple[0].T),
                                             torch.FloatTensor(three_tuple[1]),
                                             torch.Size(three_tuple[2]))
    return sparse_tensor


def sparse_to_tuple(sparse_mx):
    if not ss.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def count_idx(labels):
    dic_idx = {}
    dic_count = {}
    for index, c in enumerate(labels):
        if c not in dic_idx.keys():
            dic_idx[c] = []
            dic_count[c] = 0
        dic_idx[c].append(index)
        dic_count[c] += 1
    return dic_idx, dic_count


def divide(dic1, dic2, ratio):
    keys = list(dic1.keys())
    idx_train = []
    idx_test = []
    idx_val = []
    for i in keys:
        random.shuffle(dic1[i])
        num1 = int(dic2[i] * ratio)
        num2 = int(dic2[i] * 0.1)
        idx_train.extend(dic1[i][0:num1])
        idx_val.extend(dic1[i][num1:num1+num2])
        idx_test.extend(dic1[i][num1+num2:])
    return idx_train, idx_test, idx_val