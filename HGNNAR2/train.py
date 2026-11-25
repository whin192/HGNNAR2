import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import get_evaluation_results
from Dataloader import load_data
from model import HGNNAR2
import copy


def train(args, device):
    features, adj_hat_list, adj_norm_list, norm_weights, pose_weights, \
        labels, idx_labeled, idx_unlabeled, idx_val = load_data(args, device=device)
    print('Labeled sample:', len(idx_labeled))
    num_classes = len(np.unique(labels))
    labels = labels.to(device)
    num_view = len(adj_hat_list)
    input_dims = features.shape[1]

    model = HGNNAR2(input_dims, args.hidden_dims, args.emb_dims, num_view, num_classes,
                    args.dropout, args.alpha, args.tau, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if num_classes == 2:
        loss_function = torch.nn.BCEWithLogitsLoss()
    else:
        loss_function = torch.nn.CrossEntropyLoss()

    optimizer_ep = torch.optim.Adam(model.ep_module.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for i in range(args.pretrain_ep):
        model.train()
        loss_ep = 0.0
        for j in range(num_view):
            adj_logits = model.ep_module[j](features, adj_norm_list[j])
            loss_ep += norm_weights[j] * F.binary_cross_entropy_with_logits(adj_logits.to_dense(),
                                                                            adj_hat_list[j].to_dense(),
                                                                            pos_weight=pose_weights[j])
        optimizer_ep.zero_grad()
        loss_ep.backward()
        optimizer_ep.step()

    begin_time = time.time()
    best_acc = 0
    best_F1 = 0
    train_acc_list = []
    train_f1_list = []
    loss_list = []
    acc_list = []
    f1_list = []
    val_acc_list = []
    val_f1_list = []
    with tqdm(total=args.num_epoch, desc="Training") as pbar:
        for epoch in range(args.num_epoch):
            model.train()
            ep, nc, _ = model(features, adj_norm_list, adj_hat_list)
            optimizer.zero_grad()
            loss = caculate_loss(ep, nc, norm_weights, pose_weights, adj_hat_list,
                                 loss_function, num_view, labels, idx_labeled, args.beta)
            loss_list.append(loss.cpu().detach().numpy())
            pred_labels = torch.argmax(sum(nc), 1).cpu().detach().numpy()
            ACC_train, F1_train = get_evaluation_results(labels.cpu().detach().numpy()[idx_labeled],
                                                         pred_labels[idx_labeled])
            train_acc_list.append(ACC_train * 100)
            train_f1_list.append(F1_train * 100)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                _, output, edge_probs_list = model(features, adj_norm_list, adj_hat_list)

                output = sum(output)
                pred_labels = torch.argmax(output, 1).cpu().detach().numpy()
                ACC, F1 = get_evaluation_results(labels.cpu().detach().numpy()[idx_unlabeled],
                                                 pred_labels[idx_unlabeled])
                val_acc, val_f1 = get_evaluation_results(labels.cpu().detach().numpy()[idx_val],
                                                         pred_labels[idx_val])

                if F1 >= best_F1:
                    best_acc = ACC
                    best_F1 = F1
                    best_model = copy.deepcopy(model)

                acc_list.append(ACC * 100)
                f1_list.append(F1 * 100)
                val_acc_list.append(val_acc * 100)
                val_f1_list.append(val_f1 * 100)
                pbar.set_postfix({'Loss': '{:.6f}'.format(loss.item()),
                                  'ACC_train': '{:.2f}'.format(ACC_train * 100),
                                  'F1_train': '{:.2f}'.format(ACC_train * 100),
                                  'ACC': '{:.2f}'.format(ACC * 100),
                                  'F1': '{:.2f}'.format(F1 * 100),
                                  'best_acc': '{:.2f}'.format(best_acc * 100),
                                  'best_f1': '{:.2f}'.format(best_F1 * 100),
                                  'ACC_val': '{:.2f}'.format(val_acc * 100),
                                  'F1_val': '{:.2f}'.format(val_f1 * 100)})
                pbar.update(1)

            del output, ep, nc
            torch.cuda.empty_cache()

    cost_time = time.time() - begin_time
    model.eval()
    _, output, _ = best_model(features, adj_norm_list, adj_hat_list)
    output = sum(output)
    pred_labels = torch.argmax(output, 1).cpu().detach().numpy()
    ACC, F1 = get_evaluation_results(labels.cpu().detach().numpy()[idx_unlabeled],pred_labels[idx_unlabeled])

    print("Evaluating the model")
    print("ACC: {:.2f}, F1: {:.2f}".format(ACC*100, F1*100))

    return ACC*100, F1*100, cost_time, loss_list, val_acc_list, val_f1_list, train_acc_list, train_f1_list


def caculate_loss(ep, nc, norm_weights, pose_weights, adj_hat_list, loss_function, num_view, labels, idx_labeled, beta):
    nc = sum(nc)
    loss_nc = loss_function(nc[idx_labeled], labels[idx_labeled])
    loss_ep = 0.0
    for i in range(num_view):
        loss_ep += norm_weights[i] * F.binary_cross_entropy_with_logits(ep[i].to_dense(), adj_hat_list[i].to_dense(),
                                                                        pos_weight=pose_weights[i])

    loss = beta * loss_ep + loss_nc
    return loss
