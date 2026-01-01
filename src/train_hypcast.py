# ---- coding: utf-8 ----
# @author: Ziyang Zhang et al.
# @version: v1, Task-guided Co-clustering Framework
# This work partly uses the code from CACHE.


import json
import os
import argparse
import torch.nn as nn
from tqdm import trange
from models import Hypergraph, DEC
from preprocessing import *
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import f1_score



@torch.no_grad()
def evaluate(model, data, split_idx, eval_func, epoch, method, dname, args):
    model.eval()

    out_score_g_logits, edge_feat, node_feat, weight_tuple = model(data)
    out_g = torch.sigmoid(out_score_g_logits)

    valid_acc, valid_auroc, valid_aupr, valid_f1_macro, valid_sensitivity = eval_func(data.y[split_idx['valid']],
                                                                   out_g[split_idx['valid']], epoch, method, dname,
                                                                   args, threshold=args.threshold)
    test_acc, test_auroc, test_aupr, test_f1_macro, test_sensitivity = eval_func(data.y[split_idx['test']],
                                                               out_g[split_idx['test']],
                                                               epoch, method, dname, args,
                                                               threshold=args.threshold)
    

    return valid_acc, valid_auroc, valid_aupr, valid_f1_macro, valid_sensitivity, \
           test_acc, test_auroc, test_aupr, test_f1_macro, test_sensitivity


from sklearn.metrics import recall_score

def eval_demo(y_true, y_pred, epoch, method, dname, args, threshold=0.5):
    # Convert tensor outputs to numpy arrays
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    # Binarize predictions based on threshold
    pred = np.array(y_pred > threshold).astype(int)

    # Calculate per-label accuracy and F1 score (not used in final output)
    total_acc = []
    total_f1 = []
    for i in range(args.num_labels):
        correct = (pred[:, i] == y_true[:, i])
        accuracy = correct.sum() / correct.size
        total_acc.append(accuracy)
        f1_macro = f1_score(y_true[:, i], pred[:, i], average='macro')
        total_f1.append(f1_macro)

    # Overall accuracy and F1 score (macro)
    correct = (pred == y_true)
    accuracy = correct.sum() / correct.size
    f1_macro = f1_score(y_true, pred, average='macro')

    # Calculate per-label and overall ROC AUC
    total_auc = []
    for i in range(args.num_labels):
        roc_auc = roc_auc_score(y_true[:, i].reshape(-1), y_pred[:, i].reshape(-1))
        total_auc.append(roc_auc)
    roc_auc = roc_auc_score(y_true.reshape(-1), y_pred.reshape(-1))

    # Calculate per-label and overall AUPR
    total_aupr = []
    for i in range(args.num_labels):
        aupr = average_precision_score(y_true[:, i].reshape(-1), y_pred[:, i].reshape(-1))
        total_aupr.append(aupr)
    aupr = average_precision_score(y_true.reshape(-1), y_pred.reshape(-1))

    # Calculate sensitivity as macro recall
    sensitivity = recall_score(y_true, pred, average='macro')

    return accuracy, roc_auc, aupr, f1_macro, sensitivity






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_prop', type=float, default=0.7)
    parser.add_argument('--valid_prop', type=float, default=0.1)
    parser.add_argument('--dname', default='demo')
    parser.add_argument('--method', default='AllSetTransformer')
    parser.add_argument('--text', type=int, default=1)  # 1 for encode text
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--cuda', default='1', type=str)
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--wd', default=1e-3, type=float)
    parser.add_argument('--warmup', default=10, type=int)  # 0 for direct training
    parser.add_argument('--LearnFeat', action='store_true')
    parser.add_argument('--All_num_layers', default=1, type=int)  # hyperparameter L
    parser.add_argument('--MLP_num_layers', default=1, type=int)
    parser.add_argument('--MLP_hidden', default=48, type=int)  # hyperparameter d
    parser.add_argument('--num_cluster', type=int, default=5)  # hyperparameter K
    parser.add_argument('--Classifier_num_layers', default=2, type=int)
    parser.add_argument('--Classifier_hidden', default=64, type=int)
    parser.add_argument('--aggregate', default='mean', choices=['sum', 'mean'])
    parser.add_argument('--normtype', default='all_one')  # ['all_one','deg_half_sym']
    parser.add_argument('--add_self_loop', action='store_false')
    parser.add_argument('--normalization', default='ln')  # NormLayer for MLP. ['bn','ln','None']
    parser.add_argument('--num_features', default=0, type=int)  # placeholder
    parser.add_argument('--PMA', action='store_true')
    parser.add_argument('--heads', default=1, type=int)  # attention heads
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=10)  # hyperparameter α

    parser.set_defaults(PMA=True)
    parser.set_defaults(add_self_loop=True)
    parser.set_defaults(LearnFeat=True)

    args = parser.parse_args()

    dname = args.dname
    p2raw = '../data/raw_data/'
    
    # Automatically detect num_nodes and feature_dim from node-embeddings file
    node_feature_path = os.path.join(p2raw, dname, f'node-embeddings-{dname}')
    if os.path.exists(node_feature_path):
        with open(node_feature_path, 'r') as f:
            first_line = f.readline().strip()
            parts = first_line.split()
            args.num_nodes = int(parts[0])
            args.feature_dim = int(parts[1])
        print(f'Automatically detected num_nodes: {args.num_nodes}, feature_dim: {args.feature_dim}')
    else:
        # Fallback: try to read from processed data if exists
        processed_path = os.path.join('../data/pyg_data/hypergraph_dataset', dname, 'processed/data.pt')
        if os.path.exists(processed_path):
            # Will be detected from data after loading
            args.num_nodes = None
            args.feature_dim = None
        else:
            raise FileNotFoundError(f'Cannot find node-embeddings file at {node_feature_path}')
    
    dataset = dataset_Hypergraph(name=dname, root='../data/pyg_data/hypergraph_dataset/',
                                 p2raw=p2raw, num_nodes=args.num_nodes, text=args.text)
    data = dataset[0]
    args.num_features = dataset.num_features
    
    # Verify feature_dim matches actual data dimension
    # If text features are used, actual dimension might be 2x the embedding dimension
    actual_feature_dim = data.x.shape[1]
    if args.feature_dim is None:
        args.feature_dim = actual_feature_dim
        print(f'Using feature_dim from data: {args.feature_dim}')
    elif args.feature_dim != actual_feature_dim:
        # If there's a mismatch, use the actual dimension from data
        print(f'Warning: feature_dim from file ({args.feature_dim}) does not match data dimension ({actual_feature_dim}). Using {actual_feature_dim}.')
        args.feature_dim = actual_feature_dim
    
    # Automatically detect number of labels from data
    args.num_labels = data.y.shape[1] if len(data.y.shape) > 1 else 1
    print(f'Automatically detected num_labels: {args.num_labels}')
    
    if args.dname == 'demo':
        # Shift the y label to start with 0
        data.y = data.y - data.y.min()
    if not hasattr(data, 'n_x'):
        data.n_x = torch.tensor([data.x.shape[0]])
    if not hasattr(data, 'num_hyperedges'):
        # note that we assume the he_id is consecutive.
        data.num_hyperedges = torch.tensor(
            [data.edge_index[0].max() - data.n_x[0] + 1])

    if args.method == 'AllSetTransformer':
        data = ExtractV2E(data)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
        data = norm_contruction(data, option=args.normtype)

    # hypergraph transformer
    model = Hypergraph(args, data)
    # edge clustering
    edge_cluster = DEC(num_cluster=args.num_cluster, feat_dim=args.MLP_hidden)

    # put things to device
    if args.cuda != '-1':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    # train-valid-test split
    split_idx = rand_train_test_idx(data.y, train_prop=args.train_prop, valid_prop=args.valid_prop)
    train_idx = split_idx['train'].to(device)

    model, data, edge_cluster = (model.to(device), data.to(device), edge_cluster.to(device))

    # Create the loss function without pos_weight
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    model.reset_parameters()
    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    edge_cluster_optimizer = torch.optim.Adam(edge_cluster.parameters(), lr=args.lr, weight_decay=args.wd)

    # training logs
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('../logs/', current_time)
    os.makedirs(log_dir, exist_ok=True)
    
    # start training
    with torch.autograd.set_detect_anomaly(True):
        for epoch in trange(args.epochs):
            if epoch < args.warmup:
                """STEP ONE - WARMUP THE TRANSFORMER"""
                model.train()
                model.zero_grad()

                out_score_logits, _, _, _ = model(data)
                out = torch.sigmoid(out_score_logits)

                warmup_loss = criterion(out[train_idx], data.y[train_idx])
                warmup_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                model_optimizer.step()

            else:
                """STEP TWO - TRAIN THE WHOLE MODEL"""
                model.train()
                model.zero_grad()
                edge_cluster.train()
                edge_cluster.zero_grad()

                out_score_logits, out_edge_feat, _, _ = model(data)

                # clustering loss
                edge_cluster_loss = edge_cluster.loss(out_edge_feat, epoch)

                # classifier loss
                cls_loss = criterion(out_score_logits[train_idx], data.y[train_idx])

                # reg loss
                # Obtain the cluster assignment probabilities (edge_Q) with shape (n, k)
                edge_Q = edge_cluster.get_Q()
                # Compute the proportion for each cluster by summing over samples and dividing by n
                cluster_proportions = torch.sum(edge_Q, dim=0) / edge_Q.shape[0]
                # Create a uniform distribution vector of shape (k,) where each element is 1/k on the same device
                uniform_dist = torch.full_like(cluster_proportions, 1.0 / cluster_proportions.shape[0])
                # Compute the KL divergence between the computed distribution and the uniform distribution
                reg_loss = torch.sum(cluster_proportions * torch.log((cluster_proportions + 1e-10) / uniform_dist))

                # final loss
                model_loss = cls_loss + args.alpha * edge_cluster_loss + 10*reg_loss

                model_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                model_optimizer.step()
                edge_cluster_optimizer.step()

            eval_function = eval_demo
            valid_acc, valid_auroc, valid_aupr, valid_f1_macro, valid_sensitivity, test_acc, test_auroc, test_aupr, test_f1_macro, test_sensitivity = evaluate(model, data, split_idx, eval_function, epoch, args.method, args.dname, args)

            # training logs
            fname_valid = f'{dname}_valid_{args.method}.txt'
            fname_test = f'{dname}_test_{args.method}.txt'

            fname_valid = os.path.join(log_dir, fname_valid)
            fname_test = os.path.join(log_dir, fname_test)
            fname_hyperparameters = os.path.join(log_dir, 'hyperparameters.txt')

            # save hyperparams
            with open(fname_hyperparameters, 'w', encoding='utf-8') as f:
                args_dict = vars(args)
                f.write(json.dumps(args_dict, indent=4))

            # valid set
            with open(fname_valid, 'a+', encoding='utf-8') as f:
                f.write('Epoch: {}, ACC: {:.5f}, AUROC: {:.5f}, AUPR: {:.5f}, F1_MACRO: {:.5f}, SENSITIVITY: {:.5f}\n'
                    .format(epoch + 1, valid_acc, valid_auroc, valid_aupr, valid_f1_macro, valid_sensitivity))

            # test set
            with open(fname_test, 'a+', encoding='utf-8') as f:
                f.write('Epoch: {}, ACC: {:.5f}, AUROC: {:.5f}, AUPR: {:.5f}, F1_MACRO: {:.5f}, SENSITIVITY: {:.5f}\n'
                    .format(epoch + 1, test_acc, test_auroc, test_aupr, test_f1_macro, test_sensitivity))

    print(f'Training finished. Logs are saved in {log_dir}.')
    
    print("Generating and saving test set probabilities...")
    model.eval()
    with torch.no_grad():
        out_score_g_logits, out_edge_feat, _, _ = model(data)
        out_g = torch.sigmoid(out_score_g_logits)
        test_probabilities = out_g[split_idx['test']]
        test_probabilities_np = test_probabilities.cpu().detach().numpy()
        
        test_gt = data.y[split_idx['test']].cpu().detach().numpy()
        gt_output_path = os.path.join(log_dir, 'hyg_test_gt.csv')
        np.savetxt(gt_output_path, test_gt, fmt='%d', delimiter=',')
        print(f"Test ground-truth labels successfully saved to {gt_output_path}")

        prob_output_path = os.path.join(log_dir, 'hyg_prob.csv')
        np.savetxt(prob_output_path, test_probabilities_np, delimiter=',')
        print(f"Test probabilities successfully saved to {prob_output_path}")
    
    edge_Q = edge_cluster.get_Q()
    edge_onehot = edge_cluster.predict()
    edge_label = np.argmax(edge_onehot.cpu().detach().numpy(), axis=1)
    
    import numpy as np
    
    np.save(os.path.join(log_dir, 'edge_feat.npy'), out_edge_feat.cpu().detach().numpy())
    np.save(os.path.join(log_dir, 'edge_Q.npy'), edge_Q.cpu().detach().numpy())
    np.save(os.path.join(log_dir, 'edge_label.npy'), edge_label)
