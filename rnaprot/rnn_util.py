import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn import metrics
from rnaprot.RNNNets import GRUModel
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import shutil
import sys
import os


"""
Collection of model utility functions.

"""

################################################################################

def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.FloatTensor(target)
    return [data, target]


################################################################################

def my_embedding(batch_data):
    seq_lens = [len(s) for s in batch_data]
    padded = pad_sequence(batch_data, batch_first=True)
    packed = pack_padded_sequence(padded, seq_lens, batch_first=True, enforce_sorted=False)
    return packed.cuda()


################################################################################

def mysort(batch_data, batch_labels):
    lens = [e.shape[0] for e in batch_data]
    zipped_lists = zip(lens, batch_data, batch_labels)
    sorted_pairs = sorted(zipped_lists, key=lambda x: x[0], reverse=True)
    _, batch_data, batch_labels = zip(*sorted_pairs)
    batch_labels = torch.tensor(batch_labels).long().cuda()
    return batch_data, batch_labels


################################################################################

def train(model, optimizer, train_loader, criterion):
    model.train()
    loss_all = 0
    for batch_data, batch_labels in train_loader:
        batch_data, batch_labels = mysort(batch_data, batch_labels)
        embed = my_embedding(batch_data)
        optimizer.zero_grad()
        outputs = model(embed)
        loss = criterion(outputs[0], batch_labels)
        loss_all += loss.item() * len(batch_labels)
        loss.backward()
        optimizer.step()
    return loss_all / len(train_loader.dataset)


################################################################################

def test(loader, model, criterion):
    model.eval()
    loss_all = 0
    score_all = []
    test_labels = []
    correct = 0
    for batch_data, batch_labels in loader:
        batch_data, batch_labels = mysort(batch_data, batch_labels)
        embed = my_embedding(batch_data)
        outputs = model(embed)
        loss = criterion(outputs[0], batch_labels)
        loss_all += loss.item() * len(batch_labels)
        #output = torch.exp(outputs[0])
        #output = output.cpu().detach().numpy()[:, 1]
        #score_all.extend(output)
        correct += (outputs[0].argmax(dim=1) == batch_labels).sum().item()
        score_all.extend(outputs[0].cpu().detach().numpy()[:, 1])
        test_labels.extend(batch_labels.cpu().detach().numpy())
    #predicted_labels = [1 if s >= 0.5 else 0 for s in score_all]
    test_acc = correct / len(test_labels)
    fpr, tpr, thresholds = metrics.roc_curve(test_labels, score_all, pos_label=1)
    test_auc = metrics.auc(fpr, tpr)
    #test_acc = metrics.accuracy_score(test_labels, predicted_labels)
    test_loss = loss_all / len(loader.dataset)
    return test_loss, test_acc, test_auc




"""
    model.eval()
    site_scores = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            #l_x = len(data.x)
            #print("len.x:", l_x)
            #print("data.x:", data.x)
            #print("data.y:", data.y)
            output = model(data.x, data.edge_index, data.batch)
            output = torch.exp(output)
            output = output.cpu().detach().numpy()[:, 1]
            if min_max_norm:
                for o in output:
                    o_norm = min_max_normalize_probs(o, 1, 0, borders=[-1, 1])
                    site_scores.append(o_norm)
            else:
                site_scores.extend(output)
    return site_scores
"""


################################################################################

def test_scores(loader, model, criterion,
                min_max_norm=True):
    model.eval()
    site_scores = []
    loss_all = 0
    score_all = []
    test_labels = []
    correct = 0
    for batch_data, batch_labels in loader:
        batch_data, batch_labels = mysort(batch_data, batch_labels)
        embed = my_embedding(batch_data)
        outputs = model(embed)
        loss = criterion(outputs[0], batch_labels)
        loss_all += loss.item() * len(batch_labels)
        output = torch.exp(outputs[0])
        #output = output.cpu().detach().numpy()[:, 1]
        #score_all.extend(output)
        #output = torch.exp(output)
        output = output.cpu().detach().numpy()[:, 1]
        if min_max_norm:
            for o in output:
                o_norm = min_max_normalize_probs(o, 1, 0, borders=[-1, 1])
                site_scores.append(o_norm)
        else:
            site_scores.extend(output)

        #correct += (outputs[0].argmax(dim=1) == batch_labels).sum().item()
        #score_all.extend(outputs[0].cpu().detach().numpy()[:, 1])
        test_labels.extend(batch_labels.cpu().detach().numpy())
    #predicted_labels = [1 if s >= 0.5 else 0 for s in score_all]
    #test_acc = correct / len(test_labels)
    #fpr, tpr, thresholds = metrics.roc_curve(test_labels, score_all, pos_label=1)
    #test_auc = metrics.auc(fpr, tpr)
    #test_acc = metrics.accuracy_score(test_labels, predicted_labels)
    #test_loss = loss_all / len(loader.dataset)
    #return test_loss, test_acc, test_auc
    return site_scores


################################################################################

def min_max_normalize_probs(x, max_x, min_x,
                            borders=False):
    """
    Min-max normalization of input x, given dataset max and min.

    >>> min_max_normalize_probs(20, 30, 10)
    0.5
    >>> min_max_normalize_probs(30, 30, 10)
    1.0
    >>> min_max_normalize_probs(10, 30, 10)
    0.0
    >>> min_max_normalize_probs(0.5, 1, 0, borders=[-1, 1])
    0.0

    Formula from:
    https://en.wikipedia.org/wiki/Feature_scaling

    """
    # If min=max, all values the same, so return x.
    if (max_x - min_x) == 0:
        return x
    else:
        if borders:
            assert len(borders) == 2, "list of 2 values expected"
            a = borders[0]
            b = borders[1]
            assert a < b, "a should be < b"
            return a + (x-min_x)*(b-a) / (max_x - min_x)
        else:
            return (x-min_x) / (max_x - min_x)


################################################################################

def select_model(args, n_features, train_dataset, val_dataset,
                 model_folder, device, criterion,
                 plot_lc_folder=False,
                 verbose=False,
                 hps2auc_dic=None,
                 hps2epo_dic=None):
    """
    Run hyperparameter optimization to select optimal parameters / model.

    plot_lc_folder:
        If set, plot learning curves into this folder.

    """

    opt_batch_size = args.list_batch_size[0]
    opt_n_hidden_dim = args.list_n_hidden_dim[0]
    opt_n_hidden_layers = args.list_n_hidden_layers[0]
    opt_weight_decay = args.list_weight_decay[0]
    opt_lr = args.list_lr[0]
    opt_dr = args.list_dr[0]
    opt_val_loss = 1000000000.0
    i_round = 0
    opt_acc = 0
    opt_auc = 0
    opt_epochs = 0
    opt_dic = {}
    if plot_lc_folder:
        if not os.path.exists(plot_lc_folder):
            os.makedirs(plot_lc_folder)

    for batch_size in args.list_batch_size:
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, collate_fn=my_collate, pin_memory=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, collate_fn=my_collate, pin_memory=True)
        for n_hidden_layers in args.list_n_hidden_layers:
            for n_hidden_dim in args.list_n_hidden_dim:
                for weight_decay in args.list_weight_decay:
                    for lr in args.list_lr:
                        for dr in args.list_dr:
                            i_round += 1
                            # Hyperparameter string.
                            hp_str = str(batch_size) + "_" + str(n_hidden_layers) + "_" + str(n_hidden_dim) + "_" + str(weight_decay) + "_" + str(lr) + "_" + str(dr)
                            model_path = model_folder + "/" + hp_str
                            if verbose:
                                print("Round %i with %s" %(i_round, hp_str))
                            model = GRUModel(n_features, n_hidden_dim, n_hidden_layers,
                                             args.n_class, device,
                                             dr=dr).to(device)
                            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
                            # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

                            best_val_loss = 1000000000.0
                            best_val_acc = 0
                            best_val_auc = 0
                            elapsed_patience = 0
                            c_epochs = 0
                            tll = [] # train loss list.
                            vll = [] # validation loss list.

                            for epoch in range(0, args.epochs):
                                c_epochs += 1
                                if elapsed_patience > args.patience:
                                    break
                                train_loss = train(model, optimizer, train_loader, criterion)
                                val_loss, val_acc, val_auc = test(val_loader, model, criterion)
                                if verbose:
                                    print('Epoch {}: ({}, {}, {}, {})'.format(epoch, train_loss, val_loss, val_acc, val_auc))
                                tll.append(train_loss)
                                vll.append(val_loss)

                                if val_loss < best_val_loss:
                                    if verbose:
                                        print("Saving model ...")
                                    elapsed_patience = 0
                                    best_val_loss = val_loss
                                    best_val_acc = val_acc
                                    best_val_auc = val_auc
                                    torch.save(model.state_dict(), model_path)
                                else:
                                    elapsed_patience += 1

                            if plot_lc_folder:
                                lc_plot = plot_lc_folder + "/" + hp_str + ".lc.png"
                                assert tll, "tll empty"
                                assert vll, "vll empty"
                                create_lc_loss_plot(tll, vll, lc_plot)

                            # Store used epochs and best accuarcy for this HP combination.
                            if hps2auc_dic is not None:
                                if hp_str not in hps2auc_dic:
                                    hps2auc_dic[hp_str] = []
                                    hps2auc_dic[hp_str].append(best_val_auc)
                                else:
                                    hps2auc_dic[hp_str].append(best_val_auc)
                            if hps2epo_dic is not None:
                                if hp_str not in hps2epo_dic:
                                    hps2epo_dic[hp_str] = []
                                    hps2epo_dic[hp_str].append(c_epochs)
                                else:
                                    hps2epo_dic[hp_str].append(c_epochs)

                            if best_val_loss < opt_val_loss:
                                opt_val_loss = best_val_loss
                                opt_n_hidden_dim = n_hidden_dim
                                opt_n_hidden_layers = n_hidden_layers
                                opt_weight_decay = weight_decay
                                opt_lr = lr
                                opt_dr = dr
                                opt_acc = best_val_acc
                                opt_auc = best_val_auc
                                opt_epochs = c_epochs
                                opt_batch_size = batch_size

    opt_dic["opt_batch_size"] = opt_batch_size
    opt_dic["opt_n_hidden_dim"] = opt_n_hidden_dim
    opt_dic["opt_n_hidden_layers"] = opt_n_hidden_layers
    opt_dic["opt_weight_decay"] = opt_weight_decay
    opt_dic["opt_lr"] = opt_lr
    opt_dic["opt_dr"] = opt_dr
    opt_dic["opt_acc"] = opt_acc
    opt_dic["opt_auc"] = opt_auc
    opt_dic["opt_epochs"] = opt_epochs

    return opt_dic


################################################################################

def create_lc_loss_plot(train_loss, val_loss, out_plot):
    """
    Input two lists with loss values on training set (train_loss), and on
    validation set (val_loss). Length of the list == number of epochs.

    """
    assert train_loss, "given train_loss list empty"
    assert val_loss, "given val_loss list empty"
    l_tl = len(train_loss)
    l_vl = len(val_loss)
    assert l_tl == l_vl, "differing list lengths for train_loss and val_loss"
    # Make pandas dataframe.
    data = {'set': [], 'epoch': [], 'loss': []}
    for i,tl in enumerate(train_loss):
        epoch = i+1
        data['set'].append('train_loss')
        data['loss'].append(tl)
        data['epoch'].append(epoch)
    for i,vl in enumerate(val_loss):
        epoch = i+1
        data['set'].append('validation_loss')
        data['loss'].append(vl)
        data['epoch'].append(epoch)
    df = pd.DataFrame (data, columns = ['set','loss', 'epoch'])
    #fig, ax = plt.subplots()
    fig = plt.figure()
    sns.lineplot(data=df, x="epoch", y="loss", hue="set")
    fig.savefig(out_plot, dpi=125, bbox_inches='tight')
    plt.close(fig)


################################################################################
