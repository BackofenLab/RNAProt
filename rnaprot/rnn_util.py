import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn import metrics
from rnaprot.RNNNets import RNNModel
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import shutil
import sys
import re
import os
# New from BOHB.
from torch.nn import BCEWithLogitsLoss
#import torch.nn as nn
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB
#import pickle
#import time
# Logging needed ?
import logging
logging.basicConfig(level=logging.DEBUG) # or try logging.WARNING
#import torch.utils.data needed?


"""
Collection of functions for train, predict, test, and hyperparameter
optimization.

"""


################################################################################

def run_BOHB(args, train_dataset, val_dataset, bohb_out_folder,
             n_bohb_iter=10,
             min_budget=3,
             max_budget=27):

    #nic_name = 'lo'
    #port = None
    host = '127.0.0.1'
    run_id = 'bohb_run_id'

    NS = hpns.NameServer(run_id=run_id, host=host, port=None)
    NS.start()

    w = MyWorker(args, train_dataset, val_dataset,
                 sleep_interval=0, nameserver=host,run_id=run_id)
    w.run(background=True)

    result_logger = hpres.json_result_logger(directory=bohb_out_folder, overwrite=True)

    bohb = BOHB(configspace=w.get_configspace(),
                run_id=run_id,
                nameserver=host,
                min_budget=min_budget,
                max_budget=max_budget)

    result = bohb.run(n_iterations=n_bohb_iter)

    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()

    all_runs_max_budget = result.get_all_runs(only_largest_budget=True)
    id2conf = result.get_id2config_mapping()
    all_runs_max_budget = [r for r in all_runs_max_budget if r['loss'] != None] # get the erronous runs out
    best_run = min(all_runs_max_budget, key=lambda x: x["loss"])
    best_conf = id2conf[best_run['config_id']]
    best_conf = best_conf['config']

    # Get best AUC / ACC.
    results_json_file = bohb_out_folder + "/results.json"
    assert os.path.exists(results_json_file), "BOHB results file %s missing" %(results_json_file)

    best_loss = 1000
    best_auc = 0
    best_acc = 0

    with open(results_json_file) as f:
        for line in f:
            if re.search("loss.+val acc.+val auc", line):
                m = re.search(r"loss\": (.+?),.+val acc\": (.+?),.+val auc\": (.+?)\}", line)
                loss = float(m.group(1))
                if loss < best_loss:
                    best_loss = loss
                    best_acc = float(m.group(2))
                    best_auc = float(m.group(3))
    f.closed

    assert best_auc, "no AUCs extracted from %s" %(results_json_file)
    assert best_acc, "no ACCs extracted from %s" %(results_json_file)

    # Return best HPs, ACC, and AUC.
    return best_conf, best_acc, best_auc


################################################################################

class MyWorker(Worker):
    def __init__(self, args, train_dataset, val_dataset, **kwargs):
        super().__init__(**kwargs)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.n_class = args.n_class
        self.n_feat = args.n_feat
        self.embed = args.embed
        self.embed_vocab_size = args.embed_vocab_size
        self.add_feat = args.add_feat
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def compute(self, config, budget, **kwargs):

        train_loader = DataLoader(dataset=self.train_dataset,
                                  batch_size=config['batch_size'],
                                  collate_fn=pad_collate, pin_memory=True)
        val_loader = DataLoader(dataset=self.val_dataset,
                                batch_size=config['batch_size'],
                                collate_fn=pad_collate, pin_memory=True)

        model = RNNModel(self.n_feat, self.n_class, self.device,
                         rnn_type=config['rnn_type'],
                         rnn_n_layers=config['n_rnn_layers'],
                         rnn_hidden_dim=config['n_rnn_dim'],
                         bidirect=config['bidirect'],
                         dropout_rate=config['dropout_rate'],
                         add_fc_layer=config['add_fc_layer'],
                         embed_dim=config['embed_dim'],
                         embed=self.embed,
                         embed_vocab_size=self.embed_vocab_size,
                         add_feat=self.add_feat).to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learn_rate'], weight_decay=config['weight_decay'])
        criterion = BCEWithLogitsLoss()

        for epoch in range(int(budget)):
            train_loss = train(model, optimizer, train_loader, criterion, self.device)

        val_loss, val_acc, val_auc = test(val_loader, model, criterion, self.device)

        return({
            'loss': 1-val_auc,
            'info': {'val_acc': val_acc,
                     'val_auc': val_auc
                     }
        })

    @staticmethod
    def get_configspace():
        cs = CS.ConfigurationSpace()
        lr = CSH.UniformFloatHyperparameter('learn_rate', lower=1e-6, upper=1e-1, default_value='1e-2', log=True)
        wd = CSH.UniformFloatHyperparameter(name='weight_decay', lower=1e-8, upper=1e-1, log=True)
        embed_dim = CSH.UniformIntegerHyperparameter('embed_dim', lower=4, upper=24, default_value=10)
        n_rnn_dim = CSH.CategoricalHyperparameter('n_rnn_dim', choices=[32, 64, 96], default_value=32)
        n_rnn_layers = CSH.CategoricalHyperparameter('n_rnn_layers', choices=[1, 2, 3], default_value=2)
        dr = CSH.UniformFloatHyperparameter('dropout_rate', lower=0.0, upper=0.9, default_value=0.5, log=False)
        bidirect = CSH.CategoricalHyperparameter('bidirect', choices=[False, True], default_value=True)
        add_fc_layer = CSH.CategoricalHyperparameter('add_fc_layer', choices=[False, True], default_value=True)
        rnn_type = CSH.CategoricalHyperparameter('rnn_type', choices=[1, 2], default_value=1)
        batch_size = CSH.CategoricalHyperparameter('batch_size', choices=[30, 50, 80], default_value=50)
        cs.add_hyperparameters([lr, wd, embed_dim, n_rnn_dim, n_rnn_layers, dr, bidirect, add_fc_layer, rnn_type, batch_size])

        return cs


################################################################################

def pad_collate(batch):
    (xs, ys) = zip(*batch)
    xs_lens = [len(x) for x in xs]
    xs_pad = pad_sequence(xs, batch_first=True, padding_value=0)
    ys = torch.FloatTensor([[y] for y in ys])
    return xs_pad, ys, xs_lens


###############################################################################

def train(model, optimizer, train_loader, criterion, device):
    model.train()
    loss_all = 0
    for batch_data, batch_labels, batch_lens in train_loader:
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        outputs, _ = model(batch_data, batch_lens, len(batch_labels))
        loss = criterion(outputs[0], batch_labels)
        loss_all += loss.item() * len(batch_labels)
        loss.backward()
        optimizer.step()
    return loss_all / len(train_loader.dataset)


###############################################################################

def test(test_loader, model, criterion, device):
    model.eval()
    loss_all = 0
    score_all = []
    test_labels = []
    test_acc = 0.0
    for batch_data, batch_labels, batch_lens in test_loader:
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        outputs, _ = model(batch_data, batch_lens, len(batch_labels))
        loss = criterion(outputs[0], batch_labels)
        loss_all += loss.item() * len(batch_labels)
        acc = binary_accuracy(outputs[0], batch_labels)
        test_acc += acc.item()
        score_all.extend(outputs[0].cpu().detach().numpy())
        test_labels.extend(batch_labels.cpu().detach().numpy())
    test_acc = test_acc / len(test_loader)
    fpr, tpr, thresholds = metrics.roc_curve(test_labels, score_all, pos_label=1)
    test_auc = metrics.auc(fpr, tpr)
    test_loss = loss_all / len(test_loader.dataset)
    return test_loss, test_acc, test_auc


################################################################################

def test_scores(loader, model, device,
                min_max_norm=False):
    model.eval()
    loss_all = 0
    score_all = []
    test_acc = 0.0
    for batch_data, batch_labels, batch_lens in loader:
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        outputs, _ = model(batch_data, batch_lens, len(batch_labels))
        # score_all.extend(outputs[0].cpu().detach().numpy())
        output = outputs[0].cpu().detach().numpy()
        if min_max_norm:
            for o in output:
                o_norm = min_max_normalize_probs(o, 1, 0, borders=[-1, 1])
                score_all.append(o_norm)
        else:
            score_all.extend(output)
    return score_all


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

def test_model(args, test_loader, model_path,
               device, criterion):
    """
    Run given model on test data in test_loader, return AUC and ACC.

    """

    # Define model.
    model = define_model(args, device)
    model.load_state_dict(torch.load(model_path))
    test_loss, test_acc, test_auc = test(test_loader, model, criterion, device)
    return test_auc, test_acc


################################################################################

def train_model(args, train_loader, val_loader,
                model_path, device, criterion,
                fold_i=1,
                plot_lc_folder=False,
                verbose=False):

    """
    Train a model on training set (train_loader), using the validation set
    (val_loader) to estimate model performance.

    """

    if verbose:
        print("Training --gen-cv model ... "
        print("Reporting: (train_loss, val_loss, val_acc, val_auc)")

    # Get model + optimizer.
    model, optimizer = define_model_and_optimizer(args, device)

    best_val_loss = 1000000000.0
    best_val_acc = 0
    best_val_auc = 0
    elapsed_patience = 0
    c_epochs = 0
    tll = [] # train loss list.
    vll = [] # validation loss list.

    for epoch in range(1, args.epochs+1):
        c_epochs += 1
        if elapsed_patience >= args.patience:
            break

        train_loss = train(model, optimizer, train_loader, criterion, device)
        val_loss, val_acc, val_auc = test(val_loader, model, criterion, device)
        print('Epoch {}: ({}, {}, {}, {})'.format(epoch, train_loss, val_loss, val_acc, val_auc))

        tll.append(train_loss)
        vll.append(val_loss)

        if val_loss < best_val_loss:
            print("Saving model ... ")
            elapsed_patience = 0
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_val_auc = val_auc
            torch.save(model.state_dict(), model_path)
        else:
            elapsed_patience += 1

    if plot_lc_folder:
        lc_plot = plot_lc_folder + "/cv_fold_" + str(fold_i) + ".lc.png"
        assert tll, "tll empty"
        assert vll, "vll empty"
        create_lc_loss_plot(tll, vll, lc_plot)

    return best_val_auc, best_val_acc, c_epochs


################################################################################

def define_model(args, device):
    """
    Define and return model based on args and device.

    """
    # Define model.
    model = RNNModel(args.n_feat, args.n_class, device,
                     rnn_type=args.rnn_type,
                     rnn_n_layers=args.n_rnn_layers,
                     rnn_hidden_dim=args.n_rnn_dim,
                     bidirect=args.bidirect,
                     dropout_rate=args.dropout_rate,
                     add_fc_layer=args.add_fc_layer,
                     embed=args.embed,
                     embed_vocab_size=args.embed_vocab_size,
                     embed_dim=args.embed_dim).to(device)
    return model


################################################################################

def define_model_and_optimizer(args, device):
    """
    Define and return model and optimizer based on args and device.

    """

    # Model.
    model = define_model(args, device)

    # Optimizer.
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.learn_rate,
                                  weight_decay=args.weight_decay)
    return model, optimizer


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
