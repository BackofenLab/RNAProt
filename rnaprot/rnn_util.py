import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn import BCEWithLogitsLoss
from sklearn import metrics
from rnaprot.RNNNets import RNNModel, RNNDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import statistics
import shutil
import copy
import sys
import re
import os
# >>> BOHB <<<.
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB


"""
Collection of functions for train, predict, test, and hyperparameter
optimization.

"""


################################################################################

def run_BOHB(args, train_dataset, val_dataset, bohb_out_folder,
             n_bohb_iter=10,
             min_budget=5,
             verbose_bohb=False,
             max_budget=30):

    # Logging.
    import logging
    if verbose_bohb:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    #nic_name = 'lo'
    #port = None
    host = '127.0.0.1'
    run_id = 'bohb_run_id'

    NS = hpns.NameServer(run_id=run_id, host=host, port=None)
    NS.start()

    w = MyWorker(args, train_dataset, val_dataset,
                 nameserver=host,run_id=run_id)
    w.run(background=True)

    result_logger = hpres.json_result_logger(directory=bohb_out_folder, overwrite=True)

    # Get HPs and their spaces.
    conf_space = w.get_configspace()
    print(conf_space)

    bohb = BOHB(configspace=conf_space,
                run_id=run_id,
                nameserver=host,
                result_logger=result_logger,
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
            if re.search("loss.+val_acc.+val_auc", line):
                m = re.search(r"loss\": (.+?),.+val_acc\": (.+?),.+val_auc\": (.+?)\}", line)
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
    """
    The worker is responsible for evaluating a given model with a
    single configuration on a single budget at a time.

    Worker example:
    https://automl.github.io/HpBandSter/build/html/auto_examples/example_5_pytorch_worker.html

    """
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

    """
    Defining the Search Space:
    The parameters being optimized need to be defined. HpBandSter relies
    on the ConfigSpace package for that.

    Example of conditional HPs:
    https://automl.github.io/HpBandSter/build/html/auto_examples/
    example_5_pytorch_worker.html#sphx-glr-auto-examples-example-5-pytorch-worker-py

    """
    @staticmethod
    def get_configspace():
        cs = CS.ConfigurationSpace()
        lr = CSH.UniformFloatHyperparameter('learn_rate', lower=1e-6, upper=1e-1, default_value='1e-3', log=True)
        wd = CSH.UniformFloatHyperparameter(name='weight_decay', lower=1e-8, upper=1e-1, default_value='5e-4', log=True)
        embed_dim = CSH.UniformIntegerHyperparameter('embed_dim', lower=4, upper=24, default_value=10)
        n_rnn_dim = CSH.CategoricalHyperparameter('n_rnn_dim', choices=[32, 64, 96], default_value=32)
        n_rnn_layers = CSH.CategoricalHyperparameter('n_rnn_layers', choices=[1, 2], default_value=2)
        dr = CSH.UniformFloatHyperparameter('dropout_rate', lower=0.0, upper=0.8, default_value=0.5, log=False)
        bidirect = CSH.CategoricalHyperparameter('bidirect', choices=[False, True], default_value=True)
        add_fc_layer = CSH.CategoricalHyperparameter('add_fc_layer', choices=[False, True], default_value=True)
        rnn_type = CSH.CategoricalHyperparameter('rnn_type', choices=[1, 2], default_value=1)
        batch_size = CSH.CategoricalHyperparameter('batch_size', choices=[30, 50, 80], default_value=50)
        if self.embed:
            cs.add_hyperparameters([lr, wd, embed_dim, n_rnn_dim, n_rnn_layers, dr, bidirect, add_fc_layer, rnn_type, batch_size])
        else:
            cs.add_hyperparameters([lr, wd, n_rnn_dim, n_rnn_layers, dr, bidirect, add_fc_layer, rnn_type, batch_size])
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

def binary_accuracy(preds, y):
    """
    Accuracy calculation:
    Round scores > 0 to 1, and scores <= 0 to 0 (using sigmoid function).

    """
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum()/len(correct)
    return acc


###############################################################################

def test(test_loader, model, criterion, device,
         apply_tanh=False):
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
        # Use tanh to normalize scores from -1 to 1.
        if apply_tanh:
            output = torch.tanh(outputs[0]).cpu().detach().numpy()
        else:
            output = outputs[0].cpu().detach().numpy()
        score_all.extend(output)
        test_labels.extend(batch_labels.cpu().detach().numpy())
    #print("score_all min:", min(score_all))
    #print("score_all max:", max(score_all))
    test_acc = test_acc / len(test_loader)
    fpr, tpr, thresholds = metrics.roc_curve(test_labels, score_all, pos_label=1)
    test_auc = metrics.auc(fpr, tpr)
    test_loss = loss_all / len(test_loader.dataset)
    return test_loss, test_acc, test_auc


################################################################################

def test_scores(loader, model, device,
                apply_tanh=False):
    model.eval()
    score_all = []
    for batch_data, batch_labels, batch_lens in loader:
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        outputs, _ = model(batch_data, batch_lens, len(batch_labels))
        if apply_tanh:
            output = torch.tanh(outputs[0][:,0]).cpu().detach().numpy()
        else:
            output = outputs[0].cpu().detach().numpy()[:,0]
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

def test_model(args, test_loader, model_path, device, criterion,
               opt_dic=False):
    """
    Run given model on test data in test_loader, return AUC and ACC.

    """

    # Define model.
    model = define_model(args, device,
                         opt_dic=opt_dic)
    model.load_state_dict(torch.load(model_path))
    test_loss, test_acc, test_auc = test(test_loader, model, criterion, device)
    return test_auc, test_acc


################################################################################

def train_model(args, train_loader, val_loader,
                model_path, device, criterion,
                opt_dic=False,
                run_id="some_model",
                plot_lc_folder=False,
                verbose=False):

    """
    Train a model on training set (train_loader), using the validation set
    (val_loader) to estimate model performance.

    """

    if verbose:
        print("Reporting: (train_loss, val_loss, val_acc, val_auc)")

    # Get model + optimizer.
    model, optimizer = define_model_and_optimizer(args, device,
                                                  opt_dic=opt_dic)

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
        if verbose:
            print('Epoch {}: ({}, {}, {}, {})'.format(epoch, train_loss, val_loss, val_acc, val_auc))

        tll.append(train_loss)
        vll.append(val_loss)

        if val_loss < best_val_loss:
            if verbose:
                print("Saving model ... ")
            elapsed_patience = 0
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_val_auc = val_auc
            torch.save(model.state_dict(), model_path)
        else:
            elapsed_patience += 1

    if plot_lc_folder:
        lc_plot = plot_lc_folder + "/" + run_id + ".lc.png"
        assert tll, "tll empty"
        assert vll, "vll empty"
        create_lc_loss_plot(tll, vll, lc_plot)

    return best_val_auc, best_val_acc, c_epochs


################################################################################

def define_model(args, device,
                 opt_dic=False):
    """
    Define and return model based on args and device. Alternatively use
    hyperparameters from provided opt_dic.

    """

    # Define model.
    if opt_dic:
        model = RNNModel(opt_dic["n_feat"], opt_dic["n_class"], device,
                         rnn_type=opt_dic["rnn_type"],
                         rnn_n_layers=opt_dic["n_rnn_layers"],
                         rnn_hidden_dim=opt_dic["n_rnn_dim"],
                         bidirect=opt_dic["bidirect"],
                         dropout_rate=opt_dic["dropout_rate"],
                         add_fc_layer=opt_dic["add_fc_layer"],
                         add_feat=opt_dic["add_feat"],
                         embed=opt_dic["embed"],
                         embed_vocab_size=opt_dic["embed_vocab_size"],
                         embed_dim=opt_dic["embed_dim"]).to(device)
    else:
        model = RNNModel(args.n_feat, args.n_class, device,
                         rnn_type=args.rnn_type,
                         rnn_n_layers=args.n_rnn_layers,
                         rnn_hidden_dim=args.n_rnn_dim,
                         bidirect=args.bidirect,
                         dropout_rate=args.dropout_rate,
                         add_fc_layer=args.add_fc_layer,
                         add_feat=args.add_feat,
                         embed=args.embed,
                         embed_vocab_size=args.embed_vocab_size,
                         embed_dim=args.embed_dim).to(device)
    return model


################################################################################

def define_model_and_optimizer(args, device,
                               opt_dic=False):
    """
    Define and return model and optimizer based on args and device.
    Alternatively use hyperparameters from provided opt_dic.

    """

    if opt_dic:
        # Model.
        model = define_model(args, device,
                             opt_dic=opt_dic)

        # Optimizer.
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=opt_dic["learn_rate"],
                                      weight_decay=opt_dic["weight_decay"])
    else:
        # Model.
        model = define_model(args, device)

        # Optimizer.
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=args.learn_rate,
                                      weight_decay=args.weight_decay)
    return model, optimizer


###############################################################################

def get_saliency(loader, model, device):
    """
    Get saliency. Only tested for batch_size = 1.

    """
    model.train()
    model.dropout.eval()
    all_saliency = []
    for batch_data, batch_labels, batch_lens in loader:
        batch_data = batch_data.to(device)
        outputs, x_embed = model(batch_data, batch_lens, len(batch_labels))
        outputs[0].backward()
        saliency = x_embed.grad.data.abs().squeeze()
        all_saliency.append(saliency.cpu().detach().numpy())

    return all_saliency


################################################################################

def get_saliency_from_feat_list(feat_list, model, device,
                                got_tensors=True,
                                sal_type=1):
    """
    Given a features list from one instance, get saliencies of this instance.
    Return list of saliencies.
n_class
    feat_list:
        Feature list of tensors (if not disable got_tensors).
    model:
        Loaded model.
    got_tensors:
        Set True if feat_list contains tensors.
    sal_type:
        Type of saliency returned. 1: mean saliencies, 2: max saliencies.

    """
    from statistics import mean

    sal_feat_list = []
    if got_tensors:
        sal_feat_list.append(feat_list)
    else:
        sal_feat_list.append(torch.tensor(feat_list, dtype=torch.float))

    sal_dataset = RNNDataset(sal_feat_list, [1])
    sal_loader = DataLoader(dataset=sal_dataset, batch_size=1, collate_fn=pad_collate, pin_memory=True)
    sal_ll = get_saliency(sal_loader, model, device)
    mean_sal_list = []
    max_sal_list = []
    for scl in sal_ll[0]:
        mean_sal_list.append(mean(scl))
        max_sal_list.append(max(scl))

    l_sl = len(mean_sal_list)
    l_fl = len(feat_list)

    assert l_sl == l_fl, "len(mean_sal_list) != len(feat_list) (%i != %i)" %(l_sl, l_fl)

    if sal_type == 1:
        return mean_sal_list
    elif sal_type == 2:
        return max_sal_list
    else:
        assert False, "invalid sal_type given"


################################################################################

def list_moving_win_avg_values(in_list,
                               win_extlr=5,
                                method=1):
    """
    Take a list of numeric values, and calculate for each position a new value,
    by taking the mean value of the window of positions -win_extlr and
    +win_extlr. If full extension is not possible (at list ends), it just
    takes what it gets.
    Two implementations of the task are given, chose by method=1 or method=2.

    >>> test_list = [2, 3, 5, 8, 4, 3, 7, 1]
    >>> list_moving_window_average_values(test_list, win_extlr=2, method=1)
    [3.3333333333333335, 4.5, 4.4, 4.6, 5.4, 4.6, 3.75, 3.6666666666666665]
    >>> list_moving_window_average_values(test_list, win_extlr=2, method=2)
    [3.3333333333333335, 4.5, 4.4, 4.6, 5.4, 4.6, 3.75, 3.6666666666666665]

    """
    l_list = len(in_list)
    assert l_list, "Given list is empty"
    new_list = [0] * l_list
    if win_extlr == 0:
        return l_list
    if method == 1:
        for i in range(l_list):
            s = i - win_extlr
            e = i + win_extlr + 1
            if s < 0:
                s = 0
            if e > l_list:
                e = l_list
            # Extract portion and assign value to new list.
            new_list[i] = statistics.mean(in_list[s:e])
    elif method == 2:
        for i in range(l_list):
            s = i - win_extlr
            e = i + win_extlr + 1
            if s < 0:
                s = 0
            if e > l_list:
                e = l_list
            l = e-s
            sc_sum = 0
            for j in range(l):
                sc_sum += in_list[s+j]
            new_list[i] = sc_sum / l
    else:
        assert 0, "invalid method ID given (%i)" %(method)
    return new_list


################################################################################

def get_window_perturb_scores(args, feat_list, feat_win,
                              model_path, device,
                              load_model=True,
                              avg_win_extlr=False,
                              model_hp_dic=False):
    """
    Get perturbation scores list containing score changes when sliding
    worst scoring feature window over original feature list (stride 1).

    feat_list:
        List of feature vectors corresponding to seq, so
        len(feat_list) == len(seq)
    feat_win:
        Best or worst scoring feature list window. 2nd dimension
        has to be equal to feat_list, first dimension length of
        window (<= feat_list length).

    """
    # Checks.
    assert feat_win, "feat_win empty"
    assert feat_list, "feat_list empty"
    len_win = len(feat_win)
    len_feat_list = len(feat_list)
    # Demand window length to be <= feat_list length.
    assert len_win <= len_feat_list, "len_win > len_feat_list (%i > %i)" %(len_win, len_feat_list)
    # Check number of channel features for window and list.
    c_win_feat = len(feat_win[0])
    c_list_feat = len(feat_list[0])
    assert c_win_feat == c_list_feat, "len(feat_win[0]) == len(feat_list[0]) (%i != %i)" %(c_win_feat, c_list_feat)

    # If model object given, set model_path to model.
    model = model_path
    # If load_model set, treat model_path as path to model file and load model.
    if load_model:
        # Define and load model.
        model = define_model(args, device,
                             opt_dic=model_hp_dic)
        model.load_state_dict(torch.load(model_path))

    # Batch size.
    batch_size = args.batch_size
    if model_hp_dic:
        batch_size = model_hp_dic["batch_size"]

    # Number of sliding windows.
    n_sl_win = (len_feat_list - len_win) + 1
    # All new mutated feature lists.
    all_mut_feat = []

    # Create mutated sequences by inserting worst window in sliding window fashion.
    si = 0
    # First add the original sequence / feature list to score.
    all_mut_feat.append(torch.tensor(feat_list, dtype=torch.float))
    for i1 in range(n_sl_win):
        nfl = copy.deepcopy(feat_list)
        for i2,fv in enumerate(feat_win):
            pos = i2 + si
            nfl[pos] = feat_win[i2]
        all_mut_feat.append(torch.tensor(nfl, dtype=torch.float))
        si += 1
    c_all_mut_feat = len(all_mut_feat)
    assert c_all_mut_feat == n_sl_win+1, "c_all_mut_feat != n_sl_win+1"

    # Predict on mutated sequences.
    all_mut_feat_labels = [1]*c_all_mut_feat
    predict_dataset = RNNDataset(all_mut_feat, all_mut_feat_labels)
    predict_loader = DataLoader(dataset=predict_dataset, batch_size=batch_size, collate_fn=pad_collate, pin_memory=True)
    mut_scores = test_scores(predict_loader, model, device)
    c_mut_scores = len(mut_scores)
    assert c_mut_scores == c_all_mut_feat, "c_mut_scores != c_all_mut_feat (%i != %i)" %(c_mut_scores, c_all_mut_feat)

    # Original sequence score.
    orig_sc = mut_scores.pop(0)

    # Calculate mutation scores.
    fl_sc = [0.0]*len_feat_list
    fl_sc_avg_c = [0]*len_feat_list
    for i1,sc in enumerate(mut_scores):
        r1 = i1
        r2 = i1 + len_win
        for i2 in range(r1, r2):
            fl_sc[i2] += sc - orig_sc
            fl_sc_avg_c[i2] += 1

    assert len(fl_sc) == len_feat_list, "len(fl_sc) != len_feat_list (%i != %i)" %(len(fl_sc), len_feat_list)

    # Average score positions.
    for i,sc in enumerate(fl_sc):
        new_sc = sc / fl_sc_avg_c[i]
        fl_sc[i] = new_sc

    if avg_win_extlr:
        # Return average-profiled window mutation scores.
        return list_moving_win_avg_values(fl_sc,
                                          win_extlr=avg_win_extlr,
                                          method=1)
    else:
        # Return window mutation scores.
        return fl_sc


################################################################################

def get_single_nt_perturb_scores(args, seq, feat_list,
                                 model_path, device,
                                 load_model=True,
                                 model_hp_dic=False):
    """
    Get perturbation scores list containing score changes for each
    nucleotide at each sequence position.

    seq:
        Nucleotide sequence string, corresponding to feat_list.
    feat_list:
        List of feature vectors corresponding to seq, so
        len(feat_list) == len(seq)


    """
    # Checks.
    assert len(seq) == len(feat_list), "seq length != feat_list length (%i != %i)" %(len(seq), len(feat_list))

    # If model object given, set model_path to model.
    model = model_path
    # If load_model set, treat model_path as path to model file and load model.
    if load_model:
        # Define and load model.
        model = define_model(args, device,
                             opt_dic=model_hp_dic)
        model.load_state_dict(torch.load(model_path))

    # Perturbations on embedded or one-hot sequences.
    embed = args.embed
    if model_hp_dic:
        embed = model_hp_dic["embed"]
    # Batch size.
    batch_size = args.batch_size
    if model_hp_dic:
        batch_size = model_hp_dic["batch_size"]

    # List to store perturbation scores for plotting.
    perturb_sc_list = []
    for fv in feat_list:
        perturb_sc_list.append([0.0,0.0,0.0,0.0])

    # Alphabet.
    ab = ["A", "C", "G", "U"]

    # Nucleotide encoding (embed or one-hot).
    nt_enc_dic = {'A' : [1,0,0,0],  'C' : [0,1,0,0], 'G' : [0,0,1,0], 'U' : [0,0,0,1]}
    if embed:
        nt_enc_dic = {'A' : [1],  'C' : [2], 'G' : [3], 'U' : [4]}

    """
    # To map mutated site ID (score) to perturb_sc_list position.
    Each list entry with format: [idx1, idx2],
    where idx1 = sequence idx and idx2 = mutated nucleotide index.
    """
    map_sc2idxs = []

    # Create mutated sequences and mappings.
    sl = list(seq)
    new_seqs = []
    for idx1,nt in enumerate(sl):
        for idx2,c in enumerate(ab):
            if c != nt:
                nsl = list(seq)
                nsl[idx1] = c
                new_seqs.append(nsl)
                map_sc2idxs.append([idx1,idx2])

    # Create feature lists from mutated sequences.
    all_mut_feat = []
    # First add the original sequence / feature list to score.
    all_mut_feat.append(torch.tensor(feat_list, dtype=torch.float))
    for new_seq in new_seqs:
        #nfl = list(feat_list)
        nfl = copy.deepcopy(feat_list)
        for idx1,nt in enumerate(new_seq):
            nt_1h = nt_enc_dic[nt]
            for idx2,oh in enumerate(nt_1h):
                nfl[idx1][idx2] = oh
        all_mut_feat.append(torch.tensor(nfl, dtype=torch.float))

    c_all_mut_feat = len(all_mut_feat)
    assert c_all_mut_feat == len(new_seqs)+1, "len(all_mut_feat) != len(new_seqs)+1"

    # Predict on mutated sequences.
    all_mut_feat_labels = [1]*c_all_mut_feat
    predict_dataset = RNNDataset(all_mut_feat, all_mut_feat_labels)
    predict_loader = DataLoader(dataset=predict_dataset, batch_size=batch_size, collate_fn=pad_collate, pin_memory=True)
    mut_scores = test_scores(predict_loader, model, device)
    c_mut_scores = len(mut_scores)
    assert c_mut_scores == c_all_mut_feat, "c_mut_scores != c_all_mut_feat (%i != %i)" %(c_mut_scores, c_all_mut_feat)

    # Original sequence score.
    orig_sc = mut_scores.pop(0)
    # Fill perturb_sc_list.
    for idx,pos in enumerate(map_sc2idxs):
        seq_idx = pos[0]
        nt_idx = pos[1]
        mut_sc = mut_scores[idx]
        perturb_sc_list[seq_idx][nt_idx] = mut_sc - orig_sc
    return perturb_sc_list


################################################################################

def get_sum_list_nt_perturb_scores(perturb_sc_list,
                                   avg_win_extlr=False):
    """
    Take single nucleotide perturbation scores list (each list position
    with 4 scores corresponding to mutated nucleotide), sum up scores,
    and return summed scores list. Optionally, average-profile list before
    returning (avg_win_extlr).
    """
    assert perturb_sc_list, "perturb_sc_list empty"
    sum_sc_list = []
    for scl in perturb_sc_list:
        sum_sc_list.append(sum(scl))
    if avg_win_extlr:
        # Return average-profiled summed scores list.
        return list_moving_win_avg_values(sum_sc_list,
                                          win_extlr=avg_win_extlr,
                                          method=1)
    else:
        # Return summed scores list.
        return sum_sc_list


################################################################################

def get_best_worst_scoring_window(profile_scores_ll, all_features, win_extlr):

    """
    Go through profile scores list of lists (profile_scores_ll), to get
    best and worst scoring window. Only look at full-length windows.
    Return best+worst scoring feat_list block (stored as list, not tensor).

    profile_scores_ll:
        Profile scores list of lists (same order as all_features).
    all_features:
        All features list of lists.
    win_extlr:
        window left and right extension (so full window length ==
        win_extlr*2 + 1).

    """
    assert profile_scores_ll, "profile_scores_ll empty"
    assert all_features, "all_features empty"
    assert win_extlr, "win_extlr False"

    # Full window length.
    full_win_len = win_extlr*2 + 1

    # Find best + worst scoring window.
    worst_psl_idx = 0
    worst_psl_pos = 0
    worst_psl_score = 1000
    best_psl_idx = 0
    best_psl_pos = 0
    best_psl_score = -1000

    for psl_idx,psl in enumerate(profile_scores_ll):
        l_psl = len(psl)
        for wi in range(l_psl):
            block_s = wi - win_extlr
            block_e = wi + win_extlr + 1
            if block_s < 0:
                continue
            if block_e > l_psl:
                break
            psl_sc = psl[wi]
            if psl_sc < worst_psl_score:
                worst_psl_score = psl_sc
                worst_psl_idx = psl_idx
                worst_psl_pos = wi
            if psl_sc > best_psl_score:
                best_psl_score = psl_sc
                best_psl_idx = psl_idx
                best_psl_pos = wi

    # Extract best + worst scoring window.
    block_s = worst_psl_pos - win_extlr
    block_e = worst_psl_pos + win_extlr + 1
    worst_sc_block = all_features[worst_psl_idx][block_s:block_e]
    l_wsb = len(worst_sc_block)
    assert l_wsb == full_win_len, "len(worst_sc_block) != full_win_len (%i != %i)" %(l_wsb, full_win_len)
    block_s = best_psl_pos - win_extlr
    block_e = best_psl_pos + win_extlr + 1
    best_sc_block = all_features[best_psl_idx][block_s:block_e]
    l_wsb = len(best_sc_block)
    assert l_wsb == full_win_len, "len(best_sc_block) != full_win_len (%i != %i)" %(l_wsb, full_win_len)

    return best_sc_block, worst_sc_block


################################################################################

def get_window_predictions(args, model_path, device,
                           all_features, win_extlr,
                           model_hp_dic=False,
                           load_model=True,
                           min_max_norm=False,
                           got_tensors=True):
    """

    Predict windows on given dataset all_features. Returns position-wise
    scores in list for each dataset instance.

    args:
        Model parameters args.
    model_path:
        Path to model or model (depending on set load_model) to be used
        for window predictions.
    all_features:
        Features list.
    win_extlr:
        Extension left and right of center position.
    load_model:
        If True, treat model_path as path to model file, and load model.
        If False, treat model_path as model object (no need to load).
    model_hp_dic:
        Model (hyper)parameter dictionary (to overwrite args).
    min_max_norm:
        If True, min-max normalize scores.
    got_tensors:
        If True, assumes all_features has tensors. If False,
        converts windows to tensors.

    Return list of profile score lists, with list index corresponding
    to all_features index.

    """

    # If model object given, set model_path to model.
    model = model_path
    # If load_model set, treat model_path as path to model file and load model.
    if load_model:
        # Define and load model.
        model = define_model(args, device,
                             opt_dic=model_hp_dic)
        model.load_state_dict(torch.load(model_path))

    # Batch size from args or model hp dic.
    batch_size = args.batch_size
    if model_hp_dic:
        batch_size = model_hp_dic["batch_size"]

    # Create window graphs.
    all_win_feat = []
    seq_len_list = []
    c_all_seq_len = 0

    for feat_list in all_features:
        l_seq = len(feat_list)
        # l_seq == window graphs for sequence.
        seq_len_list.append(l_seq)
        c_all_seq_len += l_seq
        # For each position in feat_list.
        l_feat = len(feat_list)
        for wi in range(l_feat):
            reg_s = wi - win_extlr
            reg_e = wi + win_extlr + 1
            if reg_e > l_feat:
                reg_e = l_feat
            if reg_s < 0:
                reg_s = 0
            if got_tensors:
                all_win_feat.append(feat_list[reg_s:reg_e])
            else:
                all_win_feat.append(torch.tensor(feat_list[reg_s:reg_e], dtype=torch.float))

    # Checks.
    c_all_win_feat = len(all_win_feat)
    assert c_all_win_feat, "no windows extracted (size of window features list == 0)"
    assert c_all_seq_len == c_all_win_feat, "total sequence length != # of windows (%i != %i)" %(c_all_seq_len, c_all_win_feat)

    # Load dataset.
    all_win_feat_labels = [1]*c_all_win_feat
    predict_dataset = RNNDataset(all_win_feat, all_win_feat_labels)
    predict_loader = DataLoader(dataset=predict_dataset, batch_size=batch_size, collate_fn=pad_collate, pin_memory=True)

    # Predict.
    win_scores = test_scores(predict_loader, model, device)

    #win_size = win_extlr*2 + 1
    print("Maximum window score: ", max(win_scores))
    print("Maximum window score: ", min(win_scores))

    # Checks.
    c_win_scores = len(win_scores)
    assert c_win_scores == c_all_seq_len, "# window graph scores != total sequence length (%i != %i)" %(c_win_scores, c_all_seq_len)

    # Create list of profile scores.
    profile_scores_ll = []
    si = 0
    for l in seq_len_list:
        se = si + l
        profile_scores_ll.append(win_scores[si:se])
        si += l
    assert profile_scores_ll, "profile_scores_ll list empty"
    return profile_scores_ll


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
