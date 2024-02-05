import argparse
import builtins
from codecs import namereplace_errors
import math
import os
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.models as torchvision_models
from  downstream_utils import load_backbone, dataset_info, get_datasets, get_datasets_ood, dist_acc
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.linear_model import Ridge as LinReg
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score
from sklearn.preprocessing import minmax_scale
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

torchvision_model_names = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))


model_names = ['convnext'] + torchvision_model_names

parser = argparse.ArgumentParser(description='Linear probing of QD ensemble')
parser.add_argument('-data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')

# Training args
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 1024), this is the total '
                         'batch size of all GPUs on all nodes when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=4, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# Test data args
parser.add_argument('--baseline', action='store_true', help="Use pretrained or QD model")
parser.add_argument('--test_mode', default='id', type=str, help="Use ID evaluation or OOD evaluation")
parser.add_argument('--test_dataset', default='CIFAR10', type=str)
parser.add_argument('--data_root', default='', type = str)

# QD args
parser.add_argument('--ensemble_size', default=5, type=int, help='Number of members in the ensemble')
parser.add_argument('-sr', '--sample-rate', default=100, type=int,
                        metavar='N',
                        help='sample rate of training dataset (default: 100)')

# additional configs:
parser.add_argument('--pretrained', default='', type=str,
                    help='path to pretrained checkpoint')
parser.add_argument('--image_size', default=224, type=int,
                    help='image size')
parser.add_argument('-sc', '--num-samples-per-classes', default=None, type=int,
                        help='number of samples per classes.')
parser.add_argument('--moco', default=None, type=str, help="Use MOCO pretrained model or Use supervised pretrained model")
parser.add_argument('--model-type', default='branched', type=str, 
                    help='which model type to use')
parser.add_argument('--few-shot-reg', action='store_true',
                    help='do few shot regression')

def main():
    main_worker()

def main_worker(config=None):
    args = parser.parse_args()
    global best_acc1

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    models_ensemble = load_backbone(args)
    models_ensemble.eval()
        
    if args.test_mode == 'id':
        train_loader, val_loader, trainval_loader, test_loader, num_classes = get_datasets(args)
    elif args.test_mode == 'ood':
        train_loader, val_loader, trainval_loader, test_loader, num_classes = get_datasets_ood(args)

    X_train_feature, y_train, X_val_feature, y_val = get_features(
        train_loader, val_loader, models_ensemble, device = args.gpu, baseline = args.baseline, 
        num_classes = num_classes, mode = dataset_info[args.test_dataset]['mode'], model_type = args.model_type, split='train')

    X_trainval_feature, y_trainval, X_test_feature, y_test = get_features(
            trainval_loader, test_loader, models_ensemble, device = args.gpu, baseline = args.baseline, 
            num_classes = num_classes, mode = dataset_info[args.test_dataset]['mode'], model_type = args.model_type, split='test')

    print("Checking shapes of features:")
    print("X_train_feature", X_train_feature.shape)
    print("y_train", y_train.shape)
    print("X_val_feature", X_val_feature.shape)
    print("y_val", y_val.shape)
    print("X_trainval_feature", X_trainval_feature.shape)
    print("y_trainval", y_trainval.shape)
    print("X_test_feature", X_test_feature.shape)
    print("y_test", y_test.shape)

    if args.baseline:
        if dataset_info[args.test_dataset]['mode'] == 'regression':
            clf = LinearRegression(2048, num_classes, 'r2')
        elif dataset_info[args.test_dataset]['mode'] == 'pose_estimation':
            clf = LinearRegression(2048, num_classes, 'pca')
        else:
            clf = LogisticRegression(2048, num_classes, dataset_info[args.test_dataset]['metric'])
    else:
        if dataset_info[args.test_dataset]['mode'] == 'regression':
            clf = [LinearRegression(2048, num_classes, 'r2') for _ in range(args.ensemble_size + 1)]
        elif dataset_info[args.test_dataset]['mode'] == 'pose_estimation':
            clf = [LinearRegression(2048, num_classes, 'pca') for _ in range(args.ensemble_size + 1)]
        else:
            clf = [LogisticRegression(2048, num_classes, dataset_info[args.test_dataset]['metric']) for _ in range(args.ensemble_size + 1)]
        
    if dataset_info[args.test_dataset]['mode'] in ['regression', 'pose_estimation']:
        wd_range = torch.logspace(-2, 5, 100)
    else:
        wd_range = torch.logspace(-6, 5, 45)

    if not args.baseline:
        best_params = {}
        best_score = np.zeros(args.ensemble_size + 1)
        all_results = {}
        results_file = open(os.path.join("results", "{}-moco".format(args.moco) if args.moco is not None else "supervised", args.test_dataset + ".json"), 'w')
        print("Saving results in ", results_file)

        for k in range(0, args.ensemble_size+1):
            for wd in tqdm(wd_range, desc='Selecting best ridge hyperparameters for classifier ' + str(k)):
                C = set_params(clf[k], wd, dataset_info[args.test_dataset]['mode'])
                val_acc = clf[k].fit_regression(X_train_feature[k], y_train, X_val_feature[k], y_val)
                if val_acc > best_score[k]:
                    best_params[str(k)] = C
                    best_score[k] = val_acc
                
            print("Best hyper parameters.", best_params[str(k)], best_score)

        all_results['HP result'] = best_params
        all_results["best val accuracies"] = best_score.tolist()
            
        print("----------------- Linear combination search ---------------")
        # Using best regulariser find linear combination weights using the val set
        val_preds = []
        for k in range(0, args.ensemble_size+1):
            _ = set_params(clf[k], torch.tensor(best_params[str(k)]), dataset_info[args.test_dataset]['mode'], set_mode=True)
            val_pred = clf[k].get_pred(X_val_feature[k])
            val_preds.append(val_pred)
        val_preds = np.array(val_preds)    
        lstsq_weights = find_lstsq_weights(val_preds, y_val, num_classes, mode = dataset_info[args.test_dataset]['mode'])
        lstsq_weights = minmax_scale(lstsq_weights)
        all_results['weights'] = lstsq_weights.tolist()
        print("Linear combination search results on val set:", lstsq_weights)

        # From best linear combination and best regulariser, fit classifiers on train val set
        test_preds = []
        test_accuracies = []
        for k in range(0, args.ensemble_size+1):
            _ = set_params(clf[k], torch.tensor(best_params[str(k)]), dataset_info[args.test_dataset]['mode'], set_mode=True)
            test_acc = clf[k].fit_regression(X_trainval_feature[k], y_trainval, X_test_feature[k], y_test)
            test_pred = clf[k].get_pred(X_test_feature[k])
            test_preds.append(test_pred)
            test_accuracies.append(test_acc)
        
        print("Test accuracies of all ensemble members", test_accuracies)
        all_results['test accuracies'] = test_accuracies

        test_preds = np.array(test_preds)     
        test_preds = np.swapaxes(test_preds, 0, 2)
        weighted_preds = np.matmul(test_preds, lstsq_weights).squeeze(2)
        weighted_preds = np.transpose(weighted_preds)/sum(lstsq_weights)
        test_acc = clf[0].get_accuracy(weighted_preds, y_test, dataset_info[args.test_dataset]['mode'])

        all_results['Weighted test acc'] = test_acc
        print("Test Accuracy", test_acc)
        json.dump(all_results, results_file)

    else:
        # For baseline
        best_score = 0.0
        best_params = {}
        results = {}
        results_file = open(os.path.join("results", "{}-moco".format(args.moco) if args.moco is not None else "supervised", args.test_dataset + "_baseline.json"), 'w')
        for wd in tqdm(wd_range, desc='Selecting best hyperparameters for classifier'):
            C = set_params(clf, wd, dataset_info[args.test_dataset]['mode'])
            val_acc = clf.fit_regression(X_train_feature, y_train, X_val_feature, y_val)
            if val_acc > best_score:
                best_score = val_acc
                best_params["C"] = C
            
        print("Best hyper parameter for baseline", best_params["C"], best_score)
        C = set_params(clf, torch.tensor(best_params["C"]), dataset_info[args.test_dataset]['mode'], set_mode=True)
        test_acc = clf.fit_regression(X_trainval_feature, y_trainval, X_test_feature, y_test)
        results['acc'] = test_acc
        results['best param'] = best_params["C"]
        json.dump(results, results_file)

def set_params(clf, wd, mode, set_mode = False):
    # set mode is False for hyper parameter search only
    if mode in ['regression', 'pose_estimation']:
        C = wd.item()
        clf.set_params({'alpha': C})
    else:
        if set_mode:
            C = wd.item()
        else:
            C = 1. / wd.item()
        clf.set_params({'C': C})
    return C


def find_lstsq_weights(val_preds, y_val, num_classes, mode, cls = None):
    val_preds = np.array(val_preds).reshape(len(val_preds), -1)
    val_preds = np.transpose(val_preds)
    if mode == 'classification':
        if cls is None:
            y_val_ = np.eye(num_classes)[y_val].reshape(-1)
        else:
            y_val_ = y_val.reshape(-1)
    else:
        y_val_ = y_val.reshape(-1)
    lstsq_weights = LinReg(fit_intercept=False).fit(val_preds, y_val_).coef_
    lstsq_weights = np.expand_dims(lstsq_weights, 1)
    return lstsq_weights


# Testing classes and functions
def get_features(train_loader, test_loader, model, device, baseline, num_classes, mode, model_type, split='train'):
    X_train_feature, y_train = inference(train_loader, model, device, baseline, num_classes, mode, model_type, 'train' if split == 'train' else 'trainval')
    X_test_feature, y_test = inference(test_loader, model, device, baseline, num_classes, mode, model_type, 'test' if split == 'test' else 'val')
    return X_train_feature, y_train, X_test_feature, y_test

def inference(loader, model, device, baseline, num_classes, mode, model_type, split):
    model.eval()
    feature_vector = []
    labels_vector = []
    iter = 0
    for data in tqdm(loader, desc=f'Computing features for {split} set'):
        batch_x, batch_y = data
        batch_x = batch_x.cuda(device)
        if mode == 'regression':
            batch_y = F.normalize(batch_y, dim=1)
        labels_vector.extend(np.array(batch_y))

        if baseline:
            features = model(batch_x).view(batch_x.shape[0], -1)
            feature_vector.extend(features.detach().cpu().numpy())
        else:
            if model_type == 'branched':
                _, features = model(batch_x, reshape = False)
            elif model_type == 'adapters':
                features = []
                for i in range(model.N):
                    model.select_adapter_idx = [i]
                    _, feats = model(batch_x, select = False, reshape = False, train_mode=False)
                    features.append(feats[0])

            features = torch.cat(features).reshape(model.N, -1, 2048)
            feature_vector.append(features)
        iter += 1

    if not baseline:
        feature_vector = torch.cat(feature_vector, dim = 1).cpu().detach().numpy()

    feature_vector = np.array(feature_vector)
    if mode == 'classification':
        labels_vector = np.array(labels_vector, dtype=int)
    else:
        labels_vector = np.array(labels_vector, dtype=float)

    return feature_vector, labels_vector

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes, metric):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.metric = metric
        self.clf = LogReg(solver='lbfgs', multi_class='multinomial', warm_start=True)

        print('Logistic regression:')
        print(f'\t solver = LBFGS')
        print(f"\t classes = {self.num_classes}")
        print(f"\t metric = {self.metric}")

    def set_params(self, d):
        self.clf.set_params(**d)

    @ignore_warnings(category=ConvergenceWarning)
    def fit_regression(self, X_train, y_train, X_test, y_test):
        if self.metric == 'accuracy':
            self.clf.fit(X_train, y_train)
            test_acc = 100. * self.clf.score(X_test, y_test)
            return test_acc

        elif self.metric == 'mean per-class accuracy':
            self.clf.fit(X_train, y_train)
            pred_test = self.clf.predict(X_test)

            #Get the confusion matrix
            cm = confusion_matrix(y_test, pred_test)
            cm = cm.diagonal() / cm.sum(axis=1) 
            test_score = 100. * cm.mean()

            return test_score
        
        else:
            raise NotImplementedError

    def get_pred(self, X):
        return self.clf.predict_log_proba(X)

    def get_accuracy(self, y_pred, y_true, mode=None):
        if self.metric == 'accuracy':
            y_pred = y_pred.argmax(1)
            print("acc", (y_pred == y_true).astype(np.int).sum() / len(y_true))
            return accuracy_score(y_true, y_pred) * 100.

        elif self.metric == 'mean per-class accuracy':
            cm = confusion_matrix(y_true, y_pred.argmax(1))
            cm = cm.diagonal() / cm.sum(axis=1) 
            return 100. * cm.mean()
        
        else:
            raise NotImplementedError

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim, metric):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.metric = metric
        self.clf = LinReg(solver='auto')

        print('Linear regression:')
        print(f'\t solver = AUTO')
        print(f"\t classes = {self.output_dim}")
        print(f"\t metric = {self.metric}")

    def set_params(self, d):
        self.clf.set_params(**d)

    @ignore_warnings(category=ConvergenceWarning)
    def fit_regression(self, X_train, y_train, X_test, y_test):
        if self.metric == 'r2':
            self.clf.fit(X_train, y_train)
            test_acc = 100. * self.clf.score(X_test, y_test)
            return test_acc

        elif self.metric == 'pca':
            self.clf.fit(X_train, y_train)
            pred_test = self.clf.predict(X_test)
            test_score = dist_acc((pred_test - y_test)**2)
            return test_score

        elif self.metric == 'degree_loss':
            self.clf.fit(X_train, y_train)
            pred_test = self.clf.predict(X_test)
            return -(sum(abs(pred_test - y_test))) 
    
    def get_pred(self, X):
        return self.clf.predict(X)

    def get_accuracy(self, y_pred, y_true, mode):
        if mode == 'regression':
            return r2_score(y_true, y_pred) * 100.
        else:
            return dist_acc((y_pred - y_true)**2)

if __name__ == '__main__':
    main()
