import argparse
import os
import numpy as np
import time
import json
import torch
import torch.nn as nn
import torchvision.models as torchvision_models
from  downstream_utils import load_backbone, get_datasets, get_datasets_ood, dataset_info, dist_acc
from sklearn.metrics import r2_score
import sys
sys.path.append('/raid/s2265822/qd4vision/')
from supervised.pretrain_utils import accuracy, AverageMeter, ProgressMeter
from tqdm import tqdm
from sklearn.preprocessing import minmax_scale

torchvision_model_names = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))


model_names = ['convnext'] + torchvision_model_names

parser = argparse.ArgumentParser(description='Learning Fully connected layer with Gradient Descent')
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
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 1024), this is the total '
                         'batch size of all GPUs on all nodes when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', default=2e-5, type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

# DDP args
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=4, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--warmup-epochs', default=10, type=int, metavar='N',
                    help='number of warmup epochs')

# test data args
parser.add_argument('--test_dataset', default='VOC2007', type=str)
parser.add_argument('--baseline', action='store_true', help="Use pretrained or QD model")
parser.add_argument('--data_root', default='/raid/s2265822/TestDatasets/', type = str)
parser.add_argument('--test_mode', default='id', type=str, help="Use pretrained or QD model")

# additional configs:
parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')
parser.add_argument('--image_size', default=224, type=int,
                    help='image size')

# QD args
parser.add_argument('--ensemble_size', default=5, type=int, help='Number of members in the ensemble')
parser.add_argument('--few-shot-reg', action='store_true',
                    help='do few shot regression')
parser.add_argument('--shot-size', default=0.0, type=float,
                        help='number of samples per classes.')
parser.add_argument('--moco', default=None, type=str, help="Use MOCO pretrained model or Use supervised pretrained model")
parser.add_argument('--model-type', default='branched', type=str, 
                    help='which model type to use')

def main():
    main_worker()

def main_worker(config=None):
    args = parser.parse_args()
    args.model_type = 'branched'
    global best_acc1

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    models_ensemble = load_backbone(args)
    print(models_ensemble)
        
    
    if args.test_mode == 'id':
        train_loader, val_loader, trainval_loader, test_loader, num_classes = get_datasets(args)
    elif args.test_mode == 'ood':
        train_loader, val_loader, trainval_loader, test_loader, num_classes = get_datasets_ood(args)


    if args.baseline:
        classifier = nn.Linear(2048, dataset_info[args.test_dataset]['num_classes']).cuda(args.gpu)
        classifier.weight.data.normal_(0, 0.01)
        classifier.bias.data.zero_()
    else:
        # args.ensemble_size = args.ensemble_size + 1
        classifier = []
        for k in range(args.ensemble_size):
            clf = nn.Linear(2048, dataset_info[args.test_dataset]['num_classes']).cuda(args.gpu)
            clf.weight.data.normal_(0, 0.01)
            clf.bias.data.zero_()
            classifier.append(clf)
        classifier = nn.ModuleList(classifier)
        print(classifier)

    optimizer = torch.optim.SGD(classifier.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Loss function
    if dataset_info[args.test_dataset]['mode'] in ['regression', 'pose_estimation']:
        criterion = nn.L1Loss()
    elif dataset_info[args.test_dataset]['mode'] == 'classification':
        criterion = nn.CrossEntropyLoss()
        
    results = {}
    best_score = 0.0
    for epoch in range(args.epochs):
        train_acc, train_loss = train(trainval_loader, models_ensemble, args.gpu, 
                                        classifier, optimizer, dataset_info[args.test_dataset]['mode'], 
                                        criterion, epoch, args, train_mode = True)
        lr_scheduler.step()
        if args.baseline:
            test_acc, test_loss = train(test_loader, models_ensemble, args.gpu, 
                                        classifier, optimizer, dataset_info[args.test_dataset]['mode'], 
                                        criterion, epoch, args, train_mode = False)
        else:
            test_acc, test_loss, weights = evaluate_qd(test_loader, models_ensemble, args.gpu, 
                                        classifier, dataset_info[args.test_dataset]['mode'], 
                                        criterion, epoch, args)

        print("Epoch: {}, Train Loss: {}, Train Acc: {}, Val Loss: {}, Val Acc:{}".format(epoch, train_loss, train_acc, test_loss, test_acc))
        epoch_results = {}
        epoch_results["Train acc"] = train_acc
        epoch_results["Train loss"] = train_loss
        epoch_results["Val loss"] = test_loss
        epoch_results["Val acc"] = test_acc.item()

        if test_acc > best_score:
            best_score = test_acc
            if not args.baseline:   
                best_weights = weights

        results[epoch] = epoch_results
    results["Best score"] = best_score
    if not args.baseline:
        results["Best weights"] = best_weights.tolist()

    print(results)
    print("Best score: ", best_score)

    if args.baseline:
        fname = os.path.join("results", "{}-moco".format(args.moco) if args.moco is not None else "supervised", "few_shot" if args.few_shot_reg else "",  "{}_{}_{}_baseline.json".format(args.test_dataset, str(args.shot_size) if args.few_shot_reg else "", args.seed))
    else:
        fname = os.path.join("results", "{}-moco".format(args.moco) if args.moco is not None else "supervised", "few_shot" if args.few_shot_reg else "", "{}_{}_{}.json".format(args.test_dataset, str(args.shot_size) if args.few_shot_reg else "", args.seed))
    
    with open(fname, 'w') as f:
        json.dump(results, f)

def train(loader, model, device, classifier, optimizer, mode, criterion, epoch, args, train_mode):
    model.eval()
    feature_vector = []
    labels_vector = []
    iter = 0
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    ce_losses = AverageMeter('CE Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    prefix_ = "Train: [{}]".format(epoch) if train_mode else "Val: [{}]".format(epoch)
    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time, ce_losses, top1],
        prefix=prefix_)

    end = time.time()   
    model.eval()

    for data in tqdm(loader, desc=f'Training Epoch {epoch}'):
        batch_x, batch_y = data
        batch_x = batch_x.cuda(device).reshape(-1, 3, args.image_size, args.image_size)
        batch_y = batch_y.cuda(device)
        if args.test_dataset in ['leeds_sports_pose', '300w']:
            batch_y = nn.functional.normalize(batch_y, dim=1)

        if args.baseline:
            features = model(batch_x).view(batch_x.shape[0], -1)
            output = classifier(features)
            loss = criterion(output, batch_y)
        else:
            _, feats = model(batch_x, reshape=False)
            output = []
            loss = 0.0
            for k in range(args.ensemble_size):
                output.append(classifier[k](feats[k]))
                loss += criterion(output[k], batch_y)
            loss /= args.ensemble_size

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if args.baseline:
            if mode == 'classification':
                acc1, _ = accuracy(output, batch_y, topk=(1, 5))
            elif mode == 'regression':
                acc1 = r2_score(batch_y.cpu().detach().numpy().reshape(-1), output.cpu().detach().numpy().reshape(-1))
            elif mode == 'pose_estimation':
                acc1 = dist_acc((output.cpu().detach().numpy() - batch_y.cpu().detach().numpy())**2)
        else:
            acc1 = []
            if mode == 'classification':
                for k in range(args.ensemble_size):
                    a1, _ = accuracy(output[k], batch_y, topk=(1, 5))
                    acc1.append(a1.item())
            elif mode == 'regression':
                for k in range(args.ensemble_size):
                    a1 = r2_score(batch_y.cpu().detach().numpy().reshape(-1), output[k].cpu().detach().numpy().reshape(-1))
                    acc1.append(a1)
            elif mode == 'pose_estimation':
                for k in range(args.ensemble_size):
                    a1 = dist_acc((output[k].cpu().detach().numpy() - batch_y.cpu().detach().numpy())**2)
                    acc1.append(a1)
            acc1 = np.mean(acc1)
            

        ce_losses.update(loss.item(), batch_x.size(0))
        top1.update(acc1, batch_x.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        torch.cuda.empty_cache()
        if iter % 10 == 0:
            progress.display(iter)
    
    return top1.avg, ce_losses.avg
        
def evaluate_qd(loader, model, device, classifier, mode, criterion, epoch, args):
    model.eval()

    # Collect all outputs
    outputs = [[] for _ in range(args.ensemble_size)]
    labels = []
    loss = 0.0
    for data in tqdm(loader, desc=f'Validation Epoch {epoch}'):
        batch_x, batch_y = data
        batch_x = batch_x.cuda(device).reshape(-1, 3, args.image_size, args.image_size)
        batch_y = batch_y.cuda(device)
        if args.test_dataset in ['leeds_sports_pose', '300w']:
            batch_y = nn.functional.normalize(batch_y, dim=1)
        output, _ =  model(batch_x, reshape = False)
        loss_batch = 0.0
        for k in range(args.ensemble_size):
            pred = classifier[k](output[k])
            outputs[k].append(pred.cpu().detach().numpy())
            loss_batch += criterion(pred, batch_y).item()
        labels.append(batch_y.cpu().detach().numpy())
        loss += loss_batch/args.ensemble_size

    outputs = [np.concatenate(f, axis=0) for f in outputs]
    labels = np.concatenate(labels, axis=0)
    weights = find_lstsq_weights(outputs, labels, dataset_info[args.test_dataset]['num_classes'], mode)
    scaled_weights = minmax_scale(weights)
    outputs = np.array(outputs)
    outputs = np.swapaxes(outputs, 0, 2)
    weighted_preds = np.matmul(outputs, scaled_weights)/sum(scaled_weights)
    weighted_preds = np.transpose(weighted_preds.squeeze(2))
    if mode == 'classification':
        acc1 = (weighted_preds.argmax(1) == labels).astype(np.float32).mean() * 100.
    elif mode == 'regression':
        acc1 = r2_score(labels.reshape(-1), weighted_preds.reshape(-1))
    elif mode == 'pose_estimation':
        acc1 = dist_acc((weighted_preds- labels)**2)
    loss /= len(loader)
    return acc1, loss, weights
    

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
    lstsq_weights = np.linalg.lstsq(val_preds, y_val_)[0]
    lstsq_weights = np.expand_dims(lstsq_weights, 1)
    return lstsq_weights


if __name__ == '__main__':
    main()
