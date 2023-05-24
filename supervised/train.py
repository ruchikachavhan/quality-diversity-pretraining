import argparse
import code
import math
import os
import time
import shutil
import warnings
import random
import copy
import builtins
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as torchvision_models
import numpy as np
import torchvision
import models
import datasets
from pretrain_utils import train, evaluate, save_checkpoint, adjust_coeff, adjust_learning_rate, get_pairwise_rowdiff, EarlyStopping, ExponentialMovingAverage
import json
import wandb

# python train.py --multiprocessing-distributed --rank 0 --world-size 1 --dist-url "tcp://localhost:10001" --train_data imagenet1k --data /raid/imagenet1k/ --num_augs 6 --batch-size 1024 
# --lr 0.5 --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --lr-warmup-decay 0.01 --wd 2e-5 --mixup --cutmix --model-ema
# Models and arguments

torchvision_model_names = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

model_names = ['convnext'] + torchvision_model_names

parser = argparse.ArgumentParser(description='Quality Diversity for Vision: Pretraining')

# Path args
parser.add_argument('--data', default='/raid/s2265822/image-net100/',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('--output_dir', default = '/raid/s2265822/saved_models/qd4vision/', type=str)

# DDP args
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
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')

# Training parameters
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--model-type', default='branched', type=str, 
                    help='which model type to use')
parser.add_argument('--ensemble_size', default=5, type=int, help='Number of members in the ensemble')
parser.add_argument('--num_augs', default=6, type=int, help='Number of encoders')
parser.add_argument('--coeff', default=0.2, type=float, help='')
parser.add_argument('--kl_coeff', default=0.1, type=float, help='')
parser.add_argument('--baseline', action='store_true',
                    help='Use baseline (one backbone) models if true')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 4096), this is the total '
                         'batch size of all GPUs on all nodes when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument("--quality-warmup", type=int, default=20, 
                help="the number of iterations for which only supervised training will be done",)

# Learning rates, scheduler, and EMA
parser.add_argument('--lr', '--learning-rate', default=1.0, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=8e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-6)',
                    dest='weight_decay')
parser.add_argument('--lr-warmup-epochs', default=10, type=int, metavar='N',
                    help='number of warmup epochs')
parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
parser.add_argument('--train_data', default='imagenet',
                    help='path to dataset')
parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
parser.add_argument("--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)")
parser.add_argument("--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters")
parser.add_argument("--model-ema-steps", type=int, default=32, help="the number of iterations that controls how often to update the EMA model (default: 32)",)
parser.add_argument("--model-ema-decay", type=float, default=0.99998, help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",)
parser.add_argument('--no-scheduler', action='store_false',
                    help='DO not use scheduler if false')

def main():
    args = parser.parse_args()
    print(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = 4
    # torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function

        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not first GPU on each node
    if args.multiprocessing_distributed and (args.gpu != 0 or args.rank != 0):
        def print_pass(*args):
            pass
            builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed: 
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()

    ############################### Model Instantiation #########################
    # Number of classes according to dataset
    if args.train_data == 'imagenet100':
        args.num_classes = 100
    elif args.train_data == 'imagenet1k':
        args.num_classes = 1000

    if args.model_type == 'branched':
        if args.arch == 'resnet50':
            models_ensemble = models.BranchedResNet(N = args.ensemble_size, num_classes = args.num_classes, arch = args.arch)
        elif args.arch == 'convnext':
            models_ensemble = models.DiverseConvNext(N = args.ensemble_size, num_classes = args.num_classes)
    else:
        raise NotImplementedError("Only branched models are supported.")

    ############################### Optimizers and schedulers ###########################
    if args.opt.startswith("sgd"):
        optimizer = torch.optim.SGD(
            models_ensemble.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
    elif args.opt == "rmsprop":
        optimizer = torch.optim.RMSprop(
            models_ensemble.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif args.opt == "adamw":
        optimizer = torch.optim.AdamW(models_ensemble.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    scaler = torch.cuda.amp.GradScaler()

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == 'cycliclr':
         main_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr = 0.01, max_lr = args.lr
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs], verbose = True
        )
    else:
        lr_scheduler = main_lr_scheduler

    ################################ DDP ##############################
     # Run DDP only for ensemble model
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # apply SyncBN
        # local_rank = int(os.environ['LOCAL_RANK'])
        models_ensemble = torch.nn.SyncBatchNorm.convert_sync_batchnorm(models_ensemble)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            models_ensemble.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            find_unused = True 
            models_ensemble = torch.nn.parallel.DistributedDataParallel(models_ensemble, device_ids=[args.gpu], find_unused_parameters=find_unused)
        else:
            models_ensemble.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            models_ensemble = torch.nn.parallel.DistributedDataParallel(models_ensemble, find_unused_parameters=False)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        models_ensemble = models_ensemble.cuda(args.gpu)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather/rank implementation in this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    # print(models_ensemble) # print model after SyncBatchNorm

   ############################# Exponenetial moving average #############################
    if args.model_ema:
        if args.distributed:
            model_without_ddp = models_ensemble.module
        else:
            model_without_ddp = copy.deepcopy(models_ensemble)
        # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = ExponentialMovingAverage(model_without_ddp, device=torch.device(args.gpu), decay=1.0 - alpha)
    else:
        model_ema = None

    # optionally resume from a checkpoint, only for ensemble model
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch'] 
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('module.'):
                    # remove prefix
                    state_dict[k[len("module."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
            
            models_ensemble.load_state_dict(state_dict)            
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            if args.model_ema:
                model_ema.load_state_dict(checkpoint["model_ema"])

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        args.start_epoch = 0
        # infer learning rate before changing batch size
   
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).cuda(args.gpu)

    ############################# Data Loading code #############################
    if args.num_augs == 2:
        augs_list = [datasets.dorsal_augmentations, datasets.ventral_augmentations, datasets.base_augs]
    else:
        assert args.num_augs == 5
        augs_list = datasets.combinations_default
        augs_list.append(datasets.base_augs) #Last transformation corresponds to no augmentation

    if args.train_data in ['imagenet100', 'imagenet1k']:
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
    else:
        raise NotImplementedError("Only imagenet100 and imagenet1k are supported.")

    train_dataset = torchvision.datasets.ImageFolder(traindir,
                datasets.CropsTransform(augs_list))
    val_dataset = torchvision.datasets.ImageFolder(valdir,
                datasets.CropsTransform(augs_list))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    print("Training and validation data path", traindir, valdir)
    print('Training and validation data size:', len(train_dataset), len(val_dataset))

    fname = '%s_%s_%s-supervised%d_checkpoint_%s_%04d.pth.tar'%(args.arch, args.train_data, args.model_type, args.num_augs, args.ensemble_size, args.epochs)
    coeff = args.coeff

    quality = []
    diversity = []

    with wandb.init(project="QD ImageNet pretraining", name=f"experiment_{'QD4v'}",  config={
        "learning_rate": args.lr,
        "architecture": args.arch,
        "dataset": "ImageNet",
        "epochs": args.epochs,
      }):
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
                # sup_train_sampler.set_epoch(epoch)
            
            train_avg_sim, train_acc1, train_acc5, train_loss = train(train_loader, models_ensemble, criterion, 
                                optimizer, scaler, model_ema, epoch, args, coeff, args.kl_coeff)
            if not args.no_scheduler:
                lr_scheduler.step()
            test_acc1, test_acc5, val_loss, val_avg_sim = evaluate(val_loader, models_ensemble, criterion, epoch, args)

            print("-------------------------Epoch--------------------------", epoch)
            print("Training Accuracies of all backbones", train_acc1)
            print("Test Accuracies of all backbones", test_acc1)
            print("Train sim", train_avg_sim)
            print("Val sim", val_avg_sim, coeff)

            diff = get_pairwise_rowdiff(train_avg_sim).item()
            quality.append(test_acc1.mean().item())
            diversity.append(diff)
            dict = {"Q": quality, "D": diversity}
            with open(args.train_data + args.arch + "_" +  '_log_qd.json', "w") as f:
                json.dump(dict, f)
            
            np.save('train' + "_" + args.train_data+ "_similarity_matrix_"+ str(args.num_augs)+ "_KL_%s_augs.npy"%args.model_type, train_avg_sim.detach().cpu().numpy())
            np.save("val" + "_" + args.train_data+ "_similarity_matrix_"+ str(args.num_augs)+ "_KL_%s_augs.npy"%args.model_type, val_avg_sim.detach().cpu().numpy())

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank == 0): # only the first GPU saves checkpoint
                
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank == 0): # only the first GPU saves checkpoint
                    print("Saving checkpoint in", os.path.join(args.output_dir, fname))
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': models_ensemble.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        'scaler': scaler.state_dict(),
                        'model_ema': model_ema.state_dict() if args.model_ema else None,
                    }, is_best=False, filename=os.path.join(args.output_dir, fname))


if __name__ == '__main__':
    main()