import os
import torch
import numpy as np
import torch.nn as nn
import tllib.vision.datasets as datasets
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.datasets import ImageFolder
import sys
sys.path.append('/raid/s2265822/qd4vision/')
from supervised import models
from test_datasets import CelebA, FacesInTheWild300W, LeedsSportsPose, AnimalPose, Causal3DIdent, ALOI, MPII
from r2score import r2_score
from ood_datasets import domain_net_datasets, CIFAR_STL_dataset, breeds_dataset

# dataset_dict = ['ImageList', 'Office31', 'OfficeHome', "VisDA2017", "OfficeCaltech", "DomainNet", "ImageNetR",
#            "ImageNetSketch", "Aircraft", "cub200", "StanfordCars", "StanfordDogs", "COCO70", "OxfordIIITPets", "PACS",
#            "DTD", "OxfordFlowers102", "PatchCamelyon", "Retinopathy", "EuroSAT", "Resisc45", "Food101", "SUN397",
#            "Caltech101", "CIFAR10", "CIFAR100"]

generator = torch.Generator()
generator.manual_seed(0)

dataset_info = {
    'DTD': {
        'class': None, 'dir': 'dtd/', 'num_classes': 47,
        'splits': ['train', 'validation', 'test'], 'split_size': 0.8,
        'mode': 'classification', 'metric': 'accuracy'
    },
    'Aircraft': {
        'class': None, 'dir': 'Aircraft', 'num_classes': 102,
        'splits': ['train', 'val', 'test'], 'split_size': 0.8,
        'mode': 'classification', 'metric': 'mean per-class accuracy'
    },
    'StanfordCars': {
        'class': None, 'dir': 'Cars/', 'num_classes': 196,
        'splits': ['train', 'val', 'test'], 'split_size': 0.8,
        'mode': 'classification', 'metric': 'accuracy'
    },
    'CIFAR10': {
        'class': datasets.CIFAR10, 'dir': 'CIFAR10', 'num_classes': 10,
        'splits': ['train', 'val', 'test'], 'split_size': 0.7,
        'mode': 'classification', 'metric': 'accuracy'
    },
    'CIFAR100': {
        'class': datasets.CIFAR100, 'dir': 'CIFAR100_new/CIFAR100/', 'num_classes': 100,
        'splits': ['train', 'val', 'test'], 'split_size': 0.7,
        'mode': 'classification', 'metric': 'accuracy'
    },
    'OxfordFlowers102': {
        'class': None, 'dir': 'flowers/', 'num_classes': 102,
        'splits': ['train', 'validation', 'test'], 'split_size': 0.5,
        'mode': 'classification', 'metric': 'mean per-class accuracy'
    }, 
    'Caltech101': {
        'class': None, 'dir': 'Caltech101', 'num_classes': 101,
        'splits': ['train', 'val', 'test'], 'split_size': 0.7,
        'mode': 'classification', 'metric': 'mean per-class accuracy'
    },
    'celeba': {
        'class': CelebA, 'dir': 'CelebA', 'num_classes': 10,
        'splits': ['train', 'valid', 'test'], 'split_size': 0.5,
        'target_type': 'landmarks',
        'mode': 'regression', 'metric': 'r2'
    },
    '300w': {
        'class': FacesInTheWild300W, 'dir': '300W', 'num_classes': 136,
        'splits': ['train', 'valid', 'test'], 'split_size': 0.5,
        'mode': 'regression', 'metric': 'r2'
    },
    'animal_pose':{
        'class': AnimalPose, 'dir': 'animal_pose/animalpose_keypoint_new/', 'num_classes': 40,
        'splits': [], 'split_size': 0.6,
        'mode': 'pose_estimation', 'metric': 'pca'
    },
    'causal3d':{
        'class': Causal3DIdent, 'dir': 'Causal3d', 'num_classes': 10,
        'splits': ['train', 'test'], 'split_size': 0.6,
        'mode': 'regression', 'metric': 'r2'
    },
    'aloi':{
        'class': ALOI, 'dir': 'ALOI/png4/', 'num_classes': 24,
        'splits': [], 'split_size': 0.6,
        'mode': 'classification', 'metric': 'accuracy'
    },
    'mpii':{
        'class': MPII, 'dir': 'mpii', 'num_classes': 32,
        'splits': [], 'split_size': 0.6,
        'mode': 'pose_estimation', 'metric': 'pca'
    },
    'leeds_sports_pose': {
        'class': LeedsSportsPose, 'dir': 'LeedsSportsPose', 'num_classes': 28,
        'splits': ['train', 'test'], 'split_size': 0.8,
        'mode': 'regression', 'metric': 'r2'
    },
    'domainnet': {
        'class': None, 'dir': 'domainnet', 'num_classes': 40,
        'splits': [], 'split_size': 0.8,
        'mode': 'classification', 'metric': 'accuracy'
    },
    'cifarstl': {
        'class': None, 'dir': 'domainnet', 'num_classes': 10,
        'splits': [], 'split_size': 0.8,
        'mode': 'classification', 'metric': 'accuracy'
    },
    'imagenet-a': {
        'class': None, 'dir': 'domainnet', 'num_classes': 200,
        'splits': [], 'split_size': 0.8,
        'mode': 'classification', 'metric': 'accuracy'
    },
    'imagenet-r': {
        'class': None, 'dir': 'domainnet', 'num_classes': 200,
        'splits': [], 'split_size': 0.8,
        'mode': 'classification', 'metric': 'accuracy'
    },
    'imagenet-sketch': {
        'class': None, 'dir': 'domainnet', 'num_classes': 1000,
        'splits': [], 'split_size': 0.8,
        'mode': 'classification', 'metric': 'accuracy'
    },
    'living17': {
        'class': None, 'dir': 'domainnet', 'num_classes': 17,
        'splits': [], 'split_size': 0.8,
        'mode': 'classification', 'metric': 'accuracy'
    },
    'entity30': {
        'class': None, 'dir': 'domainnet', 'num_classes': 30,
        'splits': [], 'split_size': 0.8,
        'mode': 'classification', 'metric': 'accuracy'
    }
}

def get_tllib_dataset(dataset_name, root, train_transform, val_transform, sample_rate=100, num_samples_per_classes=None, test_transform=None):
    """
    When sample_rate < 100,  e.g. sample_rate = 50, use 50% data to train the model.
    Otherwise,
        if num_samples_per_classes is not None, e.g. 5, then sample 5 images for each class, and use them to train the model;
        otherwise, keep all the data.
    """
    dataset = datasets.__dict__[dataset_name]

    if sample_rate < 100:
        train_dataset = dataset(root=root, split='train', sample_rate=sample_rate, download=True, transform=train_transform)
        test_dataset = dataset(root=root, split='test', sample_rate=100, download=True, transform=val_transform)
        num_classes = train_dataset.num_classes
    else:
        train_dataset = dataset(root=root, split='train', download=False, transform=train_transform)
        num_classes = train_dataset.num_classes
        if dataset_name in ['DTD', 'OxfordFlowers102']:
            # Val split is available for these datasets
            val_dataset = dataset(root=root, split='validation', download=False, transform=val_transform)
        else:
            train_size = int(0.8*len(train_dataset))
            train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, len(train_dataset) - train_size],
                                                generator = generator)

        if test_transform is None:
            test_dataset = dataset(root=root, split='test', download=False, transform=val_transform)
        else:
            test_dataset = dataset(root=root, split='test', download=False, transform=test_transform)
        

    return train_dataset, val_dataset, test_dataset, num_classes


def split_train_val_test(dataset_name, data_root, train_transform):
    dataset = dataset_info[dataset_name]['class'](os.path.join(data_root, dataset_info[dataset_name]['dir'])
                                        ,transform = train_transform)
    train_size = int(len(dataset)* dataset_info[dataset_name]['split_size'])
    val_size = (len(dataset) - train_size)//2
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], 
                                                                generator = generator) 
    return train_dataset, val_dataset, test_dataset 

def split_train_val(dataset_name, dataset):
    train_size = int(len(dataset)* dataset_info[dataset_name]['split_size'])
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], 
                                                                generator = generator) 
    return train_dataset, val_dataset

def get_datasets(args):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    base_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize,
        ])
        
    if dataset_info[args.test_dataset]['mode'] in ['regression', 'pose_estimation'] or args.test_dataset == 'aloi':
        num_classes = dataset_info[args.test_dataset]['num_classes']
        # if no splilt is given, then divide training data into train and val
        if len(dataset_info[args.test_dataset]['splits']) == 0:
            train_dataset, val_dataset, test_dataset = split_train_val_test(args.test_dataset, 
                                dataset_info[args.test_dataset]['class'], args.data_root, base_transform)
        else: 
            # If split is given, use the split to make dataloaders
            train_dataset = dataset_info[args.test_dataset]['class'](os.path.join(args.data_root, 
                                                dataset_info[args.test_dataset]['dir']), 
                                                split = 'train', transform = base_transform)

            if len(dataset_info[args.test_dataset]['splits']) == 3:
                val_dataset = dataset_info[args.test_dataset]['class'](os.path.join(args.data_root, 
                                                dataset_info[args.test_dataset]['dir']), 
                                                split = dataset_info[args.test_dataset]['splits'][1], 
                                                transform = base_transform)

                test_dataset = dataset_info[args.test_dataset]['class'](os.path.join(args.data_root, 
                                                dataset_info[args.test_dataset]['dir']), 
                                                split = dataset_info[args.test_dataset]['splits'][2], 
                                                transform = base_transform)

            elif(len(dataset_info[args.test_dataset]['splits'])) == 2: 
                # train dataset is split into train and val
                train_dataset, val_dataset = split_train_val(args.test_dataset, train_dataset)
                test_dataset = dataset_info[args.test_dataset]['class'](os.path.join(args.data_root, 
                                                dataset_info[args.test_dataset]['dir']), 
                                                split = 'test', transform = base_transform)
        if args.few_shot_reg:
            dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset, test_dataset])
            print("Loading few shot regression %s dataset" % args.test_dataset)
            few_shot_size = int(args.shot_size * len(dataset))
            val_size = int(0.2*(len(dataset) - few_shot_size))
            test_size = len(dataset) - few_shot_size - val_size
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [few_shot_size, val_size, test_size])

    else:
        # Get all classification datasets from TLLIB
        train_dataset, val_dataset, test_dataset, num_classes = get_tllib_dataset(args.test_dataset, 
                                                                    os.path.join(args.data_root, dataset_info[args.test_dataset]['dir']),
                                                                    base_transform,
                                                                    base_transform, args.sample_rate,
                                                                    args.num_samples_per_classes)
     
    print("Datasets loaded")
    print("Train size: ", len(train_dataset), "Val size: ", len(val_dataset), "Test size: ", len(test_dataset))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    if args.few_shot_reg:
        trainval_dataset = train_dataset
    else:
        trainval_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
    
    trainval_loader = DataLoader(trainval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    return train_loader, val_loader, trainval_loader, test_loader, num_classes


def get_datasets_ood(args):
    base_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
        ])
    if args.test_dataset == 'domainnet':
        source_train, source_test, target, num_classes = domain_net_datasets(args.data_root)
    elif args.test_dataset == 'cifarstl':
        source_train, source_test, target, num_classes = CIFAR_STL_dataset(args.data_root)
    elif args.test_dataset in ['living17', 'entity30']:
        source_train, source_test, target, num_classes = breeds_dataset(root='../../imagenet1k', 
                        info_dir = os.path.join(args.data_root, 'breeds'), dataset_name = args.test_dataset)


    elif args.test_dataset in ['imagenet-a', 'imagenet-r', 'imagenet-sketch']:
        source_train = ImageFolder(os.path.join('../../imagenet1k', 'train'), transform = base_transform)
        source_test = ImageFolder(os.path.join('../../imagenet1k', 'val'), transform = base_transform)
 
        if args.test_dataset == 'imagenet-sketch':
            num_classes = 1000
            target = ImageFolder(os.path.join('../robust-imagenets', args.test_dataset, 'sketch'), transform = base_transform)
        else:
            num_classes = 200
            target = ImageFolder(os.path.join('../robust-imagenets', args.test_dataset), transform =base_transform)

        # Now choose only a subset of this dataset
        data_size = len(target)
        train_size = int(data_size*0.6)
        val_size = data_size - train_size
        _, target = torch.utils.data.random_split(target, [train_size, val_size], 
                                                                    generator = generator)

    
    trainval_dataset = torch.utils.data.ConcatDataset([source_train, source_test])
    train_loader = DataLoader(source_train, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, drop_last=False)
    val_loader = DataLoader(source_test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(target, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    trainval_loader = DataLoader(trainval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    return train_loader, val_loader, trainval_loader, test_loader, num_classes


def get_few_shot_reg_datasets(args):
    base_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
        ])
    dataset = dataset_info[args.test_dataset]['class'](os.path.join(args.data_root, dataset_info[args.test_dataset]['dir'])
                                        ,transform = base_transform)
    few_shot_size = int(args.shot_size * len(dataset))
    val_size = int(0.2*(len(dataset) - few_shot_size))
    test_size = len(dataset) - few_shot_size - val_size    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [few_shot_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                                num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                                num_workers=args.workers)
    
    trainval_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
    trainval_loader = DataLoader(trainval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    num_classes = dataset_info[args.test_dataset]['num_classes']

    return train_loader, val_loader, trainval_loader, test_loader, num_classes

def get_scores(preds, labels, test_dataset):
    if dataset_info[test_dataset]['mode'] == 'regression':
        return r2_score(labels.flatten().detach().cpu().numpy(), preds.flatten().detach().cpu().numpy())
    elif dataset_info[test_dataset]['mode'] == 'pose_estimation':
        return dist_acc((preds-labels)**2)
    elif dataset_info[test_dataset]['mode'] == 'classification':
        return accuracy(preds, labels, topk=(1, 5))[0]

def dist_acc(dists, thr=0.001):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    # dists = dists.detach().cpu().numpy()
    dist_cal = np.not_equal(dists, 0.0)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1

def load_backbone(args):
    # Model 
    if args.baseline:
        if args.arch == 'resnet50':
            print("Loading baseline")
            if args.moco:
                models_ensemble = resnet50()
            else:
                models_ensemble = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            # models_ensemble = resnet50()
        elif args.arch == 'convnext':
            models_ensemble = convnext_base(pretrained=True, num_classes=1000)
    else:
        if args.arch == 'resnet50':
            if args.moco:
                num_classes = 128
            else:
                num_classes = 1000
            if args.model_type == 'branched':
                models_ensemble = models.BranchedResNet(N = args.ensemble_size, num_classes = num_classes, arch = args.arch)
        elif args.arch == 'convnext':
            models_ensemble = models.DiverseConvNext(N = args.ensemble_size, num_classes = 1000)

    # If MOco momdel is used, change the projection layer before laoding checkpoint
    if args.moco is not None and args.moco != 'augself':
        change_projection_layer(models_ensemble, args.baseline)
        
    # optionally resume from a checkpoint, only for ensemble model
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            if args.gpu is None:
                checkpoint = torch.load(args.pretrained)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.pretrained, map_location=loc)

            if args.moco is None or args.moco != 'augself':
                state_dict = checkpoint['state_dict']
            else:
                checkpoint['epoch'] = 200

            # if not args.baseline:
            if args.moco is not None:
                if args.moco == 'augself':
                    state_dict = checkpoint['backbone']
                else:
                    for k in list(state_dict.keys()):
                        # retain only base_encoder up to before the embedding layer
                        if k.startswith('module.encoder_q.'):
                            # remove prefix
                            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                        # delete renamed or unused k
                        del state_dict[k]
            else:
                for k in list(state_dict.keys()):
                    # retain only base_encoder up to before the embedding layer
                    if k.startswith('module.'):
                        # remove prefix
                        state_dict[k[len("module."):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]

            models_ensemble.load_state_dict(state_dict, strict=False)
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # scaler.load_state_dict(checkpoint['scaler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pretrained, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    if args.baseline:
        if args.arch == 'resnet50':
            models_ensemble.feat_dim = 2048 if args.arch == 'resnet50' else 512
            try:
                if dataset_info[args.test_dataset] in ['aloi', 'animal_pose', 'mpii']:
                    models_ensemble.global_pool = nn.Identity()
            except:
                pass
            models_ensemble.fc = nn.Identity()
        elif args.arch == 'convnext':
            models_ensemble.head.fc = nn.Identity()
    else:
        if args.arch == 'resnet50':
            models_ensemble.base_model.branches_fc = nn.ModuleList([nn.Identity() for i in range(args.ensemble_size + 1)])
        elif args.arch == 'convnext':
            for ind in range(args.ensemble_size):
                models_ensemble.base_model.head[ind].fc = nn.Identity()
    
    # freeze all layers but the last fc
    for name, param in models_ensemble.named_parameters():
        # if args.few_shot_reg is None:
        if args.arch == 'resnet50':
            if "fc" not in name:
                param.requires_grad = False
        elif args.arch == 'convnext':
            if "head" not in name:
                param.requires_grad = False
        # print(name, param.requires_grad)

    # infer learning rate before changing batch size, not done in hyoer-models
    models_ensemble = models_ensemble.cuda(args.gpu)
    return models_ensemble

def change_projection_layer(model, baseline):
    if baseline:
        dim_mlp = model.fc.weight.shape[1]
        model.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, 128)
        )
    else:
        dim_mlp = model.base_model.branches_fc[0].weight.shape[1]
        model.base_model.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, model.num_classes)
        )