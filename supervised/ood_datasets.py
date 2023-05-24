import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from torchvision.datasets import CIFAR10, STL10
from robustness.tools.breeds_helpers import setup_breeds
from robustness.tools.breeds_helpers import ClassHierarchy
import sys
from torch.utils.data import Dataset
from robustness.tools.breeds_helpers import make_living17, make_entity13, make_entity30, make_nonliving26

def get_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def get_val_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

class ImageList(Dataset):
    def __init__(self, image_root, image_list_root, dataset, domain_label, dataset_name, split='train', transform=None, 
                 sample_masks=None, pseudo_labels=None, strong_transform=None, aug_num=0, rand_aug=False, freq=False):
        self.image_root = image_root
        self.dataset = dataset  # name of the domain
        self.dataset_name = dataset_name  # name of whole dataset
        self.transform = transform
        self.strong_transform = strong_transform
        self.loader = self._rgb_loader
        self.sample_masks = sample_masks
        self.pseudo_labels = pseudo_labels
        self.rand_aug = rand_aug
        self.aug_num = aug_num
        self.freq = freq
        if dataset_name == 'domainnet' or dataset_name == 'minidomainnet':
            imgs = self._make_dataset(os.path.join(image_list_root, dataset + '_' + split + '.txt'), domain_label)
        else:
            imgs = self._make_dataset(os.path.join(image_list_root, dataset + '.txt'), domain_label)
        self.imgs = imgs
        self.tgts = [s[1] for s in imgs]
        if sample_masks is not None:
            temp_list = self.imgs
            self.imgs = [temp_list[i] for i in self.sample_masks]
            if pseudo_labels is not None:
                self.labels = self.pseudo_labels[self.sample_masks]
                assert len(self.labels) == len(self.imgs), 'Lengths do no match!'

    def _rgb_loader(self, path):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def _make_dataset(self, image_list_path, domain):
        print('image path', image_list_path)
        image_list = open(image_list_path).readlines()
        images = [(val.split()[0], int(val.split()[1]), int(domain)) for val in image_list]
        return images
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        output = {}
        path, target, domain = self.imgs[index]
        # bug exists for 
        if self.dataset_name == 'domainnet' or self.dataset_name == 'minidomainnet':
            raw_img = self.loader(os.path.join(self.image_root, path))
        elif self.dataset_name in ['pacs']:
            raw_img = self.loader(os.path.join(self.image_root, self.dataset, path))
        elif self.dataset_name in ['office-home', 'office31', 'office-home-btlds']:
            raw_img = self.loader(os.path.join(self.image_root, path))
        if self.transform is not None and self.freq == False:
            img = self.transform(raw_img)
        # output['img'] = img
        # if self.pseudo_labels is not None:
        #     output['target'] = torch.squeeze(torch.LongTensor([np.int64(self.labels[index]).item()]))
        # else:
        #     output['target'] = torch.squeeze(torch.LongTensor([np.int64(self.tgts[index]).item()]))
        # output['domain'] = domain
        # output['idx'] = index

        return img, target


def domain_net_datasets(image_root, path = 'DomainNet'):
    dataset = {}
    dataset_info = {'clipart': 0, 'painting': 1, 'real': 2, 'sketch': 3}
    image_root = os.path.join(image_root, path)
    source_train = ImageList(image_root, image_root, 'sketch', dataset_info['sketch'], 'domainnet', split='train', transform=get_val_transform())
    source_test = ImageList(image_root, image_root, 'sketch', dataset_info['sketch'], 'domainnet', split='test', transform=get_val_transform())
    target_train = []
    target_test = []
    num_classes = 40
    for domain in dataset_info.keys():
        if domain != 'sketch':
            target_train.append(ImageList(image_root, image_root, domain, dataset_info[domain], 'domainnet', split='train', transform=get_val_transform()))
            target_test.append(ImageList(image_root, image_root, domain, dataset_info[domain], 'domainnet', split='test', transform=get_val_transform()))
    target = torch.utils.data.ConcatDataset(target_train + target_test)
    return source_train, source_test, target_test, num_classes

def CIFAR_STL_dataset(image_root):
    source_train = CIFAR10(image_root, train=True, transform=get_val_transform(), download=True)
    source_test = CIFAR10(image_root, train=False, transform=get_val_transform(), download=True)
    target_train = STL10(image_root, split='train', transform=get_val_transform(), download=True)
    target_test = STL10(image_root, split='test', transform=get_val_transform(), download=True)
    # target = torch.utils.data.ConcatDataset([target_train, target_test])
    num_classes = 10
    return source_train, source_test, target_test, num_classes

# Test the domain net dataset
# domainnet_info = {'clipart': 0, 'painting': 1, 'real': 2, 'sketch': 3}
# domainnet = ImageList(image_root='../../TestDatasets/DomainNet', image_list_root='../../TestDatasets/DomainNet', dataset='clipart', 
#                     domain_label=0, dataset_name='domainnet', split='train', transform=get_train_trasnform(), sample_masks=None, pseudo_labels=None, strong_transform=None, aug_num=0, rand_aug=False, freq=False)
# dataloader = torch.utils.data.DataLoader(domainnet, batch_size=64, shuffle=True, num_workers=0)
# for i, data in enumerate(dataloader):
#     print(data)
# source_train, source_test, target = domain_net_datasets(image_root='../../TestDatasets/DomainNet')
# source_train = torch.utils.data.DataLoader(source_train, batch_size=64, shuffle=True, num_workers=0)
# source_test = torch.utils.data.DataLoader(source_test, batch_size=64, shuffle=True, num_workers=0)
# target = torch.utils.data.DataLoader(target, batch_size=64, shuffle=True, num_workers=0)


# if not (os.path.exists('../../TestDatasets/breeds') and len(os.listdir('../../TestDatasets/breeds'))):
#     print("Downloading class hierarchy information into `info_dir`")
#     setup_breeds('../../TestDatasets/breeds')
# info_dir = '../../TestDatasets/breeds'
# hier = ClassHierarchy(info_dir)

# level = 3 # Could be any number smaller than max level
# superclasses = hier.get_nodes_at_level(level)
# print(f"Superclasses at level {level}:\n")
# print(", ".join([f"({si}: {hier.HIER_NODE_NAME[s]})" for si, s in enumerate(superclasses)]))

BREEDS_SPLITS_TO_FUNC = {
    'entity13': make_entity13,
    'entity30': make_entity30,
    'living17': make_living17,
    'nonliving26': make_nonliving26,
}

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

SPLITS = ['train', 'val']

MIN_NUM_TRAIN_PER_CLASS = 100

NUM_VAL_PER_CLASS = 50

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def get_classes(dir):
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def get_image_paths_breeds_class(class_dir, breeds_class):
    image_paths_breeds_class = []
    for root, _, fnames in sorted(os.walk(class_dir)):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, extensions=IMG_EXTENSIONS):
                path = os.path.join(root, fname)
                image_paths_breeds_class.append((path, breeds_class))
    return image_paths_breeds_class

def get_image_paths_by_class(data_dir, idx_to_class_id, subclasses, split):
    image_paths_and_class = []
    for idx in range(len(subclasses)):
        for subclass in subclasses[idx]:
            subclass_image_paths_breeds_class = get_image_paths_breeds_class(
                data_dir + '/' + idx_to_class_id[subclass] + '/', idx)
            image_paths_and_class.extend(subclass_image_paths_breeds_class)
            # print(data_dir + '/' + idx_to_class_id[subclass] + '/', len(subclass_image_names))
            if split == 'train':
                assert(len(subclass_image_paths_breeds_class) >= MIN_NUM_TRAIN_PER_CLASS)
            else:
                assert(len(subclass_image_paths_breeds_class) == NUM_VAL_PER_CLASS)
    return image_paths_and_class

class Breeds(Dataset):
    def __init__(self, root, breeds_name,
                 info_dir,
                 source=True, target=False, split='train', transform=None):
        super().__init__()
        if breeds_name not in BREEDS_SPLITS_TO_FUNC.keys():
            raise ValueError(f'breeds_name must be in {BREEDS_SPLITS_TO_FUNC.keys()} but was {breeds_name}')
        if split not in SPLITS:
            raise ValueError(f'split must be in {SPLITS} but was {split}')
        if not source and not target:
            raise ValueError('At least one of "source" and "target" must be True!')

        self._breeds_name = breeds_name
        self._source = source
        self._split = split
        self._transform = transform
        if os.path.isdir(root + '/BREEDS-Benchmarks'):
            self._info_dir = root + '/BREEDS-Benchmarks/imagenet_class_hierarchy/modified'
        else:
            self._info_dir = info_dir
        self._data_dir = root + '/' + split
        self._idx_to_class_id, self._class_to_idx = get_classes(self._data_dir)
        breeds_func = BREEDS_SPLITS_TO_FUNC[breeds_name]
        self._superclasses, self._subclass_split, self._label_map = breeds_func(self._info_dir, split="rand")
        self._subclasses = []

        if source:
            self._subclasses.extend(self._subclass_split[0])
        if target:
            self._subclasses.extend(self._subclass_split[1])

        self._image_paths_by_class = get_image_paths_by_class(
            self._data_dir, self._idx_to_class_id, self._subclasses, split)

        self.means = [0.485, 0.456, 0.406]
        self.stds = [0.228, 0.224, 0.225]

    def __getitem__(self, i):
        path, y = self._image_paths_by_class[i]
        x = Image.open(path)
        x = x.convert('RGB')
        if self._transform is not None:
            x = self._transform(x)
        return x, y

    def __len__(self) -> int:
        return len(self._image_paths_by_class)

    def get_num_classes(self):
        return len(self._idx_to_class_id)

def breeds_dataset(root, info_dir, dataset_name):
    source_train = Breeds(root=root, breeds_name=dataset_name, info_dir=info_dir, source=True, target=False, split='train', transform=get_val_transform())
    source_test = Breeds(root=root, breeds_name=dataset_name, info_dir=info_dir, source=True, target=False, split='val', transform=get_val_transform())
    target = Breeds(root=root, breeds_name=dataset_name, info_dir=info_dir, source=False, target=True, split='val', transform=get_val_transform())
    if dataset_name == 'living17':
        num_classes = 17
    elif dataset_name == 'entity30':
        num_classes = 30
    return source_train, source_test, target, num_classes
