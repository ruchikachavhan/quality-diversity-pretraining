import os
import numpy as np
from glob import glob
import torchvision.transforms as transforms
import torch
from torch.utils.data.dataset import Dataset
from torchvision.datasets.folder import default_loader
import torchvision
import torch.nn.functional as F
from PIL import Image
import numpy as np
from scipy.io import loadmat
from math import ceil
from collections import namedtuple
import csv
from functools import partial
import torch
import os
import PIL
import cv2
import torchvision.transforms.functional as FT
from typing import Any, Callable, List, Optional, Union, Tuple
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_file_from_google_drive, check_integrity, verify_str_arg
from torchvision.datasets.folder import pil_loader, accimage_loader, default_loader
from torchvision.datasets import ImageFolder
import json

imagenet_mean_std = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]

class FacesInTheWild300W(Dataset):
    def __init__(self, root, split, mode='indoor+outdoor', transform=None, loader=default_loader, download=False):
        self.root = root
        self.split = split
        self.mode = mode
        self.transform = transform
        self.loader = loader
        images = []
        keypoints = []
        if 'indoor' in mode:
            print('Loading indoor images')
            images += glob(os.path.join(self.root, '01_Indoor', '*.png'))
            keypoints += glob(os.path.join(self.root, '01_Indoor', '*.pts'))
        if 'outdoor' in mode:
            print('Loading outdoor images')
            images += glob(os.path.join(self.root, '02_Outdoor', '*.png'))
            keypoints += glob(os.path.join(self.root, '02_Outdoor', '*.pts'))
        images = list(sorted(images))[0:len(images)-1]
        keypoints = list(sorted(keypoints))

        split_path = os.path.join(self.root, f'{mode}_{split}.npy')
        while not os.path.exists(split_path):
            self.generate_dataset_splits(len(images))
        split_idxs = np.load(split_path)
        print(split, split_path, len(split_idxs), len(images), len(keypoints))
        self.images = [images[i] for i in split_idxs-1]
        self.keypoints = [keypoints[i] for i in split_idxs-1]

    def generate_dataset_splits(self, l, split_sizes=[0.3, 0.3, 0.4]):
        np.random.seed(0)
        print(split_sizes)
        assert sum(split_sizes) == 1
        idxs = np.arange(l)
        np.random.shuffle(idxs)
        split1, split2 = int(l * split_sizes[0]), int(l * sum(split_sizes[:2]))
        train_idx = idxs[:split1]
        valid_idx = idxs[split1:split2]
        test_idx = idxs[split2:]
        #print(max(train_idx), max(valid_idx), max(test_idx))
        print("Generated train")
        np.save(os.path.join(self.root, f'{self.mode}_train'), train_idx)
        np.save(os.path.join(self.root, f'{self.mode}_valid'), valid_idx)
        print("Generated train and val")
        np.save(os.path.join(self.root, f'{self.mode}_test'), test_idx)
        print("Generated test")

    def __getitem__(self, index):
        # get image in original resolution
        path = self.images[index]
        image = self.loader(path)
        h, w = image.height, image.width
        min_side = min(h, w)

        # get keypoints in original resolution
        keypoint = open(self.keypoints[index], 'r').readlines()
        keypoint = keypoint[3:-1]
        keypoint = [s.strip().split(' ') for s in keypoint]
        keypoint = torch.tensor([(float(x), float(y)) for x, y in keypoint])
        bbox_x1, bbox_x2 = keypoint[:, 0].min().item(), keypoint[:, 0].max().item()
        bbox_y1, bbox_y2 = keypoint[:, 1].min().item(), keypoint[:, 1].max().item()
        bbox_width = ceil(bbox_x2 - bbox_x1)
        bbox_height = ceil(bbox_y2 - bbox_y1)
        bbox_length = max(bbox_width, bbox_height)

        image = FT.crop(image, top=bbox_y1, left=bbox_x1, height=bbox_length, width=bbox_length)
        keypoint = torch.tensor([(x - bbox_x1, y - bbox_y1) for x, y in keypoint])
        
        h, w = image.height, image.width
        min_side = min(h, w)

        if self.transform is not None:
            image = self.transform(image)
        try:
            new_h, new_w = image.shape[1:]
        except:
            new_h, new_w = image.shape[2:]

        keypoint = torch.tensor([[
            ((x - ((w - min_side) / 2)) / min_side) * new_w,
            ((y - ((h - min_side) / 2)) / min_side) * new_h,
        ] for x, y in keypoint])
        keypoint = keypoint.flatten()
        keypoint = F.normalize(keypoint, dim=0)
        return image, keypoint

    def __len__(self):
        return len(self.images)



CSV = namedtuple("CSV", ["header", "index", "data"])


class CelebA(VisionDataset):
    """`Large-scale CelebFaces Attributes (CelebA) Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        split (string): One of {'train', 'valid', 'test', 'all'}.
            Accordingly dataset is selected.
        target_type (string or list, optional): Type of target to use, ``attr``, ``identity``, ``bbox``,
            or ``landmarks``. Can also be a list to output a tuple with all specified target types.
            The targets represent:

                - ``attr`` (np.array shape=(40,) dtype=int): binary (0, 1) labels for attributes
                - ``identity`` (int): label for each person (data points with the same identity are the same person)
                - ``bbox`` (np.array shape=(4,) dtype=int): bounding box (x, y, width, height)
                - ``landmarks`` (np.array shape=(10,) dtype=int): landmark points (lefteye_x, lefteye_y, righteye_x,
                  righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y)

            Defaults to ``attr``. If empty, ``None`` will be returned as target.

        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "celeba"
    # There currently does not appear to be a easy way to extract 7z in python (without introducing additional
    # dependencies). The "in-the-wild" (not aligned+cropped) images are only in 7z, so they are not available
    # right now.
    file_list = [
        # File ID                                      MD5 Hash                            Filename
        ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        # ("0B7EVK8r0v71pbWNEUjJKdDQ3dGc","b6cd7e93bc7a96c2dc33f819aa3ac651", "img_align_celeba_png.7z"),
        # ("0B7EVK8r0v71peklHb0pGdDl6R28", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_celeba.7z"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
        ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
        # ("0B7EVK8r0v71pTzJIdlJWdHczRlU", "063ee6ddb681f96bc9ca28c6febb9d1a", "list_landmarks_celeba.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]

    def __init__(
            self,
            root: str,
            split: str = "train",
            target_type: Union[List[str], str] = "landmarks",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            shots: int = None
    ) -> None:
        super(CelebA, self).__init__(root, transform=transform,
                                     target_transform=target_transform)
        self.split = split
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError('target_transform is specified but target_type is empty')

        #if download:
        #    self.download()

        #if not self._check_integrity():
        #    raise RuntimeError('Dataset not found or corrupted.' +
        #                       ' You can use download=True to download it')

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[verify_str_arg(split.lower(), "split",
                                          ("train", "valid", "test", "all"))]
        splits = self._load_csv("list_eval_partition.txt")
        identity = self._load_csv("identity_CelebA.txt")
        bbox = self._load_csv("list_bbox_celeba.txt", header=1)
        landmarks_align = self._load_csv("list_landmarks_align_celeba.txt", header=1)
        attr = self._load_csv("list_attr_celeba.txt", header=1)

        mask = slice(None) if split_ is None else (splits.data == split_).squeeze()

        self.filename = splits.index
        if shots is None:
            self.filename = [self.filename[i] for i, m in enumerate(mask) if m]
            self.identity = identity.data[mask]
            self.bbox = bbox.data[mask]
            self.landmarks_align = landmarks_align.data[mask]
            self.attr = attr.data[mask]
            # map from {-1, 1} to {0, 1}
            self.attr = torch.div(self.attr + 1, 2).to(int)
            self.attr_names = attr.header
        else:
            self.filename = [self.filename[i] for i, m in enumerate(mask) if m]
            l_shot = int(shots*len(self.filename))
            self.filename = self.filename[:l_shot]
            self.identity = identity.data[mask][:l_shot]
            self.bbox = bbox.data[mask][:l_shot]
            self.landmarks_align = landmarks_align.data[mask][:l_shot]
            self.attr = attr.data[mask][:l_shot]
            # map from {-1, 1} to {0, 1}
            self.attr = torch.div(self.attr + 1, 2).to(int)
            self.attr_names = attr.header
        

        print()

    def _load_csv(
        self,
        filename: str,
        header: Optional[int] = None,
    ) -> CSV:
        data, indices, headers = [], [], []

        fn = partial(os.path.join, self.root, self.base_folder)
        with open(fn(filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=' ', skipinitialspace=True))

        if header is not None:
            headers = data[header]
            data = data[header + 1:]

        indices = [row[0] for row in data]
        data = [row[1:] for row in data]
        data_int = [list(map(int, i)) for i in data]

        return CSV(headers, indices, torch.tensor(data_int))

    def _check_integrity(self) -> bool:
        for (_, md5, filename) in self.file_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            _, ext = os.path.splitext(filename)
            # Allow original archive to be deleted (zip and 7z)
            # Only need the extracted images
            print("CHeck integrity", filename, ext)
            if ext not in [".zip", ".7z"] and not check_integrity(fpath, md5):
                return False

        # Should check a hash of the images
        return os.path.isdir(os.path.join(self.root, self.base_folder, "img_align_celeba"))

    def download(self) -> None:
        import zipfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        for (file_id, md5, filename) in self.file_list:
            download_file_from_google_drive(file_id, os.path.join(self.root, self.base_folder), filename, md5)

        print("FOLDER", os.path.join(self.root, self.base_folder))
        with zipfile.ZipFile(os.path.join(self.root, self.base_folder, "img_align_celeba.zip"), "r") as f:
            f.extractall(os.path.join(self.root, self.base_folder))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))
        w, h = X.width, X.height
        min_side = min(w, h)

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError("Target type \"{}\" is not recognized.".format(t))

        if self.transform is not None:
            X = self.transform(X)
        new_w, new_h = X.shape[1:]

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        # transform the landmarks
        new_target = torch.zeros_like(target)
        if 'landmarks' in self.target_type:
            for i in range(int(len(target) / 2)):
                new_target[i * 2] = ((target[i * 2] - ((w - min_side) / 2)) / min_side) * new_w
                new_target[i * 2 + 1] = ((target[i * 2 + 1] - ((h - min_side) / 2)) / min_side) * new_h
                
        return X, new_target.float()

    def __len__(self) -> int:
        return len(self.attr)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)

class LeedsSportsPose(Dataset):
    def __init__(self, root, split, transform=None, loader=default_loader, download=False, shots = None):
        self.root = root
        self.split = split
        self.transform = transform
        self.loader = loader
        images = glob(os.path.join(self.root, 'images', '*.jpg'))
        images = sorted(images)
        joints = loadmat(os.path.join(self.root, 'joints.mat'))['joints']
        joints = np.array([[(joints[0, j, i], joints[1, j, i], joints[2, j, i]) for j in range(joints.shape[1])] for i in range(joints.shape[2])])

        split_path = os.path.join(self.root, f'{split}.npy')
        # while not os.path.exists(split_path):
        self.generate_dataset_splits(len(images), shots = shots)
        split_idxs = np.load(split_path)
        self.images = [images[i] for i in split_idxs]
        self.joints = [joints[i] for i in split_idxs]

    def generate_dataset_splits(self, l, split_sizes=[0.6, 0.4], shots = None):
        np.random.seed(0)
        assert sum(split_sizes) == 1
        idxs = np.arange(l)
        np.random.shuffle(idxs)
        print("shots", shots)
        if shots is None:
            split1 = int(l * split_sizes[0])
            train_idx = idxs[:split1]
            test_idx = idxs[split1:]
        else:
            print("shots", shots)
            split1 = int(l * shots)
            train_idx = idxs[:split1]
            test_idx = idxs[split1:]
        print(max(train_idx), max(test_idx))
        np.save(os.path.join(self.root, 'train'), train_idx)
        np.save(os.path.join(self.root, 'test'), test_idx)

    def __getitem__(self, index):
        # get image in original resolution
        path = self.images[index]
        image = self.loader(path)
        h, w = image.height, image.width
        min_side = min(h, w)

        # get keypoints in original resolution
        joints = self.joints[index]

        bbox_x1 = int((w - min_side) / 2) if w >= min_side else 0
        bbox_x2 = bbox_x1 + min_side
        bbox_y1 = int((h - min_side) / 2) if h >= min_side else 0
        bbox_y2 = bbox_y1 + min_side

        image = FT.crop(image, top=bbox_y1, left=bbox_x1, height=min_side, width=min_side)
        joints = torch.tensor([(x - bbox_x1, y - bbox_y1) for x, y, _ in joints])
        
        h, w = image.height, image.width
        min_side = min(h, w)

        if self.transform is not None:
            image = self.transform(image)
        try:
            new_h, new_w = image.shape[1:]
        except:
            new_h, new_w = image.shape[2:]

        joints = torch.tensor([[
            ((x - ((w - min_side) / 2)) / min_side) * new_w,
            ((y - ((h - min_side) / 2)) / min_side) * new_h,
        ] for x, y in joints])
        joints = joints.flatten()

        return image, joints

    def __len__(self):
        return len(self.images)

class Causal3DIdent(ImageFolder):
    # 3600 images over 7 classes
    # teapot, hare, dragon, cow, armadillo, horse, head
    # latents
    # (obj pos X, Y, Z), (obj rot X, Y, Z), (light angle), (obj hue), (light hue), (bg hue)
    def __init__(self, root, split, transform, target_type='latents'):
        split_path = os.path.join(root, split + 'set')
        print("PATH", split_path)
        super().__init__(split_path, transform=transform)
        self.target_type = target_type
        self.num_classes = 7
        
        self.latents = np.concatenate([np.load(os.path.join(split_path, f'latents_{i}.npy')) for i in range(self.num_classes)])

    def __getitem__(self, i):
        img, class_label = super().__getitem__(i)
        latent = self.latents[i]
        return img, latent

class AnimalPose(ImageFolder):
    def __init__(self, root, transform=None):
        self.data = json.load(open(os.path.join(root, "keypoints.json")))
        self.root = root
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform
        self.image_size = (224, 224)
        self.resize = transforms.Resize(self.image_size)

    def __getitem__(self, i):
        image_id = str(self.data['annotations'][i]['image_id'])
        image_name = self.data['images'][image_id]

        source = os.path.join(self.root, 'images', image_name)
        np_image = cv2.imread(source)        
        bbox = self.data['annotations'][i]['bbox']
        cropped_image = np_image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        keypoint = np.array(self.data['annotations'][i]['keypoints']).tolist()
        new_keypoint_list = []
        for i in range(len(keypoint)):
            if keypoint[i][0] != 0 and keypoint[i][1] != 0:
                new_keypoint = [keypoint[i][0] - bbox[0], keypoint[i][1] - bbox[1]]
            else:
                new_keypoint = [keypoint[i][0], keypoint[i][1]]
            new_keypoint_list.append(new_keypoint)
        
        resize_image = self.transform(self.resize(Image.fromarray(cropped_image)))
        new_keypoint_list = np.array(new_keypoint_list)/self.image_size
        new_keypoint_list = new_keypoint_list.reshape(-1)
        return resize_image, new_keypoint_list

    def __len__(self):
        return len(self.data['annotations'])

class ALOI(ImageFolder):
    def __init__(self, root, transform=None):
        self.root = root
        self.num_classes = len(os.listdir(self.root))
        self.samples = []
        self.image_size = (224, 224)
        for class_id in range(self.num_classes):
            class_images = os.listdir(os.path.join(self.root, str(class_id+1)))
            for img in class_images:
                self.samples.append(os.path.join(self.root, str(class_id+1), img))

        if transform is None:
            self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()])
        else:
            self.transform = transform
        self.label_dict = {'l1c1': 1, 'l1c2': 2, 'l1c3': 3, 
                    'l2c1': 4, 'l2c2': 5, 'l2c3': 6, 
                    'l3c1': 7, 'l3c2': 8, 'l3c3': 9,
                    'l4c1': 10, 'l4c2': 11, 'l4c3': 12,
                    'l5c1': 13, 'l5c2': 14, 'l5c3': 15,
                    'l6c1': 16, 'l6c2': 17, 'l6c3': 18,
                    'l7c1': 19, 'l7c2': 20, 'l7c3': 21,
                    'l8c1': 22, 'l8c2': 23, 'l8c3': 24,
                    }

    def __getitem__(self, i):
        image = Image.open(self.samples[i])
        image = self.transform(image)
        label = self.samples[i].split("/")[-1].split(".")[0].split("_")[1]
        label = self.label_dict[label] - 1
        return image, label

    def __len__(self):
        return len(self.samples)

# bbox using keypoints
def cal_bbox(pts):
    bbox_list = []
    # pts = np.load(kpts).astype(np.float16)
    y_min = int(np.min(pts[:,1]))
    y_max = int(np.max(pts[:,1]))
    x_min = int(np.min(pts[:,0]))
    x_max = int(np.max(pts[:,0]))
    bbox_list = [x_min, x_max, y_min, y_max]
    return bbox_list

class MPII(ImageFolder):
    def __init__(self, root, transform=None):
        self.root = root
        self.annot = json.load(open(os.path.join(root, "annot", "trainval.json")))
        self.samples = []
        fname = 'indices.txt'
        if not os.path.exists(os.path.join(self.root, fname)):
            self.indices_file = open(os.path.join(self.root, fname), 'w')
            for i in range(0, len(self.annot)):
                data = self.annot[i]
                joints_vis = data['joints_vis']
                joints = np.array(data['joints'])
                image = data["image"]
                image_new = str(i) + ".jpg"
                indices = np.where(np.all([joints[:, 0] > 0, joints[:, 1] > 0], axis=0))
                visible_joints = joints[indices].reshape(-1, 2)
                if visible_joints.shape[0] !=0:
                    bbox = cal_bbox(visible_joints)
                    image_path = os.path.join(root, "images", image)
                    image_np = cv2.imread(image_path)
                    cropped_image = image_np[bbox[2]:bbox[3], bbox[0]+5:bbox[1]+5]
                    if cropped_image.shape[0] !=0 and cropped_image.shape[1]!=0:
                        self.samples.append(i)
                        self.indices_file.write(str(i))
                        self.indices_file.write("\n")
        else:
           with open(os.path.join(self.root, fname), 'r') as f:
            self.samples = [line.strip() for line in f]

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform
        self.image_size = (224, 224)
        self.resize = transforms.Resize(self.image_size)

    def __getitem__(self, index):
        i = int(self.samples[index])
        data = self.annot[i]
        joints_vis = data['joints_vis']
        joints = np.array(data['joints'])
        image = data["image"]
        image_new = str(i) + ".jpg"

        indices = np.where(np.all([joints[:, 0] > 0, joints[:, 1] > 0], axis=0))

        visible_joints = joints[indices].reshape(-1, 2)
        if visible_joints.shape[0] !=0:
            bbox = cal_bbox(visible_joints)
            image_path = os.path.join(self.root, "images", image)
            image_np = cv2.imread(image_path)
            cropped_image = image_np[bbox[2]:bbox[3], bbox[0]+5:bbox[1]+5]
            if cropped_image.shape[0] !=0 and cropped_image.shape[1]!=0:
                joints[:, 0] -= bbox[0]
                joints[:, 1] -= bbox[2]
                joints[:, 0] /= cropped_image.shape[1]
                joints[:, 1] /= cropped_image.shape[0]
                joints[joints < 0] = 0.0
            
        resize_image = self.transform(self.resize(Image.fromarray(cropped_image)))
        joints = joints.reshape(-1)
        return resize_image, joints
