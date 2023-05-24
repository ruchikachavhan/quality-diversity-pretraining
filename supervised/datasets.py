import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageFilter, ImageOps
import math
import random
import itertools

from PIL import Image, ImageFilter
  
  
class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class Edges(object):
    """Canny Edges for images"""
    def __init__(self):
        self.convert_grayscale = True
    def __call__(self, x):
        # Converting the image to grayscale, as edge detection 
        # requires input image to be of mode = Grayscale (L)
        x = x.convert("L")
        # Detecting Edges on the Image using the argument ImageFilter.FIND_EDGES
        x = x.filter(ImageFilter.FIND_EDGES).convert('RGB')
        return x

img_size = 224
# Normalize
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
# Set of default augmentations
default_augmentations = [
    transforms.RandomResizedCrop(img_size, scale=(0.08, 1.)),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
    ], p=1.0),
    transforms.RandomGrayscale(p=1.0),
    transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0), 
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
]

# Set of default augmentations
default_augmentations_edges = [
    transforms.RandomResizedCrop(img_size, scale=(0.08, 1.)),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
    ], p=1.0),
    transforms.RandomGrayscale(p=1.0),
    transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0), 
    transforms.RandomApply([Edges()], p=1.0), 
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
]


# Set of dorsal augmentations
# Using solarize as per MoCO-v3
# Including Resize and then center crop because Random Resize crop would be a ventral augmentation
dorsal_augmentations = [
    transforms.Resize((img_size, img_size)),
    transforms.CenterCrop(img_size),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
    transforms.ToTensor(),
    normalize
]

tta_augmentations = [
    transforms.RandomResizedCrop((img_size, img_size)),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomVerticalFlip(p=1.0),
    transforms.RandomAffine(degrees=90, translate=(0.1, 0.1)),
    transforms.RandomRotation(270),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
    ], p=1.0),
    transforms.RandomGrayscale(p=1.0),
    transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
    transforms.ToTensor(),
    normalize
]

# tta augmentations
tta_augmentations = list(itertools.combinations(tta_augmentations[:-2], 1))
tta_augmentations = [(transforms.Resize((img_size, img_size)), ) + tta_augmentations[i]+ (transforms.ToTensor(), normalize) for i in range (len(tta_augmentations))]

combinations_default = list(itertools.combinations(default_augmentations[:-2], 1))
combinations_default = [(transforms.Resize((img_size, img_size)), ) + combinations_default[i]+ (transforms.ToTensor(), normalize) for i in range (len(combinations_default))]

combinations_default_edges = list(itertools.combinations(default_augmentations_edges[:-2], 1))
combinations_default_edges = [(transforms.Resize((img_size, img_size)), ) + combinations_default_edges[i]+ (transforms.ToTensor(), normalize) for i in range (len(combinations_default_edges))]

# Set of ventral augmentations
ventral_augmentations = [
    transforms.RandomResizedCrop(img_size, scale=(0.08, 1.)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
]

base_augs = [
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    normalize
]

class CropsTransform:
    """Returns two random crops of one image for each type of augmentation"""

    def __init__(self, augs_list):
        self.augs_list = augs_list

    def __call__(self, x):
        outputs = []
        for i in range(0, len(self.augs_list)):
            t = transforms.Compose(self.augs_list[i])
            outputs.append(t(x))
        outputs = torch.cat(outputs, dim = 0).reshape(len(self.augs_list), 3, img_size, img_size)
        return outputs