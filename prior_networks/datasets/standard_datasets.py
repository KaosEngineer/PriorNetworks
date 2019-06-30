"""
A number of classes that transform the torch vision datasets into a standard format
usable for uncertainty (e.g. out-of-domain vs. in-domain) experimentation.
"""
import context
import torch
import torchvision
from torchvision.datasets import CIFAR10


CACHE_DIR = 'scratch/test'


def get_standardised_CIFAR10():
    # data_torchvision.transforms = {
    #     'train': torchvision.transforms.Compose([
    #         torchvision.transforms.Resize(32),
    #         torchvision.transforms.Pad(4, padding_mode='symmetric'),
    #         torchvision.transforms.RandomCrop(32),
    #         torchvision.transforms.RandomRotation(15),
    #         torchvision.transforms.RandomHorizontalFlip(),
    #         torchvision.transforms.ToTensor(),
    #         # torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ]),
    #     'val': torchvision.transforms.Compose([
    #         torchvision.transforms.Resize(32),
    #         torchvision.transforms.ToTensor(),
    #         # torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ]),
    #     'ood': torchvision.transforms.Compose([
    #         torchvision.transforms.Resize(32),
    #         torchvision.transforms.Pad(4, padding_mode='symmetric'),
    #         torchvision.transforms.RandomCrop(32),
    #         torchvision.transforms.RandomRotation(15),
    #         torchvision.transforms.RandomHorizontalFlip(),
    #         torchvision.transforms.RandomVerticalFlip(),
    #         torchvision.transforms.ToTensor(),
    #         # torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ])
    # }
    id_dataset = CIFAR10(root=path,
                         transform=data_transforms['train'], train=True,
                         target_transform=target_transform(1e2,1.0),
                         download=True)
