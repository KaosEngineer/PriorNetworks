import seaborn as sns
import torch
import torch.optim as optim
from torch.utils import data
import numpy as np
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
# from torchvision.datasets import ImageNet
# import sys
# import prior_networks.models as models
# from prior_networks.priornet.losses import DirichletKLLoss, DirichletKLLossJoint
# from prior_networks.priornet.training import TrainerWithOODJoint
# from prior_networks.util_pytorch import save_model, TargetTransform, MODEL_DICT
# import foolbox

sns.set()


def main(argv=None):
    device = torch.device('cuda:0')

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'ood': transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    path = '/scratch/test'

    dataset1 = CIFAR10(root=path,
                         transform=data_transforms['train'], train=True,
                         #target_transform=TargetTransform(1e2, 1.0),
                         download=True)

    dataset2 = CIFAR100(root=path,
                         transform=data_transforms['train'], train=True,
                         #target_transform=TargetTransform(1e2, 1.0),
                         download=True)

    #print(dataset.data.shape)
    mean = np.mean(np.concatenate([dataset1.data/255.0, dataset2.data/255], axis=0), axis=(0,1,2))
    std = np.std(np.concatenate([dataset1.data/255.0, dataset2.data/255], axis=0), axis=(0,1,2))

    print(np.round(mean, 3), np.round(std, 3))


if __name__ == '__main__':
    main()
