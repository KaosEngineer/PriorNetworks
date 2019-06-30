import seaborn as sns
import torch
import torch.optim as optim
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100

from prior_networks.models.densenet import DenseNet
from prior_networks.priornet.losses import DirichletKLLoss, DirichletKLLossJoint
from prior_networks.priornet.training import TrainerWithOODJoint
from prior_networks.util_pytorch import save_model

sns.set()

class target_transform:
    def __init__(self, target_concentration, gamma, ood=False):
        self.target_concentration = target_concentration
        self.gamma = gamma
        self.ood = ood

    def __call__(self, label):
        return self.forward(label)

    def forward(self, label):
        if self.ood:
            return (0, self.target_concentration, self.gamma)
        else:
            return (label, self.target_concentration, self.gamma)

def main(argv=None):
    print(torch.cuda.device_count())
    device = torch.device('cuda:0')

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(32),
            transforms.Pad(4,padding_mode='symmetric'),
            transforms.RandomCrop(32),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
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
            transforms.Pad(4,padding_mode='symmetric'),
            transforms.RandomCrop(32),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    path = 'scratch/test'
    id_dataset = CIFAR10(root=path,
                         transform=data_transforms['train'], train=True,
                         target_transform=target_transform(1e2,1.0),
                         download=True)
    test_dataset = CIFAR10(root=path,
                           transform=data_transforms['val'], train=False,
                           #target_transform=target_transform(1e2, 1.0),
                           download=True)
    ood_dataset = CIFAR100(root=path,
                           transform=data_transforms['ood'], train=True,
                           target_transform=target_transform(0.0, 10.0, ood=True),
                           download=True)

    train_dataset = data.ConcatDataset([id_dataset, ood_dataset])

    model = DenseNet(drop_rate=0.2,
                     num_classes=10,
                     growth_rate=32,
                     num_init_features=64,
                     block_config=(6, 12, 24, 16))
    model = torch.nn.DataParallel(model)
    model.to(device)

    criterion = DirichletKLLossJoint()
    trainer = TrainerWithOODJoint(model, criterion,
                                  test_criterion=DirichletKLLoss(target_concentration=1e2),
                                  train_dataset=train_dataset,
                                  #ood_dataset=ood_dataset,
                                  test_dataset=test_dataset,
                                  optimizer=optim.SGD,
                                  device=device,
                                  checkpoint_path='./',
                                  #scheduler=optim.lr_scheduler.MultiStepLR,
                                  scheduler=optim.lr_scheduler.CosineAnnealingLR,
                                  optimizer_params={'lr': 5e-4, 'momentum': 0.9,
                                               'nesterov': True,
                                               'weight_decay': 1e-4},
                                  #scheduler_params={'milestones': [150,215]},
                                  scheduler_params={'T_max':25,'eta_min':0.0},
                                  batch_size=128)
    trainer.train(300)
    save_model(model, 'model', '/scratch/test/')

if __name__ == '__main__':
    main()
