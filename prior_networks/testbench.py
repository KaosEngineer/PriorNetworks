import seaborn as sns
import torch
import torch.optim as optim
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from torchvision.datasets import ImageNet
import sys
import prior_networks.models as models
from prior_networks.priornet.losses import DirichletKLLoss, DirichletKLLossJoint
from prior_networks.priornet.training import TrainerWithOODJoint
from prior_networks.util_pytorch import save_model, TargetTransform, model_dict
import foolbox
sns.set()


def main(argv=None):
    print(torch.cuda.device_count())
    device = torch.device('cuda:0')
    sys.exit()
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
    path = '/scratch/imagenet'

    dataset= ImageNet(root=path, split='train', download=True)
    dataset = ImageNet(root=path, split='val', download=True)
    sys.exit()

    id_dataset = CIFAR10(root=path,
                         transform=data_transforms['train'], train=True,
                         target_transform=TargetTransform(1e2, 1.0),
                         download=True)
    test_dataset = CIFAR10(root=path,
                           transform=data_transforms['val'], train=False,
                           #target_transform=target_transform(1e2, 1.0),
                           download=True)
    ood_dataset = CIFAR100(root=path,
                           transform=data_transforms['ood'], train=True,
                           target_transform=TargetTransform(0.0, 10.0, ood=True),
                           download=True)

    train_dataset = data.ConcatDataset([id_dataset, ood_dataset])


    model = models.resnet34(num_classes=10, small_inputs=True)
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
                                  scheduler=optim.lr_scheduler.MultiStepLR,
                                  #scheduler=optim.lr_scheduler.CosineAnnealingLR,
                                  optimizer_params={'lr': 1e-4, 'momentum': 0.9,
                                               'nesterov': True,
                                               'weight_decay': 1e-7},
                                  scheduler_params={'milestones': [30,50,90]},
                                  #scheduler_params={'T_max':25,'eta_min':0.0},
                                  batch_size=128)
    trainer.train(100)
    save_model(model.module,
               n_in=32,
               num_classes=10,
               n_channels=3,
               arch='resnet34',
               small_inputs=True,
               path='/scratch/test/')

    ckpt = torch.load('/scratch/test/model.tar')
    model = model_dict[ckpt['arch']](num_classes=ckpt['num_classes'],
                                     small_inputs=ckpt['small_inputs'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()



if __name__ == '__main__':
    main()
